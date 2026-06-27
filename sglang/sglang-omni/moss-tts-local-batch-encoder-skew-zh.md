# RL 训推不一致的根源其实在 inference 系统中广泛存在

最近 SGLang Omni 团队在 MOSS-TTS Local 模型上尝试了一轮 reference audio encoder 的 batching 优化。虽然最后没有合入，但过程中发现的问题很有记录价值，很开心能够分享给大家。

## 被缓存容量掩盖的真相

voice cloning 的每个请求在生成前，需要把 reference audio 编码成离散的 codec token。并发升高后，反复 encode 会拖慢 preprocessing。codec encoder 本身支持 batched encode，自然的想法就是把并发请求里的 reference 按长度分桶，同一个桶的攒成 batch 一起编码。[PR #749](https://github.com/sgl-project/sglang-omni/pull/749) 做的就是这个。

初期数据反映良好，并发 16 下 throughput 提升了 23%。我们自信满满，认为方向正确。但事后发现，我们最初进步明显的 commit 实际上改了两个变量，除了 batching 优化，顺手把 reference audio 的缓存上限从 256 调到了 1024，没有做到良好的控制变量。

这是个很严肃的问题，reference encode 的结果有一层缓存，以音频内容为 key，命中后直接复用，默认最多缓存 256 条。测试用的 SeedTTS data 包含 666 个 unique reference 文件，256 的容量容纳不下整个工作集，运行中会反复 evict 和 miss，大量 reference 被迫重算。上限提到 1024 后，666 条全部驻留，命中率接近饱和，重算基本消除。事后发现，23% 的收益其实主要来自这里，与 batching 优化无关。

意识到变量混杂后，我们重新做了严格对照：仅保留 batching 优化，缓存上限两侧都锁定 256，其余条件完全一致。结果是并发 1 吞吐 -0.8%，并发 16 吞吐 -4.4%，两个并发下均为负收益。原因并不复杂，按长度分桶使 batch 碎片化，同一轮并发中能落入同一桶的 reference 数量有限，但 load、resample、分桶、等待、调度等额外开销全部保留。在这个 workload 下，分桶带来的 batch 收益不足以覆盖这些开销。最终 batching 优化没有合入，缓存容量作为独立改动单独合了。

这是本次优化的第一个教训，我们在中学就学过控制变量法，当一个改动带来显著提升时，先确认是否混杂了其他变量。多个变量同时修改，收益归属需要仔细辨别。

## batching 会改变离散 token

在对 batch 尝试进一步性能优化的同时，我们还观察到另一个现象，同一段 reference audio，单独 encode 和放入 batch 中 encode，输出的离散 token 不一致。

有意思的是，这个“不一致”并不是随机性或 padding 引起的问题。首先，它不是随机抖动。固定 batch shape 下重复运行，结果 bit-identical；单独 encode 重复运行，结果也完全一致。差异仅出现在“单独编码”和“批量编码”两条路径之间。其次，与 padding 无关。我们专门测试了相同 frame 数的两条音频放入同一个 batch，仍然出现差异。同长度 batching 下，single 对 batch 的 code mismatch 约 5.8%，而 single 对自身重复编码的 mismatch 为 0。

我们逐层去分析这个不一致的根源，第一个 divergence 出现在 codec encoder 内某个 BF16 Linear 层的输出，根因是 GEMM shape 效应。encoder 和 quantizer 中的 `nn.Linear`、kernel size = 1 的 `WNConv1d`，底层均为矩阵乘法。GEMM 的 M 维等于 batch size 乘以序列长度。batch 变化导致 M 变化，cuBLAS 会为不同的 M 选择不同的 kernel 实现，不同 kernel 的浮点累加顺序不同。BF16 精度下浮点加法不满足结合律，`(a + b) + c` 与 `a + (b + c)` 在最低位可能相差一个 bit。累加顺序改变，结果就在 sub-ULP 量级产生扰动。encoder 中其余操作，norm、RoPE、激活函数、patching，均为逐 token 计算，与 batch shape 无关。GEMM 是这一漂移的唯一来源。

如果仅限于此，sub-ULP 量级的扰动在连续计算中没有实际影响。但 encoder 之后紧接 residual VQ 量化。量化的操作是硬判决：将连续向量映射到 codebook 中距离最近的码字，输出整数索引，不存在中间状态。对于恰好落在两个码字边界附近的帧，sub-ULP 量级的扰动足以将“最近码字”从 A 切换为 B，离散 token 直接翻转。且 RVQ 逐层计算残差，第 0 层翻转后残差改变，误差向后续层传播。

量化器在这里扮演了放大器的角色，将底层不可见的 BF16 数值噪声放大为显式的 token 翻转。这其实和 RL 里的训推不一致高度相关。相关讨论可以参考 Thinking Machines 的 [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)。

## 为什么文本 tokenizer 不存在这个问题

文本 tokenizer 是 symbolic 的。BPE 或 unigram，本质是字符串上的 merge 规则加词表查找，从离散到离散。整个过程不涉及浮点运算，不涉及 GEMM，不涉及跨样本交互。将多个字符串 batch 起来编码，底层实现是循环逐个处理，各算各的，不存在 shape 变化导致 kernel 切换的情况。文本 tokenizer 的确定性是结构赋予的，一个字符串永远映射到同一串 token，token cache 天然安全。文本侧从未需要关注这个问题。

语音不具备这一条件。音频是连续信号，无法进行 symbolic 切分，只能使用 learned neural codec，走 continuous 到 discrete 的路径。而这条路径恰好具备产生问题的两个要素：GEMM 引入的 batch-variant 数值噪声，以及紧随其后的 hard quantization。两者叠加，batch-variant 的 token 漂移就出现了。

真正的分界线在 symbolic tokenizer 与 neural tokenizer 之间。图像领域的 VQ-GAN 及各类 image tokenizer 结构相同，learned encoder 加 codebook，同样继承这一性质。语音只是因为信号连续性最强、最无法回避 neural codec，所以这个特性体现得最为典型。

## 对 RL 场景的影响

纯文本模型，non-determinism 通常到 forward pass 阶段才会遇到。但在 Omni、多模态模型中，input tokenization 阶段就已经引入。输入 token 本身随 batch shape 变化而抖动，此时尚未进入 LLM 主干。非确定性的位置被往前推了整整一个阶段。

落到 RL 场景，train-inference consistency 出现了一个文本侧 RL 从未有过的新来源：rollout 和 reference 在不同并发负载下，输入 token 可能就无法对齐。这个问题的根因与 LLM inference 中经典的数值非确定性问题相同，但出现的位置更靠前，也藏得更深。
