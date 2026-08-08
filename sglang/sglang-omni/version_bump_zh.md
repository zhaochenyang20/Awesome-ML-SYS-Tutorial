# API 对齐到浮点结合律：SGLang Omni 的 backbone 升级

前段时间我们 SGLang Omni 依赖的 SGLang Backbone 从 `0.5.12.post1` 升级到了 `0.5.16`。通常认为，我们只需要改掉重命名的 API，更新一下集成代码，然后就完事了。不过最后其实改动的大小堪称灾难性，跨越了六个 SGLang 版本，Transformers 从 `5.6.0` 升级到了 `5.12.1`，最后在 [PR #1183](https://github.com/sgl-project/sglang-omni/pull/1183) 里改动了 162 个文件。


> PS：其实从我个人对工程审美的理解来看，SGLang Omni 最好作为 SGLang 的一层上游抽象，尽量对 SGLang 黑盒使用，甚至都不需要固定在任意一个具体版本上。类似于我们在 slime 和 miles 中，希望通过约定 SGLang 的下游接口，再通过 SGLang 的 CI 进行保护，这样随时能够使用最新的 SGLang。但是很遗憾，大多数上游框架都会侵入性修改下游作为依赖，导致不得不 pin 在某个具体的 SGLang 版本上，并且对这个版本的接口进行修改。而每次修改实际上就是要把这些在特定版本上的侵入性代码重置到更新的版本上。

升级的 PR diff 巨大无比。如我上文所说的那样，SGLang Omni 很难做到对 SGLang 仅仅进行一层简单轻薄的封装：它自己拥有一条多阶段流水线、一部分 scheduler 循环、model runner 的集成、流式状态，以及进程放置逻辑。上下游两个系统之间的运行契约远不是一两个核心接口能够保护的。Scheduler 的兼容性要求一套稳定的执行协议；Qwen3-Omni 要求维护浮点运算的顺序；而 MOSS 各阶段合并进同一个进程之后，显存必须按构造顺序计算，不能再把进程预算当成一个与时间无关的数字...

这些事情构成了本次升级的噩梦，也是我们在此反思，希望以后的升级能够更为轻量。

## Scheduler API 的变化

SGLang `0.5.16` 改动了 scheduling 与 model execution 之间的边界。batch selection 现在返回 `NextBatchPlan`，`ForwardBatch` 由实时的 `ScheduleBatch` 初始化，output processing 消费的是 `GenerationBatchResult`。Omni 没法靠直接调用 SGLang 的 scheduler 循环来接住这些变化，因为它有自己的多阶段事件循环、模型专属 runner、流式行为和结果传递路径。

难的地方不在于把一种结果类型转成另一种。Scheduler 会在迭代之间对请求做过滤、合并、retract 或复用，所以挂在某个早期批次视图上的状态，可能已经描述不了下一次真正执行的批次。同时，下一步模型计算需要的 device 侧结果，和用于结束判定、流式输出、构造响应的 host 侧结果，生命周期并不相同。把它们当成同一个可以互换的对象，等于用新的方法名保留了旧集成的假设。

最终实现把这层适配收在一个很窄的执行边界里，删掉了 `0.5.12` 时代遗留的 `output_ids` 旁路，也删掉了一条走上游 `run_batch` 的不可达 fallback。模型专属 runner 现在走的是同一条批次构造和结果发布路径，不再各自带一套并行的兼容行为。这里能得到的经验比"上游 API 会变"更具体：当一个框架改变了可变执行状态归谁所有、以及这个状态什么时候可以被消费，那么只对齐新的函数签名、却保留旧的数据流，仍然是错的。

【TODO：我觉得这一段写的不够清晰，可以再写的更详细一些，读者读完了之后似乎没有学到什么。一个写作角度是，可以写 0.5.12 的时候，scheduler 的某一个对象是如何操作的，然后到 0.5.16，这个对象变成了什么样子，从而导致 SGLang Omni 需要做出什么样的更改，这样来写，可能让读者能够学到的更多。】

## 几个浮点运算改变了 Qwen3-Omni

Scheduler 这条路跑通之后，Qwen3-Omni 给出了一个更有迷惑性的故障。模型能正常启动，也能接住同样的请求，但 MMMU 的结果掉了。这种故障很容易被归到预处理、图像缩放、tokenizer 变化、模型权重、rotary position，或者干脆归到 GPU 的非确定性上，所以我们把两套栈逐层对比了一遍。

输入完全一致。`input_ids`、attention mask、pixel values、image grid 元信息、patch embedding 的输出、rotary position IDs 全都对得上。第一个差异出现在位置编码进入 vision encoder 之后，然后穿过第一个 vision block，一直传到最终的 image embedding 和 deepstack embedding。在七个真实样本上，最终 image embedding 的最大绝对差异大约在 `0.156` 到 `0.359` 之间。

根源只是几个浮点运算。Transformers `5.6` 用 CPU FP32 的行为构造双线性插值坐标，把插值权重转成位置编码表的 dtype（通常是 BF16），再按一个明确的顺序把四个角的 embedding 加起来：

```python
corners = pos_embed(indices) * weights[:, :, None]
result = corners[0] + corners[1] + corners[2] + corners[3]
```

Transformers `5.12` 把这段计算挪进了一条公共路径。它用不同的方式生成插值状态，在乘法过程中保留 FP32 权重，然后用一次 sum 归约四个角。两种实现在数学上描述的是同一个双线性插值，但 BF16 的乘法和加法不满足结合律。改变中间 dtype 和累加顺序，就改变了预训练 vision tower 看到的位置编码。

修复刻意做得很局部。[`Qwen3OmniMoeVisionEncoderCompat`](https://github.com/sgl-project/sglang-omni/blob/a8d3dd14a2784cea51937936301043f1735bfda7/sglang_omni/models/qwen3_omni/components/vision_compat.py#L13-L146) 保留了 Transformers `5.12.1` 的 encoder 结构、装饰器、输出类型、vision block 和 deepstack 行为，只把插值那段算术换回 checkpoint 原始技术栈使用的 `5.6` 版本。改完之后，预处理张量、抓下来的 vision 中间张量、最终 embedding 和 deepstack embedding 都和参考实现逐比特一致；50 样本的 MMMU 卡口回到 31/50，也就是 62%。

这是整次升级里最清晰的一个结论。**对一个预训练模型来说，兼容性包含解释它权重的那个浮点程序。** 一个依赖可以完整保留所有公开 API 和张量形状，同时通过设备放置、算子融合、归约顺序或中间精度，在数值上改变这个模型。

## EOS 不等于请求已经结束

Scheduler 的这轮 review 还顺带暴露了一个早就存在的生命周期问题。它不是 SGLang `0.5.16` 引入的，但 scheduler 的重写让它没法再被安全地忽略。Omni 把模型专属的请求数据挂在 SGLang 的 request 上，而这份请求数据又持有指回 request 的引用：

```text
Req → Omni request data → Req
```

对 TTS 和 omni 模型来说，这份请求数据可能持有参考音频、输入 embedding、hidden state、流式缓冲区，以及模型专属的解码状态。环留在那里，普通的引用计数就没法在终止时释放它，只能等 Python 的循环回收器之后来发现。把这个链接清掉听起来很简单，直到我们开始考虑在多阶段流水线里"终止"到底意味着什么。

一个自回归请求可能已经结束了，而上游阶段还有一个 stream chunk 在路上。同一个请求可能同时出现在 running batch、刚完成的 batch、一个异步 pending step，以及 stream ingress 缓冲区里。如果在 model runner 冲刷完最后一段缓冲音频之前就把请求数据摘掉，最后一个 chunk 就丢了。如果在 in-flight 的 stream ingress 还没落定之前就摘掉，一个迟到的 chunk 会被误认为是准入之前的数据，然后留在某个 pending 结构里。如果 abort 在正常终止流程正在清理这个请求的时候到达，两边可能都去跑一遍模型清理，也可能都以为对方会做。

最终的 scheduler 代码把终止处理当成一次所有权交接，而不是一次指针清零。在请求准入锁的保护下，正常完成路径先"认领"这个请求，保证只有一条路径产出终止输出。然后它让 model runner 冲刷掉剩余的流式状态，构造终止结果，跑模型专属的 finish 回调，最后才摘掉 Omni 的请求数据，并把请求记为已完成。在这个边界之后到达的 stream chunk 会被直接丢弃，而不是重新造出 pending 状态。Abort 走同一把锁：如果它在 detach 之前到达，终止流程会看到它并完成 abort 清理；如果它在 detach 之后到达，abort 路径就知道已经没有终止所有者了，于是自己做清理。

关键性质不是"每条终止路径都调用了同一个 helper"，而是锁和数据挂载这两件事合在一起，在两种可能的交错顺序下都能唯一确定一个清理所有者。这一点在最终的 [`stream_output`](https://github.com/sgl-project/sglang-omni/blob/a8d3dd14a2784cea51937936301043f1735bfda7/sglang_omni/scheduling/omni_scheduler.py#L1326-L1424) 路径里能直接看到：最后一段流式数据在 detach 之前被排空，而完成记录则把这个请求对后续 ingress 关闭掉。

这是多阶段服务里反复出现的问题。"模型生成了 EOS"和"流水线已经处理完这个请求"是两个不同的事件。清理属于第二个边界，而这个边界必须从所有并发持有者推出来，而不是从模型的 finish 标志上猜出来。

## 合并后的 MOSS 流水线为什么会 OOM

MOSS-TTS Local 暴露的是另一种所有权错误。它的默认拓扑给预处理、自回归引擎和 vocoder 分配 GPU 显存比例：

```text
preprocessing    0.15
AR engine        0.67
vocoder          0.18
```

当这些阶段跑在各自独立的进程里时，一个阶段的 fraction 就等于它所在进程的 fraction。当运维把 vocoder 合并进 pipeline 进程之后，这个完整进程声明的累计预算就变成了 `1.0`。这不是 SGLang 的 `mem_fraction_static`；它表示的是该进程里所有模型都加载完之后，各阶段预算之和。我们第一版实现在自回归引擎为 KV cache 做显存 profiling 的时候，把这个最终总数直接传了进去。

算式看着合理，实际是错的。各阶段是顺序构造的：先加载预处理，然后自回归引擎分配权重和 KV cache，最后才构造 vocoder。把最终的 `1.0` 预算给自回归引擎，等于允许它去占用一个还不存在的阶段的显存。在 H100 上，它算出大约 `66.22 GiB` 可以用于 KV；等到 vocoder 后面再要 `942 MiB` 的时候，只剩下 `442 MiB` 左右，启动直接失败。

每个构造点上正确的值，是最终进程预算的前缀和。预处理看到的是 `0.15`；加上自回归引擎之后，声明的累计预算是 `0.15 + 0.67 = 0.82`；只有等 vocoder 也加载完，这个累计预算才达到 `1.0`。最终实现按阶段构造顺序算出这些前缀和，并注入到每一个会做进程级显存 profiling 的 factory 里。实现本身足够短，可以直接读 [`_attach_process_memory_fraction_defaults`](https://github.com/sgl-project/sglang-omni/blob/a8d3dd14a2784cea51937936301043f1735bfda7/sglang_omni/pipeline/mp_runner.py#L186-L214)。

用 `0.82` 之后，自回归引擎把 KV 分配定在大约 `51.89 GiB`。vocoder 加载完成后，整个进程占用 `73.44 GiB`，对应它最终 `79.65 GiB` 的预算，之前起不来的合并拓扑成功服务了一个正常请求。

这个问题很容易被总结成"给 vocoder 留点显存"，但这个说法丢掉了真正有用的设计原则。**当一个进程是被增量组装出来的，它的预算就是分阶段的。** 放置 fraction 描述的是最终拓扑；而显存 profiling 需要的是截至当前构造点、已经加载的那些对象所对应的 fraction。测量的范围和预算的范围必须对齐。

## 这次升级要求我们验证什么

这些问题需要不同形式的证明，因为它们失败的方式不一样。Scheduler 桥接层是对着 `0.5.16` 真实的执行路径核对的，而不是从重命名的方法名推出来的。Qwen 的修复要求中间张量和最终张量逐比特一致，因为端到端的精度分数能看出回归，却定位不到回归在哪。生命周期的改动要求我们真的去跑 abort 和 stream 的各种交错，并观察请求结束后显存是否回落。MOSS 的修复要求在做 KV 决策的那个启动阶段精确读取 GPU 显存。

这个过程也帮我们把真正属于这次 pin 升级的工作，和只是碰巧在附近的改动分开了。执行桥接层和 Qwen 兼容路径是新依赖栈要求的。请求引用环是个老问题，但在动终止所有权的同时不修它，会让升级后的 scheduler 变得不安全。MOSS 的故障来自这个分支带进来的拓扑与记账的交互。其他没有同样清晰兼容性理由的性能想法，都被挡在这次升级之外。

回头看，这次升级之所以难，原因很简单：SGLang Omni 依赖的东西远不止 SGLang 导出的那层 Python 接口。它依赖一个 batch 在什么时候可以被改动、下一个 token 在迭代之间归谁所有、哪些浮点运算定义了一个模型、一个多阶段请求在什么时候才真正变得不可达，以及做显存 profiling 的时候有哪些对象是驻留的。

所以，一个 pin 住的运行时版本不只是一个可复现的安装选择。它是在声明：这些假设已经对着某一个具体的运行时验证过了。移动这个 pin，就意味着要重新把它们全部验证一遍。