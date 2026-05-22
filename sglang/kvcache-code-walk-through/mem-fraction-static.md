# 当 SGLang OOM 的时候，究竟在 OOM 什么？

在开始全力开发 SGLang Omni 两个月之际（大概今年 4 月），我们在先前的 H200 开发发机器加上 H20 CI 机器的基础上，得到了一台崭新的 H100，能够参与到社区的开发。一台机器的加入自然是令人兴奋的，于是我们火速将新的 H100 机器投入到了生产中，来解决 [#280](https://github.com/sgl-project/sglang-omni/pull/280)。

280 本身是个非常简单的 PR，看上去删改代码都接近 2000 行，实际上逻辑并不复杂。简单来说，我们之前基于 SeedTTS 数据集有一个测试 Omni Model Voice Clone 性能的脚本，还有一个测试 Omni Model Voice Clone 正确性的脚本。但是这两个脚本显然是可以合并的，测量正确性的同时自然可以测量性能，而 280 就是进行这样的合并。毫无疑问，这个任务非常清晰，AI 绝对可以做的非常完美。所以 Hao Jin（AAA 谷歌网球金教练）火速完成了 280 本身，然后对着新合并的脚本开始测试。

这里，就出现了非常严肃的问题。金教练测出来合并后的脚本 Qwen3 Omni 的 TTS WER 在全集的分数竟然到了 3.0+，注意到 WER 是一个越高越坏的分数，这让我们如临大敌。所以，我们马上展开了分析，究竟是什么原因导致正确性出现了如此大规模的回退。注意到，在 #280 之前，我们已经建立起了当时看来武装到脚趾甲的 CI，Qwen3 Omni 会在每个 commit 上验证 SeedTTS 的一个 50 samples 的 subset，确保性能和正确性没有一丝一毫的回归。所以见到新的脚本居然出现了显著的性能损失，这让我们有了一系列不太好的猜想：

1. 我们的 CI 是虚假的；之前某个 PR 把 Qwen3 Omni 的精度弄崩了，但是没有捕捉到；
2. 我们的 benchmark 错了；可能在合并两个 benchmark 脚本为一个的过程中写错了什么；

然后，我们在 SGLang Omni 开发群讨论了这个问题，发觉这个事情可能不是我们想的那么麻烦。你看，在我刚才的叙述中，其实对转换开发机器这个事情轻描淡写。在 PR 280 之前，我们都使用的是 H200 作为开发机器；但是到了 PR 280，我们换了 H100。起初我不以为意，但事实上，这才是问题所在。我记得金教练第一次给我说他在验证 280 的时候，就说之前启动 server 的脚本居然启动不了 server 了。这让我非常费解，当时周末，我一边在 Mountain View 跑步，一边寻思，“这都什么和什么呀”。然后，他拿着 Claude debug 了下，将 SGLang Omni 启动 Qwen3 Omni 的 `mem_fraction_static` 调到了 0.8，server 才成功启动。

现在回想起来，这其实就是问题本身。我默认从 H200 到 H100 的迁移没有任何问题，甚至出现了性能回归，我也没有怪罪到 H100 上。直到和聪明的群友们聊起来，我才意识到，先前 SGLang Omni 的 Thinker 居然 hard coded 了 `mem_fraction_static` 到 0.7。这么一来，80GB 的 H100 根本无法启动 Qwen3 Omni 的 Thinker（30B MoE），进而导致了上述的性能回归。即便我们把 `mem_fraction_static` 调到了 0.8，也才让 server 能够启动，但是还是有相当量的 request 因为显存压力而 fail 了。

这篇简单的博客便会来分享我们解决这一过程的思路。具体来说，我们会讨论：

1. `mem_fraction_static` 参数的实际作用。当 SGLang OOM 时，究竟在 OOM 什么？
2. 为什么 H100 上将 `mem_fraction_static` 调到 0.8 后，即便能启动 server，也有相当量的 request 因为显存原因而 fail？
3. 我们如何基于 SGLang 对 `mem_fraction_static` 的 auto-tune 机制来让 H100 重获新生？
4. 我们如何修改了 SGLang Omni 的 Error Handling 来让用户更清晰地知道问题所在？

致谢：感谢 Huapeng，Yifei，Ratish，金教练，xuesong 还有 yifei。

## `mem_fraction_static` 到底在分配什么

SGLang 官方对这个参数的描述是：

> The fraction of the memory used for static allocation (model weights and KV cache memory pool). Use a smaller value if you see out-of-memory errors.

记 `--mem-fraction-static = y`。`y` 表示：在加载模型权重之前，也即 torch.distributed.init 之后测得的可用显存按 `y : (1-y)` 切成两部分，`y` 部分给 model weights 和 KV cache pool，`(1-y)` 则预留给 activations、CUDA graph buffers 以及推理过程中的其它动态开销。

这里有一个和官方文档不太一致，却和我们 H100 踩坑直接相关的地方：文档笼统地说 OOM 就「调小 `y`」。但在我们的实际操作中，我们将 `y` 从 0.7 调到 0.8 才勉强把 server 启动起来。实际上，出现 OOM 时，究竟是调小还是调大 `y` 有更深的考虑。要弄清楚这件事，得先弄清 SGLang 在启动时的两个决定变量 `pre_model_load_memory` 和 `post_model_load_memory`。

具体来说，SGLang 使用同一个工具函数 [`get_available_gpu_memory`](https://github.com/sgl-project/sglang/blob/4fa3482/python/sglang/srt/utils/common.py#L494-L576)来读取当前可使用的显存大小：先 `torch.cuda.empty_cache()`，再用 `torch.cuda.mem_get_info` 读当前空闲显存，换算成 GB；若是 TP 等多卡场景，还会对各 rank 做一次 MIN，避免某张卡特别紧却被其它卡「平均」掉。

1. `pre_model_load_memory`：在 [`init_torch_distributed`](https://github.com/sgl-project/sglang/blob/4fa3482/python/sglang/srt/model_executor/model_runner.py#L1139-L1164) 末尾：distributed init 刚结束，尚未 `load_model`。返回值记为 `pre_model_load_memory`。它不是 H100 的物理 80 GB，而是「此刻驱动认为这张卡还能给我用多少」——已经扣掉了 CUDA context、NCCL buffer、同卡上其它进程占用的部分。
2. `post_model_load_memory`：在 [`_profile_available_bytes`](https://github.com/sgl-project/sglang/blob/4fa3482/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py#L56-L74) 中，权重已经 `load_model` 完成，LoRA 相关的预分配（若有）也做完了，但 KV pool 还没建。再调用 [`get_available_gpu_memory`](https://github.com/sgl-project/sglang/blob/4fa3482/python/sglang/srt/utils/common.py#L494-L576) 函数，记为 `post_model_load_memory`。

由此可见，我们几乎可以认为 `post_model_load_memory - pre_model_load_memory` 就是 weights 和 LoRA 相关的预分配所占用的显存大小。虽然某种意义上，我们可以通过直接计算模型的参数量 * 数据精度来得到 model weights，但是 `post_model_load_memory - pre_model_load_memory` 事实上更加准确。基于此，我们将 `pre_model_load_memory` 分为两个逻辑区域：

- 静态区（weights + KV + lora allocation etc.）：`y × pre_model_load_memory`
- 动态区（activations 等）：`(1-y) × pre_model_load_memory`

从静态区域扣除掉 weights 和 LoRA 相关的预分配所占用的显存大小，剩下的就是 KV 可用显存：

```
memory_for_kv = y × pre_model_load_memory - (pre_model_load_memory - post_model_load_memory)
```

对应源码（[`model_runner_kv_cache_mixin.py`](https://github.com/sgl-project/sglang/blob/4fa3482/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py#L56-L74)）：

```python
def _profile_available_bytes(self, pre_model_load_memory: int) -> int:
    post_model_load_memory = get_available_gpu_memory(...)

    rest_memory = post_model_load_memory - pre_model_load_memory * (
        1 - self.mem_fraction_static
    )
    ...
    return int(rest_memory * (1 << 30))
```

`pool_configurator.py` 再把返回值（GB）换成 `max_total_num_tokens`；若 `<= 0`，就抛 `Not enough memory. Please try to increase --mem-fraction-static.`——这正是我们在 H100 上看到的启动失败。

有了这段讨论，我们进一步来考虑，当 OOM 出现时，究竟应该调大还是调小 `mem_fraction_static`？实际上，这并没有标准答案，因为出现 OOM 的原因不同，决定了我们不同的调参方向。

## 三种典型的 OOM 场景

### 启动阶段 OOM（KV Cache 分配失败）

启动阶段直接失败，`rest_memory`为负，也即静态区 `y × pre_model_load_memory` 扣完 weights 后，连 KV pool 都分不出来。此时，与官方文档说的「调小 `y`」正好相反，这里要调高 `y`。

以 H100 80 GB + Qwen3-Omni 60 GB 为例（假设 init 吃掉约 2 GB，则 `pre ≈ 78 GB`）：

```
mem_fraction_static = 0.7（默认）
pre                     ≈ 78 GB
reserved = (1-y)×pre    ≈ 23.4 GB
post ≈ 78 - 60          ≈ 18 GB
memory_for_kv ≈ -5.4 GB → OOM

mem_fraction_static = 0.85
reserved                ≈ 11.7 GB
memory_for_kv ≈ 6.3 GB → 可以分配 KV Cache
```

注意到，上文的 78 GB 只是一个估计值。`pre_model_load_memory` 取决于 driver、CUDA context、NCCL/通信库以及同卡上其他进程的占用。如果同卡还有别的服务在跑(例如本机训练任务、监控 daemon)，`pre_model_load_memory` 可能直接掉到 70 GB 甚至更低，调高 `y` 都不一定救得回来 —— 这正是某些「看起来公式留够了，启动还是 OOM」的根本原因。这种场景的正确解法是先腾出同卡显存，再调 `mem_fraction_static`。

### 运行期间 SGLang 自身 OOM

启动后运行了一段时间，SGLang 本身 OOM 了。其实这是个很有趣的问题，SGLang 本身不会因为 KV cache 不够而 OOM。当某个 request 超长，kv cache 容不下这尊大佛的时候，SGLang 的策略是将这条 request retract，或者叫做退回 scheduler 队列。等到其他 request 的 kv cache 被清除出去了很多，再把这条请求重新组到 batch 中。所以，SGLang 其实不会因为 KV Cache 的分配不够（`y` 比较小，但是又容得下 weights）而 OOM。所以，如果是 SGLang 在运行时 OOM 了，只有两种可能：

1. activation 没有留够，在运行过程中，activation 直接挤爆了当前 GPU；这是个逻辑上可能发生的事情。注意，我前面说了，SGLang 不会因为 KV Cache 没分够而 OOM，但是确实可能因为 activation 没有留够而 OOM。这种情况下，理应调小 `y`，来让更多的显存分配给 activation。
2. 其实 KV cache 和 activation 都很友善，但是显卡上还有其他进程。比如说，~~假如你是学生，你实验室的同学在 GPU 上挂着令人费解的显存波动巨大的任务～～，或者说你在 RL workload 上，然后 training engine 这边的显存存在高强度的碎片或者泄露。这种情况下，逻辑上应该~~严加管理其他进程～～，但实际上我们可能仅仅可以做的就是压低 SGLang 的 mem static fraction，恰好能够 load 起来 weights，容得下更少的 KV Cache，以祈祷 activation 不会和其他进程一起挤爆 server。

这就是 SGLang 参数文档提到的 `Use a smaller value if you see out-of-memory errors.` 所以说，参数文档是对的，但是不完全对。如果 weights 直接 load 失败，应该调高 `y`，如果是在运行时 OOM，应该调小 `y`。

### Multi-Modal Encoder OOM Error

回到 SGLang Omni，还有高手 🤣 我之前提到过，当我们将 Qwen3 Omni 的 Thinker Mem Fraction 调到 0.8 后，server 成功启动。结果，这个参数测出来的 seedTTS 的 WER 还是很高，仔细一查，有大量的 request 因为 OOM 而失败，既不是 weights 直接 load 失败，也不是在运行时 activation 没有留够。这就令人费解了，仔细追查，发现是 video encoder 和 audio encoder 的显存没有分配到位，直接导致了某些 request 根本没有发给 Thinker，自然 WER 精度就很难看了。对于此，我们目前的解决方案不算优雅，暂时用了 `encoder_mem_reserve` 参数，来提前为了 video encoder 和 audio encoder 预留显存。当然，之后应该会和 SGLang VLM 的处理逻辑对齐。

## H200 vs H20 vs H100：Hopper 卡的临门一脚

这就是故事的全貌了，我们长期在 H200 开发，CI 机器使用的是 96GB 的 H20，恰好，`thinker_mem_static` 倘若被写死为 0.7，留给 KV cache 的空间大致为 `96 * 0.7 - 60 = 7.2 GB`，不多不少，真能跑起来。当然，现在我们肯定没有把 `thinker_mem_static` 写死为 0.7，而是参考 SGLang 做了 auto-tune 机制，尽量让每个机器都能够打满显存。对于 H100，现在 SGLang Omni 也能美丽运行，但是留给 multi-modal encoder 的显存空间还是比较逼仄，在一些 long sqeuence 的 video input 上，还是会有 OOM 的情况发生。当然，这符合我们的预期，我们也会进一步做些许优化，可能支持 FP8，或者学习下 SGLang 本身先进的显存管理机制😂

写到这里，这篇小短文基本结束了。南加州的五月日渐炎热。我想起来，去年大概就是这时候，我离开学校，去业界实习，六月初决定了要离开学术界，然后到了 25 年 11 月，正式开始全职工作。有时候想想，工业界真的非常不一样，创业更是如此。那些以前在学校里觉得显赫的名人，真的从工业级的视角，再去看看，便也有了诸多的不如意处，到底也难以“辩护荣辱之境，定乎内外之分”。至于我呢，参加工作的半年，心态上和认知上改变不可谓不大。SGLang Omni 是正式工作后的第一个项目，虽然过程多有波折，但是感谢各位朋友的支持和全力开发，一段新的旅程就此展开。

> 孩子们终其一生都将记得父亲如何在桌首庄严入座，被长期熬夜和苦思冥想折磨得形销骨立，因激动而颤抖着，向他们透露自己的发现：“地球是圆的，就像个橙子。”
