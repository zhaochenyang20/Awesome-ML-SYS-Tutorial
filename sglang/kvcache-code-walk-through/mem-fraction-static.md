# 当 SGLang OOM 的时候，究竟在 OOM 什么？

在开始全力开发 SGLang Omni 两个月之际，我们在先前的 H200 开发发机器加上 H20 CI 机器的基础上，得到了一台崭新的 H100，，能够参与到社区的开发。一台机器的加入自然是令人兴奋的，于是我们火速将新的 H100 机器投入到了生产中，来解决 [#280](https://github.com/sgl-project/sglang-omni/pull/280)。

280 本身是个非常简单的 PR，看上去删改代码都接近 2000 行，实际上逻辑并不复杂。简单来说，我们之前基于 SeedTTS 数据集有一个测试 Omni Model Voice Clone 性能的脚本，还有一个测试 Omni Model Voice Clone 正确性的脚本。但是这两个脚本显然是可以合并的，测量正确性的同时自然可以测量性能，而 280 就是进行这样的合并。毫无疑问，这个任务非常清晰，AI 绝对可以做的非常完美。所以 Hao Jin（AAA 湾区网球金教练）火速完成了 280 本身，然后对着新合并的脚本开始测试。

这里，就出现了非常严肃的问题。金教练测出来合并后的脚本 Qwen3 Omni 的 TTS WER 在全集的分数竟然到了 3.0+，注意到 WER 是一个越搞越坏的分数，这让我们如临大敌。所以，我们马上展开了分析，究竟是什么原因导致正确性出现了如此大规模的回退。因为注意到，在 #280 之前，我们已经建立起了当时看来武装到脚趾甲的 CI，Qwen3 Omni 会在每个 commit 上验证 SeedTTS 的一个 50 samples 的 subset，确保性能和正确性没有一丝一毫的回归。所以见到新的脚本居然出现了显著的性能损失，这让我们有了一系列不太好的猜想：

1. 我们的 CI 是虚假的；之前某个 PR 把 Qwen3 Omni 的精度弄崩了，但是没有捕捉到；
2. 我们的 benchmark 错了；可能在合并两个 benchmark 脚本为一个的过程中写错了什么；

然后，我们在 SGLang Omni 开发群讨论了这个问题，发觉这个事情可能不是我们想的那么麻烦。你看我刚才的叙述中，其实对转换开发机器这个事情轻描淡写。在 PR 280 之前，我们都使用的是 H200 作为开发机器；但是到了 PR 280，我们换了 H100。起初我不以为意，但事实上，这才是问题所在。我记得金教练第一次给我说他在验证 280 的时候，就说之前启动 server 的脚本居然启动不了 server 了。这让我非常费解，当时周末，我一边在 Mountain View 跑步，一边寻思，“这都什么和什么呀”。然后，他拿着 Claude debug 了下，将 SGLang Omni 启动 Qwen3 Omni 的 `mem_fraction_static` 调到了 0.8，server 才成功启动。

现在回想起来，这其实就是问题本身。我默认从 H200 到 H100 的迁移没有任何问题，甚至出现了性能回归，我也没有怪罪到 H100 上。直到和聪明的群友们聊起来，我才意识到，先前 SGLang Omni 的 Thinker 居然 hard coded 了 `mem_fraction_static` 到 0.7。这么一来，80GB 的 H100 根本无法启动 Qwen3 Omni 的 Thinker（30B MoE），进而导致了上述的性能回归。即便我们把 `mem_fraction_static` 调到了 0.8，也才让 server 能够启动，但是还是有相当量的 request 因为显存压力而 fail 了。

这篇简单的博客便会来分享我们解决这一过程的思路。具体来说，我们会讨论：

1. `mem_fraction_static` 参数的实际作用。当 SGLang OOM 时，究竟在 OOM 什么？
2. 为什么 H100 上将 `mem_fraction_static` 调到 0.8 后，即便能启动 server，也有相当量的 request 因为显存原因而 fail？
3. 我们如何基于 SGLang 对 `mem_fraction_static` 的 auto-tune 机制来让 H100 重获新生？
4. 我们如何修改了 SGLang Omni 的 Error Handling 来让用户更清晰地知道问题所在？

致谢：感谢 Huapeng，Yifei，Ratish，金教练和 xuesong，Let's Make SGLang Omni Great Again!

## `mem_fraction_static` 到底在分配什么

SGLang 官方对这个参数的描述是：

> The fraction of the memory used for static allocation (model weights and KV cache memory pool). Use a smaller value if you see out-of-memory errors.

实际上，`--mem-fraction-static` 参数为 `y` 的意义是：**SGLang 在 distributed init 完成、还没开始加载模型时**，把当时的可用显存(下文记为 `pre_model_load_memory`)按 `y : (1 - y)` 切两半 —— `y` 那一份留给 model weights + KV Cache，`(1 - y)` 那一份预留给 activations、CUDA graph buffers 以及同卡上其他应用。

具体公式见 [`model_runner_kv_cache_mixin.py` 中 `ModelRunnerKVCacheMixin._profile_available_bytes`](https://github.com/sgl-project/sglang/blob/4fa3482/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py#L56-L70)(本文撰写时对应上游 `main` 第 56-70 行)：

```python
def _profile_available_bytes(self, pre_model_load_memory: int) -> int:
    post_model_load_memory = get_available_gpu_memory(...)  # 模型加载完后的空闲显存（GB）

    # 核心公式
    rest_memory = post_model_load_memory - pre_model_load_memory * (
        1 - self.mem_fraction_static
    )                                                        # GB
    if self.mambaish_config is not None:
        rest_memory = self.handle_max_mamba_cache(rest_memory)

    return int(rest_memory * (1 << 30))                      # 转换为 bytes
```

随后 `pool_configurator.py` 用 `available_bytes // cell_size_per_token` 切出 `max_total_num_tokens`，若 `<= 0` 就抛出 `Not enough memory. Please try to increase --mem-fraction-static.`(见 `MemoryPoolConfig.__post_init__`)。

关键点：

1. **`reserved` 是基于 `pre_model_load_memory` 算的**，不是基于 GPU 的物理总显存。`pre_model_load_memory` 等于 distributed init 之后剩下的可用显存 —— 已经扣掉了 driver / CUDA context / NCCL buffer / 同卡上其他进程占用的部分。也就是说，**显卡上其他进程占得越多，`pre_model_load_memory` 越小，留给 KV Cache 的额度也跟着缩水**。
2. 当模型本身占了大量显存时，`post_model_load_memory` 会很小，`reserved` 又是按 `pre_model_load_memory` 的比例预留，`rest_memory` 就可能为负，对应启动阶段 OOM。
3. 在某些极端场景(比如别的进程把卡的显存占去一大块、或者 NCCL/通信库本身吃了几个 GB)，即便公式表面看起来"留够了"，由于 `pre_model_load_memory` 已经被扣得很低，KV Cache 可用空间仍然可能不足，运行期再次 OOM。

## 三种 OOM 场景

### 场景一：启动阶段 OOM（KV Cache 分配失败）

如果 `mem_fraction_static` 为 `y`，启动阶段直接 OOM 了，说明 `rest_memory` 为负，显存没分配够。**需要调高 `y`**。

以 H100 80 GB + Qwen3-Omni 60 GB 为例(假设 distributed init / driver / CUDA context 占了 ~2 GB，则 `pre_model_load_memory ≈ 78 GB`)：

```
mem_fraction_static = 0.7（默认）
pre_model_load_memory     ≈ 78 GB
reserved                  = 78 × 0.3 ≈ 23.4 GB
post_model_load_memory    ≈ 78 - 60 = 18 GB
rest = 18 - 23.4 ≈ -5.4 GB → 负数 → OOM

mem_fraction_static = 0.85
reserved                  = 78 × 0.15 ≈ 11.7 GB
rest = 18 - 11.7 ≈ 6.3 GB → 可以分配 KV Cache
```

> 注意：上面 78 GB 只是一个估计值。`pre_model_load_memory` 取决于 driver、CUDA context、NCCL/通信库以及**同卡上其他进程**的占用。如果同卡还有别的服务在跑(例如本机训练任务、监控 daemon)，`pre_model_load_memory` 可能直接掉到 70 GB 甚至更低，调高 `y` 都不一定救得回来 —— 这正是某些"看起来公式留够了，启动还是 OOM"的根本原因。这种场景的正确解法是先腾出同卡显存，再调 `mem_fraction_static`。

### 场景二：运行期间 SGLang 自身 OOM

启动后运行了一段时间，SGLang 本身 OOM 了。这说明 KV Cache 分配不够，或者 activations 空间不足（KV Cache 分太大了）。根据具体情况：

- 如果是 KV Cache 不够（并发请求太多，序列太长）：**调高 `y`**
- 如果是 activations/CUDA graph 空间不够（KV Cache 挤占了太多）：**调低 `y`**

实际中，对于 Omni 这种大模型在紧张显存下的场景，更常见的是 KV Cache 太小导致部分请求失败，server 不会直接崩溃，但**失败请求的输出质量极差**——这正是 Hao 在 [#280](https://github.com/sgl-project/sglang-omni/pull/280) 中遇到的情况：WER 飙到 3.0+，很可能就是一部分请求因为 KV Cache 不足而 OOM，产出了错误的音频。

### 场景三：其他应用把 SGLang 顶 OOM

如果启动后，同一张显卡上有其他应用抢占显存，把 SGLang 顶 OOM 了，说明 KV Cache 分配过多，没给其他应用留空间。**需要调低 `y`**。

### 总结

| 场景 | 发生阶段 | 调参方向 | 原因 |
|---|---|---|---|
| 启动 OOM（KV Cache 分配失败） | 启动时 | **调高** `y` | `reserved` 太大，`rest_memory` 为负 |
| SGLang 自身 OOM（KV Cache 不够） | 运行时 | **调高** `y` | KV Cache 太小，高并发/长序列放不下 |
| SGLang 自身 OOM（activations 不够） | 运行时 | **调低** `y` | KV Cache 太大，activations 被挤占 |
| 其他应用抢显存 | 运行时 | **调低** `y` | 给其他应用留的空间不足 |

## H100 vs H200：为什么换卡就好了

同样的 Qwen3-Omni 60 GB 模型，默认 `mem_fraction_static=0.7`，下面用近似的 `pre_model_load_memory ≈ 物理显存 - 2 GB` 估算(实际值见上一节注意事项)：

**H100 80 GB**：
```
pre_model_load_memory  ≈ 78 GB
reserved               = 78 × 0.3 ≈ 23.4 GB
post_model_load_memory ≈ 78 - 60 = 18 GB
rest = 18 - 23.4 ≈ -5.4 GB → Init OOM
```

**H200 141 GB**：
```
pre_model_load_memory  ≈ 139 GB
reserved               = 139 × 0.3 ≈ 41.7 GB
post_model_load_memory ≈ 139 - 60 = 79 GB
rest = 79 - 41.7 ≈ 37.3 GB → 充裕，KV Cache 大
```

H200 不仅显存更大（141 vs 80 GB），HBM3e 带宽也更高（4.8 vs 3.35 TB/s），LLM 推理是 memory-bound 的，带宽直接决定 token 生成速度。

这也解释了为什么 CI 上 96 GB 的 H20 恰好绕开了 OOM：

```
pre_model_load_memory  ≈ 94 GB
reserved               = 94 × 0.3 ≈ 28.2 GB
post_model_load_memory ≈ 94 - 60 = 34 GB
rest = 34 - 28.2 ≈ 5.8 GB → 刚好能跑
```

## 对 [#261](https://github.com/sgl-project/sglang-omni/pull/261) 和 [#280](https://github.com/sgl-project/sglang-omni/pull/280) 的影响

基于以上分析，我认为一个非常重要的 debug 方向是：**直接换 H200 跑一模一样的测试**，验证分数和速度会不会好起来。

- 对于 [#261](https://github.com/sgl-project/sglang-omni/pull/261)（MMSU benchmark）：华鹏在 H100 上 `mem_fraction_static=0.85` 勉强启动，但 KV Cache 只有 ~6 GB，并发能力很弱，推理很慢。换 H200 后 KV Cache 有 ~37 GB，并发和速度应该会有质的提升。
- 对于 [#280](https://github.com/sgl-project/sglang-omni/pull/280)（SeedTTS WER）：Hao 在 H100 上 WER 3.0+，很可能是 KV Cache 不足导致部分请求 OOM，产出了错误音频拉高了整体 WER。换 H200 后这些请求应该不会再 OOM。

如果换了 H200 确实有更好的表现，需要跟进以下事项：

1. **修改 SGLang 文档和 error message**：当前文档一刀切地建议「调小」，需要区分两种 OOM 场景，给用户更清晰的调控指南。
2. **H100 上的 Omni 开发方案**：对 30B MoE 模型，H100 80 GB 确实不友好。可能需要支持 thinker TP（tensor parallelism）跨多卡，让单卡放得下模型 + 足够的 KV Cache。CI 上 96 GB H20 恰好绕开了这个问题，但实际用户大概率用 H100。
3. **先在 H200 上把 [#261](https://github.com/sgl-project/sglang-omni/pull/261) 和 [#280](https://github.com/sgl-project/sglang-omni/pull/280) 解决**，H100 的问题根据上述调查结果再定方案。

## 相关源码

> 上游 `main` 还在快速迭代，下面行号以本文撰写时为准，阅读时建议直接 `grep` 函数名定位。

| 文件 | 内容 |
|---|---|
| `sglang/srt/model_executor/model_runner_kv_cache_mixin.py` 中 `_profile_available_bytes` (~L56-L70) | KV Cache 分配核心公式 |
| `sglang/srt/model_executor/pool_configurator.py` 中 `MemoryPoolConfig.__post_init__` (~L39-L44) | Init OOM 错误抛出位置 |
| `sglang/srt/model_executor/model_runner.py` 中 `init_torch_distributed` 末尾 (~L1083-L1108) | `pre_model_load_memory` 的采集点 |
| `sglang/srt/server_args.py` | `mem_fraction_static` 自动计算逻辑 |
| `sglang_omni/engines/ar/sglang_backend/server_args_builder.py` | SGLang-Omni 默认值（0.7） |
