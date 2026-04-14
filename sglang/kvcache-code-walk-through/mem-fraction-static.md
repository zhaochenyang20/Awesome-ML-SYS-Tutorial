# `mem_fraction_static` 深入解析：从一次 OOM 调参说起

## 背景

我最近意识到了一个严重的问题。Hao Jin 在 H100 上测 SeedTTS（[#280](https://github.com/sgl-project/sglang-omni/pull/280)）拿到的 WER 高达 3.0+，华鹏在 H100 上跑 MMSU benchmark（[#261](https://github.com/sgl-project/sglang-omni/pull/261)）时 Qwen3-Omni 在默认 `mem_fraction_static=0.7` 下直接启动 OOM。这两个问题很可能有同一个根因：**H100 80 GB 对 60 GB 的 Qwen3-Omni 模型来说显存太紧了**。

之前我在 S2 Pro 上就发现过：即便服务器 OOM 了，server 也会继续完成其他请求，但触发了 OOM 的那些请求的 WER 会非常高。今天华鹏跟我说，他在 H100 上需要把 `mem_fraction_static` 调到 0.85 才能启动 Qwen3-Omni，即便如此推理速度还是很慢。这让我对 SGLang 的 `mem_fraction_static` 有了更深入的认知。

## `mem_fraction_static` 到底在分配什么

SGLang 官方对这个参数的描述是：

> The fraction of the memory used for static allocation (model weights and KV cache memory pool).

实际上，`--mem-fraction-static` 参数为 `y` 的意义是：**当前显卡还完全没有启动 SGLang 时**，假设显卡的总显存是 `x`，那么 `x * y` 的显存会分给 SGLang 的 model weights 和 KV Cache，余下 `x * (1 - y)` 的显存给 activations、CUDA graph buffers 以及其他应用。

具体来说，KV Cache 的大小由以下公式决定（`model_runner_kv_cache_mixin.py` 第 111-144 行）：

```python
def profile_max_num_token(self, total_gpu_memory: int):
    available_gpu_memory = get_available_gpu_memory(...)  # 模型加载完后的空闲显存

    # 核心公式
    reserved = total_gpu_memory * (1 - mem_fraction_static)  # 从总显存算出预留区
    rest_memory = available_gpu_memory - reserved             # 空闲显存 - 预留 = KV Cache 可用
    max_num_tokens = int(rest_memory * (1 << 30)) // cell_size_per_token
```

关键点：**`reserved` 是从总显存算的，不是从空闲显存算的**。当模型本身占了大量显存时，`reserved` 可能超过实际空闲显存，导致 `rest_memory` 为负。

## 三种 OOM 场景

### 场景一：启动阶段 OOM（KV Cache 分配失败）

如果 `mem_fraction_static` 为 `y`，启动阶段直接 OOM 了，说明 `rest_memory` 为负，显存没分配够。**需要调高 `y`**。

以 H100 80 GB + Qwen3-Omni 60 GB 为例：

```
mem_fraction_static = 0.7（默认）
reserved = 80 × 0.3 = 24 GB
available = 80 - 60 = 20 GB（模型加载完后空闲）
rest = 20 - 24 = -4 GB → 负数 → OOM

mem_fraction_static = 0.85
reserved = 80 × 0.15 = 12 GB
rest = 20 - 12 = 8 GB → 可以分配 KV Cache
```

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

同样的 Qwen3-Omni 60 GB 模型，默认 `mem_fraction_static=0.7`：

**H100 80 GB**：
```
reserved = 80 × 0.3 = 24 GB
available = 80 - 60 = 20 GB
rest = 20 - 24 = -4 GB → Init OOM
```

**H200 141 GB**：
```
reserved = 141 × 0.3 = 42.3 GB
available = 141 - 60 = 81 GB
rest = 81 - 42.3 = 38.7 GB → 充裕，KV Cache 大
```

H200 不仅显存更大（141 vs 80 GB），HBM3e 带宽也更高（4.8 vs 3.35 TB/s），LLM 推理是 memory-bound 的，带宽直接决定 token 生成速度。

这也解释了为什么 CI 上 96 GB 的 H20 恰好绕开了 OOM：

```
reserved = 96 × 0.3 = 28.8 GB
available = 96 - 60 = 36 GB
rest = 36 - 28.8 = 7.2 GB → 刚好能跑
```

## 对 [#261](https://github.com/sgl-project/sglang-omni/pull/261) 和 [#280](https://github.com/sgl-project/sglang-omni/pull/280) 的影响

基于以上分析，我认为一个非常重要的 debug 方向是：**直接换 H200 跑一模一样的测试**，验证分数和速度会不会好起来。

- 对于 [#261](https://github.com/sgl-project/sglang-omni/pull/261)（MMSU benchmark）：华鹏在 H100 上 `mem_fraction_static=0.85` 勉强启动，但 KV Cache 只有 ~8 GB，并发能力很弱，推理很慢。换 H200 后 KV Cache 有 ~38 GB，并发和速度应该会有质的提升。
- 对于 [#280](https://github.com/sgl-project/sglang-omni/pull/280)（SeedTTS WER）：Hao 在 H100 上 WER 3.0+，很可能是 KV Cache 不足导致部分请求 OOM，产出了错误音频拉高了整体 WER。换 H200 后这些请求应该不会再 OOM。

如果换了 H200 确实有更好的表现，需要跟进以下事项：

1. **修改 SGLang 文档和 error message**：当前文档一刀切地建议「调小」，需要区分两种 OOM 场景，给用户更清晰的调控指南。
2. **H100 上的 Omni 开发方案**：对 30B MoE 模型，H100 80 GB 确实不友好。可能需要支持 thinker TP（tensor parallelism）跨多卡，让单卡放得下模型 + 足够的 KV Cache。CI 上 96 GB H20 恰好绕开了这个问题，但实际用户大概率用 H100。
3. **先在 H200 上把 [#261](https://github.com/sgl-project/sglang-omni/pull/261) 和 [#280](https://github.com/sgl-project/sglang-omni/pull/280) 解决**，H100 的问题根据上述调查结果再定方案。

## 相关源码

| 文件 | 内容 |
|---|---|
| `sglang/srt/model_executor/model_runner_kv_cache_mixin.py:111-144` | KV Cache 分配核心公式 |
| `sglang/srt/model_executor/model_runner_kv_cache_mixin.py:327-331` | Init OOM 错误抛出位置 |
| `sglang/srt/server_args.py:880-1029` | `mem_fraction_static` 自动计算逻辑 |
| `sglang_omni/engines/ar/sglang_backend/server_args_builder.py` | SGLang-Omni 默认值（0.7） |
