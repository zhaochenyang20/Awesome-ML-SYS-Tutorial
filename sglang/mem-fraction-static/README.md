# `mem_fraction_static` 深入解析：两种 OOM 的本质与调参方向

本文档基于 [SGLang](https://github.com/sgl-project/sglang) 的 `model_runner_kv_cache_mixin.py` 和 `server_args.py`，从源码层面解析 `mem_fraction_static` 的内存分配机制，并厘清两种 OOM 场景下**相反的调参方向**。

本文也适用于 [SGLang-Omni](https://github.com/sgl-project/sglang-omni) 的 AR 引擎阶段（thinker、talker 等），其底层复用了相同的 SGLang 内存管理逻辑。

## 问题背景

SGLang 的官方文档建议：

> If you see out-of-memory errors during serving, try to reduce the memory usage of the KV cache pool by setting a smaller value of `--mem-fraction-static`.

但在 SGLang-Omni 部署 Qwen3-Omni（60 GB 权重、H100 80 GB）时，实际遇到的错误是：

```
RuntimeError: Not enough memory. Please try to increase --mem-fraction-static.
```

注意关键词：**increase**，不是 decrease。这和官方文档的建议完全相反。这两种 OOM 发生在不同阶段，调参方向也相反。

## `mem_fraction_static` 到底是什么

`mem_fraction_static` 定义的是**总 GPU 显存中，分配给「模型权重 + KV Cache」的比例**。剩余部分留给 activations 和 CUDA graph buffers。

```
┌─────────────────────────────────────────────────────────────┐
│                       总 GPU 显存                            │
├──────────────────────────────────┬──────────────────────────┤
│     模型权重 + KV Cache           │   Activations +          │
│                                  │   CUDA Graph Buffers     │
│  mem_fraction_static × total     │  (1 - mem_fraction_static)│
│                                  │        × total           │
└──────────────────────────────────┴──────────────────────────┘
```

官方定义（`server_args.py` 第 890-893 行）：

```
GPU memory capacity = model weights + KV cache pool + activations + cuda graph buffers

mem_fraction_static = (model weights + KV cache pool) / GPU memory capacity
```

默认值为 `0.7`（SGLang-Omni 的 `server_args_builder.py` 中设定）。

## KV Cache 分配的核心公式

KV Cache 的大小在启动时由 `profile_max_num_token()` 决定。关键代码位于 `sglang/srt/model_executor/model_runner_kv_cache_mixin.py` 第 111-144 行：

```python
def profile_max_num_token(self, total_gpu_memory: int):
    available_gpu_memory = get_available_gpu_memory(...)  # 模型加载完后的空闲显存

    cell_size = self.get_cell_size_per_token(num_layers)  # 每个 token 的 KV Cache 字节数

    # 核心公式
    rest_memory = available_gpu_memory - total_gpu_memory * (1 - self.mem_fraction_static)

    return int(rest_memory * (1 << 30)) // cell_size
```

拆解这个公式：

| 变量 | 含义 | 示例（H100 80GB，Qwen3-Omni 60GB） |
|---|---|---|
| `total_gpu_memory` | GPU 总容量 | 80 GB |
| `available_gpu_memory` | 模型加载完后的空闲显存 | ~20 GB |
| `1 - mem_fraction_static` | 留给 activations 的比例 | 0.3（当 `mem_fraction_static=0.7`） |
| `total_gpu_memory * (1 - mem_fraction_static)` | 从**总显存**算出的 reserved 区域 | 80 × 0.3 = 24 GB |
| `rest_memory` | 空闲显存 - reserved = 给 KV Cache 的 | 20 - 24 = **-4 GB** |

**注意**：`reserved` 是从**总显存**而非空闲显存计算的。这意味着当模型本身占用了大量显存时，`reserved` 可能超过实际空闲显存，导致 `rest_memory` 为负。

当 `rest_memory <= 0` 时触发错误（第 327-331 行）：

```python
if self.max_total_num_tokens <= 0:
    raise RuntimeError(
        "Not enough memory. Please try to increase --mem-fraction-static."
    )
```

## 两种 OOM 及其调参方向

### 场景一：Init OOM — KV Cache 分配失败（启动阶段）

**现象**：服务启动时直接崩溃，报 `Not enough memory. Please try to increase --mem-fraction-static`。

**根因**：模型相对于 GPU 显存过大。以 Qwen3-Omni 60 GB 在 H100 80 GB 上为例：

```
available = 80 - 60 = 20 GB（空闲）
reserved  = 80 × (1 - 0.7) = 24 GB（给 activations 的预留）
rest      = 20 - 24 = -4 GB → 负数 → OOM
```

`reserved` 按总显存的 30% 计算，这个量（24 GB）超过了实际空闲显存（20 GB）。KV Cache 分不出来。

**解决**：**增大** `mem_fraction_static`，让 `reserved` 缩小。

```
mem_fraction_static = 0.85
reserved = 80 × 0.15 = 12 GB
rest     = 20 - 12 = 8 GB → 可以分配 KV Cache
```

```bash
# SGLang-Omni speech 模式
python examples/run_qwen3_omni_speech_server.py --mem-fraction-static 0.85

# SGLang-Omni text-only 模式
sgl-omni serve --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct --text-only \
  stages.4.executor.args.server_args_overrides.mem_fraction_static=0.85
```

### 场景二：Runtime OOM — 推理期间显存不足

**现象**：服务启动成功，但在处理请求时报 CUDA OOM。

**根因**：KV Cache 占用太大，activations 和 CUDA graph buffers 空间不足。常见于模型较小、`mem_fraction_static` 设得过大的场景。

**解决**：**减小** `mem_fraction_static`，让 `reserved` 增大，给 activations 更多空间。

```bash
sgl-omni serve --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct --text-only \
  stages.4.executor.args.server_args_overrides.mem_fraction_static=0.5
```

### 对比总结

| 场景 | 发生阶段 | 模型大小占比 | 调参方向 | 原理 |
|---|---|---|---|---|
| Init OOM | 启动时 KV Cache 分配 | 大（>70% GPU） | **增大**（如 0.85） | `reserved` 缩小 → `rest_memory` 变正 |
| Runtime OOM | 推理期间 | 小（<50% GPU） | **减小**（如 0.5） | KV Cache 缩小 → activations 空间增大 |

## H100 80 GB 上的数值实例

以 Qwen3-Omni（60 GB 权重）为例，空闲显存约 20 GB：

| `mem_fraction_static` | reserved | rest_memory | KV Cache 可用 | 结果 |
|---|---|---|---|---|
| 0.50 | 40.0 GB | -20.0 GB | 不可分配 | Init OOM |
| 0.70（默认） | 24.0 GB | -4.0 GB | 不可分配 | Init OOM |
| 0.80 | 16.0 GB | 4.0 GB | ~4 GB | 能启动，KV Cache 较小 |
| 0.85 | 12.0 GB | 8.0 GB | ~8 GB | 推荐值 |
| 0.90 | 8.0 GB | 12.0 GB | ~12 GB | KV Cache 大，但 activations 可能不够 |

注意 0.90 虽然 KV Cache 最大，但 activations 只剩 8 GB，高并发或长序列时可能触发 Runtime OOM。实际调参需要在两种 OOM 之间找平衡。

## 为什么 H200 上默认 0.7 没问题

H200 有 141 GB HBM3e。同样的 60 GB 模型：

```
available = 141 - 60 = 81 GB
reserved  = 141 × 0.3 = 42.3 GB
rest      = 81 - 42.3 = 38.7 GB → 充裕
```

H200 的显存足够大，默认 0.7 下 `rest_memory` 仍然很充裕。H100 80 GB 才会遇到这个问题。

## 自动计算逻辑

当用户不手动指定 `mem_fraction_static` 时，SGLang 会自动计算（`server_args.py` 第 988-1029 行）：

```python
# 预估 activations 所需显存
reserved_mem = 512 MB                          # 基础元数据
reserved_mem += max(chunked_prefill_size, 2048) * 1.5 GB  # activations
reserved_mem += cuda_graph_max_bs * 2 GB       # CUDA graph buffers
# 考虑 TP/PP 并行度的调整...

mem_fraction_static = (gpu_mem - reserved_mem) / gpu_mem
```

这个自动值通常比 0.7 更精确，但在 SGLang-Omni 中，`server_args_builder.py` 硬编码默认值为 0.7，没有走自动计算路径。

## 相关源码位置

| 文件 | 内容 |
|---|---|
| `sglang/srt/model_executor/model_runner_kv_cache_mixin.py:111-144` | KV Cache 分配核心公式 |
| `sglang/srt/model_executor/model_runner_kv_cache_mixin.py:327-331` | Init OOM 错误抛出位置 |
| `sglang/srt/server_args.py:880-1029` | `mem_fraction_static` 自动计算逻辑 |
| `sglang/srt/server_args.py:310` | 参数定义 |
| `sglang_omni/engines/ar/sglang_backend/server_args_builder.py` | SGLang-Omni 默认值（0.7） |
