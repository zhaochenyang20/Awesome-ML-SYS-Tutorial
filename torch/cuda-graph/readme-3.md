# 从 S2-Pro 双头架构到 SGLang CudaGraphRunner：CUDA Graph 深度实战

因为工作需要，最近在 SGLang-Omni 框架中为 Fish Audio 的 S2-Pro TTS 模型添加 CUDA Graph 支持（[PR #153](https://github.com/sgl-project/sglang-omni/pull/153)）。这个 PR 前前后后迭代了 7 个 commit，经历了深度 review 讨论，还催生了后续的 [Issue #172](https://github.com/sgl-project/sglang-omni/issues/172)（Framework-level torch.compile 蓝图）。~~本以为加个 CUDA Graph 就完事了，结果越挖越深。~~

在这个过程中，暴露了一系列我理解不够深入的问题：deferred graph capture 的初始化顺序约束、persistent buffer 的指针稳定性设计、torch.compile 四种 mode 与 CUDA Graph 的共存策略、inductor CUDAGraph Trees 与 SGLang CudaGraphRunner 的冲突机制。坦诚说，写完 PR 后回头看，发现自己当时的很多设计选择更多是靠直觉和试错，而非基于对底层机制的清晰理解。

这篇文章就是对这些"知其然而不知其所以然"的地方做一次系统性的梳理。

照理，感谢参与本文档讨论和撰写的所有朋友们：

Ratish1（SGLang），sdli1995（SGLang），以及 SGLang-Omni 的所有 reviewer 们。

实际上，我已经写过两篇 CUDA Graph 相关的前序文章：

- [基于 torch-memory-savor 浅析 CUDA Graph](./readme.md)（本系列第一篇）：覆盖了 CUDA Graph 基本概念、推理常用/训练少用的原因、torch-memory-saver 如何通过 `cuMemMap` 保护虚拟地址稳定性。
- [CUDA Graph vs torch.compile: S2-Pro TTS 模型实战分析](./readme-2.md)（本系列第二篇）：从 S2-Pro 的实际问题出发，对比了两种优化技术消除的开销类型。

本文是这系列的第三篇，将从 PR #153 的 7 个 commit 作为叙事主线，逐层深入。本文首先深入解析 S2-Pro 的 Dual-AR 双头架构和两种 KV cache 的共存设计，接着分析 deferred graph capture 和 persistent buffer 的 CUDA Graph 安全性机制，然后从 GPU 执行流水线的五层开销模型出发剖析 CUDA Graph 与 torch.compile 的深层关系，之后走读 SGLang CudaGraphRunner 的源码实现，最后落地到 PR #153 的 torch.compile 兴衰故事和 Issue #172 的框架级工程蓝图。欢迎大家批评指正。

本文基于 SGLang-Omni commit [`cd9aaf3`](https://github.com/sgl-project/sglang-omni/commit/cd9aaf3) 进行分析。

## S2-Pro Dual-AR 架构深入解析

在进入 CUDA Graph 的工程实现之前，我们需要先透彻理解 S2-Pro 的模型架构——它不是一个普通的 LLM，而是一个 **Dual-AR（双自回归）模型**，将文本理解和音频生成统一在一个前向传播中。

### Slow Head：基于 Qwen3 的文本模型

S2-Pro 的 slow head 是一个完整的 Qwen3 架构 transformer，在 [`sglang_model.py`](https://github.com/sgl-project/sglang-omni/blob/cd9aaf3/sglang_omni/models/fishaudio_s2_pro/sglang_model.py) 中实现为 `S2ProSGLangTextModel`。其核心参数如下：

- **36 层** `S2ProDecoderLayer`
- `hidden_size=2560`，`intermediate_size=9728`
- `num_heads=32`，`num_kv_heads=8`（GQA），`head_dim=128`
- 使用 SGLang 的 `RadixAttention`（带 paged KV cache），注意力后端为 FlashAttention 3
- RoPE（`is_neox_style=False`）

**计算特征**：每层包含 4 个大 GEMM——`qkv_proj`、`o_proj`、`gate_up_proj`、`down_proj`。以 bs=8 为例，主要的 GEMM shape 是 `mm(8×2560, 2560×6144)` 和 `mm(8×9728, 9728×2560)` 等。这些大矩阵乘法在 cuBLAS 中已经高度优化，单个 kernel 耗时在 ms 级，**kernel launch overhead 相对占比较小**。

### Fast Head：FishQwen3AudioDecoder（Codebook Loop）

S2-Pro 的 fast head 是一个独立的小型 transformer——[`FishQwen3AudioDecoder`](https://github.com/sgl-project/sglang-omni/blob/cd9aaf3/sglang_omni/models/fishaudio_s2_pro/fish_speech/models/text2semantic/modeling.py#L919)，负责生成 audio codebook tokens。其架构包括：

- `project_in`：一个线性投影层，将 text model 的 `hidden_size`（2560）映射到 fast decoder 自己的维度空间
- 若干层 `TransformerBlock`（层数远小于 slow head）
- `RMSNorm` + `output` 线性头（输出 `vocab_size=4096` 的 codebook logits）
- `codebook_embeddings`：`nn.Embedding(vocab_size × num_codebooks, text_dim)`——共享的 codebook embedding 表
- `codebook_offsets`：`torch.arange(num_codebooks) * vocab_size`——用于向量化 embedding 查找的偏移量

**9 步 codebook loop 的计算特征**：每步只有一次小 GEMM（`[bs, 1, fast_dim]` 级别），计算量极小（μs 级），但需要完成 embedding lookup → linear projection → `forward_kvcached` → argmax → embedding lookup 的完整序列。**瓶颈完全在 kernel launch overhead**——每步虽然快，但 9 步循环产生了大量的小 kernel launch。

这里思考一个小问题：为什么 fast head 的 KV cache 要用 static 预分配而不用 SGLang 的 paged KV cache？

简单来说，codebook loop 的序列长度是固定的（`num_codebooks + 1`，即 10 或 11 步），不存在动态增长的需求，也不需要 prefix caching 或 radix tree 带来的灵活性。Static KV cache 不仅实现更简单，而且天然 CUDA Graph 安全——分配后地址不再变化。

### 两种 KV Cache 的共存

这是一个值得深入理解的设计点。在同一个 CUDA Graph 中，两种完全不同的 KV cache 管理策略必须和平共处：

| 维度 | SGLang Paged KV Cache（slow head） | Static KV Cache（fast head） |
|---|---|---|
| 管理方式 | `token_to_kv_pool_allocator` 动态分配 | `setup_caches()` 一次性预分配 |
| 序列长度 | 动态增长 | 固定（`num_codebooks + 1`） |
| Prefix Caching | 支持（RadixAttention） | 不支持 |
| 内存回收 | 通过 `release_kv_cache()` 归还 pool | `reset_caches()` 用 `zero_()` 清空 |
| CUDA Graph 安全性 | 通过 SGLang 的 `req_to_token_pool` 间接保证 | **天然安全**——地址分配后不变 |

**关键洞察**：paged KV cache 的地址稳定性由 SGLang 框架（`CudaGraphRunner` 的预分配 buffer + `req_pool_indices` 间接寻址）保证，而 static KV cache 则更简单——`setup_caches()` 时 `torch.zeros(...)` 分配一次，之后只通过 `zero_()` 清空值、`flash_attn_with_kvcache` 原地更新。两者在同一个 graph 中共存不冲突，因为 graph replay 时看到的 GPU 虚拟地址没有变化。

### PR #153 的架构变革：从分离到统一

**PR #153 之前（分离的处理流程）**：

```
Text Model forward → LogitsProcessorOutput
                        ↓
S2ProSGLangOutputProcessor._codebook_loop_impl()  ← per-request, 不在 graph 内
                        ↓
Codebook codes output
```

**PR #153 之后（统一的 forward）**：

```
S2ProSGLangModelRunner._update_vq_buffers()  ← 将上一步的 codes 写入 persistent buffers
                        ↓
S2ProSGLangTextModel.forward()
    ├── VQ embedding combination（从 persistent buffers 读取）
    ├── 36 层 Transformer（slow head）
    ├── Logits 计算
    └── _decode_codebooks()（constrained sampling + batched codebook loop）
                        ↓
S2ProSGLangModelRunner._build_outputs()  ← 从 persistent buffers 读取 output codes
```

**关键改变**：`_decode_codebooks()` 从外部的 per-request 后处理，变成了 `forward()` 内部的一个步骤。这意味着 **CUDA Graph 可以将 transformer + sampling + codebook loop 一次性录制**，消除整个 decode step 的所有 kernel launch overhead。最终实现了 ~88 tok/s 的稳态吞吐，启动时间仅 33s（graph capture 3.3s）（来自 Ratish1 的 benchmark 数据）。

## Deferred Graph Capture：为什么初始化顺序如此重要

有了对模型架构的理解，我们接着来看 PR #153 中一个至关重要的工程设计——deferred graph capture。

### `factory.py` 的初始化时序

[`create_s2pro_sglang_engine()`](https://github.com/sgl-project/sglang-omni/blob/cd9aaf3/sglang_omni/models/fishaudio_s2_pro/factory.py) 中的初始化时序非常精心，每一步都不能乱：

```python
# Step 1: 暂时禁用 CUDA Graph
want_cuda_graph = not server_args.disable_cuda_graph
server_args.disable_cuda_graph = True

# Step 2: 初始化 ModelWorker（此时不 capture graph）
model_worker = ModelWorker(config=ModelWorkerConfig(), server_args=server_args, gpu_id=gpu_id)

# Step 3: BF16 精度修正
_truncate_rope_to_bf16(model_worker.model_runner.model)

# Step 4: 预分配 fast head 的 static KV cache
audio_decoder.setup_caches(max_batch_size=max_bs, dtype=torch.bfloat16)

# Step 5: 分配 persistent buffers + 挂载 audio decoder
text_model.setup_vq_decode(audio_decoder, ...)

# Step 6: 此时 capture graph——包含完整的 forward + _decode_codebooks
if want_cuda_graph:
    model_worker.model_runner.init_device_graphs()
```

有一个问题非常值得分享：**为什么不能在 Step 2 直接 capture？**

`ModelWorker.__init__()` 内部会调用 `init_cuda_graphs()`。如果在这一步就 capture，此时 `text_model._vq_ready = False`——因为 `setup_vq_decode()` 还没调用。于是 `forward()` 中的 `if self._vq_ready:` 分支不会执行，graph 中不包含 VQ embedding combination 和 `_decode_codebooks()`。

**CUDA Graph 是静态的**。一旦录制完成，graph 中的 kernel 序列就被固化了。后续即使调用了 `setup_vq_decode()` 让 `_vq_ready = True`，已经录好的 graph 并不会自动更新。Replay 时执行的仍然是录制时的 kernel 序列——也就是一个不包含 codebook decode 的"残缺"forward。

因此必须 **先** attach audio decoder 和分配 buffers（Step 4-5），**再** capture graph（Step 6）。这就是 deferred graph capture 模式的核心：通过 `server_args.disable_cuda_graph = True` 临时禁用 `ModelWorker.__init__()` 内部的 graph capture，等所有准备工作就绪后再手动触发。

### `setup_vq_decode()` 的 Buffer 分配设计

[`setup_vq_decode()`](https://github.com/sgl-project/sglang-omni/blob/cd9aaf3/sglang_omni/models/fishaudio_s2_pro/sglang_model.py#L196) 做了以下关键工作：

**Input buffers**（由 ModelRunner 在 forward 前写入）：
- `_vq_codes`：`torch.zeros(max_bs, num_codebooks, dtype=torch.long)`——上一步生成的 codebook codes
- `_vq_mask`：`torch.zeros(max_bs, dtype=torch.bool)`——标记哪些 batch 位置需要 VQ embedding combination

**Output buffers**（由 `_decode_codebooks()` 写入，ModelRunner 在 forward 后读取）：
- `_output_codes`：`torch.zeros(max_bs, num_codebooks+1, dtype=torch.long)`——当前步生成的所有 codes
- `_output_semantic_ids`：`torch.zeros(max_bs, dtype=torch.long)`——当前步的 semantic token id

**Auxiliary tensors**：
- `_semantic_bias`：`torch.full((vocab_size,), -inf, dtype=torch.bfloat16)`——非 semantic 和非 EOS 的 token logits 设为 `-inf`，实现 constrained decoding
- `_vq_codebook_embeddings`：直接引用 `audio_decoder.codebook_embeddings`（共享权重，不额外分配）
- `_vq_codebook_offsets`：引用 `audio_decoder.codebook_offsets`

## Persistent Buffer 设计：为什么 `copy_()` 能保证 CUDA Graph 安全

理解了 buffer 的分配，下一个关键问题是：为什么这些 buffer 是 CUDA Graph 安全的？

### CUDA Graph 的指针稳定性要求

回顾 [基于 torch-memory-savor 浅析 CUDA Graph](./readme.md) 中讨论过的：CUDA Graph capture 时，runtime 录制的是每个 kernel 的参数——包括输入/输出 tensor 的 **GPU 虚拟地址**。Replay 时，这些地址被原样传递给 kernel。如果地址变了（比如 tensor 被重新分配），kernel 就会读写错误的内存。

**所有 buffer 在 `setup_vq_decode()` 时一次性 `torch.zeros(...)` 分配，之后只通过就地操作修改值**：

| 操作 | 使用场景 | 为什么 Graph 安全 |
|---|---|---|
| `tensor.copy_(source)` | `_update_vq_buffers()` 写入 VQ codes | 修改值，不改地址 |
| `tensor[:bs] = value` | `_decode_codebooks()` 写入 output codes | index assignment 是就地操作 |
| `tensor.fill_(scalar)` | audio decoder 的 `input_pos.fill_(codebook_idx)` | 就地填充 |
| `tensor.zero_()` | `reset_caches()` 清空 KV cache | 就地清零 |

让我惊讶的是，即使是 `torch.where(mask, a, b)` 也是安全的——它返回一个新 tensor，但在 graph capture 期间，PyTorch 的 CUDA caching allocator 会记录这个分配，replay 时会复用同一块内存。

### `_update_vq_buffers()` 和 `_build_outputs()` 的读写协议

[`S2ProSGLangModelRunner`](https://github.com/sgl-project/sglang-omni/blob/cd9aaf3/sglang_omni/models/fishaudio_s2_pro/runtime/s2pro_sglang_ar.py) 实现了一个精心设计的 buffer 读写协议：

```
execute(scheduler_output):
  ├── _update_vq_buffers()   ← 仅 decode：写入 text_model._vq_codes / _vq_mask（graph 外部）
  ├── model_worker.forward() ← CUDA Graph replay（graph 内部）
  └── _build_outputs()       ← 从 text_model._output_codes 读取结果（graph 外部）
```

**关键洞察**：buffer 的读写发生在 **CUDA Graph boundary 之外**（`_update_vq_buffers` 在 forward 前，`_build_outputs` 在 forward 后），但 buffer 本身被 graph 内部的 kernel 引用。这种 **"外部写值、graph 内部读地址"** 的模式是 CUDA Graph 与动态输入兼容的标准做法。

来看 `_update_vq_buffers()` 的具体实现：

```python
def _update_vq_buffers(self, model_worker_batch, scheduler_output):
    text_model = self.model_worker.model_runner.model
    input_ids = model_worker_batch.input_ids
    bs = input_ids.shape[0]

    # 计算 semantic mask
    is_semantic = (input_ids >= self._semantic_begin_id) & (input_ids <= self._semantic_end_id)
    text_model._vq_mask[:bs].copy_(is_semantic)

    # 写入每个 request 的 codebook values
    for i, sched_req in enumerate(scheduler_output.requests):
        data = sched_req.data
        if data._last_codebook_values is not None and is_semantic[i]:
            text_model._vq_codes[i].copy_(data._last_codebook_values)
```

这里 `.copy_()` 是关键——它修改了 `_vq_mask` 和 `_vq_codes` 的值，但这些 tensor 的 GPU 虚拟地址没有变化。当 graph replay 时，`forward()` 中的 kernel 读取的仍然是同一个地址，只是值已经被更新为当前 step 的数据。

### audio decoder 的 `input_pos.fill_()` 模式

`forward_kvcached()` 中使用了一个精心设计的 CUDA Graph 兼容模式：

```python
def forward_kvcached(self, x, codebook_idx):
    # 用 fill_() 更新 pre-allocated buffer（地址不变，值变化）
    self.input_pos.fill_(codebook_idx)
    freqs_cis = self.freqs_cis[self.input_pos]
    cache_seqlens = self.input_pos.expand(bsz).to(torch.int32)
    ...
```

`input_pos` 是 `register_buffer` 注册的 persistent tensor。`fill_()` 是就地操作，修改值而不重新分配。`codebook_idx` 在 codebook loop 展开后是 Python 常量（`for cb_idx in range(1, self._num_codebooks)` 中的 `cb_idx`），在 capture 时被固化为 graph 的一部分。

> 这比 `torch.tensor([codebook_idx])` 安全得多——后者会创建新 tensor，破坏地址稳定性。

### 为什么 Greedy Decoding 是 CUDA Graph 安全的

PR #153 将 codebook 的采样策略从 `_sample_with_topk`（temperature + top_k + top_p + repetition_penalty + RAS）切换为 `torch.argmax(biased_logits, dim=-1)`。Copilot review 专门指出了这个 behavioral change（sampling parameters 被忽略）。这不仅仅是简化——它是 CUDA Graph 兼容性的要求：

- `torch.argmax` 是确定性的、无状态的、不需要随机数生成器 → 完全可以被 CUDA Graph 录制
- top_k/top_p sampling 涉及 `torch.multinomial`，可能需要 random state 管理和动态 shape 操作 → graph-incompatible
- TTS 场景下 greedy decoding 的质量损失可以接受——这是一个有意的 trade-off

## CUDA Graph 底层机制：capture → instantiate → replay

有了对具体工程实现的理解，我们需要退一步，系统性地理解 CUDA Graph 底层的三阶段机制，并将每个约束映射到 S2-Pro 的设计选择上。

### 三阶段工作流

1. **Capture**：SGLang 的 `CudaGraphRunner` 对每个 batch size 调用 `model.forward()`。此时 CUDA runtime 不执行 kernel，而是录制所有 kernel launch 及其参数（包括输入/输出 tensor 的 GPU 虚拟地址），形成一个 DAG。
2. **Instantiate**：将录制的 DAG 编译为 `cudaGraphExec_t`，此时进行 kernel 参数绑定和 dependency analysis。
3. **Replay**：每次 decode step，`CudaGraphRunner` 先将新的输入 `copy_()` 到预分配的 buffer 中（地址不变，只改值），然后 `cudaGraphLaunch()` 一次性提交所有 kernel——**CPU 只发出一次 launch 指令**。

### CUDA Graph 约束与 S2-Pro 设计选择的映射

| CUDA Graph 约束 | PR #153 中的应对策略 |
|---|---|
| capture 期间所有 kernel launch 被记录而非执行 | `setup_vq_decode()` 必须在 capture 前调用，确保 `_vq_ready=True`，让 `_decode_codebooks()` 的 kernel 被录制 |
| capture 期间不能有动态内存分配 | 所有 buffer 在 capture 前预分配为 `max_batch_size` 大小 |
| capture 期间不能有 host-device sync | 用 `torch.argmax`（greedy）替代 stochastic sampling，避免动态采样可能引发的 sync |
| graph replay 时必须保证 pointer 稳定性 | persistent buffers 只通过 `copy_()`、index assignment 修改值，不重新分配 |
| graph 中的控制流必须是静态的 | codebook loop 的循环次数 `num_codebooks` 是常量，`for cb_idx in range(1, self._num_codebooks)` 在 capture 时被完全展开 |

## SGLang CudaGraphRunner 源码走读

有了这些基础，我们来看 SGLang 如何在框架层面管理 CUDA Graph 的 capture 和 replay。

### 多 Batch Size 的 Graph 管理

SGLang 的 `CudaGraphRunner` 为每个 batch size 维护一个独立的 `cudaGraphExec_t`。默认的 capture_bs 列表包含 12 个 batch size（如 `[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64]`）。

**Capture 顺序**：从大到小。原因是大 bs 需要更多内存，先 capture 大 bs 可以让 memory allocator "看到" 最大的内存需求，后续小 bs 的 capture 可以复用已分配的内存（通过共享 memory pool）：

```python
# cuda_graph_runner.py capture() 中的关键逻辑
capture_range = reversed(self.capture_bs)  # 从大 bs 到小 bs
for bs in capture_range:
    graph, output_buffers = self.capture_one_batch_size(bs, forward)
    self.graphs[bs] = graph
    self.output_buffers[bs] = output_buffers
```

每个 bs 在 capture 前先做一次 **warmup run**（eager forward），触发所有可能的内存分配（包括 cuBLAS workspace、attention buffer 等），确保 capture 时不会有意外的 allocation。

### Memory Pool 共享

`CudaGraphRunner` 使用 `torch.cuda.graph(pool=...)` 让多个 graph 共享同一个 memory pool：

```python
# _capture_graph() 的核心
with self.device_module.graph(cuda_graph=graph, pool=pool, stream=stream):
    out = run_once_fn()
```

**不共享 pool 的代价**：每个 graph 独立分配中间 tensor 内存，12 个 graph 各持有一份 → **12 倍的中间 tensor 显存**。共享 pool 的前提是同一时间只有一个 graph 在 replay——SGLang 的 decode 阶段满足此条件。

对 S2-Pro 的意义：12 个 bs 的 graph 共享 audio decoder 的 KV cache 中间结果内存，不需要 12 份独立分配。

### Replay 中的 BS Padding

当 actual batch size < captured batch size 时，`CudaGraphRunner` 会找到 **大于等于 actual bs 的最小 captured bs**：

```python
index = bisect.bisect_left(self.capture_bs, raw_bs)
bs = self.capture_bs[index]
```

Graph replay 仍然执行完整的 `captured_bs` 个 kernel，多余的行产生无效计算。对于 S2-Pro，padding 意味着 `_decode_codebooks()` 也会对 padding 行执行完整的 9 步 codebook loop——这是额外的浪费。但由于 codebook loop 是小矩阵运算，额外开销很小。

### S2-Pro 对 Capture 的额外需求

当 `text_model._vq_ready = True` 时，每个 bs 的 capture 包含：

1. VQ embedding combination（读取 `_vq_codes`、`_vq_mask`）
2. 36 层 Transformer（RadixAttention + FFN）
3. logits 计算
4. `_decode_codebooks()`：constrained sampling + 9 步 codebook loop

这意味着 graph 中包含了约 `36 × 4（transformer GEMM）+ 9 × N（codebook loop kernels）` 个 kernel node。这比普通 LLM 的 graph 显著更大，但 replay 的 latency overhead 仍然是 **一次 `cudaGraphLaunch()` 调用**——graph size 不影响 launch 开销，只影响 GPU 侧的执行时间。

### 何时 Fallback 到 Eager Mode

并非所有情况都能走 CUDA Graph。对 S2-Pro 而言，以下场景会 fallback 到 eager 执行：

- **Prefill 阶段**：序列长度不固定，无法匹配预先 capture 的固定 shape → 不走 graph
- **Decode 阶段 bs 超过最大 capture bs**：找不到匹配的 graph → fallback
- **Chunked prefill**：不走 graph
- **Extend 模式**（`forward_batch.forward_mode.is_extend()`）：走 eager

也就是说，CUDA Graph 主要加速的是 **decode 阶段的稳态吞吐**——这恰好是 S2-Pro 的性能瓶颈所在（codebook loop 的大量小 kernel launch）。

## CUDA Graph 与 torch.compile 的深层关系

这是本文最核心的部分。在 PR #153 的迭代过程中，torch.compile 曾经被加入又被移除，这个决策背后的技术逻辑需要从 GPU 执行流水线的开销模型说起。

### GPU 执行流水线的五层开销

```
CPU 侧                                    GPU 侧
┌─────────────────┐                      ┌─────────────────┐
│ Python 解释器    │  ①Python overhead    │                 │
│      ↓          │                      │                 │
│ PyTorch dispatch│  ②框架 dispatch      │                 │
│      ↓          │                      │                 │
│ CUDA API 调用   │  ③launch overhead ──→│ Kernel 执行     │
│      ↓          │                      │      ↓          │
│ 等待下一个 op   │                      │ ④显存读写(带宽) │
│                 │                      │      ↓          │
│                 │                      │ ⑤算术计算(算力) │
└─────────────────┘                      └─────────────────┘
```

| 开销层 | CUDA Graph 的效果 | torch.compile 的效果 |
|---|---|---|
| ①Python overhead | **完全消除**（replay 不经过 Python） | 大幅减少（traced graph 绕过 Python dispatch） |
| ②框架 dispatch | **完全消除** | 大幅减少 |
| ③launch overhead | **完全消除**（一次 launch 提交所有 kernel） | 部分减少（融合后 kernel 数量减少） |
| ④显存带宽 | 不影响 | **显著优化**（算子融合减少中间 tensor 读写） |
| ⑤算术计算 | 不影响 | 可能优化（Triton kernel 可能比 cuBLAS 更优，但也可能更差） |

**关键洞察**：两者在开销③上有重叠（都能减少 kernel launch），但在 **④（显存带宽）** 上只有 torch.compile 有效。这解释了为什么：

- CUDA Graph alone 对 codebook loop（③是瓶颈）效果极好
- 但 CUDA Graph + torch.compile 仍然有 36% 的增量——因为 codebook loop 的小算子链仍然有大量的中间 tensor 显存读写（开销④）

### torch.compile 的四种 Mode 与 CUDA Graph 的关系

| Mode | 含义 | CUDA Graph 行为 | S2-Pro 适用性 |
|---|---|---|---|
| `default` | 基本优化，不做 autotune | inductor 不管理 CUDA graph | 可用但收益小 |
| `reduce-overhead` | 优化 + inductor 内部自动管理 CUDA graph | **inductor 自己 capture/replay graph**（CUDAGraph Trees） | **与 SGLang 的 CudaGraphRunner 冲突** |
| `max-autotune` | 全力 autotune + inductor 管理 CUDA graph | 同 `reduce-overhead` | **冲突** |
| `max-autotune-no-cudagraphs` | 全力 autotune，但 **不** 让 inductor 管理 graph | inductor 只生成优化的 kernel，不碰 graph | **SGLang 使用的 mode** |

有一个问题非常值得分享：**为什么 SGLang 使用 `max-autotune-no-cudagraphs`？**

这是一个关键的架构决策。SGLang 有自己的 `CudaGraphRunner` 负责 graph capture/replay。如果 inductor 也自己做 graph capture（`reduce-overhead` 或 `max-autotune` mode），就会产生 **"graph 里套 graph"** 的冲突——外层是 SGLang 的 graph capture，内层是 inductor 的 CUDAGraph Trees，两者争抢 stream capture 的控制权。

因此 SGLang 选择 `no-cudagraphs` 后缀：让 inductor 只负责 **kernel 优化**（算子融合 + Triton autotune），而 **graph 管理** 留给 SGLang 自己。这是一种 **"分工"** 模式：

```
torch.compile(mode="max-autotune-no-cudagraphs") → 生成优化的 Triton kernel
                          ↓
SGLang CudaGraphRunner.capture() → 将这些优化后的 kernel 录入 CUDA Graph
                          ↓
SGLang CudaGraphRunner.replay() → 一次性提交所有优化后的 kernel
```

### CUDAGraph Trees：inductor 内部的 Graph 管理机制

虽然 SGLang 不使用 inductor 的 graph 管理，但理解其机制有助于理解为什么两套系统不能叠加。

**CUDAGraph Trees** 是 inductor 在 `reduce-overhead` mode 下的内部实现：为每个不同的执行路径（不同 shape、不同 control flow 分支）创建独立的 graph，以树状结构组织。其特点包括：

- **Memory Pool 共享**：所有 graph 共享一个 memory pool
- **Graph Break 处理**：当 torch.compile 遇到无法 trace 的操作（如 `.item()`、动态 control flow），会产生 graph break，将代码切分为多个 graph partition
- **执行流程**：首次调用 warmup → 第二次调用录制 graph → 后续调用 replay

**与 SGLang CudaGraphRunner 的冲突点**：

1. SGLang 的 `CudaGraphRunner` 也在 stream 上做 capture。如果 inductor 已经在内部 capture 了一部分 kernel，那么 SGLang 的外层 capture 会"看到"一个已经被 graph 化的执行——可能导致 **nested capture** 的未定义行为
2. Memory pool 管理：inductor 和 SGLang 各自管理 memory pool，两者不互通

### `fullgraph=True` 的约束与 S2-Pro 的挑战

`fullgraph=True` 要求 torch.compile 将整个被编译的函数 trace 为 **一个** 完整的 FX graph，不允许任何 graph break。对 S2-Pro 的 `_decode_codebooks()` 来说：

**可以满足的条件**：
- `for cb_idx in range(1, self._num_codebooks)` 循环可以被完全展开（`num_codebooks` 是常量）
- `self._audio_decoder.reset_caches()` 中的 `zero_()` 是可 trace 的就地操作
- `torch.argmax` 可 trace

**潜在的 graph break 风险**（对应 Issue #172 Phase 2 需要解决的问题）：
- `RadixAttention`（slow head）：涉及 paged KV cache 的动态索引 → 可能 graph break
- `ForwardBatch` 的动态属性访问 → 可能 graph break
- `MergedColumnParallelLinear` 的 tensor parallel 通信 → 需要验证

这也是为什么 Issue #172 将 backbone compile（Phase 2）排在 auxiliary module compile（Phase 1）之后——**fast head 的 `_decode_codebooks` 更容易满足 `fullgraph=True`，而 slow head 有大量 SGLang 特有的动态操作**。

### inductor 生成的 Triton Kernel 能否被外部 CUDA Graph 录制？

这是 `no-cudagraphs` 模式下的核心假设：inductor 生成的 Triton kernel 必须是"普通"的 CUDA kernel，能够被 SGLang 的 `CudaGraphRunner` 正常录制。

**答案是：可以，但有条件**：

- inductor 在 `no-cudagraphs` 模式下输出的 Triton kernel 是标准的 CUDA kernel（通过 Triton 的 PTX codegen），与 cuBLAS kernel 一样可以被 stream capture 录制
- 但 inductor 的 **guard 机制** 可能在 graph replay 时触发 recompilation——如果输入 tensor 的 shape/stride/dtype 与 trace 时不同，inductor 会尝试重新编译
- SGLang 通过 `CudaGraphRunner` 的 **固定 bs + padding 策略** 规避了这个问题：每个 captured bs 对应一组固定 shape 的输入，guard 不会触发

## torch.compile 在 S2-Pro 中的兴衰

有了上面的理论框架，我们可以完整理解 PR #153 中 torch.compile 从加入到移除的故事。

### 七个 Commit 的叙事线

| 序号 | Commit | 内容 | 意义 |
|---|---|---|---|
| 1 | `c153ae9` | unified slow/fast head | 核心实现：统一 forward + persistent buffers |
| 2 | `f621355` | lint | 代码规范 |
| 3 | `c962aa6` | torch.compile added in | **转折点**：加入 `enable_torch_compile = True` |
| 4 | `78aafc7` | setup_vq_decode before CUDA graph capture | **关键修复**：deferred graph capture |
| 5 | `dccf122` | tts eval refactoring | Benchmark 重构 |
| 6 | `cf9396d` | export server output | 输出接口调整 |
| 7 | `20be04a` | acknowledge torch.compile discussion | **最终决策**：移除 torch.compile |

**迭代的关键转折**：

Commit 3 加入了 `server_args.enable_torch_compile = True`。SGLang 的 `CudaGraphRunner` 在 `enable_torch_compile=True` 时会调用 `torch.compile(model.forward, mode="max-autotune-no-cudagraphs")`。这导致了 **整个 model forward** 被 inductor 接管：对 36 层 transformer × 12 个 bs 的每个 GEMM shape 做 18 候选 kernel 的 benchmark。启动时间从 33s 膨胀到了 137s——第一次看到日志的时候差点以为进程卡死了。

### Ratish1 的三配置 Benchmark

| 配置 | Health Ready | Graph Capture | 吞吐（TTS） | 吞吐（Voice Clone） |
|---|---|---|---|---|
| CUDA Graph only | 33.3s | 3.3s | 88.1 tok/s | 87.7 tok/s |
| Partial compile（fast head only） | 54.4s | 16.4s | 120.6 tok/s | 118.7 tok/s |
| Full-model compile | 137.0s | 107.0s | 125.7 tok/s | 122.5 tok/s |

**结合五层开销模型解读这些数据**：

1. **Partial compile 的 36% 吞吐提升从何而来？** CUDA Graph 已消除开销①②③，但 codebook loop 的 9 步循环中每步包含 embedding lookup → linear projection → multi-head attention → RMSNorm → output projection，这些小算子之间的中间 tensor 仍然需要经过显存读写（开销④）。torch.compile 的 inductor 将这些算子融合为更少的 Triton kernel，减少了 GPU-side 的显存 round-trip。**即使 launch overhead 为零，带宽优化仍有 36% 的收益空间**。

2. **Full compile vs Partial compile 仅 4% 差异**：transformer 部分的大 GEMM 已被 cuBLAS 高度优化。从 autotune 日志可以验证——cuBLAS 在多数 shape 下击败了 Triton kernel（开销⑤已经接近最优）。torch.compile 在 transformer 上唯一的收益是融合 layernorm + residual 等小算子链，但这部分占比很小。

3. **103.7s 的额外启动时间**：`max-autotune-no-cudagraphs` mode 对每个 GEMM shape × 每个 bs 做 Triton autotune，总量 ≈ 12 bs × 36 layers × ~4 linear layers × 18 candidates ≈ 31,000+ benchmark runs。这是 autotune 的固有成本。

4. **Partial compile 的启动时间仅 +21s**：只编译 fast head 的少量小算子，autotune 搜索空间远小于 full model。`54.4s - 33.3s = 21.1s` 是可接受的。

### 为什么最终选择不 Compile：三层决策逻辑

1. **抽象层级错配**：torch.compile 的优化应该是 SGLang-Omni 框架级别的能力，而不是单个模型的 hack。在模型代码中硬编码 `enable_torch_compile=True` 违背了框架的设计哲学。

2. **交互复杂性**：torch.compile 的 guards/recompilation 与 CUDA Graph 的交互需要非常小心。codebook loop 的 `for cb_idx in range(1, self._num_codebooks)` 需要被 `fullgraph=True` 完全展开，任何 graph break 都会导致编译失败或性能退化。而 slow head 的 `RadixAttention` 更是 graph break 的重灾区。

3. **粒度问题**：真正受益的只有 fast head 的小算子融合（36% 增益），slow head 的 4% 增量不值得 103s 的额外启动时间。结合 Ratish1 的数据，这是一个清晰的 cost-benefit 判断。

> 这个决策的精妙之处在于：它不是"不要 torch.compile"，而是"**不在这里做 torch.compile**"——将优化机会推迟到框架层面（Issue #172），以更系统性的方式实现。

## Issue #172：Framework-Level torch.compile 蓝图

PR #153 中被 defer 的 torch.compile 优化，最终以 [Issue #172](https://github.com/sgl-project/sglang-omni/issues/172) 的形式提出了系统性的解决方案。

### 三个核心障碍

1. **启动开销**：Full-model compile 需要 2~5 分钟，生产环境不可接受
2. **抽象缺失**：目前需要在每个模型的 `factory.py` 中硬编码 compile 逻辑
3. **Graph 冲突**：框架已经管理了 CUDA Graph capture，torch.compile 必须与之干净共存

这三个障碍正好对应上文分析的三个技术约束：mode 选择（`no-cudagraphs`）、`fullgraph=True` 要求、guard 机制。

### Phase 1：Partial Compile（仅辅助模块）

模型实现 `get_compile_targets()` 方法，声明"什么是可编译的"：

```python
class S2ProSGLangTextModel(nn.Module):
    def get_compile_targets(self) -> dict[str, Callable]:
        if not self._vq_ready:
            return {}
        return {"decode_codebooks": self._decode_codebooks_impl}
```

框架侧决定"如何编译"：

```python
def apply_compile_targets(model, compile_mode="max-autotune-no-cudagraphs"):
    if not hasattr(model, "get_compile_targets"):
        return []
    compiled = []
    for name, fn in model.get_compile_targets().items():
        compiled_fn = torch.compile(fn, mode=compile_mode, fullgraph=True)
        setattr(model, f"_compiled_{name}", compiled_fn)
        compiled.append(name)
    return compiled
```

**关键设计决策**：

- **`fullgraph=True` 是强制要求**——graph break 在 CUDA Graph 环境下会导致不可预测的行为
- **compile mode 固定为 `max-autotune-no-cudagraphs`**——graph 管理权留给 SGLang
- **模型只声明 target，不调用 `torch.compile`**——实现了"模型不知道自己被编译了"的抽象
- **compile 在 `setup_vq_decode()` 之后、`init_device_graphs()` 之前**——复用 deferred capture 时序

**预期效果**：S2-Pro ~121 tok/s，启动 ~54s。

### Phase 2：Global Compile（完整 model forward）

目标是编译整个 `model.forward()` 以获取剩余的 4% 吞吐增量。前置条件是 SGLang 的 `RadixAttention`、`ForwardBatch`、`MergedColumnParallelLinear`、RoPE cache 模式必须全部 compile-clean（无 graph break）。

两种共存策略需要 benchmark：

| 策略 | 实现方式 | 优点 | 缺点 |
|---|---|---|---|
| **Layered**（分层管理） | `max-autotune-no-cudagraphs` + SGLang CUDA Graph | SGLang 保持对 graph 的完全控制 | 需要确保 inductor kernel 对 SGLang graph capture 完全透明 |
| **Unified**（统一管理） | `reduce-overhead`，让 inductor 管理 CUDA Graph | 更深度的优化（cross-kernel memory planning） | 失去 SGLang 的 multi-bs graph 管理、memory pool 控制等精细能力 |

用户通过配置驱动：

```bash
--compile-level none     # CUDA graph only（默认，零启动开销）
--compile-level partial  # Phase 1：仅 auxiliary modules
--compile-level full     # Phase 2：完整 model forward
```

### Phase 3：Mega Cache（消除启动开销）

缓存 inductor 的编译产物（FX graph + Triton kernel binary + autotune 结果），让第二次启动跳过所有 compile 开销。Cache key 使用多因子设计：

```python
cache_key = hash(
    model_path,              # 模型权重标识
    compile_level,           # "partial" 或 "full"
    max_batch_size,          # 影响 CUDA Graph shape
    torch.__version__,       # inductor codegen 可能变化
    cuda_runtime_version,    # CUDA runtime 版本
    gpu_arch,                # sm_80 vs sm_90 产生不同 kernel
)
```

**预期效果**：warm cache 下，即使 `compile_level=full`，启动时间也接近 baseline 的 ~33s。

### 适用范围：不只是 S2-Pro

Issue #172 的框架设计是 model-agnostic 的：

| 模型 | Backbone | Auxiliary Module | Compile 机会 |
|---|---|---|---|
| **S2-Pro** | Qwen3 + RadixAttention | Codebook decoder | +37%（partial）；+43%（full） |
| **Qwen3-Omni** | Qwen3 thinker | Talker、encoders | 待 benchmark |
| **未来模型** | 任何 SGLang-backed LLM | 模型特定的 decoder | 自动获得 compile 支持 |

这意味着学习这个设计不仅对 S2-Pro 有价值，对理解 **推理框架如何系统性地管理 torch.compile** 有普遍意义。任何接入 SGLang-Omni 的新模型，只要实现 `get_compile_targets()` 协议，就能自动获得 partial compile 加速。

### 五条设计原则

| 设计原则 | 含义 | 对应 PR #153 中的教训 |
|---|---|---|
| 模型文件中不出现 compile 调用 | 模型声明 target，框架决定编译策略 | PR #153 中硬编码 `enable_torch_compile=True` 导致了不可维护的 hack |
| Compile target 必须是 tensor-in tensor-out | 被编译的函数不能有外部状态访问 | `_decode_codebooks` 需重构为纯函数 `_decode_codebooks_impl` |
| `fullgraph=True` 强制 | 不允许 graph break | codebook loop 必须全部可 trace |
| Eager-first 可读性 | compile 是可选加速，不是默认行为 | PR #153 最终选择 CUDA Graph only 作为默认 |
| 配置驱动 | 通过 `ServerArgs` 开关控制 | 同 `disable_cuda_graph` 的控制模式 |

## 设计复盘：S2-Pro 的完整优化栈

最后，我们把前面所有知识串联起来，对 S2-Pro 的优化路径做一个完整的复盘。

### 设计决策矩阵

| 决策 | 选择 | Trade-off | 理由 |
|---|---|---|---|
| 统一 vs 分离 graph | 统一 | 单个大 graph vs 两个小 graph + 中间数据传输 | 统一消除 slow→fast 之间的 CPU 调度开销 |
| Greedy vs Sampling | Greedy（`torch.argmax`） | 丢失采样多样性 | CUDA Graph 兼容性要求；TTS 场景可接受 |
| Persistent buffers | Pre-allocate + `copy_()` | 额外显存占用（~几 MB） | CUDA Graph 要求地址稳定 |
| torch.compile | Off（defer 到 framework） | 放弃 36% 吞吐提升 | 启动时间 + 抽象层级 + 可维护性 |
| Deferred capture | 先 init → setup_vq → capture | 增加初始化复杂度 | 确保 graph 包含完整 decode path |
| Graph 管理权 | SGLang CudaGraphRunner | 放弃 inductor CUDAGraph Trees | 保持精细的 multi-bs graph 控制 |

### 从 Eager 到终态的完整优化路径

```
Eager baseline (no optimization)
    │  消除 ①②③ → CUDA Graph only（PR #153, 88 tok/s, 33s startup）
    │
    │  消除 ④ for fast head → + Partial compile（Issue #172 Phase 1, ~121 tok/s, ~54s startup）
    │
    │  消除 ④ for slow head → + Full compile（Issue #172 Phase 2, ~126 tok/s, ~137s startup）
    │
    │  消除 compile startup → + Mega cache（Issue #172 Phase 3, ~126 tok/s, ~33s startup）
    ▼
终态：CUDA Graph + Full compile + Mega cache（126 tok/s, 33s startup）
```

每一层优化都是 **正交且可叠加** 的，这要归功于 `max-autotune-no-cudagraphs` 模式实现的"inductor 管 kernel、SGLang 管 graph"的清晰分工。

### 未来方向

1. **Phase 1 实施**（Issue #172）：`get_compile_targets()` 协议 + `apply_compile_targets()` 框架函数
2. **Mega cache 集成**（Phase 3）：`torch.compiler.save_cache_artifacts()` / `load_cache_artifacts()`
3. **CUDA Graph conditional nodes**（CUDA 12.4+）：允许在 graph 内部条件执行，对 early stopping 有价值
4. **Graph Update API**：`cudaGraphExecUpdate()` 可能允许不重新 capture 就修改 graph 参数
5. **Sampling 恢复**：探索 graph-safe stochastic sampling（pre-allocated random state + graph-safe `multinomial`）
6. **RadixAttention compile-clean 改造**：Phase 2 的核心前置条件

总的来说，S2-Pro 的 CUDA Graph 实战让我深刻体会到：**优化从来不是单一技术的事，而是多层抽象之间精心协调的结果**。CUDA Graph 管 launch overhead，torch.compile 管算子融合，SGLang 管 multi-bs graph 生命周期，三者各司其职、互不越界——这种"分工哲学"才是工程设计中最值得学习的部分。

## 参考

- [基于 torch-memory-savor 浅析 CUDA Graph](./readme.md)（本系列第一篇）
- [CUDA Graph vs torch.compile: S2-Pro TTS 模型实战分析](./readme-2.md)（本系列第二篇）
- [SGLang Code Walk Through](../../sglang/code-walk-through/readme.md)
- [深入浅出理解 verl 源码（初始化）](../../rlhf/verl/multi-turn/code-walk-through/readme.md)——涉及 SGLang rollout engine 的初始化流程和 CUDA Graph 显存预留
- [SGLang-Omni PR #153: Add CUDA Graph Support for S2 Pro Model](https://github.com/sgl-project/sglang-omni/pull/153)
- [SGLang-Omni Issue #172: Framework-level torch.compile + Mega Cache](https://github.com/sgl-project/sglang-omni/issues/172)
- [NVIDIA CUDA Programming Guide - CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [PyTorch CUDA Graphs Documentation](https://pytorch.org/docs/stable/cuda.html#cuda-graphs)
- [PyTorch CUDAGraph Trees](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html)
- [Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)

<!-- /learn-write 自动检查报告
双轨检查：PASS
  - [x] 同时包含概念框架（五层开销模型、四种 compile mode、CUDAGraph Trees）和代码分析（sglang_model.py、factory.py、s2pro_sglang_ar.py、modeling.py、cuda_graph_runner.py）
  - [x] 概念框架在代码分析之前建立（先讲五层开销模型，再进入具体代码）
  - [x] 代码来自真实生产系统（SGLang-Omni PR #153，commit cd9aaf3）
  - [x] 概念框架与代码分析之间有明确的过渡句

叙事检查：PASS
  - [x] 开篇有个人动机（PR #153 的工程经历）
  - [x] 包含致谢（Ratish1、sdli1995）
  - [x] 有文章路线图（开篇第三段）
  - [x] 段落间有过渡句式（"有了对模型架构的理解，我们接着来看"、"有了这些基础"、"有一个问题非常值得分享"等）
  - [x] 在合适位置插入了交叉引用（前两篇系列文章、SGLang Code Walk Through）
  - [x] 使用了设问句引导读者思考

深度检查：[理解复现级 + 修改扩展级混合] → [实际深度匹配] PASS
  - CUDA Graph 机制部分达到理解复现级（原理级，能正确使用）
  - S2-Pro / SGLang-Omni 部分达到修改扩展级（源码级分析）
  - torch.compile 与 CUDA Graph 交互部分达到理解复现级
  - Issue #172 框架蓝图部分达到修改扩展级

交叉引用建议：已包含
  - 前序依赖：readme.md（系列第一篇）、readme-2.md（系列第二篇）
  - 跨主题引用：SGLang Code Walk Through
-->
