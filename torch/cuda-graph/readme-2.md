# CUDA Graph vs torch.compile: S2-Pro TTS 模型实战分析

## 背景

S2-Pro 是一个 TTS 模型，采用两阶段 decode 架构：
- **Slow head（文本模型）**：基于 Qwen3 架构的 transformer，36 层，使用 SGLang 的 RadixAttention + paged KV cache
- **Fast head（codebook loop）**：audio decoder，每步执行 9 次小矩阵运算（embedding lookup → linear projection → forward_kvcached），生成 10 个 codebook token

在 `s2-optimize` 分支中，我们将 codebook loop 从外部的后处理步骤移入了模型的 `forward()` 方法内，使得整个 decode step（slow head + fast head）可以被一个 CUDA graph 捕获。

## 问题：CUDA graph capture 从秒级变成了分钟级

### 现象

在 `main` 分支上，CUDA graph capture 很快（几秒）。切到 `s2-optimize` 分支后，capture 耗时变成了分钟级，日志中出现大量 Triton autotune 信息：

```
SingleProcess AUTOTUNE benchmarking takes 0.2120 seconds and 0.2071 seconds precompiling for 18 choices
AUTOTUNE mm(8x2560, 2560x6144) ...
AUTOTUNE mm(8x4096, 4096x2560) ...
AUTOTUNE mm(8x9728, 9728x2560) ...
```

### 根因

`factory.py` 中设置了 `server_args.enable_torch_compile = True`。

SGLang 的 `CudaGraphRunner` 在捕获 CUDA graph 时，如果 `enable_torch_compile=True`，会对**整个 model forward** 调用：

```python
torch.compile(model.forward, mode="max-autotune-no-cudagraphs")
```

`max-autotune` 模式下，torch.inductor 会为每个 GEMM shape 生成 Triton kernel，并对 18 个候选 kernel 做基准测试。SGLang 默认捕获 12 个 batch size（`[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64]`），其中 bs ≤ 32 的 8 个都会走 torch.compile。每个 bs 下，36 层 transformer 的每个线性层都要跑一遍 autotune，导致总耗时达到分钟级。

在 `main` 分支上，`torch.compile` 只应用于 codebook loop 这一个小函数（`_codebook_loop_impl`），而且是在 CUDA graph 之外独立运行的，所以不影响 graph capture 速度。

## 核心讨论：torch.compile 和 CUDA graph 的区别与配合

### 两者消除的开销不同

| 优化技术 | 消除的开销 | 机制 |
|---------|-----------|------|
| **CUDA graph** | kernel launch overhead（CPU→GPU 调度开销） | 将整个 forward 的所有 CUDA kernel 调用录制为一个 graph，replay 时一次性提交，零 launch 开销 |
| **torch.compile** | 算子融合（多个小 kernel 合成一个大 kernel，减少显存读写） | 通过 torch.inductor 将 PyTorch 算子编译为优化的 Triton kernel，融合相邻算子 |

### 对 Transformer（Slow Head）的效果

**CUDA graph**：有效。消除了每层 attention + FFN 中几十个 kernel 的 launch 开销。

**torch.compile**：**收益极小**。Transformer 的核心计算是大矩阵乘法（GEMM），已经由 cuBLAS 高度优化。从 autotune 日志可以验证——cuBLAS 的 `mm` 在多数 shape 下反而击败了 Triton kernel：

```
AUTOTUNE mm(8x4096, 4096x2560)
  mm                0.0163 ms  100.0%    ← cuBLAS 赢了
  triton_mm_21      0.0166 ms   98.5%
```

attention 部分使用 FlashAttention 3，也是专门优化的 kernel，torch.compile 无法改善。

### 对 Codebook Loop（Fast Head）的效果

**CUDA graph**：**非常有效**。codebook loop 的特点是 9 步循环，每步都是小矩阵运算（`[bs, 1, dim]`），计算量极小，瓶颈完全在 kernel launch overhead。CUDA graph 一招解决，拿到 80%+ 收益。

**torch.compile**：在**没有 CUDA graph** 的情况下非常有效（main 分支实测 5x 加速），因为它把多个小算子融合成一个 kernel，同时减少了 launch 次数和显存 round-trip。但**在已有 CUDA graph 的情况下**，launch 开销已被消除，torch.compile 只剩算子融合带来的显存带宽收益，增量很小。

### 为什么 SGLang 上 LLM 通常同时开两者？

LLM decode 的工作负载和 codebook loop 不同：

- **LLM decode**：每步只有一次大 GEMM（`hidden × vocab`），kernel 本身耗时长，launch overhead 占比小。torch.compile 的主要收益是**融合每层 transformer 中的 layernorm + residual + activation 等小算子链**，这些小算子是显存带宽瓶颈，融合后能显著提升吞吐。
- **Codebook loop**：9 步循环的小矩阵运算，瓶颈是 launch overhead 而非算子融合，CUDA graph 已经足够。

所以"两者同时开"的收益大小取决于模型的计算特征。对于有大量小算子链的 LLM transformer，两者叠加收益明显；对于以 launch overhead 为瓶颈的 codebook loop，CUDA graph 已经吃掉了绝大部分收益。

## 结论

对 S2-Pro 的 unified decode（slow head + fast head in one CUDA graph）：

1. **关闭 `enable_torch_compile`**：CUDA graph capture 从分钟级恢复到秒级，推理速度基本不受影响。
2. **CUDA graph 是主要优化手段**：对 codebook loop 消除 launch overhead，对 transformer 部分同样有效。
3. **torch.compile 作为可选的后续优化**：如果 benchmark 显示 codebook loop 仍有瓶颈，可以单独对 `_decode_codebooks` 做 torch.compile + warmup，然后再捕获 CUDA graph（参见 issue.md）。



# Issue: 探索对 codebook loop 单独启用 torch.compile + CUDA graph 的叠加优化

## 背景

S2-Pro 的 unified decode 将 slow head（transformer）和 fast head（codebook loop）统一到一个 `forward()` 中，整体由 CUDA graph 捕获。

当前已关闭全局 `enable_torch_compile`，因为它对 transformer 部分几乎无收益（transformer 部分已经选取了局部最优的 kernel，并不需要进一步做算子融合），却导致 CUDA graph capture 从秒级膨胀到分钟级（inductor 对每个 GEMM shape × 每个 batch size 做 Triton autotune）。

但 codebook loop（`_decode_codebooks`）的计算特征——9 步循环、小矩阵、大量 embedding/projection 小算子链——理论上仍能从 torch.compile 的算子融合中获益。在 `main` 分支上，torch.compile 对 codebook loop 实测有 5x 加速（无 CUDA graph 时）。现在 CUDA graph 已消除 launch overhead，但 torch.compile 的**显存带宽优化**（融合小算子减少中间 tensor 读写）是否还有增量收益，需要实测。

## 需要研究的内容

### 1. 单独 torch.compile `_decode_codebooks` 的可行性

在 CUDA graph capture 之前，对 `_decode_codebooks` 方法单独调用 `torch.compile`，然后做一次 warmup forward 让编译完成，再捕获 CUDA graph。

关键问题：

- torch.compile 生成的 Triton kernel 能否被 CUDA graph 正确录制？需要确认 inductor 输出的 kernel 不包含 graph-incompatible 的操作（如动态 shape 断言、host-device sync 等）
- `_decode_codebooks` 中的 `for cb_idx in range(1, self._num_codebooks)` 循环是否能被 `fullgraph=True` 正确展开
- `self._audio_decoder.reset_caches()` 等就地修改操作是否和 torch.compile 的 tracing 兼容

### 2. 实现方案

```python
# factory.py 中，setup_vq_decode 之后、init_device_graphs 之前
text_model._decode_codebooks = torch.compile(
    text_model._decode_codebooks,
    mode="max-autotune",
    fullgraph=True,
)
# Warmup: 让 torch.compile 完成编译和 autotune
with torch.no_grad():
    dummy_logits = torch.randn(max_bs, text_model.vocab_size, device="cuda", dtype=torch.bfloat16)
    dummy_hidden = torch.randn(max_bs, text_model.hidden_size, device="cuda", dtype=torch.bfloat16)
    text_model._decode_codebooks(dummy_logits, dummy_hidden)
    torch.cuda.synchronize()

# 然后再捕获 CUDA graph
model_worker.model_runner.init_device_graphs()
```

### 3. Benchmark 对比

在以下配置下跑 `benchmark_tts_speed.py`，对比 decode 阶段的 per-step latency 和总 RTF：

| 配置 | torch.compile | CUDA graph | 说明 |
|------|:---:|:---:|------|
| A | off | off | baseline eager |
| B | off | on | 当前方案 |
| C | `_decode_codebooks` only | on | 本 issue 探索的方案 |

重点关注指标：
- **decode per-step latency**：B vs C 的差异即为 torch.compile 在 CUDA graph 上的增量收益
- **CUDA graph capture 时间**：C 的 capture 时间应远小于全局 torch.compile（因为只编译 codebook loop 的小算子，不涉及 transformer 的大 GEMM autotune）
- **首次启动时间**：C 方案需要额外的 warmup 编译时间，需要评估是否可接受

## 期待的结果

1. **如果 B ≈ C**（差异 < 5%）：确认 CUDA graph 已经充分优化 codebook loop，不需要 torch.compile，关闭即为最终方案。
2. **如果 C 明显优于 B**（差异 > 10%）：将方案 C 合入，接受额外的 warmup 编译时间换取推理加速。记录具体的 latency 数字和 capture 时间。
3. **如果 C 不可行**（torch.compile 的 Triton kernel 无法被 CUDA graph 录制）：记录失败原因，确认当前方案 B 为最优，关闭此 issue。
