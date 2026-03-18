# 再探 CUDA Graph：TTS 模型中的双 CUDA Graph 优化——学习计划

## 动机定位

本系列第一篇（[基于 torch-memory-savor 浅析 CUDA Graph](./readme.md)）停留在虚拟地址保护的角度，对 CUDA Graph 的理解尚且浅薄。最近在 SGLang-Omni 框架中为 Fish Audio S2 Pro 模型添加 CUDA Graph 支持（[PR #153](https://github.com/sgl-project/sglang-omni/pull/153)），才发现 CUDA Graph 的博大精深。通过双 CUDA Graph 同时执行 S2 Pro 的 slow head 和 fast head，TPS 从 55.6 提高到了 88。结合 torch.compile 的进一步测试：

| Configuration | Startup time | Steady-state throughput |
|---|---|---|
| No compile (CUDA graph only) | ~33s | 88 tok/s |
| Partial compile (fast head only) | ~54s | 121 tok/s |
| Full-model compile | ~137s | 126 tok/s |

> 注：TPS 衡量的是 TTS 模型 LLM backbone 产生 speech codec tokens 的速度，不包含 vocoder 阶段。

### 开篇风格要点

写作时，开篇应：
1. **先回顾前序文章**，建立系列连续性（"去年 8 月，我浅浅写过..."），而不是"因为工作需要"这种模板化开头
2. **开篇就亮 benchmark 数据**（TPS 提升 + 三配置对比表格），用成果抓读者
3. **路线图用精炼编号列表**（4 条以内），不要用长段落描述文章结构
4. **torch.compile 讨论明确留作后文**——本文核心聚焦 CUDA Graph 本身
5. **致谢随意自然**（"各位大哥"风格），不需要标注公司/组织
6. **不要写"本文基于 commit xxx 进行分析"**这类模板声明——commit hash 在代码引用时自然出现即可

### 核心问题

这个 PR 暴露了一系列需要深入理解的问题：

1. **deferred graph capture 模式**：为什么 `factory.py` 需要先 `disable_cuda_graph=True` 初始化 ModelWorker，再调用 `setup_vq_decode()` 和 `setup_caches()` 之后才 `init_device_graphs()`？这背后的 CUDA Graph 约束是什么？
2. **persistent buffer 设计**：`sglang_model.py` 中的 `_vq_codes`、`_vq_mask`、`_output_codes`、`_output_semantic_ids` 四个 persistent tensor 是 CUDA Graph 安全的核心，但为什么 pre-allocate + `copy_()` 就能保证 graph 中的指针稳定性？
3. **torch.compile 的兴衰**：PR 的第三个 commit（`c962aa6`）加了一行 `server_args.enable_torch_compile = True`，然后 launch 时间从 33s 膨胀到 137s——Triton autotune 在 graph capture 期间被触发了。但 Ratish1 的 benchmark 显示 partial compile（仅 fast head）能把吞吐从 88 提到 120 tok/s，这 36% 的增量来自哪里？最终为什么还是选择了不 compile？
4. **两种 KV cache 的共存**：slow head 用 SGLang 的 RadixAttention + paged KV cache，fast head 用 `FishQwen3AudioDecoder` 的 static KV cache（`KVCache` 类，每层独立预分配 `[max_bs, num_codebooks+1, n_heads, head_dim]`）。两种 cache 如何在同一个 CUDA Graph 中共存而不冲突？
5. **CUDA Graph 与 torch.compile 的深层关系**：为什么 SGLang 选择 `max-autotune-no-cudagraphs` 而不是 `reduce-overhead`？inductor 的 CUDAGraph Trees 与 SGLang 的 `CudaGraphRunner` 为什么不能共存？`fullgraph=True` 对 codebook loop 的 trace 有什么具体挑战？
6. **Framework-level compile 的工程蓝图**（Issue #172）：如何设计 `get_compile_targets()` 协议让模型声明可编译的 target？三阶段计划（partial compile → global compile → mega cache）各自的技术挑战和 trade-off 是什么？

### 系列定位

已有两篇 CUDA Graph 相关文章：

- [基于 torch-memory-savor 浅析 CUDA Graph](./readme.md)（系列第一篇）：CUDA Graph 基本概念、推理常用/训练少用的原因、torch-memory-saver 的 `cuMemMap` 虚拟地址保护。
- [CUDA Graph vs torch.compile: S2-Pro TTS 模型实战分析](./readme-2.md)（系列第二篇，即本次草稿）：两种优化技术消除的开销类型对比。

这两篇都停留在"使用层"分析。本次学习聚焦 **CUDA Graph 与具体工程实现的结合**——**先建立 CUDA Graph 的五条核心约束作为概念框架**，再以 PR #153 为叙事主线，用概念框架分析 S2-Pro 双头架构的挑战，最后深入 deferred capture、persistent buffer、SGLang CudaGraphRunner 等工程落地代码。关于 torch.compile 的讨论（四种 mode、CUDAGraph Trees、Issue #172 蓝图），本文只做概要引出，详细分析留给后续文章。

此外，本 topic 还与以下已有文章产生关联：
- [SGLang Code Walk Through](../../sglang/code-walk-through/readme.md)：SGLang 的整体架构，理解 ModelRunner → Model 的 forward 调用链路
- [SGLang Worker 架构](../../sglang/sglang-worker/readme.md)：`init_cuda_graphs` 在 ModelRunner 初始化中的位置
- [深入浅出理解 verl 源码（初始化）](../../rlhf/verl/multi-turn/code-walk-through/readme.md)：verl 中 SGLang rollout engine 的初始化流程，涉及 CUDA Graph 显存预留

## 前置知识检查

学习本 topic 之前，建议回顾以下内容：

- [基于 torch-memory-savor 浅析 CUDA Graph](./readme.md)：理解 CUDA Graph 的 DAG 结构、capture/replay 基本概念、虚拟地址稳定性问题
- [SGLang Code Walk Through](../../sglang/code-walk-through/readme.md)：理解 SGLang 的 Server → Scheduler → ModelRunner → Model 调用链路，以及 `ForwardBatch` 的数据流转
- [CUDA Graph vs torch.compile: S2-Pro TTS 模型实战分析](./readme-2.md)：理解 S2-Pro 的 slow head / fast head 架构，以及两种优化技术的开销维度对比

## 学习路线图

> **核心原则：概念 → 模型 → 代码。** 必须先建立 CUDA Graph 的概念框架，让读者拿到"分析工具箱"，再介绍模型特点并用概念框架解释其挑战，最后才进入工程代码。绝对不能反过来。

### 第一步：CUDA Graph 概念框架——建立分析工具箱

- **深度层级**：理解复现（CUDA Graph 是依赖的基础设施）
- **目标**：让读者理解 capture/instantiate/replay 三阶段机制和五条核心约束，为后续所有工程分析提供概念基础
- **方法**：概念框架为主，辅以简单示意
- **写作位置**：文章第一个正文章节（开篇之后立即进入）

#### 1.1 capture → instantiate → replay 三阶段

1. **Capture**：CUDA runtime 录制所有 kernel launch 及其参数（包括 GPU 虚拟地址），形成 DAG
2. **Instantiate**：编译为 `cudaGraphExec_t`，做 kernel 参数绑定和 dependency analysis
3. **Replay**：`cudaGraphLaunch()` 一次性提交所有 kernel——CPU 只发一次 launch 指令

#### 1.2 五条核心约束

| 约束 | 含义 | 后续章节中的映射（此处仅提纲，不展开） |
|---|---|---|
| graph 录制的是指针地址 | replay 时必须保证 GPU 虚拟地址不变 | → persistent buffer 的 `copy_()` 设计 |
| capture 期间不能有动态内存分配 | 所有 tensor 必须预分配 | → `setup_vq_decode()` 的 buffer 设计 |
| capture 期间不能有 host-device sync | 不能调用 `.item()`、`torch.multinomial` 等 | → greedy decoding（`torch.argmax`）的选择 |
| graph 中的控制流必须是静态的 | 循环次数、分支条件在 capture 时固化 | → codebook loop 的常量展开 |
| graph 是静态的，录制后不会自动更新 | 改了代码路径必须重新 capture | → deferred graph capture 的初始化顺序 |

#### 1.3 PyTorch 封装与 Memory Pool

- `torch.cuda.CUDAGraph()` / `graph.capture_begin()` / `graph.replay()` 的 API 映射
- Memory Pool 共享（`pool=...`）：多个 graph 复用同一块中间 tensor 内存的前提和好处
- 这些概念会在后续 SGLang CudaGraphRunner 源码走读中具体落地

> 注：系列第一篇已覆盖 CUDA Graph 的基本概念和虚拟地址保护。本步骤重点是**提炼出五条核心约束**，为后续 S2-Pro 分析建立明确的参照系。

### 第二步：S2-Pro 模型架构与双 CUDA Graph 的动机

- **深度层级**：修改扩展（SGLang-Omni 是自己开发的系统）
- **目标**：先让读者理解两个 head 各自的计算特征，再基于此回答"为什么需要双 CUDA Graph"
- **方法**：模型架构 → 计算特征分析 → 核心问题推导
- **写作顺序**：
  1. **2.1 整体架构 + 为什么叫 slow/fast**：先让读者知道 S2-Pro 是 Dual-AR 模型，一个 decode step 要先跑一次大 transformer（slow head，因为慢所以叫 slow），再跑 9 次小 transformer（fast head，因为单次快所以叫 fast，但跑 9 次）。解释命名由来：slow/fast 指的是单次推理的速度——slow head 单次推理涉及 36 层大 transformer，慢；fast head 单次推理只有几层小 transformer，快，但需要自回归地跑 9 次来生成 9 个 codebook token
  2. **2.2-2.3 分别深入两个 head 的架构和计算特征**
  3. **2.4 两种 KV cache 共存**
  4. **2.5 为什么 CUDA Graph 对这个模型有帮助**：结合第一步的概念，解释 S2-Pro decode 的特点（大量重复的 kernel 序列、固定的控制流）天然适合 CUDA Graph。这一点可能比较显然，但需要显式点出，作为 2.6 的铺垫
  5. **2.6 为什么需要双 CUDA Graph**：核心论证——单 graph 只覆盖 slow head 远不够，fast head 的 launch overhead 才是瓶颈
  6. **2.7 PR #153 前后架构对比**
- **写作关键**：2.6 是本文的核心论证。但 2.5（为什么 CUDA Graph 有帮助）虽然显然，也不能跳过——它是从"概念"到"这个模型"的桥梁，让读者确认"CUDA Graph 适用于这个场景"之后，再追问"那为什么需要把两个 head 都放进去"

#### 2.1 S2-Pro Dual-AR 整体架构：为什么叫 Slow Head / Fast Head

S2-Pro 是一个 **Dual-AR（双自回归）TTS 模型**。每个 decode step 的流程是：

1. **Slow head**（text model）：一个 36 层 Qwen3 transformer，输入上一步的 token，输出下一个 semantic token 的 logits。**叫 slow 是因为单次推理慢**——36 层大 transformer，hidden_size=2560，每层 4 个大 GEMM。
2. **Fast head**（audio decoder）：一个小型 transformer，接收 slow head 的 hidden states，自回归地生成 9 个 codebook token。**叫 fast 是因为单次推理快**——层数少、维度小，单次只需 μs 级。但它要连续跑 9 次（9 个 codebook）。

命名的关键：slow/fast 描述的是**单次推理的延迟**，不是总耗时。Slow head 单次慢但只跑一次；fast head 单次快但跑 9 次。

#### 2.2 Slow Head 细节：`S2ProSGLangTextModel`（基于 Qwen3）

分析 `sglang_model.py`（commit `cd9aaf3`）中的 text model 实现：

- **模型规格**：36 层 `S2ProDecoderLayer`，`hidden_size=2560`，`intermediate_size=9728`，`num_heads=32`，`num_kv_heads=8`（GQA），`head_dim=128`
- **核心计算路径**：`embed_tokens` → 36 × (`S2ProAttention` + gate_up_proj/down_proj FFN) → `RMSNorm` → `lm_head`
- **注意力机制**：使用 SGLang 的 `RadixAttention`（带 paged KV cache），通过 `QKVParallelLinear` 做 tensor parallel，`RoPE`（非 NeoX 格式，`is_neox_style=False`）
- **计算特征**：每层包含 4 个大 GEMM（qkv_proj、o_proj、gate_up_proj、down_proj），单个 kernel 耗时长（ms 级），launch overhead 相对占比小
- **关键数值**：以 bs=8 为例，主要的 GEMM shape 是 `mm(8×2560, 2560×6144)` 和 `mm(8×4096, 4096×2560)` 等——这些在 cuBLAS 中已经高度优化

#### 2.3 Fast Head 细节：`FishQwen3AudioDecoder`（Codebook Loop）

分析 audio decoder 的实现：

- **架构**：独立的小型 transformer（层数远小于 text model），包含 `project_in`（text_dim → fast_dim 线性投影）、多层 `TransformerBlock`、`RMSNorm`、`output` 线性头
- **KV Cache**：**static KV cache**，每层独立预分配 `KVCache(max_batch_size, num_codebooks+1, n_local_heads, head_dim)`，与 SGLang 的 paged KV cache 完全独立
- **codebook_embeddings**：`nn.Embedding(vocab_size × num_codebooks, text_dim)`——共享的 codebook embedding 表，用于将 VQ codes 映射回 text model 的维度空间
- **codebook_offsets**：`torch.arange(num_codebooks) * vocab_size`——向量化 embedding 查找的偏移量，避免 per-codebook 的独立 embedding 表
- **`forward_kvcached()` 流程**：接收 `[bs, 1, dim]` 的 embedding，通过预分配的 `input_pos` buffer（`fill_(codebook_idx)` 实现 CUDA Graph 安全的标量更新），执行 attention with KV cache → norm → output projection
- **9 步循环的计算特征**：每步只有一次小 GEMM（`[bs, 1, fast_dim]` 级别），计算量极小（μs 级），但需要 embedding lookup → linear projection → forward_kvcached → argmax → embedding lookup 的完整序列，产生大量 kernel launch

#### 2.4 两种 KV Cache 的共存

这是一个值得深入理解的设计点：

- **SGLang paged KV cache**（slow head）：由 `token_to_kv_pool_allocator` 管理，支持 prefix caching 和 dynamic 分配，attention backend 为 FlashAttention 3
- **Static KV cache**（fast head）：由 `audio_decoder.setup_caches()` 预分配固定大小的 tensor，每次 `reset_caches()` 时 `zero_()` 清空，不参与 SGLang 的内存池管理
- **关键问题**：两种 cache 在 CUDA Graph capture 期间如何保证地址稳定性？paged cache 通过 SGLang 的 `req_to_token_pool` 管理，static cache 通过 `setup_caches()` 一次性分配后不再变化——后者天然 CUDA Graph 安全

#### 2.5 为什么 CUDA Graph 对这个模型有帮助

这一点可能比较显然，但需要显式点出，作为 2.6 的铺垫：

- S2-Pro 的 decode 阶段是**高度重复的固定 kernel 序列**：每个 step 都执行相同的 36 层 transformer + 9 步 codebook loop，控制流完全静态——这正是 CUDA Graph（第一步中的"静态控制流"约束）最擅长的场景
- Decode 阶段的 batch size 在一段时间内相对稳定，SGLang 的 multi-bs graph 管理可以很好地覆盖
- 模型推理是 latency-sensitive 的（TTS 需要实时生成语音），消除 CPU 侧的 launch overhead 对端到端延迟有直接帮助

确认了"CUDA Graph 适用于 S2-Pro"之后，下一个问题是：graph 应该覆盖哪些部分？

#### 2.6 核心问题：为什么需要双 CUDA Graph

**这是整篇文章的驱动问题。**

PR #153 之前，SGLang 的 CUDA Graph 只覆盖 slow head（标准 LLM transformer forward）。Fast head（codebook loop）作为 per-request 后处理运行在 graph 之外。结合 2.2 和 2.3 的计算特征分析：

- **Slow head 的 CUDA Graph 收益有限**：2.2 中分析过，slow head 的核心是 `mm(8×2560, 2560×6144)` 这样的大 GEMM，单 kernel 耗时 ms 级。消除 launch overhead 的相对收益小。
- **Fast head 没有 CUDA Graph 才是真正的瓶颈**：2.3 中分析过，codebook loop 每步是 μs 级的小算子，9 步循环产生大量 kernel launch。Launch overhead 占执行时间的主要比例——**这正是 CUDA Graph 最擅长解决的问题**。
- **两个 head 之间还有 CPU 调度开销**：slow head 的 graph replay 结束后，CPU 需要取回结果、逐 request 调用 codebook loop、再写回——这段 CPU 往返也是开销。

因此，**核心洞察：把 slow head 和 fast head 统一到一个 `forward()` 中，让一个 CUDA Graph 同时捕获两者**。TPS 从 55.6 跃升到 88，增益主要来自 fast head 的 launch overhead 消除。

但"统一到一个 graph"引入了巨大的工程复杂性——这正是后续章节存在的原因：
- 两种 KV cache 必须在同一个 graph 中共存（2.4 已分析）
- 所有动态输入必须通过 persistent buffer 传入（→ 第三步）
- 初始化顺序必须保证 graph 捕获到完整路径（→ 第三步的 deferred capture）
- Codebook loop 的循环和采样必须满足 CUDA Graph 的静态约束（→ 第四步）

#### 2.7 PR #153 之前 vs 之后的架构对比

**之前（分离的处理流程）**：
```
Text Model forward → LogitsProcessorOutput
                        ↓
S2ProSGLangOutputProcessor._codebook_loop_impl()  ← per-request, 不在 graph 内
                        ↓
Codebook codes output
```

**之后（统一的 forward）**：
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

**关键改变**：`_decode_codebooks()` 从外部的 per-request 后处理，变成了 `forward()` 内部的一个步骤。这意味着 CUDA Graph 可以将 transformer + sampling + codebook loop 一次性录制，消除整个 decode step 的所有 kernel launch overhead。

### 第三步：PR #153 的 Deferred Graph Capture 模式——为什么顺序如此重要？

- **深度层级**：修改扩展
- **目标**：从 `factory.py` 源码理解 deferred graph capture 的设计动机，以及为什么 CUDA Graph 对初始化顺序有严格约束
- **方法**：代码分析 + CUDA Graph 约束推理

#### 3.1 `factory.py` 的初始化时序分析

基于 commit `cd9aaf3` 的 `create_s2pro_sglang_engine()`：

```
Step 1: server_args.disable_cuda_graph = True
Step 2: ModelWorker.__init__()           ← 此时不 capture graph
Step 3: _truncate_rope_to_bf16()         ← BF16 精度修正
Step 4: audio_decoder.setup_caches()     ← 预分配 static KV cache
Step 5: text_model.setup_vq_decode()     ← 分配 persistent buffers + 挂载 audio decoder
Step 6: init_device_graphs()             ← 此时 capture，graph 包含完整的 forward + _decode_codebooks
```

**关键问题**：为什么不能在 Step 2 直接 capture？

- `ModelWorker.__init__()` 内部会调用 `init_cuda_graphs()`，此时 `text_model._vq_ready = False`（`setup_vq_decode()` 还没调用）
- 如果此时 capture graph，`forward()` 中的 `if self._vq_ready:` 分支不会执行 → graph 不包含 VQ embedding combination 和 `_decode_codebooks()`
- 后续即使调用了 `setup_vq_decode()`，graph 已经被录制，不会自动更新（CUDA Graph 是静态的）
- 因此必须**先** attach audio decoder 和分配 buffers，**再** capture graph

#### 3.2 `setup_vq_decode()` 的 Buffer 分配设计

分析 `sglang_model.py` 中的 `setup_vq_decode()`：

- **Input buffers**：
  - `_vq_codes`: `[max_bs, num_codebooks]` long tensor——上一步生成的 codebook codes，由 `S2ProSGLangModelRunner._update_vq_buffers()` 在 forward 前写入
  - `_vq_mask`: `[max_bs]` bool tensor——标记哪些 batch 位置是 semantic token（需要 VQ embedding combination）
- **Output buffers**：
  - `_output_codes`: `[max_bs, num_codebooks+1]` long tensor——当前步生成的所有 codes（semantic + 9 codebooks），由 `_decode_codebooks()` 写入
  - `_output_semantic_ids`: `[max_bs]` long tensor——当前步的 semantic token id
- **Auxiliary tensors**：
  - `_semantic_bias`: `[vocab_size]` BF16 tensor——将非 semantic 和非 EOS 的 token logits 设为 `-inf`，实现 constrained decoding
  - `_vq_codebook_embeddings`：直接引用 `audio_decoder.codebook_embeddings`（共享权重）
  - `_vq_codebook_offsets`：引用 `audio_decoder.codebook_offsets`
  - `_vq_scale`：`1.0 / sqrt(num_codebooks + 1)`

**CUDA Graph 安全性分析**：所有 buffer 在 `setup_vq_decode()` 时一次性 `torch.zeros(...)` 分配，之后只通过 `copy_()`、index assignment (`tensor[:bs] = ...`)、`fill_()` 等**就地操作**修改值，不改变 tensor 的底层内存地址。这正是 CUDA Graph 要求的——graph 录制时记录的是指针地址，replay 时必须保证指针不变。

#### 3.3 `_update_vq_buffers()` 和 `_build_outputs()` 的 Buffer 读写协议

分析 `s2pro_sglang_ar.py` 中 `S2ProSGLangModelRunner` 的 buffer 交互：

```
execute(scheduler_output):
  ├── _inject_vq_embeds_prefill()  ← 仅 prefill：构造 input_embeds（包含 VQ embedding）
  ├── _update_vq_buffers()         ← 仅 decode：将上一步的 codes 写入 text_model._vq_codes/_vq_mask
  ├── model_worker.forward()       ← CUDA Graph replay（或 eager 执行）
  └── _build_outputs()             ← 从 text_model._output_codes 读取结果
```

**_update_vq_buffers 详解**：
1. 计算 semantic mask：`semantic_begin_id <= input_ids < semantic_end_id`
2. 将 mask 写入 `text_model._vq_mask`
3. 遍历 batch 中每个 request，将 `last_codebook_values` 写入 `text_model._vq_codes[i]`

**_build_outputs 详解**：
1. 遍历 scheduled requests
2. 从 `text_model._output_codes[i]` 读取 `[num_codebooks+1]` 维的 code vector
3. unsqueeze 为 `[num_codebooks+1, 1]` 格式，包装为 `S2ProStepOutput`

**关键洞察**：buffer 的读写发生在 CUDA Graph boundary 之外（`_update_vq_buffers` 在 forward 前，`_build_outputs` 在 forward 后），但 buffer 本身被 graph 内部的 kernel 引用。这种"外部写值、graph 内部读地址"的模式是 CUDA Graph 与动态输入兼容的标准做法。

### 第四步：CUDA Graph 约束的工程落地——从概念到 S2-Pro 代码

- **深度层级**：理解复现 + 修改扩展
- **目标**：将第一步建立的五条约束逐一映射到 PR #153 的具体设计选择，展示概念如何在代码中落地
- **方法**：概念-代码双轨对照
- **写作关键**：每个代码设计选择都要显式回指第一步的某条约束（"这正是第一步中提到的'指针稳定性'约束的体现"）

#### 4.1 五条约束的工程映射表

| CUDA Graph 约束 | PR #153 中的应对策略 |
|---|---|
| capture 期间所有 kernel launch 被记录而非执行 | `setup_vq_decode()` 必须在 capture 前调用，确保 `_vq_ready=True`，让 `_decode_codebooks()` 的 kernel 被录制 |
| capture 期间不能有动态内存分配 | 所有 buffer（`_vq_codes`、`_output_codes` 等）在 capture 前预分配为 `max_batch_size` 大小 |
| capture 期间不能有 host-device sync | `_decode_codebooks()` 用 `torch.argmax`（greedy）替代了之前的 stochastic sampling，避免了动态采样可能引发的 sync |
| graph replay 时必须保证 pointer 稳定性 | persistent buffers 只通过 `copy_()`、index assignment 修改值，不重新分配 |
| graph 中的控制流必须是静态的 | codebook loop 的循环次数 `num_codebooks` 是常量（编译时确定），`for cb_idx in range(1, self._num_codebooks)` 在 capture 时被完全展开 |

#### 4.2 Greedy Decoding：host-device sync 约束的体现

PR #153 将采样策略从 `_sample_with_topk`（temperature + top_k + top_p + repetition_penalty + RAS）切换为 `torch.argmax(biased_logits, dim=-1)`。这个改变不仅仅是简化——它是第一步中"不能有 host-device sync"约束的直接体现：

- `torch.argmax` 是确定性的、无状态的、不需要随机数生成器 → 完全可以被 CUDA Graph 录制
- top_k/top_p sampling 涉及 `torch.multinomial`，可能需要 random state 管理和动态 shape 操作 → graph-incompatible
- Copilot review 指出了这个 behavioral change（sampling parameters 被忽略），这是一个有意的 trade-off

#### 4.3 `input_pos.fill()` 模式：指针稳定性约束的体现

`forward_kvcached()` 中使用 `self.input_pos.fill_(codebook_idx)` 来设置位置信息——这是第一步中"指针稳定性"约束的精心实践：

- `input_pos` 是 `register_buffer` 注册的 persistent tensor，地址不变
- `fill_()` 是就地操作，修改值而不重新分配
- `codebook_idx` 是 Python 常量（循环展开后），在 capture 时被固化为 graph 的一部分
- 这比 `torch.tensor([codebook_idx])` 安全得多——后者会创建新 tensor，破坏地址稳定性

### 第五步：SGLang CudaGraphRunner 源码分析——与 S2-Pro 的交互

- **深度层级**：修改扩展（SGLang 是自己开发的系统）
- **目标**：理解 `CudaGraphRunner` 如何管理多 batch size 的 graph 实例，以及 S2-Pro 的特殊需求如何与之配合
- **方法**：源码走读

#### 5.1 `CudaGraphRunner` 的 capture 流程

分析 SGLang 源码 `sglang/srt/model_executor/cuda_graph_runner.py`：

- **capture_bs 列表**：默认 `[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64]`（12 个 batch size）
- **capture 顺序**：从大到小（最大 bs 先 capture）。原因：大 bs 需要更多内存，先 capture 大 bs 可以让 memory allocator "看到" 最大的内存需求，后续小 bs 的 capture 可以复用已分配的内存
- **warmup run**：每个 bs 在 capture 前先做一次 eager forward，触发所有可能的内存分配（包括 cuBLAS workspace、attention buffer 等），确保 capture 时不会有意外的 allocation

#### 5.2 S2-Pro 对 capture 流程的影响

当 `text_model._vq_ready = True` 时，capture 的 forward 包含：

1. VQ embedding combination（读取 `_vq_codes`、`_vq_mask`）
2. 36 层 Transformer（RadixAttention + FFN）
3. logits 计算
4. `_decode_codebooks()`：constrained sampling + 9 步 codebook loop

**额外的 capture 约束**：
- audio decoder 的 static KV cache 必须在 capture 前通过 `setup_caches(max_batch_size=max_bs)` 分配好
- `_decode_codebooks()` 中的 `self._audio_decoder.reset_caches()` 是 `zero_()` 操作——就地操作，graph 安全
- codebook loop 中的 `forward_kvcached()` 涉及 audio decoder 自己的 attention 计算——这些 kernel 也会被录入 graph

**关键问题**：capture 一个包含 9 步 codebook loop 的 graph，意味着 graph 中包含了约 `36 × 4 (transformer GEMM) + 9 × N (codebook loop kernels)` 个 kernel node。这比普通 LLM 的 graph 显著更大。需要分析 graph size 对 replay latency 的影响。

#### 5.3 Graph Replay 中的 bs padding

当 actual batch size < captured batch size 时：

- `CudaGraphRunner` 将 actual input 拷贝到预分配 buffer 的前 `actual_bs` 行
- graph replay 仍然执行完整的 captured_bs 个 kernel，多余的行产生无效计算
- 对于 S2-Pro，padding 意味着 `_decode_codebooks()` 也会对 padding 行执行完整的 9 步 codebook loop——这是额外的浪费。但由于 codebook loop 是小矩阵运算，额外开销很小

#### 5.4 `can_run_cuda_graph` 判断逻辑

分析什么情况下 S2-Pro 会 fallback 到 eager mode：

- prefill 阶段：序列长度不固定 → 不走 graph
- decode 阶段：bs 超过最大 capture bs → fallback
- chunked prefill：不走 graph
- S2-Pro 的 extend 模式（`forward_batch.forward_mode.is_extend()`）：走 eager

### 第六步：CUDA Graph 与 torch.compile 的深层关系——两套优化体系如何共存

- **深度层级**：理解复现（CUDA Graph 和 torch.compile 都是依赖的基础设施）
- **目标**：从 PyTorch 内部机制理解 CUDA Graph 和 torch.compile 各自管理 GPU 执行的方式、两者的正交性与冲突点，为后续的工程决策（PR #153 和 Issue #172）提供理论基础
- **方法**：概念框架 + PyTorch 源码分析

#### 6.1 两套优化体系的本质差异

首先需要建立一个清晰的概念框架——CUDA Graph 和 torch.compile 解决的是 **GPU 执行流水线中不同阶段** 的开销：

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
| ①Python overhead | 完全消除（replay 不经过 Python） | 大幅减少（traced graph 绕过 Python dispatch） |
| ②框架 dispatch | 完全消除 | 大幅减少 |
| ③launch overhead | **完全消除**（一次 launch 提交所有 kernel） | 部分减少（融合后 kernel 数量减少） |
| ④显存带宽 | 不影响 | **显著优化**（算子融合减少中间 tensor 读写） |
| ⑤算术计算 | 不影响 | 可能优化（Triton kernel 可能比 cuBLAS 更优，但也可能更差） |

**关键洞察**：两者在开销③上有重叠（都能减少 kernel launch），但在④上只有 torch.compile 有效。这解释了为什么：
- CUDA Graph alone 对 codebook loop（③是瓶颈）效果极好
- 但 CUDA Graph + torch.compile（④的额外优化）仍然有 36% 的增量——因为 codebook loop 的小算子链仍然有大量的中间 tensor 显存读写

#### 6.2 torch.compile 的四种 mode 与 CUDA Graph 的关系

| Mode | 含义 | CUDA Graph 行为 | S2-Pro 适用性 |
|---|---|---|---|
| `default` | 基本优化，不做 autotune | inductor 不管理 CUDA graph | 可用但收益小 |
| `reduce-overhead` | 优化 + inductor 内部自动管理 CUDA graph | **inductor 自己 capture/replay graph**（CUDAGraph Trees） | **与 SGLang 的 CudaGraphRunner 冲突** |
| `max-autotune` | 全力 autotune + inductor 管理 CUDA graph | 同 reduce-overhead，inductor 自己管理 graph | **冲突** |
| `max-autotune-no-cudagraphs` | 全力 autotune，但**不**让 inductor 管理 graph | inductor 只生成优化的 kernel，不碰 graph | **SGLang 使用的 mode** |

**为什么 SGLang 使用 `max-autotune-no-cudagraphs`？**

这是一个关键的架构决策：SGLang 有自己的 `CudaGraphRunner` 负责 graph capture/replay，如果 inductor 也自己做 graph capture（`reduce-overhead` 或 `max-autotune` mode），就会产生 **"graph 里套 graph"** 的冲突——外层是 SGLang 的 graph capture，内层是 inductor 的 CUDAGraph Trees，两者争抢 stream capture 的控制权。

因此 SGLang 选择 `no-cudagraphs` 后缀：让 inductor 只负责 **kernel 优化**（算子融合 + Triton autotune），而 **graph 管理** 留给 SGLang 自己。这是一种"分工"模式：

```
torch.compile(mode="max-autotune-no-cudagraphs") → 生成优化的 Triton kernel
                          ↓
SGLang CudaGraphRunner.capture() → 将这些优化后的 kernel 录入 CUDA Graph
                          ↓
SGLang CudaGraphRunner.replay() → 一次性提交所有优化后的 kernel
```

#### 6.3 CUDAGraph Trees：inductor 内部的 graph 管理机制

虽然 SGLang 不使用 inductor 的 graph 管理，但理解其机制有助于理解为什么两套系统不能叠加：

- **CUDAGraph Trees** 是 inductor 在 `reduce-overhead` mode 下的内部实现：为每个不同的执行路径（不同 shape、不同 control flow 分支）创建独立的 graph，以树状结构组织
- **Memory Pool 共享**：所有 graph 共享一个 memory pool，"不增加额外的内存开销"
- **Graph Break 处理**：当 torch.compile 遇到无法 trace 的操作（如 `.item()`、动态 control flow），会产生 graph break，将代码切分为多个 graph partition——每个 partition 被独立 capture
- **执行流程**：首次调用 warmup → 第二次调用录制 graph → 后续调用 replay

**与 SGLang CudaGraphRunner 的冲突点**：
- SGLang 的 `CudaGraphRunner` 也在 stream 上做 capture。如果 inductor 已经在内部 capture 了一部分 kernel（CUDAGraph Trees），那么 SGLang 的外层 capture 会"看到"一个已经被 graph 化的执行——这可能导致 "graph 中嵌套 graph" 的未定义行为
- Memory pool 管理：inductor 和 SGLang 各自管理 memory pool，两者不互通

#### 6.4 `fullgraph=True` 的约束与 S2-Pro 的挑战

`fullgraph=True` 要求 torch.compile 将整个被编译的函数 trace 为**一个**完整的 FX graph，不允许任何 graph break。这对 S2-Pro 的 `_decode_codebooks()` 意味着：

**必须满足的条件**：
1. `for cb_idx in range(1, self._num_codebooks)` 循环必须能被 torch.compile 完全展开（`num_codebooks` 是常量 → 可以展开）
2. `self._audio_decoder.reset_caches()` 中的 `zero_()` 必须是 torch.compile 支持的操作 → 是，就地操作可被 trace
3. `self._audio_decoder.forward_kvcached()` 中不能有 graph break → 需要验证 attention 中的 KV cache index 操作是否 compile-clean
4. `torch.argmax` 必须可 trace → 是

**潜在的 graph break 风险**（Issue #172 Phase 2 需要解决的）：
- `RadixAttention`（slow head）：涉及 paged KV cache 的动态索引 → 可能 graph break
- `ForwardBatch` 的动态属性访问 → 可能 graph break
- `MergedColumnParallelLinear` 的 tensor parallel 通信 → 需要验证

这也是为什么 Issue #172 将 backbone compile（Phase 2）排在 auxiliary module compile（Phase 1）之后——**fast head 的 `_decode_codebooks` 更容易满足 `fullgraph=True`，而 slow head 有大量 SGLang 特有的动态操作**。

#### 6.5 inductor 生成的 Triton kernel 能否被外部 CUDA Graph 录制？

这是 `no-cudagraphs` 模式下的核心假设：inductor 生成的 Triton kernel 必须是"普通"的 CUDA kernel，能够被 SGLang 的 `CudaGraphRunner` 正常录制。

**答案是：可以，但有条件**：
- inductor 在 `no-cudagraphs` 模式下输出的 Triton kernel 是标准的 CUDA kernel（通过 Triton 的 PTX codegen），与 cuBLAS kernel 一样可以被 stream capture 录制
- 但 inductor 的 **guard 机制** 可能在 graph replay 时触发 recompilation——如果输入 tensor 的 shape/stride/dtype 与 trace 时不同，inductor 会尝试重新编译，这在 CUDA Graph replay 期间是不允许的
- SGLang 通过 `CudaGraphRunner` 的 **固定 bs + padding 策略** 规避了这个问题：每个 captured bs 对应一组固定 shape 的输入，guard 不会触发

### 第七步：PR #153 的迭代故事——torch.compile 在 S2-Pro 中的兴衰

- **深度层级**：修改扩展
- **目标**：通过 PR 的 7 个 commit 和 review 讨论，将第五步的理论知识映射到 S2-Pro 的实际工程决策
- **方法**：PR 考古 + 数据分析

#### 7.1 七个 Commit 的叙事线

| 序号 | Commit | 内容 | 意义 |
|---|---|---|---|
| 1 | `c153ae9` | "unified slow/fast head gaining huge efficiency gain" | 核心实现：统一 forward + persistent buffers |
| 2 | `f621355` | "lint" | 代码规范 |
| 3 | `c962aa6` | "torch.compile added in" | **转折点**：加入 `server_args.enable_torch_compile = True`，触发了 launch 时间问题 |
| 4 | `78aafc7` | "setup_vq_decode before CUDA graph capture" | **关键修复**：deferred graph capture 模式 |
| 5 | `dccf122` | "[refactor] tts eval for voice cloning" | Benchmark 重构 |
| 6 | `cf9396d` | "[feature] export server output of tokens" | 输出接口调整 |
| 7 | `20be04a` | "Acknowledge torch.compile discussion" | **最终决策**：移除 torch.compile，记录分析 |

**迭代的关键转折**：
- Commit 3 → 4：发现 torch.compile 会在 graph capture 期间对**整个 model forward** 触发 Triton autotune。SGLang 的 `CudaGraphRunner` 在 `enable_torch_compile=True` 时会调用 `torch.compile(model.forward, mode="max-autotune-no-cudagraphs")`，对 36 层 transformer × 12 个 bs 的每个 GEMM shape 做 18 候选 kernel 的 benchmark。
- Commit 7：基于 Ratish1 的 benchmarking 结果，正式移除 torch.compile。

#### 7.2 Ratish1 的三配置 Benchmark 深入分析

| 配置 | Health Ready | Graph Capture | 吞吐（TTS） | 吞吐（Voice Clone） |
|---|---|---|---|---|
| CUDA Graph only | 33.3s | 3.3s | 88.1 tok/s | 87.7 tok/s |
| Partial compile（fast head only） | 54.4s | 16.4s | 120.6 tok/s | 118.7 tok/s |
| Full-model compile | 137.0s | 107.0s | 125.7 tok/s | 122.5 tok/s |

**结合第五步的理论框架解读这些数据**：

1. **Partial compile 的 36% 吞吐提升从何而来？** 回顾 5.1 的开销层分析：CUDA Graph 已消除开销①②③，但 codebook loop 的 9 步循环中每步包含 embedding lookup → linear projection → multi-head attention → RMSNorm → output projection，这些小算子之间的中间 tensor 仍然需要经过显存读写（开销④）。torch.compile 的 inductor 将这些算子融合为更少的 Triton kernel，减少了 GPU-side 的显存 round-trip。**即使 launch overhead 为零，带宽优化仍有 36% 的收益空间**。
2. **Full compile vs Partial compile 仅 4% 差异**：transformer 部分的大 GEMM（`mm(8×2560, 2560×6144)` 等）已被 cuBLAS 高度优化。autotune 日志证实 cuBLAS 在大多数 shape 下击败 Triton kernel（开销⑤已经接近最优）。torch.compile 在 transformer 上唯一的收益是融合 layernorm + residual 等小算子链，但这部分占比很小。
3. **103.7s 的额外启动时间**：`max-autotune-no-cudagraphs` mode 对每个 GEMM shape × 每个 bs 做 Triton autotune，总量 = 12 bs × 36 layers × ~4 linear layers × 18 candidates ≈ 31,000+ benchmark runs。这是 autotune 的固有成本，与 CUDA Graph 无关。
4. **Partial compile 的启动时间仅 +21s**：只编译 fast head（codebook decoder 的少量小算子），autotune 搜索空间远小于 full model，`54.4s - 33.3s = 21.1s` 是可接受的。

#### 7.3 为什么最终选择不 compile？——三层决策逻辑

**zhaochenyang20 的判断**（来自 PR review）：

1. **抽象层级错配**："Hard-coding mega cache into a single model file isn't the right abstraction... should live at the framework level"——torch.compile 的优化应该是 SGLang-Omni 框架级别的能力，而不是单个模型的 hack。结合 5.2 的 mode 分析：选择 `max-autotune-no-cudagraphs` 本身就是一个框架级决策（让 SGLang 管 graph，让 inductor 管 kernel），不应该在模型代码中硬编码
2. **交互复杂性**：结合 5.4 和 5.5 的分析——torch.compile 的 guards/recompilation 与 CUDA Graph 的交互需要非常小心。codebook loop 的 `for cb_idx in range(1, self._num_codebooks)` 需要被 `fullgraph=True` 完全展开，任何 graph break 都会导致编译失败或性能退化。而 slow head 的 `RadixAttention` 更是 graph break 的重灾区
3. **粒度问题**："Compiling the entire model forward is wasteful"——结合 Ratish1 的数据，真正受益的只有 fast head 的小算子融合（36% 增益），slow head 的 4% 增量不值得 103s 的额外启动时间

**sdli1995 的 mega cache 建议**：使用 `torch.compiler.load_cache_artifacts()` / `save_cache_artifacts()` 可以缓存 inductor 的编译结果（FX graph + Triton kernel binary），让后续启动跳过 autotune。实测可以将 compile 时间降低到 2s（LLM）+ 10s（dual AR loop）。但这被推迟到 [Issue #172](https://github.com/sgl-project/sglang-omni/issues/172)。

### 第八步：Issue #172——Framework-Level torch.compile 的工程蓝图

- **深度层级**：修改扩展（SGLang-Omni 是自己开发的系统）
- **目标**：理解 PR #153 中被 defer 的 torch.compile 优化如何在框架层面被系统性地实现，这是 PR #153 设计决策的"下半场"
- **方法**：Issue 分析 + 架构设计评审

#### 8.1 Issue #172 的核心问题

[Issue #172](https://github.com/sgl-project/sglang-omni/issues/172) 的标题是 "Framework-level fine-grained `torch.compile` + Mega Cache for Omni models"。它要解决的三个障碍：

1. **启动开销**：Full-model compile 需要 2~5 分钟，生产环境不可接受
2. **抽象缺失**：目前需要在每个模型的 `factory.py` 中硬编码 compile 逻辑
3. **Graph 冲突**：框架已经管理了 CUDA Graph capture，torch.compile 必须与之干净共存

这三个障碍正好对应第五步中分析的三个技术约束：5.2 的 mode 选择、5.4 的 fullgraph 要求、5.5 的 guard 机制。

#### 8.2 三阶段实施计划

**Phase 1：Partial Compile（仅辅助模块）**

- **目标**：编译 auxiliary modules（如 S2-Pro 的 codebook decoder），backbone 保持 eager + CUDA Graph
- **框架 API 协议**：模型实现 `get_compile_targets()` 方法，返回 `dict[str, Callable]`

```python
# 模型侧：声明"什么是可编译的"
class S2ProSGLangTextModel(nn.Module):
    def get_compile_targets(self) -> dict[str, Callable]:
        if not self._vq_ready:
            return {}
        return {"decode_codebooks": self._decode_codebooks_impl}
```

```python
# 框架侧：决定"如何编译"（新文件 engines/omni/compile.py）
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

**关键设计决策分析**：
- **`fullgraph=True` 是强制要求**——结合 5.4 的分析，graph break 在 CUDA Graph 环境下会导致不可预测的行为
- **compile mode 固定为 `max-autotune-no-cudagraphs`**——结合 5.2 的分析，graph 管理权必须留给 SGLang
- **模型只声明 target，不调用 `torch.compile`**——实现了"模型不知道自己被编译了"的抽象
- **compile 在 `setup_vq_decode()` 之后、`init_device_graphs()` 之前**——复用 PR #153 的 deferred capture 时序

**预期效果**：S2-Pro ~121 tok/s，启动 ~54s（对比 CUDA Graph only 的 88 tok/s / 33s）

**Phase 2：Global Compile（完整 model forward）**

- **目标**：编译整个 `model.forward()`，获取剩余的 4% 吞吐增量
- **前置条件**：SGLang 的 `RadixAttention`、`ForwardBatch`、`MergedColumnParallelLinear`、RoPE cache 模式必须全部 compile-clean（无 graph break）

**两种共存策略需要 benchmark**：

| 策略 | 实现方式 | 优点 | 缺点 |
|---|---|---|---|
| **Layered**（分层管理） | `mode="max-autotune-no-cudagraphs"` + SGLang 的 CUDA Graph capture | SGLang 保持对 graph 的完全控制 | 需要确保 inductor kernel 对 SGLang graph capture 完全透明 |
| **Unified**（统一管理） | `mode="reduce-overhead"`，让 inductor 管理 CUDA Graph | 更深度的优化（inductor 可以做 cross-kernel memory planning） | 失去 SGLang 的 multi-bs graph 管理、memory pool 控制等精细能力 |

**关键挑战**：
- `RadixAttention` 涉及 paged KV cache 的动态索引——这是最可能产生 graph break 的地方
- `ForwardBatch` 的动态属性（`forward_mode`、`extend_seq_lens` 等）在不同 prefill/decode 模式下变化
- S2-Pro 的 `_vq_ready` 条件分支在 capture 时被固化，但 torch.compile 的 trace 需要处理两条路径

**配置接口**：
```bash
--compile-level none     # CUDA graph only（默认，零启动开销）
--compile-level partial  # Phase 1：仅 auxiliary modules
--compile-level full     # Phase 2：完整 model forward
```

**Phase 3：Mega Cache（消除启动开销）**

- **目标**：缓存 inductor 的编译产物（FX graph + Triton kernel binary + autotune 结果），让第二次启动跳过所有 compile 开销
- **Cache Key 设计**：

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

- **两种实现方案**：
  1. **Inductor-native**：设置 `TORCHINDUCTOR_CACHE_DIR`，利用 inductor 内置的 FX graph + kernel 缓存
  2. **`torch.export` + `torch._inductor.aot_compile`**：预编译为 `.so` 动态库（更多控制，更复杂）

- **Invalidation 策略**：cache key 任一变化时重新编译；提供 `--clear-compile-cache` CLI 命令手动重置
- **预期效果**：warm cache 下，即使 `compile_level=full`，启动时间也接近 baseline 的 ~33s

#### 8.3 Issue #172 的五条设计原则——与 PR #153 决策的呼应

| 设计原则 | 含义 | 对应 PR #153 中的教训 |
|---|---|---|
| 模型文件中不出现 compile 调用 | 模型声明 target，框架决定编译策略 | PR #153 中 `factory.py` 硬编码 `enable_torch_compile=True` 导致了不可维护的 hack |
| Compile target 必须是 tensor-in tensor-out | 被编译的函数不能有外部状态访问 | `_decode_codebooks` 访问 `self._audio_decoder`——需要重构为纯函数 `_decode_codebooks_impl` |
| `fullgraph=True` 强制 | 不允许 graph break | codebook loop 的 `for` 循环、`forward_kvcached` 调用必须全部可 trace |
| Eager-first 可读性 | compile 是可选加速，不是默认行为 | PR #153 最终选择 CUDA Graph only 作为默认，compile 作为 opt-in |
| 配置驱动 | 通过 `ServerArgs` 开关控制，不需要改代码 | PR #153 通过 `server_args.disable_cuda_graph` 控制 graph 行为，compile 应该同样如此 |

#### 8.4 适用范围：不只是 S2-Pro

Issue #172 的框架设计是 model-agnostic 的：

| 模型 | Backbone | Auxiliary Module | Compile 机会 |
|---|---|---|---|
| **S2-Pro** | Qwen3 + RadixAttention | Codebook decoder | +37%（aux）；+43%（full） |
| **Qwen3-Omni** | Qwen3 thinker | Talker、encoders | 待 benchmark |
| **未来模型** | 任何 SGLang-backed LLM | 模型特定的 decoder | 自动获得 compile 支持 |

这意味着学习这个设计不仅对 S2-Pro 有价值，对理解 **推理框架如何系统性地管理 torch.compile** 有普遍意义。

#### 8.5 需要深入理解的开放问题

1. **Phase 2 的 RadixAttention compile 可行性**：paged KV cache 的动态页表索引是否能被 `fullgraph=True` 处理？还是需要 SGLang 侧做改造？
2. **Layered vs Unified 策略的 benchmark**：在 S2-Pro 上实测两种策略的吞吐差异和 memory footprint 差异
3. **Mega cache 的 invalidation 精确性**：`torch.__version__` 粒度太粗（minor version 可能不影响 codegen），GPU arch 粒度太细（同 arch 不同 GPU 可能性能最优 kernel 不同）
4. **compile target 的纯函数化改造**：`_decode_codebooks` 当前访问 `self._audio_decoder`、`self._semantic_bias` 等外部状态——如何重构为满足"tensor-in tensor-out"约束？

### 第九步：PyTorch CUDA Graph 封装层与 Memory Pool 机制

- **深度层级**：理解复现
- **目标**：理解 PyTorch 如何将 CUDA runtime 的 graph API 封装为 Python 友好的接口，以及 memory pool 如何支撑多 graph 共存
- **方法**：代码分析

#### 9.1 核心 API 映射

| PyTorch API | CUDA Runtime API | S2-Pro 中的使用场景 |
|---|---|---|
| `torch.cuda.CUDAGraph()` | `cudaGraph_t` + `cudaGraphExec_t` | `CudaGraphRunner` 为每个 bs 维护一个实例 |
| `graph.capture_begin()` | `cudaStreamBeginCapture()` | capture 开始前需要确保所有 buffer 已分配 |
| `graph.capture_end()` | `cudaStreamEndCapture()` | capture 结束后得到完整的 DAG |
| `graph.replay()` | `cudaGraphLaunch()` | 每次 decode step 的实际执行 |
| `torch.cuda.graph()` 上下文管理器 | 封装 begin + end | SGLang 不直接用这个，而是更底层的控制 |

#### 9.2 Memory Pool 与 graph 共享

- `torch.cuda.graph(pool=...)` 允许多个 graph 共享同一个 memory pool，这是 SGLang 管理 12 个 bs graph 的基础
- **不共享 pool 的代价**：每个 graph 独立分配中间 tensor 内存，12 个 graph 各持有一份 → 12 倍的中间 tensor 显存
- **共享 pool 的前提**：同一时间只有一个 graph 在 replay（SGLang 的 decode 阶段满足此条件）
- 对 S2-Pro 的意义：12 个 bs 的 graph 共享 audio decoder 的 KV cache 中间结果内存，不需要 12 份独立分配
- **CUDA 12.4+ 的改进**：每个 kernel launch 的 device memory 从 64KB 降低，减少了多 graph 的显存 overhead

#### 9.3 torch.compile 与 PyTorch CUDA Graph 封装的交互

当 `mode="max-autotune-no-cudagraphs"` 时：
- inductor 生成的 compiled function 就是一个普通的 Python callable，内部调用优化后的 Triton/cuBLAS kernel
- 这个 callable 可以被 `torch.cuda.graph()` 或 SGLang 的 `CudaGraphRunner` 正常 capture
- inductor **不**在内部做任何 graph capture（没有 CUDAGraph Trees）

当 `mode="reduce-overhead"` 时（SGLang 不使用）：
- inductor 在内部使用 CUDAGraph Trees，自己管理 capture/replay
- 如果外层再包一层 `torch.cuda.graph()`，会产生 nested capture 的未定义行为

### 第十步：设计复盘——统一 Graph 的完整 Trade-off 分析

- **深度层级**：修改扩展
- **目标**：将前面所有知识串联，对 PR #153 和 Issue #172 做完整的设计复盘
- **方法**：概念框架 + 数据分析

#### 10.1 设计决策矩阵

| 决策 | 选择 | Trade-off | 理由 | 未来演进（Issue #172） |
|---|---|---|---|---|
| 统一 vs 分离 graph | 统一 | 单个大 graph vs 两个小 graph + 中间数据传输 | 统一消除 slow→fast 之间的 CPU 调度开销 | 保持统一——compile 不改变 graph 结构 |
| Greedy vs Sampling | Greedy (`torch.argmax`) | 丢失采样多样性 | CUDA Graph 兼容性要求；TTS 场景下 greedy 可接受 | 探索 graph-safe sampling |
| Persistent buffers vs Dynamic tensors | Persistent | 额外显存占用（~几 MB） | CUDA Graph 要求地址稳定 | 不变——compile 不影响 buffer 设计 |
| torch.compile | Off（defer 到 framework） | 放弃 36% 吞吐提升 | 启动时间 trade-off + 抽象层级 + 可维护性 | Phase 1: partial compile 恢复 36% 增益；Phase 3: mega cache 消除启动开销 |
| Deferred capture | 先 init → setup_vq → capture | 增加初始化复杂度 | 确保 graph 包含完整的 decode path | compile warmup 插入 setup_vq 和 capture 之间 |
| Graph 管理权 | SGLang CudaGraphRunner | 放弃 inductor 的 CUDAGraph Trees | 保持精细的 multi-bs graph 控制 | `max-autotune-no-cudagraphs` 不变；不考虑 `reduce-overhead` |

#### 10.2 S2-Pro 的完整优化栈（从 eager 到终态）

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

每一层优化都是**正交且可叠加**的，这要归功于 `max-autotune-no-cudagraphs` 模式实现的"inductor 管 kernel、SGLang 管 graph"的清晰分工。

#### 10.3 未来优化方向

1. **Phase 1 实施**（Issue #172）：`get_compile_targets()` 协议 + `apply_compile_targets()` 框架函数 + S2-Pro 的 `_decode_codebooks_impl` 注册
2. **Mega cache 集成**（Issue #172 Phase 3）：`torch.compiler.save_cache_artifacts()` / `load_cache_artifacts()`，或 `TORCHINDUCTOR_CACHE_DIR` 环境变量
3. **CUDA Graph conditional nodes**（CUDA 12.4+）：可能允许在 graph 内部进行条件执行，对 early stopping（遇到 `im_end_id` 提前结束）有价值——但需要 PyTorch 层面的支持
4. **Graph Update API**：`cudaGraphExecUpdate()` 可能允许在不重新 capture 的情况下修改 graph 中的某些参数，减少 multi-bs capture 的开销
5. **Sampling 恢复**：探索将 stochastic sampling 以 CUDA Graph 兼容的方式实现（pre-allocated random state + graph-safe `multinomial`）
6. **RadixAttention compile-clean 改造**：这是 Phase 2 的核心前置条件，需要 SGLang 上游的配合

### 第十一步：Piecewise CUDA Graph——SGLang 主仓库的另一条路径

- **深度层级**：修改扩展（SGLang 是自己开发的系统）
- **目标**：理解 SGLang 主仓库中 Piecewise CUDA Graph（PCG）的设计动机、实现机制，以及它与 PR #153 的 monolithic graph 方案之间的对比和关联
- **方法**：概念框架 + 源码走读
- **写作位置**：文章末尾，在设计复盘之后、参考之前。作为 CUDA Graph 话题的拓展——从"一个模型的 CUDA Graph 优化"上升到"推理框架级别的 CUDA Graph 架构"

#### 11.1 动机：Monolithic Graph 的局限性

PR #153 中的方案是 **monolithic CUDA Graph**——将整个 `forward()` 作为一个 graph 捕获。这对 S2-Pro 的 decode 阶段（固定 bs、固定 kernel 序列）效果很好，但对 SGLang 主仓库面对的更广泛场景，monolithic graph 有根本性的局限：

1. **不可捕获的操作**：FlashAttention、MoE dispatch（DeepEP 等）等操作本身不能或不适合被 CUDA Graph 捕获——它们需要动态 shape 或有内部的 host-device sync。Monolithic graph 无法绕过这些操作。
2. **Prefill/Extend 的动态 shape**：Prefill 阶段的 token 数量变化范围大（从几个到几千），不可能为每个 token 数都预先 capture 一个 monolithic graph。
3. **显存压力**：Monolithic graph 为每个 batch size 各持有一份完整的 graph 和中间 tensor，显存占用大。

这些局限正是 Piecewise CUDA Graph 要解决的问题。

#### 11.2 核心思路：切分 + 分段捕获

**Piecewise CUDA Graph 的核心思路**：不把整个 forward 作为一个 graph，而是在**不可捕获操作的边界**（如 attention kernel、MoE dispatch）处切分，将 forward 拆成若干个小 subgraph，每个 subgraph 独立 capture。

```
Monolithic Graph（PR #153 方案）：
┌──────────────────────────────────────────────────────┐
│ 整个 forward() 作为一个 graph                         │
│ VQ combine → 36 层 Transformer → logits → codebook   │
└──────────────────────────────────────────────────────┘

Piecewise Graph（SGLang 主仓库方案）：
┌──────────┐  eager  ┌──────────┐  eager  ┌──────────┐
│ Subgraph │→ attn  →│ Subgraph │→ attn  →│ Subgraph │→ ...
│ (FFN等)  │  kernel  │ (FFN等)  │  kernel  │ (FFN等)  │
└──────────┘         └──────────┘         └──────────┘
```

每个 subgraph 覆盖的是"两个不可捕获操作之间"的可捕获部分（如 FFN、layernorm、residual 等）。不可捕获操作（attention、MoE dispatch）仍然以 eager 模式执行。

#### 11.3 Split Points 与三阶段执行

**Split Points 机制**：
- 通过 `@register_split_op` 装饰器声明切分点（如 MoE forward dispatch）
- 编译时自动在这些位置切开 FX graph，产生若干个 subgraph
- 每个 subgraph 独立编译（`torch.compile`）和 capture

**三阶段执行模型**（每个 subgraph）：
1. **Compilation**：`torch.compile` 编译 subgraph，处理动态 shape
2. **CUDA Graph Capture**：为预定义的 token 长度（4, 8, 12, ..., 2048+）capture 每个 subgraph
3. **Steady-State Replay**：运行时找到最近的 captured size，pad 后 replay

**Capture Size Schedule**（默认）：
```
4-32:      步长 4
48-256:    步长 16
288-512:   步长 32
640-1024:  步长 64
1280-4096: 步长 256
4608-max:  步长 512
```

这个 schedule 的设计逻辑：小 token 数（decode 阶段）需要更精细的粒度以减少 padding 浪费；大 token 数（prefill 阶段）对粒度不敏感。

#### 11.4 与 PR #153 Monolithic Graph 的对比

| 维度 | Monolithic Graph（PR #153） | Piecewise CUDA Graph |
|---|---|---|
| **捕获范围** | 整个 forward | 每层/每段独立 |
| **Attention 处理** | 被 graph 包含（RadixAttention） | 在 split point 处以 eager 执行 |
| **适用阶段** | 仅 decode（固定 bs） | Decode + Prefill（多种 token 数） |
| **不可捕获操作** | 必须绕过或替代 | 在切分点处自然支持 |
| **Memory Pool** | 每个 bs 一个 graph 共享一个 pool | 全局 shared pool，所有 subgraph + 所有 capture size 共享 |
| **与 torch.compile 的关系** | 正交（先 compile 再 capture） | **内嵌**（每个 subgraph 先 compile 再 capture） |
| **复杂度** | 简单直接 | 框架层复杂，但对模型开发者透明 |

**关键洞察**：PR #153 的 monolithic graph 能工作，是因为 S2-Pro 的 decode 阶段满足了 CUDA Graph 的所有五条约束——尤其是 RadixAttention 恰好在 decode 阶段可被捕获（固定 bs、固定 seq 位置）。但对 SGLang 主仓库面对的更广泛场景（多种 attention backend、MoE 模型、变长 prefill），piecewise 方案是更通用的解。

#### 11.5 源码走读要点

关键代码文件：
- `sglang/srt/model_executor/piecewise_cuda_graph_runner.py`——主 runner，管理 capture 和 replay
- `sglang/srt/compilation/cuda_piecewise_backend.py`——per-subgraph 的 compile + capture
- `sglang/srt/compilation/backend.py`——graph 切分逻辑
- `sglang/srt/compilation/compilation_config.py`——split points、capture sizes 配置
- `sglang/srt/server_args.py`——`--enable-piecewise-cuda-graph` 等 CLI 参数

需要关注的设计点：
1. **Reverse capture order**：和 `CudaGraphRunner` 一样，从大 token 数到小 token 数 capture，复用 memory pool
2. **Global shared memory pool**：所有 subgraph × 所有 capture size 共享一个 pool
3. **与 torch.compile 的紧密集成**：每个 subgraph 先经过 `torch.compile`（inductor）优化，再 capture 为 CUDA Graph——这正是 PR #153 和 Issue #172 中讨论的"inductor 管 kernel、SGLang 管 graph"分工模式的框架级实现
4. **Eager fallback**：capture 失败或超出 max tokens 时自动 fallback

#### 11.6 对 S2-Pro / SGLang-Omni 的启示

Piecewise CUDA Graph 的思路是否可以应用到 SGLang-Omni？

- **当前 S2-Pro decode 不需要**：decode 阶段的 monolithic graph 已经足够（固定 bs、所有操作可捕获、性能已经很好）
- **但 prefill 阶段可能受益**：S2-Pro 的 prefill 目前走 eager，如果 prefill 成为瓶颈，piecewise graph 可以覆盖 prefill 中 attention 之外的部分
- **更广的意义**：SGLang-Omni 未来接入更复杂的模型（多模态、MoE）时，piecewise 方案可能成为必选项
- **Issue #172 Phase 2 的关联**：Phase 2 要 compile 整个 `model.forward()`，但 RadixAttention 可能 graph break——piecewise 的"在不可捕获操作处切分"思路，正是一种优雅的解法

#### 11.7 相关 Issue 和状态

- [Feature Roadmap for Prefill (Piecewise) CUDA Graph - Issue #11490](https://github.com/sgl-project/sglang/issues/11490)
- [TODO: Piecewise CUDA Graph Default Enable - Issue #18130](https://github.com/sgl-project/sglang/issues/18130)
- [Docs: Add doc for piecewise CUDA graph - Issue #18267](https://github.com/sgl-project/sglang/issues/18267)
- 当前状态：**已默认启用**，可通过 `--disable-piecewise-cuda-graph` 关闭

## 推荐资源

### 官方文档
- [NVIDIA CUDA Programming Guide - CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [CUDA Runtime API - Graph Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html)
- [PyTorch CUDA Graphs](https://pytorch.org/docs/stable/cuda.html#cuda-graphs)
- [PyTorch torch.compile Documentation](https://pytorch.org/docs/stable/torch.compiler.html)
- [PyTorch CUDAGraph Trees](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html)——理解 inductor 内部的 graph 管理机制，以及为什么 SGLang 选择 `no-cudagraphs` 模式

### 代码仓库（需要确定 commit hash）
- SGLang `CudaGraphRunner`（monolithic）：`sglang/srt/model_executor/cuda_graph_runner.py`
- SGLang `PiecewiseCudaGraphRunner`：`sglang/srt/model_executor/piecewise_cuda_graph_runner.py`
- SGLang Piecewise compilation backend：`sglang/srt/compilation/cuda_piecewise_backend.py`、`sglang/srt/compilation/backend.py`
- SGLang-Omni PR #153（merge commit `cd9aaf3`）：https://github.com/sgl-project/sglang-omni/pull/153
  - `sglang_omni/models/fishaudio_s2_pro/sglang_model.py`——统一模型实现
  - `sglang_omni/models/fishaudio_s2_pro/factory.py`——deferred graph capture
  - `sglang_omni/models/fishaudio_s2_pro/runtime/s2pro_sglang_ar.py`——buffer 读写协议
  - `sglang_omni/models/fishaudio_s2_pro/fish_speech/models/text2semantic/modeling.py`——audio decoder（`FishQwen3AudioDecoder`）
- SGLang-Omni Issue #172：https://github.com/sgl-project/sglang-omni/issues/172 ——Framework-level torch.compile 的三阶段工程蓝图
- torch-memory-saver：https://github.com/fzyzcjy/torch_memory_saver
- PyTorch CUDA Graph wrapper：`torch/cuda/graphs.py`
- PyTorch Inductor CUDAGraph Trees：`torch/_inductor/cudagraph_trees.py`——理解 inductor 如何在 `reduce-overhead` mode 下内部管理 graph

### 社区文章
- [Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) — PyTorch 官方博客
- [CUDA Graphs in the Deep Learning Ecosystem](https://developer.nvidia.com/blog/cuda-graphs/) — NVIDIA 开发者博客
- [torch.compile, the missing manual](https://docs.google.com/document/d/1y811KBmTLBEaYMCFBfbgLzrdSN0VRz51CkLMpTk7vAU) — PyTorch 团队的 torch.compile 内部文档

## 文章结构建议

- **文章类型**：code-walkthrough + sys-design 混合（以 PR #153 为叙事主线，CUDA Graph 工程实现为核心，torch.compile 讨论为概要引出）
- **建议路径**：`torch/cuda-graph/readme-3.md`
- **建议标题**：`再探 CUDA Graph：TTS 模型中的双 CUDA Graph 优化`
- **系列归属**：CUDA Graph 系列，第三篇（前两篇分别是浅析和 S2-Pro 对比分析）
- **开篇结构**（参照用户实际写作风格）：
  1. 先回顾系列第一篇，承认当时理解浅薄，引出本次 PR 的工程实践
  2. **紧接着亮出 benchmark 数据表格**（三配置对比 + TPS 提升数字），用成果抓读者
  3. 简短路线图（4 条编号列表，不超过一句话/条）
  4. 随意风格致谢
  5. **不要写模板化声明**（如"本文基于 commit xxx 进行分析"）
- **章节顺序的核心原则：概念 → 模型 → 代码**：
  - 必须先建立 CUDA Graph 的概念框架（capture/replay 三阶段、指针稳定性、静态控制流约束），让读者拿到"分析工具"
  - 再介绍 S2-Pro 模型特点，用概念框架解释"这个模型为什么对 CUDA Graph 有挑战"
  - 最后才进入代码实现，展示概念如何在工程中落地
  - **绝对不能反过来**——没有概念基础，读者无法理解代码中的设计选择
- **预计章节**：
  1. **CUDA Graph 概念框架**：capture → instantiate → replay 三阶段机制、指针稳定性约束（graph 录制的是 GPU 虚拟地址）、静态控制流要求、不能有动态内存分配、不能有 host-device sync——建立读者的"分析工具箱"
  2. **S2-Pro 模型：为什么它对 CUDA Graph 有挑战**：Dual-AR 双头架构（slow head + fast head）、9 步 codebook loop 的小 kernel launch 瓶颈、两种 KV cache 共存——用第 1 章的概念框架解释每个特点带来的 CUDA Graph 约束
  3. **Deferred Graph Capture**：`factory.py` 的初始化时序——为什么必须先 setup 再 capture（映射到第 1 章的"graph 是静态的"概念）
  4. **Persistent Buffer 设计**：pre-allocate + `copy_()` 的指针稳定性（映射到第 1 章的地址稳定性约束）、buffer 读写协议
  5. **SGLang CudaGraphRunner 源码走读**：多 bs graph 管理、memory pool 共享、eager fallback 条件
  6. **CUDA Graph 与 torch.compile 的关系**：五层开销模型、四种 compile mode、`no-cudagraphs` 的分工哲学（概要，详细分析留后续文章）
  7. **torch.compile 在 S2-Pro 中的兴衰**：Ratish1 benchmark、三层决策逻辑
  8. **Issue #172 概要**：三阶段计划引出（详细分析留后续文章）
  9. **Piecewise CUDA Graph**：SGLang 主仓库的另一条路径——monolithic vs piecewise 的设计对比、split points 机制、三阶段执行模型、与 S2-Pro 方案的关系和启示
  10. **设计复盘**：S2-Pro 完整优化栈、未来方向（包含 piecewise 的展望）

## 草稿完成度分析

草稿路径：`torch/cuda-graph/readme-2.md`

### 已完成部分
- S2-Pro 的双头架构描述（slow head + fast head 的基本概念）
- CUDA Graph capture 从秒级变分钟级的问题分析与根因定位
- CUDA Graph vs torch.compile 的开销对比表格（消除 launch overhead vs 算子融合）
- 对 Transformer（slow head）和 Codebook Loop（fast head）的效果分析
- 为什么 SGLang LLM 通常同时开两者的讨论
- 结论：关闭 torch.compile、CUDA Graph 为主要优化手段
- Issue：对 codebook loop 单独启用 torch.compile + CUDA Graph 的叠加优化探索方案

### 完成度
- **相对于草稿自身的主题（CUDA Graph vs torch.compile 对比分析）**：~90%
- **相对于本学习计划的完整目标**：~15%

### 待补充部分（按优先级）
1. **【高优先级】CUDA Graph 与 torch.compile 的深层关系**（第五步）：GPU 执行流水线的五层开销模型、四种 compile mode 的 graph 管理策略差异、`max-autotune-no-cudagraphs` 的"分工"哲学、CUDAGraph Trees 机制与 SGLang CudaGraphRunner 的冲突分析、`fullgraph=True` 约束在 S2-Pro 中的具体挑战——草稿的对比停留在"消除不同开销"的高层描述，缺乏对 PyTorch 内部 graph 管理机制的深入分析
2. **【高优先级】Issue #172 的框架级工程蓝图**（第七步）：三阶段实施计划、`get_compile_targets()` 协议设计、Phase 1/2/3 的技术细节与挑战、五条设计原则与 PR #153 教训的映射、Layered vs Unified 共存策略的 benchmark 计划——草稿完全没有覆盖
3. **【高优先级】PR #153 的完整代码实现分析**（第一、二步）：`setup_vq_decode()`、persistent buffers、deferred capture、`_decode_codebooks()`、`_update_vq_buffers()/_build_outputs()` 的 buffer 读写协议
4. **【高优先级】S2-Pro 模型架构的源码级深入分析**（第一步）：`FishQwen3AudioDecoder` 的 `forward_kvcached()`、static KV cache 设计、`input_pos.fill_()` 的 graph 安全模式、两种 KV cache 的共存
5. **【中优先级】SGLang CudaGraphRunner 源码走读**（第四步）：多 bs graph 管理机制、capture 顺序与内存策略、与 S2-Pro persistent buffers 的交互
6. **【中优先级】PR 迭代故事**（第六步）：torch.compile 的加入→问题→移除过程、Ratish1 的 benchmark 数据深度解读（结合五层开销模型）、sdli1995 的 mega cache 建议
7. **【中优先级】PyTorch CUDA Graph 封装层**（第八步）：memory pool 共享机制、torch.compile `no-cudagraphs` 模式下 inductor kernel 对外部 graph capture 的透明性
