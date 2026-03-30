# SGLang Omni Code Walkthrough

本文档从开发者视角对 SGLang Omni（全模态推理框架）进行代码走读，追踪一个多模态请求（文本 + 图片 + 音频）从提交到返回文本与语音结果的全过程。SGLang Omni 以 **多阶段异步流水线** 为核心架构，支持多模态输入（文本、图片、视频、音频）和多模态输出（文本、语音）。

## 目录

- [架构总览](#架构总览)
- [Pipeline 整体架构](#pipeline-整体架构)
  - [Coordinator](#coordinator)
  - [Control Plane](#control-plane)
  - [Stage](#stage)
  - [Worker](#worker)
  - [Executor](#executor)
- [请求处理全流程](#请求处理全流程)
  - [Stage 1: Preprocessing（预处理）](#stage-1-preprocessing预处理)
  - [Stage 2-3: Image Encoder & Audio Encoder（编码器）](#stage-2-3-image-encoder--audio-encoder编码器)
  - [Stage 4: Aggregate（聚合）](#stage-4-aggregate聚合)
  - [Stage 5: Thinker（主模型推理）](#stage-5-thinker主模型推理)
  - [Stage 6: Decode（解码输出）](#stage-6-decode解码输出)
  - [Stage 7-9: Speech Pipeline（语音生成流水线）](#stage-7-9-speech-pipeline语音生成流水线)
- [OmniEngine: 调度与执行引擎](#omniengine-调度与执行引擎)
- [核心数据结构](#核心数据结构)
- [关键设计模式](#关键设计模式)

---

## 架构总览

SGLang Omni 的核心思想是将一个多模态请求拆分为多个处理阶段（Stage），每个阶段作为独立进程运行，通过 ZMQ 控制面和共享内存数据面进行通信。请求在各阶段之间流转，状态（`PipelineState`）不断被丰富和转换。

```
                          ┌─────────────────────────────────────────────────────────────┐
                          │                      Coordinator                            │
                          │   (请求路由 / 完成聚合 / Abort 广播)                          │
                          └──────┬──────────────────────────────────────────────┬────────┘
                                 │ SubmitMessage                                │ CompleteMessage
                                 ▼                                              ▲
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ preprocessing │──▶│ image_encoder│──▶│  aggregate   │──▶│   thinker    │──▶│    decode     │
│    (CPU)      │  ││ audio_encoder│  │    (CPU)      │   │   (GPU:0)    │  ││   (CPU)       │
└──────────────┘   └──────────────┘   └──────────────┘   └──────┬───────┘   └──────────────┘
                        (GPU)              ▲ 等待全部               │               (Terminal)
                                           │ 编码器完成             │ stream
                                           │                       ▼
                                                          ┌──────────────┐
                                                          │  talker_ar   │
                                                          │   (GPU:1)    │
                                                          └──────┬───────┘
                                                                 │ stream
                                                                 ▼
                                                          ┌──────────────┐
                                                          │code_predictor│◀─┐
                                                          │   (GPU:1)    │  │ feedback
                                                          └──────┬───────┘──┘
                                                                 │ stream
                                                                 ▼
                                                          ┌──────────────┐
                                                          │   code2wav   │
                                                          │   (GPU:1)    │
                                                          └──────────────┘
                                                              (Terminal)
```

以 Qwen3-Omni（含语音输出）为例，完整流水线包含 **9 个 Stage**：

| Stage | 位置 | 设备 | 作用 |
|-------|------|------|------|
| preprocessing | 入口 | CPU | 文本 tokenize、多媒体解析 |
| image_encoder | 编码 | GPU | Vision Transformer 编码图片/视频 |
| audio_encoder | 编码 | GPU | 音频 Mel 频谱编码 |
| aggregate | 聚合 | CPU | 合并文本 tokens 与编码器输出 |
| thinker | 推理 | GPU:0 | MoE Transformer 主模型，生成文本 token |
| decode | 输出 | CPU | 文本后处理（Terminal） |
| talker_ar | 语音 | GPU:1 | 语音 codec token 自回归生成 |
| code_predictor | 语音 | GPU:1 | RVQ 多层码预测 |
| code2wav | 语音 | GPU:1 | Vocoder 合成音频波形（Terminal） |

---

## Pipeline 整体架构

### Coordinator

[`Coordinator`](https://github.com/sgl-project/sglang/blob/main/python/sglang_omni/pipeline/coordinator.py) 是整个流水线的入口和出口，负责：

1. **请求提交**：接收用户请求，封装为 `StagePayload`，通过 `CoordinatorControlPlane.submit_to_stage()` 发送 `SubmitMessage` 到入口 Stage（preprocessing）。
2. **完成聚合**：流水线可能有多个 Terminal Stage（如 `decode` 和 `code2wav`），Coordinator 等待所有 Terminal Stage 完成后，合并 `partial_results` 并返回给调用方。
3. **Abort 广播**：通过 PUB/SUB 模式向所有 Stage 广播 `AbortMessage`。
4. **流式输出**：通过 `stream()` 方法逐步 yield 中间结果。

关键方法：

- `submit(request_id, request)` — 提交请求并等待完成
- `stream(request_id, request)` — 提交请求并流式返回
- `run_completion_loop()` — 后台协程，持续接收 `CompleteMessage` / `StreamMessage`
- `abort(request_id)` — 广播取消信号

### Control Plane

[`ControlPlane`](https://github.com/sgl-project/sglang/blob/main/python/sglang_omni/pipeline/control_plane.py) 基于 ZMQ 实现进程间通信，定义了以下消息类型：

| 消息类型 | 方向 | 用途 |
|---------|------|------|
| `SubmitMessage` | Coordinator → 入口 Stage | 初始请求提交 |
| `DataReadyMessage` | Stage → Stage | 数据就绪通知（含共享内存元信息） |
| `CompleteMessage` | Terminal Stage → Coordinator | 请求完成 |
| `StreamMessage` | Stage → Coordinator | 流式中间结果 |
| `AbortMessage` | Coordinator → 所有 Stage (PUB/SUB) | 请求取消 |
| `ShutdownMessage` | Coordinator → Stage | 关闭信号 |

Control Plane 分为两个实现：
- **`CoordinatorControlPlane`**：Coordinator 端，管理到各 Stage 的 PUSH socket 和接收 completion 的 PULL socket。
- **`StageControlPlane`**：Stage 端，提供 `recv()` 阻塞接收和 `send_to_stage()` / `send_complete()` 路由功能。

### Stage

[`Stage`](https://github.com/sgl-project/sglang/blob/main/python/sglang_omni/pipeline/stage/runtime.py) 代表流水线中的一个处理节点，每个 Stage 运行在独立进程中。

Stage 的核心职责：
1. **消息路由**：接收 `SubmitMessage` 或 `DataReadyMessage`，分派给内部 Worker。
2. **输入聚合**：部分 Stage（如 `aggregate`）需要等待多个上游 Stage 的数据全部到达后才能开始处理，使用 `AggregatedInputHandler` 实现。
3. **Abort 监听**：后台协程持续监听 `AbortMessage`，收到后取消正在处理的请求。
4. **流式块路由**：通过 `StreamQueue` 将上游的流式数据块转发给对应 Worker。

核心执行循环 `Stage.run()`:
```
while not shutdown:
    msg = control_plane.recv()           # 阻塞等待消息
    if SubmitMessage:
        input_handler.receive(msg)       # 记录输入
        router.enqueue(work)             # 分发给 Worker
    elif DataReadyMessage:
        input_handler.receive(msg)       # 聚合输入
        if all_inputs_ready:
            router.enqueue(work)         # 全部就绪后分发
    elif StreamChunk:
        stream_queue.put(request_id, chunk)  # 放入流式队列
```

### Worker

[`Worker`](https://github.com/sgl-project/sglang/blob/main/python/sglang_omni/pipeline/worker/runtime.py) 是 Stage 内部的实际处理单元，负责调用 Executor 执行计算。

Worker 处理单个请求的流程：

```python
async def _process_request(work):
    # 1. 从共享内存/Relay 加载输入数据
    payloads = await data_plane.read_payload(work.shm_metadata)
    merged = _merge_payloads(work, payloads)

    # 2. 提交给 Executor 执行
    executor.add_request(merged)

    # 3. 等待结果
    result = await executor.get_result()

    # 4. 路由到下一个 Stage 或完成
    next_stages = get_next(request_id, result)
    if next_stages:
        for stage in next_stages:
            _send_to_next(request_id, stage, result)   # DataReadyMessage
    else:
        _send_complete(request_id, result)              # CompleteMessage
```

对于流式 Stage，Worker 额外运行 `_stream_send_loop()` 后台任务，将 Executor 产出的流式块通过 `DataPlaneAdapter` 写入共享内存，并发送 `DataReadyMessage` 通知下游。

**同 GPU 零拷贝优化**：当上下游 Stage 在同一 GPU 上时（如 `talker_ar` → `code_predictor`），使用 CUDA IPC（`ForkingPickler`）实现 tensor 零拷贝传输，避免通过共享内存中转。

### Executor

[`Executor`](https://github.com/sgl-project/sglang/blob/main/python/sglang_omni/executors/interface.py) 是抽象执行接口，定义了统一的请求处理 API：

```python
class Executor(ABC):
    async def add_request(payload: StagePayload) -> None    # 提交请求
    async def get_result() -> StagePayload                   # 获取结果
    async def abort(request_id: str) -> None                 # 取消请求
    def set_stream_fn(fn) -> None                            # 设置流式回调
```

框架提供了以下 Executor 实现：

| Executor | 用途 | 特点 |
|----------|------|------|
| `PreprocessingExecutor` | CPU 预处理（tokenize、媒体加载） | 支持同步/异步处理函数 |
| `DirectModelExecutor` | 直接运行 torch 模型（无调度） | 支持 batch 和 streaming 两种模式 |
| `EngineExecutor` | 包装 OmniEngine（带调度器） | 用于 Thinker、Talker AR 等需要连续解码的场景 |
| `FusedExecutor` | 串联多个 Executor | 中间结果在同一 Worker 内传递，减少 IPC 开销 |

---

## 请求处理全流程

以一个包含图片和音频的多模态请求为例，追踪其完整生命周期。

### Stage 1: Preprocessing（预处理）

**Executor**: `PreprocessingExecutor`
**设备**: CPU
**核心类**: [`Qwen3OmniPreprocessor`](https://github.com/sgl-project/sglang/blob/main/python/sglang_omni/models/qwen3_omni/components/preprocessor.py)

预处理阶段将原始用户输入转化为模型可消费的格式：

1. **多媒体加载**（异步并行）：
   - **图片**: `ImageMediaIO` 支持 PIL、URL、Base64、文件路径，统一转为 RGB PIL.Image
   - **音频**: `AudioMediaIO` 支持 WAV（PCM/IEEE-float）、WebM/Opus、MP3、OGG、FLAC，统一转为 16kHz 单声道 float32 numpy 数组。使用轻量级 `_resample_linear()` 替代 librosa 进行推理时重采样
   - **视频**: `VideoMediaIO` 支持 TorchCodec/TorchVision 后端帧提取，可选同时提取视频中的音频（`use_audio_in_video` 标志）

2. **缓存键计算**：在加载媒体前基于文件路径计算 content-addressable 缓存键（包含 FPS、采样率等参数），实现编码器输出的请求级缓存。

3. **消息构建 & Tokenize**：
   - 将多模态内容注入 HF 格式的消息中，插入占位符 token（`<|image_pad|>`、`<|audio_pad|>`、`<|video_pad|>`）
   - 应用 chat template 后调用 HF Processor 进行 tokenize

4. **输出 `PipelineState`**：
   ```python
   {
       "prompt": {"input_ids": [...], "attention_mask": [...], "prompt_text": "..."},
       "mm_inputs": {"image": [...], "audio": [...], "video": [...]},
       "encoder_inputs": {
           "image_encoder": {"pixel_values": tensor, "image_grid_thw": tensor},
           "audio_encoder": {"input_features": tensor, "audio_feature_lengths": tensor}
       }
   }
   ```

预处理完成后，数据同时路由到 `image_encoder` 和 `audio_encoder` 两个 Stage（并行执行）。

### Stage 2-3: Image Encoder & Audio Encoder（编码器）

**设备**: GPU

#### Image Encoder

**核心类**: [`Qwen3OmniImageEncoder`](https://github.com/sgl-project/sglang/blob/main/python/sglang_omni/models/qwen3_omni/components/image_encoder.py)

- 基于 27 层 Vision Transformer（ViT），patch_size=16，spatial_merge_size=2
- 输入 `pixel_values [B, C, H, W]`，输出 `image_embeds [n_tokens, 3584]`
- **多尺度特征**：通过 `deepstack_visual_indexes=[8, 16, 24]` 从中间层提取特征（deepstack_visual_embeds），供后续 Thinker 使用
- **优化**：`_optimize_patch_embed` 将 Conv3d 重写为 Linear，获得 7-15× 的推理加速

#### Audio Encoder

**核心类**: [`Qwen3OmniAudioEncoder`](https://github.com/sgl-project/sglang/blob/main/python/sglang_omni/models/qwen3_omni/components/audio_encoder.py)

- 基于 32 层 Transformer 编码器，输入 128 维 Mel 频谱特征
- 输入 `input_features [B, n_mels, T]`，输出 `audio_embeds [n_tokens, 3584]`
- 支持 500-token chunk 流式处理

两个编码器独立运行，结果分别通过 `DataReadyMessage` 发送到 `aggregate` Stage。

### Stage 4: Aggregate（聚合）

**设备**: CPU
**输入聚合**: `AggregatedInputHandler`

Aggregate Stage 等待三路输入全部到达：
1. Preprocessing 的原始 `PipelineState`（含 `input_ids`、`attention_mask`）
2. Image Encoder 的输出（`image_embeds`、`deepstack_visual_embeds`）
3. Audio Encoder 的输出（`audio_embeds`、`audio_output_lengths`）

通过 `merge_for_thinker()` 函数将三路数据合并，构建 `thinker_inputs`：

```python
thinker_inputs = {
    "input_ids": [...],
    "attention_mask": [...],
    "image_embeds": tensor,         # [n_img_tokens, 3584]
    "image_grid_thw": tensor,
    "video_embeds": tensor,         # [n_video_tokens, 3584]
    "video_grid_thw": tensor,
    "audio_embeds": tensor,         # [n_audio_tokens, 3584]
    "deepstack_visual_embeds": [...],  # 中间层特征
}
```

### Stage 5: Thinker（主模型推理）

**Executor**: `EngineExecutor`（包装 `OmniEngine`）
**设备**: GPU:0
**核心模型**: [`Qwen3OmniMoeThinkerTextModel`](https://github.com/sgl-project/sglang/blob/main/python/sglang_omni/models/qwen3_omni/thinker.py)

Thinker 是整个系统的核心——一个 28 层 MoE Transformer，负责理解多模态输入并生成文本。

#### 模型结构

```
Qwen3OmniMoeThinkerTextModel
├── embed_tokens: VocabParallelEmbedding (vocab=3584 → hidden=2048)
├── layers[0..27]: Qwen3OmniMoeThinkerTextDecoderLayer
│   ├── input_layernorm: RMSNorm
│   ├── self_attn: Qwen3OmniMoeThinkerTextAttention
│   │   ├── qkv_proj: Fused QKV projection
│   │   ├── o_proj: Output projection
│   │   ├── q_norm, k_norm: RMSNorm (per-head)
│   │   ├── rotary_emb: RoPE (YARN scaling)
│   │   └── attn: RadixAttention (with KV cache)
│   ├── post_attention_layernorm: RMSNorm
│   └── mlp: Qwen3OmniMoeThinkerTextSparseMoeBlock
│       ├── gate: ReplicatedLinear (router)
│       └── experts: FusedMoE (128 experts, top-8 routing, intermediate=768)
└── norm: RMSNorm
```

**MoE 配置**：每个 token 从 128 个 expert 中选择 8 个（top-k routing），经 renormalize 后加权求和。

#### Forward 流程

1. **Token 嵌入**: `input_ids` → `embed_tokens` → `[seq_len, 2048]`
2. **多模态融合**: 在 placeholder token 位置通过 `masked_scatter` 注入编码器输出
   - 视觉 placeholder → `image_embeds` / `video_embeds`
   - 音频 placeholder → `audio_embeds`
3. **28 层 Transformer**：RMSNorm → GQA Attention (28 heads, 4 kv heads) → RMSNorm → MoE → Residual
4. **输出**: hidden states → `lm_head` → logits → sampling → `output_ids`

**优化**：
- `fused_qk_norm_rope` kernel：将 QK Norm 和 RoPE 融合为单个 bfloat16 kernel（约 3× 加速）
- YARN RoPE scaling：将上下文从 8K 扩展到 32K
- RadixAttention：高效 KV cache 管理

#### 输出分流

Thinker 输出后进行 fan-out：
- **text 分支** → `decode` Stage（文本后处理）
- **speech 分支** → `talker_ar` Stage（语音生成），同时传递：
  - `thinker_embeds`: token embeddings
  - `thinker_hidden[layer_24]`: 第 24 层的 hidden states（供 Talker 做跨模态对齐）

### Stage 6: Decode（解码输出）

**设备**: CPU（Terminal Stage）

将 Thinker 的 `output_ids` 解码为文本，构建最终的文本响应。支持流式文本输出。

### Stage 7-9: Speech Pipeline（语音生成流水线）

当启用语音输出时，Thinker 的输出额外流入三级语音流水线，全部部署在 GPU:1 上。

#### Stage 7: Talker AR

**Executor**: `EngineExecutor`（包装 `OmniEngine`）
**核心类**: [`Qwen3OmniMoeTalkerTextModel`](https://github.com/sgl-project/sglang/blob/main/python/sglang_omni/models/qwen3_omni/talker.py)

Talker 是一个 20 层 MoE Transformer（128 experts, top-6 routing），与 Thinker 的关键区别是额外包含 **Shared Expert**（每层一个固定的 dense MLP，与 routed experts 并行执行后门控组合）。

**Prefill 输入构建**（[`build_prefill_input`](https://github.com/sgl-project/sglang/blob/main/python/sglang_omni/models/qwen3_omni/components/talker_input.py)）：

Talker 不直接使用 Thinker 的 logits，而是使用 Thinker 的 **embeddings** 和 **hidden states** 作为 prefill：
1. 解析 chat template，按 `<|im_start|>` 分段
2. 对文本位置：`text_projection(thinker_embed)` → talker hidden
3. 对多模态位置：`hidden_projection(thinker_hidden[layer_24])` → talker hidden
4. 拼接所有段 → Talker 的 prefill input

**自回归解码**：
```
每步:
  1. 采样 codec token ~ p(t | context)
  2. 提取 talker_hidden
  3. 发送 (talker_hidden, codec_token) → code_predictor (stream)
  4. 接收 code_predictor 的 feedback (summed_embeddings)
  5. 将 feedback 注入下一步的输入
```

**Feedback 机制**：Talker 与 Code Predictor 之间存在双向流式通信。Talker 发出 codec token 后进入 `WAITING_FEEDBACK` 状态，等待 Code Predictor 返回各层 codec embedding 的加权和，作为下一步解码的额外上下文。

#### Stage 8: Code Predictor

**核心类**: [`_CodePredictorWrapper`](https://github.com/sgl-project/sglang/blob/main/python/sglang_omni/models/qwen3_omni/components/code_predictor_executor.py)

Code Predictor 是一个 5 层 dense Transformer（hidden=1024, vocab=2048），接收 Talker 的每一步输出：

1. 输入：`talker_hidden [1024]` + `layer0_code`（Talker 采样的 codec token）
2. 嵌入 layer-0 code，与 talker_hidden 拼接
3. 自回归生成 15 个额外 code（layer 1-15），组成 16 层 RVQ codes
4. 收集所有层的 codec embedding，求和 → `summed_embeddings`
5. **双路输出**：
   - `codes [16]` → code2wav（语音合成）
   - `summed_embeddings` → talker_ar（feedback）

#### Stage 9: Code2Wav

**核心类**: [`_Code2WavStreamingExecutor`](https://github.com/sgl-project/sglang/blob/main/python/sglang_omni/models/qwen3_omni/components/code2wav_executor.py)

Code2Wav 使用 HF 的 `Qwen3OmniMoeCode2Wav`（neural codec decoder / vocoder）将 RVQ codes 合成为音频波形：

1. 累积 code chunks 直到达到 `stream_chunk_size`
2. `_decode_incremental()`: 将 codes `[num_chunks, 16]` 输入 vocoder
3. 裁剪左侧上下文伪影（`left_context_size`）
4. 流式输出 float32 音频块（24kHz）
5. 最终拼接所有音频块

---

## OmniEngine: 调度与执行引擎

[`OmniEngine`](https://github.com/sgl-project/sglang/blob/main/python/sglang_omni/engines/omni/engine.py) 是 Thinker 和 Talker AR 的执行引擎，将 Scheduler（请求生命周期管理）和 ModelRunner（无状态模型执行）统一在一起。

### 请求生命周期

```
WAITING ──schedule()──▶ RUNNING ──update()──▶ FINISHED
   │                       │                      │
   │                       ▼                      │
   │              WAITING_FEEDBACK                 │
   │                  (Talker)                     │
   │                       │                      │
   │              resume_request()                 │
   │                       │                      │
   └───abort_request()─────┴──────────▶ ABORTED ──┘
```

**[`Scheduler`](https://github.com/sgl-project/sglang/blob/main/python/sglang_omni/engines/omni/scheduler.py)** 的核心方法：
- `add_request(request_id, data)` — 请求入队为 `WAITING` 状态
- `schedule()` — 通过 `BatchPlanner` 选择请求并构建 batch，返回 `SchedulerOutput`
- `update(scheduler_output, model_output)` — 根据模型输出更新请求状态，由 `IterationController` 判断是否完成
- `stream(request_id)` — 返回异步生成器，逐步 yield 中间输出

**[`ModelRunner`](https://github.com/sgl-project/sglang/blob/main/python/sglang_omni/engines/omni/model_runner.py)** 是无状态执行器，`execute(scheduler_output)` 调用模型 forward 并通过 `OutputProcessor` 转换输出。

### 执行模式

OmniEngine 支持两种执行模式：

#### Normal 模式（`_step_normal`）

顺序执行：
```
schedule() → execute() → update()
```

#### Overlap 模式（`_step_overlap`）

流水线化 GPU 计算和 CPU 状态更新：
```
┌─────────────────┬──────────────────┐
│    GPU: forward(step N)            │
├─────────────────┼──────────────────┤
│    CPU: update(step N-1)           │
└─────────────────┴──────────────────┘
```

关键实现：
1. `_step_overlap()` 通过 `asyncio.run_in_executor()` 将 GPU forward 提交到线程池
2. 在等待 GPU 的同时，CPU 处理上一步的 `_process_pending_result()`
3. 启发式策略：连续 prefill 时禁用 overlap（优化 TTFT）

### Runtime Protocol 接口

OmniEngine 通过一组 Protocol 接口实现模型无关的调度逻辑：

| Protocol | 职责 |
|----------|------|
| `BatchPlanner` | 选择请求 + 构建 batch（`select_requests`, `build_batch`） |
| `ResourceManager` | 内存资源管理（`can_allocate`, `allocate`, `free`） |
| `IterationController` | 判断请求是否完成（`update_request`, `is_finished`） |
| `InputPreparer` | 将 `SchedulerOutput` 转为模型输入 dict |
| `OutputProcessor` | 将模型输出转为 `RequestOutput` |
| `CacheManager` | 可选的输出缓存（`get`, `put`, `clear`） |

这些接口定义在 [`engines/omni/runtime/interfaces.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang_omni/engines/omni/runtime/interfaces.py)，由具体模型实现。

---

## 核心数据结构

### PipelineState

[`PipelineState`](https://github.com/sgl-project/sglang/blob/main/python/sglang_omni/models/qwen3_omni/io.py) 是贯穿整个流水线的状态容器，每个 Stage 在其基础上丰富数据：

```python
@dataclass
class PipelineState:
    raw_inputs: Any                    # 用户原始输入
    prompt: PromptInputs               # {input_ids, attention_mask, prompt_text}
    mm_inputs: dict[str, Any]          # {image: [...], audio: [...], video: [...]}
    encoder_inputs: dict[str, dict]    # {image_encoder: {...}, audio_encoder: {...}}
    encoder_outs: dict[str, Any]       # 编码器输出 {image_embeds, audio_embeds, ...}
    thinker_inputs: dict[str, Any]     # 合并后的 thinker 输入
    thinker_out: ThinkerOutput         # {output_ids, step, is_final, extra_model_outputs}
    engine_outputs: dict[str, Any]     # 最终解码结果
    stream_state: dict[str, Any]       # 流式输出状态追踪
```

### Scheduler 相关类型

定义在 [`engines/omni/types.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang_omni/engines/omni/types.py)：

- **`SchedulerStatus`** 枚举: `WAITING` / `RUNNING` / `WAITING_FEEDBACK` / `FINISHED` / `ABORTED`
- **`SchedulerRequest`**: 包含 `request_id`、`status`、`data`（模型特定的 opaque 数据）、时间戳
- **`SchedulerOutput`**: 包含选中的 requests 列表、`batch_data`、`step_id`
- **`RequestOutput`**: 单个请求的输出，包含 `finished` 标志和 `finish_reason`（"stop" / "length" / "abort"）
- **`ModelRunnerOutput`**: 一个 batch 的输出，`Dict[request_id → RequestOutput]`

### Control Plane 消息

定义在 [`proto/messages.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang_omni/proto/messages.py)：

- **`DataReadyMessage`**: 包含 `request_id`、`from_stage`、`to_stage`、`shm_metadata`（共享内存位置）、`chunk_id`（流式序号）、`is_done`、`error`
- **`CompleteMessage`**: 包含 `request_id`、`from_stage`、`success`、`result`、`error`
- **`AbortMessage`**: 仅包含 `request_id`

消息序列化使用 msgpack，跨进程传输使用 ZMQ。

---

## 关键设计模式

### 1. 声明式配置 → 运行时编译

流水线通过 [`PipelineConfig`](https://github.com/sgl-project/sglang/blob/main/python/sglang_omni/models/qwen3_omni/config.py) 声明式定义 Stage DAG，运行时由 `compile_pipeline()` 编译为可执行的 `Stage` + `Worker` + `Executor` 实例。每个 Stage 的 executor factory、路由函数、GPU 分配、relay 设备等均在配置中指定：

```python
# Qwen3OmniSpeechPipelineConfig 中的 Stage 定义示例
StageConfig(
    name="thinker",
    executor_factory=create_sglang_thinker_executor,
    next_stages=["decode", "talker_ar"],     # fan-out
    stream_to="talker_ar",                    # 流式转发
    relay_device="cuda",
)
```

GPU 分配通过 `gpu_placement` 字典控制：
```python
gpu_placement = {
    "thinker": 0,                 # GPU 0
    "talker_ar": 1,               # GPU 1
    "code_predictor": 1,          # GPU 1
    "code2wav": 1,                # GPU 1
}
```

### 2. Stage Payload 作为状态容器

`StagePayload` 不是简单的"模型输入"，而是请求在当前流水线位置的**完整状态快照**。每个 Stage 丰富状态而非消费状态——后续 Stage 始终可以访问前序 Stage 写入的所有字段。

### 3. Overlap 调度（GPU/CPU 流水线化）

OmniEngine 的 overlap 模式通过 `asyncio.run_in_executor()` 实现 GPU 计算与 CPU 状态更新的并行，显著提升吞吐量。启发式策略在连续 prefill 场景下禁用 overlap 以优化首 token 延迟（TTFT）。

### 4. Feedback 门控

Talker AR 与 Code Predictor 之间的双向通信通过 `WAITING_FEEDBACK` 状态和 `StreamQueue` 实现：
- Talker 每生成一个 codec token 后暂停，等待 feedback
- Code Predictor 处理完毕后将 `summed_embeddings` 通过 feedback 通道返回
- OmniEngine 的 `_check_feedback()` 检测到 feedback 后调用 `resume_request()` 恢复 Talker 解码

### 5. 多 Terminal 完成聚合

一个请求可以有多个 Terminal Stage（如文本的 `decode` + 语音的 `code2wav`）。Coordinator 等待所有 Terminal Stage 的 `CompleteMessage` 到达后，合并 `partial_results` 再返回调用方。

### 6. 同 GPU 零拷贝

当相邻 Stage 在同一 GPU 时（如 `talker_ar` → `code_predictor` 在 GPU:1），Worker 使用 CUDA IPC (`ForkingPickler`) 直接传递 tensor handle，避免 GPU→CPU→GPU 的拷贝开销。

---

## Acknowledge

本文档基于 SGLang Omni 代码进行整理。感谢 SGLang 社区的贡献者们。
