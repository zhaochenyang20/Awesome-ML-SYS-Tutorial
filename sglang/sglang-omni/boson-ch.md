# Boson AI × SGLang 正式发布 Higgs Audio v3 TTS：面向语音智能体的实时可控语音合成

*Boson AI & SGLang-Omni 团队*

*SGLang-Omni 团队*

[SGLang-Omni](https://github.com/sgl-project/sglang-omni) 现已支持 Boson AI [Higgs Audio v3 TTS](https://www.boson.ai/blog/higgs-audio-v3-tts) 的端到端 serving。Higgs Audio v3 TTS 是一款面向对话场景的 text-to-speech 模型：它可以在低延迟下生成自然、有表现力的语音，也允许开发者在文本流里直接控制情绪、风格、韵律和音效。模型在 [100 种语言上达到个位数 WER/CER](https://www.boson.ai/blog/higgs-audio-v3-tts)，也支持零样本声音克隆。

我们接入 Higgs，并不只是为了多支持一个 TTS 模型。Higgs 代表了一类越来越重要的生成模型：端到端生成过程不再是一条单独的自回归 decode loop，而是由多个计算特性各不相同的 stage 串联或交错完成。SGLang-Omni 正是为这类 multi-stage 模型重新设计的推理框架。

<iframe
  width="960"
  height="540"
  src="https://www.youtube.com/embed/i2PJeaywDew"
  title="Higgs Audio v3 TTS Demo"
  frameborder="0"
  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
  allowfullscreen
></iframe>

## 认识 Higgs Audio v3 TTS

### 为自然对话而生

对话式 TTS 的难点，不只是把一段完整文字读得好听。真实的语音智能体往往只能先拿到半句话，甚至几个字，就需要开始回应；后续文本还在持续到来，生成出来的声音却不能前后割裂。Higgs Audio v3 TTS 从设计上就面向这种流式对话场景：它不需要等到完整句子或标点出现，就可以开始合成语音，并在后续文本继续输入时保持音色、情绪和语速的一致。

从架构上看，Higgs 基于 Qwen3-4B backbone，是一个约 4B 参数的自回归解码器。模型消费交错排列的文本 token 和音频 token。音频会先由 Higgs Tokenizer 编码成 25 fps、8 路离散 codebook，再通过 delayed pattern 交错排列；多 codebook embedding 被融合后送入 backbone，最后由融合的多 codebook head 解码回 24 kHz 波形。整个生成过程在文本块与音频块之间交替推进，使得每个新的音频片段都能同时参考提示音频和已经生成的上下文。

### 质量稳定，多语言支持

Higgs Audio v3 TTS 在 Boson AI 内部的 **Higgs-Multilingual** 评测集上覆盖 111 种语言和方言，其中 **[100 种语言的 WER/CER 达到个位数](https://www.boson.ai/blog/higgs-audio-v3-tts)**。在公开多语言声音克隆 benchmark 上，v3 在 Seed-TTS、CV3 和 MiniMax-Multilingual 的 macro-average WER/CER 也都保持在个位数。零样本声音克隆只需要一小段参考音频；同一段参考音频也可以跨语言使用。

下表展示零样本声音克隆场景下的 WER/CER（↓，%）。所有数字均按对应 benchmark 的语言集合做 macro-average，指标和归一化方式可复现。

| Benchmark | Languages | WER/CER ↓ |
|---|---:|---:|
| Seed-TTS | 2 | 1.11 |
| CV3 | 9 | 4.41 |
| MiniMax-Multilingual | 23 | 2.74 |
| Higgs-Multilingual | 111 | 3.61 |

各语言的 Seed-TTS 细分指标，以及 WavLM 说话人相似度，请参阅 [SGLang-Omni Higgs Cookbook](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html)。

### 丰富可控制的演绎特性

Higgs Audio v3 TTS 不只追求音质，也重视可控性。开发者可以把控制标记直接写进输入文本，在同一段文本流里切换情绪、说话风格、语速、音高，插入停顿，甚至触发音效：

```text
<|emotion:amusement|><|prosody:expressive_high|>Wait, wait, that was kind of hilarious. <|sfx:laughter|>Hehe, no, seriously, I was not ready for that.
```

控制标记覆盖 20 多种情绪（`<|emotion:elation|>`、`<|emotion:anger|>`、`<|emotion:sadness|>` 等）、风格（`<|style:singing|>`、`<|style:whispering|>`、`<|style:shouting|>`）、韵律（`<|prosody:speed_very_slow|>`、`<|prosody:pitch_high|>`、`<|prosody:pause|>`、`<|prosody:long_pause|>`）以及音效（`<|sfx:cough|>`、`<|sfx:laughter|>`、`<|sfx:sigh|>` 等）。不同类别的标记可以组合使用。完整列表见 [SGLang-Omni Higgs Cookbook](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html#inline-control-tokens)。

## 使用 SGLang-Omni 服务 Higgs

Higgs 的 serving 和优化基于 [SGLang-Omni](https://github.com/sgl-project/sglang-omni) 完成。和标准 LLM 不同，Higgs 这类现代 TTS 模型很难被塞进一条统一的自回归解码循环里。它们的端到端生成过程天然包含多个 stage：有的 stage 更像标准 AR decoding，有的 stage 更像轻量函数式计算，有的 stage 则需要持续接收 chunk 并流式输出音频。

SGLang-Omni 的目标，就是让这类 multi-stage 模型可以用统一而清晰的方式被服务起来：每个 stage 按自己的计算特性调度，stage 之间用低开销通信连接，显存和进程拓扑则在框架层统一管理。

### 基于高性能 SGLang Backend 的多阶段解码

单阶段模型已经有非常成熟的推理路径：自回归 LLM 由 SGLang 主线优化，扩散模型由 SGLang-Diffusion 支持。SGLang-Omni 关注的是另一类模型：端到端生成过程被拆成多个计算特性不同的 stage。Higgs 是一个典型例子；Qwen3-Omni 的 Thinker → Talker → MTP，Fish Audio S2-Pro 的串行嵌套 Dual-AR，以及 Ming-Omni、LLaDA2.0-Uni 等全模态模型，也都属于这一类。

因此，SGLang-Omni 的运行时从一开始就是围绕 stage 抽象设计的。模型配置静态声明 pipeline 里的所有 stage、GPU 放置和进程拓扑；placement 和拓扑层负责拉起对应的 worker；Coordinator 负责在 stage 之间路由请求；每个 Stage 作为 IO 外壳接收上游数据，再交给内部 Scheduler 执行。

不同 stage 可以选择不同的调度器。自回归 stage（例如 Qwen3-Omni 的 Thinker）通常使用 `OmniScheduler`，保留 continuous batching、prefill/decode 混合调度、KV cache 管理、tree cache、CUDA Graph 等 SGLang 主线能力，同时适配 omni 请求对象和流式输出。非自回归 stage（例如小型 encoder、聚合器）可以使用 `SimpleScheduler`，本质上就是一个清晰的 get → forward → put 循环。流式 stage 则使用 `StreamingSimpleScheduler` 管理 chunk 和 done 的生命周期，例如 Higgs 流式模式下的 vocoder。

在这个设计里，Stage 之间的接口是统一的，但每个 stage 内部可以采用最适合自身计算特性的调度方式。为了让这一套机制真正跑得快，我们重点处理了三件事：

- **分层通信。** 控制消息走 ZMQ/msgpack，包括 submit、data-ready、stream、complete、shutdown、abort 等轻量信号；真正的大块 tensor 数据走 relay 数据面，可选择 `shm`、`nccl`、`nixl`、`mooncake` 等后端。同进程内可以本地派发，符合条件的同 GPU 流式 chunk 可以走 CUDA IPC，跨进程通信则严格遵守统一的 stage 契约。
- **进程-GPU-阶段拓扑。** Pipeline 在配置中声明 stage、路由、流式边、进程组、GPU 放置、tensor-parallel 大小，以及可选的融合 stage group。非 TP stage 显式声明进程组；TP stage 展开为每个 rank 一个进程，由 rank 0 持有对外 IO。紧凑的单机共置部署和更大规模的拆分/TP 部署，本质上只是同一套拓扑描述的不同实例。
- **显存隔离。** 在 multi-stage 场景下，GPU 显存不再是单个 scheduler 的全局比例，而是 stage 级别的资源契约。每个占用 GPU 的 stage 可以声明 `runtime.resources.total_gpu_memory_fraction`；placement 会按 GPU 汇总预算并做启动前校验。多个 stage 共享同一张卡时，显存预算必须被显式声明，避免某个 stage 在运行时挤占其他 stage 的空间。

### 复用 Omni 专用优化

接入 Higgs 的过程中，我们也把一些反复出现的 omni 优化抽象成了可复用模块。这样做的目的很简单：类似的计算过程不应该在不同模型里重复写一遍，性能优化也应该沉淀在框架里，而不是散落在各个模型的定制代码中。

- **支持 CUDA Graph 的反馈运行器。** Higgs 的 `tts_engine` 默认开启 CUDA Graph 捕获，并使用专门面向“自回归 + 多 codebook 反馈循环”的模型运行器。这个运行器统一静态 buffer 分配、延迟捕获时机，并对 Python 侧 gather/scatter 开销做了额外处理。同一套接口也可以支持 Qwen3-Omni、Fish Audio S2-Pro 等模型中的一步前瞻异步解码。
- **流式 vocoder 调度器。** Higgs、Qwen3-Omni、Fish Audio S2-Pro 等模型都需要类似的流式音频生命周期：按请求初始化状态，累积上游 code chunk，在上下文足够时尽早输出音频窗口，在 `stream_done` 时刷新缓冲区，并向客户端返回简洁的最终 payload。具体 codec 和窗口逻辑仍然由模型自己定义，但服务生命周期可以由框架统一管理。

有了这些抽象，新的 multi-stage 模型不需要从零写一套充满 if-else 的定制 pipeline。开发者只需要把模型拆成合适的调度片段，选择对应的 scheduler 和 model runner hook，声明拓扑和显存契约；剩下的路由、流式输出、数据搬运、进程放置和 stage 级资源隔离，都交给框架完成。

### Multi-stage 模型生态

Higgs 已经加入 SGLang-Omni 支持的 TTS 与 omni 模型生态：

| Model | Type | Notes |
|---|---|---|
| [Higgs Audio v3 TTS](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b) | TTS | 声音克隆、流式、100 种语言 |
| [Fish Audio S2-Pro](https://huggingface.co/fishaudio/s2-pro) | TTS | 声音克隆、流式 |
| [Voxtral TTS](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) | TTS | 命名音色、流式、9 种语言 |
| [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | TTS | 声音克隆、流式、10 种语言 |
| [MOSS-TTS-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5) | TTS | 声音克隆、流式、31 种语言 |
| [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | Omni | 文本/图像/音频/视频 → 文本 + 音频 |
| [Ming-flash-omni-2.0](https://huggingface.co/inclusionAI/Ming-flash-omni-2.0) | Omni | 流式 TTS |
| [LLaDA2.0-Uni](https://huggingface.co/inclusionAI/LLaDA2.0-Uni) | Multimodal | 文本 + 图像理解与生成 |

这些模型的外在形态不同，但在推理系统看来，它们共享同一种更底层的问题：如何把多个异构 stage 组织成一条稳定、高效、可扩展的生成流水线。也正因为如此，在 SGLang-Omni 上接入 Higgs 时，我们的主要工作集中在声明 pipeline（`preprocessing → audio_encoder → tts_engine → vocoder`）和补齐模型专属 hook，而不是重新搭一套服务栈。

### Higgs 端到端优化

在框架抽象之外，我们也围绕 Higgs pipeline 做了一轮端到端优化。这里列出主要工作；更完整的实现和跟踪请参考 [Higgs 优化路线图 (#478)](https://github.com/sgl-project/sglang-omni/issues/478) 与 [代码仓库](https://github.com/sgl-project/sglang-omni)。

- **自回归 backbone**：支持解码循环的 [CUDA Graph 捕获](https://github.com/sgl-project/sglang-omni/pull/503)、omni 自回归循环的 [异步（一步前瞻）解码](https://github.com/sgl-project/sglang-omni/pull/590)，并将每步 device-to-host 同步 [合并为单次传输](https://github.com/sgl-project/sglang-omni/pull/572)。
- **编码器**：将 [预处理并入编码器阶段](https://github.com/sgl-project/sglang-omni/issues/576)，为参考音频添加 [LRU 缓存](https://github.com/sgl-project/sglang-omni/pull/563)（[#605](https://github.com/sgl-project/sglang-omni/pull/605)），并支持 [批量音频编码器](https://github.com/sgl-project/sglang-omni/pull/610)。
- **声码器**：支持 [批量 vocoder 解码](https://github.com/sgl-project/sglang-omni/pull/574)。
- **缓存**：使用 `extra_key` 命名空间按参考音频划分 RadixAttention 缓存，让重复的声音克隆请求可以复用 prefix。
- **调度与流式**：废弃 Higgs 早期的 [定制调度器](https://github.com/sgl-project/sglang-omni/pull/476)，改用共享的 `OmniScheduler`，并落地真正的 SSE [流式](https://github.com/sgl-project/sglang-omni/pull/597) [调度器](https://github.com/sgl-project/sglang-omni/pull/614)，显著降低首包音频延迟。

### 性能

我们在 Seed-TTS 英文全集上测试 Higgs（每次运行 **N=1088**）。客户端通过 `--max-concurrency` 扫描并发度；服务端使用 Higgs server（`max_running_requests=16`，bf16，开启 CUDA Graph）。下表每一行都是 **3 次运行的平均值**。硬件为 **1× H100**。

| Concurrency | Throughput (req/s) | Mean latency | RTF (per-req) | audio_s/s |
|---:|---:|---:|---:|---:|
| 1 | 1.62 | 617 ms | 0.147 | 6.89 |
| 2 | 2.70 | 742 ms | 0.180 | 11.37 |
| 4 | 5.45 | 733 ms | 0.177 | 22.84 |
| 8 | 8.91 | 898 ms | 0.217 | 37.38 |
| 16 | 14.74 | 1079 ms | 0.262 | 61.84 |

- **Concurrency**：客户端最大在途请求数（`--max-concurrency`）。
- **Throughput (req/s)**：完成请求数除以 benchmark 总墙钟时间。
- **Mean latency**：单个请求端到端平均耗时，从发送请求到收到完整响应。
- **RTF (per-req)**：处理时间与生成音频时长之比，低于 1 表示快于实时。
- **audio_s/s**：生成音频总秒数除以 benchmark 总墙钟时间。

复现方式见 [benchmark 脚本](https://github.com/sgl-project/sglang-omni/blob/main/benchmarks/eval/benchmark_tts_seedtts.py)。

## 快速上手

完整说明请参阅 [SGLang-Omni Higgs Cookbook](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html)。下面给出最短可运行路径。

### 安装与服务

```bash
docker pull lmsys/sglang-omni:dev
docker run -it --gpus all --shm-size 32g --ipc host --network host --privileged \
  lmsys/sglang-omni:dev /bin/zsh

git clone git@github.com:sgl-project/sglang-omni.git && cd sglang-omni
uv venv .venv -p 3.12 && source .venv/bin/activate
uv pip install -v -e .
```

```bash
hf download bosonai/higgs-audio-v3-tts-4b

sgl-omni serve \
  --model-path bosonai/higgs-audio-v3-tts-4b \
  --port 8000
```

### 零样本合成

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, how are you?"}' \
  --output output.wav
```

参考输出：

<audio controls>
  <source src="https://sgl-project.github.io/sglang-omni/_static/audio/higgs-1.wav" type="audio/wav">
</audio>

### 声音克隆

声音克隆时可以提供参考音频，也建议同时提供参考文本（`text`），通常能显著提升克隆质量：

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Have a nice day and enjoy south california sunshine.",
    "references": [{
      "audio_path": "https://sgl-project.github.io/sglang-omni/_static/audio/male-voice.wav",
      "text": "Hey, Adam here. Let'\''s create something that feels real, sounds human, and connects every time."
    }],
    "temperature": 0.8,
    "top_k": 50,
    "max_new_tokens": 1024
  }' \
  --output output.wav
```

参考输入：

<audio controls>
  <source src="https://sgl-project.github.io/sglang-omni/_static/audio/male-voice.wav" type="audio/wav">
</audio>

参考输出：

<audio controls>
  <source src="https://sgl-project.github.io/sglang-omni/_static/audio/higgs-2.wav" type="audio/wav">
</audio>

### 流式生成

在请求体中设置 `"stream": true` 后，客户端可以通过 [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) 接收音频，不必等整段生成完成才开始播放。vocoder 会增量输出 WAV chunk，从而降低首包音频延迟。下面命令中的 `-N` 用来关闭 curl 输出缓冲，让 SSE 事件到达后立即打印：

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "https://sgl-project.github.io/sglang-omni/_static/audio/male-voice.wav",
      "text": "Hey, Adam here. Let'\''s create something that feels real, sounds human, and connects every time."
    }],
    "stream": true
  }'
```

如果需要原始 PCM 流式输出（不带 SSE JSON），请参考 [Higgs TTS cookbook](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html#streaming)。

参考输出：

<audio controls>
  <source src="https://sgl-project.github.io/sglang-omni/_static/audio/higgs-4.wav" type="audio/wav">
</audio>

### 内联控制标记

控制标记可以直接写入 `input` 字段，不同类别可以组合使用。一般来说，每轮开头可以先放演绎类标记（emotion、style、speed/pitch/expressive prosody）；`<|prosody:pause|>` / `<|prosody:long_pause|>` 放在需要停顿的位置；每个 `<|sfx:…|>` 后面紧跟对应拟声词。完整列表见 [cookbook](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html#inline-control-tokens)。

**情绪：amusement + laughter**

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "<|emotion:amusement|><|prosody:expressive_high|>Wait, wait, that was kind of hilarious. <|sfx:laughter|>Hehe, no, seriously, I was not ready for that.",
    "temperature": 0.8,
    "top_k": 50,
    "max_new_tokens": 1024
  }' \
  --output output.wav
```

参考输出：

<audio controls>
  <source src="https://sgl-project.github.io/sglang-omni/_static/audio/control-tokens-test1.wav" type="audio/wav">
</audio>

**情绪：anger + shouting**

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "<|emotion:anger|><|style:shouting|>No, that is not okay! We cannot ship something that sounds broken, delayed, and unnatural.",
    "temperature": 0.8,
    "top_k": 50,
    "max_new_tokens": 1024
  }' \
  --output output.wav
```

参考输出：

<audio controls>
  <source src="https://sgl-project.github.io/sglang-omni/_static/audio/control-tokens-test2.wav" type="audio/wav">
</audio>

**情绪：surprise + screaming**

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "<|emotion:surprise|><|prosody:pitch_high|><|sfx:screaming|>Ah! Wait, I almost forgot! Higgs Audio v3 also supports over one hundred languages.",
    "references": [{
      "audio_path": "https://sgl-project.github.io/sglang-omni/_static/audio/ref_voice.wav",
      "text": "It was the night before my birthday. Hooray! It’s almost here! It may not be a holiday, but it’s the best day of the year."
    }],
    "temperature": 0.8,
    "top_k": 50,
    "max_new_tokens": 1024
  }' \
  --output output.wav
```
参考输出：

<audio controls>
  <source src="https://sgl-project.github.io/sglang-omni/_static/audio/control-tokens-test5.wav" type="audio/wav">
</audio>

**组合示例：**

下面用一段高考英语听力风格的双人短对话，展示 emotion、sound effects 和 prosody 标记如何组合使用：

<details>
<summary>命令</summary>

Part 1 — 询问缺课内容：

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "<|emotion:contemplation|>Hi David, I missed the biology class today because I caught a cold. <|sfx:cough|>Ahem! Sorry, Could you tell me what the teacher covered?",
    "references": [{
      "audio_path": "https://sgl-project.github.io/sglang-omni/_static/audio/female-voice.wav",
      "text": "By repeating what students say, teachers can demonstrate that they are listening. By extending what students say."
    }],
    "temperature": 0.8,
    "top_k": 50,
    "max_new_tokens": 1024
  }' \
  --output part1.wav
```

Part 2 — 说明课堂内容：

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "<|emotion:enthusiasm|>Sure, no problem! We learned how plants make food through photosynthesis, and <|prosody:long_pause|> there will be a quiz this Friday.",
    "references": [{
      "audio_path": "https://sgl-project.github.io/sglang-omni/_static/audio/male-voice.wav",
      "text": "Hey, Adam here. Let'\''s create something that feels real, sounds human, and connects every time."
    }],
    "temperature": 0.8,
    "top_k": 50,
    "max_new_tokens": 1024
  }' \
  --output part2.wav
```

Part 3 — 表示感谢：

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "<|emotion:relief|>Oh, that is really helpful. Thank you!",
    "references": [{
      "audio_path": "https://sgl-project.github.io/sglang-omni/_static/audio/female-voice.wav",
      "text": "By repeating what students say, teachers can demonstrate that they are listening. By extending what students say."
    }],
    "temperature": 0.8,
    "top_k": 50,
    "max_new_tokens": 1024
  }' \
  --output part3.wav
```

拼接（句间约 0.6 秒间隔）：

```bash
ffmpeg -y \
  -i part1.wav -f lavfi -t 0.6 -i anullsrc=r=24000:cl=mono \
  -i part2.wav -f lavfi -t 0.6 -i anullsrc=r=24000:cl=mono \
  -i part3.wav \
  -filter_complex "[0:a][1:a][2:a][3:a][4:a]concat=n=5:v=0:a=1" \
  gaokao_listening.wav
```

</details>

参考输出：

<audio controls>
  <source src="https://sgl-project.github.io/sglang-omni/_static/audio/gaokao-listening.wav" type="audio/wav">
</audio>

### Demo

也可以一条命令启动后端和浏览器 UI：

```bash
CUDA_VISIBLE_DEVICES=0 ./playground/higgs/start.sh
```

> **TODO [ DEMO VIDEO PLACEHOLDER ] @yichi**

## Roadmap

对 SGLang-Omni 来说，跑通 Higgs 是一个重要节点，但不是终点。我们接下来会继续推进几件事：

- **跟进上游 SGLang**（[#658](https://github.com/sgl-project/sglang-omni/issues/658)）：升级到最新 SGLang，让自回归 backbone 继续继承主线在 CUDA/PyTorch 构建矩阵、kernel、调度和 speculative decoding 等方向的收益。
- **按模型重构**（[#661](https://github.com/sgl-project/sglang-omni/issues/661)）：延续 [RFC #188](https://github.com/sgl-project/sglang-omni/issues/188)，形成更清晰的 per-model 抽象。我们希望新模型接入越来越接近“声明拓扑 + 挂载 hook”，而不是在框架各处堆积特殊分支。
- **端到端 RL**（[#663](https://github.com/sgl-project/sglang-omni/issues/663)）：将 SGLang-Omni 作为 omni 与 TTS 模型的高吞吐 rollout 后端，并支持显式 reward 目标，进一步打通 serving 与 post-training。

跨节点 multi-stage pipeline 和更完整的 diffusion stage 支持也在推进中。我们相信，清晰的 stage 抽象、统一的调度器接口、分层通信和跨 stage 显存预算，会让这些能力自然长在同一套框架上。

## 欢迎大家加入

SGLang-Omni 仍在快速演进。我们希望它成为 multi-stage 生成模型的一套通用推理基础设施：新模型不必从零搭 pipeline，不必在十几个文件里写特殊判断，而是把计算过程拆成清晰的 stage，声明拓扑、挂载 hook，剩下的调度、通信、显存管理和流式服务都由框架承担。

如果你也在思考多阶段推理框架该怎么设计，或者对 TTS、omni、多模态生成、推理系统、RL rollout 后端这些问题感兴趣，欢迎参与 SGLang-Omni。无论你更擅长 kernel、调度、通信、模型接入还是 benchmark，我们都欢迎贡献者加入讨论、提交 PR，一起把这套框架继续往前推进。

## 致谢

**SGLang-Omni** — Haoguang Cai, Shangming Cai, Qiujiang Chen, Jiaxin Deng, Wenyao Gao, Yifei Gao, Jingwen Gu, Yitong Guan, Chenchen Hong, Hao Jin, Xinli Jing, Shenggui Li, Junrong Lin, Xinyu Lu, Yuan Luo, Ratish Palanisamy, Mick Qian, JinTao Qu, Shuai Shi, Chao Wang, Richard Wang, Shuwen Wang, Zijie Xia, Yuhao Yang, Xuesong Ye, Yin Yue, Fan Yin, Gaokai Zhang, Xiaoyu Zhang, Yichi Zhang, Chenyang Zhao.

**Higgs Audio v3 TTS (Boson AI)** — Mu Li, Alex Smola, Lindsey Allen. Silin Meng, Ke Bai. Ruskin Raj Manku, Huapeng Zhou, Dongming Shen, Jonah Mackey, Erik Li, Weisu Yin, Yizhi Liu, Xinyu Wang, Hao Yu.

## 延伸阅读

- **模型：** [`bosonai/higgs-audio-v3-tts-4b`](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b)
- **博客：** [Higgs Audio v3 TTS](https://www.boson.ai/blog/higgs-audio-v3-tts)
- **服务框架：** [SGLang-Omni on GitHub](https://github.com/sgl-project/sglang-omni)
- **文档：** [SGLang-Omni docs](https://sgl-project.github.io/sglang-omni/) · [Higgs TTS cookbook](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html)
- **Higgs 优化路线图：** [#478](https://github.com/sgl-project/sglang-omni/issues/478)
- **设计背景：** *SGLang-Omni: Redesigning the Inference Framework for Multi-Stage Generative Models*