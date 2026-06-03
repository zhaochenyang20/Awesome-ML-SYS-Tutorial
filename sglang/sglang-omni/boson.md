# Higgs Audio v3: A Chat-Native TTS Model, Powered by SGLang-Omni

*Boson AI & SGLang-Omni Team*

Boson AI and the SGLang community are jointly releasing **Higgs Audio v3 Generation**, a chat-native text-to-speech model built for real-time, expressive, and controllable speech in voice-agent settings — speaking 100+ languages with state-of-the-art accuracy and directable inline control over emotion, style, prosody, and sound effects. The model is served end-to-end by [**SGLang-Omni**](https://github.com/sgl-project/sglang-omni), the multi-stage inference framework we built together with the LMSYS team.

> **[ DEMO VIDEO — hero ] @huapeng**

## Meet Higgs Audio v3 Generation

### Chat-Native by Design

A good conversational TTS model should be able to start speaking given only half a sentence — or even just a few words — and keep going as the rest of the text streams in. Higgs Audio v3 Generation was designed from the ground up for this kind of turn-taking: speech begins within milliseconds, never has to wait for a punctuation mark, and stays consistent in voice, emotion, and pace as more text arrives. The result is a voice that feels like it is *listening and answering*, rather than transcribing a finished script.

Architecturally, Higgs is a ~4B autoregressive decoder built on a Qwen3-4B backbone. It consumes interleaved text and audio tokens; audio is encoded into 8 discrete codebooks at 25 fps, staggered via a delay pattern, mapped to backbone hidden states through a fused multi-codebook embedding, and decoded back to 24 kHz waveform through a fused multi-codebook head. Multi-turn generation interleaves text and audio chunks so each new chunk is grounded on the reference and prior context.

### Multilingual, with Quality that Holds

Out of the box the model speaks **100+ languages and dialects**, with **90+ languages reaching single-digit WER/CER** on internal multilingual evaluations. Across the standard public benchmarks, v3 sets Boson's highest accuracy to date while *also* pushing WavLM speaker similarity up, rather than trading one for the other. Zero-shot voice cloning needs only a short reference clip — and works across languages from the same reference.

WER/CER (↓, %) and WavLM speaker similarity (↑, ×100), macro-averaged per benchmark, zero-shot voice cloning at release:

> *[Need Update] @xinli*

| Benchmark | Languages | WER/CER ↓ | SIM ↑ |
|---|---:|---:|---:|
| Seed-TTS | 2 | 2.02 | 67.91 |
| CV3 | 9 | 7.82 | 66.27 |
| MiniMax-Multilingual | 23 | 3.17 | 75.92 |

*(Per-language breakdowns are in the [SGLang Omni Higgs Cookbook](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html).)*

> *We need to align the model card with the SGLang Omni Higgs Cookbook. Ideally we should update the results of benchmarks after the latest checkpoint is released and PR it to huggingface. @xinli*

### Directable: Control the Delivery from the Text Stream

Beyond raw quality, v3 is built to be *directed*. Inline control tags let you change emotion, switch speaking style, adjust speed and pitch, insert pauses, and trigger sound effects — all mid-utterance, all from the text stream:

```text
I can't believe it! <|emotion:surprise|> <|prosody:pause|> <|style:whispering|> Higgs and SGLang are absolutely incredible.
```

The tag families cover 20+ emotions (`<|emotion:elation|>`, `<|emotion:anger|>`, `<|emotion:sadness|>`, …), styles (`<|style:singing|>`, `<|style:whispering|>`, `<|style:shouting|>`), prosody (`<|prosody:speed_very_slow|>`, `<|prosody:pitch_high|>`, `<|prosody:pause|>`, `<|prosody:long_pause|>`), and sound effects (`<|sfx:cough|>`, `<|sfx:laughter|>`, `<|sfx:sigh|>`, …). Tags from different categories can be combined freely. The full catalogue lives in the [model card](https://huggingface.co/boson-sglang/higgs-audio-v3-generation-4B-base).

---

## Serving Higgs with SGLang-Omni

Higgs is served and optimized by [**SGLang-Omni**](https://github.com/sgl-project/sglang-omni), the omni framework of SGLang. Unlike a standard single-stage autoregressive LLM, modern TTS and omni models, such as Higgs, do not fit into one uniform decode loop. SGLang-Omni gives these models a multi-stage runtime, and then layers reusable omni-specific fast paths on top.

### Multi-stage Decoding with a High-Performance SGLang Backend

Single-stage models such as autoregressive LLMs and diffusion models are already pushed hard by SGLang main and SGLang-Diffusion. SGLang-Omni targets **multi-stage decoding** as the next serving regime, where end-to-end generation is split into heterogeneous stages with different resource bottlenecks. Higgs is a good example. So is Qwen3-Omni (Thinker → Talker → MTP), Fish Audio S2-Pro (a serially-nested Dual-AR), and other fully omni-modal models like Ming-Omni and LLaDA2.0-Uni.

SGLang-Omni's multi-stage runtime is built around this heterogeneity. In SGLang-Omni's `HTTP - Coordinator - Stage - Scheduler - Model-Runner - Model Forward` design, the model configuration statically declares the stages, GPU placement, and process topology required for serving; the runtime placement and topology layers prepare the workers, and the coordinator routes requests across them. The `stage` acts as an IO shell handling the direction of the data flow. It streams out the data to the next targeted stage, and as the next stage's `stage` IO shell receives the data, transmits it to the scheduler layer. SGLang-Omni provides multiple high-performance schedulers for different tasks across stages. AR stages, such as the Qwen3-Omni thinker, usually use `OmniScheduler`, supporting continuous batching, mixed prefill/decode, KV-cache management, tree cache, and CUDA-Graph support, with omni-native request objects and streaming output. Non-AR stages, such as small encoders and aggregators, use `SimpleScheduler` for function-style work. Streaming stages use `StreamingSimpleScheduler` for chunk/done request lifecycles, such as Higgs' vocoder under streaming mode. The scheduler decides when work runs, and it passes the job to the model runner just like SGLang main, and the model runner decides how the forward path is prepared, executed, and post-processed.

A few more details for multi-stage design to enable high-performance omni-modal serving:

- **Layered communication.** A ZMQ/msgpack control plane carries lightweight signals such as submit, data-ready, stream, complete, shutdown, and abort. Tensor payloads move through the relay data plane, with `shm`, `nccl`, `nixl`, and `mooncake` backends available. Same-process edges can use local dispatch, and eligible same-GPU streaming chunks can use CUDA IPC, while cross-process edges keep the same stage-level contract.
- **Process-GPU-stage topology.** Pipelines declare stages, routing, streaming edges, process groups, GPU placement, tensor-parallel size, and optional fused stage groups in config. Non-TP stages explicitly declare their process group; TP stages expand into per-rank processes, with rank 0 owning external stage IO. This makes compact colocated deployments and larger split/TP deployments variations of the same topology description rather than different serving stacks.
- **Memory isolation.** GPU memory is a stage-level resource contract instead of a global fraction to effectively support multi-stage colocation on one GPU. This makes it possible for multiple schedulers under multiple stage shells to live on a single GPU. Each GPU-backed stage can declare `runtime.resources.total_gpu_memory_fraction`; placement validation sums budgets per GPU and requires explicit budgets when multiple process groups share a card. It greatly reduces developers' burden to adapt the framework onto different workloads and computation devices.

### Reusing Omni-specific optimizations

SGLang-Omni turns recurring omni optimizations into reusable modules, making performance optimization easier, and enhancing codebase readability.

- **CUDA-Graph-friendly feedback runners.** Higgs' `tts_engine` enables CUDA-Graph capture by default and uses the Model-Runner designed for the AR + multi-codebook feedback loop through unified static buffer assignment and deferred capture, with special extra handling over Python-side gather/scatter. The same runner interface supports one-step-lookahead async decode for Qwen3-Omni, Fish Audio S2-Pro, and more across SGLang-Omni.
- **Streaming vocoder schedulers.** Higgs, Qwen3-Omni, Fish Audio S2-Pro, and more reuse the shared streaming scheduler lifecycle where they initialize per-request state, accumulate incoming code chunks, emit audio windows as soon as enough context is available, flush on `stream_done`, and return a slim final payload for streaming clients. The codec and windowing logic stay model-specific, but the serving lifecycle is shared.

With the overall design, a new multi-stage model does not need a bespoke pipeline with if-else branches scattered across the codebase. Developers only need to partition the model into scheduling segments, choose the right scheduler/model-runner hooks, declare the topology and memory contract, and let the framework handle routing, streaming, data movement, process placement, and stage-level resource isolation.

### Growing Multi-stage Model Ecosystem

Higgs joins a roster of TTS and omni models already supported by SGLang-Omni:

| Model | Type | Notes |
|---|---|---|
| [Higgs Audio v3](https://huggingface.co/boson-sglang/higgs-audio-v3-generation-4B-base) | TTS | Voice cloning, streaming, 100+ languages |
| [Fish Audio S2-Pro](https://huggingface.co/fishaudio/s2-pro) | TTS | Voice cloning, streaming |
| [Voxtral TTS](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) | TTS | Named voices, streaming, 9 languages |
| [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | TTS | Voice cloning, streaming, 10 languages |
| [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | Omni | Text/image/audio/video → text + audio |
| [Ming-flash-omni-2.0](https://huggingface.co/inclusionAI/Ming-flash-omni-2.0) | Omni | Streaming TTS |
| [LLaDA2.0-Uni](https://huggingface.co/inclusionAI/LLaDA2.0-Uni) | Multimodal | Text + image understanding and generation |

The same scheduler interface, layered communication, and memory budgeting back all of them — which is exactly why onboarding Higgs was largely a matter of declaring its pipeline (`preprocessing → audio_encoder → tts_engine → vocoder`) rather than building a serving stack from scratch.

### Optimizing Higgs End-to-End

On top of the framework, the team drove a full performance pass across every stage of the Higgs pipeline. We list the levers by name here; the implementations and tracking live in the [Higgs optimization roadmap (#478)](https://github.com/sgl-project/sglang-omni/issues/478) and the [repository](https://github.com/sgl-project/sglang-omni).

- **AR backbone** — [CUDA-Graph capture](https://github.com/sgl-project/sglang-omni/pull/503) for the decode loop, [async (one-step lookahead) decode](https://github.com/sgl-project/sglang-omni/pull/590) for the omni AR loop, and [batching the per-step D2H syncs](https://github.com/sgl-project/sglang-omni/pull/572) into a single transfer.
- **Encoder** — [fusing preprocessing into the encoder stage](https://github.com/sgl-project/sglang-omni/issues/576), an [LRU cache](https://github.com/sgl-project/sglang-omni/pull/563) for [reused reference audio](https://github.com/sgl-project/sglang-omni/pull/605), and a [batched audio encoder](https://github.com/sgl-project/sglang-omni/pull/610).
- **Vocoder** — [batched vocoder decode](https://github.com/sgl-project/sglang-omni/pull/574).
- **Caching** — a RadixAttention cache keyed per reference audio (`extra_key` namespacing), so repeated voice-cloning references hit cache.
- **Scheduling & streaming** — [dropping the bespoke scheduler](https://github.com/sgl-project/sglang-omni/pull/476) in favor of the shared `OmniScheduler`, plus real SSE [streaming](https://github.com/sgl-project/sglang-omni/pull/597) [schedulers](https://github.com/sgl-project/sglang-omni/pull/614) for low time-to-first-audio.

### Performance

Throughput on Seed-TTS English (full set):

> *[Update later] @huapeng add the detailed throughput, latency, TTFT, RTF, etc. Also, add reproduce instructions.*

| Concurrency | Mean latency | RTF (per-req) | audio_s / s |
|---:|---:|---:|---:|
| 1 | 4,637 ms | 0.526 | 1.90 |
| 16 | 7,138 ms | 0.747 | 12.88 |
| 32 | 10,188 ms | 0.865 | 16.94 |

## Try it Yourself

Detailed instructions are in the [SGLang Omni Higgs Cookbook](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html).

### Install and Serve

> *TODO: Release docker and update the instructions, and release version. @huapeng*

```bash
docker pull frankleeeee/sglang-omni:dev
docker run -it --gpus all --shm-size 32g --ipc host --network host --privileged \
  frankleeeee/sglang-omni:dev /bin/zsh

git clone git@github.com:sgl-project/sglang-omni.git && cd sglang-omni
uv venv .venv -p 3.12 && source .venv/bin/activate
uv pip install -v -e .
```

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
hf download boson-sglang/higgs-audio-v3-generation-4B-base

sgl-omni serve \
  --model-path boson-sglang/higgs-audio-v3-generation-4B-base \
  --port 8000
```

### Zero-shot synthesis

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, how are you?"}' \
  --output output.wav
```

Reference output:

<audio controls>
  <source src="https://sgl-project.github.io/sglang-omni/_static/audio/higgs-1.wav" type="audio/wav">
</audio>

### Voice cloning

Supplying the reference transcript (`text`) materially improves cloning fidelity:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Have a nice day and enjoy the southern California sunshine.",
    "references": [{
      "audio_path": "https://.../reference.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }]
  }' \
  --output output.wav
```

Reference input:

<audio controls>
  <source src="https://sgl-project.github.io/sglang-omni/_static/audio/higgs-3.wav" type="audio/wav">
</audio>

Reference output:

<audio controls>
  <source src="https://sgl-project.github.io/sglang-omni/_static/audio/higgs-2.wav" type="audio/wav">
</audio>

### Streaming

Set `"stream": true` to receive audio over Server-Sent Events and start playback before generation finishes — the vocoder emits incremental WAV chunks, dramatically lowering time-to-first-audio:

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Get the trust fund to the bank early.", "stream": true}'
```

Reference output:

<audio controls>
  <source src="https://sgl-project.github.io/sglang-omni/_static/audio/higgs-4.wav" type="audio/wav">
</audio>

### Inline Control Tokens

Embed control tokens directly in the `input` field. Tokens from different categories can be combined.

**Emotion: surprise**

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "I cant believe it! <|emotion:surprise|> <|prosody:pause|> <|style:whispering|> Higgs Model and SGLang are absolutely incredible."
  }' \
  --output output.wav
```

Reference output:

<audio controls>
  <source src="https://sgl-project.github.io/sglang-omni/_static/audio/control-tokens-test1.wav" type="audio/wav">
</audio>

**Prosody: speed_slow**

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "<|emotion:enthusiasm|> Welcome to the show! <|prosody:pause|> <|prosody:speed_slow|> Today we have something truly special for you."
  }' \
  --output output.wav
```
Reference output:

<audio controls>
  <source src="https://sgl-project.github.io/sglang-omni/_static/audio/control-tokens-test2.wav" type="audio/wav">
</audio>

**Combine them together:**

Here is an example of combining emotion, prosody and style tokens together:

<details>
<summary>Commands</summary>

Part 1 — female asks:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "<|prosody:pitch_high|> <|prosody:speed_slow|> Excuse me. Can you tell me how much the shirt is?",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_103675.wav",
      "text": "Excuse me. Can you tell me how much the shirt is?"
    }],
    "temperature": 0.5,
    "top_k": 30,
    "seed": 404
  }' \
  --output part1.wav
```

Part 2 — male answers:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "<|prosody:speed_very_slow|> <|prosody:expressive_low|> Yes, it is nine fifteen.",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }],
    "temperature": 0.5,
    "top_k": 30,
    "seed": 43
  }' \
  --output part2.wav
```

Part 3 — female reads the question:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "<|prosody:speed_slow|> <|prosody:expressive_low|> Question: How much is the shirt?",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_103675.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }],
    "temperature": 0.5,
    "top_k": 30,
    "seed": 44
  }' \
  --output part3.wav
```

Concatenate (~0.6 s gap between lines):

```bash
ffmpeg -y \
  -i part1.wav -f lavfi -t 0.6 -i anullsrc=r=24000:cl=mono \
  -i part2.wav -f lavfi -t 0.6 -i anullsrc=r=24000:cl=mono \
  -i part3.wav \
  -filter_complex "[0:a][1:a][2:a][3:a][4:a]concat=n=5:v=0:a=1" \
  gaokao_listening.wav
```

</details>

Reference output:

<audio controls>
  <source src="https://sgl-project.github.io/sglang-omni/_static/audio/gaokao-listening.wav" type="audio/wav">
</audio>

### Demo

A one-command playground launches the backend and a browser UI:

```bash
CUDA_VISIBLE_DEVICES=0 ./playground/higgs/start.sh
```

> **[ DEMO VIDEO PLACEHOLDER ] @yichi**

## Roadmap
 
This release is a milestone, not a finish line. The near-term tracks, each tracked in the open:
 
- **Tracking upstream SGLang** ([#658](https://github.com/sgl-project/sglang-omni/issues/658)) — moving onto the latest SGLang so the AR backbone inherits main's newest gains (updated CUDA / PyTorch build matrix, kernel improvements, the latest scheduling and speculative-decoding work) for free.
- **Per-model refactor** ([#661](https://github.com/sgl-project/sglang-omni/issues/661)) — continuing the direction of [RFC #188](https://github.com/sgl-project/sglang-omni/issues/188): a cleaner per-model abstraction that drives new-model integration toward "declare a topology and plug in hooks," keeping the codebase lean as the model zoo grows.
- **End-to-end RL** ([#663](https://github.com/sgl-project/sglang-omni/issues/663)) — an RFC for using SGLang-Omni as a high-throughput rollout backend for omni and TTS models with explicit reward targets, closing the loop between serving and post-training.
Cross-node multi-stage pipelines and fuller diffusion-stage support are also in flight. With clean stage abstraction, a unified scheduler interface, layered communication, and cross-stage memory budgeting already in place, these should land with calm and grace rather than another from-scratch build.

## Join us

SGLang-Omni is an open community project, and it is still growing fast. Cross-node multi-stage pipelines, fuller diffusion-stage support, and end-to-end RL training integration are all underway. If multi-stage inference is the kind of problem you find beautiful — whether you come from a systems background or arrive halfway, whether you specialize in kernel optimization or scheduling logic — **we are actively recruiting contributors**. Come build a truly industrial-grade omni-serving stack with us: open a PR, join the discussion, or say hi in the community channels linked below.

## Acknowledgments

**Higgs Audio v3 (Boson AI)** — Lead: Mu Li, Alex Smola, Lindsey Allen. Silin Meng, Ke Bai. Ruskin Raj Manku, Huapeng Zhou, Silin Meng, Dongming Shen. Jonah Mackey, Ke Bai, Ruskin Raj Manku, Erik Li, Weisu Yin, Yizhi Liu. 

**SGLang-Omni** — Haoguang Cai, Shangming Cai, Qiujiang Chen, Jiaxin Deng, Wenyao Gao, Yifei Gao, Jingwen Gu, Yitong Guan, Chenchen Hong, Hao Jin, Xinli Jing, Shenggui Li, Junrong Lin, Xinyu Lu, Yuan Luo, Ratish Palanisamy, Mick Qian, Jinjiang Qu, Shuai Shi, Chao Wang, Richard Wang, Shuwen Wang, Zijie Xia, Yuhao Yang, Xuesong Ye, Fan Yin, Gaokai Zhang, Xiaoyu Zhang, Yichi Zhang, Chenyang Zhao.

## Learn More

- **Model:** [`boson-sglang/higgs-audio-v3-generation-4B-base`](https://huggingface.co/boson-sglang/higgs-audio-v3-generation-4B-base)
- **Serving framework:** [SGLang-Omni on GitHub](https://github.com/sgl-project/sglang-omni)
- **Documentation:** [SGLang-Omni docs](https://sgl-project.github.io/sglang-omni/) · [Higgs TTS cookbook](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html)
- **Higgs optimization roadmap:** [#478](https://github.com/sgl-project/sglang-omni/issues/478)
- **Design background:** *SGLang-Omni: Redesigning the Inference Framework for Multi-Stage Generative Models*