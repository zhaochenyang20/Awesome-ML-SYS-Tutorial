# Higgs Audio v3 TTS on SGLang-Omni: Real-Time, Controllable Speech for Voice Agents

*Boson AI & SGLang-Omni Team*

Today we are announcing end-to-end serving for [**Higgs Audio v3 TTS**](https://www.boson.ai/blog/higgs-audio-v3-tts) on [**SGLang-Omni**](https://github.com/sgl-project/sglang-omni). Higgs Audio v3 TTS is Boson AI's text-to-speech model for conversational voice agents: it generates natural and expressive speech at low latency, supports [100 languages with single-digit WER/CER](https://www.boson.ai/blog/higgs-audio-v3-tts), and lets developers control emotion, style, prosody, and sound effects directly from the input text stream.

For us, serving Higgs is not just about adding one more TTS model. Higgs represents a broader class of generation workloads where the end-to-end path is no longer a single autoregressive decode loop. Instead, generation is split across multiple stages with different compute patterns, latency requirements, and memory behavior. SGLang-Omni is the inference framework we built for exactly this class of multi-stage models.

<iframe
  width="960"
  height="540"
  src="https://www.youtube.com/embed/i2PJeaywDew"
  title="Higgs Audio v3 TTS Demo"
  frameborder="0"
  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
  allowfullscreen
></iframe>

## Meet Higgs Audio v3 TTS

### Designed for Real Conversations

A good conversational TTS model cannot wait for a fully polished paragraph. In a real voice-agent setting, the model may only see half a sentence, or even a few words, before it needs to start speaking. As more text arrives, the generated voice still has to remain coherent in speaker identity, emotion, and pace.

Higgs Audio v3 TTS was designed for that streaming interaction pattern. It can begin synthesis before a full sentence or punctuation mark arrives, then continue as the text stream grows while preserving a stable delivery.

Architecturally, Higgs is a roughly 4B-parameter autoregressive decoder built on a Qwen3-4B backbone. It consumes interleaved text and audio tokens. Audio is encoded by the Higgs Tokenizer into 8 discrete codebooks at 25 fps, staggered with a delayed pattern, mapped into the backbone hidden states through a fused multi-codebook embedding, and decoded back into a 24 kHz waveform through a fused multi-codebook head. Generation alternates between text and audio chunks, so each new audio segment is grounded in both the reference audio and the context generated so far.

### Multilingual Quality

On Boson AI's internal **Higgs-Multilingual** suite, which covers 111 languages and dialects, Higgs Audio v3 TTS reaches [**single-digit WER/CER on 100 languages**](https://www.boson.ai/blog/higgs-audio-v3-tts). On public multilingual voice-cloning benchmarks, v3 also achieves macro-averaged single-digit WER/CER on Seed-TTS, CV3, and MiniMax-Multilingual. Zero-shot voice cloning only needs a short reference clip, and the same reference can be used across languages.

The table below reports WER/CER (↓, %) for zero-shot voice cloning. Each number is macro-averaged over the language set of the corresponding benchmark, using reproducible metrics and normalization.

| Benchmark | Languages | WER/CER ↓ |
|---|---:|---:|
| Seed-TTS | 2 | 1.11 |
| CV3 | 9 | 4.41 |
| MiniMax-Multilingual | 23 | 2.74 |
| Higgs-Multilingual | 111 | 3.61 |

Per-language Seed-TTS breakdowns and WavLM speaker similarity are available in the [SGLang Omni Higgs Cookbook](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html).

### Controlling Delivery from the Text Stream

Higgs Audio v3 TTS is also designed to be controllable. Developers can put control tags directly into the input text to change emotion, switch speaking style, adjust speed and pitch, insert pauses, or trigger sound effects within the same utterance:

```text
<|emotion:amusement|><|prosody:expressive_high|>Wait, wait, that was kind of hilarious. <|sfx:laughter|>Hehe, no, seriously, I was not ready for that.
```

The tag families cover 20+ emotions (`<|emotion:elation|>`, `<|emotion:anger|>`, `<|emotion:sadness|>`, ...), styles (`<|style:singing|>`, `<|style:whispering|>`, `<|style:shouting|>`), prosody (`<|prosody:speed_very_slow|>`, `<|prosody:pitch_high|>`, `<|prosody:pause|>`, `<|prosody:long_pause|>`), and sound effects (`<|sfx:cough|>`, `<|sfx:laughter|>`, `<|sfx:sigh|>`, ...). Tags from different categories can be combined. The full catalogue is in the [SGLang Omni Higgs Cookbook](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html#inline-control-tokens).

## Serving Higgs with SGLang-Omni

Higgs is served and optimized on [**SGLang-Omni**](https://github.com/sgl-project/sglang-omni). Unlike a standard LLM, Higgs and many modern TTS or omni models do not fit naturally into one uniform autoregressive decode loop. Their end-to-end generation path contains multiple stages: some look like standard AR decoding, some are lightweight function-style computation, and some continuously consume chunks and stream audio back.

The goal of SGLang-Omni is to serve this kind of model with a clean runtime structure: each stage is scheduled according to its own compute pattern, stages communicate through low-overhead channels, and GPU placement, process topology, and memory budgets are managed by the framework.

### Multi-stage Decoding with a High-Performance SGLang Backend

Single-stage models already have strong serving paths: autoregressive LLMs are optimized by SGLang main, and diffusion models are supported by SGLang-Diffusion. SGLang-Omni focuses on a different regime: models whose end-to-end generation is split into multiple stages with different compute characteristics. Higgs is one example. Qwen3-Omni's Thinker → Talker → MTP pipeline, Fish Audio S2-Pro's serially nested Dual-AR design, and fully omni-modal models such as Ming-Omni and LLaDA2.0-Uni fall into the same category.

This is why the SGLang-Omni runtime is built around the stage abstraction. A model configuration statically declares the stages in the pipeline, their GPU placement, and the process topology. The placement and topology layers prepare the workers. The Coordinator routes requests between stages. Each Stage acts as an IO shell: it receives data from upstream stages, hands work to its internal Scheduler, and streams outputs to downstream stages.

Different stages can use different schedulers. AR stages, such as the Qwen3-Omni Thinker, usually use `OmniScheduler`, which preserves SGLang's continuous batching, mixed prefill/decode scheduling, KV cache management, tree cache, and CUDA Graph support while adapting them to omni-native request objects and streaming outputs. Non-AR stages, such as small encoders and aggregators, can use `SimpleScheduler`, which is essentially a clear get → forward → put loop. Streaming stages use `StreamingSimpleScheduler` to manage chunk and done lifecycles, such as the Higgs vocoder in streaming mode.

The interface between stages is uniform, but each stage can choose the execution strategy that matches its own compute pattern. To make this practical and fast, we focus on three pieces of infrastructure:

- **Layered communication.** Lightweight control messages, including submit, data-ready, stream, complete, shutdown, and abort, go through a ZMQ/msgpack control plane. Tensor payloads move through the relay data plane, with `shm`, `nccl`, `nixl`, and `mooncake` backends available. Same-process edges can use local dispatch, eligible same-GPU streaming chunks can use CUDA IPC, and cross-process edges keep the same stage-level contract.
- **Process-GPU-stage topology.** Pipelines declare stages, routing, streaming edges, process groups, GPU placement, tensor-parallel size, and optional fused stage groups in config. Non-TP stages explicitly declare their process group. TP stages expand into per-rank processes, with rank 0 owning external stage IO. Compact colocated deployments and larger split/TP deployments are different instances of the same topology description, not separate serving stacks.
- **Memory isolation.** In a multi-stage runtime, GPU memory is a stage-level resource contract rather than one global scheduler fraction. Each GPU-backed stage can declare `runtime.resources.total_gpu_memory_fraction`; placement validation sums budgets per GPU before startup. When multiple process groups share a card, those budgets must be explicit, so one stage cannot silently consume memory reserved for another.

### Reusing Omni-Specific Optimizations

While integrating Higgs, we also pulled recurring omni optimizations into reusable framework modules. Similar compute patterns should not be reimplemented in each model, and performance work should live in the runtime rather than being scattered across model-specific pipelines.

- **CUDA-Graph-friendly feedback runners.** Higgs' `tts_engine` enables CUDA Graph capture by default and uses a model runner designed for the AR + multi-codebook feedback loop. The runner handles static buffer assignment, deferred capture, and extra care around Python-side gather/scatter. The same runner interface also supports one-step-lookahead async decode for Qwen3-Omni, Fish Audio S2-Pro, and other SGLang-Omni models.
- **Streaming vocoder schedulers.** Higgs, Qwen3-Omni, Fish Audio S2-Pro, and related models all need a similar streaming audio lifecycle: initialize per-request state, accumulate incoming code chunks, emit audio windows as soon as enough context is available, flush on `stream_done`, and return a compact final payload to streaming clients. Codec and windowing logic remain model-specific, but the serving lifecycle is shared.

With these abstractions, a new multi-stage model does not need a bespoke pipeline with if-else branches scattered across the codebase. Developers partition the model into scheduling segments, choose the right scheduler and model-runner hooks, declare topology and memory contracts, and let the framework handle routing, streaming, data movement, process placement, and stage-level resource isolation.

### A Growing Multi-Stage Model Ecosystem

Higgs now joins the TTS and omni models already supported by SGLang-Omni:

| Model | Type | Notes |
|---|---|---|
| [Higgs Audio v3 TTS](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b) | TTS | Voice cloning, streaming, 100 languages |
| [Fish Audio S2-Pro](https://huggingface.co/fishaudio/s2-pro) | TTS | Voice cloning, streaming |
| [Voxtral TTS](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) | TTS | Named voices, streaming, 9 languages |
| [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | TTS | Voice cloning, streaming, 10 languages |
| [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | Omni | Text/image/audio/video → text + audio |
| [Ming-flash-omni-2.0](https://huggingface.co/inclusionAI/Ming-flash-omni-2.0) | Omni | Streaming TTS |
| [LLaDA2.0-Uni](https://huggingface.co/inclusionAI/LLaDA2.0-Uni) | Multimodal | Text + image understanding and generation |

These models look different from the outside, but at the inference-system level they share the same underlying problem: how to organize multiple heterogeneous stages into a stable, efficient, and extensible generation pipeline. That is why onboarding Higgs on SGLang-Omni was mostly about declaring its pipeline (`preprocessing → audio_encoder → tts_engine → vocoder`) and adding model-specific hooks, rather than building a serving stack from scratch.

### Optimizing Higgs End-to-End

Beyond the framework abstraction, we also optimized the Higgs pipeline end to end. The main pieces are listed below; implementation details and tracking live in the [Higgs optimization roadmap (#478)](https://github.com/sgl-project/sglang-omni/issues/478) and the [repository](https://github.com/sgl-project/sglang-omni).

- **AR backbone**: [CUDA Graph capture](https://github.com/sgl-project/sglang-omni/pull/503) for the decode loop, [async one-step-lookahead decode](https://github.com/sgl-project/sglang-omni/pull/590) for the omni AR loop, and [batching per-step D2H syncs](https://github.com/sgl-project/sglang-omni/pull/572) into a single transfer.
- **Encoder**: [fusing preprocessing into the encoder stage](https://github.com/sgl-project/sglang-omni/issues/576), an [LRU cache](https://github.com/sgl-project/sglang-omni/pull/563) for [reused reference audio](https://github.com/sgl-project/sglang-omni/pull/605), and a [batched audio encoder](https://github.com/sgl-project/sglang-omni/pull/610).
- **Vocoder**: [batched vocoder decode](https://github.com/sgl-project/sglang-omni/pull/574).
- **Caching**: RadixAttention cache partitioned by reference audio with `extra_key` namespacing, so repeated voice-cloning references can reuse prefix cache.
- **Scheduling and streaming**: [dropping the bespoke scheduler](https://github.com/sgl-project/sglang-omni/pull/476) in favor of the shared `OmniScheduler`, plus real SSE [streaming](https://github.com/sgl-project/sglang-omni/pull/597) [schedulers](https://github.com/sgl-project/sglang-omni/pull/614) to reduce time to first audio.

### Performance

We evaluate Higgs on the full Seed-TTS EN set (**N=1088** per run). The client sweeps `--max-concurrency` against a Higgs server configured with `max_running_requests=16`, bf16, and CUDA Graph enabled. Each row reports the **mean of 3 runs** on **1× H100**.

| Concurrency | Throughput (req/s) | Mean latency | RTF (per-req) | audio_s/s |
|---:|---:|---:|---:|---:|
| 1 | 1.62 | 617 ms | 0.147 | 6.89 |
| 2 | 2.70 | 742 ms | 0.180 | 11.37 |
| 4 | 5.45 | 733 ms | 0.177 | 22.84 |
| 8 | 8.91 | 898 ms | 0.217 | 37.38 |
| 16 | 14.74 | 1079 ms | 0.262 | 61.84 |

- **Concurrency**: Maximum number of in-flight client requests (`--max-concurrency`).
- **Throughput (req/s)**: Completed requests divided by total benchmark wall-clock time.
- **Mean latency**: Average end-to-end time per request, from sending the request to receiving the full response.
- **RTF (per-req)**: Average ratio of processing time to generated audio duration per request. Values below 1 are faster than real time.
- **audio_s/s**: Total seconds of audio produced divided by total benchmark wall-clock time.

To reproduce the results, follow the [benchmark script](https://github.com/sgl-project/sglang-omni/blob/main/benchmarks/eval/benchmark_tts_seedtts.py).

## Try it Yourself

Detailed instructions are in the [SGLang Omni Higgs Cookbook](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html). The commands below show the shortest path to a working setup.

### Install and Serve

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

For voice cloning, we recommend providing both the reference audio and the reference transcript (`text`), which usually improves cloning quality:

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

Reference input:

<audio controls>
  <source src="https://sgl-project.github.io/sglang-omni/_static/audio/male-voice.wav" type="audio/wav">
</audio>

Reference output:

<audio controls>
  <source src="https://sgl-project.github.io/sglang-omni/_static/audio/higgs-2.wav" type="audio/wav">
</audio>

### Streaming

Set `"stream": true` to receive audio over [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events). The client can start playback before the full generation finishes because the vocoder emits incremental WAV chunks. The `-N` flag disables curl's output buffering so SSE events print as they arrive:

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

For raw PCM streaming (no SSE JSON), see the [Higgs TTS cookbook](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html#streaming).

Reference output:

<audio controls>
  <source src="https://sgl-project.github.io/sglang-omni/_static/audio/higgs-4.wav" type="audio/wav">
</audio>

### Inline Control Tokens

Control tokens can be embedded directly in the `input` field, and tokens from different categories can be combined. In general, put delivery tokens such as emotion, style, speed, pitch, or expressive prosody at the beginning of each turn; place `<|prosody:pause|>` / `<|prosody:long_pause|>` where the pause should happen; and pair each `<|sfx:…|>` with matching onomatopoeia right after it. The full catalogue is in the [cookbook](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html#inline-control-tokens).

**Emotion: amusement + laughter**

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

Reference output:

<audio controls>
  <source src="https://sgl-project.github.io/sglang-omni/_static/audio/control-tokens-test1.wav" type="audio/wav">
</audio>

**Emotion: anger + shouting**

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

Reference output:

<audio controls>
  <source src="https://sgl-project.github.io/sglang-omni/_static/audio/control-tokens-test2.wav" type="audio/wav">
</audio>

**Emotion: surprise + screaming**

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
Reference output:

<audio controls>
  <source src="https://sgl-project.github.io/sglang-omni/_static/audio/control-tokens-test5.wav" type="audio/wav">
</audio>

**Combined example:**

The example below combines emotion, sound effects, and prosody tokens in a short Gaokao-style English listening dialogue between two speakers:

<details>
<summary>Commands</summary>

Part 1 — she asks about the missed class:

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

Part 2 — he explains what was covered:

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

Part 3 — she thanks him:

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

You can also launch the backend and browser UI with one command:

```bash
CUDA_VISIBLE_DEVICES=0 ./playground/higgs/start.sh
```

> **TODO [ DEMO VIDEO PLACEHOLDER ] @yichi**

## Roadmap

For SGLang-Omni, serving Higgs end to end is an important milestone, but it is not the finish line. We are continuing to push on several tracks:

- **Tracking upstream SGLang** ([#658](https://github.com/sgl-project/sglang-omni/issues/658)): moving to the latest SGLang so AR backbones continue to inherit improvements from mainline SGLang, including CUDA/PyTorch build updates, kernel improvements, scheduling, and speculative decoding.
- **Per-model refactor** ([#661](https://github.com/sgl-project/sglang-omni/issues/661)): continuing the direction of [RFC #188](https://github.com/sgl-project/sglang-omni/issues/188) with a cleaner per-model abstraction. We want new-model integration to look more like "declare topology and plug in hooks" than adding special branches across the framework.
- **End-to-end RL** ([#663](https://github.com/sgl-project/sglang-omni/issues/663)): using SGLang-Omni as a high-throughput rollout backend for omni and TTS models with explicit reward targets, further connecting serving and post-training.

Cross-node multi-stage pipelines and fuller diffusion-stage support are also in progress. With stage abstraction, a unified scheduler interface, layered communication, and cross-stage memory budgeting already in place, these capabilities can grow within the same framework instead of requiring another serving stack.

## Join us

SGLang-Omni is still moving quickly. We want it to become a general inference foundation for multi-stage generative models: new models should not need a serving stack from scratch, nor special-case logic scattered across a dozen files. They should be expressible as clear stages, topology declarations, and model-specific hooks, with scheduling, communication, memory management, and streaming handled by the framework.

If you are interested in multi-stage inference, TTS, omni models, multimodal generation, inference systems, or RL rollout backends, we would love to work with you. Whether your strength is kernels, scheduling, communication, model onboarding, or benchmarking, contributions and discussions are welcome.

## Acknowledgments

**SGLang-Omni** — Haoguang Cai, Shangming Cai, Qiujiang Chen, Jiaxin Deng, Wenyao Gao, Yifei Gao, Jingwen Gu, Yitong Guan, Chenchen Hong, Hao Jin, Xinli Jing, Shenggui Li, Junrong Lin, Xinyu Lu, Yuan Luo, Ratish Palanisamy, Mick Qian, JinTao Qu, Shuai Shi, Chao Wang, Richard Wang, Shuwen Wang, Zijie Xia, Yuhao Yang, Xuesong Ye, Yin Yue, Fan Yin, Gaokai Zhang, Xiaoyu Zhang, Yichi Zhang, Chenyang Zhao.

**Higgs Audio v3 TTS (Boson AI)** — Mu Li, Alex Smola, Lindsey Allen. Silin Meng, Ke Bai. Ruskin Raj Manku, Huapeng Zhou, Dongming Shen, Jonah Mackey, Erik Li, Weisu Yin, Yizhi Liu, Xinyu Wang, Hao Yu.

## Learn More

- **Model:** [`bosonai/higgs-audio-v3-tts-4b`](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b)
- **Blog:** [Higgs Audio v3 TTS](https://www.boson.ai/blog/higgs-audio-v3-tts)
- **Serving framework:** [SGLang-Omni on GitHub](https://github.com/sgl-project/sglang-omni)
- **Documentation:** [SGLang-Omni docs](https://sgl-project.github.io/sglang-omni/) · [Higgs TTS cookbook](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html)
- **Higgs optimization roadmap:** [#478](https://github.com/sgl-project/sglang-omni/issues/478)
- **Design background:** *SGLang-Omni: Redesigning the Inference Framework for Multi-Stage Generative Models*
