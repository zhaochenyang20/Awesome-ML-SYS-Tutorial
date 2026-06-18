# MOSS-TTS Local Transformer v1.5 on SGLang-Omni: High-Fidelity Native-Streaming Speech Generation

OpenMOSS Team, MOSI.AI & SGLang-Omni Team

Today we are announcing end-to-end serving for **MOSS-TTS-Local-Transformer-v1.5** on **SGLang-Omni** working with MOSI.AI, OpenMOSS team and SGLang-Omni team.

MOSS-TTS-Local-Transformer-v1.5 is an open text-to-speech model that brings 48 kHz stereo generation, zero-shot voice cloning, long-form synthesis, multilingual speech, and native streaming into a single Local Transformer architecture. It has a distinctive inference shape. Text and reference audio are first converted into a multi-channel prompt. A Qwen3-4B global transformer advances the sequence frame by frame. A local transformer expands each global frame into 12 RVQ audio codebooks. A stateful MOSS-Audio-Tokenizer-v2 decoder then turns those generated codes into streaming 48 kHz stereo waveform chunks.

That path is model-native and quality-driven, but it is also demanding for inference systems. It mixes neural reference encoding, backbone autoregressive decoding, frame-local codebook sampling, and stateful vocoder execution in one request lifecycle. The serving runtime must batch what can be batched, stream what should be streamed, keep per-request state alive across stages, and protect GPU memory for both the AR engine and the codec.

SGLang-Omni handles this by serving MOSS as a multi-stage pipeline rather than forcing it into a single LLM decode loop. On top of the stage runtime, we added MOSS-specific optimizations across the full path: reference-audio caching, batched codec encoding, CUDA-Graph-friendly frame decoding, GPU-native cache keys, compiled seeded sampling, native streaming vocoder scheduling, vocoder CUDA Graph replay, and colocated memory budgeting. The result is a production-ready OpenAI-compatible speech API for one of the most capable open TTS models available today.

## Meet MOSS-TTS Local Transformer v1.5

MOSS-TTS-Local-Transformer-v1.5 is the second flagship model in the MOSS-TTS v1.5 family. It continues the Audio Tokenizer + LLM autoregressive paradigm and upgrades the audio codec, backbone architecture, training scale, and streaming path for high-fidelity speech generation.

The model is designed around a simple goal: produce natural, long, multilingual speech while preserving the voice identity and acoustic detail needed for realistic zero-shot cloning. It supports direct text-to-speech generation, voice cloning from a short reference clip, continuation, duration control, explicit pause control such as `[pause 3.2s]`, and long-form generation up to 10 minutes. It covers 31 major world languages and was trained on roughly 4 million hours of multilingual speech data.

![MOSS-TTS Local Transformer v1.5 model architecture](images/moss-local-transformer-arch.png)

At the audio interface, MOSS-TTS Local Transformer v1.5 uses **MOSS-Audio-Tokenizer-v2**, a high-quality neural audio tokenizer with an encoder and decoder totaling about 2B parameters. The tokenizer runs at a 12.5 Hz frame rate, supports variable bitrate compression from 0.125 kbps to 4 kbps, reconstructs 48 kHz stereo audio, and represents speech with residual vector quantization (RVQ). This gives the language-model side of the system a discrete acoustic space that carries both semantic and fine-grained audio information.

At the generation core, the model uses a **Qwen3-4B backbone** with a **Global Transformer + Local Transformer** architecture. The global transformer models text semantics, multilingual context, speaker identity, and prosody from the prompt and reference audio. The local transformer then generates acoustic codes inside each frame. The Local variant uses the first 12 RVQ layers at 12.5 Hz and one local transformer layer, forming a pure autoregressive path that is naturally streamable at the frame level.

Each sequence position is a multi-channel row rather than a scalar token. The MOSS Local layout is `[T, 13]`: one text/control channel and 12 audio codebook channels. Text positions carry a text token in channel 0 and audio padding in the remaining channels. Audio-frame positions carry a slot/control token in channel 0 and one RVQ code from each of the 12 audio codebooks. This representation is central to both model quality and serving complexity: it lets the model reason over text and audio in a unified autoregressive stream, while requiring the inference engine to handle multi-channel embeddings, frame-local sampling, and feedback into the next frame.

On public evaluation sets, MOSS-TTS Local Transformer v1.5 achieves strong multilingual and voice-cloning quality:

| Benchmark | WER (lower is better) | SIM (higher is better) |
|---|---:|---:|
| Seed-TTS-Eval | 5.10% | 69.23% |
| CV3-Eval | 7.48% | 61.59% |
| MiniMax Multilingual | 6.37% | 75.31% |
| X Voice | 20.48% | 63.00% |

These are model-level offline evaluation results. The serving benchmarks later in this post use a different evaluation pipeline and should be read as end-to-end system measurements.

MOSS-TTS Local Transformer v1.5 was trained at thousand-card scale on Alibaba Cloud's PPU-ZW810 cluster. The serving work described below focuses on making this model fast, stable, streamable, and easy to deploy through SGLang-Omni.

## Why MOSS Needs a Multi-Stage Serving Runtime

A standard LLM serving engine is usually optimized around one hidden assumption: generation is one repeated model loop. MOSS-TTS Local Transformer v1.5 has a different shape. A request passes through three heterogeneous stages:

1. **Preprocessing and reference encoding:** text is tokenized, reference audio is loaded, and the reference waveform is encoded into RVQ codes.
2. **Autoregressive TTS engine:** a Qwen3 backbone and local transformer generate multi-channel frame rows one frame at a time.
3. **Streaming vocoder:** generated RVQ rows are decoded by a stateful MOSS codec decoder into 48 kHz waveform chunks.

![MOSS-TTS Local inference pipeline](images/tts-opt-moss-pipeline.png)

The important point is that each stage is computationally meaningful. Reference encoding runs a neural codec encoder; AR generation is not one scalar next-token step but a frame-level loop with 12 sequential codebook samples; the vocoder is a neural decoder that must preserve streaming state across chunks. This creates a mix of memory-bound backbone work, launch-bound local sampling, stateful codec execution, and cross-stage GPU memory pressure.

SGLang-Omni is designed for exactly this regime: heterogeneous stages, stage-specific schedulers, streaming intermediate data, and resource isolation across a multi-model generation path.

## Serving MOSS with SGLang-Omni

SGLang-Omni serves MOSS-TTS Local Transformer v1.5 as a three-stage pipeline:

```text
preprocessing -> tts_engine -> vocoder
```

The **preprocessing** stage parses the OpenAI-compatible speech request, prepares the multi-channel prompt, and encodes reference audio when voice cloning is used. The **tts_engine** stage is backed by `OmniScheduler`, inheriting SGLang's continuous batching, KV cache management, RadixAttention, and CUDA Graph support while adapting them to MOSS's `[T, 13]` request format. The **vocoder** stage consumes generated rows as a stream and returns audio chunks from a persistent codec streaming session.

This layout follows the model's natural computation graph. It also gives every optimization a clear home: encoder caching belongs in preprocessing, continuous batching and KV reuse belong in the AR engine, slot management belongs in the vocoder, and memory budgeting belongs at stage boundaries. For users, the complexity is hidden behind `/v1/audio/speech`, with support for synthesis, voice cloning, streaming PCM, duration control, pause markup, language hints, style instructions, seeds, and sampling parameters.

## Optimizing MOSS End-to-End

[TODO: Yichi xinyu please help to review this part]

Once the pipeline was functionally complete, we optimized it stage by stage. The guiding principle was simple: each bottleneck should be solved where it naturally appears, and each optimization should preserve a clear fallback path for production.

| Stage | Optimization | Main Benefit |
|---|---|---|
| Preprocessing | Batched reference encoding | Amortizes codec encoder cost across concurrent requests |
| Preprocessing | Content-addressed LRU cache + single-flight | Removes repeated reference encoding and avoids cold-start duplication |
| AR engine | Backbone CUDA Graph + frame-decode CUDA Graph | Removes launch overhead from both backbone decode and local codebook micro-loop |
| AR engine | Decode state pool | Provides fixed-address GPU state for graph replay and request lifecycle management |
| AR engine | GPU-native Radix row hash | Removes per-frame CPU hashing and D2H synchronization |
| AR engine | Compiled seeded sampler | Fuses the hot sampling path while preserving deterministic per-request sampling |
| Vocoder | Stateful streaming session + slot management | Enables frame-level audio streaming with request isolation |
| Vocoder | Dual-threshold coalesced steps | Balances time to first audio and steady-state throughput |
| Vocoder | Vocoder CUDA Graph | Speeds up short streaming decode steps |
| Cross-stage | Explicit memory budgeting | Prevents codec and AR memory pressure from interfering with each other |

### Reference Audio Encoding

Voice cloning workloads often reuse the same speakers across many prompts. In MOSS, that pattern matters because reference encoding runs a neural codec encoder before AR generation can begin.

![Reference audio cache:](images/tts-opt-encoder-cache.png)

SGLang-Omni combines batched reference encoding with a content-addressed LRU cache. Concurrent references can be encoded together, and repeated references are keyed by audio content rather than by path, so copied or renamed files still reuse the same encoded RVQ result. A single-flight path also merges concurrent misses for the same speaker, preventing a cold-cache burst from launching duplicate codec encodes.

In SeedTTS English evaluation on 2x H100 at concurrency 16, increasing the reference cache capacity from 256 to 1024 entries improved throughput by **32.0%** and reduced mean latency by **24.3%**. The memory cost was small because encoded code tensors are compact; the larger cache mainly prevents eviction of the active speaker working set.

### AR Engine

The MOSS AR engine has two levels of computation: the Qwen3 backbone and the local transformer frame-decode loop. SGLang-Omni captures both with CUDA Graphs, but keeps them separate because they have different structure and ownership.

![CUDA Graph execution](images/tts-opt-cuda-graph.png)

The backbone graph uses SGLang's standard CUDA Graph path for causal LM decode. The MOSS-specific frame graph captures the local transformer micro-loop for a full frame, including stop/continue sampling, 12 sequential codebook projections, codebook feedback, and feedback embedding assembly for the next frame. This removes the launch overhead that otherwise dominates a small but highly sequential local loop.

To make graph replay possible, MOSS keeps per-request decode state in a persistent GPU-side pool. Feedback embeddings, sampling parameters, seeds, counters, and audio history live at stable addresses across frames. SGLang-Omni also moves the generated-row radix hash to the GPU, avoiding a per-frame CPU hash and D2H synchronization. Finally, the 13 per-frame sampling operations use a seeded GPU sampler and a narrow compile scope to fuse the hot sampling path without changing the backbone or local transformer execution.

On SeedTTS English evaluation at concurrency 16, the compiled seeded sampler improved throughput by **12.3%**, reduced mean latency by **11.1%**, and reduced mean RTF by **10.5%**. The narrow compile scope was intentional: it targets a real hot path while avoiding broad compile-time overhead and graph changes in the backbone or local transformer.

### Streaming Vocoder

The vocoder stage turns generated RVQ frames into audio chunks. Because MOSS-Audio-Tokenizer-v2 supports stateful streaming decode, SGLang-Omni keeps a persistent codec streaming session inside the vocoder executor.

The scheduler manages stream slots, an offline fallback slot, chunk thresholds, and coalesced decode steps. The first chunk can use a small threshold to reduce time to first audio, while later chunks use larger windows to improve throughput. When several requests have enough pending frames, the scheduler decodes them together in one codec call.

Short streaming chunks are launch-heavy, so SGLang-Omni also captures common vocoder frame counts with CUDA Graphs. The implementation keeps codec state buffers at stable addresses and updates them in place, allowing graph replay across streaming steps.

The speedup is largest for short streaming chunks:

| Frames per Step | Eager | CUDA Graph | Speedup |
|---:|---:|---:|---:|
| 4 | 66.3 ms | 30.1 ms | 2.20x |
| 5 | 65.8 ms | 30.7 ms | 2.14x |
| 8 | 65.6 ms | 34.0 ms | 1.93x |
| 13 | 65.4 ms | 40.4 ms | 1.62x |
| 25 | 74.8 ms | 58.3 ms | 1.28x |
| 100 | 222.9 ms | 215.3 ms | 1.04x |

The graph path has safe fallback to eager decode and is covered by streaming/non-streaming consistency checks.

### Memory Budgeting

Compact deployment is important for users who want the simplest path from model download to serving. In the default MOSS Local config, preprocessing, AR generation, and vocoder execution can be colocated on one GPU. SGLang-Omni therefore gives the AR engine an explicit colocated memory contract and reserves headroom for codec runtime allocations and streaming state.

This is a small change with visible impact. In a single-card colocated configuration at concurrency 8, explicit codec memory budgeting improved throughput by **8.9%** and reduced mean RTF by **8.4%**. More importantly, it makes the deployment behavior more predictable under memory pressure.

## Performance

We evaluate the optimized MOSS-TTS Local Transformer v1.5 serving path on the SeedTTS English set with 1088 samples. The results below come from the full CI evaluation after the vocoder CUDA Graph path was enabled, using 2x GPU and client concurrency 16. ASR scoring uses Qwen3-ASR-1.7B, and speaker similarity uses WavLM-Large finetune.

| Mode | Completed / Failed | Throughput | Audio Throughput | Mean Latency | Mean RTF | WER |
|---|---:|---:|---:|---:|---:|---:|
| Non-streaming | 1088 / 0 | 5.976 req/s | 26.303 audio s/s | 2.669 s | 0.644 | 1.75% |
| Streaming | 1088 / 0 | 2.909 req/s | 12.804 audio s/s | 5.474 s | 1.322 | 2.14% |

For non-streaming requests, the system reaches **5.976 requests per second** at concurrency 16 with a mean RTF of **0.644**. This mode is the best fit for throughput-oriented workloads where the client receives the final WAV after generation finishes.

For streaming requests, the system emits incremental audio chunks through the vocoder stage. At concurrency 16, the average inter-chunk interval is **0.109 seconds**, and the average number of audio chunks per request is **8.82**. The streaming path trades some high-concurrency throughput for incremental delivery because the vocoder runs more frequently on smaller chunks and shares GPU time with the AR engine. This is an inherent latency-throughput trade-off in streaming TTS serving, and SGLang-Omni exposes the scheduler controls needed to continue improving that balance.

The quality numbers remain stable across serving modes. In the same CI evaluation, non-streaming WER is **1.75%**, and streaming WER is **2.14%**. Streaming and non-streaming artifact consistency checks pass, giving us confidence that the streaming scheduler and CUDA Graph paths preserve the model's audio semantics.

The individual optimization measurements should not be summed into one headline speedup because they were collected under different hardware and concurrency settings. They are more useful as a map of where the system spends time: reference caching removes redundant encoder work, frame CUDA Graphs remove local-loop launch overhead, sampler compilation improves the hot sampling path, vocoder CUDA Graphs accelerate short streaming chunks, and memory budgeting stabilizes colocated deployment.

## Try It Yourself

Detailed instructions are available in the [SGLang-Omni MOSS-TTS-Local cookbook](https://sgl-project.github.io/sglang-omni/cookbook/moss_tts_local.html). The commands below show the shortest path from a clean container to a working speech endpoint, followed by the most common request patterns.

### Install and Serve

```bash
docker pull lmsysorg/sglang-omni:dev
docker run -it --gpus all --shm-size 32g --ipc host --network host --privileged \
  lmsysorg/sglang-omni:dev /bin/zsh

git clone git@github.com:sgl-project/sglang-omni.git
cd sglang-omni
uv venv .venv -p 3.12
source .venv/bin/activate
uv pip install -v -e .

hf download OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5

sgl-omni serve \
  --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
  --port 8000
```

The default server layout colocates the AR backbone and codec/vocoder on one GPU. A matching explicit config is also available at `examples/configs/moss_tts_local.yaml` for users who want to inspect or customize the pipeline topology.

### Zero-Shot Synthesis

MOSS-TTS-Local can synthesize speech without a reference clip. The response is a WAV file by default:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "SGLang-Omni is a great project for high-fidelity speech generation."}' \
  --output output.wav
```

### Voice Cloning

For voice cloning, provide a reference audio clip and its transcript. The `references` field accepts `audio_path` as a local path readable by the server, an HTTP(S) URL, or a base64 data URI. Supplying the transcript usually improves speaker similarity:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }]
  }' \
  --output output.wav
```

The shorthand fields `ref_audio` and `ref_text` are also accepted.

#### Python

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": "Get the trust fund to the bank early.",
        "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
        "ref_text": "We asked over twenty different people, and they all said it was his.",
    },
)
response.raise_for_status()

with open("output.wav", "wb") as output_file:
    output_file.write(response.content)
```

### Reference Audio Sources

Reference audio can be sent as a local file path, a URL, or an inline base64 data URI. Data URIs are useful when the client owns the audio bytes and does not want to expose a separate file server:

```python
import base64

import requests

reference_url = "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav"
reference_response = requests.get(reference_url)
reference_response.raise_for_status()

reference_audio = (
    "data:audio/wav;base64,"
    + base64.b64encode(reference_response.content).decode("ascii")
)

response = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": "SGLang-Omni is a great project!",
        "ref_audio": reference_audio,
        "ref_text": "We asked over twenty different people, and they all said it was his.",
    },
)
response.raise_for_status()

with open("output_data_uri.wav", "wb") as output_file:
    output_file.write(response.content)
```

The server caches and coalesces reference encodes. Reusing the same reference clip can skip codec re-encoding, which is especially useful for fixed speaker pools.

### Streaming

Set `"stream": true` and `"response_format": "pcm"` to receive raw 48 kHz PCM chunks as they are produced. Pipe the stream through `ffmpeg` when you want a playable WAV file:

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his.",
    "stream": true,
    "response_format": "pcm"
  }' \
  | ffmpeg -f s16le -ar 48000 -ac 1 -i pipe:0 output_stream.wav
```

### Duration Control

MOSS-TTS-Local can condition on a target duration token count. The count is measured in codec frames; a larger count usually yields longer audio. You can set it with an inline `${token:N}` prefix or with the `token_count` field:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "${token:150}A sentence with an explicit duration target.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his."
  }' \
  --output output_duration_tokens.wav
```

The duration token count is a control hint rather than an exact wall-clock duration. It is useful for making generated clips shorter or longer while preserving the model's natural pacing.

### Pronunciation, Style, and Language Hints

Inline markup that the model understands is passed through unchanged. This includes pause markers such as `[pause 0.5s]`, as well as pronunciation controls such as Pinyin and IPA. The optional `language` field guides multilingual generation, and `instructions` carries a free-text style directive:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Today we are serving MOSS-TTS Local Transformer v1.5 on SGLang-Omni. [pause 0.5s] The model supports high-fidelity native streaming speech.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his.",
    "language": "English",
    "instructions": "Use a natural conversational style."
  }' \
  --output output_markup.wav
```

### Generation Parameters

MOSS-TTS-Local exposes the usual speech-generation controls through the OpenAI-compatible request body:

| Parameter | Notes |
|---|---|
| `input` | Text to synthesize; may include a `${token:N}` duration prefix and inline markup |
| `references` | Reference clips for cloning; each item has `audio_path` and `text` |
| `ref_audio` / `ref_text` | Shorthand for `references[0].audio_path` and `references[0].text` |
| `stream` | Set to `true` for streaming output |
| `response_format` | Use `pcm` with streaming raw chunks |
| `language` | Optional target-language hint |
| `instructions` | Optional free-text style directive |
| `token_count` / `duration_tokens` | Target duration in codec frames |
| `max_new_tokens` | Maximum generated frames |
| `temperature`, `top_p`, `top_k` | Sampling controls; single values apply to both text and audio channels |
| `repetition_penalty` | Audio repetition penalty |
| `seed` | Non-negative integer for reproducible sampling on a fixed server configuration |

The model has separate text-channel and audio-channel sampling defaults. A single `temperature`, `top_p`, or `top_k` applies to both; channel-specific fields can be used when more control is needed.

### Benchmarking And Performance

[TODO: Yichi please help to review this part]

To reproduce SeedTTS-style serving measurements, run the benchmark client against the local server:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
  --port 8000 \
  --ref-format references \
  --token-count auto \
  --output-dir results/moss_tts_en \
  --lang en \
  --max-concurrency 16
```

## Roadmap

【TODO：Jiaxin Please help to change】

Serving MOSS-TTS Local Transformer v1.5 is an important step for SGLang-Omni, and we are continuing to push the system in several directions.

**Pool-native frame CUDA Graph.** The current frame-decode graph already uses persistent state pools, but some staging remains around sampling parameters and generated rows. A more native pool-to-pool graph path can further reduce host-device movement and simplify the launch/resolve boundary.

**Adaptive streaming scheduling.** Streaming TTS has a real latency-throughput trade-off. We are exploring load-aware chunk sizing, priority-aware slot scheduling, and better coalescing policies so low-load requests receive fast first audio while high-load deployments recover more throughput.

**Broader compilation coverage.** The codec encoder and Qwen3 backbone still have room for targeted compilation experiments. We will continue to evaluate compile scope carefully, prioritizing steady-state gains that do not damage cold-start latency or output consistency.

**Wider benchmark coverage.** Current measurements focus on SeedTTS English in CI. We plan to expand coverage to Chinese, multilingual evaluation, long-form generation, multiple speaker pools, different reference lengths, and production-like traffic mixes.

**General multi-stage model onboarding.** The long-term goal of SGLang-Omni is to make new TTS and omni models easy to serve without building a custom inference stack each time. A model should be expressed as stages, topology, memory contracts, and model-specific hooks, while scheduling, communication, streaming, placement, and resource isolation stay in the framework.

## Join Us

SGLang-Omni is moving quickly toward a general inference foundation for multi-stage generative models. TTS models like MOSS show why this direction matters: model quality increasingly comes from heterogeneous generation paths, and inference systems need to handle those paths directly rather than forcing them into a single-loop abstraction.

If you are interested in TTS, omni models, streaming inference, CUDA Graphs, scheduling, communication, model onboarding, benchmarking, or production serving, we would love to work with you. Contributions, issues, discussions, and new model integrations are welcome.

## Acknowledgments

【Yichi：Please help to change】

**SGLang-Omni** - Haoguang Cai, Shangming Cai, Qiujiang Chen, Yuhao Chen, Jiaxin Deng, Wenyao Gao, Yifei Gao, Jingwen Gu, Yitong Guan, Zhihao Guo, Chenchen Hong, Hao Jin, Xinli Jing, Xiangrui Ke, Shenggui Li, Junrong Lin, Estella Liu, Xinyu Lu, Yuan Luo, Ratish Palanisamy, Mick Qian, JinTao Qu, Shuai Shi, Yijiang Tian, Chao Wang, Richard Wang, Shuwen Wang, Zijie Xia, Yuhao Yang, Xuesong Ye, Fan Yin, Yue Yin, Gaokai Zhang, Xiaoyu Zhang, Yichi Zhang, Chenyang Zhao.

**MOSS-TTS Local Transformer v1.5** - Yitian Gong, Kuangwei Chen, Zhicheng Zhang, Botian Jiang, Yiyang Zhang, Kang Yu, Yang Gao, Xiaogui Yang, Qinyuan Chen, Zhaoye Fei, Shimin Li, Xipeng Qiu.

## Learn More

- **Model:** [OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5)
- **Serving framework:** [SGLang-Omni on GitHub](https://github.com/sgl-project/sglang-omni)
- **Documentation:** [SGLang-Omni docs](https://sgl-project.github.io/sglang-omni/)
- **MOSS-TTS-Local cookbook:** [MOSS-TTS-Local in SGLang-Omni](https://sgl-project.github.io/sglang-omni/cookbook/moss_tts_local.html)
- **MOSS optimization roadmap:** [#637](https://github.com/sgl-project/sglang-omni/issues/637)
- **Design background:** [SGLang-Omni: Redesigning the Inference Framework for Multi-Stage Generative Models](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/sglang-omni/why-sglang-omni-en.md)
