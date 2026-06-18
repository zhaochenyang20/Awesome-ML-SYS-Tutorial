# MOSS-TTS Local Transformer v1.5 on SGLang-Omni: High-Fidelity Native-Streaming Speech Generation

OpenMOSS Team, MOSI.AI & SGLang-Omni Team

OpenMOSS and MOSI.AI recently released **MOSS-TTS-Local-Transformer-v1.5**, an open text-to-speech model that brings 48 kHz stereo generation, zero-shot voice cloning, long-form synthesis, multilingual speech, and native streaming into a single Local Transformer architecture. Together with the OpenMOSS team, we are making this model available through **SGLang-Omni** as an end-to-end high-performance serving stack.

MOSS-TTS Local Transformer v1.5 has a distinctive inference shape. Text and reference audio are first converted into a multi-channel prompt. A Qwen3-4B global transformer advances the sequence frame by frame. A local transformer expands each global frame into 12 RVQ audio codebooks. A stateful MOSS-Audio-Tokenizer-v2 decoder then turns those generated codes into streaming 48 kHz stereo waveform chunks.

That path is model-native and quality-driven, but it is also demanding for inference systems. It mixes neural reference encoding, backbone autoregressive decoding, frame-local codebook sampling, and stateful vocoder execution in one request lifecycle. The serving runtime must batch what can be batched, stream what should be streamed, keep per-request state alive across stages, and protect GPU memory for both the AR engine and the codec.

SGLang-Omni handles this by serving MOSS as a multi-stage pipeline rather than forcing it into a single LLM decode loop. On top of the stage runtime, we added MOSS-specific optimizations across the full path: reference-audio caching, batched codec encoding, CUDA-Graph-friendly frame decoding, GPU-native cache keys, compiled seeded sampling, native streaming vocoder scheduling, vocoder CUDA Graph replay, and colocated memory budgeting. The result is a production-ready OpenAI-compatible speech API for one of the most capable open TTS models available today.

## Meet MOSS-TTS Local Transformer v1.5

MOSS-TTS-Local-Transformer-v1.5 is the second flagship model in the MOSS-TTS v1.5 family. It continues the Audio Tokenizer + LLM autoregressive paradigm and upgrades the audio codec, backbone architecture, training scale, and streaming path for high-fidelity speech generation.

The model is designed around a simple goal: produce natural, long, multilingual speech while preserving the voice identity and acoustic detail needed for realistic zero-shot cloning. It supports direct text-to-speech generation, voice cloning from a short reference clip, continuation, duration control, explicit pause control such as `[pause 3.2s]`, and long-form generation up to 10 minutes. It covers 31 major world languages and was trained on roughly 4 million hours of multilingual speech data.

![MOSS-TTS Local Transformer v1.5 model architecture: text and reference audio are tokenized into a multi-channel sequence, processed by a decoder-only global transformer, expanded frame by frame through local transformers, and decoded back into waveform by the audio detokenizer.](images/moss-local-transformer-arch.png)

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

A standard LLM serving engine is usually optimized around one hidden assumption: generation is a single model loop. A request is tokenized, prefilling builds KV cache, decoding repeatedly runs the same transformer, and each step emits one next token. The serving system can focus on one model's KV cache, one scheduler, one attention path, and one token stream.

MOSS-TTS Local Transformer v1.5 has a different shape. Its end-to-end path is a heterogeneous pipeline:

1. **Preprocessing and reference encoding.** Text is tokenized, reference audio is loaded, and the reference waveform is encoded by the MOSS codec encoder into RVQ codes.
2. **Autoregressive TTS engine.** A Qwen3 backbone and local transformer generate multi-channel frame rows, one audio frame at a time.
3. **Streaming vocoder.** The generated RVQ rows are consumed by a stateful MOSS codec decoder and incrementally converted into 48 kHz stereo waveform chunks.

![MOSS-TTS Local inference pipeline: text and reference audio form a multi-channel prompt, the decoder-only LLM produces frame-level globals, local transformers generate RVQ blocks, and the audio detokenizer reconstructs speech.](images/tts-opt-moss-pipeline.png)

Each stage carries real compute. Reference audio encoding is not a lightweight preprocessing step: the codec encoder is a neural model, and a single reference clip can take on the order of hundreds of milliseconds to encode. The AR engine is not a vanilla next-token loop: each generated frame requires a backbone step and then a local codebook micro-loop. The vocoder is not a formatting step: it is another neural model that maintains streaming state and produces audio chunks as soon as enough frames are available.

The AR engine is the most unusual part from a serving perspective. For each frame, MOSS first fuses the 13-channel input row into a single hidden vector. The backbone runs once and returns the global context for that frame. The local transformer then performs a stop/continue decision and sequentially samples 12 RVQ codebook tokens. After each sampled codebook token, its embedding is fed back into the local transformer so the next codebook can condition on the previous ones. The sampled 12-codebook row is then fused into a feedback embedding and used as the next frame's backbone input.

This means a single frame involves one backbone forward, a local transformer micro-loop, 13 sampling operations, 12 audio embedding lookups, output-head projections, feedback assembly, scheduler bookkeeping, and downstream streaming. The local loop is small in FLOPs but rich in kernel launches and sequential dependencies. If executed eagerly with Python orchestration, launch overhead quickly becomes a first-order bottleneck.

The vocoder adds another serving dimension: stateful streaming. MOSS-Audio-Tokenizer-v2 can decode frames incrementally. SGLang-Omni needs to hold a persistent streaming codec session, assign slots to active requests, decide when enough frames have accumulated for an audio chunk, batch compatible slots together, flush remaining frames when a request finishes, and release resources without disturbing other streams. Streaming is not just an HTTP feature here; it is an execution mode of the codec decoder.

Finally, the stages share GPU memory. In compact deployments, the reference encoder, AR backbone, local transformer, KV cache, decoder state, vocoder activations, and streaming buffers can live on the same GPU. A single global memory fraction is too coarse: the AR engine and codec need explicit resource contracts so one stage does not silently starve another.

These are exactly the conditions SGLang-Omni is meant to handle: heterogeneous stages, stage-specific schedulers, streaming intermediate data, and resource isolation across a multi-model generation path.

## Serving MOSS with SGLang-Omni

SGLang-Omni serves MOSS-TTS Local Transformer v1.5 as a three-stage pipeline:

```text
preprocessing -> tts_engine -> vocoder
```

The **preprocessing** stage receives the OpenAI-compatible speech request. It tokenizes the input text, parses control fields such as reference audio, reference transcript, language hints, style instructions, duration token targets, and inline markup, then prepares the multi-channel prompt for the AR engine. For voice cloning, it runs the MOSS codec encoder on the reference audio and inserts the resulting RVQ rows into the prompt.

The **tts_engine** stage is the autoregressive core. It is backed by SGLang's high-performance scheduler infrastructure through `OmniScheduler`, while using a MOSS-specific model runner for the local transformer feedback loop. The stage inherits continuous batching, KV cache management, RadixAttention, CUDA Graph support for the backbone path, and streaming outputs, while adapting the request object and output processing to the `[T, 13]` multi-channel format.

The **vocoder** stage consumes generated frame rows as a stream. It strips the text/control channel, accumulates the 12 audio codebook channels per request, and calls the MOSS codec decoder in a persistent streaming session. As audio chunks become available, the stage sends them back to the client. When the AR engine emits a done signal, the vocoder flushes pending frames, releases the request slot, and returns the final payload.

This layout is intentionally close to the model's natural computation graph. The AR engine does not need to wait for a full utterance before the vocoder starts work. The vocoder does not need to know about text tokenization or KV cache. The preprocessing stage can cache and batch reference encoding independently of the AR decode loop. Each stage has a focused contract.

The same pipeline also gives the runtime clear places to optimize. Encoder caching belongs in preprocessing. Continuous batching and KV reuse belong in the AR engine. Slot management and chunk lifecycle belong in the streaming vocoder. Memory budgeting belongs at stage boundaries. This separation keeps MOSS-specific hooks small while allowing SGLang-Omni's shared runtime to handle routing, scheduling, data movement, process placement, streaming, and resource isolation.

In the default single-GPU configuration, all three stages run in a compact colocated topology. The pipeline config declares the stage factories, GPU placement, streaming edge from `tts_engine` to `vocoder`, model path, and memory settings. Split deployments can place the codec and AR engine on different devices while keeping the same stage-level contract.

For users, this complexity is hidden behind a familiar endpoint. MOSS is served through `/v1/audio/speech`, with support for basic synthesis, reference-based voice cloning, streaming PCM chunks, duration token control, inline pause markup, language hints, style instructions, seeds, and sampling parameters.

## Optimizing MOSS End-to-End

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

Voice cloning workloads often reuse the same speakers across many prompts. In MOSS, that pattern matters because reference encoding is a real neural computation. The preprocessing stage must run the reference waveform through the MOSS codec encoder before AR generation can begin.

![Reference audio cache: MOSS reference audio is hashed by content, looked up in an LRU cache, and only encoded on a miss.](images/tts-opt-encoder-cache.png)

SGLang-Omni uses a batched reference encoder for MOSS. A background worker collects concurrent encode requests, forms a small batch, deduplicates identical paths within the batch, and runs the codec encoder once per unique reference. The batching window is kept short so burst handling improves GPU utilization without adding noticeable request delay.

Caching is the larger win for steady production traffic. The `CachedReferenceEncoder` uses a content-addressed key rather than a filename key. This is important because a reference voice may be copied, renamed, downloaded to different paths, or provided through different request forms. If the audio content is the same, the encoded RVQ result can be reused.

The cache key is built with a layered strategy. A stat-tuple memo key provides a fast path for unchanged files. Sentinel byte reads from the beginning, middle, and end of the file detect content changes that reuse the same size and timestamp. A full content hash is computed only when needed. This keeps exact-match semantics without forcing a full file read on every request.

The cache also uses single-flight deduplication. If several concurrent requests ask for the same uncached reference, the first request performs the encode while the others wait for its future. This avoids the thundering-herd behavior that would otherwise appear when a popular speaker first enters the cache.

Encoded codes are stored on CPU as compact integer tensors. Each retrieval returns a clone with the dtype and device expected by downstream code, so callers cannot mutate shared cache entries and the rest of the pipeline sees one consistent contract.

In SeedTTS English evaluation on 2x H100 at concurrency 16, increasing the reference cache capacity from 256 to 1024 entries improved throughput by **32.0%** and reduced mean latency by **24.3%**. The memory cost was small because encoded code tensors are compact; the larger cache mainly prevents eviction of the active speaker working set.

### AR Engine

The MOSS AR engine has two levels of computation: the Qwen3 backbone and the local transformer frame-decode loop. SGLang-Omni captures both with CUDA Graphs, but they are captured separately because they have different structure and ownership.

![CUDA Graph execution: eager decode launches many small kernels, while graph replay records the step once and replays forward, sampling, and state updates as one call.](images/tts-opt-cuda-graph.png)

The **backbone graph** uses SGLang's standard CUDA Graph path for causal LM decode. It covers the attention and KV-cache path for the Qwen3 backbone, including batch-size buckets and the scheduler integration that SGLang already uses for high-throughput autoregressive serving.

The **frame-decode graph** is MOSS-specific. It captures the local transformer micro-loop for a full frame: the initial local transformer step, stop/continue sampling, 12 sequential codebook projections, 12 codebook samples, codebook embedding feedback, local transformer steps between codebooks, and feedback embedding assembly for the next frame. Capturing this loop removes hundreds of small kernel launches from the eager path.

The frame graph is captured for a set of batch-size buckets. Requests whose active batch size does not exactly match a bucket are padded to the next bucket and then trimmed after graph replay. Requests requiring features that cannot be represented in the fixed graph topology, such as certain repetition-penalty paths, fall back to eager decoding. This keeps the fast path static without sacrificing correctness.

CUDA Graph replay requires stable memory addresses. MOSS therefore uses `MossTTSLocalDecodeStatePool`, a persistent GPU-side state pool allocated before graph capture. Each active request owns one row in the pool. The row holds feedback embeddings, sampling temperatures, top-p and top-k values, seeds, generation counters, sampling counters, repetition-penalty state, and audio-token history. A reserved padding row provides a stable no-op target for graph padding and future in-graph routing.

One subtle part of the design is how the backbone sees the next frame input. MOSS needs to feed the fused embedding of the generated frame back into the next backbone step. Instead of constructing a new `[1, 13]` row on the host, the runner writes feedback embeddings into a staging embedding table and passes integer row indices as input IDs. The backbone graph then performs a normal embedding lookup, while MOSS-specific multi-channel fusion remains isolated in the model runner and frame graph.

SGLang's radix cache also needs a key for each generated row. A CPU-side hash would introduce a GPU-to-CPU synchronization every frame. MOSS replaces that with a GPU-native polynomial row hash over all 13 channels. The hash is folded into the special-token-safe ID range so it does not collide with scheduler control tokens. Stop rows keep the original end token so completion logic remains simple.

Sampling is another hot path. Each frame performs 13 seeded samples: one binary stop/continue decision and 12 audio codebook samples. SGLang-Omni uses a GPU-native seeded sampler whose randomness is derived from the request seed, frame index, and channel index. This makes generation independent of batch neighbors and reproducible across concurrency settings for a fixed server configuration. Compiling the sampler with a narrow `torch.compile` scope fuses the sampling kernel chain without changing the surrounding model execution.

On SeedTTS English evaluation at concurrency 16, the compiled seeded sampler improved throughput by **12.3%**, reduced mean latency by **11.1%**, and reduced mean RTF by **10.5%**. The narrow compile scope was intentional: it targets a real hot path while avoiding broad compile-time overhead and graph changes in the backbone or local transformer.

### Streaming Vocoder

The vocoder stage turns generated RVQ frames into audio chunks. MOSS-Audio-Tokenizer-v2 supports stateful streaming decode, so SGLang-Omni keeps a persistent codec streaming session alive inside the vocoder executor.

The session manages a fixed pool of slots. Stream slots are assigned to active streaming requests. A separate offline slot is reserved for non-streaming or fallback decode, preventing one mode from blocking the other. Each slot owns codec streaming state across decode steps, including the internal decoder state needed to make incremental output consistent.

Streaming decode is triggered by thresholds. The first threshold is small: by default, the vocoder can emit the first chunk after 5 frames, about 0.4 seconds of audio at 12.5 Hz. This reduces time to first audio. The steady-state threshold is larger: by default, later chunks use 25 frames, about 2 seconds of audio, which improves throughput and amortizes decode overhead.

The scheduler also performs coalesced steps. When one request becomes ready to decode, the vocoder checks whether other active slots can join the same step. Joined requests decode the same number of pending frames in one batched codec call. This shares GPU work and a single output transfer across multiple streams, while still letting each request maintain its own lifecycle.

Short vocoder chunks launch many kernels relative to their compute size. To reduce that overhead, SGLang-Omni captures the MOSS codec decoder as CUDA Graphs for common frame counts. The key challenge is state: the codec decoder maintains KV cache and position offsets across streaming steps. The implementation keeps those buffers at stable addresses and updates them in place, so graph replay remains valid across steps.

The speedup is largest for short streaming chunks:

| Frames per Step | Eager | CUDA Graph | Speedup |
|---:|---:|---:|---:|
| 4 | 66.3 ms | 30.1 ms | 2.20x |
| 5 | 65.8 ms | 30.7 ms | 2.14x |
| 8 | 65.6 ms | 34.0 ms | 1.93x |
| 13 | 65.4 ms | 40.4 ms | 1.62x |
| 25 | 74.8 ms | 58.3 ms | 1.28x |
| 100 | 222.9 ms | 215.3 ms | 1.04x |

The production path includes safe fallback. If graph capture runs out of memory, if free memory is below the configured threshold, or if replay fails, the vocoder transparently falls back to eager decode. Output consistency is preserved: streaming and non-streaming artifacts are checked for consistency in CI, and the graph path has unit tests around stateful decode behavior.

### Memory Budgeting

Compact deployment is important for users who want the simplest path from model download to serving. In the default MOSS Local config, preprocessing, AR generation, and vocoder execution can be colocated on one GPU. That makes memory budgeting a first-class part of serving.

SGLang's default memory profiling is designed for a standalone LLM engine. MOSS needs to leave headroom for codec runtime allocations and streaming state. SGLang-Omni therefore gives the AR engine an explicit colocated memory contract. The default single-GPU config sets a total GPU memory fraction and reserves a codec runtime margin. The effective AR KV-cache allocation is reduced accordingly, preventing the AR engine from consuming memory needed by the vocoder.

This is a small change with visible impact. In a single-card colocated configuration at concurrency 8, explicit codec memory budgeting improved throughput by **8.9%** and reduced mean RTF by **8.4%**. More importantly, it makes the deployment behavior more predictable under memory pressure.

## Performance

We evaluate the optimized MOSS-TTS Local Transformer v1.5 serving path on the SeedTTS English set (N=1088). All optimizations are enabled (AR CUDA Graph, frame-decode CUDA Graph, vocoder CUDA Graph, compiled seeded sampler). Each data point is the mean of 3 runs.

### Single-GPU (1× H100 80GB, colocate)

**Non-streaming:**

| Concurrency | Throughput (qps) | RTF | Latency mean (s) |
|---:|---:|---:|---:|
| 2  | 2.974 | 0.157 | 0.676 |
| 4  | 4.870 | 0.192 | 0.821 |
| 8  | 6.111 | 0.310 | 1.306 |
| 16 | 6.144 | 0.623 | 2.593 |

**Streaming:**

| Concurrency | Throughput (qps) | RTF | Latency mean (s) | TTFP (ms) |
|---:|---:|---:|---:|---:|
| 2  | 2.256 | 0.206 | 0.888 | 261 |
| 4  | 2.649 | 0.356 | 1.509 | 726 |
| 8  | 2.633 | 0.726 | 3.033 | 2239 |
| 16 | 2.635 | 1.458 | 6.045 | 5227 |

### Dual-GPU (2× GPU, concurrency 16)

| Mode | Throughput | Audio Throughput | Mean Latency | Mean RTF | WER |
|---|---:|---:|---:|---:|---:|
| Non-streaming | 5.976 req/s | 26.303 audio s/s | 2.669 s | 0.644 | 1.75% |
| Streaming | 2.909 req/s | 12.804 audio s/s | 5.474 s | 1.322 | 2.14% |

Non-streaming throughput scales well with concurrency, reaching **6.1 qps on a single H100** and **6.0 qps on 2× GPU** at concurrency 16. The streaming path trades throughput for incremental delivery — the vocoder runs more frequently on smaller chunks and shares GPU time with the AR engine. Improving high-concurrency streaming scalability is on the roadmap.

Quality remains stable across serving modes: non-streaming WER is **1.75%** and streaming WER is **2.14%** in the dual-GPU evaluation, confirming that the CUDA Graph and streaming scheduler paths preserve the model's audio semantics.

## Try It Yourself

Detailed instructions are available in the SGLang-Omni MOSS-TTS-Local cookbook. The commands below show the shortest path from a clean container to a working speech endpoint, followed by the most common request patterns.

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

### Benchmarking

To reproduce the serving benchmarks, start the server and run the benchmark client:

**1. Start the server:**

```bash
sgl-omni serve \
  --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
  --port 8000
```

**2. Run the benchmark** (against the running server):

```bash
# Non-streaming, concurrency 16
python -m benchmarks.eval.benchmark_tts_seedtts \
  --use-existing-server --generate-only \
  --base-url http://localhost:8000 \
  --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
  --ref-format references --lang en --token-count auto \
  --max-concurrency 16 \
  --output-dir results/moss_perf_nostream_c16

# Streaming, concurrency 16
python -m benchmarks.eval.benchmark_tts_seedtts \
  --use-existing-server --generate-only \
  --base-url http://localhost:8000 \
  --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
  --ref-format references --lang en --token-count auto \
  --max-concurrency 16 \
  --output-dir results/moss_perf_stream_c16 \
  --stream
```

## Roadmap

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

**SGLang-Omni** - Haoguang Cai, Shangming Cai, Qiujiang Chen, Yuhao Chen, Jiaxin Deng, Wenyao Gao, Yifei Gao, Jingwen Gu, Yitong Guan, Zhihao Guo, Chenchen Hong, Hao Jin, Xinli Jing, Xiangrui Ke, Shenggui Li, Junrong Lin, Estella Liu, Xinyu Lu, Yuan Luo, Ratish Palanisamy, Mick Qian, JinTao Qu, Shuai Shi, Yijiang Tian, Chao Wang, Richard Wang, Shuwen Wang, Zijie Xia, Yuhao Yang, Xuesong Ye, Fan Yin, Yue Yin, Gaokai Zhang, Xiaoyu Zhang, Yichi Zhang, Chenyang Zhao.

**MOSS-TTS Local Transformer v1.5** - Yitian Gong, Kuangwei Chen, Zhicheng Zhang, Botian Jiang, Yiyang Zhang, Kang Yu, Yang Gao, Xiaogui Yang, Qinyuan Chen, Zhaoye Fei, Shimin Li, Xipeng Qiu.

## Learn More

- **Model:** [OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5)
- **Serving framework:** [SGLang-Omni on GitHub](https://github.com/sgl-project/sglang-omni)
- **Documentation:** [SGLang-Omni docs](https://sgl-project.github.io/sglang-omni/)
- **MOSS-TTS-Local cookbook:** [MOSS-TTS-Local in SGLang-Omni](https://sgl-project.github.io/sglang-omni/cookbook/moss_tts_local.html)
- **MOSS optimization roadmap:** [#637](https://github.com/sgl-project/sglang-omni/issues/637)
- **Design background:** *SGLang-Omni: Redesigning the Inference Framework for Multi-Stage Generative Models*
