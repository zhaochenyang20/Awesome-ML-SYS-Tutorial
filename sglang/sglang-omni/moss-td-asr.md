# SGLang-Omni Day-0 Support for MOSS ASR Model: An ASR Engineering Field Guide

The community recently collaborated to bring [MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) (MOSS-TD for short) to [SGLang-Omni](https://github.com/sgl-project/sglang-omni), covering model support, correctness fixes, and performance optimization. This post records the engineering lessons we accumulated along the way: from how ASR models work, to the MOSS-TD architecture and optimization strategies, to benchmark design and results.

MOSS-TD's distinguishing feature is multi-speaker diarization with support for up to ~90 minutes of audio. We built long-sequence, multi-speaker ASR datasets into CI specifically for this model, giving performance work a clear target.

------

## ASR Model Primer

ASR (Automatic Speech Recognition) converts audio input into text output: **input is audio waveform, output is text.**

MOSS-TD follows the Audio LLM paradigm — a Whisper encoder produces continuous embeddings, which are projected by an FFN adaptor into a decoder-only LLM that autoregressively generates the transcript with speaker labels and timestamps.

![MOSS-TD ASR Inference Pipeline](images/moss-td-pipeline.svg)

*Figure 1. MOSS-TD inference pipeline: Whisper encoder → FFN adaptor → Qwen3 LLM prefill → AR decode → text with speaker labels and timestamps.*

![What the Data Looks Like at Each Stage](images/moss-td-data-shapes.svg)

*Figure 2. Data shapes along the pipeline: 16 kHz waveform → log-mel spectrogram `[T, 80]` → encoder hidden states `[T, 1024]` → LLM embeddings `[T/4, 1024]` (12.5 tokens/s, ~67k tokens for 90 min) → diarized text.*

The pipeline has three stages:

1. **Encoder.** Waveform → log-mel spectrogram (80 bins) → 24-layer Whisper Transformer → 4× Time Merge (concatenate every 4 frames) → FFN adaptor (Linear→SiLU→Linear, 4096→1024). The output is a sequence of continuous float vectors in the LLM's embedding space.

2. **LLM Prefill.** The audio embeddings replace placeholder tokens in the prompt. Qwen3 processes the full prompt in one forward pass to build the KV cache.

3. **AR Decode.** Qwen3 generates text tokens one at a time — transcript with speaker labels `[S01]`/`[S02]` and timestamps — until EOS.

| Component    | Spec                                           |
| ------------ | ---------------------------------------------- |
| Architecture | `MossTranscribeDiarizeForConditionalGeneration` |
| Audio Encoder | Whisper encoder (24 layers, d_model=1024)      |
| Adapter      | FFN: Linear→SiLU→Linear→LayerNorm, 4096→1024 |
| Text Decoder | Qwen3 (28 layers, hidden=1024, GQA 16/8)      |
| Output       | Speaker-labeled transcript with start/end timestamps |
| Endpoint     | `/v1/audio/transcriptions`                     |

### Chunked Prefill for Long Audio

For long audio (up to ~90 minutes), the encoded token sequence can reach tens of thousands of tokens. **Chunked Prefill** splits it into 4096-token chunks, processing one per scheduling step and interleaving decode steps for other requests between chunks. Stream output is suppressed during chunked prefill to avoid emitting intermediate states as if they were transcript output.

![Chunked Prefill for Long Audio](images/moss-td-chunked-prefill.svg)

*Figure 3. Chunked prefill splits a long token sequence into 4096-token chunks, interleaving other requests' decode steps between chunks to keep the GPU busy.*

### ASR vs TTS

ASR and TTS share much of the same serving infrastructure in SGLang-Omni — both use `OmniScheduler`, share CUDA Graph / KV cache management / continuous batching, and run on the same Qwen3 backbone. The real differences lie in what they encode, what they generate, and how their pipelines are organized:

| Dimension         | ASR (MOSS-TD)                                | TTS (Higgs / MOSS-TTS)                      |
| ----------------- | -------------------------------------------- | -------------------------------------------- |
| Audio representation | Continuous features (mel → encoder hidden states) | Discrete codec tokens (RVQ multi-codebook)  |
| Data flow         | Audio → text                                 | Text → audio                                 |
| Decoder / Vocoder | Not needed — output is plain text            | Vocoder required to reconstruct waveform     |
| Typical input length | Very long (MOSS-TD supports ~90 min)       | Short (reference voice: a few seconds)       |
| Pipeline stages   | Single stage (encoder + LLM)                 | Multi-stage (encoder → AR engine → vocoder)  |
| Streaming         | Stream text output (incremental transcript)  | Stream audio output + streaming vocoder      |

![ASR vs TTS Architecture Comparison](images/moss-td-asr-vs-tts.svg)

*Figure 4. ASR vs TTS architecture. ASR: Whisper encoder → FFN adaptor → Qwen3 decoder, which generates text directly. TTS (MOSS-TTS-Local-v1.5): codec encoder → Qwen3-4B backbone → local transformer → vocoder.*

ASR does not need a vocoder at all — it simply generates text. But ASR inputs can be much longer than TTS inputs, shifting the optimization focus toward the AR decode loop and long-sequence memory management. For the full TTS optimization story, see [Optimizing TTS Inference](tts-optimization.md).

------

## Where Time Is Spent: Profiling

Before optimizing, we profiled on a single H100 (CUDA Graph, bf16) to find the bottlenecks:

![MOSS-TD Profiling Breakdown](images/moss-td-profiling.svg)

*Figure 5. Inference time breakdown by stage. AR Decode dominates at low concurrency; encoder share rises at high concurrency with short audio.*

| Audio Length | Concurrency | Encoder | LLM Prefill | AR Decode |
| -----------: | ----------: | ------: | ----------: | --------: |
| 5 s          | 1           | 8.9%    | 14.7%       | 76.4%     |
| 5 s          | 4           | 20.0%   | 22.3%       | 57.7%     |
| 5 s          | 16          | 38.2%   | 29.7%       | 32.1%     |
| 60 s         | 1           | 4.0%    | 2.1%        | 94.0%     |
| 60 s         | 4           | 5.0%    | 4.6%        | 90.4%     |
| 60 s         | 16          | 13.7%   | 9.5%        | 76.8%     |
| 20 min       | 1           | 4.7%    | 0.8%        | 94.5%     |
| 20 min       | 4           | 9.2%    | 1.9%        | 88.9%     |
| 20 min       | 16          | 11.6%   | 2.6%        | 85.7%     |

Two takeaways:

- At c=1 with long audio, AR Decode takes 94%+ of total time — the leverage is almost entirely in the decode loop.
- At c=16 with short audio, encoder + prefill together account for 68%, making encoder-side optimizations worthwhile.

------

## Optimization Strategies

Our optimization stack reuses the core infrastructure built for TTS, with ASR-specific adaptations. If you have read [the TTS optimization blog](tts-optimization.md), you will recognize the same ideas — CUDA Graph, async decode, encoder caching — applied to a simpler pipeline (no vocoder, no multi-codebook generation).

![MOSS-TD Optimization Strategies](images/moss-td-opt-overview.svg)

*Figure 6. Optimization strategies mapped to pipeline stages. Encoder gets CUDA Graph + Torch Compile + LRU cache; decode gets CUDA Graph + async decode + stream output.*

### Encoder

**CUDA Graph.** The Whisper encoder works on fixed 30-second windows: input audio is split into 30 s chunks (the last one padded to 30 s), and each chunk goes through one encoder forward. Since every chunk has the same shape, the only variable is how many chunks a request carries — so the encoder is bucketed by chunk count (default up to 8 chunks, covering ~4 min of audio), and each bucket captures a CUDA graph, eliminating per-call kernel launch overhead.

**Torch Compile** (opt-in). An opt-in setting replaces the encoder CUDA graph with `torch.compile(mode="reduce-overhead")`, which adds kernel fusion on top of its own internal CUDA graph. The two approaches are mutually exclusive — `reduce-overhead` already owns a CUDA graph, so stacking a manual one is redundant (and nesting is illegal). Torch Compile trades slower cold start (one-time per-bucket compilation) for lower steady-state latency. Prefer it for encoder-bound, high-concurrency short-audio workloads.

**LRU Cache.** The Whisper encoder forward is deterministic for identical input — same waveform always produces the same embeddings. We cache encoder outputs on CPU (max 64 entries, 4 GB), keyed by a content hash of the input waveform. On a hit, the stored tensor is transferred back to GPU asynchronously and the encoder is skipped entirely.

![Encoder LRU Cache Flow](images/moss-td-encoder-cache.svg)

*Figure 7. LRU cache flow: content-hash the waveform, check the cache. On a hit, skip the encoder entirely and transfer cached embeddings back to GPU. On a miss, run the encoder and store the result.*

Unlike TTS where the same reference voice is reused across many prompts (high hit rate), ASR inputs are typically unique in production. The cache is most useful during request retries, A/B testing with different decode parameters, and development iteration. Being honest about limited hit rate in production is more useful than pretending the cache always helps.

### AR Decode

**CUDA Graph.** The LLM decode step pads batch size to predefined buckets (1, 2, 4, 8, ...) and replays a captured CUDA graph, eliminating kernel launch overhead on every token. The mechanism is the same as for TTS decode — see [CUDA Graph in the TTS blog](tts-optimization.md#cuda-graph) for the full discussion of static-path challenges, fixed-address shadow buffers, and gather/scatter patterns.

![Eager vs CUDA Graph](images/moss-td-cuda-graph.svg)

*Figure 8. Eager mode dispatches each kernel individually with CPU gaps in between; CUDA Graph records the sequence once and replays it as a single GPU operation.*

**Async Decode.** Same one-step lookahead as TTS: launch the current decode step's GPU work, then resolve the previous step's host-side work (D2H copy, finish detection, result dispatch) in parallel. Falls back to synchronous mode at batch size 1, where the host-side work is too small to justify the overlap. Two alternating pinned host buffers prevent read/write races between the GPU's async D2H write and the CPU's read.

This brings a solid qps gain at high concurrency. The work also fixed a KV slot leak caused by lookahead overrun. For the full mechanism — launch/resolve event loop, ping-pong buffers, lookahead guard — see [Asynchronous Decode + Lookahead in the TTS blog](tts-optimization.md#asynchronous-decode--lookahead).

![Synchronous vs Asynchronous Decode](images/moss-td-async-decode.svg)

*Figure 9. Synchronous decode wastes GPU cycles waiting for CPU resolve. Async decode overlaps GPU forward N with CPU resolve N-1 via ping-pong pinned buffers, hiding the CPU work entirely.*

**Stream Output.** During AR decode, transcript text is emitted incrementally via SSE so that users see partial results as they are generated rather than waiting for the full sequence. Three mechanisms control when to emit:

1. **Rate limiting** (default 50 ms): tokens accumulate in a per-request buffer and flush only when enough time has elapsed. The first token goes out immediately; EOS always triggers a flush.
2. **Chunked prefill suppression**: all emission is suppressed during chunked prefill to prevent intermediate states from being misinterpreted as transcript.
3. **Incomplete UTF-8 handling**: if the accumulated tokens end with the Unicode replacement character (an incomplete multi-byte sequence split across token boundaries), emission is held until the next token completes the sequence.

### Batched Inference

Encoder-side mel alignment and LLM-side sequence packing allow multiple requests to be processed together more efficiently. On the encoder side, mel spectrograms of different lengths are aligned for batched Whisper forward passes. On the LLM side, multiple requests' token sequences are packed into the same batch for prefill and decode, making better use of GPU compute under concurrency.

------

## Benchmark Results

*Benchmark results are being re-measured and will be added here.*

We prepared two datasets for multi-speaker ASR:

- **movies800times**: short-sequence dataset, 800 dialog clips
- **aishell4_long**: long-sequence dataset, 20 long-form meeting recordings

Both datasets are currently under private license — contact the MOSS team for access.

Key metrics we report: **RTF** (Real-Time Factor) is processing time divided by input audio duration — `<1` means faster than real time. **audio_s/s** is total audio seconds processed per wall-clock second, which measures how much batching delivers real throughput gains.

------

## Model Usage

See the [MOSS-TD cookbook](https://sgl-project.github.io/sglang-omni/cookbook/moss_transcribe_diarize.html) for deployment and usage instructions.

------

## What's Next

Several optimization efforts are still in progress:

- **Streaming audio input**: accept audio as it arrives and transcribe incrementally, instead of waiting for the full clip to upload
- **Piecewise prefill CUDA Graph**: capture chunked prefill as a CUDA graph
- **Tensor parallelism**: TP integration for multi-GPU deployment

------

## Acknowledgments

Thanks for the joint effort of the OpenMOSS team and SGLang-Omni team.

**MOSS Team:** Donghua Yu, Zhengyuan Lin, Hanfu Chen, Yiyang Zhang, Yang Gao, Zhaoye Fei, Qinyuan Cheng, Shimin Li, Xipeng Qiu.

**SGLang-Omni Team:** Yijiang Tian, Xinli Jing, Xiangrui Ke, Zhihao Guo, Ruoqi Zhang, Lifan Shen, Jintao Qu, Xuxiang Tian, Kaige Li, Ratish P, Haoguang Cai, Zijie Xia, Chenchen Hong, Xuesong Ye, Jingwen Gu, Jiaxin Deng, Jiaxuan Luo, Xinyu Lu, Hao Jin, Chenyang Zhao, Yichi Zhang.

## Learn More

- **Model:** [OpenMOSS-Team/MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)
- **Serving framework:** [SGLang-Omni on GitHub](https://github.com/sgl-project/sglang-omni)
- **Cookbook:** [MOSS-TD in SGLang-Omni](https://sgl-project.github.io/sglang-omni/cookbook/moss_transcribe_diarize.html)
- **ASR optimization roadmap:** [tracking issue on GitHub](https://github.com/sgl-project/sglang-omni/issues/924)
- **TTS optimization blog:** [Optimizing TTS Inference](tts-optimization.md)
- **Encoder skew analysis:** [The Root Cause of RL Training-Serving Skew](moss-tts-local-batch-encoder-skew.md)
