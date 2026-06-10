# Optimizing TTS Inference: Engineering Lessons from Profiling to Streaming in SGLang Omni

*Yichi Zhang — June 2026*

*Originally published on [Medium](https://medium.com/@yichizhang602/optimizing-tts-inference-engineering-lessons-from-profiling-to-streaming-in-sglang-omni-00d06e3fc78d)*

---

<span style="color:red">**[Request to change]** 图片要更好看一点</span>

A TTS (Text-to-Speech) model converts written text into natural-sounding spoken audio. It powers accessibility for the visually impaired, enables hands-free interaction with devices, brings life to virtual assistants and audiobooks, and makes technology usable for everyone, anywhere.

Optimizing Text-to-Speech (TTS) inference looks a lot like LLM optimization on paper, but the actual engineering bottlenecks are entirely different. For example, TTS has unique multi-stage architecture, which fits well in our SGLang Omni engine structure. In this blog, instead of just listing our optimizations, we break down the mechanical sympathy required to make this pipeline fast—the bottlenecks we hit, the host-to-device pitfalls, and the architectural trade-offs we made along the way. We hope this blog can be a reference material and inspire our future TTS model support, and help all readers understand more on TTS model support & optimization.

## The Higgs TTS Pipeline Under the Hood

![Higgs TTS Pipeline](images/tts-opt-pipeline.png)

To optimize the system, we first have to understand the computing process through four stages of the Higgs pipeline:

- **Preprocessing (CPU):** Text tokenization and reference audio loading. This is purely IO-bound and handles no GPU compute.
- **Audio Encoder (GPU):** Uses [HiggsAudioCodec](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer), a neural audio codec based on the DAC (Descript Audio Codec) architecture with a semantic encoder branch. The encoder converts the reference audio waveform into discrete tokens via Residual Vector Quantization (RVQ), producing an output of shape `[T, 8]`, where T is the number of time steps and 8 represents parallel codebook tokens per step.
- **TTS Engine (GPU Backbone):** The core autoregressive (AR) model based on a Qwen3 LLM architecture. It generates the multi-codebook tokens step-by-step.
- **Vocoder (GPU):** A DAC decoder that converts the generated tokens back into an audible waveform. Note: even though Vocoder and Encoder share the same single instance in implementation, they don't share the same weight on model level.

### Know Before Dive In: the Codebook & RVQ

LLMs operate on text tokens — discrete symbols from a finite vocabulary. To apply the same autoregressive framework to audio, we need a way to turn a continuous waveform into a sequence of discrete tokens, and back. This is where neural audio codecs come in: models like SoundStream, EnCodec, and DAC learn to compress audio into compact token sequences that an LLM-style backbone can generate.

The core tokenization technique these codecs share is Residual Vector Quantization (RVQ) ([Zeghidour et al., 2021](https://arxiv.org/abs/2107.03312)). A single vector quantizer (one codebook) can only approximate the audio signal coarsely. RVQ improves on this by stacking multiple codebooks in series: the first codebook quantizes the signal, then the second codebook quantizes the residual error left by the first, the third quantizes what the second missed, and so on. Each layer captures progressively finer detail — CB0 encodes coarse structure (pitch, rhythm, energy), while deeper codebooks encode subtle textures and high-frequency harmonics. Decoding simply sums all codebook contributions to reconstruct the audio.

This layered structure is fundamental to TTS inference optimization. Because the codebooks form a strict hierarchy — each one refining the previous — generating them matters: producing all codebooks independently in parallel degrades quality (higher codebooks can't adapt to what lower ones emitted), while generating them fully sequentially multiplies latency by the number of codebooks. Every multi-codebook TTS system must navigate this trade-off, and the delay pattern (discussed next) is the most widely adopted solution.

### Architectural Gotchas: The Delay Pattern

The delay pattern is a scheduling strategy for multi-codebook autoregressive generation, first introduced in Meta's [MusicGen (2023)](https://arxiv.org/abs/2306.05284). The core idea: instead of generating all codebooks in lockstep (which sacrifices quality) or sequentially (which sacrifices speed), stagger them so that each codebook starts in steps after CB0. This way, when CB_i samples at step t, it can condition on CB_{i-1}'s output from step t-1 — giving each layer causal access to its coarser neighbor without serializing the entire generation.

To balance this, Higgs uses a Delay Pattern where Codebook i is delayed by i time steps relative to Codebook 0.

- Step 0: CB0 activates
- Step 1: CB0 + CB1 activate
- Step 2: CB0 + CB1 + CB2 activate
- ... and so on.
- Step 7: CB0 + CB1 + … CB7 activate
- … decode for more steps until complete, and start winding-down
- Step n-8: CB0 hits EOC and stop; CB1 – CB7 still activate
- Step n-7: CB0 + CB1 hits EOC stop; CB2 – CB7 still activate
- … until finish at the end

![Delay Pattern State Machine](images/tts-opt-delay-pattern.png)

This introduces a state machine with four phases: Delay Stage (staggered activation), Active Stage (normal sampling), Wind Down (triggered when CB0 hits the End-of-Codes token), and Finished. And here is a graph to show it more in detail.

The trade-off of the delay pattern is it adds exactly N-1 extra AR steps to every generation (7 steps for 8 codebooks) — which is the ramp-up and wind-down, as shown in the graph. For a typical 250-step generation, this is a ~3% overhead. The exchange is better audio quality, and fully parallel generation.

Because we need CUDA Graph compatibility (discussed in the optimization section), the entire delay pattern state machine is implemented as pure tensor operations. The key function is `batched_step_direct()` in `sampler.py`, which manages four state variables per request: `delay_count` (tracks ramp-up progress), `eoc_countdown` (wind-down timer, initialized to -1), `generation_done` (terminal flag), and `last_codes` (last emitted multi-codebook row). Each step uses `torch.where` to compute:

- a delay mask `cb_idx > delay_count` that forces not-yet-active codebooks to emit BOC tokens
- state transitions — incrementing `delay_count` during ramp-up, setting `eoc_countdown = N-2` when CB0 emits EOC, decrementing it during wind-down, and setting `generation_done` when the countdown hits zero.

All branching is expressed as tensor-level conditionals, making the function a single static compute graph that CUDA Graph can capture and replay.

So in summary, the delay pattern converts a naive 8-way parallel generation (fast but low quality) or 8-way sequential generation (high quality but 8× slower) into a staggered pipeline that is only N-1 steps longer than parallel, while preserving the causal codebook hierarchy that RVQ-based audio quality depends on.

The delay pattern is not unique to Higgs — it has been widely adopted in multi-codebook audio generation models. Including but not limited to MOSS-TTS (OpenMOSS) and Parler-TTS. And in SGLang-Omni supported model, there're other models adopted different structure, such as FishAudio S2-Pro decouples the RVQ codebook into Slow AR (semantic) + Fast AR (acoustic codebooks in parallel).

**The Catch:**
Our original implementation uses if-else statements and for loops to control the flow of this delay pattern, which needs to be completely re-written in tensor operations to support CUDA Graph. Will be discussed later in the CUDA Graph chapter.

## Stage 0 Support: RadixCache and SGLang Scheduler Backend

Before any of the optimizations discussed below, we first needed a working TTS serving baseline built on SGLang's infrastructure. The foundational work landed in commit `60c6e75` (2026-02-27), which introduced RadixCache support and `torch.compile` integration for the FishAudio DualAR architecture — the first TTS model served through the SGLang-Omni pipeline. The full SGLang scheduler integration (paged KV cache, RadixAttention, batch planning) followed in commit `92dbd45` (2026-03-09) for S2-Pro. Higgs TTS support was added later in PR #428 (commit `4d6be58`, 2026-05-17), bringing the Higgs Audio v3 model onto the same SGLang scheduler backend with a custom HiggsScheduler and model runner. This baseline — SGLang's continuous batching scheduler with RadixCache for KV reuse — is the starting point from which all subsequent optimizations in this blog are measured.

## Profiling: Where is the Time Actually Spent?

Before writing code, we profiled the naive pipeline and identified three core bottlenecks:

- **AR Decode Dominates:** A typical 10-second speech request requires 250 decode steps. Every single step involves a backbone forward pass, head projection, sampling, and Device-to-Host (D2H) synchronization. A tiny 0.1ms overhead per step inflates end-to-end latency by nearly 25ms.
- **The Encoder is Heavy but Static:** A single encoding pass takes 50–100ms. However, in production, users often reuse the same reference audio across multiple prompts.
- **Vocoder Queuing:** The vocoder is fast (~10ms per call), but under high concurrency, multiple AR generation loops finish at the exact same time, creating a massive serial bottleneck at the vocoder stage.

With those bottlenecks, we applied targeted optimization strategies. Here is a macro-level graph to show our strategies, and we will be discussing each strategy in the next section.

![Optimization Strategies Overview](images/tts-opt-strategies.png)

## Layer-by-Layer Optimizations

### Encoder: Bypassing the Compute with LRU Caching

**Why:** The encoder converts a reference audio clip into delayed codec tokens. In production, users frequently reuse the same reference voice across many prompts (e.g., a fixed narrator voice for an audiobook). Each encoding pass costs 50–100ms of GPU time, but the output is deterministic for identical input audio. This makes it a textbook caching opportunity.

**How:** We introduced an LRU cache keyed by the audio waveform content. On a cache hit, the encoder stage is skipped entirely.

For text, SGLang's RadixCache enables prefix sharing — two prompts that start with the same tokens can reuse partial KV cache. However, audio caching is fundamentally different: there is no meaningful prefix relationship in the time-frequency domain, so we use strictly exact-match lookup. The cache key is a content hash of the input audio: SHA3-64 for file paths (with stat-based memoization to avoid re-reading stable files), or xxh3_64 for raw bytes/base64 input. Two audio clips that produce the same hash are a hit; everything else is a miss.

We also experimented with online batched encoding by bucketing incoming audio by length. While it improved raw throughput on paper, it created a new problem in production: GPU utilization shifted from smooth patterns to intermittent spikes, causing severe resource contention with the concurrent AR decode loops. We ultimately moved batched encoding offline (used strictly for bulk server warmups) and kept online encoding isolated.

<span style="color:red">**[Placeholder]** Can we add more experimental data or evidence to support that with the batching, autoregression is going to be unstable and spike GPU usage will come and take a toll on us.</span>

### AR Decode: Shaving Off Every Microsecond

Since AR decode is our primary bottleneck, we need to put the majority of work into optimizing this step. In our implementation, we focused on eliminating kernel launch overhead and synchronization stalls by CUDA Graph and CPU-GPU async decode, separately.

#### CUDA Graph Migration

**Why:** Because each AR step launches a sequence of tiny kernels, the CPU launch overhead was killing performance. Therefore we captured the entire decode loop inside a CUDA Graph to eliminate those numerous small launch overheads.

**How:** However, CUDA Graph records a fixed sequence of kernel launches and replays it every time we reach the point, so the execution path must be static — any Python branch that depends on runtime data breaks the recording (for example, if/else statements). Therefore, if we wanted to use CUDA Graphs, to make this work, we had to eliminate all Python if-else branching in the model's forward path, rewriting the delay pattern state machine into in-place tensor operations with fixed memory addresses.

To make the path static, we pre-allocated fixed-address GPU buffers for every piece of per-request decode state — the delay counter, the EOC countdown, the `generation_done` flag, the last emitted codes, and the sampled-code output — all shaped `[max_batch, …]` and allocated once at startup. Each step overwrites these in place at the same addresses, so the graph replays without being rebuilt.

Around the captured graph, the runner copies the active rows' state from the request pool into these fixed "shadow" buffers before the step, lets the graph read and update them in place, then scatters the results back to the pool afterward — all GPU-to-GPU. Finally it packs the step's outputs (codes + done flags) into a single staging buffer so the whole step returns to the CPU in one copy, which is what lets us make that copy non-blocking next.

We will discuss further on how we achieve GPU-CPU async decode in a later section of this blog.

#### Merging D2H Synchronizations

**Why:** Our baseline implementation performed three separate Device-to-Host (D2H) synchronizations per AR step to check tokens and states, creating repeated pipeline stalls.

**How:** We optimized this by consolidating all intermediate data into a single staging tensor named `_cg_collect_staging`. Now, we execute exactly one combined transfer per step using the slice: `combined_cpu = staging[:n_real].cpu()`. This single change dramatically reduced pipeline latency.

#### Asynchronous Decode + Lookahead

**Why:** The vanilla pattern of CPU–GPU synchronization is as shown in the picture below. GPU and CPU process in the same flow and will stop and wait for each other. We discovered this pattern is inefficient since the D2H sync time can stack up to very high during AR decoding. To hide the remaining D2H synchronization time, we want to discover a pattern to let GPU and CPU work simultaneously and not wait for each other.

**How:** The async decode splits each step into two halves — a GPU-side **launch** and a CPU-side **resolve** — that run one step apart:

The overall timeline would be as shown in the picture:

The event loop implements this as: each iteration launches the current step (enqueue GPU work + async D2H + record event), then resolves the previous step (check event, read host buffer, process results). When the batch size drops below 2, it falls back to synchronous execution, since the async fixed overhead will be larger than the performance gains from overlapping.

![Async Decode Timeline](images/tts-opt-async-decode.png)

**GPU launch (step N):** Before replaying the CUDA Graph, the runner gathers each active request's current state (delay counter, last emitted codes, done flag, etc.) from a shared pool into the graph's fixed-address buffers. The graph then runs the forward pass and sampling, writes updated state back to the pool, and packs the step's outputs (all 8 codebook codes plus completion flags) into a single staging tensor. This staging tensor is copied to a pinned host buffer asynchronously — the GPU does not wait for the copy to finish. A CUDA event is recorded right after the copy is enqueued, serving as a "data is ready" signal for the CPU.

**CPU resolve (step N-1):** While step N runs on the GPU, the CPU processes step N-1's results. It checks the CUDA event (non-blocking) to see if the D2H copy has landed. In the common case it has — the CPU reads the host buffer and runs per-request bookkeeping: appending codes to each request's output, detecting end-of-generation, emitting streaming audio chunks, and removing finished requests. If the copy hasn't landed yet (rare), the CPU blocks briefly until it does.

**The ping-pong buffer:** Since the GPU is writing step N's results to a host buffer while the CPU is simultaneously reading step N-1's results, they cannot share the same buffer. We allocate two pinned host buffers and alternate between them each step. At step N the GPU writes to buffer A while the CPU reads from buffer B; at step N+1 the roles flip. This avoids a data race that CUDA stream ordering alone cannot prevent — stream ordering governs GPU-side operations, but the CPU's read of pinned memory is not synchronized by the stream.

**Lookahead guard:** Because launch runs before resolve, a request that finished at step N-1 (via EOC) is still present in step N's batch — the CPU hasn't had a chance to remove it yet. The runner detects this by checking the request's done flag in the pool before launching, and routes finished requests to a dummy padding row. The graph still runs over these slots (CUDA Graph requires a fixed batch shape), but their outputs are discarded during the next resolve. This prevents double-counting finished requests.

Therefore, CPU and GPU can be each separated worker to give us more space for parallelization, they communicate through a shared ping-pong buffer.

#### Torch Compile

We also evaluated `torch.compile` as a potential optimization shortcut. However, since our manual CUDA Graph migration had already eliminated the bulk of kernel launch overhead, `torch.compile` offered only marginal throughput improvements. Plus, it introduced a massive compilation penalty during model warmup, severely damaging our cold-start latency. We ultimately chose to remove it—as a pragmatic engineering trade-off favoring fast system initialization over redundant runtime optimizations.

### Vocoder: Batched Decoding and Windowed Streaming

#### Batched Decoding

**Why:** Under high concurrency, multiple AR decode loops finish at nearly the same time — they enter the pipeline together, generate similar-length utterances, and race to the vocoder stage simultaneously. With 16 concurrent requests each taking ~15ms to vocode, the last request in line waits 240ms just for its turn — turning a fast stage into a tail-latency killer.

**How:** We batch vocoder calls using a short collection window (2ms / up to 4 requests). Before decoding, each request's delayed codes are un-delayed (reversing the delay pattern) and special tokens (BOC/EOC) are clamped to valid codec range. To avoid wasting compute on padding, we use bucketed batching — grouping sequences by length so that each batch contains only same-length items. Sequences that share a length are stacked and decoded in a single GPU call; sequences with unique lengths decode individually. This eliminates the tail-latency problem: instead of 16 serial vocoder calls, we issue a handful of batched calls.

![Batched Decoding](images/tts-opt-batched-decoding.png)

#### Windowed Streaming

**Why:** Without streaming, the user hears nothing until the entire AR decode loop completes — hundreds of steps of silence. So we want to stream the process to minimize TTFB (time to first byte). But you can't just naively chop the code sequence into chunks and decode each independently like LLM: neural audio codecs produce audible clicks at every splice boundary because the codec's internal convolution state is disrupted. On top of that, the delay pattern means the trailing rows in any mid-stream snapshot have incomplete high-layer codebooks — decoding them injects noise.

**How:** To manage this irreducible latency boundary smoothly, we tuned three parameters for our streaming window:

- **Stride (75 frames):** Accumulates roughly 3 seconds of delayed codes before triggering a vocoder decode step.
- **Overlap (8 frames):** Looks back into the previous window to eliminate seam artifacts and clicks during audio stitching.
- **Holdback (4 frames):** Retains trailing frames where high-layer codebooks are still incomplete, preventing noise injection during mid-stream decodes.

![Windowed Streaming Detail](images/tts-opt-streaming.png)

For streaming, we accumulate codes until a stride threshold (75 frames, ~1 second of audio) before triggering a decode — amortizing kernel launch overhead across a meaningful chunk. When decoding, we overlap by looking 8 frames back into the previously decoded region, re-decoding them jointly with new tokens so the codec sees continuous context across boundaries. We then extract only the delta (new samples past the overlap) and crossfade-blend it with the held-back tail of the previous chunk using a linear fade-in/fade-out envelope — smoothing any residual amplitude mismatch at the splice point.

Finally, a holdback of 4 frames retains the trailing rows where high-layer codebooks are still filling in due to the delay pattern. These incomplete rows are only released on the final flush when the full sequence is available.

The delay pattern also creates an irreducible startup cost: the vocoder needs at least N rows (N = number of codebooks) just to reverse the pattern and produce the first audio frame. Combined with the stride, the actual TTFB lands at ~300–400ms at our measured RTF — well under the 500ms conversational threshold.

## Benchmark Result

Referring directly to the data from the sglang-omni benchmark reference (`benchmarks/eval/benchmark_tts_seedtts.py`):

Throughput on Seed-TTS (full set, EN=1088, ZH=2020). Client max-concurrency sweep against a Higgs server (`max_running_requests=16`, bf16, CUDA Graph on, `torch.compile` off). Single reference run. Hardware: 1× H200. Last verified: 2026-05-25.

![Benchmark Results 1](images/tts-opt-benchmark-1.png)

![Benchmark Results 2](images/tts-opt-benchmark-2.png)

<span style="color:red">**[Placeholder]** Confirm this is the data we want, also we need to get the baseline data for comparison.</span>

## Conclusion

The best system optimizations don't come from blindly applying trendy techniques; they come from a deep understanding of the theory, desire for building elegant systems, and clean engineering trade-offs.

If you are interested in our project, please go to [sglang-omni repo](https://github.com/sgl-project/sglang-omni) to give it a try, to experience the exhilarating performance. If you are interested in making a contribution, I'd love to talk, please don't hesitate to reach out.

## Join Us

SGLang-Omni is an open community project, and it is still growing fast. Cross-node multi-stage pipelines, fuller diffusion-stage support, and end-to-end RL training integration are all underway. If multi-stage inference is the kind of problem you find beautiful — whether you come from a systems background or arrive halfway, whether you specialize in kernel optimization or scheduling logic — **we are actively recruiting contributors**. Come build a truly industrial-grade omni-serving stack with us: open a PR, join the discussion, or say hi in the community channels linked below.

## Acknowledgments

**SGLang-Omni** — Haoguang Cai, Shangming Cai, Qiujiang Chen, Jiaxin Deng, Wenyao Gao, Yifei Gao, Jingwen Gu, Yitong Guan, Chenchen Hong, Hao Jin, Xinli Jing, Shenggui Li, Junrong Lin, Xinyu Lu, Yuan Luo, Ratish Palanisamy, Mick Qian, JinTao Qu, Shuai Shi, Chao Wang, Richard Wang, Shuwen Wang, Zijie Xia, Yuhao Yang, Xuesong Ye, Yue Yin, Fan Yin, Gaokai Zhang, Xiaoyu Zhang, Yichi Zhang, Chenyang Zhao.

**Higgs Audio v3 TTS (Boson AI)** — Mu Li, Alex Smola, Lindsey Allen. Silin Meng, Ke Bai. Ruskin Raj Manku, Huapeng Zhou, Dongming Shen, Jonah Mackey, Erik Li, Weisu Yin, Yizhi Liu, Xinyu Wang, Hao Yu.

## Learn More

- **Model:** [boson-sglang/higgs-audio-v3-generation-4B-base](https://huggingface.co/boson-sglang/higgs-audio-v3-generation-4B-base)
- **Serving framework:** [SGLang-Omni on GitHub](https://github.com/sgl-project/sglang-omni)
- **Documentation:** [SGLang-Omni docs](https://sgl-project.github.io/sglang-omni/) · [Higgs TTS cookbook](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html)
- **Higgs optimization roadmap:** [#478](https://github.com/sgl-project/sglang-omni/issues/478)
- **Design background:** *SGLang-Omni: Redesigning the Inference Framework for Multi-Stage Generative Models*
