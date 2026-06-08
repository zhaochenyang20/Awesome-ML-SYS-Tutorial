# Optimizing TTS Inference: Engineering Lessons from Profiling to Streaming in SGLang Omni

*Yichi Zhang — June 2026*

*Originally published on [Medium](https://medium.com/@yichizhang602/optimizing-tts-inference-engineering-lessons-from-profiling-to-streaming-in-sglang-omni-00d06e3fc78d)*

---

Optimizing Text-to-Speech (TTS) inference looks a lot like LLM optimization on paper, but the actual engineering bottlenecks are entirely different. Instead of just listing our optimizations, this post breaks down the mechanical sympathy required to make this pipeline fast—the bottlenecks we hit, the host-to-device pitfalls, and the architectural trade-offs we made along the way.

## The Higgs TTS Pipeline Under the Hood

![Higgs TTS Pipeline](images/tts-opt-pipeline.png)

To optimize the system, we first have to understand how data moves through the four stages of the Higgs pipeline:

- **Preprocessing (CPU):** Text tokenization and reference audio loading. This is purely IO-bound and handles no GPU compute.
- **Audio Encoder (GPU):** Uses HiggsAudioCodec (a DAC-like neural audio codec) to convert the reference audio waveform into discrete tokens. Output shape is `[T, 8]`, where T is the number of time steps and 8 represents parallel codebook tokens per step.
- **TTS Engine (GPU Backbone):** The core autoregressive (AR) model based on a Qwen3 LLM architecture. It generates the multi-codebook tokens step-by-step.
- **Vocoder (GPU):** A DAC Decoder that converts the generated tokens back into an audible waveform. It shares weights with the Encoder.

## Two Architectural Gotchas: Embeddings & The Delay Pattern

Higgs differs from standard LLMs in two major ways:

- **Fused Multi-Codebook Embedding:** Instead of maintaining eight independent embedding tables, Higgs concatenates them. We look up all codebooks in a single pass, outputting a tensor of shape `[B, N, V]` (Batch size, 8 Codebooks, Vocabulary size).
- **The Multi-Codebook Delay Pattern:** This is the trickiest part of the architecture. Higgs generates 8 tokens per AR step, but they have a strict hierarchy. Codebook 0 handles coarse structure (pitch, intonation), while Codebook 7 handles high-frequency textures. Generating them completely in parallel ruins audio quality; generating them sequentially ruins throughput.

To balance this, Higgs uses a Delay Pattern where Codebook i is delayed by i time steps relative to Codebook 0.

- Step 0: CB0 activates
- Step 1: CB0 → CB1 activate
- Step 2: CB0 → CB1 → CB2 activate
- … and so on.

![Delay Pattern State Machine](images/tts-opt-delay-pattern.png)

This introduces a state machine with four phases: Delay Stage (staggered activation), Active Stage (normal sampling), Wind Down (triggered when CB0 hits the End-of-Character token), and Finished.

**The Catch:** Our original implementation uses if-else statements and for loops to control the flow of this delay pattern, which needs to be completely re-written in tensor operations to support CUDA Graph. Will be discussed later in the CUDA Graph chapter.

## Profiling: Where is the Time Actually Spent?

Before writing code, we profiled the naive pipeline and identified three core bottlenecks:

- **AR Decode Dominates:** A typical 10-second speech request requires 400 to 800 decode steps. Every single step involves a backbone forward pass, head projection, sampling, and Device-to-Host (D2H) synchronization. A tiny 0.1ms overhead per step inflates end-to-end latency by nearly 80ms.
- **The Encoder is Heavy but Static:** A single encoding pass takes 50–100ms. However, in production, users often reuse the same reference audio across multiple prompts.
- **Vocoder Queuing:** The vocoder is fast (~10ms per call), but under high concurrency, multiple AR generation loops finish at the exact same time, creating a massive serial bottleneck at the vocoder stage.

With those bottlenecks, we applied targeted optimization strategies. Here is a macro-level graph to show our strategies, and we will be discussing each strategy in the next section.

![Optimization Strategies Overview](images/tts-opt-strategies.png)

## Layer-by-Layer Optimizations

### 1. Encoder: Bypassing the Compute with LRU Caching

**Why:** Since the fastest compute is the compute you don't do, we introduced an LRU Cache for the reference audio. If a user sends multiple prompts using the same reference voice, we skip the Encoder entirely and fetch the pre-computed delayed tokens instantly.

We also experimented with online batched encoding by bucketing incoming audio by length. While it improved raw throughput on paper, it created a new problem in production: GPU utilization shifted from smooth patterns to intermittent spikes, causing severe resource contention with the concurrent AR decode loops. We ultimately moved batched encoding offline (used strictly for bulk server warmups) and kept online encoding isolated.

### 2. AR Decode: Shaving Off Every Microsecond

Since AR decode is our primary bottleneck, we focused on eliminating kernel launch overhead and synchronization stalls.

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

**Why:** The vanilla pattern of CPU–GPU synchronization has the GPU and CPU processing in the same flow, stopping and waiting for each other. We discovered this pattern is inefficient since the D2H sync time can stack up to very high during AR decoding. To hide the remaining D2H synchronization time, we want to discover a pattern to let GPU and CPU work simultaneously and not wait for each other.

**How:** We implemented pipeline one-step overlapping. The GPU immediately kicks off the computation for step t+1 without waiting for the CPU to finish processing the sampling results from step t. This keeps the GPU fully utilized and completely masks the host transfer overhead.

The key insight is that the token causal chain can close entirely on the GPU, without waiting for the CPU, and that's why we will be able to compute t+1 step on GPU even without CPU's result on step t.

The overall timeline would be as shown in the picture:

![Async Decode Timeline](images/tts-opt-async-decode.png)

On the GPU, the sampler pool `[max_bs+1, 8]` stores the most recent `last_codes` for each request; each request gets a row in the pool. And KV cache stores paged attention KV for decoding computation. CG active buffer stores the copy gathered from the sampler pool. And CG active buffer is a fixed address buffer so it is compatible with CUDA Graph.

On the CPU, it doesn't move codebook token to anywhere, it only reads from the host staging buffer which is copied from the CG active buffer through D2H copy. And CPU fill in three things: row table: `request_id → pool row index`; sampler params: `temperature, top_p, top_k`; Asynchronously handle last step's output: add more request / detect EOC and remove finished request → reflected in the next step's `cg_row_indices`, and in gather step, the CG active buffer will record `row_indices`. By this way, we make async updates possible.

Therefore, CPU and GPU can be each separated worker to give us more space for parallelization, they communicate through a shared ping-pong buffer. (ping-pong buffer means it's two buffer sections and one is read while the other one is written in one simultaneous step, and flip the role in next step) By this design, we will be able to achieve async decode.

#### Torch Compile

We also evaluated `torch.compile` as a potential optimization shortcut. However, since our manual CUDA Graph migration had already eliminated the bulk of kernel launch overhead, `torch.compile` offered only marginal throughput improvements. Plus, it introduced a massive compilation penalty during model warmup, severely damaging our cold-start latency. We ultimately chose to remove it—as a pragmatic engineering trade-off favoring fast system initialization over redundant runtime optimizations.

### 3. Vocoder: Batched Decoding and Windowed Streaming

#### Batched Decoding

**Why:** Under high concurrency, multiple AR decode loops finish at nearly the same time — they enter the pipeline together, generate similar-length utterances, and race to the vocoder stage simultaneously. With 16 concurrent requests each taking ~15ms to vocode, the last request in line waits 240ms just for its turn — turning a fast stage into a tail-latency killer.

**How:** We collect requests that arrive within a tight batching window (`max_batch_wait_ms=2ms`) and decode them in a single GPU call. The mechanics: all codebook tensors are zero-padded to the max sequence length in the batch, stacked into a single `[B, num_codebooks, max_len]` tensor, and decoded through one `codec.decode_batch()` invocation. After decoding, each request's waveform is trimmed back to its true length (`length × samples_per_frame`), discarding the padding region.

![Batched Decoding](images/tts-opt-batched-decoding.png)

#### Windowed Streaming

**Why:** Without streaming, the user hears nothing until the entire AR decode loop completes — hundreds of steps of silence. So we want to stream the process to minimize TTFB (time to first byte). But you can't just naively chop the code sequence into chunks and decode each independently like LLM: neural audio codecs produce audible clicks at every splice boundary because the codec's internal convolution state is disrupted. On top of that, the delay pattern means the trailing rows in any mid-stream snapshot have incomplete high-layer codebooks — decoding them injects noise.

**How:** To manage this irreducible latency boundary smoothly, we tuned three parameters for our streaming window:

- **Stride (75 frames):** Accumulates roughly 1 second of delayed codes before triggering a vocoder decode step.
- **Overlap (8 frames):** Looks back into the previous window to eliminate seam artifacts and clicks during audio stitching.
- **Holdback (4 frames):** Retains trailing frames where high-layer codebooks are still incomplete, preventing noise injection during mid-stream decodes.

![Windowed Streaming Detail](images/tts-opt-streaming.png)

For streaming, we accumulate codes until a stride threshold (75 frames, ~1 second of audio) before triggering a decode — amortizing kernel launch overhead across a meaningful chunk. When decoding, we overlap by looking 8 frames back into the previously decoded region, re-decoding them jointly with new tokens so the codec sees continuous context across boundaries. We then extract only the delta (new samples past the overlap) and crossfade-blend it with the held-back tail of the previous chunk using a linear fade-in/fade-out envelope — smoothing any residual amplitude mismatch at the splice point.

Finally, a holdback of 4 frames retains the trailing rows where high-layer codebooks are still filling in due to the delay pattern. These incomplete rows are only released on the final flush when the full sequence is available.

The delay pattern also creates an irreducible startup cost: the vocoder needs at least N rows (N = number of codebooks) just to reverse the pattern and produce the first audio frame. Combined with the stride, the actual TTFB lands at ~300–400ms at our measured RTF — well under the 500ms conversational threshold.

## Benchmark

Referring directly to the data from the sglang-omni cookbook:

Throughput on Seed-TTS EN (full set, N=1088 per run). Client — max-concurrency sweep against a Higgs server (`max_running_requests=16`, bf16, CUDA Graph on). Each row is the mean of 3 runs. Hardware: 1× H100.

![Benchmark Results 1](images/tts-opt-benchmark-1.png)

![Benchmark Results 2](images/tts-opt-benchmark-2.png)

## Conclusion

The best system optimizations don't come from blindly applying trendy techniques; they come from a deep understanding of the theory, desire for building elegant systems, and clean engineering trade-offs.

If you are interested in our project, please go to [sglang-omni repo](https://github.com/sgl-project/sglang-omni) to give it a try, to experience the exhilarating performance. If you are interested in making a contribution, I'd love to talk, please don't hesitate to reach out.

## Join Us

SGLang-Omni is an open community project, and it is still growing fast. Cross-node multi-stage pipelines, fuller diffusion-stage support, and end-to-end RL training integration are all underway. If multi-stage inference is the kind of problem you find beautiful — whether you come from a systems background or arrive halfway, whether you specialize in kernel optimization or scheduling logic — **we are actively recruiting contributors**. Come build a truly industrial-grade omni-serving stack with us: open a PR, join the discussion, or say hi in the community channels linked below.

## Acknowledgments

**Higgs Audio v3 (Boson AI)** — Lead: Mu Li, Alex Smola, Lindsey Allen. Pre-train: Silin Meng, Ke Bai. Post-train: Ruskin Raj Manku, Huapeng Zhou. Data: Silin Meng, Dongming Shen. Evaluation: Jonah Mackey, Ke Bai, Ruskin Raj Manku. Inference: Huapeng Zhou, Silin Meng, Erik Li, Weisu Yin, Yizhi Liu. Release: Alex Chen, Ke Bai, Silin Meng.

**SGLang-Omni** — Haoguang Cai, shangming cai, Qiujiang Chen, Jiaxing Deng, Wenyao Gao, Yifei Gao, Jingwen Gu, yitong guan, Chenchen Hong, Hao Jin, Xinli Jing, Shenggui Li, Junrong Lin, Xinyu Lu, Yuan Luo, Ratish Palanisamy, Qian Mick, jinjiang qu, Shuai Shi, Chao Wang, Richard Wang, Suwen Wang, Zijie Xia, Yuhao Yang, Xuesong Ye, Fan Yin, Gaokai Zhang, Xiaoyu Zhang, Yichi Zhang, Chenyang Zhao

## Learn More

- **Model:** [boson-sglang/higgs-audio-v3-generation-4B-base](https://huggingface.co/boson-sglang/higgs-audio-v3-generation-4B-base)
- **Serving framework:** [SGLang-Omni on GitHub](https://github.com/sgl-project/sglang-omni)
- **Documentation:** [SGLang-Omni docs](https://sgl-project.github.io/sglang-omni/) · [Higgs TTS cookbook](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html)
- **Higgs optimization roadmap:** [#478](https://github.com/sgl-project/sglang-omni/issues/478)
- **Design background:** *SGLang-Omni: Redesigning the Inference Framework for Multi-Stage Generative Models*
