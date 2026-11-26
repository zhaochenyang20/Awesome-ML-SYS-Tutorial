# Getting Qwen3-Omni to Speak Sooner

A user has finished asking a question. Text starts appearing on screen, but the headphones are still silent. To get Qwen3-Omni speaking sooner, we need to follow that first bit of audio through the entire inference pipeline: where the work runs, where it waits, and where a synchronization point or data handoff holds it up.

With ordinary text generation, delivery starts as soon as the first token arrives. Speech generation has more work to do: turn the meaning into discrete audio codes, then turn those codes into a waveform. Talker needs output from Thinker before it can continue. Even after Talker produces a frame, Code2Wav may not yet have enough frames for a decoding window. Data also has to move between stages. A result can be finished on the GPU and still be sitting in a queue.

We start by breaking down that time. How long did each stage compute? How long did it wait before it could start? Once it finished, how long did the result take to reach the next stage? Those answers tell us whether to reduce computation, change the size of each unit of work, or address a synchronization point or handoff. After the change, we follow the same request again to see whether the time we saved actually reaches the user.

This article follows that path, from input preparation and message delivery to frame-by-frame generation and waveform decoding. The implementations differ, but we keep asking the same practical questions. Which work has to wait for new input? Which state hasn't changed and can be reused? What does the next stage need, and when does it need it?

## 1. Build a cost model along the request path

### 1.1 The stages behind a piece of audio

The speech path has seven configured stages: `preprocessing`, `image_encoder`, `audio_encoder`, `thinker`, `decode`, `talker_ar`, and `code2wav`. The image and audio encoders run as needed for the input. Predictor runs inside Talker.

Input first passes through preprocessing and any required encoders. Thinker handles understanding and text generation. Generated text returns through the decode path, while the state needed for speech generation goes to Talker. Talker then generates codec frames step by step, with its internal Predictor filling in the remaining codebooks. Code2Wav takes these discrete codes and uses the left context to produce a continuous waveform.

![A request produces text through Thinker and audio through Talker and Code2Wav. Data readiness, queueing, and delivery create separate timing boundaries.](images/qwen3-omni-serving-optimization/01-request-path.png)

*Figure 1: Follow compute, waiting, and delivery along the request path. Arrows represent data dependencies. A deployment can colocate stages on one GPU or place them separately.*

These stages can overlap, but each downstream stage still has to wait for the data it needs. Talker needs enough input, and Code2Wav needs a valid window. Even after the data arrives, a busy downstream stage may keep it queued. Timing a single `forward()` misses all of this waiting.

In an early short-prompt trace, GPU compute took about 3.2–3.8 ms, while the interval from prefill to the first output reached 76.3 ms. The host issued roughly 2,400 CUDA API calls. This historical trace pointed the investigation toward host dispatch, synchronization, and waiting. [Original record](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/51fa3246427078066c72c2c117ff345ba260d4a5/sglang/sglang-omni/qwen3-omni-serving-optimization-zh.md)

### 1.2 First decide which event the user is waiting for

| Metric | Observed event | What it tells us |
| --- | --- | --- |
| TTFT | The client receives the first text | How soon the user can see an answer |
| TTFA | The client receives the first validated, nonempty PCM audio chunk | How soon the service starts delivering audio |
| Inter-chunk gap | The arrival interval between adjacent audio chunks | Whether subsequent audio arrives in time |
| E2E latency | The interval from the request to the defined completion event | How long the full request takes |
| Request throughput | Completed requests / measurement window | The service's sustained request-processing capacity |
| WER | Transcription errors relative to the reference text | Whether the output preserves the intended spoken content |

The TTFA measured here does not include an actual player's buffering or the playback device's latency. An early first PCM chunk does not guarantee that later chunks will keep up. RTF divides generation time by audio duration; we still need to read it alongside the full output duration and its quality.

The choice of optimization also depends on which metric we want to improve. Waiting for more requests can make batching more efficient, but it may delay the first audio. A smaller first window can deliver audio sooner, while increasing the frequency of later scheduling work. Whether E2E includes writing WAV files and metadata to disk also affects the numbers. The results below retain their original timing boundaries, and percentages from different experiments cannot simply be added together. We use C to denote request concurrency: C16, for example, means a concurrency of 16.

## 2. Thinker: move input preparation out of repeated execution

Thinker prefill handles the input sequence and multimodal positions, and prepares the hidden states needed downstream. Request content changes each time, but not all dynamic work has to wait until the model executes. Here, we organize that information first, then let the model compute over the batch.

### 2.1 Record positions early and merge the whole batch at once

The old multimodal merge iterated over “request × modality,” calling operations such as `any()` and `sum().item()` on GPU masks. Python needed the device result before it could decide what to do next, potentially making the CPU stop and wait for the GPU. More requests in a batch meant repeating these small synchronizations.

After the change, request construction records placeholder positions up front. The merge then prepares indices for the whole batch and uses batched tensor operations to put encoder outputs in the correct positions. This moves the dynamic work into input preparation, so model execution no longer has to keep asking the GPU, “Does this row contain audio?” Position encoding received a similar change, replacing item-by-item construction with batched operations. [Multimodal merge](https://github.com/sgl-project/sglang-omni/pull/1161) · [MRoPE](https://github.com/sgl-project/sglang-omni/pull/1160)

![Move request-specific placeholder discovery before execution, then gather encoder outputs and scatter them into the batch in one tensor path.](images/qwen3-omni-serving-optimization/02-input-preparation.png)

*Figure 2: Organize dynamic positions first, then move data in a batch. The optimization removes device-scalar synchronization from the request loop; encoder outputs must still land at their corresponding token positions.*

In a component measurement with 8,192 multimodal tokens, the single-modality merge fell from 2.108 ms to 1.005 ms, while interleaved image/video input fell from 7.825 ms to 2.311 ms. The batched path also needs a temporary source buffer, about 32 MiB in this measurement. These results cover the merge operation itself; a stable, independently measured end-to-end TTFT gain has not been established. [Component measurements](https://github.com/sgl-project/sglang-omni/pull/1161)

If information is already known before model computation, organize it during input preparation where possible. The same approach applies to other changes: fewer host/device round trips, and more regular data for the execution that follows.

### 2.2 Define output ownership when reusing execution buffers

Text output mainly uses logits. Speech output also needs to hand hidden states from selected layers to Talker. Each eager execution can produce new output tensors; with CUDA Graphs, output addresses must stay fixed, and later replays reuse the same buffer. Downstream readers must know which request a result belongs to and when it might be overwritten.

Here, hidden states get static buffers. The forward pass writes to fixed addresses, and downstream consumers read the valid rows and token ranges for the current request. Buffer capacity has to be budgeted together with request and graph limits. If initialization fails, the state left by capture also needs to be cleared so the next execution does not reuse invalid contents. [Static hidden buffers](https://github.com/sgl-project/sglang-omni/pull/1380) · [Initialization recovery](https://github.com/sgl-project/sglang-omni/pull/1532)

![A stable graph buffer is reused across replays. Request identity, valid rows, and ownership must remain explicit until each consumer finishes.](images/qwen3-omni-serving-optimization/03-buffer-ownership.png)

*Figure 3: Stable addresses satisfy only the replay requirement. Consumers must still read the correct request range and complete the necessary handoff before a later replay overwrites the buffer.*

We will run into this again with Code2Wav. Execution can be reused, but each output's lifetime still needs to be managed separately. Both have to be considered before asynchronous handoff is safe.

## 3. Pipeline: let consumers use ready data sooner

Finishing model computation does not mean a message immediately reaches the next stage. It may be waiting for a timer or a thread wake-up, or stuck behind an extra relay or expensive transport initialization. To address these waits, we need to follow each handoff along the message path.

### 3.1 Handle waiting, kernel submission, and transport setup separately

The early Audio Encoder path had three fixed costs. After the first message arrived, it still waited about 50 ms. Its multilayer eager forward issued many small kernels. Even transferring a small tensor could spend substantial time opening a CUDA IPC handle.

Each cost gets its own treatment. The wait becomes configurable, allowing an idle path to process requests that have already arrived. Suitable encoder shapes are captured as graphs for repeated execution. Small audio payloads use an appropriate pooled relay, avoiding disproportionate setup costs for a small tensor. A tensor tens of MiB in size and one of a few hundred KiB do not necessarily belong on the same transport path. [Wait policy](https://github.com/sgl-project/sglang-omni/pull/1564) · [Encoder graphs and transport](https://github.com/sgl-project/sglang-omni/pull/1628)

In a historical colocated test on a single H200, these changes reduced C1 TTFT p50 from 171 ms to 74 ms and TTFA p50 from 297 ms to 208 ms, while increasing request throughput by 16.1%. That baseline still included the old 50 ms wait. The combined result therefore cannot be added to the separate benefit of removing the wait, or interpreted as a pure incremental gain over a later main branch. The C8 throughput change was within noise, and C32 was roughly flat. The quality check used 96 samples for WER, and speaker similarity stayed within the eager execution variability band. [Experiment record](https://github.com/sgl-project/sglang-omni/pull/1628)

### 3.2 Handle work that has already arrived in one wake-up

The old outbox path crossed between a worker thread and the event loop for every message it read. After the change, the first read still blocks. Once it returns a message, nonblocking reads continue processing messages already in the queue. Each round handles at most 64 messages, then yields.

This path does not add a wait for future messages. It handles ready work within the same wake-up while preserving ordering, completion, and cancellation semantics. In an H100 FP8 colocated C16 SeedTTS comparison, output token throughput rose from 100.42 to 104.88 tok/s, about 4.4%, and TTFA p95 fell from 1.5665 s to 1.2991 s, about 17.1%. In a separate deterministic check, text, token counts, chunk counts, and WAV hashes matched across 160 request pairs. [Outbox drain](https://github.com/sgl-project/sglang-omni/pull/1384)

![Ready messages are drained within one wake-up, while consumer-side joins remove a forwarding stage without removing input dependencies.](images/qwen3-omni-serving-optimization/04-pipeline-handoff.png)

*Figure 4: Reduce the handoffs for data that is already ready. Batched draining handles messages already in the queue; consumer-side joins move the job of waiting for all inputs to the actual consumer.*

Another relay can be removed entirely. In the historical speech path, `mm_aggregate` only waited for inputs to arrive and then forwarded them. Thinker and Talker now each wait for the data they need. The request path drops from eight stages to seven, with input joining handled by the actual consumers. The text-only path keeps its original aggregation stage. [Consumer-side join](https://github.com/sgl-project/sglang-omni/pull/1548)

All required inputs still have to arrive; the intermediate forwarding step is what goes away. When checking whether a stage can be removed, first ask whether it changes the data or execution policy, then whether the consumer can take on that work directly.

### 3.3 Send only what will be read, and append without recopying old data

To reduce transfers, start by checking what downstream code actually reads. The generated assistant text stream and the prompt's multimodal hidden states serve different purposes. The generated text stream does not need to carry both an embedding and an auxiliary hidden tensor. This path keeps the embedding and falls back to hidden states when the embedding is missing; the prompt's multimodal conditioning is still retained where needed. [Thinker → Talker data path](https://github.com/sgl-project/sglang-omni/pull/1574)

The data also determines which transport path to use. Small CPU stream chunks of at most 16 KiB, with no tensors in their metadata, can go directly into control messages. Larger tensors continue through the data channel. Talker prefill keeps tensors throughout, removing the round trip through `cpu().tolist()` and back into a newly constructed tensor.

The pending-text queue can be examined the same way. Every time the old implementation appended new rows, it called `torch.cat` to join them with the old rows that had not yet been consumed. A longer backlog meant more old data copied repeatedly. With a queue of device tensor chunks, a cursor tracks consumption in FIFO order. Appending only adds a new chunk; an old chunk is released after it has been consumed. [Pending-text queue](https://github.com/sgl-project/sglang-omni/pull/1611)

![Transmit the fields the consumer needs, then append immutable tensor chunks to a FIFO instead of concatenating the unconsumed backlog on every arrival.](images/qwen3-omni-serving-optimization/05-payload-queue.png)

*Figure 5: The consumer determines which fields are sent, and the queue stores data in chunks. A cursor tracks consumption, so appending new content no longer recopies unconsumed rows.*

In an H100 FP8 colocated comparison, all 4,200 end-to-end requests per arm succeeded. The original 31,120 queue-level cat calls were removed, along with 457.4 MiB of repeated copies of old rows. C16 request throughput changed from 7.351 to 7.428 req/s, about 1%, leaving the overall end-to-end result roughly flat. The repeated copying was removed, but this does not support a claim of a matching reduction in user latency. [Mechanism and end-to-end results](https://github.com/sgl-project/sglang-omni/pull/1611)

## 4. Talker: update state only when it changes

Thinker's prefill happens at the start of a request. Talker runs an autoregressive loop throughout the speech output. It generates part of the codec information, then its internal Predictor fills in the remaining codebooks; the current frame's result feeds into the next frame. An extra synchronization or unnecessary copy in this loop keeps adding cost as more frames are generated.

### 4.1 Rebuild when state changes

If batch membership and sampling parameters have not changed, rebuilding sampling state, masks, and metadata for every frame produces no new information. We keep that state across steps and explicitly update it when requests join or leave, or when parameters or ownership change. [Sampling state reuse](https://github.com/sgl-project/sglang-omni/pull/1043)

Sampling state is used every frame; rebuilding can wait until the state it depends on changes. Once a request leaves, its old state must not occupy the space assigned to a new request.

![Sampling metadata is rebuilt when membership or parameters change. Predictor attention consumes shared K/V heads without materializing repeated copies.](images/qwen3-omni-serving-optimization/06-talker-state.png)

*Figure 6: In Talker's repeated execution path, sampling metadata is updated when its inputs change, and attention uses GQA to express K/V sharing. Both changes reduce data movement or preparation work in the frame loop.*

In the profile, pageable H2D events fell from 15.13 to 4.26 per frame, and stream synchronizations on the forward thread fell from 16.09 to 5.23 per frame. These are event counts and cannot be directly converted into milliseconds saved. The corresponding end-to-end changes remained close to the variation between repeated runs. [Measurements](https://github.com/sgl-project/sglang-omni/pull/1043)

### 4.2 Let the backend express sharing directly

Predictor attention also has intermediate tensors that do not need to be created. The old path explicitly expanded the K/V heads before running attention. With native GQA, the backend can express K/V sharing directly, avoiding some small copy kernels. [Native GQA](https://github.com/sgl-project/sglang-omni/pull/1164)

Predictor already runs inside a CUDA Graph, so this change reduces memory work within replay. CUDA Graphs reuse the submission sequence, but unnecessary tensor expansion can still remain inside the graph. Finding that cost means looking further into what actually executes within a frame.

This follows the same approach as input preparation: check whether the values a computation depends on have changed, then decide which preparation steps need to run again. The same rule helps with maintenance. Whenever we add state, we need to spell out what it depends on and which events invalidate it.

## 5. Code2Wav: organize computation around the streaming output protocol

Talker produces codec frames one at a time. Code2Wav turns them into waveform windows. Each window contains new frames and may also include left context. Before choosing execution shapes, we need to settle when the first window goes out, how many frames each later window adds, and which part of the waveform belongs to the current output.

### 5.1 Match execution shapes to actual window lengths

In the default serial path, each new chunk contains 10 frames, with up to 25 frames of left context. As context accumulates, typical input lengths for normal windows are `T=10/20/30/35`.

| Window | Left context | New frames | Total input length |
| --- | ---: | ---: | ---: |
| First | 0 | 10 | 10 |
| Second | 10 | 10 | 20 |
| Third | 20 | 10 | 30 |
| Subsequent full windows | 25 | 10 | 35 |

The history frames participate in the current window's computation, but the output is cropped to deliver only the waveform corresponding to new frames. Capturing CUDA Graphs at these actual lengths can reduce launch overhead without making every small window execute at the maximum shape. [Exact-shape graph](https://github.com/sgl-project/sglang-omni/pull/1101)

![Streaming chunk geometry defines graph keys. Ready windows are grouped into supported batch buckets under a fixed memory budget.](images/qwen3-omni-serving-optimization/07-code2wav-windows.png)

*Figure 7: The output protocol determines the time dimension first; ready windows then determine the batch. Graph coverage is constrained by both shape and memory budget.*

If we choose a larger batch or broader graph coverage first and make requests wait for it, users hear the first audio later. We need to establish valid windows and delivery timing first, then fit the execution strategy to those requirements.

### 5.2 Batch ready windows with bounded waiting and memory use

When windows from several requests are ready, bounded batching can execute them together, dispatching when the batch condition or waiting deadline is reached. The zero-wait configuration forms batches only from windows that are already ready; it does not deliberately wait for future windows. The later chunk-aligned graph design brings valid windows, batch buckets, and the memory budget into the same scheduling rules. [Bounded batching](https://github.com/sgl-project/sglang-omni/pull/1126) · [Chunk-aligned graphs](https://github.com/sgl-project/sglang-omni/pull/1237)

In the H100 Code2Wav component experiment, the harness simulated arriving Talker codec frames, with 20 windows per request and three repeats. Under a 2% graph memory budget, the zero-wait configuration improved component throughput by about 11–15% at the tested C≥8 levels, while C1 was roughly unchanged. This configuration retained 12 graphs covering `B=1/2/4` and `T=10/20/30/35`, using about 634 MB. When larger batches exceeded the budget, execution was split into smaller batches. [Component experiment](https://github.com/sgl-project/sglang-omni/pull/1237)

These measurements cover the vocoder component and do not establish an end-to-end speedup for the full Qwen3-Omni system. The experiment explicitly enabled batched mode, which remains off by default. Waveforms were compared within a tolerance, with a worst-case SNR of 35.26 dB and no failed checks. Longer waits and larger memory budgets belong to separate configurations and cannot be folded into these results.

This path also needs to handle missing graph coverage and execution failures. If a graph for a larger batch is missing, the scheduler first splits the work into smaller captured batches, falling back to serial execution when necessary. Unsupported shapes can select eager execution before replay. Once replay has started, execution errors must still be reported as failures, with subsequent requests taking the fallback path according to the disabled state. Silently rerunning the same request could consume output twice or hide corrupted state.

## 6. Follow computation through to delivery

Once the GPU finishes a window, the result still needs to be copied, cropped, converted, and sent to the client. If those steps occupy the scheduler thread for too long, other ready work waits behind them. Following the request this far means looking at output handling alongside computation.

### 6.1 Overlap output handling while preserving ownership

In the CUDA serial Code2Wav path, each stream has an output pipeline with a depth of 2. It draws pinned slots from a shared pool and tracks asynchronous copies with CUDA events, allowing the scheduler thread to continue with other work. Once a copy completes, CPU output handling can overlap with subsequent GPU computation. The first and final windows remain synchronous. [Output overlap](https://github.com/sgl-project/sglang-omni/pull/1567)

After a window replay, FP32 conversion, asynchronous D2H into the slot, and event recording are all enqueued on the same CUDA stream. This ordering ensures that the next replay cannot overwrite the borrowed graph output before the copy completes. After completion, the host first copies the valid result into independent CPU memory, releases the slot, and then sends that independent result. Slot reuse is now separate from downstream consumption.

![A depth-two output pipeline separates borrowed graph output from slot-owned pinned data. Completion events and an owned CPU copy separate delivery from safe slot reuse.](images/qwen3-omni-serving-optimization/08-code2wav-output.png)

*Figure 8: After D2H completes, take an independent CPU copy before releasing the shared slot and delivering the result. Output handling can overlap with computation, while the first and final windows retain their synchronous boundaries.*

Every arriving Talker frame triggers an event check. As soon as completion is observed, the corresponding result is emitted. When the request ends, the pending window is sent as a separate message before decoding the final tail, preserving message boundaries.

Cancelling a request does not make slots that are still in use immediately reclaimable. Abort first moves them into a retired queue; the scheduler checks completion before reclaiming them. If a copy or event-recording operation fails, the slot is quarantined so its buffer cannot be reused while completion is unknown. The later shared `PinnedTransferSlot` implementation preserves these ownership states. [Output slot lifecycle](https://github.com/sgl-project/sglang-omni/pull/1567) · [Shared transfer slot](https://github.com/sgl-project/sglang-omni/pull/1759)

The existing profile supports reduced scheduler-thread occupancy, but does not establish a reliable end-to-end speedup. Protocol-level comparisons checked message consistency for identical inputs. The later transfer-slot change also ran bitwise comparisons for 20 deterministic requests and 12 real-weight replay cases. These checks cover their specific inputs and implementations; they do not establish quality for the complete service.

### 6.2 After each change, return to the same request

How much input preparation Thinker can do ahead of execution affects how regular its execution path can be. Talker first checks whether state has changed before deciding whether to rebuild it in the frame loop. For the pipeline and Code2Wav, we need to know what consumers need and when they need it before arranging data transfers and windows.

Those questions guide the choice of mechanism. Batched indexing reduces repeated queries of device state; unconsumed data stays in a FIFO; GQA expresses K/V sharing, and CUDA Graphs reuse the submission sequence. Each choice also comes with constraints. Static addresses need clear ownership, batching must account for first-window timing, and broader graph coverage needs memory.

After making a change, we still need to look at the user's request. Multimodal merge became faster as a component, without a consistent, isolated TTFT improvement. The pending-text queue eliminated repeated copying, while end-to-end performance stayed roughly unchanged. Outbox draining improved both output-token throughput and first-audio tail latency in its historical configuration. Keeping those results distinct tells us whether to keep shortening the current path or look elsewhere for waiting.

This is how we work on Qwen3-Omni: follow the request path, measure it, explain the results, and make changes. After a local change, we first check which work was actually removed, then measure complete requests to see whether that work was part of the user's wait.
