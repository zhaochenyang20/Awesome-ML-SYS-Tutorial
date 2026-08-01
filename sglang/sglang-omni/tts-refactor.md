# Lifecycle Management and Framework Abstractions: Refactoring TTS Serving in SGLang Omni

**How should a general-purpose serving framework support speech models with vastly different architectures and runtime requirements?**

This was one of the questions we repeatedly returned to while designing SGLang Omni. Ideally, integrating a new model should be straightforward:

1. Declare the pipeline topology
2. Implement the model-specific computation
3. Let the framework handle scheduling, communication, and lifecycle management

Before this refactor, however, the TTS serving stack was still far from that goal. To add a new model, developers had to implement not only the core generation path—from text to acoustic latents and eventually to waveforms—but also a substantial amount of serving infrastructure inside the model directory. This included engine startup and scheduling, cross-process state transport, streaming vocoder state management, and cleanup on failures or request cancellation.

In practice, integrating a new model often meant rebuilding half of the serving stack. That was clearly not a sustainable boundary. We therefore spent a month redefining the responsibilities between models and the framework:

> **Models should focus on generation algorithms. The framework should own the recurring serving lifecycle.**

## The Refactoring Space and Challenges

As described in [Optimizing TTS Inference: Engineering Lessons from Profiling to Streaming in SGLang Omni](./tts-optimization.md), a typical TTS inference pipeline consists of three major stages:

> Reference audio encoding → autoregressive audio-token generation → waveform decoding with a vocoder

This shared pipeline creates natural opportunities to reuse scheduling, caching, and request-lifecycle infrastructure.

The real challenge came from the model-specific optimizations we had previously introduced to improve inference performance. Many of these optimizations were deeply integrated into serving concerns such as batching, cache-key construction, and streaming state management. For example:

- **Higgs** does not natively support streaming, so we implemented incremental output using overlapping windows and crossfading.
- **MOSS-TTS-Local** uses a separate causal Transformer vocoder. Serving it efficiently requires persistent codec sessions, CUDA Graph slot allocation, and batching across concurrent requests.
- **FishAudio S2-Pro** uses multiple codebooks, which means its KV cache must distinguish not only token IDs but also embedding-based inputs.

These optimizations directly affect throughput, latency, and streaming quality. They also allow model-specific assumptions to leak into the serving lifecycle.

The refactor therefore had to satisfy two goals at once: preserve the performance advantages of the existing implementations while moving as much repeated control flow as possible into the framework.

<div align="center">
  <img src="images/tts-opt-pipeline-overview.svg" alt="A multi-stage SGLang Omni TTS pipeline covering preprocessing, reference audio encoding, autoregressive generation, and vocoder decoding" width="78%">
  <p><em>Figure 1: A typical multi-stage TTS inference pipeline. The framework schedules the standard stages, while each model retains control over its generation logic and stage-specific computation.</em></p>
</div>

## What Did We Abstract?

As of July 30, 2026, the refactor had removed a net total of **2,840 lines of non-test implementation code**.

<div align="center">
  <a href="https://luojiaxuan.github.io/sglang-omni/tts-refactor/">
    <img src="images/tts-refactor-progress-2026-07-30.png" alt="The TTS Refactor Progress page showing a net reduction of 2,840 lines of non-test implementation code" width="96%">
  </a>
  <p><em>Figure 2: Progress snapshot from July 30, 2026. For the latest statistics and a commit-by-commit breakdown, see <a href="https://luojiaxuan.github.io/sglang-omni/tts-refactor/">TTS Refactor Progress</a>.</em></p>
</div>

Most of the deleted code came from serving mechanisms that had previously been reimplemented for each model. Examples include:

- Handwritten `to_dict` and `from_dict` methods for state transport, which are easy to break when a newly added field is not propagated everywhere.
- LRU eviction and concurrency control for reference-audio caching.
- Cleanup logic for interrupted or failed streaming requests.

After the refactor, the framework provides a shared skeleton for engine startup, state transport, caching, and vocoder lifecycle management. Model directories retain their own codec sessions, checkpoint parsing, sampling logic, and generation algorithms.

We also enforced a strict rule when defining this boundary:

> **Shared framework code must never branch on model names.**

Every model-specific difference must instead be represented through hooks, explicit state fields, or capability metadata.

<div align="center">
  <img src="images/tts-refactor-before-after.svg" alt="Before-and-after comparison showing six independent TTS serving stacks consolidated behind shared framework interfaces and model-specific hooks" width="96%">
  <p><em>Figure 3: Models retain their generation and codec implementations, while the framework manages the recurring serving lifecycle.</em></p>
</div>

The key abstractions introduced during the refactor include [`TtsEngineBuilder`](https://github.com/sgl-project/sglang-omni/pull/923), [`DeclarativeStateBase`](https://github.com/sgl-project/sglang-omni/pull/1050), [`ReferenceEncodeService`](https://github.com/sgl-project/sglang-omni/pull/926), [`BatchVocoderBase`](https://github.com/sgl-project/sglang-omni/pull/940), [`StreamingVocoderBase`](https://github.com/sgl-project/sglang-omni/pull/936), and [`OmniScheduler`](https://github.com/sgl-project/sglang-omni/pull/937). These are intentionally narrow, composable interfaces rather than a monolithic base class. A backend can reuse engine startup without adopting the streaming-vocoder abstraction, or reuse state transport without sharing its codec implementation.

## Shared Lifecycles Amplify Hidden Assumptions

Once previously isolated model logic moves into a shared framework path, an assumption that used to affect only one backend can suddenly affect many.

Caching is a good example. A conventional token-based cache assumes that two inputs are equivalent when their token IDs are identical. For FishAudio S2-Pro, however, part of the acoustic conditioning from the reference audio is not represented in the token IDs at all. Reusing the standard cache identity rules would therefore associate requests with the wrong reference conditions.

During the migration, we identified and fixed three particularly important classes of problems.

### A Cache Key Cannot Depend on Token IDs Alone

FishAudio S2-Pro represents reference audio using multiple VQ codebooks. Only codebook 0 is converted into prompt token IDs; the remaining codebooks enter the model as embeddings.

This creates a subtle failure mode. Two requests may contain different acoustic information while sharing both the same text prompt and the same first-layer codebook values. A standard Radix Cache would treat their prefixes as identical. The later request could then reuse the KV cache from the earlier one, causing the generated speech to inherit the wrong reference voice.

**Solution:** As part of the migration to [`OmniScheduler`](https://github.com/sgl-project/sglang-omni/pull/937), we established an explicit rule: every input or piece of state that can affect the KV state—such as embeddings or adapters—must contribute a fingerprint to `Req.extra_key`.

### Late Chunks After Request Cleanup

A streaming vocoder keeps decoding state for each request ID and clears it when the request completes or is aborted. A normally completed request processes all of its audio chunks before terminating. When a request is aborted, however, an audio chunk produced earlier by an upstream stage may still be in flight and reach the vocoder only after that state has been cleared.

The old logic treated a request ID missing from the state table as the first chunk of a new request and created fresh decoding state. By then, the original abort signal had already propagated through the pipeline. The “resurrected” request would never receive another cleanup signal, so its new state could never be released. For MOSS-TTS-Local, this invalid request could permanently occupy both a codec session and a CUDA Graph slot.

**Solution:** [`StreamingVocoderBase`](https://github.com/sgl-project/sglang-omni/pull/936) introduces a tombstone mechanism. When a request completes or is aborted, the system temporarily retains a tombstone for its request ID. Any late-arriving chunk that matches the tombstone is discarded immediately. Tombstones are then evicted after a configured retention period.

### A “Deadlock” in the Single-Flight Error Path

Reference-audio encoding is relatively expensive, while the same speaker reference is often reused across many requests. To avoid duplicate work, the framework applies a single-flight mechanism: concurrent requests for the same reference audio share one encoding operation, and the remaining requests wait for its result.

We found a failure mode in the error path. If audio loading or cache insertion failed without clearing the coordination state, subsequent requests using the same reference audio could wait indefinitely for a result that would never arrive.

**Solution:** Encoding, cache insertion, and error propagation now belong to the same lifecycle. If any step fails, the framework immediately clears the in-flight state and forwards the error to every waiting request.

## Validating the Abstractions with New Models

Successfully migrating existing models only proves that the new interfaces are compatible with the old implementations. It does not prove that the abstractions are general enough for new architectures.

To test that, we integrated three additional TTS models: Ming-Omni-TTS, ZONOS2, and Audar-TTS. The goal was twofold:

- Determine whether a new model could be integrated with substantially less serving code.
- Verify that the shared abstractions could be reused without sacrificing correctness or performance.

Ming-Omni-TTS provides one example of a substantially different generation path. Its autoregressive backbone produces hidden states, which are passed to a FlowLoss/CFM tail to sample continuous acoustic latents. An AudioVAE then decodes those latents into waveforms. Despite these architectural differences, the model was able to reuse the same engine, reference-encoding, and state-transport interfaces.

Audar-TTS provides another useful case study. It is an Arabic text-to-speech model. To measure directly whether the new framework reduced integration costs, we implemented Audar-TTS twice, once against each framework. Before the refactor, we used the old framework as the baseline and added production-ready model support. After the refactor, we used the shared framework as the baseline and implemented exactly the same capabilities.

On the old framework, integrating Audar-TTS required 797 lines of code, excluding tests and documentation. With the new framework, that number fell to 619 lines, a reduction of 22.3%. Within those totals, the additional code required to take the model from “it runs” to production-ready fell from 222 lines to 77 lines, a reduction of 65.3%. The difference came primarily from repeated serving mechanisms such as reference-audio caching, error handling, and request cleanup, which the framework now provides. The resulting model-side code is also clearer and easier to maintain.

This reduction in integration code did not change the model’s computation, and performance remained effectively unchanged. We compared the two implementations on the same inputs: across 28 paired requests, they produced exactly the same 285-code sequences and identical 24 kHz waveforms; on a separate set of 50 Arabic sentences, their acoustic codes, floating-point waveforms, and PCM-WAV hashes all matched. On an H100, stage-sum latency, real-time factor, and engine throughput changed by −0.13%, −0.13%, and +0.16%, respectively, all within normal measurement variance.

<div align="center">
  <img src="images/tts-refactor-audar-validation.svg" alt="Comparison of the original and refactored Audar-TTS implementations across code size, output consistency, and performance" width="88%">
  <p><em>Figure 4: The refactor reduces the amount of production-serving code required on the model side while preserving outputs and performance.</em></p>
</div>

The refactor did not change the model’s numerical behavior or generation algorithm. What it removed was the repeated caching, scheduling, and lifecycle code required to move a model from “it runs” to “it serves reliably at high performance.”

## Where We Drew the Boundary

Sampling, codec sessions, MoE execution, codebook layouts, and waveform post-processing remain inside model-specific directories. These components directly determine how a model generates and decodes audio, and their constraints vary significantly across architectures. Trying to force them behind a single abstraction would produce a large and brittle interface rather than meaningful reuse.

During the refactor, we also explored several broader abstractions based on profiling results, including batched reference encoding and a shared decode-state pool. Load testing showed, however, that these optimizations were highly workload-dependent. They produced meaningful gains only for particular combinations of models and traffic patterns, and the improvements did not consistently transfer to other backends. Because their semantics and benefits had not yet converged, we chose not to promote them into the framework prematurely. Individual models can continue to explore and adopt these optimizations where they are useful.

The abstractions that did move into the shared layer are the serving lifecycles that repeatedly appeared across models and had developed stable semantics:

- Engine construction and startup.
- State transport across pipeline stages.
- Reference-audio caching.
- Batch and streaming vocoder scheduling.
- Request-state cleanup.
- Shared scheduler validation.

After six existing backends were migrated successfully, the integration of Ming-Omni-TTS, ZONOS2, and Audar-TTS provided further evidence that these interfaces can support new model architectures.

The goal of a refactor is not to move every possible line of code into the framework. The better contract is:

> **The framework owns the stable mechanisms required for reliable, high-performance serving. Models remain lightweight and retain the generation logic and optimization strategies that best fit their architectures.**

Future progress, updated statistics, and a commit-by-commit breakdown will continue to be published on the [TTS Refactor Progress](https://luojiaxuan.github.io/sglang-omni/tts-refactor/) page.

---

## Acknowledgments

We would like to thank everyone who contributed to the core roadmap and its related or exploratory pull requests:

[Yuhao Chen](https://github.com/AkazaAkane), [Jiaxin Deng](https://github.com/JiaxinD), [Jingwen Gu](https://github.com/JingwenGu0829), [Chenchen Hong](https://github.com/Hayden727), [Yizhuo Huang](https://github.com/YzXiao101), [Xiangrui Ke](https://github.com/keke0315), [Xinyu Lu](https://github.com/SandyLuXY), [Jiaxuan Luo](https://github.com/luojiaxuan), [Ratish P](https://github.com/Ratish1), [Xinhao Tan](https://github.com/XinhaoTheo), [Xuesong Ye](https://github.com/yxs), [Yue Yin](https://github.com/MelodyyyYin), [Gaokai Zhang](https://github.com/GaokaiZhang), [Yichi Zhang](https://github.com/Ccyest), and [Chenyang Zhao](https://github.com/zhaochenyang20).

The complete pull-request history and review discussions are recorded in [issue #985](https://github.com/sgl-project/sglang-omni/issues/985).
