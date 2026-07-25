# From Six TTS Stacks to One Serving Framework: Refactoring TTS in SGLang Omni

At the end of [Why SGLang-Omni](./why-sglang-omni-en.md), we described an ideal for onboarding a new model: *declare the pipeline topology, implement the model-specific computation, and leave scheduling, communication, and lifecycle management to the framework.*

The TTS subsystem was the hardest place to make that ideal concrete. Higgs, MOSS-TTS, MOSS-TTS-Local, Qwen3-TTS, FishAudio S2-Pro, and Voxtral-TTS do not share one synthesis algorithm. They use different codebook layouts, autoregressive structures, cache requirements, and vocoders. What they did share was everything wrapped around the math—and each model had rebuilt that machinery locally.

The completed TTS refactor turned those six model-local stacks into a set of explicit serving contracts. The clearest results are not just line-count reductions:

| Proof point | Result |
|---|---|
| **Six existing TTS backends** | Migrated onto shared contracts for engine bootstrap, state transport, reference encoding, vocoder lifecycle, capabilities, and scheduling. |
| **FishAudio scheduler migration** | Deleted the 591-line `FishScheduler`; the migration PR landed at net **−816 lines** while preserving accuracy and performance. |
| **Controlled Audar-TTS A/B** | Reduced production-equivalent integration code from **797 to 619 lines** and the production capability premium from **222 to 77 lines (−65.3%)**, with byte-identical outputs and performance parity. |
| **New-model validation** | Ming-Omni-TTS, Audar-TTS, and ZONOS2 exercised the shared surfaces with architectures that were not used to design the original six-model code. |

Our litmus test was simple:

> **A model contributor should implement model semantics and hooks—not copy, fork, or modify a framework scheduling state machine.**

---

## The Pipeline Was Already Shared; Its Lifecycle Was Not

TTS inference in SGLang Omni is a multi-stage pipeline. Preprocessing, reference-audio encoding, autoregressive generation, and waveform decoding have different compute profiles, memory lifetimes, and batching opportunities. For the broader architecture, see [Why SGLang-Omni](./why-sglang-omni-en.md); for performance work inside individual stages, see [Optimizing TTS Inference](./tts-optimization.md).

<div align="center">
  <img src="images/tts-opt-pipeline-overview.svg" alt="TTS inference pipeline from preprocessing through audio encoding and autoregressive generation to the vocoder" width="78%">
  <p><em>Figure 1. A typical multi-stage TTS inference pipeline in SGLang Omni.</em></p>
</div>

The refactor focused on the layer *around* those stages: the mechanics every production implementation needs but no model directory should own by itself.

## The Duplication Was Around the Math, Not in It

Looking only at computation, the six TTS backends appeared unrelated. Looking at lifecycle, the same structure repeated:

- engine bootstrap resolved checkpoints, built server arguments, initialized model-specific state, captured CUDA Graphs, created adapters, and assembled schedulers in nearly the same order;
- every model manually serialized pipeline state across stage boundaries;
- reference-audio paths independently implemented cache keys, LRU eviction, same-key deduplication, and failure handling;
- batch and streaming vocoders independently managed request state, chunk thresholds, flush ordering, aborts, and terminal results;
- capability differences lived in scattered conditionals instead of one declarative surface.

This is **mechanism duplication**: functions compute different values but repeatedly manage the same control flow. It is harder to notice than copy-pasted math because the duplicated code sits around model-specific kernels and therefore looks model-specific too.

<div align="center">
  <img src="images/tts-refactor-before-after.svg" alt="Before and after diagram showing six model-local serving stacks consolidated into shared framework contracts with model-specific hooks" width="96%">
  <p><em>Figure 2. The refactor did not merge model algorithms. It moved repeated lifecycle mechanics below a stable hook boundary.</em></p>
</div>

The boundary principle was:

> **Framework owns reusable mechanics. Model directories own model semantics.**

That principle led to three non-negotiable design rules:

1. **No model-name conditionals in shared code.** Differences are expressed through hooks, capabilities, or declarative fields—not `if model == ...`.
2. **Migrate or delete.** A model does not keep a hidden legacy path after adopting a shared surface.
3. **The hardest consumer validates the API.** An abstraction is not accepted because the simplest model fits it.

## The Contract Stack

The outcome is not one giant `TTSBaseModel`. It is a small stack of contracts, each with a deliberately narrow ownership boundary.

| Shared surface | The framework owns | The model owns |
|---|---|---|
| [`TtsEngineBuilder`](https://github.com/sgl-project/sglang-omni/pull/923) | invariant bootstrap order, server-argument plumbing, deferred CUDA Graph setup, scheduler assembly, and post-wiring | checkpoint quirks, model setup, compile policy, adapters, and model-specific callbacks |
| [`DeclarativeStateBase`](https://github.com/sgl-project/sglang-omni/pull/1050) and typed tensors | field emission rules, codecs, round-trip transport, dtype/shape-preserving tensor payloads | which state exists and how each field should travel |
| [`ReferenceEncodeService`](https://github.com/sgl-project/sglang-omni/pull/926) | byte-bounded LRU caching, same-key single-flight, waiter failure propagation, no-poison retries, and cache statistics | input normalization, identity keys, codec execution, artifact dtype/device policy, and revalidation |
| [`BatchVocoderBase`](https://github.com/sgl-project/sglang-omni/pull/940) | scheduler wiring and `prepare → decode batch → store` orchestration | the three corresponding hooks and waveform semantics |
| [`StreamingVocoderBase`](https://github.com/sgl-project/sglang-omni/pull/936) | request-state registry, contract latching, chunk/flush/terminal ordering, abort behavior, and coalesced-step failure isolation | codec sessions, cursor math, CUDA Graph slots, decode plans, and emitted audio shape |
| [Model capabilities](https://github.com/sgl-project/sglang-omni/pull/957) and [`OmniScheduler`](https://github.com/sgl-project/sglang-omni/pull/937) | declarative feature discovery plus common request, batch, KV-cache, abort, and finish lifecycles | sampling, stop semantics, cache fingerprints, row layouts, and model-side buffers |

The important property is composability. A backend can adopt the engine builder without using a streaming vocoder, or use typed state transport without sharing its codec implementation. Each contract removes one class of lifecycle duplication without pretending all TTS models are the same.

## Refactoring Without Changing the Model

A serving refactor can be logically clean and still be operationally dangerous. We therefore classified changes by behavioral risk and raised the acceptance gate with the risk.

| Risk | What changes | Required evidence |
|---|---|---|
| **Green** | structural extraction with the same runtime behavior | CPU contract tests and wire/output equivalence |
| **Yellow** | output-equivalent behavior through a new runtime path | accuracy gate plus targeted concurrency/failure tests |
| **Red** | lifecycle semantics, batching, or resource behavior | accuracy plus throughput, latency, and RTF comparison |

The migration order mattered. Low-risk contracts landed first and became test infrastructure for later lifecycle changes. By the time we migrated streaming vocoders and a bespoke scheduler, the shared state, engine, and capability surfaces were already stable.

### Design from the Hardest Consumer

Higgs and MOSS-TTS-Local both stream audio, but their internal requirements are very different. Higgs uses windowed chunking with overlap and crossfade. MOSS-TTS-Local maintains a persistent causal-transformer codec session across chunks, allocates per-request CUDA Graph slots, coalesces requests into one decode step, isolates failures, and latches stream metadata once generation begins.

An API designed around Higgs alone would have needed to be redesigned as soon as MOSS-TTS-Local arrived. Instead, [`StreamingVocoderBase`](https://github.com/sgl-project/sglang-omni/pull/936) was shaped around the harder consumer. The shared layer owns participant selection, plan/execution sequencing, per-request state, completion, and failure isolation; MOSS-TTS-Local keeps its persistent session and graph mechanics behind hooks. Once that worked, the [Higgs migration](https://github.com/sgl-project/sglang-omni/pull/939) needed only a subset of the surface.

This was validated with 18 CPU contract tests for the shared lifecycle and full SeedTTS runs where throughput, RTF, and time-to-first-audio remained flat within run noise.

### Migrate, Then Delete

We avoided `enable_legacy`, per-model escape branches, and “temporary” duplicate implementations. The old path was deleted in the migration PR. This made every shared contract real: production traffic had to exercise it, and reviewers never had to reason about two subtly different lifecycles indefinitely.

## Migration Is an Audit: Fish Meets OmniScheduler

FishAudio S2-Pro originally shipped with a bespoke scheduler because the shared scheduler could not express its requirements at the time. Once `OmniScheduler` had matured, [PR #937](https://github.com/sgl-project/sglang-omni/pull/937) removed the 591-line `FishScheduler`; the full PR landed at net −816 lines.

The deletion was useful, but the stronger result was what the migration exposed.

**Vocabulary bounds became an enforced contract.** Fish semantic tokens live in the tokenizer's added vocabulary. The shared scheduler validates sampled token IDs, which forced Fish to configure `Req.vocab_size` from the complete tokenizer rather than a smaller base vocabulary.

**Radix-cache identity became explicit.** A Fish reference prompt can have the same primary codebook sequence while differing in auxiliary codebooks. Token IDs alone are therefore insufficient cache identity. The migration added a fingerprint of the reference VQ codes so distinct prompts cannot incorrectly share KV cache state.

**Impossible requests failed early and clearly.** The shared scheduler pre-validates request capacity. Fish therefore had to clamp the generation budget to the remaining context length instead of admitting a request that could never fit and relying on a later stop condition.

Independent implementations can look correct because they never exercise the same invariants. Consolidation makes every model traverse the framework's safety checks. In that sense, migration is not only deduplication—it is an audit.

## Declarative State: Move Correctness Into the Structure

State transport is especially dangerous in a multi-process pipeline. Before the refactor, each model maintained hand-written `to_dict` and `from_dict` methods. Adding a state field without updating both methods could silently drop data between stages.

[PR #1050](https://github.com/sgl-project/sglang-omni/pull/1050) replaced six hand-written serializer pairs—313 lines in total—with `DeclarativeStateBase` and `wire(...)` field metadata. Emission policy and codec choice now live beside the field definition, while the framework derives both directions of transport.

The validation was stronger than “the tests passed”:

- rich and default-only states for all six models produced byte-identical normalized wire dumps before and after the change—12 out of 12 comparisons;
- a field-complete round-trip contract test pins every transported attribute;
- the change removed a net 116 non-test lines;
- typed tensor transport records bytes, wire dtype, and shape under one codec rather than forcing every model to invent an encoding.

Later, [Ming-Omni-TTS adopted the same surface](https://github.com/sgl-project/sglang-omni/pull/1103). Its continuous acoustic latents required floating-point tensor transport, so the shared typed-tensor primitive gained an explicit float32 wire policy and the model-local encode/decode helper pairs were deleted. New requirements improved the common primitive instead of spawning another private protocol.

## New Models Became the Acceptance Test

A framework refactor is not proven by how elegantly it rewrites old code. It is proven when unfamiliar models arrive and do not need to fork the framework.

### Ming-Omni-TTS: A Continuous-Latent Outlier

Ming-Omni-TTS is not a conventional logits-to-codebook pipeline. A BailingMoE autoregressive backbone emits hidden states; a FlowLoss/CFM tail samples continuous acoustic latents; those latents feed the next autoregressive step; an AudioVAE later decodes the latent sequence.

That architecture kept its feedback loop, tensor-parallel behavior, tail graphs, and model math local. It still reused the shared engine builder, reference-encode service, capability metadata, checkpoint resolution, declarative state, and typed-tensor transport. The framework handled lifecycle; the model kept the algorithm.

### ZONOS2: Shared Surfaces Without Hiding Complexity

ZONOS2 combines a MoE autoregressive backbone, reference-speaker encoding, delayed DAC codebooks, and streaming waveform decode. Its integration adopted `TtsEngineBuilder`, `ReferenceEncodeService`, `StreamingVocoderBase`, model capabilities, declarative state, shared checkpoint resolution, and the keyed tensor reference-encode hook.

The code that remained model-local is equally important: the MoE backbone, DAC vocoder mechanics, sampler, radix fingerprinting, decode-state pool, and text normalization. A good abstraction reduces duplicated ownership without erasing legitimate differences.

### Audar-TTS: A Controlled A/B

Audar-TTS provided the cleanest experiment because the same model was implemented twice: once against a pre-refactor stack and once against the shared framework.

| Metric | Without shared framework | With shared framework | Change |
|---|---:|---:|---:|
| Minimal integration, non-test/non-doc LOC | 575 | 542 | −5.7% |
| Production-equivalent integration LOC | 797 | 619 | −22.3% |
| **Production capability premium** | **222** | **77** | **−65.3%** |

The *production capability premium* is the code added after a minimal demo to make the backend production-ready: reference caching, lifecycle safety, error handling, and contract coverage. The shared framework reduced that premium by almost two thirds.

<div align="center">
  <img src="images/tts-refactor-audar-validation.svg" alt="Audar TTS controlled comparison showing lower production integration code and production capability premium with identical outputs and performance parity" width="88%">
  <p><em>Figure 3. The strongest refactor result: less model-owned production glue, with unchanged output and performance.</em></p>
</div>

The equivalence checks were unusually strict. All 28 paired requests produced identical 285-code sequences and identical 24 kHz waveforms. A separate 50-sentence Arabic run produced 50 out of 50 identical acoustic-code, float-waveform, and PCM-WAV hashes. On H100, stage-sum latency changed by −0.13%, RTF by −0.13%, and engine throughput by +0.16%—performance parity, not a claimed speedup.

That distinction matters. The framework did not make Audar's model math faster. It made production-grade serving behavior substantially cheaper to integrate without changing the result.

## What We Deliberately Did Not Abstract

Completing the roadmap did not mean turning every recurring noun into a base class.

- **Model math stays local:** sampling, codec sessions, latent feedback, MoE layers, codebook layout, and waveform post-processing.
- **A shared surface needs evidence:** batch reference encoding remained profile-gated rather than becoming a framework feature without a demonstrated bottleneck.
- **Model-specific resource pools stay model-specific until their invariants converge:** decode-state pooling was not forced into a premature universal API.
- **Capabilities describe differences; conditionals do not hide them:** the framework can discover whether a model supports reference audio, batch vocoding, streaming, CUDA Graphs, or compilation without branching on model names.

This restraint is part of the refactor. The goal was not maximal inheritance. It was minimal, enforceable contracts around mechanics that were genuinely shared.

## Takeaways

**Look for duplicated lifecycle, not only duplicated computation.** The six backends had different algorithms but repeated the same bootstrap, transport, cache, scheduler, and vocoder control flows. Changing the question from “what does this function compute?” to “what lifecycle does it manage?” revealed the framework boundary.

**A shared framework earns its existence by enforcing invariants.** Fish gained vocabulary checks, correct cache identity, and early capacity validation. Declarative state gained field-complete round-trip contracts. Streaming gained explicit completion and failure semantics.

**New-model onboarding is the final benchmark.** Ming and ZONOS2 showed that the contracts can support architectures outside the original design center. Audar quantified the result: less production integration code, dramatically less capability glue, identical outputs, and performance parity.

---

The completed roadmap and full PR history are tracked in [SGLang Omni issue #985](https://github.com/sgl-project/sglang-omni/issues/985). The linked PRs preserve authorship, reviews, benchmark artifacts, and the detailed behavior class for each migration.
