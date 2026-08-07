# Inside the Runtime Upgrade to SGLang v0.5.16 in SGLang Omni

When we started upgrading SGLang Omni from SGLang `0.5.12.post1` to `0.5.16`, I expected a difficult dependency bump: fix renamed APIs, update the integrations, and move on. That expectation did not survive the first serious read of the upstream diff. The upgrade crossed six SGLang releases, moved Transformers from `5.6.0` to `5.12.1`, and eventually changed 162 files in [PR #1183](https://github.com/sgl-project/sglang-omni/pull/1183).

The large diff was only a symptom. SGLang Omni is not a thin wrapper around SGLang: it owns a multi-stage pipeline, parts of the scheduler loop, model-runner integration, streaming state, and process placement. Once a downstream framework owns those boundaries, an upstream pin is no longer just a package version. It describes a runtime contract between two systems.

This became the driving question for the upgrade: **how do we prove that SGLang Omni still means the same thing after the runtime beneath it changes?** The answer turned out to be different at each boundary. Scheduler compatibility required preserving an execution protocol; Qwen3-Omni required preserving floating-point arithmetic; terminal cleanup required defining the last owner of a request; and colocated MOSS stages required accounting for GPU memory in construction order rather than treating a process budget as one timeless number.

## The Scheduler API Was Only the Visible Change

SGLang `0.5.16` changed the boundary between scheduling and model execution. Batch selection now returns a `NextBatchPlan`, `ForwardBatch` is initialized from the live `ScheduleBatch`, and output processing consumes a `GenerationBatchResult`. Omni could not adopt those changes by calling SGLang's scheduler loop directly because it owns a different multi-stage event loop, model-specific runners, streaming behavior, and result plumbing. The upgrade therefore had to reproduce the part of the execution contract that those custom paths bypass.

The difficult part was not converting one result type into another. The scheduler can filter, merge, retract, or reuse requests between iterations, so state attached to an earlier view of a batch may no longer describe the batch that executes next. At the same time, the device-side result needed by the following model step and the host-side result used for finish detection, streaming, and response construction have different lifetimes. Treating them as one interchangeable object preserves the old integration's assumptions under new method names.

The final implementation keeps that adaptation in one narrow execution boundary, removes the obsolete `0.5.12` `output_ids` side channel, and removes an unreachable fallback through upstream `run_batch`. Model-specific runners now enter through the same batch-construction and result-publication path instead of carrying parallel compatibility behavior. The broader lesson is more precise than “upstream APIs can change”: when a framework changes who owns mutable execution state and when that state may be consumed, matching the new signatures while preserving the old dataflow is still incorrect.

## A Few Floating-Point Operations Changed Qwen3-Omni

Once the scheduler path was working, Qwen3-Omni presented a more deceptive failure. The model started successfully and accepted the same requests, but its MMMU result had regressed. That kind of failure is easy to misattribute to preprocessing, image resizing, tokenizer changes, model weights, rotary positions, or general GPU nondeterminism, so we compared the two stacks layer by layer.

The inputs were identical. `input_ids`, attention masks, pixel values, image-grid metadata, patch-embedding output, and rotary position IDs all matched. The first difference appeared after positional embeddings entered the vision encoder, and it propagated through the first vision block into the final and deepstack image embeddings. Across seven real samples, the maximum absolute difference in the final image embeddings ranged from roughly `0.156` to `0.359`.

The underlying change was only a few floating-point operations. Transformers `5.6` constructed bilinear-interpolation coordinates with CPU FP32 behavior, converted the interpolation weights to the positional-embedding table's dtype—normally BF16—and combined the four corner embeddings in an explicit order:

```python
corners = pos_embed(indices) * weights[:, :, None]
result = corners[0] + corners[1] + corners[2] + corners[3]
```

Transformers `5.12` moved the calculation into a shared path. It generated the interpolation state differently, retained FP32 weights during multiplication, and reduced the four corners with a sum. Both implementations describe the same bilinear interpolation mathematically, but BF16 multiplication and addition are not associative. Changing the intermediate dtype and accumulation order changed the positional embeddings seen by the pretrained vision tower.

The fix was deliberately local. [`Qwen3OmniMoeVisionEncoderCompat`](https://github.com/sgl-project/sglang-omni/blob/a8d3dd14a2784cea51937936301043f1735bfda7/sglang_omni/models/qwen3_omni/components/vision_compat.py#L13-L146) retains the Transformers `5.12.1` encoder structure, decorators, output type, vision blocks, and deepstack behavior. It replaces only the interpolation arithmetic with the `5.6` sequence used by the checkpoint's original stack. After that change, preprocessing tensors, captured intermediate vision tensors, final embeddings, and deepstack embeddings were bit-identical to the reference; the 50-sample MMMU gate recovered to 31/50, or 62%.

This was the clearest result of the entire upgrade. **For a pretrained model, compatibility includes the floating-point program that interprets its weights.** A dependency can preserve every public API and tensor shape while changing the model numerically through device placement, fusion, reduction order, or intermediate precision.

## EOS Did Not Mean the Request Was Finished

The scheduler review also surfaced a pre-existing lifecycle problem. It was not caused by SGLang `0.5.16`, but the scheduler rewrite made it impossible to ignore safely. Omni attached model-specific request data to the SGLang request, while that request data held a reference back to the request:

```text
Req → Omni request data → Req
```

For TTS and omni models, the request data can retain reference media, input embeddings, hidden states, streaming buffers, and model-specific decode state. Leaving the cycle intact means ordinary reference counting cannot release it at terminal time; Python's cyclic collector must discover it later. Clearing the link sounds trivial until we consider what “terminal” means in a multi-stage pipeline.

An autoregressive request can finish while an upstream stage still has a stream chunk in flight. The same request may remain visible through the running batch, the just-completed batch, an asynchronous pending step, and a stream-ingress buffer. If request data is detached before the model runner flushes its final buffered audio, the final chunk can be lost. If it is detached before in-flight stream ingress settles, a late chunk can be mistaken for pre-admission data and retained in a pending structure. If an abort arrives while normal terminalization is already cleaning the request, both sides can either run the model cleanup or assume the other side will do it.

The final scheduler code treats terminalization as an ownership handoff rather than a pointer clear. Under the request-admission lock, normal completion claims the request so that only one path performs terminal output. It then asks the model runner to flush remaining stream state, constructs the terminal result, runs the model-specific finish callback, and only then detaches the Omni request data and records the request as completed. Stream chunks arriving after that boundary are dropped instead of recreating pending state. Abort uses the same lock: if it arrives before detach, terminalization observes it and completes abort cleanup; if it arrives after detach, the abort path knows there is no terminal owner left and performs the cleanup itself.

The important property is not that every terminal path calls the same helper. It is that the lock and the data attachment together identify exactly one cleanup owner for both possible interleavings. This is visible in the final [`stream_output`](https://github.com/sgl-project/sglang-omni/blob/a8d3dd14a2784cea51937936301043f1735bfda7/sglang_omni/scheduling/omni_scheduler.py#L1326-L1424) path: final stream data is drained before detachment, while completion recording closes the request against later ingress.

This is a recurring problem in multi-stage serving. “The model generated EOS” and “the pipeline has finished with this request” are different events. Cleanup belongs at the second boundary, and that boundary has to be derived from all concurrent owners rather than guessed from the model's finish flag.

## Why the Merged MOSS Pipeline OOMed

MOSS-TTS Local exposed a different ownership mistake. Its default topology assigns GPU fractions to preprocessing, the autoregressive engine, and the vocoder:

```text
preprocessing    0.15
AR engine        0.67
vocoder          0.18
```

When those stages run in separate processes, a stage fraction and its process fraction are the same. When an operator merges the vocoder into the pipeline process, the declared cumulative budget for the complete process becomes `1.0`. This is not SGLang's `mem_fraction_static`; it represents the sum of the stage budgets after every model in that process has been loaded. Our first implementation passed that final total to the autoregressive engine while it was profiling memory for the KV cache.

The arithmetic looked reasonable and was wrong. Stages are constructed sequentially: preprocessing is loaded first, then the AR engine allocates its weights and KV cache, and only afterward is the vocoder constructed. Giving the AR engine the final `1.0` budget allowed it to allocate memory belonging to a stage that did not exist yet. On the H100 run, it calculated about `66.22 GiB` as available for KV; when the vocoder later requested another `942 MiB`, only around `442 MiB` remained and startup failed.

The correct value at each construction point is the prefix of the final process budget. Preprocessing sees `0.15`; after adding the AR engine, the declared cumulative budget is `0.15 + 0.67 = 0.82`; only after the vocoder is loaded does that cumulative budget reach `1.0`. The final implementation computes those prefix totals in stage-construction order and injects the appropriate value into each factory that performs process-scoped memory profiling. The implementation is small enough to read directly in [`_attach_process_memory_fraction_defaults`](https://github.com/sgl-project/sglang-omni/blob/a8d3dd14a2784cea51937936301043f1735bfda7/sglang_omni/pipeline/mp_runner.py#L186-L214).

With `0.82`, the AR engine sized the KV allocation at approximately `51.89 GiB`. Once the vocoder finished loading, the complete process used `73.44 GiB` against its final `79.65 GiB` budget, and the previously failing merged topology served a valid request.

This issue is easy to describe as “reserve memory for the vocoder,” but that framing misses the useful design principle. **A process budget is phase-dependent when the process is assembled incrementally.** Placement fractions describe the final topology; memory profiling needs the fraction represented by objects that have been loaded up to the current construction point. The scope of the measurement and the scope of the budget must match.

## What the Upgrade Required Us to Verify

These problems needed different forms of proof because they failed in different ways. The scheduler bridge was checked against the actual `0.5.16` execution path rather than inferred from renamed methods. The Qwen fix required bit-identical intermediate and final tensors, because an end-to-end accuracy score could show the regression but could not localize it. The lifecycle change required exercising abort and stream interleavings while watching memory settle after requests. The MOSS fix required reading GPU memory at the exact startup phase where the KV decision was made.

That process also helped separate work that truly belonged in the pin bump from changes that merely happened to be nearby. The execution bridge and Qwen compatibility path were required by the new dependency stack. The request cycle was older, but touching terminal ownership without fixing it would have made the upgraded scheduler unsafe. The MOSS failure came from a topology and accounting interaction carried by the branch. Other performance ideas that did not have an equally clear compatibility argument were kept out of the final upgrade.

In retrospect, the upgrade was difficult for a simple reason: SGLang Omni depends on more than SGLang's exported Python surface. It depends on when a batch may be mutated, how the next token is owned between iterations, which floating-point operations define a model, when a multi-stage request truly becomes unreachable, and which objects are resident when memory is profiled.

A pinned runtime version is therefore not just a reproducible installation choice. It is a statement that these assumptions have been verified against one concrete runtime. Moving the pin means establishing them again.
