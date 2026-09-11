# A Practical Guide to Upgrading SGLang Omni's Backbone

For the Chinese version, see [SGLang Omni Backbone Version Upgrade Guide](./version_bump_guide_zh.md).

SGLang Omni depends heavily on SGLang's internal APIs. Of its roughly 512 Python files, 68 directly import internal modules under `sglang.srt.*`. These modules are implementation details of SGLang, and their private interfaces come with no compatibility guarantees. A backbone version bump may look like a dependency update, but it also requires adapting the existing code to changes in those interfaces.

> All examples in this guide come from the `0.5.16 → 0.5.17` upgrade ([PR #1477](https://github.com/sgl-project/sglang-omni/pull/1477)). It touched 10 files, adding 80 lines and removing 43. That is far smaller than the previous upgrade, which touched 162 files, but a type checker still could not catch either of the two issues that took the most time. For a full retrospective on the earlier `0.5.12.post1 → 0.5.16` upgrade, see [From API Alignment to Floating-Point Associativity: Upgrading SGLang Omni's Backbone](./version_bump.md). This guide does not repeat those cases.

## 1. The actual scope

The three subsystems with the most import sites are `layers`, `managers` (where the scheduler lives), and `model_executor`:

| Subsystem | Imports |
|---|---:|
| `sglang.srt.layers` | 69 |
| `sglang.srt.managers` | 68 |
| `sglang.srt.utils` | 27 |
| `sglang.srt.model_executor` | 26 |
| `sglang.srt.models` | 22 |
| `sglang.srt.sampling` | 20 |
| `sglang.srt.server_args` | 15 |
| `sglang.srt.model_loader` | 12 |
| `sglang.srt.distributed` | 11 |
| everything else (`platforms`, `configs`, `mem_cache`, `compilation`, `runtime_context`, `speculative`, `multimodal`, `environ`, `dllm`, `disaggregation`, `arg_groups`, `kernels`) | ~50 |

These counts cover the `sglang_omni/` and `sglang_omni_router/` packages. Apart from `sglang.kernels`, every import path is under `sglang.srt.*`, so none of these APIs should be expected to remain stable.

Recount them at the start of every upgrade. As features are added, Omni's dependency on SGLang internals will keep growing, so the result from the previous upgrade cannot simply be reused. The distribution also helps identify which upstream changes may affect Omni and, in turn, which entries need attention in Step 1.

`sglang_omni/vendor/sglang/` is where version compatibility code belongs. Any logic that selects a different implementation based on the SGLang version should be kept there rather than scattered elsewhere. The compatibility layer can itself break; see [Section 5](#5-do-not-infer-the-version-from-interface-existence-alone).

## 2. Two classes of change and what they mean for the process

Class A: a name is removed or renamed, or a function signature changes. These issues always raise an exception: `ImportError`, `TypeError: unexpected keyword argument`, or `TypeError: missing required argument`. A type checker can catch them before the code runs.

Class B: names and function signatures stay the same, but their semantics change. The call still resolves, accepts the same arguments, and returns normally, but no longer has its original effect. These issues raise no exception and static tools cannot find them. They surface only when real requests are run through a real model.

The two classes differ by an order of magnitude in cost, and the work should be ordered accordingly. Static scans are cheap, so run them early. When estimating the schedule, plan around Class B, because that is where most of the time goes.

> In `0.5.17`, there were 6 Class A changes and 2 Class B changes. The 6 Class A changes were fixed within half a day; the 2 Class B changes took the rest of the time.
>
> One example in detail: `0.5.17` turned `ServerArgs` into a read-only startup record and moved the resolved configuration into a set of objects. The SGLang source calls these objects config bags (class `_ConfigBag`), and callers use functions such as `get_exec()` and `get_parallel()` to read the corresponding settings. The signature of `ServerArgs.override()` did not change, and the call still returned normally, but its changes were no longer propagated to the relevant config bag. At the same time, `get_num_allocatable_reqs` now had to read `get_parallel().pp_max_micro_batch_size` instead of `get_server_args().pp_max_micro_batch_size`. The write therefore returned normally but had no effect. The scheduler read `None` and stopped on the first batch when it tried to evaluate `None - int`.
>
> The conclusion from the previous retrospective was that "Omni depends on SGLang's concrete behavior, and the interface never pinned that behavior down." This upgrade presented an even subtler problem: the interface did not change at all, but its behavior did.

## 3. Order of work

### Step 0: compare dependency metadata and decide whether the image needs to be rebuilt

This takes about ten minutes and does not depend on any later adaptation work, but it affects the schedule for the entire upgrade.

```bash
pip download --no-deps sglang==<old-version> -d /tmp/sgl-old
pip download --no-deps sglang==<new-version> -d /tmp/sgl-new
# unpack both, then compare METADATA / requires_dist
```

Pay particular attention to flashinfer and torch. `.github/scripts/validate_omni_env_reusable.sh` requires them to be loaded from the image's site-packages, not from the venv, so that the prebuilt JIT cache in the image can be reused. If the new `sglang` requires a newer flashinfer version—possibly just as a result of dependency resolution—installing the project puts that newer version in the venv, and the check fails. The version requirement comes from the `sglang` package itself, not from `pyproject.toml`, and cannot be fixed in the PR. This requires both of the following:

1. Rebuild the CI image. `docker/Dockerfile` pins the flashinfer version and `COPY`s `/root/.cache/flashinfer/<version>`.
2. Update the digest in six workflow files and the Dockerfile.

Both `/docker` and `.github` have CODEOWNERS, so an image rebuild has to wait for the relevant owners to arrange it.

- The target SGLang version directly determines the base image, the flashinfer version, and the JIT cache version. All three are known before adaptation work starts, and nothing found during adaptation will change the image contents.
- Local adaptation and validation do not depend on the image. The rebuild is needed for CI's environment-reuse check and for JIT cache reuse; it does not affect whether the code can run locally. Install the new `sglang` and its dependencies for local work.
- Local adaptation and the image rebuild are on separate critical paths and should proceed in parallel. Waiting until adaptation is complete adds an unpredictable delay to the overall schedule.

> For `0.5.17`, the metadata diff was:
>
> | Requirement | 0.5.16 | 0.5.17 |
> |---|---|---|
> | `helion` | `==0.2.6` | `==1.4` |
> | `flashinfer_python[cu13]` | `==0.6.14` | `==0.6.15.post1` |
> | `sgl-deep-gemm` | `==0.1.4.post1` | `==0.1.5.post1` |
> | `av` on Linux ARM | unpinned | `==16.1.0` |
> | `xxhash` | absent | added |
>
> The flashinfer version change made CI fail outright with `flashinfer must come from the image, not /data/omni-ci/pr-1477/omni` and `Torch and FlashInfer must use the image installation for JIT cache reuse`. Updating the digest required 22 changes. The image rebuild was not requested until the fourth day of the upgrade, and CI had been failing since its first run.

### Step 1: read the changelog and identify the scope of the changes

Start with the Breaking Changes and Dependencies sections to estimate the amount of architectural work involved.

Be clear about the changelog's limits. It usually does not record every change to private interface signatures, and most of what Omni uses is private. Even when a change is listed, the changelog often says what changed without explaining exactly how the old code will fail. It can point the investigation in the right direction, but it cannot replace checking each item.

It is still best to compare the code directly: unpack the old and new wheels, then diff only the modules the project actually imports.

> The `0.5.17` changelog contained more than 600 entries in over 20 categories, but covered only 2 of the 8 changes required for this upgrade.
>
> | Change | Covered by the changelog |
> |---|---|
> | `ServerArgs` made read-only, config moved to config bags | yes, under Breaking Changes |
> | `sglang.jit_kernel` folded into `sglang.kernels` | yes, under Kernel Library |
> | `SamplingBatchInfo`'s `vocab_mask` / `apply_mask_func` merged into `grammar_mask` | no |
> | `SchedulerLogprobResultProcessor` dropped `server_args` | no |
> | `SchedulerDPAttnAdapter` gained a required `model_runner` | no |
> | `SchedulerLoadInquirer` gained three required telemetry accessors | no |
> | `pp_max_micro_batch_size` read moved to `get_parallel()` | no |
> | token clamp reads `dcp_size` | no |
>
> On the config split, the changelog said, "Code that mutated ServerArgs at runtime must route through the new accessors." That statement is accurate, but it does not say that the old call still returns normally. That omission is what led to the problem.

### Step 2: establish a performance baseline

Collect the baseline for both versions on the same machine and with the same method. Warm up before every measurement and repeat each run. The main rule is that the first execution of any code path is not a valid measurement, because there are many reasons a code path may not be fully warmed up.

Run `.github/scripts/delete_gpu_process.sh` between runs, just as the CI workflows do. GPU memory left over from the previous run is the second biggest source of false regressions, after an unwarmed cache.

Both versions must also use the dependencies pinned for that version. A performance test compares two complete dependency stacks, not just two versions of one package.

The evaluation tools live in `benchmarks/eval/`. CI checks the same metrics that need to appear in the PR description, so collect them in the same form from the start.

> For `0.5.17`, four apparent performance regressions were investigated. All four were false alarms.
>
> | Observation | Actual cause |
> |---|---|
> | Qwen3-TTS throughput dropped | it took four runs to reach steady state |
> | TTS stage-2 TTFC p95 was 0.5838 | repeat runs returned to the 0.506–0.529 baseline range |
> | `ws_stream` latency p95 was 13.82s, above the threshold | both repeat runs were within the threshold |
> | MMMU at 0.959 qps and 16.05s latency | the inductor cache was not warm; three consecutive runs passed after cleanup |
>
> The dependency sets were flashinfer 0.6.14, helion 0.2.6, and sgl-deep-gemm 0.1.4.post1 for `0.5.16`; and 0.6.15.post1, 1.4, and 0.1.5.post1 for `0.5.17`.

### Step 3: update the pin and run static checks

After installing the new version, run a type checker against the existing code and the new internals. Class A changes can be fixed together at this stage, at low cost.

The project does not currently configure a type checker, but running mypy or using an LSP with type checking can still find problems quickly, even though each tool produces a different amount of noise. In my experience, `pyrefly` has a much better signal-to-noise ratio than `ty`, and both are considerably faster than the more widely used `pyright`. `ty`, however, produces enough noise to bury useful diagnostics.

**`try/except ImportError` hides import failures.** When an import fails, these branches fall back to another implementation, and the program silently switches to a slower path. There is no exception or log, only a performance drop with no diagnostic clue. Retired import paths should therefore be found with a static search, including references inside `try/except ImportError`, rather than left for runtime testing to uncover.

> **Tip:** grep can find matching text, but it also matches comments and strings, and it cannot tell that an import is inside a `try` block that catches `ImportError`. ast-grep matches the syntax tree and avoids both problems. To find imports from a known retired module:
>
> ```bash
> ast-grep --lang python --pattern 'from sglang.jit_kernel.$$$A import $$$B'
> ```
>
> To list all imports inside `try/except ImportError` blocks for review before the upgrade (there were 17 under `sglang_omni/` at the time of writing):
>
> ```bash
> ast-grep --lang python --pattern 'try:
>     $$$BODY
> except ImportError:
>     $$$H' sglang_omni/
> ```
>
> This covers both `from X import Y` and `import X`, while excluding `try` blocks that catch other exceptions. Check two things for every match: whether the import still succeeds with the new version, and, if it does not, whether the fallback still behaves as intended.

It is also a good idea to commit each fix separately and explain the reason in the commit subject. This prepares material for the PR description and keeps the Class A fixes separate from the Class B fixes that follow.

> For `0.5.17`, this step produced three commits:
>
> ```
> fix(qwen3-omni): drop the SamplingBatchInfo grammar-mask kwargs
> fix(moss-tts-local): import flash_attn_varlen_func from kernels.ops
> fix(scheduler): adapt to the 0.5.17 scheduler-component contracts
> ```
>
> The second commit is the `try/except ImportError` case described above. The MOSS-TTS-Local vocoder falls back to SDPA in that exception handler. Since `sglang.jit_kernel` was retired in `0.5.17`, the import always fails.

### Step 4: start every model covered by CI

The CI model matrix is the list of required tests and defines the scope the project actually commits to supporting.

| Workflow | Models | Checks |
|---|---|---|
| `test-asr-ci.yaml` | MOSS-Transcribe-Diarize; Fun-ASR or Qwen3-ASR (selectable) | WER, RTF, throughput |
| `test-tts-ci.yaml` | Higgs or MOSS-TTS-Local (selectable), 5 stages | WER, SIM, TTFC, latency, streaming consistency, router DP2 stress test |
| `test-qwen3-omni-ci.yaml` | Qwen3-Omni, 11 stages | thinker length, TTS WER and SIM, MMMU and MMSU accuracy and speed, talker, video |
| `omni-ci.yaml` → PR Test | — | full unit test suite |

Passing the static checks does not mean the code will run. Class B changes usually surface only at this stage.

When switching repeatedly between two versions, a common approach is to unpack the other `sglang` version into a separate directory, then use `PYTHONPATH` to put that directory first on `sys.path`. This makes `import sglang` load that copy instead of the installed one. This is usually called a shadow. Switching then requires changing one environment variable rather than reinstalling the full dependency stack.

A shadow has two limitations. First, it replaces only `sglang`; all other dependencies still come from the current environment, so the resulting stack matches neither the old version nor the new one exactly. It is useful for a quick check that the code runs, but not for collecting the comparison data from Step 2. That data must be collected with each version's complete dependency stack.

Second, a shadow works only if the worker inherits the parent process's environment. CI tests start workers through `start_server_from_cmd` (`benchmarks/benchmarker/utils.py`). It first copies `os.environ`, then applies the `env` supplied by the caller. A test that does not pass `PYTHONPATH` keeps the parent's shadow setting. If a test sets `PYTHONPATH` in `process_env`, that value overrides the shadow and the worker runs the installed version. The two test runs then use the same code and incorrectly report no regression.

Before comparing versions, find the `process_env` passed to `launch_managed_router` and check which category each test falls into. Tests in the second category must be run after installing each version in turn. As of the current main branch, `test_tts_serving_ci.py` is in this category: it sets `PYTHONPATH` to the project root in `process_env` (`tests/test_model/test_tts_serving_ci.py:296`), and its benchmark subprocess does the same (`:369` in the same file).

Record exactly which tests were run. A table of passing results should not imply that the full matrix was covered.

> Both Class B changes in `0.5.17` surfaced at this stage. The corresponding fixes were also the two most time-consuming commits:
>
> ```
> fix(scheduler): route the pp_max_micro_batch_size default through the context
> fix(scheduler): mirror the 0.5.17 step counters and batch launch timestamp
> ```
>
> In this upgrade, 12 of the 17 GPU tests were run. TTS stages 3–5 and the four Qwen3-Omni video jobs were not run, and the PR states that explicitly.

### Step 5: reproduce a regression before investigating its cause

The rule from Step 2 applies here as well: a performance drop must not be treated as a regression until it has been reproduced in a clean environment. Reproduction usually takes minutes; finding the root cause may take days. Reproduce first, then investigate.

Once a regression is confirmed, add temporary logging to the compatibility layer on a running server to record the values each call site actually reads and writes. Unlike static inspection, these logs show actual runtime behavior rather than an inference.

These logs cover only the call sites exercised by that run; the rest remain unknown. Base conclusions on measured results, and state clearly which paths were not covered. Overstating the scope will mislead later investigation.

> Across the repository, there were 23 `override_server_args` call sites covering 13 fields. Temporary logging was added to the helper on a running Qwen3-TTS server. The three call sites reached by that run all yielded clear results, and none was affected; only `pp_max_micro_batch_size` was actually broken. The remaining callers were on model paths not reached by the run, and the PR marks them as unaudited.
>
> Based on those measurements, the PR reached a narrow conclusion: writing through `ServerArgs.override` and later reading the value back from `ServerArgs` still works, but writing a value for a config bag consumer to read does not. It would be wrong to summarize this as "all of these overrides are broken."

### Step 6: open the PR after pre-commit and all unit tests pass

`pre-commit` runs autoflake, isort, black, and ruff locally. The `lint` job in CI runs the same checks.

The fake objects used in place of SGLang components in unit tests (`tests/unit_test/fakes.py`) hard-code the structure of upstream interfaces. Whenever upstream code reads another field, that field must be added to the fakes. Otherwise the tests either fail outright or keep passing even though they no longer reflect real behavior; the latter is more dangerous. Updating the fakes is part of the adaptation work, not final cleanup.

Before attributing any failure to the upgrade, check whether it reproduces on the old version. That check usually takes one command, while a wrong attribution can waste hours of investigation.

The PR description must include a version-difference table, a one-line explanation of each adaptation, accuracy and performance comparisons, and a clear list of anything that was not run.

> For `0.5.17`, 3 of the 9 substantive commits changed only tests:
>
> ```
> test(scheduler): give the scheduler doubles a dcp_size-bearing server_args
> test(scheduler): model the 0.5.17 runtime-context contract in the doubles
> test: adapt two merged-in suites to the 0.5.17 contract
> ```
>
> The only failing unit test, `test_mp_runner_startup_failure_includes_child_factory_traceback`, failed in exactly the same way on `0.5.16`: the test allowed 10 seconds for startup, while a cold `import sglang_omni.pipeline.stage_workers` took 18.6 seconds on that machine. The failure was due to that machine's performance, not the version upgrade.

## 4. Dealing with a main branch that keeps moving

The main branch continues to change during an upgrade, and the scope of the adaptation may grow with it. Two principles help. First, keep the upgrade PR small—make minimal changes and do not include unrelated refactoring—to reduce conflicts. Second, merge it as soon as possible, because the longer the branch remains open, the more it costs to maintain. The exact timing still depends on the project's overall roadmap.

After every merge from main, check again for performance regressions and for new instances of the same compatibility problems.

> During the `0.5.17` upgrade, changes from main were merged into the upgrade branch three times in four days. One commit existed only to re-adapt tests brought in by a merge. While CI was blocked, two more relevant changes landed on main: `models/moss_tts/vocoder_decoder.py` was merged into `audio_tokenizer.py`, deleting the original file modified by the PR and moving its `jit_kernel` import with it; and a deterministic-inference feature added two new `override_server_args` call sites.

## 5. Do not infer the version from interface existence alone

`override_server_args` in `sglang_omni/vendor/sglang/server_args.py` is intended to keep the handling for different versions in one place. It selects an implementation as follows:

```python
legacy_override = getattr(server_args, "override", None)
if callable(legacy_override):
    legacy_override(source, **fields)
    return
# get_context().override(...) / declare_late_resolution(...)
```

`ServerArgs.override` still exists in `0.5.17`, so this code always enters the first branch. The two paths below it for the new version never run. The temporary fix was for the scheduler to bypass this code and call `get_context().override(...)` directly.

More generally, when upstream deprecates an API by leaving the interface in place but removing its effect, checking only whether the function exists silently selects the broken legacy branch and prevents the new-version path from running. Use the version number instead, or check an observable result: write a value, then read it back through the accessor that the consumer actually uses. The presence of the method alone is not enough.

As of the current main branch, this small issue is still not fixed. PR #1477 only worked around it by calling `get_context().override(...)` directly in `omni_scheduler.py`; it did not change the shim itself. The fix belongs in the shim.

That would solve only this case. It cannot stop upstream from deprecating another interface in the same way. Such a change raises no exception and cannot be found by static tools. It is a Class B change, and it surfaces only when real models are run in Step 4.

## 6. Checklist

- [ ] Compare dependency metadata; if the image needs to be rebuilt, ask the CODEOWNERS for a rebuild on day 0
- [ ] Read the Breaking Changes and Dependencies sections of the changelog; compare private APIs separately
- [ ] Collect a baseline on the target machine, with warmup and repeated measurements
- [ ] Update the pin and scan statically for type errors
- [ ] Check retired import paths statically, including references inside `try/except ImportError`
- [ ] Verify that every model covered by CI starts and serves requests successfully
- [ ] Reproduce every suspected regression on a clean GPU before investigating its cause
- [ ] Before attributing a failure to the upgrade, check whether it also reproduces on the old version
- [ ] Update unit-test fakes to match the new interfaces
- [ ] Pass `pre-commit` and the full unit test suite
- [ ] State in the PR which tests were not run
