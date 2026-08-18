# A Practical Guide to Upgrading the SGLang Backbone

A Chinese version is available at [SGLang backbone 升级实操指南](./version_bump_guide_zh.md).

SGLang Omni does not consume SGLang as a library; it consumes it as a framework. Roughly 68 of its ~512 Python files import internal modules from under `sglang.srt.*`, which upstream treats as implementation detail and changes without a deprecation cycle. Every backbone version bump is therefore a dependency update in form and a port across private interfaces in substance.

> **Where the examples come from:** every example below is drawn from `0.5.16 → 0.5.17` ([PR #1477](https://github.com/sgl-project/sglang-omni/pull/1477)), which touched 10 files, adding 80 lines and removing 43. That is far smaller than the previous upgrade's 162 files, yet the two problems that took longest to resolve again sat in places a type checker cannot see. The full retrospective on the previous `0.5.12.post1 → 0.5.16` upgrade is [From API Alignment to Floating-Point Associativity](./version_bump.md); this guide does not repeat its case analysis.

## 1. How large the upgrade surface actually is

The heaviest dependencies are on `layers`, `managers` (where the scheduler lives), and `model_executor`:

| Subsystem | Import sites |
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

The counts cover the `sglang_omni/` and `sglang_omni_router/` packages. Apart from `sglang.kernels`, all of it sits under `sglang.srt.*`, so none of these APIs should be assumed stable.

Recount this table at the start of every upgrade; the surface grows with feature work and the previous conclusion does not carry over. Its direct use is to bound the review: upstream organizes its release notes around its own module layout, and this table decides which of those entries can reach Omni. Step 1 uses it as the filter.

`sglang_omni/vendor/sglang/` is the designated home for version-conditional code. Anything that has to branch on the SGLang version belongs there rather than scattered across call sites. That layer can fail too — see [section 5](#5-checking-that-an-interface-exists-is-not-checking-that-it-still-works).

## 2. Two classes of failure, and what they imply for the process

**Class A: the symbol changed.** Removed, renamed, or a changed signature. These always raise: `ImportError`, `TypeError: unexpected keyword argument`, `TypeError: missing required argument`. A type checker finds all of them before any code runs.

**Class B: the symbol survived, its meaning changed.** The call still resolves, still accepts the same arguments, still returns normally, and no longer produces the original effect. Nothing raises, static tooling cannot observe it, and only real requests against a real model expose it.

The two classes differ by an order of magnitude in cost, which sets the ordering for everything below: **run static scanning as early as possible because it is cheap, and estimate the schedule from Class B because that is what dominates it.**

> **0.5.17 example:** the split was 6 Class A and 2 Class B. The six Class A fixes were done in half a day; the two Class B ones consumed the rest.
>
> The representative Class B case was the config split. `0.5.17` turned `ServerArgs` into a read-only startup record and moved resolved config onto a set of per-domain namespace objects that the SGLang source calls config bags (class `_ConfigBag`), read through accessors such as `get_exec()` and `get_parallel()`. The signature of `ServerArgs.override()` did not change at all and the call still returned normally, but it no longer wrote through to the config bag. Meanwhile `get_num_allocatable_reqs` moved its read from `get_server_args().pp_max_micro_batch_size` to `get_parallel().pp_max_micro_batch_size`. The result: the write reported success, the reader took its value from somewhere else, the scheduler read `None`, and the process died on `None - int` while handling the first batch.
>
> The previous retrospective concluded that "Omni depends on SGLang's concrete behavior, and the interface never pinned that behavior down." This is the extreme form of that: the interface did not change by a single character, only its behavior did.

## 3. Order of work

### Step 0: diff the dependency metadata and decide whether an image rebuild needs to be queued now

This takes about ten minutes, depends on no adaptation work, and determines the shape of the whole schedule.

```bash
pip download --no-deps sglang==<old> -d /tmp/sgl-old
pip download --no-deps sglang==<new> -d /tmp/sgl-new
# unpack both, then compare METADATA / requires_dist
```

Pay particular attention to flashinfer and torch. `.github/scripts/validate_omni_env_reusable.sh` requires both to resolve to the image's site-packages rather than the venv, so that the JIT cache baked into the image stays valid. If the new sglang's transitive dependencies demand a higher flashinfer, installing the project puts a second copy inside the venv and that check fails immediately. The requirement comes from the `sglang` package itself, is unrelated to what `pyproject.toml` says, and **cannot be fixed inside the PR**. What it needs is:

1. A rebuilt CI image. `docker/Dockerfile` pins the flashinfer version and `COPY`s `/root/.cache/flashinfer/<version>`.
2. A digest update, across six workflow files plus the Dockerfile.

`/docker` and `.github` both have CODEOWNERS, so the rebuild is work that queues behind someone else. **Raise the request on day 0 rather than waiting for the adaptation to finish.** Three reasons:

- What goes into the image is determined entirely by the target SGLang version — the base image, the flashinfer version, and the JIT cache version — all known before the first line of adaptation code is written. Nothing discovered during adaptation changes the image's contents, so deferring the request yields no information.
- Local upgrade and validation do not depend on the image. The image exists to satisfy the CI environment-reuse check and to allow JIT cache reuse; it has nothing to do with whether the code runs. Locally, installing the new sglang and letting it pull its own dependencies is enough.
- Local adaptation and the image rebuild therefore sit on different critical paths and should run in parallel. Scheduling the rebuild after the adaptation simply adds queue time you do not control on top of the timeline.

The only risk of asking early is that the adaptation turns out to be unworkable and the rebuild becomes unnecessary. Handle that by agreeing on a cancellation deadline when you file the request, not by deferring it. If some CI jobs cannot be reproduced locally because of GPU count, the rebuild is directly on the critical path and should be requested even sooner.

> **0.5.17 example:** the diff came out as follows.
>
> | Requirement | 0.5.16 | 0.5.17 |
> |---|---|---|
> | `helion` | `==0.2.6` | `==1.4` |
> | `flashinfer_python[cu13]` | `==0.6.14` | `==0.6.15.post1` |
> | `sgl-deep-gemm` | `==0.1.4.post1` | `==0.1.5.post1` |
> | `av` on Linux ARM | unpinned | `==16.1.0` |
> | `xxhash` | absent | added |
>
> The flashinfer entry blocked CI outright, failing with `flashinfer must come from the image, not /data/omni-ci/pr-1477/omni` and `Torch and FlashInfer must use the image installation for JIT cache reuse`. The digest update covered 22 occurrences. The rebuild request was only raised on day four, and CI had been failing since its very first run.

### Step 1: read the changelog, and establish what it does not cover

Read **Breaking Changes** and **Dependencies** first; together they bound the architectural work.

Then calibrate. A changelog documents the public product, so drift in private interface signatures falls outside its scope by construction — and most of what Omni depends on is private. Depth varies as well: even for entries that are covered, a changelog usually gives the conclusion rather than the failure mode. Its role is to point you at where to look, not to enumerate what breaks.

So add a mechanical comparison: unpack both wheels and diff only the modules the project actually imports.

> **0.5.17 example:** the changelog ran to 600-plus entries across 20-plus categories, and covered 2 of the 8 changes this upgrade required.
>
> | Change | Covered by the changelog |
> |---|---|
> | `ServerArgs` made read-only, config moved to config bags | yes, under Breaking Changes |
> | `sglang.jit_kernel` folded into `sglang.kernels` | yes, under Kernel Library |
> | `SamplingBatchInfo`'s `vocab_mask` / `apply_mask_func` merged into `grammar_mask` | no |
> | `SchedulerLogprobResultProcessor` dropped `server_args` | no |
> | `SchedulerDPAttnAdapter` gained a required `model_runner` | no |
> | `SchedulerLoadInquirer` gained three required telemetry accessors | no |
> | `pp_max_micro_batch_size` read relocated to `get_parallel()` | no |
> | token clamp reads `dcp_size` | no |
>
> The note for the config split read "Code that mutated ServerArgs at runtime must route through the new accessors." That is entirely accurate, and it still does not say that the old call keeps returning successfully — which is the actual failure mode.

### Step 2: establish a performance baseline, with warmup on every measurement

Collect the baseline on the same machine, by the same method, for both versions, with warmup and repeats on every measurement. The core constraint: **the first execution of any code path on either version is not a valid measurement.**

Run `.github/scripts/delete_gpu_process.sh` between runs, as the CI workflows do. GPU memory still held by a previous run is the second-largest source of phantom regressions, after an unwarmed cache.

Each version must also run the dependency stack its own pin implies. What you are comparing is therefore two complete stacks, not two versions of one package, and the write-up has to say so.

The measurement harness lives in `benchmarks/eval/`. The metrics CI asserts on are the same ones the PR description will need, so collect them in that shape from the start.

> **0.5.17 example:** four suspected performance regressions were investigated and none reproduced.
>
> | Observation | Actual cause |
> |---|---|
> | Qwen3-TTS throughput down | needed four runs to reach steady state |
> | TTS stage-2 TTFC p95 at 0.5838 | fell back into the 0.506–0.529 baseline range on repeat |
> | `ws_stream` latency p95 at 13.82s, over threshold | within threshold on both repeats |
> | MMMU 0.959 qps, 16.05s latency | unwarmed inductor cache; passed three consecutive clean runs |
>
> The two stacks were: `0.5.16` with flashinfer 0.6.14, helion 0.2.6, sgl-deep-gemm 0.1.4.post1; `0.5.17` with 0.6.15.post1, 1.4, 0.1.5.post1.

### Step 3: update the pin and run a static scan

Install the new version and let a type checker validate the existing call sites against the new internals. Class A failures can all be fixed here, at very low cost.

The project does not configure a type checker, but running one ad hoc is still worth it. On this codebase `pyrefly` had a markedly better signal-to-noise ratio than `ty`, which produced enough noise to bury the useful diagnostics.

**`try/except ImportError` hides import failures.** Such branches fall back to an alternative implementation when the import fails, so the model silently switches to a slower path — no exception, no log, just degraded performance with no diagnostic trail. Retired import paths must therefore be located statically, including references inside `try/except ImportError`, rather than waiting for runtime to surface them.

> **Tip:** grep can find them, but it also matches comments and strings, and it cannot express the structural relation "the import is nested inside a `try` that catches `ImportError`". ast-grep matches on the syntax tree and handles both. To find real imports of a specific retired module:
>
> ```bash
> ast-grep --lang python --pattern 'from sglang.jit_kernel.$$$A import $$$B'
> ```
>
> To survey every import guarded by `ImportError`, producing a list to confirm one by one before the upgrade (17 matches under `sglang_omni/` at the time of writing):
>
> ```bash
> ast-grep --lang python --pattern 'try:
>     $$$BODY
> except ImportError:
>     $$$H' sglang_omni/
> ```
>
> This form covers both `from X import Y` and `import X`, and excludes `try` blocks that catch other exceptions. For each hit, confirm two things: whether the import still holds on the new version, and if not, whether its fallback is still the intended behavior.

Committing each fix separately, with the reason in the commit subject, is recommended. It gives the PR description ready-made material and keeps the Class A fixes separable from the Class B ones that follow.

> **0.5.17 example:** this step produced three commits.
>
> ```
> fix(qwen3-omni): drop the SamplingBatchInfo grammar-mask kwargs
> fix(moss-tts-local): import flash_attn_varlen_func from kernels.ops
> fix(scheduler): adapt to the 0.5.17 scheduler-component contracts
> ```
>
> The second corresponds to the `try/except ImportError` case above: the MOSS-TTS-Local vocoder falls back to SDPA in that handler, and `sglang.jit_kernel` was retired in `0.5.17`, so the import necessarily fails.

### Step 4: bring up every model CI covers

The CI model matrix is the must-run list; it is the project's enforced definition of what "supported" means.

| Workflow | Models | Checks |
|---|---|---|
| `test-asr-ci.yaml` | MOSS-Transcribe-Diarize; Fun-ASR or Qwen3-ASR (selectable) | WER, RTF, throughput |
| `test-tts-ci.yaml` | Higgs or MOSS-TTS-Local (selectable), 5 stages | WER, SIM, TTFC, latency, streaming consistency, router DP2 stress |
| `test-qwen3-omni-ci.yaml` | Qwen3-Omni, 11 stages | thinker length, TTS WER and SIM, MMMU and MMSU accuracy and speed, talker, video |
| `omni-ci.yaml` → PR Test | — | full unit test suite |

Passing the static scan is not evidence that things run; Class B failures normally surface only here.

When switching back and forth between two versions, a common technique is to unpack the other sglang build into a directory and put that directory ahead of `sys.path` via `PYTHONPATH`, so `import sglang` resolves to it instead of the installed copy. This is usually just called a shadow. Its appeal is that switching costs one environment variable instead of reinstalling the whole stack.

A shadow has two limits. First, it replaces only sglang itself; the rest of the dependencies remain whatever is installed, so what you get is a hybrid stack matching neither side completely. That makes it suitable for quickly checking whether the code runs, but not for producing the comparison numbers Step 2 calls for, which have to be collected under each side's complete stack.

**Second, a shadow only takes effect when the worker inherits the parent process environment.** CI tests start workers through `start_server_from_cmd` (`benchmarks/benchmarker/utils.py`), which builds from `os.environ.copy()` and then applies the caller's `env` over it. Tests that pass no `PYTHONPATH` inherit the shadow; tests that pin `PYTHONPATH` in their `process_env` override it, and the worker runs the installed version instead. In the latter case both sides run the same code, and the run will produce a false "no regression" conclusion.

Before starting a comparison, confirm which category each test falls into by following the fixture down to the `process_env` argument passed to `launch_managed_router`. Tests in the second category must be run against a real installation of each version. As of current main, `test_tts_serving_ci.py` is in that category: it pins `PYTHONPATH` to the project root in `process_env` (`tests/test_model/test_tts_serving_ci.py:296`), and its benchmark subprocess does the same (`:369` in the same file).

Record test coverage honestly, so that a table of passing results does not imply complete coverage.

> **0.5.17 example:** both Class B failures surfaced only at this step, and produced the two most time-consuming commits.
>
> ```
> fix(scheduler): route the pp_max_micro_batch_size default through the context
> fix(scheduler): mirror the 0.5.17 step counters and batch launch timestamp
> ```
>
> On coverage, 12 of the 17 GPU tests were run; TTS stages 3–5 and the four Qwen3-Omni video jobs were not, and the PR says so.

### Step 5: when chasing a regression, reproduce before you localize

The Step 2 constraint applies here as well: **no slowdown counts as a regression until it reproduces on a clean GPU.** Reproduction costs minutes and root-causing costs days, so the order should not be inverted.

Once a regression is confirmed, one useful move is to add temporary logging to the compatibility layer on a running server, recording what each call site actually reads and writes at the moment it is called. Compared with reviewing call sites statically, this gives you real behavior rather than inference.

But it only covers the call sites that run actually reaches; the rest remain unknown. **Keep conclusions inside what was measured**, and state explicitly what was not covered. Overstating the blast radius sends the next reader down the wrong path.

> **0.5.17 example:** the repository had 23 `override_server_args` call sites across 13 fields. After adding temporary logging to that helper on a running Qwen3-TTS server, three of them yielded definite conclusions and none of the three was affected; the one that actually broke was `pp_max_micro_batch_size`. The remaining call sites belong to model paths that run did not exercise, and the PR marks them explicitly as unaudited.
>
> The formulation finally used was: writing through `ServerArgs.override` and later reading it back off `ServerArgs` still works; writing and then expecting a config bag reader to observe the value does not. That can be acted on directly, whereas "these overrides have all stopped working" cannot.

### Step 6: open the PR once pre-commit and the full unit test suite pass

`pre-commit` runs autoflake, isort, black, and ruff locally, and the `lint` job runs the same set, so a clean local run predicts a green lint job.

The fake objects that stand in for SGLang components in unit tests (`tests/unit_test/fakes.py`) hard-code the shape of upstream's interfaces: when upstream's new code reads one more field, those fakes have to carry it, or the tests either fail outright or keep passing while no longer matching real behavior. The second outcome is the dangerous one. Updating them is therefore part of the port, not a cleanup afterwards.

**Before attributing any failure to the upgrade, verify whether it reproduces on the old pin.** Verification usually takes a single command, while a wrong attribution costs hours of investigation.

The PR description should carry the version diff table, a one-line rationale per adapted call site, the accuracy and performance comparison, and an explicit statement of what was not run.

> **0.5.17 example:** 3 of the 9 substantive commits touched only tests.
>
> ```
> test(scheduler): give the scheduler doubles a dcp_size-bearing server_args
> test(scheduler): model the 0.5.17 runtime-context contract in the doubles
> test: adapt two merged-in suites to the 0.5.17 contract
> ```
>
> The one failing unit test, `test_mp_runner_startup_failure_includes_child_factory_traceback`, reproduced identically on `0.5.16`: the test allows 10s for startup, while a cold `import sglang_omni.pipeline.stage_workers` takes 18.6s on that machine. That is a property of the machine, not a regression.

## 4. Keeping up with a moving main branch

The upgrade happens on a main branch that keeps moving, and the adaptation surface grows while you validate. Two consequences follow. First, **keep the upgrade PR minimal in scope**, with no incidental refactoring, to contain the conflict surface. Second, **land it quickly**, because the cost of maintaining the branch rises the longer it lives.

After every merge from main, run these grep checks item by item:

```bash
# retired module paths
git grep -n "sglang\.jit_kernel"

# call sites whose contract has relocated
git grep -n "override_server_args\|get_global_server_args"
git grep -n "SamplingBatchInfo("

# anything reaching into the config split
git grep -n "sglang\.srt\.server_args\|sglang\.srt\.runtime_context"
```

New code merged from main was written against the old pin and will reintroduce patterns you have already corrected.

> **0.5.17 example:** the upgrade branch merged from main three times in four days, with one commit devoted purely to re-adapting tests that arrived with a merge. While CI was blocked, main changed twice more: it deleted a file the PR had modified — `models/moss_tts/vocoder_decoder.py` was folded into `audio_tokenizer.py`, taking its `jit_kernel` import along — and it added two new `override_server_args` call sites in a deterministic-inference feature.

## 5. Checking that an interface exists is not checking that it still works

`override_server_args` in `sglang_omni/vendor/sglang/server_args.py` exists precisely to hold the version boundary in one place. It dispatches like this:

```python
legacy_override = getattr(server_args, "override", None)
if callable(legacy_override):
    legacy_override(source, **fields)
    return
# newer-version paths: get_context().override(...) / declare_late_resolution(...)
```

`ServerArgs.override` still exists on `0.5.17`, so the shim always enters the first branch and the two paths prepared for newer versions below it are unreachable. In the end the scheduler bypassed the shim and called `get_context().override(...)` directly.

The general conclusion: **when upstream deprecates an API by keeping the interface and removing its effect, deciding the version from "does the symbol exist" silently selects the unreachable path.** The dispatch condition should be the version number, or an observable effect — write a value, then read it back through the accessor consumers actually use — rather than whether the method is still there.

As of current main this shim is still unfixed: PR #1477 worked around it by calling `get_context().override(...)` directly in `omni_scheduler.py`, and left the shim's own dispatch condition untouched. The fix itself is one-off and belongs inside the shim, where a single change covers every call site.

But fixing it only settles this one case. No patch prevents upstream from deprecating another interface the same way next time, and changes of that kind raise no exception and are invisible to static tooling — they are Class B, and only the real runs in Step 4 will expose them.

## 6. Checklist

- [ ] Dependency metadata diffed; if an image rebuild is needed, the request goes to CODEOWNERS on day 0
- [ ] Changelog's Breaking Changes and Dependencies read; the private API surface diffed separately
- [ ] Baseline collected on the target machine, with warmup and repeats
- [ ] Pin updated; static scan clean (`pyrefly`)
- [ ] Retired import paths checked statically, including references inside `try/except ImportError`
- [ ] Every model CI covers starts up and serves correctly
- [ ] Every suspected regression reproduced on a clean GPU before localization begins
- [ ] Every failure verified on the old pin before attribution
- [ ] Unit test fake objects updated to the new interfaces
- [ ] `pre-commit` clean, full unit test suite passing
- [ ] The PR states what was not run
