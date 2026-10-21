# A Practical Guide to Upgrading SGLang Omni's Backbone

For the Chinese version, see [SGLang Omni Backbone Version Upgrade Guide](./version_bump_guide_zh.md).

SGLang Omni depends heavily on SGLang's internal APIs. Of its roughly 512 Python files, 68 directly import internal modules under `sglang.srt.*`. These modules are implementation details of SGLang, and their private interfaces come with no compatibility guarantees. A backbone bump looks like a dependency update, but in practice it means adapting the existing code to changes in those private interfaces.

> Most examples in this guide come from the `0.5.16 → 0.5.17` upgrade ([PR #1477](https://github.com/sgl-project/sglang-omni/pull/1477), later superseded by the `0.5.18` upgrade in [PR #1719](https://github.com/sgl-project/sglang-omni/pull/1719) and never merged on its own). It touched 10 files, adding 80 lines and removing 43. That is far smaller than the previous upgrade, which touched 162 files, but as before, neither of the two most time-consuming issues was visible to a type checker. For a full retrospective on the earlier `0.5.12.post1 → 0.5.16` upgrade, see [From API Alignment to Floating-Point Associativity: Upgrading SGLang Omni's Backbone](./version_bump.md). This guide does not repeat those cases.

The sglang-omni repository also has an official guide, [Bumping the SGLang Pin](https://github.com/sgl-project/sglang-omni/blob/main/docs/developer_reference/bump_version.md) ([PR #1766](https://github.com/sgl-project/sglang-omni/pull/1766)). It lists the files that must change together, the places where Omni depends on SGLang, and the requirements for updating the image and validating the bump. This guide instead lays out the practical steps in order; the two are meant to be read side by side.

Counts, code locations, and the CI configuration in this guide change as the project evolves and are only there to illustrate the steps. Undated ones reflect August 2026; dated ones were checked against main on that date.

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
> Here is one of the Class B changes in detail. `0.5.17` turned `ServerArgs` into a read-only startup record and moved the resolved configuration into a set of objects. The SGLang source calls these objects config bags (class `_ConfigBag`), and callers use functions such as `get_exec()` and `get_parallel()` to read the corresponding settings. The signature of `ServerArgs.override()` did not change, and the call still returned normally, but its changes were no longer propagated to the relevant config bag. At the same time, `get_num_allocatable_reqs` now had to read `get_parallel().pp_max_micro_batch_size` instead of `get_server_args().pp_max_micro_batch_size`. The write therefore returned normally but had no effect: the scheduler read `None` and crashed on the first batch evaluating `None - int`.
>
> The conclusion from the previous retrospective was that "Omni depends on SGLang's concrete behavior, and the interface never pinned that behavior down." This upgrade hit a case that needs even more care: the interface did not change at all, but its behavior did.

## 3. Order of work

### Step 0: compare dependency metadata and decide whether the image needs to be rebuilt

This takes about ten minutes and does not depend on any later adaptation work, but it affects the schedule for the entire upgrade.

```bash
pip download --no-deps sglang==<old-version> -d /tmp/sgl-old
pip download --no-deps sglang==<new-version> -d /tmp/sgl-new
# unpack both, then compare METADATA / requires_dist
```

Pay particular attention to flashinfer and torch. `.github/scripts/validate_omni_env_reusable.sh` requires them to be loaded from the image's site-packages, not from the venv, so that the prebuilt JIT cache in the image can be reused. If the new `sglang` requires a newer flashinfer (possibly only as a result of dependency resolution), installing the project puts that newer version in the venv, and the check fails. The requirement comes from the `sglang` package itself, not from `pyproject.toml`, so it cannot be fixed inside the PR. The only way through is to do both of the following:

1. Rebuild the CI image. `docker/Dockerfile` pins the flashinfer version and `COPY`s `/root/.cache/flashinfer/<version>`.
2. Update the digest in every workflow file and the Dockerfile.

Both `/docker` and `.github` have CODEOWNERS, so an image rebuild has to wait until the owners schedule it.

- The target SGLang version directly determines the base image, the flashinfer version, and the JIT cache version. All three are known before adaptation work starts, and nothing found during adaptation will change the image contents.
- Local adaptation and validation do not depend on the image. The rebuild is needed for CI's environment-reuse check and for JIT cache reuse; it does not affect whether the code can run locally. Install the new `sglang` and its dependencies for local work.
- Local adaptation and the image rebuild are on separate critical paths and should proceed in parallel. Requesting the rebuild only after adaptation is complete adds an unpredictable wait to the overall schedule.

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

The reliable approach is to diff the code itself: unpack the old and new wheels, then diff only the modules the project actually imports.

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
> On the config split, the changelog said, "Code that mutated ServerArgs at runtime must route through the new accessors." That statement is accurate, but it does not say that the old call still returns normally, which is exactly what caused the failure described above.

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

The project does not currently configure a type checker, but running mypy or using an LSP with type checking can still find problems quickly, even though each tool produces a different amount of noise. In my experience, `pyrefly` has a much better signal-to-noise ratio than `ty`, whose noise is enough to bury the useful diagnostics; both are considerably faster than the more widely used `pyright`.

**`try/except ImportError` hides import failures.** When an import fails, these branches fall back to another implementation, and the program silently switches to a slower path. There is no exception or log, only a performance drop with no diagnostic clue. Retired import paths should therefore be found with a static search, including references inside `try/except ImportError`, rather than left for runtime testing to uncover.

> **Tip:** grep can find matching text, but it also matches comments and strings, and it cannot tell that an import is inside a `try` block that catches `ImportError`. ast-grep matches the syntax tree and avoids both problems. To find imports from a known retired module:
>
> ```bash
> ast-grep --lang python --pattern 'from sglang.jit_kernel.$$$A import $$$B'
> ```
>
> To list every `try` block with an `except` clause that catches `ImportError` or `ModuleNotFoundError`, for review before the upgrade:
>
> ```bash
> ast-grep scan --inline-rules '
> id: guarded-import
> language: python
> rule:
>   kind: try_statement
>   has:
>     kind: except_clause
>     has:
>       kind: identifier
>       regex: ^(ImportError|ModuleNotFoundError)$
>       stopBy:
>         kind: block' sglang_omni/
> ```
>
> This covers both `from X import Y` and `import X`, and also matches `except ImportError as exc:`, tuple forms such as `except (ImportError, OSError):`, and `ModuleNotFoundError`. A plain pattern like `try: ... except ImportError: ...` misses all of these: as of 2026-10-21, it finds 38 blocks under `sglang_omni/` on main, while the rule above finds 70. `stopBy: block` keeps the search out of the handler body, so a `raise ImportError(...)` inside an unrelated handler does not match. Check two things for every match: whether the import still succeeds with the new version, and, if it does not, whether the fallback still behaves as intended.

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

This table shows the CI matrix as of August 2026. Since then, the TTS, ASR, and Omni stages have each gained models (for example Qwen3-TTS, Whisper, CosyVoice3, and MiniCPM-o), and the CI labels have been split into per-stage and per-model labels. Take the actual list of required tests from `.github/workflows/` and the CI section of the PR template at the time of the upgrade.

Passing the static checks does not mean the code will run. Class B changes usually surface only at this stage.

When switching repeatedly between two versions, a common approach is to unpack the other `sglang` version into a separate directory, then use `PYTHONPATH` to put that directory first on `sys.path`. This makes `import sglang` load that copy instead of the installed one. This is usually called a shadow. Switching then requires changing one environment variable rather than reinstalling the full dependency stack.

A shadow has two limitations. First, it replaces only `sglang`; all other dependencies still come from the current environment, so the resulting stack matches neither the old version nor the new one exactly. It is useful for a quick check that the code runs, but not for collecting the comparison data from Step 2. That data must be collected with each version's complete dependency stack.

Second, a shadow works only if the worker inherits the parent process's environment. CI tests start workers through `start_server_from_cmd` (`benchmarks/benchmarker/utils.py`). It first copies `os.environ`, then applies the `env` supplied by the caller. A test that does not pass `PYTHONPATH` keeps the parent's shadow setting. If a test sets `PYTHONPATH` in `process_env`, that value overrides the shadow and the worker runs the installed version. The two test runs then use the same code and incorrectly report no regression.

Before comparing versions, find the `process_env` passed to `launch_managed_router` and check which category each test falls into. Tests in the second category must be run after installing each version in turn. As of 2026-10-21, `test_tts_serving_ci.py` on main is in this category: it sets `PYTHONPATH` to the project root in `process_env` (`tests/test_model/test_tts_serving_ci.py:297`), and its benchmark subprocess does the same (`:379` in the same file).

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

> For `0.5.17`, there were 23 `override_server_args` call sites across the repository, covering 13 fields. Temporary logging was added to the helper on a running Qwen3-TTS server. The three call sites reached by that run all yielded clear results, and none was affected; only `pp_max_micro_batch_size` was actually broken. The remaining callers were on model paths not reached by the run, and the PR marks them as unaudited.
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

### Step 7: after the merge, recalibrate the CI thresholds in a separate PR

Every CI gate is backed by a reference value stored in the code: the worst value from several clean runs on the H100 machine CI uses. The threshold is computed at import time by multiplying the reference by a fixed slack. After a backbone upgrade, throughput, latency, and RTF shift across the board, and accuracy, WER, and speaker similarity can move slightly as well, but the references still describe the old image. A threshold that is too loose will miss later real regressions; one that is too tight makes CI fail at random on unrelated PRs. Once the upgrade PR is merged, every model covered by CI must therefore be recalibrated, for both speed and quality references; the [official bump guide](https://github.com/sgl-project/sglang-omni/blob/main/docs/developer_reference/bump_version.md#the-pr) requires the same.

As of 2026-10-21, the references and slack factors on main live in these files:

| File | Coverage | Speed slack (higher-is-better / lower-is-better) |
|---|---|---|
| `tests/test_model/tts_ci_config.py` | all TTS presets | 0.75 / 1.25 |
| `tests/test_model/test_tts_serving_ci.py` | TTS serving | same as TTS |
| `tests/test_model/omni_ci_config.py` | Qwen3-Omni, MiniCPM-o | 0.875 / 1.125 |
| `tests/test_model/asr_ci_config.py` | Fun-ASR, Qwen3-ASR, Whisper | 0.9 / 1.1 |
| `tests/test_model/test_asr_ci_multi_speaker.py` | MOSS-Transcribe-Diarize | 0.9 / 1.1 (0.8 / 1.2 for AISHELL4-long and GoogleTime) |
| `tests/test_ci/tts_mps_ci_config.py` | TTS MPS | 0.75 / 1.25 |

Speed references are multiplied by the slack for their direction (`apply_slack` in `tests/utils/__init__.py`). WER uses 1.25 by default (`apply_wer_slack`; ASR passes 1.1), and UTMOS uses 0.97 (`apply_mos_slack`). Accuracy and TTS speaker similarity get no slack and are compared with the reference directly, except in TTS MPS, where similarity is multiplied by 0.97. Calibration changes only the references, never the slack.

The calibration tool is the `calibrate-h100-ci` skill in a private maintainer repository (linked from `tests/README.md`). Before [PR #2132](https://github.com/sgl-project/sglang-omni/pull/2132) it lived in the sglang-omni repository as `.claude/skills/tune-ci-thresholds/`. As of that last in-repo version (2026-09-12), the procedure was:

1. Stop the CI runners on the host and wait for running jobs to finish. Calibration shares the machine with CI, and CI jobs add CPU, GPU, and bandwidth contention. Restore CI as soon as the last measurement finishes rather than after the report is written.
2. Pin each GPU group to its NUMA-local CPU cores (`OMNI_CI_CPUSET`, applied with `sched_setaffinity` in `tests/test_model/conftest.py`). The measurements that set thresholds always run on the group with the fewest CPU cores, so the resulting thresholds are conservative for every group; the other groups are for measurement only.
3. Run the precheck: dependency versions, freed GPU memory, assets, and a CUDA smoke test, while recording an environment fingerprint (image digest, driver, torch and sglang versions). Warm the FlashInfer JIT cache and the UTMOS and speaker-similarity models beforehand; a cold start produces invalid rounds.
4. Run each stage 5 times and take the worst value over the clean rounds: the minimum for higher-is-better metrics, the maximum for lower-is-better ones. No percentiles, no trimming.
5. A round that deviates sharply (MAD z-score above 3.5 and at least 20% away from the nearest value) is marked destructive, discarded whole, and replaced by two new rounds, up to 15 rounds per stage. If three or more rounds are rejected, or the values split into two clusters, the whole stage is rerun once. If it is still unstable after that, every observation is kept, the worst one sets the reference, and the stage is flagged for review. An attempt interrupted by CPU contention is discarded and rerun as well.
6. The tool produces a before/after comparison for every reference. A person reviews it before the values are written back to the files above. After writing, regenerate the plan and confirm that every reference now matches.

The calibration PR changes only reference values. Its description states the measured commit, the image digest, the sglang and torch versions, the GPUs and cpuset, every reference's per-round results and change, and any reference deliberately left unchanged, with the reason. After it merges, rerun CI with the labels for the affected models.

A few points to keep in mind:

- Calibrate after the upgrade PR merges, on the merged commit, in a separate PR. Calibrating on the upgrade branch measures an image that main does not run, and mixes threshold changes into the upgrade PR's A/B review. Conversely, a gate that main also fails should not be adjusted in the upgrade PR; it belongs to a calibration PR of its own.
- Do not pass `-x` to pytest. An assert that fails against the old threshold is still a valid observation, and `-x` stops the remaining metrics from being collected.
- A single bad round can set the final gate. The warmup and GPU-cleanup rules from Step 2 apply here too.
- Calibration is not the Step 2 baseline. The baseline decides whether the upgrade itself introduced a regression; calibration happens after the upgrade has been accepted and brings the CI thresholds in line with the new dependency stack.

> For `0.5.16`, the first pass of the post-merge calibration ([PR #1260](https://github.com/sgl-project/sglang-omni/pull/1260)) took the raw worst of 5 rounds, so a single contended or cold-start round decided the gate. MMSU throughput was pinned at 12.1 req/s against a clean 54.9, and VideoAMME latency at 32.9s against a clean 8.7s. Those gates could not have caught a real 4x regression. After destructive rounds were rejected and the references recomputed, the result went from 117 loosened and 35 tightened to 70 tightened, 12 loosened, and 89 unchanged.
>
> For `0.5.18`, the post-merge calibration ([PR #1811](https://github.com/sgl-project/sglang-omni/pull/1811)) mostly tightened stale gates; MMSU throughput, for example, went from 54.9 to 78.8 req/s.

## 4. Dealing with a main branch that keeps moving

The main branch continues to change during an upgrade, and the scope of the adaptation may grow with it. Two principles follow. First, keep the upgrade PR minimal and leave unrelated refactoring out of it, to reduce conflicts. Second, merge it as soon as possible, because the longer the branch stays open, the more it costs to maintain. The exact timing still depends on the project's overall roadmap.

After every merge from main, check again for performance regressions and for new instances of the same compatibility problems.

> During the `0.5.17` upgrade, changes from main were merged into the upgrade branch three times in four days. One commit existed only to re-adapt tests brought in by a merge. While CI was blocked, two more relevant changes landed on main: `models/moss_tts/vocoder_decoder.py` was merged into `audio_tokenizer.py`, deleting the original file modified by the PR and moving its `jit_kernel` import with it; and a deterministic-inference feature added two new `override_server_args` call sites.

## 5. Do not infer the version from interface existence alone

Compatibility code often decides which SGLang version it is running against by probing whether an interface exists. When upstream deprecates an API by leaving the interface in place but removing its effect, checking only whether the function exists silently selects the broken legacy branch and prevents the new-version path from running. Use the version number instead, or check an observable result: write a value, then read it back through the accessor that the consumer actually uses. The presence of the method alone is not enough.

Fixing one compatibility shim cannot stop upstream from deprecating another interface in the same way. Such a change raises no exception and cannot be found by static tools. It is a Class B change, and it surfaces only when real models are run in Step 4.

> For `0.5.17`, `override_server_args` in `sglang_omni/vendor/sglang/server_args.py`, which is meant to keep the handling for different versions in one place, selected an implementation as follows:
>
> ```python
> legacy_override = getattr(server_args, "override", None)
> if callable(legacy_override):
>     legacy_override(source, **fields)
>     return
> # get_context().override(...) / declare_late_resolution(...)
> ```
>
> `ServerArgs.override` still exists in `0.5.17`, so this code always entered the first branch, and the two paths below it for the new version never ran. The upgrade PR worked around it by having the scheduler bypass this code and call `get_context().override(...)` directly. The shim itself was fixed later in [PR #1719](https://github.com/sgl-project/sglang-omni/pull/1719): instead of probing for `ServerArgs.override`, it checks whether the `server_args` passed in is the record already published to the runtime context, and writes through `get_context().override(...)` or `declare_late_resolution(...)` (since renamed `declare_resolution(...)`) accordingly.

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
- [ ] After the merge, recalibrate every model's CI thresholds on the merged commit, in a separate PR
