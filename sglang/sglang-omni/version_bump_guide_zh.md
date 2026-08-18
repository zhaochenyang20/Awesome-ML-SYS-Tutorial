# SGLang Omni 的 backbone 版本升级指南

英文版本见 [A Practical Guide to Upgrading SGLang Omni's Backbone](./version_bump_guide.md)。

SGLang Omni 深度依赖 SGLang 的内部 API：约 512 个 Python 文件中，有 68 个直接导入了 `sglang.srt.*` 下的内部模块。这些模块属于 SGLang 的内部实现，私有接口不保证保持兼容。因此，每次升级 backbone 版本，表面上只是更新依赖，实际上还要让现有代码适配私有接口的变化。

> 本文的举例均取自 `0.5.16 → 0.5.17`（[PR #1477](https://github.com/sgl-project/sglang-omni/pull/1477)）。这次升级改动 10 个文件，新增 80 行、删除 43 行，规模远小于上一次涉及 162 个文件的升级；但最耗时的两个问题，类型检查器同样无法发现。上一次 `0.5.12.post1 → 0.5.16` 升级的完整复盘见 [API 对齐到浮点结合律：SGLang Omni 的 backbone 升级](./version_bump_zh.md)，本文不再展开其中的案例。

## 一、实际涉及的范围

导入点最多的是 `layers`、`managers`（scheduler 所在）与 `model_executor` 三个子系统，具体分布如下：

| 子系统 | import 点 |
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
| 其余（`platforms`、`configs`、`mem_cache`、`compilation`、`runtime_context`、`speculative`、`multimodal`、`environ`、`dllm`、`disaggregation`、`arg_groups`、`kernels`） | 约 50 |

上述统计覆盖 `sglang_omni/` 与 `sglang_omni_router/` 两个包。除 `sglang.kernels` 外，其余导入路径均位于 `sglang.srt.*` 下，因此不能指望其中的任何 API 保持稳定。

每次升级开始时都应重新统计表中数据。随着功能增加，Omni 对 SGLang 内部模块的依赖也会不断扩大，不能直接沿用上一次的结果。此外，表中的依赖分布可以帮助判断哪些条目可能影响 Omni，并据此筛选第1步需要阅读的内容。

`sglang_omni/vendor/sglang/` 是存放版本兼容代码的目录。凡是需要按 SGLang 版本选择不同实现的逻辑，都应集中到这里，不应分散在各个调用点。不过，兼容层本身也可能失效，详见[第五节](#五版本判断不能只看接口是否存在)。

## 二、两种变更及其对流程的约束

A 类：变量名、函数签名发生变更。 变量名被移除、重命名，或函数签名发生变化。这类问题必然抛出异常：`ImportError`、`TypeError: unexpected keyword argument`、`TypeError: missing required argument`。代码运行前，类型检查器就能识别出这类问题。

B 类：变量名、函数签名保持不变，语义变更。调用仍能正常解析，也接受相同参数并正常返回，但是不再发挥原有作用。这类问题不会抛出异常，静态工具也无法发现，只有用真实模型处理实际请求时才会暴露。

两类变更的处理成本相差一个量级，因此后续步骤的先后必须据此安排：静态扫描成本很低，应该尽早执行；估算工期时则应以 B 类为准，因为这类问题会占据大部分工期。

> 0.5.17： 本次共出现 6 个 A 类变更和 2 个 B 类变更。6 个 A 类在半天内修复完毕，2 个 B 类占用了其余工时。
>
> 一个详细的例子：`0.5.17` 将 `ServerArgs` 改为只读的启动记录，并把解析后的配置迁移到一类对象中。SGLang 源码将这些对象称为 config bag（类名 `_ConfigBag`），调用方通过 `get_exec()`、`get_parallel()` 等函数来读取相应配置。`ServerArgs.override()` 的签名没有任何变化，调用也能正常返回，但修改的内容不会再同步到 config bag。与此同时，`get_num_allocatable_reqs` 的正确写法变成了从 `get_parallel().pp_max_micro_batch_size` 读取，而不再是用 `get_server_args().pp_max_micro_batch_size`。结果就是，写入操作虽然正常返回，但其实没生效，scheduler 因而读到 `None`，并在处理第一个 batch 时因计算 `None - int` 而终止。
>
> 上一次复盘的结论是「Omni 依赖的是 SGLang 的具体行为，而接口并没有把这些行为定义下来」。这次升级遇到了一个更加需要仔细处理的问题，那就是接口完全未变，行为却已经改变。

## 三、执行顺序

### 第 0 步：比对依赖的metadata，确定是否需要重新构建镜像

这一步只需约十分钟，也不依赖后续的适配工作，却会影响整个升级的排期。

```bash
pip download --no-deps sglang==<旧版本> -d /tmp/sgl-old
pip download --no-deps sglang==<新版本> -d /tmp/sgl-new
# 解包后比对两侧的 METADATA / requires_dist
```

需要重点关注 flashinfer 与 torch这两个包。`.github/scripts/validate_omni_env_reusable.sh` 要求从镜像的 site-packages 加载，而不能从 venv 加载。这是为了复用镜像中预先生成的 JIT cache。若新版 `sglang` 的依赖要求更高版本的 flashinfer（有可能只是解析出来的结果），安装项目时会在 venv 中另行安装新版 flashinfer，该检查随即失败。这个版本要求来自 `sglang` 包自身，与 `pyproject.toml` 的写法无关，无法在 PR 内解决，只能执行以下两项操作：

1. 重新构建 CI 镜像。`docker/Dockerfile` 中固定了 flashinfer 版本，并 `COPY` 了 `/root/.cache/flashinfer/<版本>`。
2. 更新 digest。涉及六个工作流文件与 Dockerfile。

`/docker` 与 `.github` 均配置了 CODEOWNERS，因此镜像重建需要等待相关负责人安排。

- 目标 SGLang 版本会直接决定基础镜像、flashinfer 版本与 JIT cache 版本。这三项在开始编写适配代码前就已经确定，后续适配也不会改变镜像内容。
- 本地升级与验证不依赖镜像。重建镜像是为了通过 CI 的环境复用检查并复用 JIT cache，并不影响代码能否在本地运行；在本地安装新版 `sglang` 及其依赖即可。
- 本地适配和验证与镜像重建位于不同的关键路径上，应当并行推进。如果等适配完成后再重建镜像，就会额外增加一段不可控的等待时间，延长总工期。


> 0.5.17 ： 比对结果如下。
>
> | 依赖 | 0.5.16 | 0.5.17 |
> |---|---|---|
> | `helion` | `==0.2.6` | `==1.4` |
> | `flashinfer_python[cu13]` | `==0.6.14` | `==0.6.15.post1` |
> | `sgl-deep-gemm` | `==0.1.4.post1` | `==0.1.5.post1` |
> | `av`（Linux ARM） | 未固定 | `==16.1.0` |
> | `xxhash` | 无 | 新增 |
>
> 其中，flashinfer 的版本变化直接导致 CI 无法通过，失败信息为 `flashinfer must come from the image, not /data/omni-ci/pr-1477/omni` 与 `Torch and FlashInfer must use the image installation for JIT cache reuse`。digest 更新涉及 22 处修改。镜像重建请求直到升级开始后的第四天才提出，CI 自首次运行起便持续失败。

### 第 1 步：阅读 changelog，确定变化范围

首先阅读 Breaking Changes 与 Dependencies 两节，预估架构层面的工作量。

同时要明确 changelog 的范围。它通常不会完整记录私有接口签名的变化，而 Omni 依赖的大多正是私有接口。即使列出了某项变更，changelog 往往也只说明具体变更，不说明旧代码的具体失效方式。因此，changelog 只能指出排查方向，不能代替逐项检查。

因此最好还是直接比对代码，具体步骤是解包新旧两个 wheel，只对项目实际导入的模块执行 diff。

> 0.5.17 changelog 包含 600 余条记录、20 余个分类，而本次升级需要完成的 8 项改动中，changelog 只覆盖了 2 项。
>
> | 改动 | changelog 是否覆盖 |
> |---|---|
> | `ServerArgs` 只读化，配置迁移至 config bag | 覆盖，见 Breaking Changes |
> | `sglang.jit_kernel` 并入 `sglang.kernels` | 覆盖，见 Kernel Library |
> | `SamplingBatchInfo` 的 `vocab_mask` / `apply_mask_func` 合并进 `grammar_mask` | 未覆盖 |
> | `SchedulerLogprobResultProcessor` 移除 `server_args` | 未覆盖 |
> | `SchedulerDPAttnAdapter` 新增必填的 `model_runner` | 未覆盖 |
> | `SchedulerLoadInquirer` 新增三个必填的遥测 accessor | 未覆盖 |
> | `pp_max_micro_batch_size` 的读取位置迁移至 `get_parallel()` | 未覆盖 |
> | token clamp 读取 `dcp_size` | 未覆盖 |
>
> 关于配置拆分，changelog 的原文是「Code that mutated ServerArgs at runtime must route through the new accessors」。这句话本身准确，却没有说明旧写法仍会正常返回，而也导致踩了坑。

### 第 2 步：建立性能基线

两个版本的基线在同一台机器上、用同一种方法分别采集，每次测量前都要预热并重复执行。核心是：任何代码路径的首次执行结果都不能作为有效测量。因为会有各种原因导致预热问题。

两次运行之间执行 `.github/scripts/delete_gpu_process.sh`，CI 工作流也采用同样的做法。前一次运行留下的显存占用，是造成假的回退的第二大原因，仅次于 cache 未预热。

此外，两个版本都必须使用各自 pin 对应的依赖。因为性能测试实际比较的是两套完整的依赖，而不只是一个包的两个版本。

评测工具位于 `benchmarks/eval/`。CI 检查的指标也正是 PR 描述需要呈现的指标，因此从一开始就应按同一口径采集。

> 0.5.17： 本次共排查四例疑似性能回退，但都是虚惊一场。
>
> | 观察结果 | 实际成因 |
> |---|---|
> | Qwen3-TTS 吞吐下降 | 需执行至第四次才进入稳态 |
> | TTS stage-2 TTFC p95 为 0.5838 | 重复执行后回落至 0.506–0.529 的基线区间 |
> | `ws_stream` 延迟 p95 为 13.82s，超出阈值 | 两次重复执行均在阈值内 |
> | MMMU 0.959 qps、16.05s 延迟 | inductor cache 未预热，清理后连续三次通过 |
>
> 两个版本使用的依赖分别是：`0.5.16` 对应 flashinfer 0.6.14、helion 0.2.6、sgl-deep-gemm 0.1.4.post1，`0.5.17` 对应 0.6.15.post1、1.4、0.1.5.post1。

### 第 3 步：更新 pin，执行静态扫描

安装新版本后，用类型检查器根据新版的内部实现检查现有代码。A 类变更可以在这一步集中修复，成本很低。

虽然项目目前没有配置类型检查器，但使用mypy或者带tyecheck的LSP，都能帮助快速发现问题，尽管他们各自有着不同的噪音。我个人体验是`pyrefly` 的信噪比显著优于 `ty`，而二者速度比更主流的`pyright`快不少。但`ty`产生的噪声足以淹没有效的诊断信息。

**`try/except ImportError` 会掩盖导入失败。** 导入失败后，这类分支会回退到备用实现，程序也会在没有提示的情况下改用性能较低的路径，行为是既不抛出异常，也不产生日志，仅表现为性能下降，而且没有任何诊断线索。因此，非常推荐通过静态搜索找出已经失效的导入路径，包括 `try/except ImportError` 内部的引用，不能等到运行时才发现。

> **提示：** grep 虽然能找到匹配文本，但也会匹配到注释和字符串，而且它无法识别「导入位于捕获 `ImportError` 的 `try` 块内」这种语法结构。ast-grep 根据语法树匹配，可以同时避开这两个问题。若要准确查找对某个已失效模块的导入：
>
> ```bash
> ast-grep --lang python --pattern 'from sglang.jit_kernel.$$$A import $$$B'
> ```
>
> 若要列出所有位于 `try/except ImportError` 内的导入，供升级前逐条确认，可执行以下命令（本文写作时在 `sglang_omni/` 下找到 17 处）：
>
> ```bash
> ast-grep --lang python --pattern 'try:
>     $$$BODY
> except ImportError:
>     $$$H' sglang_omni/
> ```
>
> 该写法同时覆盖 `from X import Y` 与 `import X`，并排除捕获其他异常的 `try`。需要逐条确认两件事：该导入在新版本中是否仍能成功；若不能，回退逻辑是否仍符合预期。

同时，建议将每项修复各自提交一个 commit，并在 commit 标题中说明原因。这样既能直接为 PR 描述准备材料，也能把 A 类修复与后续的 B 类修复分开。

> **0.5.17 实例：** 这一步共提交了三个 commit。
>
> ```
> fix(qwen3-omni): drop the SamplingBatchInfo grammar-mask kwargs
> fix(moss-tts-local): import flash_attn_varlen_func from kernels.ops
> fix(scheduler): adapt to the 0.5.17 scheduler-component contracts
> ```
>
> 其中第二个 commit 对应上述 `try/except ImportError` 情形：MOSS-TTS-Local 的 vocoder 会在该异常处理分支中回退到 SDPA；由于 `sglang.jit_kernel` 在 `0.5.17` 中已经退役，这项import必然失败。

### 第 4 步：启动 CI 覆盖的全部模型

CI 的模型矩阵就是必测清单，也界定了项目实际承诺支持的范围。

| workflow | 模型 | 检查项 |
|---|---|---|
| `test-asr-ci.yaml` | MOSS-Transcribe-Diarize；Fun-ASR 或 Qwen3-ASR（可选） | WER、RTF、吞吐 |
| `test-tts-ci.yaml` | Higgs 或 MOSS-TTS-Local（可选），5 个 stage | WER、SIM、TTFC、延迟、流式一致性、router DP2 压测 |
| `test-qwen3-omni-ci.yaml` | Qwen3-Omni，11 个 stage | thinker 长度、TTS WER 与 SIM、MMMU 与 MMSU 准确率及速度、talker、video |
| `omni-ci.yaml` → PR Test | — | 全部单元测试 |

静态扫描通过，并不说明代码一定能够运行，B 类变更通常只有到这一步才会暴露。

需要在两个版本之间反复切换时，常见做法是把另一版 `sglang` 解包到单独的目录，再通过 `PYTHONPATH` 将该目录放到 `sys.path` 最前面，使 `import sglang` 加载这个版本，而不是已经安装的版本。这种做法通常称为 shadow。切换时只需修改一个环境变量，无须反复重装整套依赖。

shadow 有两项限制。第一，它只替换 `sglang` 本身，其余依赖仍来自当前安装环境，因此组合出的依赖既不完全对应旧版，也不完全对应新版。它适合快速确认代码能否正常运行，却不适合生成第 2 步所需的对比数据；这些数据必须在两个版本各自完整的依赖下采集。

第二，shadow 只有在 worker 继承父进程环境时才会生效。CI 测试通过 `start_server_from_cmd`（`benchmarks/benchmarker/utils.py`）启动 worker。会先复制 `os.environ`，再叠加调用方传入的 `env`。未传入 `PYTHONPATH` 的测试会保留父进程的 shadow 设置；如果在 `process_env` 中固定了 `PYTHONPATH`，该设置就会被覆盖，worker 实际运行的是已经安装的版本。此时两个版本使用的是同一份代码，测试却会错误地得出「无回退」的结论。

开始对比前，应找到传给 `launch_managed_router` 的 `process_env`，逐个确认测试属于哪一类。后一类测试必须分别安装对应版本后再执行。截至当前 main，`test_tts_serving_ci.py` 属于后一类：它在 `process_env` 中将 `PYTHONPATH` 固定为项目根目录（`tests/test_model/test_tts_serving_ci.py:296`），其 benchmark 子进程也采用同样的设置（同文件 `:369`）。

必须如实记录实际执行了哪些测试，不能用一张全部通过的结果表暗示已经完整覆盖。

> 0.5.17： 两个 B 类变更都在这一步暴露，修复这两个问题的 commit 也最耗时。
>
> ```
> fix(scheduler): route the pp_max_micro_batch_size default through the context
> fix(scheduler): mirror the 0.5.17 step counters and batch launch timestamp
> ```
>
> 本次实际执行了 17 个 GPU 测试中的 12 个，TTS stage 3–5 与四个 Qwen3-Omni video job 未执行，该情况已写入 PR。

### 第 5 步：排查回退时，先复现再定位

第 2 步的约束在这里同样适用：任何性能下降，在干净的环境里复现之前，都不能判定为回退。 复现通常只需几分钟，定位根因却可能需要几天，因此必须先复现，再定位。

确认回退后，可以在运行中的 server 上为兼容层加入临时日志，记录每个调用点当时实际读取/写入了哪些值。与静态检查相比，这些日志反映的是运行时的真实行为，而不是推测。

但日志只能覆盖本次运行实际触发的调用点，未触发的部分仍然无法判断。结论必须来自实测结果，并明确列出哪些路径没有覆盖；夸大影响范围会误导后续排查。

> 0.5.17： 全仓共有 23 个 `override_server_args` 调用，涉及 13 个字段。在一台正在运行的 Qwen3-TTS server 上为该 helper 加入临时日志后，实际触发的三个调用放都得到了明确结论，且均未受影响；真正失效的只有 `pp_max_micro_batch_size`。其余调用方位于本次运行未触发的模型路径中，PR 已将其明确标为未审计。
>
> 基于这次实测，PR 最终将结论限定为：通过 `ServerArgs.override` 写入后再从 `ServerArgs` 读回的路径仍然有效。如果写入后，某个 config bag 的使用方要读取该值，那么这条路径已经失效。前面这种表述可以直接当作结论，在这个场景里，我们不可以简单地说「这些 override 已全部失效」。

### 第 6 步：pre-commit 与全部单元测试通过后提交 PR

`pre-commit` 在本地执行 autoflake、isort、black、ruff；CI里的`lint` job 也执行同一组检查。

单元测试中用于替代 SGLang 组件的 fake 对象（`tests/unit_test/fakes.py`）固定了上游接口的结构。上游代码每多读取一个字段，这些 fake 对象就必须补上该字段；否则，测试要么直接报错，要么在已经偏离真实行为的情况下继续通过，后一种情况更加危险。因此，更新这些 fake 对象本身就是适配工作的一部分，而不是最后的收尾工作。

将任何失败归因于本次升级前，都必须先在旧版本上确认能否复现。 验证通常只需执行一条命令，而一旦误判，可能浪费数小时进行排查。

PR 描述必须包含版本差异表、每项适配的一行说明、准确率与性能对照，以及对未执行项目的明确说明。

> **0.5.17 实例：** 9 个实质性 commit 中有 3 个仅涉及测试。
>
> ```
> test(scheduler): give the scheduler doubles a dcp_size-bearing server_args
> test(scheduler): model the 0.5.17 runtime-context contract in the doubles
> test: adapt two merged-in suites to the 0.5.17 contract
> ```
>
> 唯一失败的单元测试 `test_mp_runner_startup_failure_includes_child_factory_traceback` 在 `0.5.16` 上也出现了完全相同的失败：该测试为启动预留 10s，而该机器上冷启动 `import sglang_omni.pipeline.stage_workers` 需要 18.6s。问题来自这台机器的性能，而不是版本回退。

## 四、应对持续变化的 main 分支

升级期间，main 分支仍在持续变化，需要适配的范围就可能继续扩大。因此有两条原则。第一，升级 PR 应尽量缩小改动范围（make minimal changes），不要附带额外重构，以减少冲突。第二，应尽快合入；分支存在的时间越长，维护成本就越高，不过这一点也得看整个项目roadmap安排。

每次merge main后，建议重新检查性能是否回退，以及是否有一样的变更需要处理：

> 0.5.17： 该升级分支在四天内三次从 main 合入变更，其中一个 commit 只用于重新适配合入的测试。在 CI 阻塞期间，main 又合入了两项相关变更：一项将 `models/moss_tts/vocoder_decoder.py` 并入 `audio_tokenizer.py`，删除了本 PR 修改过的原文件，并一并迁移其中的 `jit_kernel` 导入；另一项在确定性推理特性中新增了两个 `override_server_args` 调用点。

## 五、版本判断不能只看接口是否存在

`sglang_omni/vendor/sglang/server_args.py` 中的 `override_server_args`，专门用于把不同版本的处理逻辑集中在一处。它按以下方式选择具体实现：

```python
legacy_override = getattr(server_args, "override", None)
if callable(legacy_override):
    legacy_override(source, **fields)
    return
# get_context().override(...) / declare_late_resolution(...)
```

`ServerArgs.override` 在 `0.5.17` 中依然存在，因此执行这段代码时，每次都会进入第一个分支，后面为新版本准备的两条路径永远不会执行。这里暂时的修法是让scheduler 绕过这段代码，直接调用 `get_context().override(...)`。

更通用的结论是：当上游通过「保留接口但使其不再生效」来废弃 API 时，只凭「函数是否存在」判断，会在不报错的情况下选中已经失效的旧分支，使新版本路径不生效。 实际测试时应依据版本号，或者检查可以实际观察到的结果，比如写入一个值，再通过使用方实际调用的 accessor 将其读回，而不能只看方法是否仍然存在。

截至当前 main，这个小问题仍未修复。PR #1477 只是在 `omni_scheduler.py` 中直接调用 `get_context().override(...)`，从而绕过了它，并没有修改 shim 自身的逻辑。这个问题应直接在 shim 中修复。

不过，这只能解决当前这一处，无法防止上游下次以同样方式废弃其他接口。这类改动既不会抛出异常，也无法被静态工具发现，属于本文开头所说的 B 类，只有在第 4 步实际运行模型时才会暴露。

## 六、检查清单

- [ ] 比对依赖元数据；如需重建镜像，第 0 天即向 CODEOWNERS 提出镜像重建请求
- [ ] 阅读 changelog 的 Breaking Changes 与 Dependencies；私有 API 得另行比对
- [ ] 在目标机器上采集基线，包含 warmup 与重复测量
- [ ] 更新 pin；静态扫描类型问题
- [ ] 静态检查已废弃的import路径，包含 `try/except ImportError` 内部的引用
- [ ] CI 覆盖的每个模型均可启动并正常服务
- [ ] 每个疑似回退都在干净的GPU上复现，之后再开始归因
- [ ] 每个失败都先在旧版本上验证能否复现，再归因
- [ ] 单元测试的 fake 对象已按新接口更新
- [ ] `pre-commit` 通过，全部单元测试通过
- [ ] PR 中注明未执行项
