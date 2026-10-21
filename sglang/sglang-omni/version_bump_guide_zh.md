# SGLang Omni的backbone版本升级指南

英文版本见[A Practical Guide to Upgrading SGLang Omni's Backbone](./version_bump_guide.md)。

SGLang Omni深度依赖SGLang的内部API：约512个Python文件中，有68个直接导入了`sglang.srt.*`下的内部模块。这些模块属于SGLang的内部实现，私有接口不保证保持兼容。因此，每次升级backbone版本，表面上只是更新依赖，实际上还要让现有代码适配私有接口的变化。

> 本文的举例主要取自`0.5.16 → 0.5.17`（[PR #1477](https://github.com/sgl-project/sglang-omni/pull/1477)，后被`0.5.18`升级[PR #1719](https://github.com/sgl-project/sglang-omni/pull/1719)取代，未单独合入）。这次升级改动10个文件，新增80行、删除43行，规模远小于上一次涉及162个文件的升级；但最耗时的两个问题，类型检查器同样无法发现。上一次`0.5.12.post1 → 0.5.16`升级的完整复盘见[API对齐到浮点结合律：SGLang Omni的backbone升级](./version_bump_zh.md)，本文不再展开其中的案例。

sglang-omni仓库中另有一份官方文档[Bumping the SGLang Pin](https://github.com/sgl-project/sglang-omni/blob/main/docs/developer_reference/bump_version.md)（[PR #1766](https://github.com/sgl-project/sglang-omni/pull/1766)），系统列出了升级时需要同步修改的文件、Omni依赖SGLang的各处接口，以及镜像更新与验证的要求。本文则按执行顺序整理具体做法，二者可以对照阅读。

文中的统计数字、代码位置与CI配置会随项目演进而变化，仅用于说明做法：未注明日期的为2026年8月的状态，注明日期的则核对于当日的main。

## 一、实际涉及的范围

导入点最多的是`layers`、`managers`（scheduler所在）与`model_executor`三个子系统，具体分布如下：

| 子系统 | import数 |
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
| 其余（`platforms`、`configs`、`mem_cache`、`compilation`、`runtime_context`、`speculative`、`multimodal`、`environ`、`dllm`、`disaggregation`、`arg_groups`、`kernels`） | 约50 |

上述统计覆盖`sglang_omni/`与`sglang_omni_router/`两个包。除`sglang.kernels`外，其余导入路径均位于`sglang.srt.*`下，因此不能指望其中的任何API保持稳定。

每次升级开始时都应重新统计表中数据。随着功能增加，Omni对SGLang内部模块的依赖也会不断扩大，不能直接沿用上一次的结果。此外，表中的依赖分布可以帮助判断哪些条目可能影响Omni，并据此筛选第1步需要阅读的内容。

`sglang_omni/vendor/sglang/`是存放版本兼容代码的目录。凡是需要按SGLang版本选择不同实现的逻辑，都应集中到这里，不应分散在其他地方。不过，兼容层本身也可能失效，详见[第五节](#五版本判断不能只看接口是否存在)。

## 二、两种变更及其对流程的约束

A类：变量名、函数签名发生变更。变量名被移除、重命名，或函数签名发生变化。这类问题必然抛出异常：`ImportError`、`TypeError: unexpected keyword argument`、`TypeError: missing required argument`。代码运行前，类型检查器就能识别出这类问题。

B类：变量名、函数签名保持不变，语义变更。调用仍能正常解析，也接受相同参数并正常返回，但是不再发挥原有作用。这类问题不会抛出异常，静态工具也无法发现，只有用真实模型处理实际请求时才会暴露。

两类变更的处理成本相差一个量级，因此后续步骤的先后必须据此安排：静态扫描成本很低，应该尽早执行；估算工期时则应以B类为准，因为这类问题会占据大部分工期。

> 0.5.17：本次共出现6个A类变更和2个B类变更。6个A类在半天内修复完毕，2个B类占用了其余工时。
>
> 一个详细的例子：`0.5.17`将`ServerArgs`改为只读的启动记录，并把解析后的配置迁移到一类对象中。SGLang源码将这些对象称为config bag（类名`_ConfigBag`），调用方通过`get_exec()`、`get_parallel()`等函数来读取相应配置。`ServerArgs.override()`的签名没有任何变化，调用也能正常返回，但修改的内容不会再同步到config bag。与此同时，`get_num_allocatable_reqs`的正确写法变成了从`get_parallel().pp_max_micro_batch_size`读取，而不再是用`get_server_args().pp_max_micro_batch_size`。结果就是，写入操作虽然正常返回，但其实没生效，scheduler因而读到`None`，并在处理第一个batch时因计算`None - int`而终止。
>
> 上一次复盘的结论是「Omni依赖的是SGLang的具体行为，而接口并没有把这些行为定义下来」。这次升级遇到了一个更加需要仔细处理的问题，那就是接口完全未变，行为却已经改变。

## 三、执行顺序

### 第0步：比对依赖的metadata，确定是否需要重新构建镜像

这一步只需约十分钟，也不依赖后续的适配工作，却会影响整个升级的排期。

```bash
pip download --no-deps sglang==<旧版本> -d /tmp/sgl-old
pip download --no-deps sglang==<新版本> -d /tmp/sgl-new
# 解包后比对两侧的 METADATA / requires_dist
```

需要重点关注flashinfer与torch这两个包。`.github/scripts/validate_omni_env_reusable.sh`要求从镜像的site-packages加载，而不能从venv加载。这是为了复用镜像中预先生成的JIT cache。若新版`sglang`的依赖要求更高版本的flashinfer（有可能只是解析出来的结果），安装项目时会在venv中另行安装新版flashinfer，该检查随即失败。这个版本要求来自`sglang`包自身，与`pyproject.toml`的写法无关，无法在PR内解决，只能执行以下两项操作：

1. 重新构建CI镜像。`docker/Dockerfile`中固定了flashinfer版本，并`COPY`了`/root/.cache/flashinfer/<版本>`。
2. 更新digest。涉及所有工作流文件与Dockerfile。

`/docker`与`.github`均配置了CODEOWNERS，因此镜像重建需要等待相关负责人安排。

- 目标SGLang版本会直接决定基础镜像、flashinfer版本与JIT cache版本。这三项在开始编写适配代码前就已经确定，后续适配也不会改变镜像内容。
- 本地升级与验证不依赖镜像。重建镜像是为了通过CI的环境复用检查并复用JIT cache，并不影响代码能否在本地运行；在本地安装新版`sglang`及其依赖即可。
- 本地适配和验证与镜像重建位于不同的关键路径上，应当并行推进。如果等适配完成后再重建镜像，就会额外增加一段不可控的等待时间，延长总工期。


> 0.5.17：比对结果如下。
>
> | 依赖 | 0.5.16 | 0.5.17 |
> |---|---|---|
> | `helion` | `==0.2.6` | `==1.4` |
> | `flashinfer_python[cu13]` | `==0.6.14` | `==0.6.15.post1` |
> | `sgl-deep-gemm` | `==0.1.4.post1` | `==0.1.5.post1` |
> | `av`（Linux ARM） | 未固定 | `==16.1.0` |
> | `xxhash` | 无 | 新增 |
>
> 其中，flashinfer的版本变化直接导致CI无法通过，失败信息为`flashinfer must come from the image, not /data/omni-ci/pr-1477/omni`与`Torch and FlashInfer must use the image installation for JIT cache reuse`。digest更新涉及22处修改。镜像重建请求直到升级开始后的第四天才提出，CI自首次运行起便持续失败。

### 第1步：阅读changelog，确定变化范围

首先阅读Breaking Changes与Dependencies两节，预估架构层面的工作量。

同时要明确changelog的范围。它通常不会完整记录私有接口签名的变化，而Omni依赖的大多正是私有接口。即使列出了某项变更，changelog往往也只说明具体变更，不说明旧代码的具体失效方式。因此，changelog只能指出排查方向，不能代替逐项检查。

因此最好还是直接比对代码，具体步骤是解包新旧两个wheel，只对项目实际导入的模块执行diff。

> 0.5.17：changelog包含600余条记录、20余个分类，而本次升级需要完成的8项改动中，changelog只覆盖了2项。
>
> | 改动 | changelog是否覆盖 |
> |---|---|
> | `ServerArgs`只读化，配置迁移至config bag | 覆盖，见Breaking Changes |
> | `sglang.jit_kernel`并入`sglang.kernels` | 覆盖，见Kernel Library |
> | `SamplingBatchInfo`的`vocab_mask` / `apply_mask_func`合并进`grammar_mask` | 未覆盖 |
> | `SchedulerLogprobResultProcessor`移除`server_args` | 未覆盖 |
> | `SchedulerDPAttnAdapter`新增必填的`model_runner` | 未覆盖 |
> | `SchedulerLoadInquirer`新增三个必填的遥测accessor | 未覆盖 |
> | `pp_max_micro_batch_size`的读取位置迁移至`get_parallel()` | 未覆盖 |
> | token clamp读取`dcp_size` | 未覆盖 |
>
> 关于配置拆分，changelog的原文是「Code that mutated ServerArgs at runtime must route through the new accessors」。这句话本身准确，却没有说明旧写法仍会正常返回，这也正是我们踩坑的地方。

### 第2步：建立性能基线

两个版本的基线在同一台机器上、用同一种方法分别采集，每次测量前都要预热并重复执行。核心是：任何代码路径的首次执行结果都不能作为有效测量。因为会有各种原因导致预热问题。

两次运行之间执行`.github/scripts/delete_gpu_process.sh`，CI工作流也采用同样的做法。前一次运行留下的显存占用，是造成假的回退的第二大原因，仅次于cache未预热。

此外，两个版本都必须使用各自pin对应的依赖。因为性能测试实际比较的是两套完整的依赖，而不只是一个包的两个版本。

评测工具位于`benchmarks/eval/`。CI检查的指标也正是PR描述需要呈现的指标，因此从一开始就应按同一口径采集。

> 0.5.17：本次共排查四例疑似性能回退，但都是虚惊一场。
>
> | 观察结果 | 实际成因 |
> |---|---|
> | Qwen3-TTS吞吐下降 | 需执行至第四次才进入稳态 |
> | TTS stage-2 TTFC p95为0.5838 | 重复执行后回落至0.506–0.529的基线区间 |
> | `ws_stream`延迟p95为13.82s，超出阈值 | 两次重复执行均在阈值内 |
> | MMMU 0.959 qps、16.05s延迟 | inductor cache未预热，清理后连续三次通过 |
>
> 两个版本使用的依赖分别是：`0.5.16`对应flashinfer 0.6.14、helion 0.2.6、sgl-deep-gemm 0.1.4.post1，`0.5.17`对应0.6.15.post1、1.4、0.1.5.post1。

### 第3步：更新pin，执行静态扫描

安装新版本后，用类型检查器根据新版的内部实现检查现有代码。A类变更可以在这一步集中修复，成本很低。

虽然项目目前没有配置类型检查器，但使用mypy或者带typecheck的LSP，都能帮助快速发现问题，尽管它们各自有着不同的噪音。我个人体验是`pyrefly`的信噪比显著优于`ty`，而二者速度比更主流的`pyright`快不少。但`ty`产生的噪声足以淹没有效的诊断信息。

**`try/except ImportError`会掩盖导入失败。** 导入失败后，这类分支会回退到备用实现，程序也会在没有提示的情况下改用性能较低的路径，行为是既不抛出异常，也不产生日志，仅表现为性能下降，而且没有任何诊断线索。因此，非常推荐通过静态搜索找出已经失效的导入路径，包括`try/except ImportError`内部的引用，不能等到运行时才发现。

> **提示：** grep虽然能找到匹配文本，但也会匹配到注释和字符串，而且它无法识别「导入位于捕获`ImportError`的`try`块内」这种语法结构。ast-grep根据语法树匹配，可以同时避开这两个问题。若要准确查找对某个已失效模块的导入：
>
> ```bash
> ast-grep --lang python --pattern 'from sglang.jit_kernel.$$$A import $$$B'
> ```
>
> 若要列出所有带有捕获`ImportError`或`ModuleNotFoundError`的`except`子句的`try`块，供升级前逐条确认，可执行以下命令：
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
> 该写法同时覆盖`from X import Y`与`import X`，也能匹配`except ImportError as exc:`、`except (ImportError, OSError):`这类元组写法以及`ModuleNotFoundError`。直接用`try: ... except ImportError: ...`这样的pattern会漏掉以上所有情形：截至2026-10-21的main，在`sglang_omni/`下它只能找到38处，上面的规则能找到70处。`stopBy: block`让搜索不进入异常处理分支的函数体，因此无关分支里的`raise ImportError(...)`不会被误匹配。需要逐条确认两件事：该导入在新版本中是否仍能成功；若不能，回退逻辑是否仍符合预期。

同时，建议将每项修复各自提交一个commit，并在commit标题中说明原因。这样既能直接为PR描述准备材料，也能把A类修复与后续的B类修复分开。

> 0.5.17：这一步共提交了三个commit。
>
> ```
> fix(qwen3-omni): drop the SamplingBatchInfo grammar-mask kwargs
> fix(moss-tts-local): import flash_attn_varlen_func from kernels.ops
> fix(scheduler): adapt to the 0.5.17 scheduler-component contracts
> ```
>
> 其中第二个commit对应上述`try/except ImportError`情形：MOSS-TTS-Local的vocoder会在该异常处理分支中回退到SDPA；由于`sglang.jit_kernel`在`0.5.17`中已经退役，这项import必然失败。

### 第4步：启动CI覆盖的全部模型

CI的模型矩阵就是必测清单，也界定了项目实际承诺支持的范围。

| workflow | 模型 | 检查项 |
|---|---|---|
| `test-asr-ci.yaml` | MOSS-Transcribe-Diarize；Fun-ASR或Qwen3-ASR（二选一） | WER、RTF、吞吐 |
| `test-tts-ci.yaml` | Higgs或MOSS-TTS-Local（二选一），5个stage | WER、SIM、TTFC、延迟、流式一致性、router DP2压测 |
| `test-qwen3-omni-ci.yaml` | Qwen3-Omni，11个stage | thinker长度、TTS WER与SIM、MMMU与MMSU准确率及速度、talker、video |
| `omni-ci.yaml` → PR Test | — | 全部单元测试 |

上表为2026年8月的CI矩阵。此后TTS、ASR、Omni三个stage都陆续加入了新模型（例如Qwen3-TTS、Whisper、CosyVoice3、MiniCPM-o），CI label也拆分成了按stage与按模型两类。实际的必测清单，以升级时`.github/workflows/`与PR模板中CI一节的内容为准。

静态扫描通过，并不说明代码一定能够运行，B类变更通常只有到这一步才会暴露。

需要在两个版本之间反复切换时，常见做法是把另一版`sglang`解包到单独的目录，再通过`PYTHONPATH`将该目录放到`sys.path`最前面，使`import sglang`加载这个版本，而不是已经安装的版本。这种做法通常称为shadow。切换时只需修改一个环境变量，无须反复重装整套依赖。

shadow有两项限制。第一，它只替换`sglang`本身，其余依赖仍来自当前安装环境，因此组合出的依赖既不完全对应旧版，也不完全对应新版。它适合快速确认代码能否正常运行，却不适合生成第2步所需的对比数据；这些数据必须在两个版本各自完整的依赖下采集。

第二，shadow只有在worker继承父进程环境时才会生效。CI测试通过`start_server_from_cmd`（`benchmarks/benchmarker/utils.py`）启动worker。会先复制`os.environ`，再叠加调用方传入的`env`。未传入`PYTHONPATH`的测试会保留父进程的shadow设置；如果在`process_env`中固定了`PYTHONPATH`，该设置就会被覆盖，worker实际运行的是已经安装的版本。此时两个版本使用的是同一份代码，测试却会错误地得出「无回退」的结论。

开始对比前，应找到传给`launch_managed_router`的`process_env`，逐个确认测试属于哪一类。后一类测试必须分别安装对应版本后再执行。截至2026-10-21的main，`test_tts_serving_ci.py`属于后一类：它在`process_env`中将`PYTHONPATH`固定为项目根目录（`tests/test_model/test_tts_serving_ci.py:297`），其benchmark子进程也采用同样的设置（同文件`:379`）。

必须如实记录实际执行了哪些测试，不能用一张全部通过的结果表暗示已经完整覆盖。

> 0.5.17：两个B类变更都在这一步暴露，修复这两个问题的commit也最耗时。
>
> ```
> fix(scheduler): route the pp_max_micro_batch_size default through the context
> fix(scheduler): mirror the 0.5.17 step counters and batch launch timestamp
> ```
>
> 本次实际执行了17个GPU测试中的12个，TTS stage 3–5与四个Qwen3-Omni video job未执行，该情况已写入PR。

### 第5步：排查回退时，先复现再定位

第2步的约束在这里同样适用：任何性能下降，在干净的环境里复现之前，都不能判定为回退。复现通常只需几分钟，定位根因却可能需要几天，因此必须先复现，再定位。

确认回退后，可以在运行中的server上为兼容层加入临时日志，记录每个调用点当时实际读取/写入了哪些值。与静态检查相比，这些日志反映的是运行时的真实行为，而不是推测。

但日志只能覆盖本次运行实际触发的调用点，未触发的部分仍然无法判断。结论必须来自实测结果，并明确列出哪些路径没有覆盖；夸大影响范围会误导后续排查。

> 0.5.17：全仓共有23个`override_server_args`调用，涉及13个字段。在一台正在运行的Qwen3-TTS server上为该helper加入临时日志后，实际触发的三个调用点都得到了明确结论，且均未受影响；真正失效的只有`pp_max_micro_batch_size`。其余调用方位于本次运行未触发的模型路径中，PR已将其明确标为未审计。
>
> 基于这次实测，PR最终将结论限定为：通过`ServerArgs.override`写入后再从`ServerArgs`读回的路径仍然有效。如果写入后，某个config bag的使用方要读取该值，那么这条路径已经失效。前面这种表述可以直接当作结论，在这个场景里，我们不可以简单地说「这些override已全部失效」。

### 第6步：pre-commit与全部单元测试通过后提交PR

`pre-commit`在本地执行autoflake、isort、black、ruff；CI里的`lint` job也执行同一组检查。

单元测试中用于替代SGLang组件的fake对象（`tests/unit_test/fakes.py`）固定了上游接口的结构。上游代码每多读取一个字段，这些fake对象就必须补上该字段；否则，测试要么直接报错，要么在已经偏离真实行为的情况下继续通过，后一种情况更加危险。因此，更新这些fake对象本身就是适配工作的一部分，而不是最后的收尾工作。

将任何失败归因于本次升级前，都必须先在旧版本上确认能否复现。验证通常只需执行一条命令，而一旦误判，可能浪费数小时进行排查。

PR描述必须包含版本差异表、每项适配的一行说明、准确率与性能对照，以及对未执行项目的明确说明。

> 0.5.17：9个实质性commit中有3个仅涉及测试。
>
> ```
> test(scheduler): give the scheduler doubles a dcp_size-bearing server_args
> test(scheduler): model the 0.5.17 runtime-context contract in the doubles
> test: adapt two merged-in suites to the 0.5.17 contract
> ```
>
> 唯一失败的单元测试`test_mp_runner_startup_failure_includes_child_factory_traceback`在`0.5.16`上也出现了完全相同的失败：该测试为启动预留10s，而该机器上冷启动`import sglang_omni.pipeline.stage_workers`需要18.6s。问题来自这台机器的性能，而不是版本回退。

### 第7步：合入后，单独提PR重新calibrate CI阈值

CI中的每个gate都对应一个写在代码里的参考值（reference），也就是在CI所用的H100机器上多次干净运行得到的最差值；阈值在import时由参考值乘以固定的slack得出。升级backbone后，吞吐、延迟、RTF会整体移动，准确率、WER、speaker similarity也可能小幅变化，但参考值描述的仍然是旧镜像。阈值过松，就拦不住之后真正的回退；阈值过紧，则会让无关PR的CI随机失败。因此，升级PR合入后必须对CI覆盖的所有模型重新calibrate一次，速度类与质量类的参考值都要更新，[官方升级文档](https://github.com/sgl-project/sglang-omni/blob/main/docs/developer_reference/bump_version.md#the-pr)也有同样的要求。

截至2026-10-21的main，参考值与slack的位置如下：

| 文件 | 覆盖范围 | 速度slack（越高越好 / 越低越好） |
|---|---|---|
| `tests/test_model/tts_ci_config.py` | TTS各preset | 0.75 / 1.25 |
| `tests/test_model/test_tts_serving_ci.py` | TTS serving | 沿用TTS |
| `tests/test_model/omni_ci_config.py` | Qwen3-Omni、MiniCPM-o | 0.875 / 1.125 |
| `tests/test_model/asr_ci_config.py` | Fun-ASR、Qwen3-ASR、Whisper | 0.9 / 1.1 |
| `tests/test_model/test_asr_ci_multi_speaker.py` | MOSS-Transcribe-Diarize | 0.9 / 1.1（AISHELL4-long与GoogleTime为0.8 / 1.2） |
| `tests/test_ci/tts_mps_ci_config.py` | TTS MPS | 0.75 / 1.25 |

速度类参考值按指标方向乘以上表的slack（`tests/utils/__init__.py`中的`apply_slack`）；WER默认乘以1.25（`apply_wer_slack`，ASR传入的是1.1）；UTMOS乘以0.97（`apply_mos_slack`）；准确率与TTS的speaker similarity不加slack，直接与参考值比较。calibration只改参考值，不改slack。

calibration工具是维护者私有仓库中的`calibrate-h100-ci` skill（`tests/README.md`中有链接），在[PR #2132](https://github.com/sgl-project/sglang-omni/pull/2132)之前以`.claude/skills/tune-ci-thresholds/`的形式放在sglang-omni仓库中。按迁出前（2026-09-12）的版本，流程如下：

1. 先停掉CI机器上的runner，等正在执行的job结束。calibration与CI共用同一台机器，CI任务会带来额外的CPU、GPU与带宽争用。最后一轮测量结束后立即恢复CI，不要等报告写完。
2. 每组GPU绑定到其NUMA本地的CPU核（`OMNI_CI_CPUSET`，由`tests/test_model/conftest.py`通过`sched_setaffinity`生效）。写入阈值所用的测量固定在同一组GPU上进行，其余组只做测量。
3. 执行precheck：检查依赖版本、GPU显存是否已释放、资源文件，并跑一次CUDA smoke test，同时记录镜像digest、驱动、torch与sglang版本等环境指纹。FlashInfer JIT cache、UTMOS与speaker similarity模型都要提前预热，否则冷启动会产生无效的轮次。
4. 每个stage重复运行5轮，取所有干净轮次中的最差值：越高越好的指标取最小值，越低越好的取最大值。不取百分位，也不去掉极值。
5. 明显偏离的轮次判为destructive（MAD z-score大于3.5，且与最近的值相差20%以上），整轮作废并补跑两轮，单个stage最多15轮。被CPU争用打断的尝试同样作废重跑。
6. 工具为每个参考值生成before / after对照，由人审阅后再写回上表中的文件。写回后重新生成一次计划，确认所有参考值都已一致。

calibration PR只修改参考值。PR描述中写明测量所用的commit、镜像digest、sglang与torch版本、GPU与cpuset、每个参考值在各轮的结果与变化量，以及刻意没有更新的参考值及原因。合入后，用对应模型的label重跑一次CI。

需要注意以下几点：

- calibration必须在升级PR合入之后、基于合入后的commit进行，并单独提PR。在升级分支上calibrate，测到的是main并未运行的镜像，阈值改动也会混进升级PR的A/B审阅。反过来，升级时如果遇到一个main本身也过不了的gate，也不应在升级PR中调整，它属于单独的calibration PR。
- 运行pytest时不要加`-x`。在旧阈值下失败的assert仍是有效的观测，加了`-x`，后续指标就无法采集。
- 一轮异常值就能决定最终的gate。第2步关于预热与清理显存的要求，在这里同样适用。
- calibration与第2步的基线不是一回事：第2步的基线用于判断升级本身是否引入了回退，calibration则是在升级已被接受之后，让CI阈值跟上新的依赖栈。

> 0.5.16：合入后的calibration（[PR #1260](https://github.com/sgl-project/sglang-omni/pull/1260)）第一遍直接取5轮中的最差值，结果一轮受争用或冷启动影响的数据就决定了gate：MMSU吞吐的参考值被压到12.1 req/s（干净轮次为54.9），VideoAMME延迟被拉到32.9s（干净轮次为8.7s）。这样的gate连真实的4倍回退都拦不住。加入destructive轮次剔除后重新计算，结果从117项放宽、35项收紧，变为70项收紧、12项放宽、89项不变。
>
> 0.5.18：合入后的calibration（[PR #1811](https://github.com/sgl-project/sglang-omni/pull/1811)）以收紧为主，例如MMSU吞吐的参考值从54.9提高到78.8 req/s。

## 四、应对持续变化的main分支

升级期间，main分支仍在持续变化，需要适配的范围就可能继续扩大。因此有两条原则。第一，升级PR应尽量缩小改动范围（make minimal changes），不要附带额外重构，以减少冲突。第二，应尽快合入；分支存在的时间越长，维护成本就越高，不过这一点也得看整个项目roadmap安排。

每次merge main后，建议重新检查性能是否回退，以及是否出现了同类兼容问题：

> 0.5.17：该升级分支在四天内三次从main合入变更，其中一个commit只用于重新适配合入的测试。在CI阻塞期间，main又合入了两项相关变更：一项将`models/moss_tts/vocoder_decoder.py`并入`audio_tokenizer.py`，删除了本PR修改过的原文件，并一并迁移其中的`jit_kernel`导入；另一项在确定性推理特性中新增了两个`override_server_args`调用点。

## 五、版本判断不能只看接口是否存在

兼容层常通过探测某个接口是否存在来判断SGLang的版本。但当上游通过「保留接口但使其不再生效」来废弃API时，只凭「函数是否存在」判断，会在不报错的情况下选中已经失效的旧分支，使新版本路径不生效。实际测试时应依据版本号，或者检查可以实际观察到的结果，比如写入一个值，再通过使用方实际调用的accessor将其读回，而不能只看方法是否仍然存在。

修好某一处兼容层，也无法防止上游下次以同样方式废弃其他接口。这类改动既不会抛出异常，也无法被静态工具发现，属于本文开头所说的B类，只有在第4步实际运行模型时才会暴露。

> 0.5.17：`sglang_omni/vendor/sglang/server_args.py`中的`override_server_args`专门用于把不同版本的处理逻辑集中在一处，当时按以下方式选择具体实现：
>
> ```python
> legacy_override = getattr(server_args, "override", None)
> if callable(legacy_override):
>     legacy_override(source, **fields)
>     return
> # get_context().override(...) / declare_late_resolution(...)
> ```
>
> `ServerArgs.override`在`0.5.17`中依然存在，因此执行这段代码时，每次都会进入第一个分支，后面为新版本准备的两条路径永远不会执行。升级PR的修法是让scheduler绕过这段代码，直接调用`get_context().override(...)`。shim本身后来在[PR #1719](https://github.com/sgl-project/sglang-omni/pull/1719)中修复：不再探测`ServerArgs.override`是否存在，而是判断传入的`server_args`是否就是已经发布到runtime context的那份记录，再决定通过`get_context().override(...)`还是`declare_resolution(...)`写入。

## 六、检查清单

- [ ] 比对依赖元数据；如需重建镜像，第0天即向CODEOWNERS提出镜像重建请求
- [ ] 阅读changelog的Breaking Changes与Dependencies；私有API得另行比对
- [ ] 在目标机器上采集基线，包含warmup与重复测量
- [ ] 更新pin；静态扫描类型问题
- [ ] 静态检查已废弃的import路径，包含`try/except ImportError`内部的引用
- [ ] CI覆盖的每个模型均可启动并正常服务
- [ ] 每个疑似回退都在干净的GPU上复现，之后再开始归因
- [ ] 每个失败都先在旧版本上验证能否复现，再归因
- [ ] 单元测试的fake对象已按新接口更新
- [ ] `pre-commit`通过，全部单元测试通过
- [ ] PR中注明未执行项
- [ ] 合入后，基于合入的commit单独提PR，重新calibrate所有模型的CI阈值
