# SGLang backbone 升级实操指南

英文版本见 [A Practical Guide to Upgrading the SGLang Backbone](./version_bump_guide.md)。

SGLang Omni 并非将 SGLang 作为库使用，而是作为框架使用：约 512 个 Python 文件中有 68 个直接导入 `sglang.srt.*` 之下的内部模块。上游将这些模块视为内部实现，可以不经废弃周期直接变更。因此每次 backbone 版本升级在形式上是一次依赖更新，实质上是一次针对私有接口变更的移植工作。

> **实例来源：** 全文的实例均取自 `0.5.16 → 0.5.17`（[PR #1477](https://github.com/sgl-project/sglang-omni/pull/1477)），该次升级改动 10 个文件，新增 80 行、删除 43 行。其规模远小于上一次涉及 162 个文件的升级，但处理耗时最长的两个问题同样出现在类型检查器无法检测的环节。上一次 `0.5.12.post1 → 0.5.16` 升级的完整复盘见 [API 对齐到浮点结合律：SGLang Omni 的 backbone 升级](./version_bump_zh.md)，本文不重复其中的个案分析。

## 一、升级影响面的实际规模

依赖量最大的是 `layers`、`managers`（scheduler 所在）与 `model_executor` 三个子系统，分布如下：

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

上述计数覆盖 `sglang_omni/` 与 `sglang_omni_router/` 两个包。除 `sglang.kernels` 外，其余全部位于 `sglang.srt.*` 之下，不应对其中任何 API 抱有稳定性预期。

这张表应在每次升级开始时重新统计，依赖面随功能开发持续扩大，上一次的结论不能直接沿用。它的直接用途是界定审查范围：上游的发布说明按其自身的模块划分组织，而这张表决定其中哪些条目可能到达 Omni，第 1 步的阅读即以此为筛选依据。

`sglang_omni/vendor/sglang/` 是存放版本条件代码的指定位置。所有需要按 SGLang 版本分叉的逻辑都应收敛在该目录内，而非分散于各调用点。该层自身同样可能失效，详见[第五节](#五版本判断不能只看接口是否存在)。

## 二、两类失效及其对流程的约束

**A 类：符号发生变更。** 符号被移除、重命名，或签名改变。此类失效必然抛出异常：`ImportError`、`TypeError: unexpected keyword argument`、`TypeError: missing required argument`。在执行任何代码之前，类型检查器即可全部识别。

**B 类：符号保留，语义变更。** 调用仍可解析，仍接受相同参数，仍正常返回，但不再产生原有效果。此类失效不抛出任何异常，静态工具无法观察，只有在真实模型上执行真实请求时才会暴露。

两类失效的成本相差一个量级，这决定了后续步骤的排序依据：**静态扫描应尽早执行，因其成本极低；工期估算则应以 B 类为准，因其成本占主导。**

> **0.5.17 实例：** 失效分布为 6 个 A 类、2 个 B 类。6 个 A 类在半天内修复完毕，2 个 B 类占用了其余工时。
>
> B 类的典型案例是配置拆分：`0.5.17` 将 `ServerArgs` 改为只读的启动记录，解析后的配置迁移至一组按域划分的命名空间对象，SGLang 源码称其为 config bag（类名 `_ConfigBag`），分别经 `get_exec()`、`get_parallel()` 等访问器读取。`ServerArgs.override()` 的签名未作任何改动，调用仍然正常返回，但不再写穿至 config bag。与此同时，`get_num_allocatable_reqs` 的读取位置由 `get_server_args().pp_max_micro_batch_size` 迁移至 `get_parallel().pp_max_micro_batch_size`。最终结果是：写入调用报告成功，读取方改从另一处取值，scheduler 读取到 `None`，并在处理第一个 batch 时因 `None - int` 终止。
>
> 上一次复盘的结论是「Omni 依赖的是 SGLang 的具体行为，而接口并没有把这些行为定义下来」。本次是该结论的极端情形：接口的字面形式完全未变，仅行为发生了变更。

## 三、执行顺序

### 第 0 步：比对依赖 metadata，确定镜像重建是否需要立即排期

该步骤耗时约十分钟，不依赖任何适配工作，但决定整个升级的排期方式。

```bash
pip download --no-deps sglang==<旧版本> -d /tmp/sgl-old
pip download --no-deps sglang==<新版本> -d /tmp/sgl-new
# 解包后比对两侧的 METADATA / requires_dist
```

需要重点关注 flashinfer 与 torch。`.github/scripts/validate_omni_env_reusable.sh` 要求二者解析至镜像的 site-packages 而非 venv，以保证镜像内预先生成的 JIT cache 有效。若新版 `sglang` 的传递依赖要求更高版本的 flashinfer，安装项目时就会在 venv 内产生一份副本，该检查随即失败。这类版本要求来自 `sglang` 包自身，与 `pyproject.toml` 的写法无关，**无法在 PR 内解决**，所需操作为：

1. 重新构建 CI 镜像。`docker/Dockerfile` 中固定了 flashinfer 版本，并 `COPY` 了 `/root/.cache/flashinfer/<版本>`。
2. 更新 digest。涉及六个 workflow 文件与 Dockerfile。

`/docker` 与 `.github` 均配置了 CODEOWNERS，因此镜像重建需要排队由相关负责人处理。**该请求应在第 0 天提出，不应等待适配完成。** 理由有三：

- 镜像的构成完全由目标 SGLang 版本决定，具体包括基础镜像、flashinfer 版本与 JIT cache 版本三项，在开始编写适配代码前即已确定。适配过程中的任何发现都不会改变镜像的内容，因此推迟提出请求不产生任何信息增益。
- 本地升级与验证不依赖镜像。镜像用于满足 CI 环境复用检查的要求，并支持 JIT cache 复用，与能否运行无关，本地安装新版 sglang 并由其拉取自身依赖即可。
- 本地适配验证与镜像重建因此位于不同的关键路径上，应当并行。将镜像重建串行安排在适配之后，等于把不受自身控制的排队时间叠加到工期上。

提前提出请求的唯一风险是适配最终不可行、镜像重建失去必要性。应对方式是在提出请求时约定一个撤销期限，而非推迟提出。若某些 CI job 受限于本地 GPU 数量无法在本地复现，镜像重建则直接位于关键路径上，更应尽早提出请求。

> **0.5.17 实例：** 比对结果如下。
>
> | 依赖 | 0.5.16 | 0.5.17 |
> |---|---|---|
> | `helion` | `==0.2.6` | `==1.4` |
> | `flashinfer_python[cu13]` | `==0.6.14` | `==0.6.15.post1` |
> | `sgl-deep-gemm` | `==0.1.4.post1` | `==0.1.5.post1` |
> | `av`（Linux ARM） | 未固定 | `==16.1.0` |
> | `xxhash` | 无 | 新增 |
>
> 其中 flashinfer 一项直接阻塞了 CI，失败信息为 `flashinfer must come from the image, not /data/omni-ci/pr-1477/omni` 与 `Torch and FlashInfer must use the image installation for JIT cache reuse`。digest 更新涉及 22 处修改。镜像重建请求在升级开始后第四天才提出，CI 自首次运行起持续失败。

### 第 1 步：阅读 changelog，并确认其覆盖边界

首先阅读 **Breaking Changes** 与 **Dependencies** 两节，二者界定架构级的工作量。

其次校准预期。changelog 面向的是公开产品，私有接口签名的变动在结构上不属于其覆盖范围，而 Omni 依赖的多数是私有签名。覆盖内容的详略程度同样有差异：即便是已覆盖的条目，changelog 给出的通常也只是结论而非失效方式。changelog 的作用是指示排查方向，而非枚举失效点。

因此需要补充一次机械比对：解包两个 wheel，仅针对项目实际导入的模块执行 diff。

> **0.5.17 实例：** changelog 包含 600 余条记录、20 余个分类，而本次升级所需的 8 项改动中，changelog 覆盖了其中 2 项。
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
> 其中配置拆分一条的提示为「Code that mutated ServerArgs at runtime must route through the new accessors」，该提示准确无误，但并未说明旧写法会继续正常返回，而后者才是实际的失效方式。

### 第 2 步：建立性能基线，且每项测量均需 warmup

基线须在同一台机器、以同一方法、对两个版本分别采集，每次测量均包含 warmup 与重复测量。核心约束是：**任一代码路径在任一版本上的首次执行结果均不构成有效测量。**

两次运行之间须执行 `.github/scripts/delete_gpu_process.sh`，CI workflow 亦采用同样做法。前次运行残留的显存是伪回退的第二大来源，仅次于未预热的 cache。

此外，两个版本须使用各自 pin 对应的依赖栈。比较对象因此是两套完整依赖栈，而非单个包的两个版本，结论中须明确说明这一点。

测量 harness 位于 `benchmarks/eval/`。CI 断言所用指标与 PR 描述所需指标一致，应从一开始即按该形式采集。

> **0.5.17 实例：** 共排查四例疑似性能回退，全部未能复现。
>
> | 观察结果 | 实际成因 |
> |---|---|
> | Qwen3-TTS 吞吐下降 | 需执行至第四次才进入稳态 |
> | TTS stage-2 TTFC p95 为 0.5838 | 重复执行后回落至 0.506–0.529 的基线区间 |
> | `ws_stream` 延迟 p95 为 13.82s，超出阈值 | 两次重复执行均在阈值内 |
> | MMMU 0.959 qps、16.05s 延迟 | inductor cache 未预热，清理后连续三次通过 |
>
> 两个版本的依赖栈分别为：`0.5.16` 对应 flashinfer 0.6.14、helion 0.2.6、sgl-deep-gemm 0.1.4.post1，`0.5.17` 对应 0.6.15.post1、1.4、0.1.5.post1。

### 第 3 步：更新 pin，执行静态扫描

安装新版本后，由类型检查器以新的内部实现校验现有调用点。A 类失效可在此步骤内集中修复，成本极低。

项目自身未配置类型检查器，但仍建议临时执行一次。在本代码库中，`pyrefly` 的信噪比显著优于 `ty`，后者产生的噪声足以掩盖有效诊断信息。

**`try/except ImportError` 会掩盖导入失败。** 此类分支在导入失败时回退至备用实现，模型将静默切换至性能更低的路径，既不抛出异常也不产生日志，仅表现为性能下降且缺乏诊断线索。因此，已退役的导入路径须直接静态定位，且须覆盖 `try/except ImportError` 内部的引用，不应等待其在运行时暴露。

> **提示：** grep 可以定位，但它会命中注释与字符串，也无法表达「导入嵌在捕获 `ImportError` 的 `try` 内」这一结构关系。ast-grep 按语法树匹配，两者都能解决。定向查找某个已退役模块的真实导入：
>
> ```bash
> ast-grep --lang python --pattern 'from sglang.jit_kernel.$$$A import $$$B'
> ```
>
> 普查全部被 `ImportError` 保护的导入，得到一份升级前逐条确认的清单（本文写作时在 `sglang_omni/` 下命中 17 处）：
>
> ```bash
> ast-grep --lang python --pattern 'try:
>     $$$BODY
> except ImportError:
>     $$$H' sglang_omni/
> ```
>
> 该写法同时覆盖 `from X import Y` 与 `import X`，并排除捕获其他异常的 `try`。逐条确认两件事：该导入在新版本上是否仍然成立；若不成立，其 fallback 是否仍是预期行为。

建议每个修复单独形成一个 commit，并在 commit 标题中说明原因。这一做法既为 PR 描述提供现成材料，也便于将 A 类修复与后续的 B 类修复区分开。

> **0.5.17 实例：** 该步骤形成三个 commit。
>
> ```
> fix(qwen3-omni): drop the SamplingBatchInfo grammar-mask kwargs
> fix(moss-tts-local): import flash_attn_varlen_func from kernels.ops
> fix(scheduler): adapt to the 0.5.17 scheduler-component contracts
> ```
>
> 其中第二个 commit 对应上述 `try/except ImportError` 情形：MOSS-TTS-Local 的 vocoder 在该异常处理分支中回退至 SDPA，而 `sglang.jit_kernel` 在 `0.5.17` 上已退役，该导入必然失败。

### 第 4 步：启动 CI 覆盖的全部模型

CI 的模型矩阵即必测清单，它是项目实际强制执行的「支持范围」定义。

| workflow | 模型 | 检查项 |
|---|---|---|
| `test-asr-ci.yaml` | MOSS-Transcribe-Diarize；Fun-ASR 或 Qwen3-ASR（可选） | WER、RTF、吞吐 |
| `test-tts-ci.yaml` | Higgs 或 MOSS-TTS-Local（可选），5 个 stage | WER、SIM、TTFC、延迟、流式一致性、router DP2 压测 |
| `test-qwen3-omni-ci.yaml` | Qwen3-Omni，11 个 stage | thinker 长度、TTS WER 与 SIM、MMMU 与 MMSU 准确率及速度、talker、video |
| `omni-ci.yaml` → PR Test | — | 全部单元测试 |

静态扫描通过并不构成可运行性证明，B 类失效通常仅在此步骤暴露。

在两个版本之间反复切换时，一种常见做法是把另一版 sglang 解包到某个目录，再用 `PYTHONPATH` 把该目录插到 `sys.path` 前面，使 `import sglang` 解析到它而非已安装的那份，通常直接称为 shadow。它的好处是切换只需改一个环境变量，无需每次重装整套依赖。

shadow 有两处限制。第一，它只替换 sglang 本身，其余依赖仍是已安装的版本，得到的是一套两边都不完全符合的混合栈；因此它适合快速确认代码能否跑通，不适合用来产出第 2 步要求的对比数字，那些数字须在各自完整的依赖栈下采集。

**第二，shadow 只在 worker 继承父进程环境时生效。** CI 测试经 `start_server_from_cmd`（`benchmarks/benchmarker/utils.py`）启动 worker，该函数以 `os.environ.copy()` 为基底，再用调用方传入的 `env` 覆盖。未传 `PYTHONPATH` 的测试会继承 shadow；在 `process_env` 中固定 `PYTHONPATH` 的则会将其覆盖，worker 实际运行已安装版本。后一种情况下两侧运行的是同一份代码，并将据此得出「无回退」的错误结论。

开始对比前须逐个确认所用测试属于哪一类，方法是沿 fixture 追到 `launch_managed_router` 的 `process_env` 实参。属于后一类的测试须分别实际安装对应版本后执行。截至当前 main，`test_tts_serving_ci.py` 属于后一类，它在 `process_env` 中将 `PYTHONPATH` 固定为项目根目录（`tests/test_model/test_tts_serving_ci.py:296`），其 benchmark 子进程同样如此（同文件 `:369`）。

测试覆盖情况须如实记录，避免以全部通过的结果表暗示完整覆盖。

> **0.5.17 实例：** 两个 B 类失效均仅在此步骤暴露，对应两个耗时最长的 commit。
>
> ```
> fix(scheduler): route the pp_max_micro_batch_size default through the context
> fix(scheduler): mirror the 0.5.17 step counters and batch launch timestamp
> ```
>
> 测试覆盖方面，本次执行了 17 个 GPU 测试中的 12 个，TTS stage 3–5 与四个 Qwen3-Omni video job 未执行，该事实已写入 PR。

### 第 5 步：排查回退时，先复现再定位

第 2 步的约束在此适用：**任何性能下降在干净 GPU 上复现之前均不得判定为回退。** 复现耗时以分钟计，根因定位耗时以天计，复现与定位的顺序不应颠倒。

确认回退后，可以在运行中的 server 上给兼容层加一层临时日志，记录每个调用点在调用时刻实际读到和写入的配置状态。相比静态审阅调用点，这样拿到的是真实行为而非推测。

但这样只覆盖该次运行实际经过的调用点，其余的仍然未知。**结论须限定在实测范围内**，并明确写出哪些没有覆盖到。夸大影响范围会把后续读者引向错误的排查方向。

> **0.5.17 实例：** 全仓共 23 个 `override_server_args` 调用点，覆盖 13 个字段。在一台运行中的 Qwen3-TTS server 上给该 helper 加临时日志后，其中三个调用点得到了确切结论，均未受影响；真正失效的是 `pp_max_micro_batch_size` 一处。其余调用点属于该次运行未经过的模型路径，PR 中明确标注为未审计。
>
> 最终采用的表述是：通过 `ServerArgs.override` 写入、随后从 `ServerArgs` 读回，该路径仍然有效；写入后期望某个 config bag 的读取方观察到该值，该路径已失效。这一表述可直接作为行动依据，而「这些 override 已全部失效」不能。

### 第 6 步：pre-commit 与全部单元测试通过后提交 PR

`pre-commit` 在本地执行 autoflake、isort、black、ruff，`lint` job 执行同一组检查，本地通过即可预期 lint job 通过。

单元测试里替代 SGLang 组件的 fake 对象（`tests/unit_test/fakes.py`）写死了上游接口的形状：上游的新代码多读一个字段，这些 fake 就得跟着补上，否则测试要么直接报错，要么在已经不符合真实行为的情况下继续通过。后一种更危险。因此更新它们属于适配工作本身，而非收尾环节。

**任何失败在归因于本次升级之前，须先在旧 pin 上验证是否复现。** 验证通常只需执行一条命令，而误判的代价是数小时的排查。

PR 描述须包含版本差异表、各适配点的一行理由、准确率与性能对照，以及未执行项的明确说明。

> **0.5.17 实例：** 9 个实质性 commit 中有 3 个仅涉及测试。
>
> ```
> test(scheduler): give the scheduler doubles a dcp_size-bearing server_args
> test(scheduler): model the 0.5.17 runtime-context contract in the doubles
> test: adapt two merged-in suites to the 0.5.17 contract
> ```
>
> 唯一失败的单元测试 `test_mp_runner_startup_failure_includes_child_factory_traceback` 在 `0.5.16` 上以完全相同的形式复现：该测试为启动预留 10s，而该机器上冷启动 `import sglang_omni.pipeline.stage_workers` 需要 18.6s。这是机器性能特征，不是回退。

## 四、应对持续变化的 main 分支

升级工作在持续变动的 main 分支上进行，适配面会在验证期间持续扩大。由此得到两条推论。其一，**升级 PR 应保持最小改动范围**，不附带额外重构，以控制冲突面。其二，**应尽快合入**，升级分支的维护成本随存续时间延长而增加。

每次从 main 合入变更后，须逐项执行以下 grep 检查：

```bash
# 已退役的模块路径
git grep -n "sglang\.jit_kernel"

# 契约位置发生迁移的调用点
git grep -n "override_server_args\|get_global_server_args"
git grep -n "SamplingBatchInfo("

# 依赖配置拆分的位置
git grep -n "sglang\.srt\.server_args\|sglang\.srt\.runtime_context"
```

从 main 合入的新代码是针对旧 pin 编写的，会重新引入已修正的旧写法。

> **0.5.17 实例：** 该升级分支在四天内三次从 main 合入变更，其中一个 commit 专用于重新适配随合并引入的测试。在 CI 阻塞期间，main 分支又发生了两项变更：一是删除了本 PR 修改过的文件，`models/moss_tts/vocoder_decoder.py` 被合并进 `audio_tokenizer.py`，其中的 `jit_kernel` 导入一并迁移；二是在确定性推理特性中新增了两个 `override_server_args` 调用点。

## 五、版本判断不能只看接口是否存在

`sglang_omni/vendor/sglang/server_args.py` 中 `override_server_args` 的设计目的正是将版本边界收敛至单点。其分派逻辑如下：

```python
legacy_override = getattr(server_args, "override", None)
if callable(legacy_override):
    legacy_override(source, **fields)
    return
# 新版本路径：get_context().override(...) / declare_late_resolution(...)
```

`ServerArgs.override` 在 `0.5.17` 上依然存在，因此该 shim 始终进入第一个分支，其下为新版本准备的两条路径均为不可达代码。最终 scheduler 绕过该 shim，直接调用 `get_context().override(...)`。

更一般的结论是：**当上游以「保留接口但使其不再生效」的方式废弃 API 时，用「符号是否存在」来判断版本会静默选中不可达路径。** 分派条件应换成版本号，或换成可观察效果——写入一个值，再经由消费方实际使用的 accessor 读回——而不是检查方法还在不在。

截至当前 main，这个 shim 仍未修复：PR #1477 绕过了它，在 `omni_scheduler.py` 中直接调用 `get_context().override(...)`，shim 自身的分派条件没有改动。修复本身是一次性的，且应落在 shim 内，一处改动即可覆盖全部调用点。

但修好它只解决这一处。一个补丁无法预防上游下次以同样方式废弃另一个接口，而这类改动既不抛异常、静态工具也看不到，属于本文开头所说的 B 类，只能靠第 4 步的实跑暴露。

## 六、检查清单

- [ ] 比对依赖 metadata；如需重建镜像，第 0 天即向 CODEOWNERS 提出镜像重建请求
- [ ] 阅读 changelog 的 Breaking Changes 与 Dependencies；私有 API 依赖面另行比对
- [ ] 在目标机器上采集基线，包含 warmup 与重复测量
- [ ] 更新 pin；静态扫描无残留问题（`pyrefly`）
- [ ] 静态检查已退役的导入路径，包含 `try/except ImportError` 内部的引用
- [ ] CI 覆盖的每个模型均可启动并正常服务
- [ ] 每个疑似回退均在干净 GPU 上复现后才进入定位阶段
- [ ] 每个失败均在旧 pin 上验证后才作归因
- [ ] 单元测试的 fake 对象已跟上新接口
- [ ] `pre-commit` 通过，全部单元测试通过
- [ ] PR 中已注明未执行项
