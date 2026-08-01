# 生命周期与框架抽象：SGLang Omni TTS Serving 重构

**一个通用的 Serving 框架，应该怎样管理复杂且差异巨大的语音模型？**

这是我们设计 SGLang Omni 时反复讨论的问题。理想的接入方式应该很简单：**声明流水线拓扑，实现模型特定的计算，然后把调度、通信和生命周期管理全部交给框架。**

但在重构前，TTS（Text-to-Speech）Serving 模块离这个目标还有一段距离。每接入一个新模型，开发者除了要实现文本到 latent、再到波形的生成逻辑，还要在模型目录里额外维护一整套 Serving 机制：从 engine 的启动和调度、跨进程的状态传输，再到 vocoder 的流式生命周期和异常清理。接一个新模型，几乎等于重新搭半套 Serving 栈。

这显然不是一个令人满意的状态。为此，我们专门花费了一个月时间重新划定这条边界：**让模型专注于生成算法，让框架统一接管重复的生命周期。**

## 重构空间与挑战

如同我们在 [TTS 性能优化实战](./tts-optimization-zh.md)中介绍的那样，一个 TTS 模型的推理链路主要如下：

> 参考音频编码 → 自回归生成 audio token → vocoder 解码波形

这条主链路为调度、缓存和请求生命周期提供了复用空间。真正增加重构难度的，是我们此前为了提高推理效率，针对不同模型做了大量贴合其结构的优化。这些优化已经深入 batching、Cache Key 和 streaming state 等 Serving 细节。

比如，**Higgs** 原生不支持流式，我们通过 window 和 crossfade 实现增量输出；**MOSS-TTS-Local** 带有独立的因果 transformer vocoder，需要维护持久化的 codec session、分配 CUDA Graph slot 并跨请求合批；**FishAudio S2-Pro** 的多 codebook 结构则要求 KV Cache 在 Token ID 之外继续识别 embedding 输入。

这些优化直接影响吞吐、延迟和流式体验，也让模型差异进入了 Serving 生命周期。重构需要保留已有的强大性能优势，并且尽量将重复的控制流抽象到框架中。

<div align="center">
  <img src="images/tts-opt-pipeline-overview.svg" alt="SGLang Omni TTS 从预处理、参考音频编码、自回归生成到 vocoder 解码的多阶段流水线" width="78%">
  <p><em>图 1：典型的多阶段 TTS 推理流水线。框架负责统一调度各个标准阶段，而阶段内的具体生成逻辑与计算仍由各模型独立实现。</em></p>
</div>

## 我们完成了哪些抽象？

截至 2026 年 7 月 30 日，这次重构净删除了 **2840 行 non-test 实现代码**。

<div align="center">
  <a href="https://luojiaxuan.github.io/sglang-omni/tts-refactor/">
    <img src="images/tts-refactor-progress-2026-07-30.png" alt="TTS Refactor Progress 页面显示 non-test 实现代码净删除 2840 行" width="96%">
  </a>
  <p><em>图 2：2026 年 7 月 30 日的进展快照。最新统计和逐 commit 明细见 <a href="https://luojiaxuan.github.io/sglang-omni/tts-refactor/">TTS Refactor Progress</a>。</em></p>
</div>

删掉的代码主要集中在以往反复手写的 Serving 机制上：比如状态传输的 `to_dict` / `from_dict`（手写这部分代码极易因漏改而丢失字段）、参考音频的 LRU 淘汰与并发控制，以及流式请求的异常清理。

重构后，框架提供 engine 启动、状态传输、缓存和 vocoder 生命周期的公共骨架；模型目录保留了特有的 codec session、checkpoint 解析和生成逻辑。边界的制定原则也非常严格：**共享代码不能含有对模型名称的特判。所有差异必须通过 Hook、显式字段或 capability metadata 来表达。**

<div align="center">
  <img src="images/tts-refactor-before-after.svg" alt="六套独立的 TTS Serving 栈重构为公共框架接口与模型 Hook 的前后对比" width="96%">
  <p><em>图 3：模型保留生成和 codec 的具体实现，框架统一管理重复的 Serving 生命周期。</em></p>
</div>

对应到实现上，关键的几个组件包括 [`TtsEngineBuilder`](https://github.com/sgl-project/sglang-omni/pull/923)、[`DeclarativeStateBase`](https://github.com/sgl-project/sglang-omni/pull/1050)、[`ReferenceEncodeService`](https://github.com/sgl-project/sglang-omni/pull/926)、[`BatchVocoderBase`](https://github.com/sgl-project/sglang-omni/pull/940)、[`StreamingVocoderBase`](https://github.com/sgl-project/sglang-omni/pull/936) 和 [`OmniScheduler`](https://github.com/sgl-project/sglang-omni/pull/937)。它们是可以按模型需求组合的窄接口，比如一个后端可以只用引擎启动而不用流式 vocoder，也可以只用状态传输而不共享 codec 实现。

## 公共生命周期会放大隐含假设

把各模型分散的逻辑收编进框架后，原来只影响一个模型的隐含假设，可能沿着公共路径影响多个后端。例如，缓存通常根据 Token ID 判断两段输入是否相同；FishAudio S2-Pro 的参考音频还有一部分声学信息没有写进 Token ID，沿用原来的判断就会复用错误的参考条件。迁移过程中，我们重点排查并修复了以下三个问题：

### Cache Key 不能只看 Token ID

FishAudio S2-Pro 的参考音频包含多个 VQ codebook，只有 codebook 0 会变成 prompt token IDs，其余 codebook 则通过 embedding 输入模型。于是出现了一种很难察觉的情形：两段参考音频即使携带不同的声学信息，只要两次请求的文本 prompt 和第一层编码都相同，普通的 Radix Cache 就会把它们当作相同的 Prefix。后发起的请求会错误复用先前请求的 KV Cache，生成结果也会带上错误的参考音色。

**解法：** 在迁移到 [`OmniScheduler`](https://github.com/sgl-project/sglang-omni/pull/937) 时，我们明确了规范：凡是会影响 KV state 的输入或状态（如 embedding、adapter），其 fingerprint 都必须显式写入 `Req.extra_key`。

### 请求已经结束，音频分片还在路上

流式 vocoder 会根据 request ID 保存每条请求的解码状态，并在请求完成或中断时清理。正常完成的请求会先处理完音频分片再结束；但请求被中断时，上游此前生成的某个音频分片可能仍在传输，并在状态清理后才抵达 vocoder。

旧逻辑看到一个不在状态表里的 request ID，会把这个分片当作新请求的第一块数据，重新创建解码状态。此时原请求的中止信号已经传播完毕，被“复活”的请求再也等不到清理通知，新建的状态也就无法正常清理。对于 MOSS-TTS-Local，这还会让无效请求一直占着 codec session 和 CUDA Graph slot。

**解法：** [`StreamingVocoderBase`](https://github.com/sgl-project/sglang-omni/pull/936) 引入了 Tombstone（墓碑）机制。系统会为已完成或中断的请求保留 Tombstone；晚到的音频分片一旦匹配到 Tombstone，便会被直接丢弃。这些 Tombstone 最后再根据时间统一淘汰。

### Single-flight 异常路径的“死锁”

参考音频编码是一笔不小的开销，而同一个音色往往会被反复使用。对于同一段参考音频的并发请求，框架只执行一次编码，其他请求等待并复用这次结果。我们发现异常路径里存在一个隐患：如果读取音频或写入缓存时失败，而用于协调并发请求的等待状态没有被清理，后续使用同一段音频的请求就会一直等不到结果。

**解法：** 把编码、写入缓存和错误通知放在同一个处理流程中。任一步骤失败时，系统都会立即清理等待状态，并把错误通知给所有等待中的请求。

## 新模型验证：我们的抽象是否足够？

旧模型跑通只能说明接口兼容已有代码，但还不足以证明抽象的通用性。为此，我们新增了 Ming-Omni-TTS、ZONOS2 和 Audar-TTS 三个 TTS 模型来进一步检验这套抽象：一方面考察接入新模型是否足够便捷，另一方面验证新模型能否快速复用已有的抽象并保持良好性能。

以 Ming-Omni-TTS 为例，它的自回归 backbone 输出 hidden state，再由 FlowLoss/CFM tail 采样得到 continuous acoustic latent，最后交给 AudioVAE 解码。这条路径复用了同一套 engine、reference encode 和 state transport 接口。

另一个例子是 Audar-TTS，这是一个阿拉伯语 TTS 模型。为了直接衡量新框架是否降低了模型接入成本，我们让 Audar-TTS 在两套框架上各实现了一次。重构前，我们以旧框架为基线，增加了一版生产级模型支持；重构后，又以共享框架为基线，实现了完全相同的模型能力。

在旧框架上，接入 Audar-TTS 需要增加 797 行代码（不含测试和文档）；使用新框架后，这个数字降到 619 行，减少了 22.3%。其中，从模型“能跑”到生产级可用所需的额外代码由 222 行降到 77 行，减少了 65.3%。减少的主要是参考音频缓存、错误处理和请求清理等重复的 Serving 机制，它们现在由框架统一提供，代码结构更加清晰、易于维护。

代码减少以后，模型计算逻辑和性能没有改变。两版实现使用同一批输入逐项对拍：28 组配对请求生成的 285-code 序列和 24 kHz 波形完全一致；另一组 50 句阿拉伯语测试中，acoustic code、float waveform 和 PCM-WAV hash 也全部一致。H100 上的 stage-sum latency、RTF 和 engine throughput 分别变化 −0.13%、−0.13% 和 +0.16%，均在测量波动范围内。

<div align="center">
  <img src="images/tts-refactor-audar-validation.svg" alt="Audar-TTS 新旧实现的代码量、输出一致性与性能对照" width="88%">
  <p><em>图 4：模型侧的生产级接入代码减少，输出和性能保持一致。</em></p>
</div>

这次重构没有牺牲精度，也没有改变模型的计算逻辑。它减少的是把模型从“能跑”推进到“稳定提供高性能服务”所需的缓存、调度和生命周期代码。

## 结语：重构的职责边界

在这次重构中，sampling、codec session、MoE、codebook layout 以及波形后处理等逻辑依然留在模型自己的目录里。这些逻辑直接决定了模型如何生成和解码，而各家的约束天差地别，强行统一只会导致接口变得臃肿。

在此期间，我们还结合 profiling 探索过更进一步的抽象，例如 batch reference encoding 和 decode-state pool。但压测后发现，这些优化过于“挑场景”——只在特定模型或 workload 下有明显收益，换到其他模型上，收益并不能稳定复现。既然无法普遍适用，我们就没有将它们提前抽象到框架中，而是继续由各模型按需探索优化。

最终沉淀到公共层的，都是那些在多个模型里反复出现且语义已经收敛的 Serving 生命周期：engine 启动、跨阶段状态传输、参考音频缓存、batch/streaming vocoder 的调度与状态清理，以及 scheduler 的公共校验。6 个既有后端平滑迁移完成后，Ming-Omni-TTS、ZONOS2 和 Audar-TTS 的顺利接入也再次验证了这套接口能够支撑新的模型架构。

重构的目标并不是把所有代码都收编进框架。最理想的默契是：**由框架来扛下稳定、高性能 Serving 必需的公共机制；让模型轻装上阵，保留最契合自身的生成逻辑与优化路径。**

后续进展、统计口径和逐 commit 明细会继续更新到 [TTS Refactor Progress](https://luojiaxuan.github.io/sglang-omni/tts-refactor/) 页面。

---

## 致谢

感谢核心 Roadmap、关联 PR 以及探索 PR 的所有贡献者：

[Yuhao Chen](https://github.com/AkazaAkane), [Jiaxin Deng](https://github.com/JiaxinD), [Jingwen Gu](https://github.com/JingwenGu0829), [Chenchen Hong](https://github.com/Hayden727), [Yizhuo Huang](https://github.com/YzXiao101), [Xiangrui Ke](https://github.com/keke0315), [Xinyu Lu](https://github.com/SandyLuXY), [Jiaxuan Luo](https://github.com/luojiaxuan), [Ratish P](https://github.com/Ratish1), [Xinhao Tan](https://github.com/XinhaoTheo), [Xuesong Ye](https://github.com/yxs), [Yue Yin](https://github.com/MelodyyyYin), [Gaokai Zhang](https://github.com/GaokaiZhang), [Yichi Zhang](https://github.com/Ccyest), [Chenyang Zhao](https://github.com/zhaochenyang20)

完整的 PR 历史和评审讨论记录在 [issue #985](https://github.com/sgl-project/sglang-omni/issues/985) 里。
