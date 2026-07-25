# 从六个 TTS 服务栈到一个统一的 Serving 框架：SGLang Omni TTS 重构纪实

在 SGLang-Omni 的开发中，我们历时差不多一个月，完成了一次对 TTS（Text-to-Speech）子系统的彻底重构。

之所以要做这次重构，是我们决定回归“第一性原理”来重新思考：**一个通用的 Serving 框架，到底应该如何优雅地管理复杂且差异巨大的语音模型？**

在讨论 SGLang-Omni 的愿景时，我们曾描述过接入新模型的理想状态：**声明流水线拓扑，实现模型特定的计算，然后把调度、通信和生命周期管理统统交给框架。**

然而在此之前，TTS 子系统是距离这个理想最远的一块“法外之地”。Higgs、MOSS-TTS、MOSS-TTS-Local、Qwen3-TTS、FishAudio S2-Pro 以及 Voxtral-TTS，这六个模型并没有共享同一种合成算法。它们有着不同的 codebook 布局、自回归结构、缓存需求和声码器（vocoder），看起来毫无共性。但实际上，它们共享的是**包裹在数学计算外围的所有工程逻辑**——只不过，每个模型都把这套复杂的机制重新造了一遍。

这次完整的 TTS 重构，将这六个本地的 serving 栈变成了一套明确的接口契约（serving contracts）。最直观的战果不仅仅是代码行数的骤减：

| 验证点 | 核心成果 |
| --- | --- |
| **六个现有的 TTS 后端** | 成功迁移到共享契约上，统一了引擎启动、状态传输、参考音频编码、声码器生命周期、能力声明和调度。 |
| **FishAudio 调度器迁移** | 删除了 591 行的 `FishScheduler`；整个迁移 PR 净减少 **816 行** 代码，同时保持精度和性能不变。 |
| **Audar-TTS 的严格 A/B 测试** | 生产级集成代码从 **797 行减少到 619 行**，“生产能力溢价（用来补齐缓存、容错等生产特性的胶水代码）”从 **222 行降至 77 行（下降 65.3%）**，且输出达到字节级一致，性能持平。 |
| **新模型验证** | Ming-Omni-TTS、Audar-TTS 和 ZONOS2 完美适配了这些共享接口，即便它们的架构在最初设计这六个老模型时根本没出现过。 |

我们检验重构成功的唯一标准很简单：

> **模型的贡献者只应该去实现模型的语义和 Hook，而不是去复制、Fork 或魔改框架的调度状态机。**

---

## 流水线是共享的，但生命周期却不是

SGLang Omni 中的 TTS 推理是一个多阶段流水线。预处理、参考音频编码、自回归生成和波形解码，都有着截然不同的计算特征、显存生命周期和 Batching 机会。重构的火力主要集中在这些阶段**外围**的那一层：那些每个生产级实现都需要，但不该由任何一个单一模型目录独自承担的底层机制。

<div align="center">
  <img src="images/tts-opt-pipeline-overview.svg" alt="SGLang Omni TTS 从预处理、参考音频编码、自回归生成到声码器解码的多阶段流水线" width="78%">
  <p><em>图 1：SGLang Omni 中典型的多阶段 TTS 推理流水线。</em></p>
</div>

## 冗余在数学之外，而不在数学之中

如果只看计算逻辑，这六个后端天差地别。但如果看生命周期，同样的结构在不断重复：

- engine 启动时都在按照几乎相同的顺序解析 checkpoint、构建参数、捕获 CUDA Graph；
- 每个模型都在跨阶段手动序列化流水线状态；
- 参考音频路径各自实现了一套 LRU Cache、相同的 Key 去重和错误处理；
- 不同的声码器都在独立管理请求状态、Chunk 阈值、Flush 顺序和中止逻辑。

这就是**机制冗余（Mechanism duplication）**。它比直接复制粘贴数学公式更难被察觉，因为这些重复代码包裹在模型特定的 Kernel 周围，看起来似乎也是模型独有的一部分。

<div align="center">
  <img src="images/tts-refactor-before-after.svg" alt="六套模型本地 Serving 栈重构为共享框架契约与模型 Hook 的前后对比" width="96%">
  <p><em>图 2：重构没有合并各模型的算法，而是把重复的生命周期机制下沉到稳定的 Hook 边界之下。</em></p>
</div>

因此，我们确立了边界原则：

> **框架拥有可复用的机制，模型目录拥有模型的语义。**

并衍生出三条不可妥协的设计铁律：

1. **共享代码中绝不允许出现模型名称的条件分支。** 差异通过 Hook、能力声明或字段来表达，绝对不能写 `if model == ...`。
2. **迁移即删除。** 采用共享接口后，绝不允许模型保留隐藏的历史遗留路径。
3. **由最硬核的消费者来验证 API。** 绝不能因为最简单的模型能跑通，就轻易拍板一个抽象设计。

---

## 契约栈（The Contract Stack）

重构的结果并不是诞生了一个巨大无比的 `TTSBaseModel`，而是一个精简的契约栈，每一层都有着极其明确的所有权边界。

| 共享表面（Shared Surface） | 框架负责拥有的能力 | 模型负责拥有的能力 |
| --- | --- | --- |
| [`TtsEngineBuilder`](https://github.com/sgl-project/sglang-omni/pull/923) | 恒定的启动顺序、参数管道、延迟的 CUDA Graph 设置、调度器组装等 | Checkpoint 解析怪癖、模型特定初始化、编译策略等 |
| [`DeclarativeStateBase`](https://github.com/sgl-project/sglang-omni/pull/1050) | 字段发射规则、编解码器、底层往返传输、保持 dtype/shape 的张量 Payload | 存在哪些状态，以及每个字段该如何传输 |
| [`ReferenceEncodeService`](https://github.com/sgl-project/sglang-omni/pull/926) | 有字节上限的 LRU 缓存、相同 Key 并发合并（Single-flight）、无缓存投毒的重试等 | 输入归一化、Identity Keys、编解码执行、产物设备策略等 |
| [`BatchVocoderBase`](https://github.com/sgl-project/sglang-omni/pull/940) | 调度器编排与 `准备 → 批量解码 → 存储` 编排 | 波形语义与对应的三个 Hook 实现 |
| [`StreamingVocoderBase`](https://github.com/sgl-project/sglang-omni/pull/936) | 请求状态注册表、Chunk/Flush/Terminal 顺序、中止行为、故障隔离 | Codec 会话、游标计算、CUDA Graph 槽位、解码计划等 |
| [模型能力](https://github.com/sgl-project/sglang-omni/pull/957)与 [`OmniScheduler`](https://github.com/sgl-project/sglang-omni/pull/937) | 声明式特性发现、通用的请求、Batch、KV Cache 与生命周期管理 | 采样、停止语义、缓存指纹、行布局与模型端 Buffer |

最重要的特性是**可组合性（Composability）**。每个契约都消除了某一类生命周期的冗余，但绝不强求所有 TTS 模型长得一模一样。

---

## 三个反直觉的踩坑实录

最有价值的 Bug 往往不是简单的漏写校验，而是当模型特有的假设被移到共享生命周期后暴露出的深层矛盾。

### 1. 相同的 Token ID，并不意味着相同的 Prefix（FishAudio 的踩坑）

FishAudio S2-Pro 的参考音频包含多个 VQ codebook，但只有 codebook 0 会变成 prompt token IDs，其余的会作为 Embedding 喂给模型。因此，两个参考音频可能在调度器看来 Token ID 一模一样，但声学信息完全不同。如果直接用常规的 Radix Cache，缓存复用会导致生成的语音部分条件反射到错误的参考音频上。

迁移到 [`OmniScheduler`](https://github.com/sgl-project/sglang-omni/pull/937) 后，我们将所有参考 VQ codebook 的指纹写入 `Req.extra_key`，让缓存身份覆盖完整的模型输入，而不只是它的 Token 投影。

**结论：** Cache Prefix 的身份不能等同于 Token 的身份。任何通过 Embedding、侧信道或 Adapter 注入的隐藏条件信息，都必须参与 Cache Key 的哈希计算。

### 2. 僵尸请求（Zombie Requests）的复活

在流式流水线中，如果声码器发出了最终结果并清除了请求状态，此时网络中延迟的一块音频代码才刚刚抵达，天真的状态注册表会认为这是一个新请求，从而创建一个“僵尸流”。它不仅会在客户端收到最终结果后继续乱发音频，还可能永久占用 CUDA Graph 槽位。

在提取 [`StreamingVocoderBase`](https://github.com/sgl-project/sglang-omni/pull/936) 时，我们将“已完成”变成显式 Tombstone：任何属于已完成或已中止请求的晚到 Chunk 都会被丢弃，绝不允许重新创建状态。

**结论：** 仅仅删除请求状态是不够的。在异步流水线中，“已完成（Finished）”必须作为一个持久化的负状态（Tombstone）存留足够长的时间，用来拒绝在途的无效消息。

### 3. 不存任何东西，也能让缓存永久投毒

在 [`ReferenceEncodeService`](https://github.com/sgl-project/sglang-omni/pull/926) 中，我们用了相同 Key 单飞（Single-flight）机制：一个 Leader 负责编码，Followers 等待。如果 Leader 在编码成功后，写入缓存时抛出异常，此时没有坏数据被缓存——但这个 Key 被“投毒”了。所有的 Followers 会对着一个不再存在的 Leader 傻等直到超时，并且以后的请求也会无限重复这个悲剧。

共享服务因此把编码、输入重校验和缓存写入视为同一个失败域：任一步骤抛出异常，都必须删除 In-flight Key，并用同一个异常唤醒所有 Followers；反过来，Follower 自己超时也不能擅自取消仍然有效的 Leader。

**结论：** Single-flight 的正确性不仅是“计算一次”。Leader 的任何异常退出路径都必须干净地唤醒并解散所有跟随者，且跟随者绝对不能负责清理 Leader 的残局。

---

## 迁移就是一场审计：Fish 与 OmniScheduler 的融合

原本 FishAudio S2-Pro 自带了一个 591 行的专属调度器，因为当时的共享调度器无法表达它的需求。随着 `OmniScheduler` 的成熟，我们在 [PR #937](https://github.com/sgl-project/sglang-omni/pull/937) 中将 Fish 迁移了过去，最终 PR 净删除了 816 行代码。

这次迁移不仅是去重，更是一次架构审计：

- **词表边界变成了强制契约：** 共享调度器强制校验采样的 Token ID，逼迫 Fish 必须根据完整的 Tokenizer 配置词表大小，而非使用一个较小的基础词表。
- **不可能的请求会尽早失败：** 共享调度器会预先校验请求容量，Fish 因此必须在入口处根据剩余上下文长度截断请求预算，而不是把注定装不下的请求放进来等它自然 OOM。

各自独立的实现之所以看起来“正确”，往往是因为它们从未经过同样的底盘校验。收敛到统一框架，强制每个模型都必须趟过框架的护城河。

---

## A/B 测试：Audar-TTS 的硬核验证

Audar-TTS 提供了一个最纯粹的对照实验：这个模型被实现了两次，一次是基于重构前的旧栈，一次是基于共享框架。

- **生产能力溢价（Production capability premium）：** 即为了让一个最小化 Demo 达到生产可用（加入缓存、生命周期安全、错误处理等）所需要增加的胶水代码。共享框架让这部分代码暴降了 **65.3%**（从 222 行降至 77 行）。
- **绝对的等价性：** 在严格的对比中，数十个配对请求产出了**字节级完美一致**的声学代码和 PCM-WAV 哈希。
- **性能持平：** 在 H100 上，延迟和 RTF 波动不到 0.15%，吞吐量微涨 0.16%。

<div align="center">
  <img src="images/tts-refactor-audar-validation.svg" alt="Audar-TTS 新旧实现的代码量、输出一致性与性能 A/B 验证" width="88%">
  <p><em>图 3：模型拥有的生产胶水代码显著减少，同时保持输出与性能不变。</em></p>
</div>

请注意，框架并没有让 Audar 的模型计算本身变快。它做到的是：**在不改变任何输出和性能的前提下，让接入生产级服务的成本变得极其低廉。**

---

## 我们刻意没有抽象的东西

完成这份 Roadmap 并不意味着要把所有反复出现的名词都变成基类。克制，也是重构的重要一环：

- **模型的数学逻辑必须留在本地：** 采样逻辑、Codec 会话、MoE 层、波形后处理等，框架绝不插手。
- **没有证据，就不做抽象：** 比如批量参考音频编码，在没有证明其是真正的性能瓶颈前，绝不盲目做成框架级 Feature。
- **用能力声明代替条件分支：** 框架应该去动态发现模型是否支持流式或 CUDA Graph，而不是去硬编码判断模型名字。

---

## 总结

寻找生命周期的重复，而非仅仅是计算的重复；用最硬核的新模型去丈量框架的边界。一个优秀的共享框架之所以存在，是因为它能强制保证底层的工程一致性。当新模型接入时，开发者终于可以把全部精力放在算法本身，这才是基础设施应有的样子。

---

## 致谢

这次重构由 [SGLang Omni issue #985](https://github.com/sgl-project/sglang-omni/issues/985) 统一追踪。感谢所有在核心 Roadmap、直接关联的补充 PR，以及后来被替代但保留了设计贡献的探索 PR 中担任作者的贡献者（按 GitHub 用户名字母序）：

[@AkazaAkane](https://github.com/AkazaAkane)、[@GaokaiZhang](https://github.com/GaokaiZhang)、[@Hayden727](https://github.com/Hayden727)、[@keke0315](https://github.com/keke0315)、[@luojiaxuan](https://github.com/luojiaxuan)、[@MelodyyyYin](https://github.com/MelodyyyYin)、[@SandyLuXY](https://github.com/SandyLuXY)、[@XinhaoTheo](https://github.com/XinhaoTheo) 和 [@YzXiao101](https://github.com/YzXiao101)。

完整的 PR 历史、评审讨论、测试证据和被替代方案都保留在 issue #985 的关联记录中。
