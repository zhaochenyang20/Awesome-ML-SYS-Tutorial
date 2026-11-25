# 说话是一种行动：将 TTS 变成 interactive model 的 tool

| 模型 | 打断回放中的残缺回复 | 调用语音工具的首轮 prompt | 三句话请求的首音频延迟 |
|---|:---:|:---:|:---:|
| Inkling 2 Mini | TODO → **TODO** | TODO → **TODO** | TODO ms → **TODO ms**（直接文本：TODO ms） |
| Inkling 2 Small | TODO → **TODO** | TODO → **TODO** | TODO ms → **TODO ms**（直接文本：TODO ms） |
| Qwen3.8-27B | TODO → **TODO** | TODO → **TODO** | TODO ms → **TODO ms**（直接文本：TODO ms） |

## TL;DR

模型越来越聪明，人与模型之间却仍是一问一答。人和人的交流是 interactive 的：一边感知一边回应，会插话，被打断时会立刻调整。SGLang Omni 用开源模型构建 interactive 系统，第一步是 ASR、LLM、TTS 组成的级联，由一个持续运行交互管理的 Session 统一调度。旧的级联会把 LLM 写出的每个 token 都念出来，于是只剩一个两难：关掉 thinking，模型很笨；打开 thinking，它又啰嗦到没法念出来。我们想在这个级联里接入工具，仔细想过之后，发现最好的形式是让 TTS 本身成为第一个工具：对 LLM 来说，说话就是一次工具调用，和 Google 搜索、查天气没有任何区别。现在 LLM 只把值得说的话传给 `tts_start`；Session 决定语音何时停止，并通过 `heard_prefix` 告诉模型用户听到了多少。在 Inkling 2 Mini、Inkling 2 Small 和 Qwen3.8-27B 上，流式输出工具参数、再预填充调用开头之后，首音频中位延迟平均从 TODO ms 降到 TODO ms，旧的直接文本路径为 TODO ms。

## 为什么要做 interactive model

模型的智力在快速增长，人与模型协作的方式却几乎没有变：写一段 prompt，等一段回复。人和人的交流不是这样。双方共享同一段时间和情境，一边感知一边回应：对方话音未落就开始回应，看到对方皱眉就换个说法，情况一变就立刻改变方向。话轮之间的间隙只有 200 毫秒左右，比准备一句话所需的 600 毫秒还短（Levinson & Torreira, 2015），约 40% 的说话人交替是重叠的（Heldner & Edlund, 2010）。轮次式系统把这一切压扁成对讲机式的轮流，让人去适应机器：一口气说完，不能停顿，不能插话，只能等。它只能靠静音去猜你何时说完，所以要么在你停下来思考时抢话，要么让每一轮都多等一段。

更深的问题在于模型本身。语言模型以单线程的方式经历世界：用户说话时，它只能等；它生成时，世界对它是静止的。它看到的是一段 prompt，不是一条流，察觉不到你犹豫了、移开了视线，察觉不到自己答偏了，也感觉不到 10 秒已经过去。有一种观点认为，人是一台每 200 毫秒接收一次输入的 thinking machine。Thinking Machines Lab 的 interaction model 以 200 毫秒为单位持续读入音频、视频和文字，同时产出回应，才有可能学会什么时候该回应、什么时候该等、什么时候该停。

交互也不是一层可以事后套上的外壳。每一刻都有一个决定要做：开口、等待、停下，还是先去查点什么。这些决定和回答本身一样需要智力；交给静音阈值和手写规则，它们就会比模型本身笨，并随着模型进步而愈发落后。所以，目标不是一个更快的轮次式系统。全双工，也就是同时感知和回应，只是手段；真正的目标是一个会交互的模型，它的交互能力随智力一起增长，贯穿它能感知和产出的每一种模态。

## 我们的级联

原生全双工模型，如 Thinking Machines Lab 的 interaction model、OpenAI 的 GPT-Live 和 Moshi（Défossez et al., 2024），端到端地学习交互，但它们需要强大的全模态基座模型，而这样的开源模型很少。轮次式的全模态模型依然一轮一轮地回应。级联把 VAD、ASR、LLM 和 TTS 串起来，用的正是开源社区最不缺的部件：每一块都能随着更好的开源模型单独升级，出了问题也能定位到具体是哪一块。代价是信息损失：ASR 会丢掉语气，也会把多人的话混成一个人。SGLang Omni 的[流式级联](https://www.sglang.io/blog/omni-interactive-runtime)由一个 Session 统一调度这些部件：它持续运行交互管理，判断用户何时结束输入、何时插话，负责停止播放，并让对话历史与实际播出的内容保持一致。级联是通往 interactive model 的过渡。在开源模型能独立做出所有交互决定之前，我们的原则是：模型能做好的决定交给模型，模型还感知不到的信号由系统协调，并随着模型变强逐步交还给模型。

## 为什么 TTS 应该是一个工具

想象一下，你正在驾驶电动汽车，询问车内的智能助手今天天气如何。你期望听到什么，又期望看到什么？你想听到的只是一句"今天 22 度"，想看到的却可以是中控屏上一整周的天气。在这句话背后，模型已经先查过了天气。人想听到的东西要简洁，想看到的东西可以丰富。这样的交互体现出高度的智能：模型清楚地知道该思考什么，该展示什么，又该念出什么。

我们之前的级联不是这样工作的。ASR、LLM、TTS 是同一层级的节点，LLM 写出的每个 token 都会被 TTS 念出来。表面上的问题是有些 token TTS 根本念不对：表情符号、Markdown 的星号、表格。这些可以过滤掉。本质问题在于，开源 LLM 大多是为 coding 训练的，说起话来也像写代码：又长又有结构，一开口就是成千上万个 token。关掉 thinking，模型就变笨，说的内容要么不对，要么调不好工具；打开 thinking，模型变聪明了，可没有人想听完它写的每一个字。

把每个 token 都念出来，还会让模型不知道用户到底听到了什么。用户插话时，旧的级联会把历史里的回复截断到已播放的部分，以"…"结尾，再加一条系统说明。模型会在上下文里学到这种错误的输出方式：即便这一轮没有被打断，它也给不出完整的回答，而是以省略号结尾。沿用 [#3255](https://github.com/sgl-project/sglang-omni/pull/3255) 的设置，给模型输入每条回复都被打断的 20 轮历史，TODO% 的最终回答以省略号结尾。

所以 TTS 不能什么都念，得由模型来挑。与此同时，我们本来就想让级联系统能调用工具：车里那个智能助手，得先查到天气才能回答。仔细想过之后，我们认为 TTS as a Tool 是同时解决这两件事的最好形式，于是 TTS 成了我们的第一个工具。我们把 TTS 从流水线中的一个节点降级为一个工具；对 LLM 来说，`tts_start` 和 Google 搜索、查天气没有任何区别。模型决定说什么；系统决定语音何时必须停止，并记录用户大概听到了什么。两者之间的接口是一次工具调用。这也是我们在"说话"这件事上，对运行时那篇文章留下的问题给出的回答：交互行为有多少该由模型学会，有多少该由系统来协调。

## 系统设计

Session、LLM 和 TTS 之间的协作遵循三条规则：

- **只有 Session 会改变状态。** LLM 发出动作，TTS 执行动作，只有 Session 决定语音是否在播。节点之间从不直接通信，所以它们对状态的认知不会漂移。
- **每个动作只有一个所有者。** `tts_start` 属于 LLM，`tts_close` 属于 Session，不会有人为"谁停了什么"产生竞争。
- **每次调用只有一个应答。** 一次 `tts_start` 对应一个工具响应，所以只需要 `tool_call_id` 这一个 ID。

| 动作 | 调用方 | 作用 |
|---|---|---|
| `tts_start(text)` | LLM | 把文本切分成句子，然后逐句合成并播放 |
| `tts_close` | Session，在用户开始说话时 | 立即停止播放并清空缓冲区。它不在 LLM 的工具列表中。 |

一个 Session 同一时间只播放一段语音。播放中再收到 `tts_start`，会得到 `BUSY`，而不是替换掉当前语音。

<div align="center">
  <img src="images/tts-tool-sequence.png" alt="一次 tts_start 调用：LLM 调用 tts_start，Session 让 TTS 说话；语音要么播完并记录为 completed，要么用户开始说话，Session 向 TTS 发送 tts_close、向 LLM 发送 stop，并记录为带 heard_prefix 的 interrupted" width="90%">
</div>

每次 `tts_start` 都恰好以一种结果结束：

<div align="center">
  <img src="images/tts-tool-outcomes.png" alt="tts_start 的结果：被拒绝的调用或没有可说内容的调用返回 failed；被接受的语音最终为 completed、带 heard_prefix 的 interrupted，或带 heard_prefix 的 failed" width="90%">
</div>

`heard_prefix` 是文本中根据播放记录判断用户已经听到的部分（[计算方式](#计算-heard_prefix)）。

**打断走快速路径。** 语音活动检测（VAD）一检测到用户说话，Session 就立即关闭 TTS 并停止 LLM，不去询问 LLM。如果先让用户的话经过 ASR 和 LLM，就会多播出一段音频，这段音频会被记为"已听到"，可用户当时正在说话，其实没听到。

## 设计权衡

### 开不开 thinking？

整个设计正是从这个权衡出发的。不开 thinking，模型对这类工具的 zero-shot 泛化很差（很遗憾，我们发现，模型在不开 thinking 的时候，几乎没有 zero-shot 操作工具的能力）：把 `tts_start` 作为可选项时，TODO 个首轮 prompt 中有 TODO 个调用了它。开了 thinking，模型不用强制也会调用，而且因为 TTS 是工具，thinking 的内容永远不会被念出来。代价是延迟：写完整段文本、出第一段音频之前，中位要等 TODO s。因此本文的 profile 仍然关闭 thinking，并强制调用（见[下一节](#让真实模型用起来)）。怎样让 thinking 便宜到可以一直开着，仍是开放问题。

| 指标 | Inkling 2 Mini | Inkling 2 Small | Qwen3.8-27B |
|---|---:|---:|---:|
| 关 thinking：可选的 `tts_start` 被调用 | TODO/TODO | TODO/TODO | TODO/TODO |
| 开 thinking：可选的 `tts_start` 被调用 | TODO/TODO | TODO/TODO | TODO/TODO |
| 开 thinking：首音频中位延迟 | TODO s | TODO s | TODO s |

### 调用何时得到应答？

一段语音会持续几秒，模型需要知道它是怎么结束的。如果立刻用"已接受"应答 `tts_start`，之后再单独上报完成情况，每次调用就有两个应答；要把它们关联起来就得再加一个 ID，之后每个操作都要带着它。

我们只在模型需要据此行动时才应答：被拒绝的调用立即应答，被接受的调用在语音结束时应答一次。一次调用，一个结果，不需要第二个 ID。代价是模型在说话期间什么也不知道，但这几乎没有损失，因为这段时间本来也没有东西会 prompt 它。

### 谁可以停止一段语音？

我们考虑过让 LLM 自己关闭语音、提供能改写正在播放句子的 `tts_edit`、让新的 `tts_start` 替换当前语音，以及把并发的 start 排队。最终选了最窄的方案：只有 VAD 会触发关闭，并且要经过 Session；第二个 start 得到 `BUSY`。

LLM 不知道自己的语音播到了哪里，也没有在这类接口上训练过，没法指望它在正确的时刻停下。edit 需要句子切分、锁定点、版本号，还要能丢弃过时的音频。close 本身就会返回停止位置，"关闭再重来"能以少得多的机制满足同样的需求。代价是模型无法在一段语音中途自我纠正。

### 模型如何得知听到了什么？

Session 把停止位置写进工具结果，并随着播放报告的到达更新它：默认按句子边界估计，开启对齐器时用词级时间戳。下一个 prompt 会带上新的输入和这个结果。被打断的语音进入历史有三种方式：

| 方案 | 模型看到的内容 |
|---|---|
| 把回复截断到已听到的部分（旧级联） | 一条以省略号结尾的助理回复，外加一条系统说明 |
| 在用户消息中加入打断说明 | 已听到的文本，夹在用户对这次打断的描述里 |
| 保留调用，把 `heard_prefix` 放进工具响应（我们的方案） | 生成的文本在调用里，已听到的文本在调用结果里 |

我们的方案把模型写了什么和实际播了什么分开记录，调用里保留模型被打断时已经写出的全部文本。在 Qwen3.8-27B 的一次回放中，调用里写的是 "The parcel code is V2P2."（取件码是 V2P2。），`heard_prefix` 停在 "The pickup desk is next to the main entrance."（取件台在正门旁边。）。被问到是否已经念过取件码时，模型下一次 `tts_start` 说的是 "Not yet read. The code is V2P2."（还没念。取件码是 V2P2。）TODO：Inkling 2 Mini 和 Inkling 2 Small 的同一回放。

## 让真实模型用起来

**强制调用。** 设置 `tool_choice: required` 后，TODO 个首轮 prompt 中有 TODO 个调用了 `tts_start`。如果服务端忽略了 `tool_choice`，模型写出了纯文本，这段文本会被当作一次 `tts_start` 调用来朗读，结果也按同样方式上报。

**结束一轮。** 如果每个 prompt 都要求调用，模型就无法选择沉默。我们试过给它一个 `silent` 工具：在 Session 的第一次输入上，TODO 个普通问题中有 TODO 个模型选择了沉默；一段语音完成之后，TODO 个样本中有 TODO 个重复了答案（改用 `silent` 动作时为 TODO/TODO）。所以现在系统在每次用户输入、一次成功的语音之后就结束这一轮，并把结果放进下一个 prompt。只有调用被拒绝或语音失败时才会再次 prompt 模型，每次输入最多四次。

**首音频延迟。** SGLang 自带的工具调用 parser（Qwen3.8-27B 用 `qwen3_coder`，Inkling 2 Mini 和 Inkling 2 Small 用 TODO）要到闭合标签才释放 text 参数，合成器只能等整段参数写完。我们做了两处改动：

1. 我们的 chat service 在模型书写的同时流式输出 `text` 参数，模型还在写后面的内容时，第一句已经在合成了。
2. 我们[预填充调用开头](https://github.com/sgl-project/sglang-omni/pull/4472)，作为一条让模型续写的 assistant 消息（`continue_final_message: true`，`tool_choice: auto`），解码直接从要说出的文本开始。Qwen3.8-27B 的开头是 14 个 token 的 `<tool_call>\n<function=tts_start>\n<parameter=text>\n`；Inkling 2 Mini 和 Inkling 2 Small 的开头为 TODO。

| 配置（请求一句 / 三句） | Inkling 2 Mini | Inkling 2 Small | Qwen3.8-27B |
|---|---:|---:|---:|
| TTS 朗读模型的文本（旧流水线） | TODO / TODO ms | TODO / TODO ms | TODO / TODO ms |
| `tts_start`，SGLang 的 parser | TODO / TODO ms | TODO / TODO ms | TODO / TODO ms |
| `tts_start`，流式参数 | TODO / TODO ms | TODO / TODO ms | TODO / TODO ms |
| `tts_start`，流式参数 + 预填充开头 | TODO / TODO ms | TODO / TODO ms | TODO / TODO ms |

每种配置、每种回复长度都回放同样的 15 段音频，TODO 个轮次中有 TODO 个产出了音频。表中为中位数，计时从客户端发出包含 VAD 检测到的语音结束的数据包开始，到收到第一个非空音频块为止。三句话请求上，流式节省了 TODO ms，预填充又节省了 TODO ms（成对中位数）；最终的工具路径比直接文本慢 TODO ms（一句话请求上慢 TODO ms）。

**失败。** 失败的语音不重试，直接返回 `failed`。这个 profile 的 TTS 超时为 10 s 而不是 120 s。如果合成在播了几句之后失败，响应会带上 `heard_prefix` 和一条说明，让模型只说剩下的部分。没有这条说明时，TODO 个样本中有 TODO 个模型从头重说；加上之后为 TODO/TODO。

| 行为 | Inkling 2 Mini | Inkling 2 Small | Qwen3.8-27B |
|---|---:|---:|---:|
| `tool_choice: required` 下首轮 prompt 调用 `tts_start` | TODO/TODO | TODO/TODO | TODO/TODO |
| Session 第一次输入时，普通问题选择沉默 | TODO/TODO | TODO/TODO | TODO/TODO |
| 语音完成后重复答案（`silent` 工具） | TODO/TODO | TODO/TODO | TODO/TODO |
| 语音完成后重复答案（`silent` 动作） | TODO/TODO | TODO/TODO | TODO/TODO |
| 合成失败后从头重说（无说明） | TODO/TODO | TODO/TODO | TODO/TODO |
| 合成失败后从头重说（有说明） | TODO/TODO | TODO/TODO | TODO/TODO |

## 评估

我们用 Inkling 2 Mini、Inkling 2 Small 和 Qwen3.8-27B，基于 [SGLang Omni `02e5103`](https://github.com/sgl-project/sglang-omni/tree/02e5103ce0428b8a37e2339116b96a516c0eb36f) 运行了[四项评估](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/pull/311#issuecomment-7124364587)：Qwen3.8-27B 跑在一张 H200 上，Inkling 模型跑在 TODO 上。正文数字取三个模型的平均，表格列出每个模型。延迟结果见上一节。

**Full-Duplex-Bench。** 在 4 个并发 session 下运行直接文本路径和预填充工具路径，使用 100 个 v1.5 样本对（每类 25 个）和 30 个 v1.0 样本（每个公开子集 10 个）。本地 Qwen 评判模型把直接文本路径中 TODO/25 的 backchannel 用例判为"继续原来的讲话"，工具路径为 TODO/25。判为"做出回应"的用例从 TODO/25 变为 TODO/25，v1.5 的平均响应延迟从 TODO s 变为 TODO s。以上均为单次运行，需要授权的 CANDOR 子集留待以后。

| 直接文本 → 工具路径 | Inkling 2 Mini | Inkling 2 Small | Qwen3.8-27B |
|---|---:|---:|---:|
| 判为"继续原来的讲话"的 backchannel（/25） | TODO → TODO | TODO → TODO | TODO → TODO |
| 判为"做出回应"的用例（/25） | TODO → TODO | TODO → TODO | TODO → TODO |
| v1.5 平均响应延迟 | TODO s → TODO s | TODO s → TODO s | TODO s → TODO s |

**多次打断后的残缺率。** 按 [#3255](https://github.com/sgl-project/sglang-omni/pull/3255#experiments) 的话题重建四段 20 轮对话，把此前每条回复的前 30% 标记为已听到，每段对话、每种历史格式采样 32 个最终回答。截短回复加系统说明：TODO/128 个回答以省略号结尾（TODO%）；英文内联说明：TODO/128；带 `heard_prefix` 的 `tts_start`：TODO/128。

| 历史格式（以省略号结尾的回答，/128） | Inkling 2 Mini | Inkling 2 Small | Qwen3.8-27B |
|---|---:|---:|---:|
| 截短回复加系统说明 | TODO | TODO | TODO |
| 英文内联说明 | TODO | TODO | TODO |
| 带 `heard_prefix` 的 `tts_start` | TODO | TODO | TODO |

**历史回放。** 两种条件下，工具参数里都有一个随机编码；"已听到"条件还把它放进了 `heard_prefix`。被问到是否念过这个编码时，模型在 32 对中有 TODO 对两种条件下都回答正确，64 个用例中有 TODO 个在下一次 `tts_start` 里给出了正确编码。这些是带预设播放历史的直接 chat 请求，测的是模型如何使用这份记录，而不是实时播放的对齐。

| 指标 | Inkling 2 Mini | Inkling 2 Small | Qwen3.8-27B |
|---|---:|---:|---:|
| 两种条件下都回答正确（/32 对） | TODO | TODO | TODO |
| 下一次 `tts_start` 给出正确编码（/64） | TODO | TODO | TODO |

## 局限与开放问题

剩下的问题，大多关乎控制权该放在哪里。

**Thinking 与工具调用。** 我们希望开着 thinking，但用户不会等，工具调用也要时间。一个思路是在模型思考或调用工具时先说一句附和或过渡语，比如在半秒内先说"好，我查一下今天的天气"。难点在于，这句话和工具返回后的语音是两次合成，要让它们听起来像一口气说下来的。

**屏幕。** 驾驶电动汽车的人应该听到"今天 22 度"，在中控屏上看到一整周的天气。渲染的问题我们还没有好的答案：模型要渲染的内容，该如何决定？TTS 作为工具，让念出来的内容变得相当聪明，也让显示的内容可以和念出来的不同；但现在的 demo 显示的仍是念出来的话，渲染的内容还需要设计。

**更多工具。** 目前唯一的工具就是 TTS，接下来是 Google 搜索和查天气。异步工具和"每次用户输入只说一次"相冲突："我查一下，然后告诉你"需要第二段语音。结果在语音结束后才到达时，由谁来触发它：一条系统规则，还是一次新的模型 prompt？

**自我纠正。** `tts_close` 或 edit 什么时候该交给模型？前提是模型能知道自己的语音播到了哪里。

**引导（steering）。** 模型处于工具循环中途时用户开口，系统应该立即打断，还是等循环跑完再加入用户的话？

**多说话人。** ASR 不带上下文，也不区分说话人，多人的话会被混成一个人。要在多大程度上支持多说话人，还需要想清楚。

**打断与拒识。** 打断走快速路径：VAD 一检测到人声，Session 就关掉 TTS。VAD 只判断有没有人在说话，不理解语义，信号也不经过 LLM，模型无法据此判断该不该停。它的阈值又很低，背景音乐、远处有人说话、几个人同时聊天，都会打断语音；而在同样嘈杂的环境里，ASR 往往能认出用户的声音，不把背景音乐混进去。系统需要拒识：用户说与对话无关的话、旁人插话时，知道不该停下。这本质上是语义判断，可能要牺牲一些打断延迟。可行的方向包括给 VAD 加语义探针、用拒识器只接收目标说话人的声音，以及让用户自己调节 VAD 阈值。

如今由 Session 来停止语音，是因为它能看到连续的输入和播放时间线。如果未来的模型能直接利用这些信号，`tts_close` 就可以变成模型的工具。

## 工程附录

### 流式工具调用 Parser

SGLang 0.5.23 的 `qwen3_coder` parser（Qwen3.8-27B 使用）只在闭合标签处释放字符串参数，回复的文本要到 TODO s 才一次性到达。`sglang_omni.models.streaming_cascade.chat_service` 就是 `sglang.launch_server` 加上一个继承 SGLang detector 的 parser，在模型书写的同时释放字符串参数；换用后，第一句在 TODO s 时就已完整。由于 chat service 依赖 SGLang 的内部实现，有一个测试专门检查它产出的调用与 SGLang 的 detector 一致。

Inkling 2 Mini 和 Inkling 2 Small：TODO（使用的 SGLang parser、用它时回复文本何时到达、流式后第一句何时完整）。

### 续写调用开头

prompt 要求调用时，`llm` 节点把调用开头作为 assistant 消息发送，并设置 `continue_final_message: true` 和 `tool_choice: auto`。chat service 把这段开头拼在第一个流式 delta 前面，所以回复仍以流式 `tts_start` 调用的形式到达。SGLang 自己的 server 会把调用剩余部分（连同闭合标签）作为 content 返回，dialog 会把它们也念出来，所以这个选项需要配合 chat service。加上开头后，第一句与不加开头时一致的样本对：Inkling 2 Mini 为 TODO/TODO，Inkling 2 Small 为 TODO/TODO，Qwen3.8-27B 为 TODO/TODO。

### 计算 `heard_prefix`

- 前缀结束于最后一个音频已播完的句子边界。
- 播放位置取自客户端的 `truncate` 报告，并以服务端实际发送的音频为上限。
- 开启强制对齐器（forced aligner）时，位置精确到词。如果一个句子在音频完整之前就被截断，它没有词级时间戳，这时级联自身的识别器会转写已交付的音频，保留转写结果能确认的该句最长前缀。
- 之后到达的报告会改写 `heard_prefix`，下一个 prompt 拿到的总是最新位置。
- 语音失败时，位置取自合成器的 `turn.done`，而不是 `reply.progress`：开启对齐器时，`reply.progress` 会在 `turn.done` 之后才到达 dialog，其中的词位置可能落在句子结束之后。

这条路径复用了运行时文章附录（"Committing Conversation History from Playback"）中的播放反馈，新的只是它的去处：工具响应。

## 使用与致谢

### 使用

Inkling 2 Mini 和 Inkling 2 Small 的 profile 分别为 `TODO` 和 `TODO`，需要 TODO。Qwen3.8-27B 的 profile 名为 `qwen3-asr-qwen3.8-27b-vivian-tts-tool`，需要两张 GPU，或者一张 H200 并使用指南中给出的显存比例。

```bash
hf download Qwen/Qwen3-ASR-1.7B
hf download wilsonzheng0327/Qwen3-ASR-1.7B-dflash --revision d62bd4051e68282e02f93de774036a35f22f1373
hf download TODO  # Inkling 2 Mini
hf download TODO  # Inkling 2 Small
hf download Qwen/Qwen3.8-27B
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base
hf download Qwen/Qwen3-ForcedAligner-0.6B
```

```bash
PYTHONPATH=. aligner-venv/bin/python examples/streaming_cascade/alignment_worker.py --model Qwen/Qwen3-ForcedAligner-0.6B --device cuda:1 --language English --port 8010
```

```bash
# Inkling 2 Mini
sgl-omni serve --config examples/streaming_cascade/profiles/TODO/launch.yaml --enable-realtime --port 8765
# Inkling 2 Small
sgl-omni serve --config examples/streaming_cascade/profiles/TODO/launch.yaml --enable-realtime --port 8765
# Qwen3.8-27B
sgl-omni serve --config examples/streaming_cascade/profiles/qwen3-asr-qwen3.8-27b-vivian-tts-tool/launch.yaml --enable-realtime --port 8765
```

[Full-Duplex Guide](https://github.com/sgl-project/sglang-omni/blob/main/docs/developer_reference/full_duplex_guide.md) 中的 "A live demo of the reference profile" 一节（由 [#4473](https://github.com/sgl-project/sglang-omni/pull/4473) 加入）介绍了对齐器环境、单张 H200 的部署布局以及浏览器 demo。同一份指南中的 "Speaking Through a Tool" 一节完整描述了这个 profile。

### 致谢

`tts_start` 路径在 [sgl-project/sglang-omni#4428](https://github.com/sgl-project/sglang-omni/pull/4428) 中合入。感谢所有为此做出贡献的人：Jingwen Gu、Yuhao Chen、Jiaxin Deng、Wenyao Gao、Jiadong Guo、Jeffero Qu、Junnan Li、Qun Lou、Jiaxian Luo、Jinyan Su、Xuesong Ye、Wilson Zheng、Chenyang Zhao。
