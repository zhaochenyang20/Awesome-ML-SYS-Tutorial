# 我的语音输入法与 Hybrid 模型的前缀缓存 学习计划

> 文章类型：**工程实践博客**（以个人工程实践为叙事主线，含关键源码走读，但不写成源码走读文）

## 驱动问题

**在 Gated DeltaNet 这类 hybrid 模型上，"稳定前缀的内容对上了"为什么不等于"能从稳定前缀的边界恢复"？如果恢复位置不由前缀长度决定，那它由什么决定？**

这个问题**不能**在文章开头提出。读者需要先拿到三样东西才有能力理解它：

1. **负载层面**：我的语音输入法是一个 5500 : 40 的病态负载，它的可用性完全建立在"长 system 能被 cache 住"这一个假设上。
2. **概念层面**：全注意力的 KV cache 为什么可以裁到任意位置（可寻址性），而 recurrent state 为什么不行（in-place、沿序列滚动）。
3. **模型层面**：Qwen3.8 的 64 层里有 48 层是 Gated DeltaNet、16 层是 Gated Attention——这条序列的恢复粒度由"最弱的一侧"决定。

**驱动问题的出现位置**：在"第七步：驱动问题与现场日志"，即负载画像（第一步）、概念框架（第二~四步）、模型特征（第五~六步）都讲完之后。此时贴出那条真实日志——共同前缀 5514/5558（99.2%）、restore 落在 4536、重算 1020 token / 3.5s、decode 14 token 只用 0.48s——读者会自己产生"命中了却还要重算 1000 个 token"的困惑，问题在这一刻自然浮现。后续所有工程内容（真实链路、源码、三次尝试、SGLang 对比）都在回答它。

> 注意：**禁止**把驱动问题写成"如何让 Ollama 变快"。那是结果，不是问题。真正的问题是"恢复粒度由什么决定"，`num_batch` 只是它的推论。

---

## 动机定位

这篇文章的起点不是一个技术好奇心，是我自己每天都在用的东西突然变慢了。

我在 macOS 上的语音输入法不是一个现成产品，是 ASR + LLM 润色拼起来的一套本地流水线。ASR 负责把声音转成"声学上最可能的字"，但它不知道我平时在聊什么，同音词替换是常态；真正把话变成人话的是后面那个 LLM，而它靠的是一段大约 5500 token 的 system prompt——里面是我的私人词表和纠错规则，人名、项目代号、术语、口癖，全是个人信息。所以这套东西**只能**在本地跑。前天晚上我把润色那一段从 Qwen3.6 换成 Qwen3.8，每句话的等待时间从可以忍受变成了三秒半，而三秒半对一个输入法来说等于不可用。

这个 topic 填补的是一个我此前一直默认成立、但在 hybrid 模型上彻底失效的假设：**"把稳定的长 system prompt 拼在最前面，前缀缓存就只付短 user 的钱"**。我在 [从 KV Cache 到 Zero Overhead Scheduling，一文读懂 SGLang 的调度巧思](../../sglang/scheduler/readme.md) 里写过 RadixAttention 怎么做前缀复用，那套心智模型的前提是——KV cache 是 per-token、append-only、按位置可寻址的，所以可以裁到 LCP 的任意位置。Qwen3.8 把这个前提拆了：48/64 的层跑 Gated DeltaNet，recurrent state 是沿序列滚下来的一坨定长张量，in-place 更新，没有"裁到第 k 个 token"这种操作。前缀缓存于是退化成"快照 + 重放"，而快照打在哪里，就成了性能的全部。

它同时把我几篇文章之间一条没被显式串起来的线接上了。[Power Up Speculative Decoding In Reinforcement Learning](../../rlhf/slime/spec/readme.md) 里我讲过 MTP 怎么加速 decode——这次的实验恰好从反面确认了 MTP 的边界：接受率 1.0，decode 只占 0.48s，而 prefill 占 3.5s，MTP 一点忙都帮不上。更巧的是，llama.cpp 里那个能做**精确**短距离回滚的 `n_rs_seq` 快照环，恰恰是为 MTP/EAGLE3 这类 draft 机制准备的，深度只有 `draft.n_max`，兜不住一句 user。另一头，[一文理解 special tokens 和 chat template](../../transformers/special_tokens/special_tokens.md) 讲的 chat template 渲染，在这里变成了一个性能问题的直接成因——Ollama 在 Go 侧自己渲染模板、只把渲染好的字符串丢给 llama-server 的 `/completion`，于是 llama-server 永远不知道 user 消息从哪里开始，也就没法在消息边界打点。

在知识图谱里，这是第一篇**离开数据中心、写本地单并发推理**的文章，也是第一篇正面处理 **hybrid / linear attention 的 cache 语义**的文章。它和 SGLang 那条线的接口很清楚：llama.cpp 用的是"位置寻址的 checkpoint"，SGLang 的 MambaRadixCache 用的是"内容寻址的 radix tree + state fork"——同一个问题的两种解法，正好构成一次有真实参照的对比（不是稻草人）。

---

## 前置知识检查

- [从 KV Cache 到 Zero Overhead Scheduling，一文读懂 SGLang 的调度巧思](../../sglang/scheduler/readme.md)：本文第二步"全注意力前缀缓存为什么可裁"的全部前提都在这里。读者需要先有 RadixAttention / prefix cache / LCP 匹配的心智模型，第三步的"为什么这套在 recurrent 上不成立"才有对照物。
- [Power Up Speculative Decoding In Reinforcement Learning](../../rlhf/slime/spec/readme.md)：需要 MTP / draft-verify 的基本概念。本文两处用到：(1) 排除 decode 侧嫌疑（MTP 接受率 1.0 但请求仍慢）；(2) 解释 llama.cpp 里 `n_rs_seq` 这个精确回滚环为什么只为 draft 服务、深度只有 `draft.n_max`。
- [一文理解 special tokens 和 chat template](../../transformers/special_tokens/special_tokens.md)：需要理解 chat template 把 `[system, user]` 渲染成一条扁平 token 序列的过程。本文第八步会指出：一旦渲染发生在 Ollama 的 Go 侧，消息边界信息就在送进 llama-server 之前丢失了。
- [当 SGLang OOM 的时候，究竟在 OOM 什么？](../../sglang/kvcache-code-walk-through/mem-fraction-static.md)（选读）：第十一步讨论 SGLang 把 Mamba pool 和 KV cache pool 分开管理、以及 elastic memory pool 时，有这篇的显存账本会更好读。

---

## 学习路线图

> 顺序硬约束：概念框架 → 具体模型/场景 → 工程代码。绝对不能反过来。
>
> 第一步是**负载画像**（问题从哪来），不是分析工具，也不是被分析的模型。它在概念之前不违反上述约束——真正的分析框架（第二~四步）仍然严格早于模型（第五~六步），而模型严格早于所有代码（第八步起）。

### 第一步：语音输入流水线的设计思路

- **深度层级**：建立直觉
- **从何推导**：起点
- **目标**：让读者理解这条流水线**为什么长成这样**，从而理解它的负载为什么是 5500 : 40 这种极端形状，并在最后接上一句要害——这套方案的可用性完全押在"长 system 能被 cache 住"这一个假设上。
- **方法**：设计推演（不是产品介绍，是"每一处设计从什么约束推出来"）
- **需要独立展开的概念**（每一条都要给出"为什么不是另一种做法"，而不是罗列特性）：
  - **为什么是 ASR + LLM 两段，而不是端到端语音模型**。端到端把"听清"和"听懂我"耦合成一件事，我没法单独换掉其中一段。两段式的代价是多一跳延迟，收益是两段可以独立迭代——这次我只换了润色那一段，ASR 一个字没动。这条**必须**用真实存在的替代方案（端到端 omni 模型）来对比，不能构造假想方案。
  - **为什么润色这一段非要强模型**。这是全章最需要展开的一条，**禁止**一句"LLM 效果好"带过。要说清 ASR 的错误类型：它输出的是声学上最可能的字，中文同音词密度又高，"预填充"和"预填冲"在声学上没有区别。纠正它需要的不是更好的声学模型，而是**先验**——知道我常提到的人叫什么、我在做的事情叫 prefill。这个先验没有别的载体，只能是 system prompt。
  - **为什么 system prompt 会长到 5500 token**。它是私人词表 + 纠错规则 + 风格约束的集合：人名、项目代号、术语表、我的口癖、我不希望被"改好"的写法。**它只会越长不会越短**——这一点很重要，因为它意味着"把 system 写短一点"从一开始就不是一个选项。
  - **为什么必须在本地**。那 5500 token 几乎全是个人信息。这不是性能选择，是前提。
  - **为什么并发度是 1、而且不能攒批**。语音输入是人类串行行为，说完一句才有下一句；输入法要边说边出字，攒批等于让用户干等。推论很硬：**所有吞吐向的优化对我一文不值，我只关心单条请求的延迟。** 这条要显式写出来，因为它解释了后面为什么可以毫不犹豫地拿冷启动换单条延迟。
  - **为什么前面挂一层只打日志的代理**。不改写 model 名，纯粹为了能看到每条请求的真实耗时构成。可观测性先行——没有它，这次 debug 根本无从下手。这是个小设计，但值得一句。
- **本步的收尾判断（全文枢纽，必须写）**：5500 : 40 的比例意味着，如果 system 每次都要重算，这套方案在第一天就不成立。**我整个语音输入法的可用性，建立在"长 system 能被 cache 住"这一个假设上。** 这句话为第七步的驱动问题预埋了全部张力。
- **需要 chenyang 补充的事实**（我无法从背景推出，写作时请提供，或明确说"略过"）：
  - ASR 那一段的具体形态：Qwen3-ASR 跑在什么上面（MLX / whisper.cpp / 别的），本地还是别处
  - 输入法层面怎么触发：快捷键、常驻监听、还是别的
  - 从触发到出字的端到端延迟预算是多少——这个数字决定了"3.5 秒"到底有多不可接受
  - 从 Qwen3.6 换到 Qwen3.8 的真实动机（润色质量不够？还是就想试试新模型？）
  - 那层日志代理是自己写的还是现成的
- **参考资源**：本机配置与日志

### 第二步：全注意力前缀缓存的"可裁剪性"

- **深度层级**：建立直觉
- **从何推导**：第一步的结论是"我押在长 system 能被 cache 住这一个假设上"。那么这个假设在过去为什么一直成立？这一步把它拆开看。
- **目标**：把"前缀缓存"拆成两个独立的能力——**内容能匹配**（LCP）和**状态能裁剪**（truncate to LCP）。读者平时把它们当成一件事，本文全部的张力来自它们可以被拆开。
- **方法**：概念框架
- **需要独立展开的概念**：
  - **KV cache 的可寻址性**（必须展开为子步骤，不能一句话带过）：为什么 KV 是 per-token 的、append-only 的、位置可索引的；因此 `keep_first(n_past)` 是 O(1) 的元数据操作，不需要重算任何东西。
  - **LCP 匹配 → 裁剪 → 增量 prefill 的三步流程**：用第一步给出的真实数字讲。system 5500 token、user 40 token，第二条请求的代价 ≈ 40 个 token 的 prefill，而不是 5540。
  - **总结性判断**：全注意力下"稳定 system 拼在前面"之所以是个**万能技巧**，是因为匹配和裁剪这两个能力恰好都成立，而且成立得非常廉价。
- **参考资源**：[SGLang scheduler 一文](../../sglang/scheduler/readme.md) 中 RadixAttention 的部分

### 第三步：线性注意力的状态为什么不能裁

- **从何推导**：第二步把前缀缓存拆成"匹配"和"裁剪"，并指出裁剪之所以廉价是因为 KV 按位置可寻址。这一步问：如果一个注意力机制**根本不按位置存东西**，裁剪会变成什么？
- **深度层级**：建立直觉（算法侧）+ 理解复现（引擎侧）
- **目标**：建立"recurrent state 是一坨沿序列滚下来的定长张量、in-place 更新、没有第 k 个 token 的位置"这一物理直觉，从而推出：LCP 匹配仍然成立，裁剪彻底不成立。
- **方法**：概念框架 + 引擎源码注释佐证
- **需要独立展开的概念**：
  - **Gated DeltaNet 的状态更新**（必须展开，不能当黑盒）：从最朴素的线性注意力 `S_t = S_{t-1} + v_t k_t^T` 出发，说清"状态是被一路加/衰减上去的，不是被一个个存下来的"，再加上 gating（自适应遗忘）和 delta rule（先擦后写）这两层。**不要**只写"一种线性注意力"。
  - **类比精确性要求**：优先用读者已有的 LLM 管线知识做类比。好的类比方向是"KV cache ↔ 一本可以翻到任意一页的账本；recurrent state ↔ 一个只有当前余额、没有流水的账户"——余额是对的，但你没法把它"退回到第 4536 笔交易之后"。**避免**"压缩/摘要"这类不精确的比喻。
  - **O(1) 显存的代价是 O(1) 的可寻址性**：本步的总结性判断。线性注意力换来的常数显存，代价恰好是它丢掉了位置维度。
  - **引擎侧的确证**：llama.cpp 在 `llama_memory_recurrent::seq_rm` 里直接写了 `// models like Mamba or RWKV can't have a state partially erased at the end of the sequence because their state isn't preserved for previous tokens`；能力枚举里也有 `COMMON_CONTEXT_SEQ_RM_TYPE_FULL = 2, // can seq_rm full sequences only`。这两处是概念的源码级背书，**应当在概念章节就引用**，而不是留到代码章节。
- **参考资源**：
  - Gated Delta Networks: Improving Mamba2 with Delta Rule（https://arxiv.org/abs/2412.06464）
  - `llama-memory-recurrent.cpp`：https://github.com/ggml-org/llama.cpp/blob/7e4c0a96880dae4fc4268ad441f8a6446bd5460a/src/llama-memory-recurrent.cpp#L170-L186
  - `common.h` 的 `common_context_seq_rm_type`：https://github.com/ggml-org/llama.cpp/blob/7e4c0a96880dae4fc4268ad441f8a6446bd5460a/common/common.h#L970-L975

### 第四步：checkpoint 是唯一的补救，以及它引入的三个新自由度

- **从何推导**：第三步的结论是"不能裁"。既然不能裁，唯一剩下的办法就是**在若干离散位置把整个状态存一份快照，恢复到快照，然后从快照往后重放**。这一步展开这个补救方案本身。
- **深度层级**：理解复现
- **目标**：让读者意识到 checkpoint 不是"另一种前缀缓存"，而是一种**恢复粒度被离散化**的前缀缓存——代价从"新增 token 数"变成"到最近可用快照的距离"。
- **方法**：概念框架
- **需要独立展开的概念**：
  - **快照 + 重放的代价模型**：给出公式化的直觉——`cost ≈ (prompt_end − nearest_valid_checkpoint)`，而**不是** `cost ≈ (prompt_end − LCP)`。这一行是全文的枢纽，后面所有实验都在验证它。
  - **checkpoint 引入的三个自由度**：打在哪里（位置策略）、能存几个（数量上限）、多近算太近（最小间距）。这三个自由度**必须在这里列出来**，因为后面三次尝试恰好各自去动了其中一个，而只有一个动对了。
  - **快照不是免费的**：每个 checkpoint 要存一整份 recurrent state（llama.cpp 日志里会打出 MiB 数），所以"到处打点"不是可选项——这是为什么会有数量上限和最小间距。
  - **总结性判断**：checkpoint 把一个连续问题（裁到任意位置）近似成了一个离散问题（跳到最近的点），近似的质量完全取决于点打得准不准。
- **参考资源**：llama.cpp Discussion #19264（Enable Partial Prompt Cache Reuse for Recurrent Models via State Checkpointing）https://github.com/ggml-org/llama.cpp/discussions/19264

### 第五步：Qwen3.8 全貌

- **深度层级**：摘要提取
- **从何推导**：第四步给出的代价模型是模型无关的。要知道它在**我这台机器上**具体有多贵，得先交代跑的是什么模型。
- **目标**：交代模型定位——来源、参数量级、用途、链接。**只讲全貌，不讲计算特征**（计算特征留给第六步，避免信息重复）。
- **方法**：概念框架
- **要点**：Qwen 团队 2026 年 8 月发布，27B dense，原生 262,144 context（可扩到 1M），带 MTP，训练时就是 hybrid 架构。本地跑的是 Q4_K_M GGUF（tag `qwen3.8` / `qwen3.8:latest`，与 `qwen3.8:27b-mtp-q4_K_M` 同一份权重）。
- **参考资源**：
  - https://huggingface.co/Qwen/Qwen3.8-27B
  - https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B

### 第六步：Qwen3.8 的 cache 特征分析

- **从何推导**：从第五步的全貌自然引出——"抛开模型能力上的设计，从 cache 的角度看，这 64 层是怎么分的？"
- **深度层级**：建立直觉
- **目标**：把第三步的抽象结论落到这个模型的具体数字上，得出"这条序列的恢复粒度由最弱的一侧决定"。
- **方法**：概念框架 + 数值例子
- **需要独立展开的概念**：
  - **层布局**：64 层 = 16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))。即 **48 层线性注意力 + 16 层全注意力**。Gated DeltaNet 侧 48 个 V head / 16 个 QK head，head dim 128；Gated Attention 侧 24 个 Q head / 4 个 KV head，head dim 256。hidden 5120。
  - **混合内存的"木桶效应"**（本步的核心判断，需要展开而非一句话）：16 层全注意力的 KV **是**可以裁到 LCP 的，但 48 层 recurrent **不能**。引擎必须让整条序列的位置指针保持一致，所以恢复位置只能取两者中更保守的那个。**四分之三的层说了算。**
  - **KV cache 行为分析**（同时回答"是什么"和"为什么"）：全注意力那 16 层是常规 paged/连续 KV，跨请求可复用、可截断，因为每个 token 的 K/V 被独立保存；DeltaNet 那 48 层每个 seq 只有一份定长 state，不跨 token 保存，因为它的定义就是把历史压进一个固定张量。
  - **一句对上一代的交代**：Qwen3.6 也是同架构，所以严格说这不是"换模型带来的新问题"，而是我在 3.6 上没有把它逼出来。写作时如果记得 3.6 上的真实体感，值得补一句；如果不记得，**不要编**。
- **参考资源**：Qwen3.8-27B model card 的 architecture 段；对照 Qwen3-Next / Kimi Linear 的 3:1 布局

### 第七步：驱动问题与现场日志

- **从何推导**：第六步得出"恢复粒度由 48 层 recurrent 说了算"。第一步说过我押在"长 system 能被 cache 住"上。那么在真实负载上，这个粒度到底是多少，我押的那一注还在不在？
- **深度层级**：理解复现
- **目标**：**这是文章的驱动问题出现的位置。** 用一条真实日志把矛盾摆到读者面前，让问题自己浮现。
- **方法**：真实日志 + 设问
- **内容**：
  - 日志：共同前缀 5514/5558（99.2%），restore 落在 4536，重算 ~1020 token / ~3.5s；decode 14 token 只有 0.48s；MTP `draft-mtp` 接受率接近 1.0。
  - **先排除 decode 侧**：MTP 在工作，接受率 1.0，decode 占比极小。慢的是 prefill。这一步必须做，否则读者会怀疑是 decode 的问题；也顺带回扣 [slime spec 一文](../../rlhf/slime/spec/readme.md)——MTP 只加快 decode，它对这段 prefill 无能为力。
  - **驱动问题在此提出**：LCP 显示 99.2% 相同，引擎也确实认出了 system，但恢复位置退到了 4536。**内容对上了，为什么边界对不上？1020 这个数字是从哪来的？**
- **参考资源**：本机日志

### 第八步：Ollama → llama-server 的真实链路

- **从何推导**：第七步问的是"点打在哪里"。要回答它，必须先知道**是谁在打点**——我以为我在用 Ollama，实际上真正决定 checkpoint 位置的是它拉起的 llama-server 子进程，以及 Ollama 传给它的那几个参数。
- **深度层级**：理解复现（实际做到源码级，因为这个问题只有源码能回答）
- **目标**：拆掉"Ollama 是一个黑盒推理引擎"这个印象，画出完整链路，并定位三处决定性的实现细节。
- **方法**：代码分析
- **博客化提示**：这一步和第九步是全文源码密度最高的两章，但它是**博客不是源码走读**——每段代码只保留决定性的那几行，其余用文字带过。判断标准：如果一段代码不参与回答"1020 从哪来"，就不要贴。
- **需要独立展开的概念**：
  - **链路本身**：Ollama 通过 `LLAMA_CPP_VERSION` 固定一个 llama.cpp 版本（当前是 tag `b10434`），打上 `llama/compat/` 的补丁后编译出 llama-server，再作为子进程拉起。所以 llama-server 的 CLI 参数、日志格式、环境变量（`LLAMA_ARG_*`）在 Ollama 场景下**全部有效**——这也解释了为什么之前设 `LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT` 时确实能在进程里看到它。
  - **决定性细节 A —— `num_batch` 同时设 `-b` 和 `-ub`**：`params = append(params, "-b", strconv.Itoa(opts.NumBatch), "-ub", strconv.Itoa(opts.NumBatch))`。这一行是后面"改 `num_batch` 有效"的**唯一原因**，必须在这里就点出来，但先不揭晓为什么（留给第九步的 `4 + n_ubatch`）。
  - **决定性细节 B —— 预渲染 prompt 走 `/completion`**：Ollama 在 Go 侧渲染 chat template，用 `--no-jinja --chat-template chatml` 让 llama-server 别去解析模型自带模板，然后把渲染好的字符串 POST 到 `/completion`。**llama-server 收到的是一条扁平字符串，不是 `[system, user]`。**（此处回扣 [special tokens 和 chat template](../../transformers/special_tokens/special_tokens.md)。）
  - **决定性细节 C —— 请求体里没有 `message_delimiters`**：`llamaServerCompletionRequest` 结构体逐字段列出来，`prompt / cache_prompt / n_predict / ...` 一应俱全，**唯独没有** `message_delimiters`。这是消息边界信息丢失的最后一环。
  - **总结性判断**：不是 llama.cpp 不肯在消息边界打点，而是在这条链路上，它压根不知道消息边界在哪。
- **参考资源**（均带 commit hash，pin 在 ollama `e5a81899d014a847a08d47393351908b53d74008`）：
  - `-b`/`-ub` 绑定：https://github.com/ollama/ollama/blob/e5a81899d014a847a08d47393351908b53d74008/llm/llama_server.go#L585-L591
  - `appendJinjaArgs` 与 "Go-rendered chat paths send already-rendered prompts through completion endpoints" 注释：https://github.com/ollama/ollama/blob/e5a81899d014a847a08d47393351908b53d74008/llm/llama_server.go#L776-L785
  - `llamaServerCompletionRequest` 结构体：https://github.com/ollama/ollama/blob/e5a81899d014a847a08d47393351908b53d74008/llm/llama_server.go#L1393-L1415
  - `/completion` 端点：https://github.com/ollama/ollama/blob/e5a81899d014a847a08d47393351908b53d74008/llm/llama_server.go#L1628
  - `llama/README.md`（`LLAMA_CPP_VERSION` 的更新流程）：https://github.com/ollama/ollama/blob/e5a81899d014a847a08d47393351908b53d74008/llama/README.md

### 第九步：llama.cpp 的打点与恢复规则（源码）

- **从何推导**：第八步确认了 llama-server 收到的是一条没有边界信息的扁平 prompt。这一步进 llama.cpp 源码，看它在这种输入下**还剩哪些打点路径**，以及 1020 这个数字的出处。
- **深度层级**：理解复现（源码级）
- **目标**：把第七步的驱动问题彻底回答掉——1020 = `4 + n_ubatch`，而 `n_ubatch = num_batch = 1024`。
- **方法**：代码分析（按执行顺序逐段）
- **需要独立展开的概念**：
  - **打点路径一：消息边界（在这条链路上失效）**。`spans.is_user_start(...)` 会在 user 消息起点处切断 batch 以便打点，`last_user_message_pos()` 还会特别保证最后一条 user 消息前面一定有点。但 `message_spans` 来自请求体的 `message_delimiters` 字段，默认 `json::array()`——**空数组进，空 spans 出**，这条路径整条哑火。这里要和第八步的细节 C 显式扣上。
  - **打点路径二：prompt 尾部的两个固定偏移（实际生效的那条）**。`static const int checkpoint_offsets[] = {4 + n_ubatch, 4};` 加上 `const int n_last = std::min(n_batch, offset);`。逐步分析：
    - 因为 Ollama 令 `n_batch == n_ubatch == num_batch`，`4 + n_ubatch` 会被 `min` 夹回 `n_batch`，所以**深点实际落在 `end − num_batch`**，浅点落在 `end − 4`。
    - **用 ablation 数据反验**：`num_batch=32` → 重算 32–37；`=1024` → 重算 1019–1029。分毫不差。这是一次漂亮的"源码预测 → 实验确认"闭环，**必须写出来**，它是全文最有说服力的一段。
    - 这两个点的设计意图（来自 PR #20288）：深点用于会话转向时的大幅回退，浅点用于最后一条 user 消息被小幅修改时的快速恢复。
  - **恢复路径：为什么"结尾那个点"必然作废**。`pos_min_thold`、`find_if` 的 `cur.pos_max > pos_next` 与 `cur.pos_min < pos_min_thold` 判据、找不到时打出的 `forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory)`。要讲清：上一条请求尾部那个 `end − 4` 的点身上带着**旧的 user**，新请求的 user 不同，它 `pos_max > pos_next`，直接被 erase；于是最近可用的只剩 `end − num_batch`。
  - **最小间距 `-cms` 的真实作用域**。`checkpoint_min_step` 默认 8192，只在两处起作用：`create_checkpoint` 里驱逐离前一个点太近的旧点，以及创建时的 `n_tokens_start > back().n_tokens + checkpoint_min_step` 判据——**但那个判据被 `is_last_user_message || near_prompt_end` 短路了**。尾部那两个点恰恰都在 `near_prompt_end` 里，所以它们**根本不受 cms 约束**。这就是"把 cms 从 8192 改成 32 毫无变化"的源码级解释。
  - **一个没被用上的机制：`n_rs_seq`**。`llama_memory_recurrent::seq_rm` 里其实有一条精确回滚路径——`// partial rollback via per-token snapshot index (bounded by n_rs_seq)`，能把 recurrent state 精确退回 `rollback` 个 token。但 `cparams.n_rs_seq = params.speculative.need_n_rs_seq()`，只在 `DRAFT_MTP` / `EAGLE3` 等场景下等于 `draft.n_max`。也就是说：**MTP 顺手给了我一个精确回滚环，但它只有几 token 深，兜不住一句 user。**
- **参考资源**（均 pin 在 Ollama 当前使用的 llama.cpp tag `b10434` = `7e4c0a96880dae4fc4268ad441f8a6446bd5460a`）：
  - `checkpoint_offsets` 与 `min(n_batch, offset)`：https://github.com/ggml-org/llama.cpp/blob/7e4c0a96880dae4fc4268ad441f8a6446bd5460a/tools/server/server-context.cpp#L3464-L3483
  - 消息边界打点 `is_user_start`：https://github.com/ggml-org/llama.cpp/blob/7e4c0a96880dae4fc4268ad441f8a6446bd5460a/tools/server/server-context.cpp#L3423-L3462
  - `message_delimiters` → `message_spans`：https://github.com/ggml-org/llama.cpp/blob/7e4c0a96880dae4fc4268ad441f8a6446bd5460a/tools/server/server-context.cpp#L4195-L4210
  - 恢复与失败日志：https://github.com/ggml-org/llama.cpp/blob/7e4c0a96880dae4fc4268ad441f8a6446bd5460a/tools/server/server-context.cpp#L3253-L3305
  - `cms` 被 `near_prompt_end` 短路：https://github.com/ggml-org/llama.cpp/blob/7e4c0a96880dae4fc4268ad441f8a6446bd5460a/tools/server/server-context.cpp#L3526-L3540
  - `-cms` / `-ctxcp` 参数定义与默认值：https://github.com/ggml-org/llama.cpp/blob/7e4c0a96880dae4fc4268ad441f8a6446bd5460a/common/arg.cpp#L1687-L1704 、https://github.com/ggml-org/llama.cpp/blob/7e4c0a96880dae4fc4268ad441f8a6446bd5460a/common/common.h#L613-L614
  - `n_rs_seq` 的精确回滚与它的来源：https://github.com/ggml-org/llama.cpp/blob/7e4c0a96880dae4fc4268ad441f8a6446bd5460a/src/llama-memory-recurrent.cpp#L178-L188 、https://github.com/ggml-org/llama.cpp/blob/7e4c0a96880dae4fc4268ad441f8a6446bd5460a/common/common.h#L386-L392
  - PR #20288（尾部双 checkpoint 的设计动机）：https://github.com/ggml-org/llama.cpp/pull/20288

### 第十步：三次尝试的复盘（合并为一步）

- **从何推导**：第九步给出的三条打点路径（消息边界 / 尾部偏移 / 最小间距），恰好对应第四步列出的三个自由度。三次尝试各去动了一个——这一步逐一展开，**用第九步的源码回过头解释每次的成败**。它们是同一个故事的三个侧面，不是三个独立话题，所以合并为一个 `##`。
- **深度层级**：理解复现
- **写作要求**：这一步必须写成**设计演进**，而不是"失败1/失败2/成功"的流水账。演进路径：
  - **baseline（不做任何事）**：每条短请求 ~3.95s，重算 ~1020 token。
  - **中间方案 A：固定垫文**。假设是"两个点贴得太近所以留不住，撑开它们"。对照实验做得很干净：不垫 / 垫进 user（800/1600/3200 字）/ 垫进 system（1600/3200 字）。稳态结果一律是共同前缀 99%+、restore 仍在"结尾往前约 1020"、prefill 仍是约 3.6s。**为什么不够好**：偏移量是相对 **prompt 结尾**算的，不是相对稳定前缀的长度算的；垫文只是把那一对点整体后移，间距纹丝不动。附带一个被实验否定的判断——当时以为"垫进 system"和"垫进 user"会因为角色分界不同而有差别，实际上 llama-server 收到的本来就是一条扁平字符串（第八步细节 B），根本没有角色这回事。**顺带一个反直觉的推论**：在这套机制下，把 system 写得更长只会让"从旧点重算到新结尾"更贵——和过去的直觉完全相反。
  - **中间方案 B：调 `-cms`**。假设是"最小间距 8192 太大，把点挤掉了"，设成 32，环境变量确认生效。日志纹丝不动：restore 4536、重算 1017 token、3.46s。**为什么不够好**：`cms` 只约束"创建时离上一个点太近就不创建 / 驱逐"，而尾部那两个点走的是 `near_prompt_end` 分支，**被短路了根本不看 cms**；何况它们本来就隔了 1020，把下限调到 32 也不会凭空多出点来。（后来从启动脚本里删掉了。）
  - **最终方案：减小 `num_batch`**。从中间方案的失败里推出的正确认识——回退距离 ≈ 一个 ubatch，而请求形态改不了、打点位置也改不了，那就**让"回退一个 batch"这件事本身变便宜**。
- **需要独立展开的内容**：
  - **ablation 的实验方法必须交代**：每个 `num_batch` 先 `ollama stop` 卸掉模型和 KV，再单独冷启动预热 1 次 + 3 次短 user。条件之间不共用 cache；条件内部的 cache 复用是**故意保留**的，因为那就是线上形态。方法论本身值得写——博客读者最容易从这里学到东西。
  - **ablation 表**（原样保留）：

    | num_batch | 短请求重算 token | 短请求总耗时 | 冷启动灌 system |
    |---|---|---|---|
    | 32 | 32–37 | 0.88s | 51.9s |
    | 64 | 63–69 | 0.91s | 28.4s |
    | 128 | 127–133 | 1.01s | 17.7s |
    | 256 | 255–261 | 1.40s | 16.7s |
    | 512 | 511–517 | 2.19s | 16.2s |
    | 1024 | 1019–1029 | 3.95s | 16.9s |

    decode 恒定在约 273ms，与 batch 无关——这一列也要点出来，它再次确认瓶颈只在 prefill。
  - **trade-off 必须讲透，并回扣第一步**：`num_batch` 越小短请求越快，但冷启动灌 5500 token 越亏（128 以下开始明显劣化）。我的日常形态是"启动 preload 一次 + 大量短请求"——**这正是第一步里"并发度 1、只关心单条延迟"的直接推论**，所以我可以毫不犹豫地拿冷启动换单条延迟。选 **64**：短请求比 1024 快约 3s，冷启动 28s，一天付一次。32 只再省约 40ms，冷启动却要 52s，不值。
  - **落地注意事项**：客户端每条请求的 `options` 里都要带 `"num_batch": 64`，漏一条就可能触发按 1024 重载。
  - **线上验证**（比实验短句更长的真实 user）：restore 从 4536 变成 5492，重算 146 token / 1.17s（这句 user 本身约 125 个新 token），整请求 2.75s，其中 decode 44 token 占 1.44s，MTP 接受率仍是 1.0。**146 不是 64**，因为这句 user 比实验短句长——但仍远小于 1020。这个"对不上但解释得通"的数字**必须解释**，否则读者会以为哪里出了错。
- **需要替代方案对比表的设计决策**：`num_batch` 的取值。建议维度：短请求延迟 / 冷启动耗时 / 每日冷启动次数下的摊销成本 / 显存占用 / 适用场景（长会话 vs 一次性短请求）。
- **参考资源**：本机 ablation 日志 + 第九步的源码

### 第十一步：对比分析——SGLang 的 MambaRadixCache

- **从何推导**：第十步的结论是"在 llama.cpp 这条链路上，最优解是把回退距离调小"。但这只是**位置寻址**这条技术路线的局部最优。这一步跳出来问：一个真正为 hybrid 模型设计的生产级引擎，会怎么解这个问题？
- **深度层级**：修改扩展（这是我自己的系统）
- **目标**：把 llama.cpp 的"位置寻址 checkpoint"和 SGLang 的"内容寻址 radix tree + state fork"放在一起，让读者看到同一个约束下两种完全不同的工程答案。**这是真实参照的对比，不是稻草人。**
- **方法**：概念框架 + 代码分析
- **需要独立展开的概念**：
  - **同一组约束的重述**：SGLang 那边对 Mamba state 的三条判断和第三、四步完全一致——(1) in-place 更新、不能回滚；(2) state 比单 token 的 KV 大好几个数量级；(3) 大多数 SSM forward kernel 是 "all or nothing" 的可复用性。这里要显式说"这正是第三章里那条约束"，让映射融进行文。
  - **MambaRadixCache 的三件事**：match（返回 state 非空且 key 是输入前缀的最佳节点，需要**拷贝** state）、insert（chunked prefill / decode 之后把 KV 和 Mamba state 插进树，需要从 request **fork** 一份 state 快照）、evict（两条独立 LRU，KV 必须从叶到根淘汰，Mamba state 可以从任意节点淘汰）。
  - **最关键的差别（本步的核心判断）**：llama.cpp 的 checkpoint 是**按位置**打的——"结尾往前 N 个 token"，和 prompt 内容无关；SGLang 是**按内容**打的——radix tree 的节点边界就是共享前缀的边界。所以在"长 system + 短 user"这种前缀高度重合的负载上，SGLang 天然会把点打在 system 结束处，而 llama.cpp 只能打在结尾往前一个 ubatch。**同样是"不能裁只能快照"，快照放在哪里决定了一切。**
  - **统一/合并设计的澄清**：SGLang 把内存池拆成 Mamba pool（request 级分配）和 KV cache pool（token 级分配），用 `HybridReqToTokenPool` 绑定生命周期、`HybridLinearKVPool` 做 layer id 映射。要说清**什么变了**（分配粒度、淘汰策略）和**什么没变**（全注意力那 16 层的 KV 语义完全照旧）。
  - **spec decoding 的处理**：每个 draft token 一个独立 Mamba cache slot，接受后把最后一个被接受的 slot 提升为主 state。这与第九步 `n_rs_seq` 那个"为 draft 服务的浅回滚环"是同一类思路的两种实现，值得并置一句。
  - **一句克制的判断**：不要写成"SGLang 完胜"。llama.cpp 面对的是单机、单并发、要在 Metal 上跑、还要能被 Ollama 静态编译进去的约束，两个点的位置策略在那个约束下是合理的工程折衷。这句话能让对比显得可信。
- **参考资源**：
  - Hybrid Models Meet SGLang: More than Full Attention（PyTorch Blog）：https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/
  - `mamba_radix_cache.py`：https://github.com/sgl-project/sglang/blob/385903b0acd69455cb688b5cb5e3afcc0fd91598/python/sglang/srt/mem_cache/mamba_radix_cache.py
  - PR #14792（mamba radix cache for overlap scheduler）：https://github.com/sgl-project/sglang/pull/14792
  - Feature tracking issue #12867：https://github.com/sgl-project/sglang/issues/12867

### 第十二步：还没走的路

- **从何推导**：第十一步说明了"内容寻址"才是更根本的解法。回到本地这条链路，有没有办法把 llama.cpp 已经写好的那条消息边界路径**接通**？
- **深度层级**：理解复现
- **目标**：给出比 `num_batch=64` 更根本的几条路，并诚实标注它们目前的状态（我还没做 / 需要上游改动 / 大概率不划算）。
- **方法**：概念框架 + 源码依据
- **内容**：
  - **路子一：让请求带上 `message_delimiters`**。llama-server 的 `/completion` **已经接受**这个字段（第九步已引），只要 Ollama 在 `llamaServerCompletionRequest` 里加上它，`is_user_start` 那条路径就能通电，点会直接打在最后一条 user 消息的起点——也就是 system 的结尾。那样短请求的代价就真的只剩短 user 本身，而且和 `num_batch` 无关，冷启动也不用付小 batch 的税。这是一个可以给 Ollama 提的 PR。
  - **路子二：绕过 Ollama，直接用 llama-server 的 `/v1/chat/completions`**。Ollama 自己也有走这条路的分支。结构化 messages 进去，delimiters 由 llama-server 自己算。代价是要自己管模型加载和生命周期——对一个每天要用几百次的输入法来说，这个代价不小。
  - **路子三（存疑，需实测）**：`n_rs_seq` 目前只由 speculative 配置驱动、深度等于 `draft.n_max`。如果它能被独立配置到"覆盖一句 user 的长度"，短请求就是**精确回滚**而非重放。但显存代价是线性的（`n_rows = mem_size * (1 + n_rs_seq)`），一句 100 token 的 user 要 100 份 recurrent state 快照——**大概率不划算**。这条要写清楚"为什么看起来诱人但很可能不成立"，而不是含糊带过。
  - **上游状态**：hybrid/recurrent 的 checkpoint 恢复在 llama.cpp 上仍是活跃问题（issue #22384、#24055），这套机制本身还在演进，值得持续跟。
- **参考资源**：
  - https://github.com/ggml-org/llama.cpp/issues/22384
  - https://github.com/ggml-org/llama.cpp/issues/24055
  - https://github.com/ggml-org/llama.cpp/discussions/19264

---

## 推荐资源

### 官方文档

- Qwen3.8-27B model card：https://huggingface.co/Qwen/Qwen3.8-27B
- Qwen3.8-27B SGLang cookbook：https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B
- Qwen3-Next Usage（SGLang）：https://docs.sglang.io/basic_usage/qwen3.html
- llama.cpp server 参数说明（`-ctxcp` / `-cms` / `-b` / `-ub`）：https://github.com/ggml-org/llama.cpp/blob/7e4c0a96880dae4fc4268ad441f8a6446bd5460a/common/arg.cpp#L1687-L1704

### 代码仓库

- llama.cpp checkpoint 打点与恢复（Ollama 当前 pin 的 tag `b10434`）：https://github.com/ggml-org/llama.cpp/blob/7e4c0a96880dae4fc4268ad441f8a6446bd5460a/tools/server/server-context.cpp#L3464-L3483
- llama.cpp recurrent memory 的 `seq_rm` 语义：https://github.com/ggml-org/llama.cpp/blob/7e4c0a96880dae4fc4268ad441f8a6446bd5460a/src/llama-memory-recurrent.cpp#L150-L200
- Ollama 拉起 llama-server 的参数拼装：https://github.com/ollama/ollama/blob/e5a81899d014a847a08d47393351908b53d74008/llm/llama_server.go#L585-L591
- SGLang MambaRadixCache：https://github.com/sgl-project/sglang/blob/385903b0acd69455cb688b5cb5e3afcc0fd91598/python/sglang/srt/mem_cache/mamba_radix_cache.py

### 社区文章

- Hybrid Models Meet SGLang: More than Full Attention：https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/
- Gated Delta Networks: Improving Mamba2 with Delta Rule：https://arxiv.org/abs/2412.06464
- Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention：https://arxiv.org/abs/2605.22791

### 拓展阅读与对比分析

- llama.cpp Discussion #19264：recurrent 模型的 partial prompt cache reuse 提案，本文第四步概念的原始出处：https://github.com/ggml-org/llama.cpp/discussions/19264
- llama.cpp PR #20288：尾部双 checkpoint（`4 + n_ubatch` / `4`）的设计讨论：https://github.com/ggml-org/llama.cpp/pull/20288
- llama.cpp Issue #22384 / #24055：hybrid/recurrent checkpoint 恢复的已知缺陷：https://github.com/ggml-org/llama.cpp/issues/22384 、https://github.com/ggml-org/llama.cpp/issues/24055
- SGLang Issue #12867：Hybrid Linear LLM 支持的总 tracking issue，可以看到 hicache、page size > 1、确定性推理这些还没做完的方向：https://github.com/sgl-project/sglang/issues/12867
- LMCache 对 hybrid attention 模型的处理：https://docs.lmcache.ai/mp/hybrid_models.html
- ik_llama.cpp Issue #1762：另一个 fork 上同类问题的独立复现，可作旁证：https://github.com/ikawrakow/ik_llama.cpp/issues/1762

---

## 文章结构建议

- **文章类型**：**工程实践博客**。叙事主线是"我自己每天用的东西变慢了 → 拆流水线 → 提假设 → 做对照实验 → 读源码 → 落地"，源码服务于叙事而不是相反。相对纯 code-walkthrough 的三点调整：
  1. 开篇和第一步要有足够的**个人叙事密度**（为什么造这套东西、为什么必须本地、三秒半为什么不可接受）。
  2. 第八、九步的源码要**收敛**——只留参与回答"1020 从哪来"的那几处，其余用文字带过。
  3. 全文保留失败路径的**真实时间顺序**，不要事后诸葛亮地重排成"正确推理链"。垫文和 `-cms` 这两次白跑是这篇博客最有价值的部分之一。
- **建议路径**：`engineer/hybrid-prefix-cache/readme.md`
  - 放 `engineer/` 是因为叙事主线是一次完整的本地工程实践。若后续想把第十一步的 SGLang 部分扩写成主体，可改挂 `sglang/`。
- **系列归属**：暂不归入现有系列。它同时挂靠 SGLang scheduler / prefix cache 这条线和 spec decoding 那条线，交叉引用即可。
- **预计章节**：

  1. `#` 标题 + 开篇。先亮 ablation 表（`num_batch` 从 1024 到 64，短请求 3.95s → 0.91s），再交代动机——我在 macOS 上的语音输入法是 ASR + LLM 润色的本地流水线，前天晚上把润色那段从 Qwen3.6 换成 Qwen3.8 之后，每句话要等三秒半。坦诚承认此前对 linear attention 的 cache 语义基本没概念，是被逼着现学的。路线图 4 条以内。
  2. `##` 我的语音输入法是怎么搭的（第一步）
  3. `##` 全注意力前缀缓存的可裁剪性（第二步）
  4. `##` 线性注意力的状态为什么不能裁（第三步）
  5. `##` Checkpoint：快照与重放（第四步）
  6. `##` Qwen3.8（第五、六步，两个 `###`：全貌 / cache 特征）
  7. `##` 一条 3.5 秒的短请求（第七步，**驱动问题在此**）
  8. `##` Ollama 到底在跑什么（第八步）
  9. `##` 点打在哪里：llama.cpp 源码（第九步，`###` 子节按打点路径切分：消息边界 / 尾部偏移 / 最小间距 / 恢复判据 / 没被用上的 `n_rs_seq`）
  10. `##` 三次尝试（第十步，`###` 子节：固定垫文 / `-cms` / `num_batch`；子节标题直接编码对应的自由度）
  11. `##` 换一种打点方式：SGLang 的 MambaRadixCache（第十一步）
  12. `##` 还没走的路（第十二步）
  13. `##` 致谢

- **一句话总结**（可放在路线图之后或全文结尾）：以前，稳定 system 拼在前面，前缀缓存只付短 user 的钱；Qwen3.8 上稳定 system 仍然对得上，但恢复粒度是一个 batch——要让一条条短请求变快，改的是 `num_batch`，不是把 system 写得更长。

- **图表建议**：
  - mermaid：语音输入流水线全景（触发 → ASR → 润色 LLM → 上屏），标出哪一段是这篇文章要优化的。放在第一步。
  - mermaid：Ollama → llama-server 子进程 → `/completion` 的链路图，标出 `-b/-ub` 和"消息边界信息在此丢失"两个点。放在第八步。
  - mermaid 或表格：一条 prompt 上两个 checkpoint 的位置示意（`end − 4` 与 `end − num_batch`），以及新请求到来时哪个失效。放在第九步。
  - markdown 表格：`num_batch` ablation 表、`num_batch` 取值的替代方案对比表、llama.cpp checkpoint vs SGLang MambaRadixCache 的多维对比表。
  - **禁止** ASCII 字符画。

---

## 写作前需要 chenyang 确认的事项

第一步的设计推演里有几处我无法从已知背景推出，写作时请补充，或明确说"略过"：

1. ASR 那一段的具体形态：Qwen3-ASR 跑在什么上面（MLX / whisper.cpp / 别的），本地还是别处
2. 输入法层面怎么触发：快捷键、常驻监听、还是别的
3. 从触发到出字的端到端延迟预算——这个数字决定"3.5 秒"到底有多不可接受
4. 从 Qwen3.6 换到 Qwen3.8 的真实动机（润色质量不够？还是就想试试新模型？）
5. 那层日志代理是自己写的还是现成的
6. Qwen3.6 时期的短请求延迟体感——如果记得，第六步可以补一句对照；不记得就不要编

---

## 自检清单确认

- [x] 每个步骤都标注了"从何推导"，推导链从"我押在长 system 能被 cache 住这一个假设上"一路贯到"改 `num_batch`"
- [x] 步骤顺序：负载画像（一）→ 概念（二~四）→ 模型（五~六）→ 驱动问题（七）→ 代码（八~十）→ 对比与展望（十一~十二）。概念严格早于模型，模型严格早于所有代码
- [x] 驱动问题已显式标注，位置在读者拿到负载画像、概念框架与模型特征之后
- [x] 需要独立展开的核心概念已逐步骤标注（两段式流水线的设计权衡、system prompt 作为先验的载体、KV 可寻址性、Gated DeltaNet 状态更新、快照代价模型、三个自由度、混合内存木桶效应、位置寻址 vs 内容寻址）
- [x] 做了广泛视野搜索：不止用户给的 Ollama 场景，还覆盖了 llama.cpp 上游 PR/issue/discussion、SGLang MambaRadixCache、LMCache、ik_llama.cpp fork
- [x] 模型步骤区分了"全貌"（第五步）与"计算/cache 特征"（第六步）
- [x] 第十步展示了 baseline → 中间方案 A/B → 最终方案的演进路径，并标注了需要替代方案对比表的决策
- [x] 三次尝试合并为一个 `##`（同一设计自由度的三个侧面）
- [x] 没有设立独立的"概念/约束映射"步骤，映射要求融入第十、十一步的行文
- [x] 所有外部代码引用均带 commit hash，未引用 main 分支行号
- [x] 未引用任何 `[Pending Review]` 文章
- [x] 博客化要求已落到具体条目：叙事密度、源码收敛、保留真实失败顺序
