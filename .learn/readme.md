# /learn Agent Proposal

## 项目背景

Awesome-ML-SYS-Tutorial 是我从事 ML System 研究以来的完整学习笔记合集。从 2024 年 8 月在科研中使用 SGLang 开始，到现在超过 5K Stars，这个 repo 记录了一个 ML SYS researcher 从入门到前沿的全部历程。它的内容覆盖 RL infra、在线/离线推理系统、以及 AI Infra 基本功等等，后续还要写入我对搭建 Agent Infra 的思考。

README 中列举的所有文章，正序阅读是研究领域的技术前沿，倒序阅读则是从零开始一步步学习 ML System 的完整路径。这不只是一堆零散笔记的堆叠，而是一套有内在逻辑的知识体系——后面的文章自然地建立在前面文章的基础之上，形成了一条可复现的学习曲线。我基于这套知识体系，开发了 /learn agent，旨在更好地理解、利用并且扩充这套知识体系。

这套笔记体系对我个人的技术成长帮助极大，但它目前仍然是一个被动的笔记仓库——文章写好后，累积在此处，但是内在的知识逻辑在我脑中，读者需要自行摸索学习路径。随着文章数量增长到近百篇，涵盖数十个 topic，/learn agent 的目标就是把这套"存在于脑中的学习系统"外化为一个可交互的工具——能够理解这个 repo 的知识结构，理解作者的学习风格，然后在学习新内容、撰写新文章、回顾旧知识时提供有针对性的辅助。它不只是一个学习助手，而是希望把"如何正确地学习 ML System"这件事本身工具化，利用 AI Agent 的力量，极大扩展我自身的学习效率和知识范畴。

## 学习模式

这套笔记体系有两条核心原则：

1. **概念 → 模型 → 代码**：先建设知识框架、再结合具体工程代码。辅助理解的代码来自真实生产系统，不是仅仅是教学示例。
2. **递进推导，不是平铺罗列**：每一层概念必须从前一层自然推导而来，形成一条逻辑链，而不是独立罗列的 checklist。约束从机制推导、推论从约束推导、工程设计从推论推导——读者在任何一个节点都能追溯到"为什么是这样"。

接下来我们来讨论具体的写作模式。

### 动机驱动 + 概念引入

几乎所有文章都以一个真实的工程需求或困惑作为起点，而非"今天来学习 X 概念"。譬如：

- `rlhf/sys-design/readme-1.md` 开头："因为工作需要，最近终于得空能够再次深入去学习思考主流 RL 框架的系统设计。"——从工作需要出发，而非学术兴趣。
- `torch/cuda-graph/readme.md` 开头："前几天和标哥在研究 torch-mem-savor 的时候发现自己对 CUDA Graph 知之甚少"——从一次具体的协作中发现知识盲区。
- `torch/torch-distributed/readme.md` 开头：完整记录了怜悯告诉他"你只需要学习 torch.dist"的对话，甚至包含了自己把 `torch.dist` 误认为计算距离的笑话——学习的触发点是一个具体的开发任务（给 OpenRLHF 写 weight_update 接口）。
- `sglang/scheduler/readme.md` 开头："想要系统性学习 SGLang Scheduler 的想法已经有一年了...这个事情必然是悬在我心上的，否则经常看到在 SGLang 的各种技术讨论群里，大家聊的内容有许多我都毫不了解，让人产生了深深的技术焦虑。"——动机来自社区讨论中的知识焦虑。

### 建立系统性认知框架

在进入代码之前，先用大篇幅建立概念性的理解框架。这个框架通常包含：分类对比、设计权衡、历史脉络。譬如：

- `rlhf/verl/readme.md`：在分析 verl 代码之前，用整段篇幅从 single-controller vs multi-controller 的设计哲学讲起，从通讯压力、鲁棒性、SPMD 模式层层推进，最后才说"理解了这些概念后，我们可以正式进入 veRL 的 introduction 了"。
- `rlhf/sys-design/readme-2.md`：在分析 FSDP 代码之前，先从 DP → DDP → DeepSpeed Zero Stage 1/2/3 逐级推演，建立完整的"通讯换显存"谱系，明确标注"总的来说，zero 就是通讯换显存的典型代表"之后，才进入 FSDP 本身。
- `rlhf/slime/fp8/readme.md`：在讨论 FP8 在 RL 中的应用之前，先从 Tensor Core 硬件演进（Volta → Hopper → Blackwell）讲起，再讲 FP8 格式（E4M3 vs E5M2）、scale 选择、量化策略，建立完整的低精度计算知识体系后，才进入"FP8 + RL"的主题。
- `rlhf/OpenRLHF/readme.md`：Reward、Advantage、Actor Loss、Critic Loss 四个计算步骤，每一步都先给数学公式建立理论理解，再给完整的 Python 代码实现。

### 代码落地——永远是真实系统的代码

概念框架建立后，转入代码分析。代码永远来自真实的生产框架（SGLang、verl、slime、Megatron），带有具体的 GitHub commit 链接和文件路径。譬如：

- `sglang/code-walk-through/readme-CN.md`：每个组件都附带精确到行号的 GitHub 链接，如 `TokenizerManager` 链接到 `tokenizer_manager.py#L88`。
- `rlhf/verl/multi-turn/code-walk-through/readme.md`：明确基于 commit `76f63cffa5` 进行分析，保证读者可以精确对照源码。
- `rlhf/sys-design/readme-1.md`：直接贴出 slime 的 `update_weights_from_tensor` 完整代码，逐行分析每个步骤的显存行为。

### 从知识概念到具体实现的过渡

概念轨到代码轨的过渡总是用一个明确的转折信号：

- "我们还是回到主线上"（`rlhf/sys-design/readme-1.md`）
- "有了这些认识，我们再来看"（`rlhf/verl/readme.md`）
- "有了这个直观的理解，我们回顾下"（`rlhf/verl/readme.md`）
- "好了，有了这些基础"（`rlhf/sys-design/readme-1.md`）

## 叙述逻辑

### 开篇结构

文章开头有一套稳定的三段式结构：

1. 个人动机/情感：总是坦诚交代自己的写作动机，经常包含个人感受，譬如：

   - `rlhf/partial-rollout/readme.md`："苦于工作繁琐，一直没有空拜读。正好最近去西雅图出差，在 LAX 往返 SEA 的路上终于有时间虔诚拜读这一工作。越读起来，越有一种文人相赏，难逢知音的感觉。"
   - `rlhf/verl/readme.md`："众所周知，我一直在 SGLang team 负责端茶倒水 + RLHF。"
   - `rlhf/rl-walk-through/part-1.md`："出于我个人的学习习惯，我对理论不感兴趣，好在对于 ML SYS researcher 而言，这样的直观已然可以带来巨大的帮助。"

2. 致谢：几乎每篇重要文章都有详细的致谢列表，标明每个贡献者的学校/公司，譬如：

   - `rlhf/sys-design/readme-1.md`："照理，感谢参与本文档讨论和撰写的所有朋友们：zhuoran yin（CMU），changyi yang（CMU）..."，甚至加了"排名按照微信群的成员顺序 😂"。
   - `rlhf/slime/code-walk-through/readme.md`："Mao Cheng @ Meta, Zhuoran Yin @ CMU, Ji Li @ Ant Group..."

3. 文章路线图：在正文开始前，概述本文将讨论什么，譬如：

   - `torch/cuda-graph/readme.md`："所以今天的文档尝试理解这些内容：1. 什么是 CUDA Graph；2. CUDA Graph 在推理中的应用；3. 为什么 CUDA Graph 在训练中很少使用..."
   - `sglang/scheduler/readme.md`："这篇文章会 follow 以上三篇文章的思路，深入理解 SGLang 的调度系统"

### 复杂概念的拆解方式

我有一套特征鲜明的概念拆解策略，譬如：

1. 先下定义，再举例，最后给代码：

   - `rlhf/verl/readme.md` 中 single-controller 的讲解：先给出加粗的核心定义（"在一个复杂的工作流程中，single controller 只有一个程序负责管理..."），再用 PPO 的工作流程举例，最后才进入 veRL 的具体实现。

2. 数值化的具体例子：喜欢用具体数字来让抽象概念可感，譬如：

   - `rlhf/sys-design/readme-1.md`："假设这个参数的 size 是 `[1024, 1024]`，FSDP 的 TP size 是 4，而 SGLang 的 TP size 是 2。因此在更新参数开始前，每个 rank 上在 FSDP engine 内有 `[256, 1024]` 大小的 tensor..."
   - `rlhf/sys-design/readme-2.md`："考虑大小为 XB 的模型，这三部分会有 12X 的显存占用。"

3. 自问自答 / 设问句：

   - `rlhf/sys-design/readme-1.md`："有一个问题非常值得分享：为什么 slime 需要先将 megatron 的 model weights offload 掉再 upload 上来，直接保留在 GPU 上不行吗？"
   - `rlhf/verl/readme.md`："这里思考一个小问题，为什么不能拿着 training engine 得到的 logits 做 sampling 然后 decode，貌似也可以用去 rollout？简单来说，太慢了。"

4. `<details>` 折叠块：可选的深入内容用折叠块包裹，保持主线清晰，譬如：

   - `rlhf/OpenRLHF/readme.md` 中所有的伪代码都放在 `<details><summary>伪代码</summary>` 中。

### 文章间的交叉引用和知识链路

repo 内形成了一张密集的知识图谱，引用方式有三种：

1. 前序依赖：明确告诉读者需要先读什么，譬如：

   - `rlhf/OpenRLHF/readme.md`："实际上，我已经写过一篇前序文章：浅析主流 Alignment 算法与 NeMo-Aligner 框架，但真的理解清楚了 RLHF 的计算流还是在读了这两篇文章之后..."
   - `sglang/scheduler/readme.md`："这篇文章会 follow 以上三篇文章的思路"，引用了 sys-design 系列的三篇。

2. 同系列衔接：系列文章之间有明确的编号和衔接，譬如：

   - `rlhf/sys-design/` 系列：readme-1（权重更新）→ readme-2（FSDP）→ readme-3（Megatron）→ readme-4（MoE EP）→ readme-5（RDMA）。
   - `rlhf/verl/multi-turn/code-walk-through/` 系列：readme（初始化）→ readme-2（Rollout）→ readme-3（Make Experience）...
   - `rlhf/rl-walk-through/` 系列：part-1 到 part-8，系统覆盖 RL 基础理论。

3. 跨主题引用：不同主题之间的知识桥梁，譬如：

   - `torch/nccl/readme.md` 引用 `torch/torch-distributed/readme.md`："在 torch-distributed 的后记中已经介绍过了。"
   - `rlhf/sys-design/readme-1.md` 引用 SGLang 架构图："具体的 SGLang 架构可以参考此图"，链接到 `sglang/code-walk-through/` 的 SVG。
   - `sglang/constraint-decoding/readme.md` 引用 SGLang 官方文档和 X-Grammar 论文。

## 行文风格

1. 谦逊但自信："非常惭愧，从那之后，我也没有理解到二者的关系"（`sglang/scheduler/readme.md`）、"无所谓，我会出手！"（`torch/torch-distributed/readme.md`）。
2. 适度幽默："排名按照微信群的成员顺序 😂"、"~~便于绕开 sudo 用户直接 kill 僵尸进程~~"（`engineer/how-to-use-docker/readme.md`）。
3. 敢于表达主观判断："让我惊讶的是..."、"我个人觉得..."、"其中的酸甜苦辣，自不必多说"。
4. 使用加粗和引用块来标记关键结论。


## 学习深度

任何领域的研究都可以做到深不可测，对于学习一个概念的深度需要适度控制。学习深度按"与我工作目的的相近程度"来分层，核心系统学到能修改扩展，相邻系统学到能理解复现，纯理论学到能建立直觉。

1. 能够修改和扩展核心系统

对于 SGLang、verl、slime 这些自己直接参与的项目，学习深度达到了源码级别，能够理解设计权衡并提出改进方案，譬如：

- `rlhf/sys-design/readme-1.md`：不只分析了 verl 的权重更新，还对比分析了 slime 的桶更新策略，最后横向对比三种权重更新方式，给出自己的判断："知易行难...权重更新接口开始的。RL 系统无非就是需要把 inference engine 接进去...可是其中的心酸滋味，自然只有真的打磨过，才能体会。"
- `sglang/code-walk-through/readme-CN.md`：完整追踪了一个请求从输入到输出的全链路（Server → TokenizerManager → Scheduler → TpModelWorker → ModelRunner → AttentionBackend → DetokenizerManager），每个组件都有初始化和核心函数分析。
- `rlhf/slime/spec/readme.md`：不只理解 speculative decoding 的原理，还设计并实现了 online SFT 训练 draft model 的方案，包含完整的训练目标构造、训练输入构造和训练流设计。

2. 能够理解和复现相邻的基础设施

对于 NCCL、CUDA Graph、FSDP 等相邻基础设施，学到理解其设计思想和使用方式，能复现关键操作，但不深入实现细节。

- `torch/nccl/readme.md`：覆盖了 NCCL 的概念、操作类型、Ring/Tree Algorithm、通信协议，以及两台集群的 `nvidia-smi topo -m` 输出分析。止步于"能读懂拓扑输出并做出正确的开发决策"，没有深入 NCCL 内核实现。
- `torch/cuda-graph/readme.md`：理解了 CUDA Graph 为什么在推理中大量使用（确定性操作流）、为什么在训练中很少使用（动态性），以及 torch-memory-savor 如何保护 CUDA Graph（虚拟地址与物理内存分离管理）。止步于理解原理，没有实现自己的 CUDA Graph 管理器。
- `torch/torch-distributed/readme.md`：通过实际代码练习掌握了 `init_process_group`、`all_reduce`、`broadcast` 等 API，但没有深入 PyTorch 分布式通信的底层实现。

3. 建立直觉和概念框架

对于 RL 理论、量化理论等，明确止步于"直觉层面"，譬如：

- `rlhf/rl-walk-through/part-1.md`：开篇直说"出于我个人的学习习惯，我对理论不感兴趣，好在对于 ML SYS researcher 而言，这样的直观已然可以带来巨大的帮助"。保留了"平凡到看似愚蠢的定义"，但目标是建立直觉而非证明定理。
- `rlhf/partial-rollout/readme.md`：对 Kimi K1.5 的技术报告做了全面但选择性的解读——RL Recipe、采样策略、Long2Short 训练都有覆盖，但最浓墨重彩的是"RL Infra"部分（Partial Rollout），因为这直接关联作者自己的系统工作。
- `sglang/constraint-decoding/readme.md`：讲清了 constraint decoding 的概念、基本原理、X-Grammar 的算法优化和系统优化，止步于"能使用 SGLang 的 structured output 功能并理解其原理"，没有去实现 PDA。止步于建立直觉而非证明定理。

4. 深度一致性分析

深度并不是在所有 topic 之间一致的，而是沿着一条清晰的"距离梯度"递减，譬如：

```
自己开发的系统 (SGLang/verl/slime/SGLang-Omni) → 源码级，能修改扩展
                ↓
依赖的基础设施 (FSDP/NCCL/CUDA Graph) → 原理级，能正确使用
                ↓
相关的算法理论 (PPO/RL Theory/量化理论) → 直觉级，能指导系统设计
                ↓
读到的论文 (Kimi K1.5/SWE-Bench) → 摘要级，提取对系统工作有用的信息
```

作为 ML SYS researcher，系统实现是核心战场，理论是弹药库，论文是情报来源。当然，我后续可能会对学习的深度进行调整，也会加入更多类型的系统，甚至包括自己开发的 Agent 系统。

## /learn Agent 设计方案

基于以上分析，/learn agent 的核心定位是：**一个完全理解我学习风格的 ML System 学习助手**。它不是通用的知识问答工具，而是基于这个 repo 的知识体系，帮助我高效学习新 topic 并保持风格一致性的专属工具。

### 硬约束

以下约束条件优先级最高，贯穿所有子命令：

1. **中文优先**：所有写作、计划、审查均以中文进行。我的工作流是先完成中文版本，再翻译为英文。/learn agent 只需要考虑中文内容的生成和审查，翻译是 /learn-review 的独立步骤。
2. **排除 [Pending Review] 文章**：参考 README.md 和 README-cn.md 中的内容，标记为 [Pending Review] 的文章并非由我本人完成，**绝对不能**作为构建 /learn agent 的风格参考或知识来源。
3. **源码引用必须带 commit hash**：所有对外部代码的引用必须包含具体的 commit hash（如 `https://github.com/org/repo/blob/<commit>/path/to/file.py#L123`），而非引用 main 分支上的行号。引用 main 上的行号会随代码变更迅速失效。
4. **信息获取不设限制**：agent 拥有最高的信息获取权限——可以读取外部 PR、搜索网页、访问 GitHub 代码库、获取官方文档。读取的信息越全面越好，没有被明令禁止的权限。

### /learn-plan

**输入**：

单独输入一个新的学习计划，譬如："我想学习 FlashAttention 的实现"；并且可能还有一份未完成的草稿，譬如：transformers/omni/readme.md

**输出**：一份符合我风格的学习大纲，包含：

- **动机定位**：这个 topic 与我的现有知识体系（repo 中已有内容）的关系。它填补了哪个知识空白？与哪些已有文章产生关联？
- **前置知识检查**：基于 repo 中已有的文章，列出学习这个 topic 之前应该回顾的内容（带 repo 内链接）。
- **学习路线图**：按照我的"概念框架 → 代码落地"双轨模式，规划学习步骤。每步标注对应的深度层级（修改扩展 / 理解复现 / 建立直觉）。
- **推荐资源**：基于我的历史引用习惯（知乎文章、官方文档、GitHub 代码），推荐最适合的学习资源。
- **草稿完成情况**：如果输入内容包含草稿，则基于已经完成的草稿部分，分析这部分内容对于整个学习计划的完成度。并且进一步为草稿未完成的部分制定新的学习大纲。如果输入内容不包含草稿，则基于学习计划生成新的学习大纲。

### /learn-write

**输入**：

/learn-plan 生成的学习大纲，并且可能还有未完成的草稿。

**输出**：

- **完善内容**：基于已有的草稿或者学习大纲，完成剩余部分的写作；
- **风格保持**：完成的文章内容必须符合我的行文习惯（开篇三段式、概念拆解方式、过渡句式、深度把握）；
- **交叉引用建议**：基于 repo 中已有内容，建议应该在哪里添加交叉引用，以及添加什么内容；
- **深度校准**：根据 topic 在你的"距离梯度"中的位置，建议当前内容是否过深或过浅。

**质量保障**：agent 输出完整草稿后，由我逐字逐句审阅并修改。对于涉及源码分析的部分，agent 必须先读完相关源码才能写作，且所有源码引用必须附带具体 commit hash 的链接（参见硬约束第 3 条）。agent 不应等待逐段确认，而是一次性输出完整内容，由我事后审阅。

### /learn-review

**输入**：

一篇已经完成的文章草稿，并且可能还有其对应的学习大纲。

**输出**：

- **内容检查**：检查草稿（主要是中文）是否存在符合我写作风格，是否完成了我的学习大纲，是否存在错误的引用；
- **内容翻译**：在确保内容检查毫无问题的情况下，得到我的允许后，将草稿翻译为英文。注意格式要严格一致；

### /learn-add

**输入**：

一篇或多篇已发布文章的路径（如 `torch/cuda-graph/readme-3.md`）。

**输出**：

- **元信息提取**：自动读取文章内容，提取 title、topics、depth、references、series 等元信息，生成 knowledge-graph.json 条目；
- **引用关系同步**：更新被引用文章的 `referenced_by` 字段，更新系列信息；
- **确认后写入**：展示将要添加的条目，经我确认后才写入 knowledge-graph.json。

**收录条件**：只收录我本人的文章，且必须已在 README.md 中列出、未标记 [Pending Review]。`/learn-write` 和 `/learn-review` 不会自动触发知识图谱更新。

### 交互方式

### 主入口

在 Claude Code 中通过 skill 调用，譬如：

```
/learn-plan 我最近在完成 SGLang Omni 框架支持 Fish Audio S2 模型的过程中，发现同时为 transformers 和 codex loop 两个部分同时开启 CUDA Graph 后，性能有了显著的提升。先前我对 CUDA Graph 的理解程度不深，我现在希望能够基于这个 PR https://github.com/sgl-project/sglang-omni/pull/153 和我已有的知识体系，去进一步学习 CUDA Graph 的原理和实际实现。注意，这个过程需要先对 Fish Audio S2 Pro 这个模型进行架构分析，理解为什么会存在快慢两个 CUDA Graph 的需求。

/learn-write 我一级完成了 omni 模型的学习草稿 transformers/omni/readme.md，我希望你参考学习计划 transformers/omni/learn-plan.md 帮我完成剩余部分的写作，重点要强调我在开发 SGLang Omni 的过程中，对不同组件在资源组上的排列方式的思考。

/learn-review 根据 omni 路径下的 readme.md 和 plan.md，检查中文文章的完成程度。

/learn-add torch/cuda-graph/readme-3.md
```

### 输出形式

- **Markdown 文件**：/learn plan 和 /learn write 的输出为 Markdown 格式，可以直接作为文章骨架。默认按照 topic 在 repo 下创建合理的新路径，并且自动保存到该路径下。
- **交互式追问**：所有功能都支持追问，例如 "/learn plan" 输出大纲后，可以继续 "把第三步展开"，也可按照反馈，继续优化大纲。

### 上下文管理：`.learn/` 目录结构

```
.learn/
├── readme.md                    # 本 proposal
├── skill.md                     # /learn skill 的 prompt 定义（四个子命令的 system prompt 和工具配置）
├── index/
│   ├── knowledge-graph.json     # repo 知识图谱（文章元信息 + 引用关系 + topic 标签）
│   └── style-guide.md           # 从第一部分分析中提炼的风格指南（机器可读版本）
├── templates/
│   ├── code-walkthrough.md      # code walk through 类文章的模板
│   ├── sys-design.md            # 系统设计分析类文章的模板
│   ├── paper-reading.md         # 论文阅读笔记类文章的模板
│   └── tutorial.md              # 教程类文章的模板
└── config.md                    # agent 配置（深度偏好、引用风格等可调参数）
```

1. `knowledge-graph.json`

统一存储 repo 中所有文章的元信息和相互关系，通过扫描 repo 自动生成，也支持手动补充。提取的信息包括：

- 文章标题、路径、语言（中文/英文/双语）
- 文章间的引用关系（通过 markdown 链接分析）
- topic 标签（通过标题和内容关键词提取）
- 深度层级（modify-extend / understand-reproduce / intuition）
- 发布状态（published / pending-review / draft）
- 系列归属和顺序
- 前置依赖
- 外部链接（知乎、GitHub 代码链接）

示例结构：

```json
{
  "articles": [
    {
      "path": "rlhf/sys-design/readme-1.md",
      "title": "RL 系统深思：深入理解权重更新机制",
      "topics": ["weight-update", "verl", "slime", "co-locate", "handle-tuple"],
      "depth": "modify-extend",
      "status": "published",
      "references_to": ["sglang/code-walk-through/readme-CN.md"],
      "referenced_by": ["rlhf/sys-design/readme-2.md", "sglang/scheduler/readme.md"],
      "series": "rl-sys-design",
      "series_order": 1,
      "prerequisites": ["rlhf/OpenRLHF/readme.md"],
      "external_links": {
        "zhihu": "https://zhuanlan.zhihu.com/p/1925210722704531547"
      }
    }
  ],
  "series": {
    "rl-sys-design": {
      "name": "RL 系统深思",
      "articles": ["readme-1.md", "readme-2.md", "readme-3.md", "readme-4.md", "readme-5.md"]
    }
  }
}
```

2. `style-guide.md`

将第一部分的分析成果编码为机器可用的风格指南，包含：
- 开篇结构要求（动机 → 致谢 → 路线图）
- 概念拆解的标准流程（定义 → 数值例子 → 代码）
- 过渡句式的模板库
- 深度层级判定规则
- 交叉引用的触发条件

### skill.md 结构

`skill.md` 是 /learn agent 的 prompt 定义文件，定义四个子命令各自的 system prompt 和行为。大致结构如下：

```
skill.md
├── 公共部分
│   ├── 角色定义：你是 chenyang 的个人 ML System 学习助手
│   ├── 硬约束注入：中文优先、排除 Pending Review、commit hash 引用、信息获取不设限
│   ├── 风格指南引用：读取 .learn/index/style-guide.md
│   └── 知识图谱引用：读取 .learn/index/knowledge-graph.json
│
├── /learn-plan
│   ├── 触发条件：用户输入包含学习主题描述
│   ├── 工具权限：Glob, Grep, Read, WebFetch, WebSearch（用于搜索 repo 和外部资源）
│   ├── 执行流程：
│   │   1. 扫描 knowledge-graph.json 定位相关已有文章
│   │   2. 若用户提供了草稿路径，读取草稿分析完成度
│   │   3. 若用户提供了外部链接（PR、文档），获取并分析
│   │   4. 生成学习大纲（动机定位 → 前置知识 → 路线图 → 推荐资源）
│   │   5. 将大纲保存为 learn-plan.md 到对应目录
│   └── 输出格式：Markdown 文件
│
├── /learn-write
│   ├── 触发条件：用户输入引用了 learn-plan.md 或学习大纲
│   ├── 工具权限：全部工具（需要读源码、读外部代码、写文件）
│   ├── 执行流程：
│   │   1. 读取 learn-plan.md 和已有草稿
│   │   2. 对涉及源码分析的部分，先读取相关源码（必须完成后才能写作）
│   │   3. 一次性输出完整文章草稿
│   │   4. 对输出进行三维度自动检查（双轨、叙事、深度）
│   └── 输出格式：Markdown 文件，保存到对应目录
│
├── /learn-review
│   ├── 触发条件：用户输入引用了已完成的草稿
│   ├── 工具权限：Glob, Grep, Read（检查阶段）；Write（翻译阶段，需用户明确允许）
│   ├── 执行流程：
│   │   1. 读取草稿和对应的 learn-plan.md
│   │   2. 检查：风格合规、大纲完成度、引用正确性
│   │   3. 输出检查报告
│   │   4. 若用户要求翻译，在确认检查无问题后，生成格式严格一致的英文版本
│   └── 输出格式：检查报告（终端文本）+ 可选的英文翻译文件
│
└── /learn-add
    ├── 触发条件：用户主动调用，传入已发布文章路径
    ├── 工具权限：Glob, Grep, Read, Write
    ├── 执行流程：
    │   1. 验证文章满足收录条件（本人文章 + 在 README 中 + 非 Pending Review）
    │   2. 读取文章内容，提取元信息（title, topics, depth, references 等）
    │   3. 更新被引用文章的 referenced_by 和系列信息
    │   4. 展示条目内容，经用户确认后写入 knowledge-graph.json
    └── 输出格式：确认提示 + 写入 knowledge-graph.json
```

### 风格保持机制

1. System Prompt 中注入风格指南

`/learn-plan` 和 `/learn-write` 的 system prompt 包含 `.learn/index/style-guide.md` 的完整内容，确保 agent 的每次输出都符合风格约束。

2. 基于模板的输出生成

对于 /learn plan 和 /learn write，输出基于 `.learn/templates/` 中的模板生成。模板本身来自对 repo 中文章结构的抽象。

例如，code-walkthrough 模板可能包含：

```markdown
# [框架名] [模块名] Code Walk Through

[1-2 句个人动机]

[致谢列表]

本文基于 [commit hash] 进行分析。

## 核心架构
[架构图 + 模块分解]

## [模块 1]
### 概念
[先建立理解框架]

### 代码分析
[贴出关键代码，逐步分析，逐级向下引出]

## [模块 2]
...
```

当然，考虑到本人行文风格的灵活程度，写作风格是明确的，而模板仅供参考。

### 三维度自动检查

`/learn-write` 在生成或检查内容时，自动进行三个维度的验证：

1. **双轨检查**：是否同时包含概念框架和代码分析？代码是否来自真实系统？概念框架是否在代码之前建立？
2. **叙事检查**：开篇是否有动机？段落间是否有过渡？是否有交叉引用？内容是否符合我的行文习惯？
3. **深度检查**：当前 topic 在"距离梯度"中的位置是什么？内容深度是否匹配？
4. **递进推导检查**（新增）：每一节的开头是否显式说明了它从前一节的什么推导而来？约束/概念映射是否融入行文而非独立成表？概念章节是否有足够深度（子步骤 + 类比 + 总结判断）？文章是否有明确的驱动问题？

### 可扩展性

当其他人向 repo 贡献文章时，/learn 会将新的内容视作草稿，后续经由我本人的确认和修改，才会正式被发布。

- **风格一致性审核**：`/learn-review` 检查新文章是否符合 repo 的整体风格。
- **知识图谱更新**：通过 `/learn-add` 命令手动触发，只收录我本人在 README 主页正式发布（无 [Pending Review] 标记）的文章。`/learn-write` 和 `/learn-review` 不会自动更新知识图谱。更新原则如下：
  - **需要更新**：新文章正式发布（我本人的文章 + 已在 README.md 中列出 + 未标记 [Pending Review]） → 通过 `/learn-add` 新增条目。
  - **需要更新**：文章文件被删除或路径变更 → 通过 `/learn-add` 同步删除或更新对应条目及引用字段。
  - **需要更新**：文章内容发生实质性变化（交叉引用、系列归属、深度层级、外部链接） → 通过 `/learn-add` 更新对应字段。
  - **不需要更新**：仅 README 排版调整，文章文件本身未变。
  - **不需要更新**：文章的非结构性修改（typo、润色、补充细节），不涉及引用关系或元信息变化。

### `.learn/` 作为独立可复用模块

`.learn/` 目录应该完全独立于 repo 之外，其他人 clone 后即可完全复用。所有有关 /learn 的信息和实现都该记录在 `.learn/` 目录下。

### 实现路径建议

为确保落地质量，建议分三期实现：

**Phase 1（最小可用：验证 /learn-plan）**：

- 编写 `skill.md`，先只包含 /learn-plan 子命令的完整 prompt
- 手动编写 `style-guide.md`（基于本 proposal 的学习模式分析）
- 通过脚本半自动生成 `knowledge-graph.json`
- 实现 `/learn-plan` 功能，实际跑几个 topic 验证效果
- 根据验证反馈迭代 skill.md 和 style-guide.md

**Phase 2（核心写作能力）**：

- 在 skill.md 中加入 /learn-write 子命令
- 建立文章模板（code-walkthrough、sys-design、paper-reading、tutorial），模板从已验证的 plan 输出中抽象
- 实现 `/learn-write` 功能
- 能够在得到我同意的情况下，自动去更新 `knowledge-graph.json`、`style-guide.md` 和 `templates/` 的内容

**Phase 3（审查与翻译）**：

- 在 skill.md 中加入 /learn-review 子命令
- 实现 `/learn-review` 的内容检查功能
- 实现 `/learn-review` 的中译英功能（作为独立步骤，需用户明确允许后执行）

## 经验教训与改进（基于 CUDA Graph 系列文章的实际迭代）

> 以下改进来自 `torch/cuda-graph/readme-2.md` 和 `readme-3.md` 的实际写作过程。用户几乎重写了整篇文章，每次重写都暴露了 /learn 工作流的具体缺陷。这些教训必须反映到 /learn-plan、/learn-write、/learn-review 的所有环节中。

### 1. 递进推导 vs 平铺罗列（最核心的教训）

**问题**：/learn-write 生成的概念章节是 checklist 式的——"五条约束"作为独立表格罗列，"三阶段机制"用 1-2 句话概括。用户重写后，每一层概念从前一层自然推导而来：构造过程 → 约束（从构造过程推导）→ 推论（一 bs 一 graph，从约束推导）→ 显存机制（从推论推导）。

**改进**：
- `/learn-plan` 中每个步骤必须标注 **"从 X 推导"**，明确推导链条
- `/learn-write` 生成的概念章节禁止 checklist 式罗列；约束/推论必须用过渡句（"启动阶段执行的操作能够进一步推导出..."）从前一层推导出来
- `/learn-review` 增加 **"递进推导检查"** 维度：每一节的开头是否显式说明了它从前一节的什么推导而来

### 2. 概念深度不足

**问题**：/learn-write 把概念章节当作"概要"——每个阶段 1-2 句话概括。用户期望的是每个阶段一整段展开：Capture 要讲到节点保存什么信息、边如何推断；Instantiate 要展开三个子步骤，配以类比（"录制脚本 vs 编译为可执行二进制"、"烘焙"、"焊死"）。

**改进**：
- `/learn-write` 的概念章节不能只做"概要"。每个核心机制需要：(a) 内部展开为子步骤或子维度，(b) 至少一个生动类比或比喻辅助理解，(c) 末尾附总结性判断（"总的来说，CUDA Graph 是一个较为脆弱的静态图操作"）
- 对于有独立价值的概念（如显存共享机制），即使当前步骤只是铺垫，也应独立展开为子节，而不是一句话带过

### 3. 文章必须有驱动问题

**问题**：初版 plan 没有明确"为什么需要双 CUDA Graph"这个核心问题。文章缺少"脊柱"——读者不知道后续所有工程复杂性是为了什么。

**改进**：
- `/learn-plan` 在设计章节结构时，必须首先识别 **文章的驱动问题**（"这篇文章要回答的核心问题是什么？"）
- 驱动问题的位置必须经过推导才能出现：先给读者足够的背景（概念 + 模型特征），再在读者有能力理解的位置提出问题
- Plan 中应显式标注"这是文章的驱动问题"

### 4. 概念 → 模型 → 代码的顺序必须强制

**问题**：初版 plan 和文章都是"模型架构 → 代码实现 → CUDA Graph 概念"——读者还不知道 CUDA Graph 的约束是什么，就被扔进了工程代码里。

**改进**：
- `/learn-plan` 的步骤顺序必须遵循 **概念 → 模型 → 代码** 的三阶段结构：
  1. 先建立概念框架（机制、约束、推论）——给读者"分析工具"
  2. 再介绍具体模型/场景，用概念框架解释"为什么这个场景有挑战"
  3. 最后进入工程代码，展示概念如何落地
- 绝对不能反过来。这是硬约束，不是建议。

### 5. 约束映射应融入行文，不要独立成表

**问题**：/learn-write 生成了一个独立的"约束映射表"章节，将五条约束与工程设计一一对应。这是罗列式的写法。用户在实际文章中将映射融入每个工程章节（"这直接对应五条约束中的最后一条"），作为推导的自然组成部分。

**改进**：
- `/learn-write` 不要生成独立的"约束/概念映射表"章节
- 每个工程设计讨论中，应自然地引用概念约束（"这正是第一章中'指针稳定性'约束的体现"），让映射成为推导链的一部分

### 6. 开篇风格偏好

**问题**：/learn-write 生成的开篇是"因为工作需要"式的模板化开头。用户更喜欢：先回顾前序文章建立系列连续性 → 紧接亮出 benchmark 数据 → 精炼编号列表路线图 → 随意致谢。

**改进**：
- 开篇不要使用"因为工作需要"这类模板句式
- 系列文章的开篇应先回顾前序文章（"去年 8 月，我浅浅写过..."）
- 有 benchmark 数据时，开篇就亮出（用表格），用成果抓读者
- 路线图用 4 条以内的精炼编号列表，不要用长段落
- 不要写"本文基于 commit xxx 进行分析"这类模板声明——commit hash 在代码引用时自然出现即可
- 致谢随意自然（"各位大哥"风格），不需要标注公司/组织

### 7. 允许引用真实人物和个人化知识来源

**问题**：/learn-write 从不引用真实人物。用户会写"兰青老师曾给我分享过的一种定义——CUDA Graph 就是一种 cache"。

**改进**：
- 当概念有明确的个人化知识来源时（某人的分享、某次对话），应保留这种引用
- 这种引用增加文章的人格化和可信度，是用户行文风格的重要组成部分

### 8. 格式约束

**问题**：/learn-write 使用 ASCII 艺术字画流程图和架构图，在不同设备间排版会完全崩溃。

**改进**：
- 禁止使用 ASCII 艺术字（`┌┐└┘│├─` 等字符画）
- 流程图/架构图使用 mermaid
- 对比/分类使用 markdown 表格
- 优化路径等线性结构使用 markdown 表格或编号列表

### 9. /learn-plan 应考虑更广泛的知识版图

**问题**：CUDA Graph 的 plan 完全聚焦于 PR #153，遗漏了 SGLang 主仓库中 Piecewise CUDA Graph 这一重要工作。用户不得不手动补充。

**改进**：
- `/learn-plan` 在设计学习路线时，不应只聚焦于用户给定的直接输入（一个 PR、一个 issue）
- 应主动搜索 **相关仓库中同一 topic 的其他重要工作**（如 SGLang 主仓库中的 piecewise CUDA graph、vLLM 中的类似实现）
- 在 plan 中标注"拓展阅读"或"对比分析"步骤，将视野从单一案例扩展到框架级理解

### 10. 章节间过渡必须是推导，不是泛泛连接

**问题**：/learn-write 生成的章节过渡是泛泛的（"有了这些基础，我们来看..."），没有说明具体从前一节的什么推导出当前节的什么。

**改进**：
- 每一节的开篇过渡句必须 **具体引用前一节的某个结论或某个悬念**：
  - 好："上一章确立了'把两个 AR 统一到一个 graph'的决策，并列出了它引入的三项工程复杂性。从这一章开始，我们逐项展开"
  - 好："回顾开篇的 benchmark 表格：CUDA Graph only 达到 88 tok/s，但 partial compile 还能+36%——这 36% 从何而来？"
  - 差："有了这些基础，我们来看..."
  - 差："接下来讨论..."

### 11. 模型介绍应先交代全貌，再进入计算细节

**问题**：/learn-write 直接从实现层面描述模型（"36 层 Transformer，hidden_size=2560"），缺少模型的全局介绍。用户重写后：先给出模型定位（"Fish Audio 推出的 5B 参数语音生成模型"）、用途（TTS）、HuggingFace 链接，再用"抛开模型架构上的设计"过渡到计算模式分析。

**改进**：
- 模型介绍分两层：第一层交代模型全貌（名称、来源、参数量级、用途、链接），第二层才进入计算特征（GEMM shape、kernel 耗时）
- "抛开架构设计，从计算模式来看"这样的过渡句有助于读者理解文章的分析视角

### 12. 展示设计演进路径，而非直接给出最终方案

**问题**：/learn-write 直接给出"统一 CUDA Graph"方案，没有交代中间的思考过程。用户重写后：先说明"只覆盖 slow head"的 baseline → 提出"双 CUDA Graph"的自然想法（Fish Audio 早期内部方案）→ 分析双 graph 的不足 → 推导出"统一到一个 graph"的最终方案。

**改进**：
- `/learn-write` 在介绍设计方案时，必须展示**自然的演进路径**：baseline → 中间方案 → 最终方案
- 中间方案（哪怕最终没被采用）帮助读者理解 WHY——为什么最终方案是这样而不是那样
- 这种演进路径也是递进推导原则在设计分析中的体现

### 13. 设计决策需要替代方案对比表

**问题**：/learn-write 只呈现最终方案，没有与替代方案做结构化对比。用户重写后，添加了"统一单 CUDA Graph vs 双 CUDA Graph"的五维度对比表（graph 数量、CPU 调度、工程复杂度、memory pool、灵活性）。

**改进**：
- 对于重要的设计决策，`/learn-write` 应生成替代方案对比表
- 对比表帮助读者理解 trade-off，而非只看到"我们选了 A"

### 14. 明确区分"改变了什么"和"没改变什么"

**问题**：统一 CUDA Graph 后，读者容易误以为两个 AR 过程的状态管理也被耦合了。用户显式澄清："统一 graph 只意味着 kernel 被录进同一个 DAG，并不意味着状态管理有任何耦合。改变的只有 kernel 的组织方式，不改变状态管理。"

**改进**：
- 当一个设计决策合并/统一了两个子系统时，必须显式说明：什么被改变了（kernel 组织），什么没被改变（状态管理、KV cache 独立性）
- 这种澄清防止读者产生错误的心理模型

### 15. 工程挑战应分组为一个章节，而非独立成章

**问题**：/learn-write 将 deferred capture、buffer 分配、persistent buffer 各自作为独立的 `##` 章节。用户重写后合并为一个 `##`（"具体实现"），三者作为 `###` 子节——它们是同一个实现故事的不同方面，不是独立的话题。

**改进**：
- 当多个工程挑战来自同一个设计决策（统一 CUDA Graph）时，应合并为一个 `##` 章节，各挑战作为 `###` 子节
- 子节标题应直接编码对应的 CUDA Graph 约束（"避免动态内存分配"、"避免 host-device sync"），让读者从标题就能看到每个子节在回应哪条约束

### 16. 解释"为什么不用 X"时，分析 X 解决的问题与当前场景的错位

**问题**：/learn-write 解释 fast head 不用 paged KV cache 时只说"序列长度固定"。用户重写后：先分析 paged KV cache 的核心价值（多请求不同长度、动态增长、page 粒度共享显存），再逐条说明 fast head 为什么绕开了这些场景（"长度固定、同步推进、用完即弃"）。

**改进**：
- 解释"为什么不用 X"时，不能只说"不需要 X 的功能"
- 正确的模式：先分析 X 解决什么问题 → 再分析当前场景为什么这些问题不存在 → 结论：X 的复杂性没有收益

### 将以上改进反映到三个子命令

| 改进项 | /learn-plan | /learn-write | /learn-review |
|---|---|---|---|
| 递进推导 | 每步标注"从 X 推导" | 禁止 checklist 罗列 | 增加推导链检查维度 |
| 概念深度 | 标注哪些概念需要独立展开 | 每个机制展开为子步骤+类比+总结 | 检查概念章节是否足够深 |
| 驱动问题 | 显式标注驱动问题及其位置 | 确保驱动问题在正确位置出现 | 检查是否有清晰的驱动问题 |
| 概念→模型→代码 | 步骤顺序硬约束 | 章节顺序硬约束 | 检查顺序是否正确 |
| 约束映射融入行文 | 不设独立映射步骤 | 不生成独立映射表 | 检查映射是否自然融入 |
| 开篇风格 | — | 遵循回顾+数据+路线图+致谢 | 检查开篇是否模板化 |
| 格式约束 | — | 禁止 ASCII 艺术字 | 检查是否有 ASCII 图 |
| 广泛视野 | 主动搜索同 topic 的相关工作 | — | — |
| 推导式过渡 | — | 每节开篇具体引用前节结论 | 检查过渡句是否具体 |
| 模型全貌先行 | 模型步骤区分"全貌"和"计算特征"两层 | 先交代模型定位/来源/用途，再进入计算细节 | 检查模型介绍是否缺少全貌 |
| 设计演进路径 | 标注 baseline → 中间方案 → 最终方案 | 展示自然演进，不直接给最终方案 | 检查是否有替代方案讨论 |
| 替代方案对比表 | 标注需要对比的设计决策 | 为重要决策生成替代方案对比表 | 检查是否缺少对比 |
| 区分改变/不改变 | — | 统一设计时显式说明什么被改变、什么没被改变 | 检查是否有可能的误解 |
| 工程章节分组 | 同源挑战归为一步 | 同一设计决策的工程挑战合并为一个 `##`，各用 `###` | 检查章节粒度是否过细 |
| "为什么不用X"的分析 | — | 先分析 X 解决什么 → 场景为什么不需要 → 结论 | 检查是否只说"不需要"而没说"为什么不需要" |