# /learn Agent Proposal

## 项目背景

### 这个 repo 是什么

Awesome-ML-SYS-Tutorial 是 chenyang 从事 ML System 研究以来的完整学习笔记合集。从 2024 年 8 月在科研中使用 SGLang 开始，到现在超过 4.5K Stars，这个 repo 记录了一个 ML SYS researcher 从入门到前沿的全部历程。它的内容覆盖 RL infra、在线/离线推理系统、以及 AI Infra 基本功三大领域。

这个 repo 有一个独特的性质：**README 中列举的所有文章，正序阅读是研究领域的技术前沿，倒序阅读则是从零开始一步步学习 ML System 的完整路径。** 这意味着它不只是一堆零散笔记的堆叠，而是一套有内在逻辑的知识体系——后面的文章自然地建立在前面文章的基础之上，形成了一条可复现的学习曲线。

### 为什么做 /learn

这套笔记体系对 chenyang 本人的成长帮助极大，但它目前仍然是"被动的"——文章写好就放在那里，知识图谱存在于作者的脑中，学习路径需要读者自行摸索。随着文章数量增长到近百篇，涵盖数十个 topic，几个问题开始浮现：

1. **知识回顾的成本越来越高**：一年前写的文章，自己都需要重新读一遍才能回忆起来。框架版本迭代后，一些内容可能已经过时，但没有机制来标记。
2. **新 topic 的学习缺乏系统引导**：每次学习一个新 topic 时，哪些已有知识是前置依赖？该学到什么深度？这些判断目前全靠个人经验，没有工具化。
3. **其他人难以复用这条学习路径**：虽然 repo 开源，但"倒序阅读"这条路径是隐含的。一个新手面对近百篇文章，不知道从哪里开始，也不知道哪些可以跳过。
4. **写作风格的一致性依赖人工自觉**：随着贡献者增多（目前致谢列表中涉及数十位来自 CMU、Amazon、Meta、蚂蚁、阿里等机构的朋友），保持整个 repo 的风格统一变得困难。

/learn agent 的目标就是把这套"存在于脑中的学习系统"外化为一个可交互的工具——能够理解这个 repo 的知识结构，理解作者的学习风格，然后在学习新内容、撰写新文章、回顾旧知识时提供有针对性的辅助。

### 更深的动机

README 中有一段话道出了整个 repo 的精神内核：

> 兜兜转转一年，这就是目前让我坚持继续做 Infra 的决心，**为了做出正确的基础建设，为社区得到正确的结论做出自己的贡献**。

chenyang 相信，大量论文得出的 RL 结论建立在可能漏洞极多的 RL infra 上，而构建正确的基础设施是确保正确结论的前提。这个信念驱动了他投入大量时间在 SGLang RL 社区的 infra 开发上，也驱动了这个 repo 的持续更新。

/learn agent 是这个信念的自然延伸：**如果正确的 infra 需要正确的理解，那么一个帮助人系统性地建立正确理解的工具，就是在更上游做基础建设。** 它不只是一个学习助手，而是希望把"如何正确地学习 ML System"这件事本身工具化，让更多人能够沿着这条被验证过的路径，更高效地建立对 RL infra、推理系统和 AI Infra 基本功的正确理解。

### 当前阶段

当前阶段的唯一交付物是本 proposal。在 chenyang 审核通过之前，不会开始任何实现工作。

---

## 第一部分：写作与学习模式深度分析

### 维度一：学习习惯 — 宏观认知 + 代码落地的双轨模式

**总结：你的学习模式是"先建框架、再落代码"，且代码永远来自真实生产系统，不是教学示例。**

这个模式在 repo 中高度一致，可以归纳为三个阶段：

#### 阶段一：动机驱动的概念引入

几乎所有文章都以一个真实的工程需求或困惑作为起点，而非"今天来学习 X 概念"。

- `rlhf/sys-design/readme-1.md` 开头："因为工作需要，最近终于得空能够再次深入去学习思考主流 RL 框架的系统设计。"——从工作需要出发，而非学术兴趣。
- `torch/cuda-graph/readme.md` 开头："前几天和标哥在研究 torch-mem-savor 的时候发现自己对 CUDA Graph 知之甚少"——从一次具体的协作中发现知识盲区。
- `torch/torch-distributed/readme.md` 开头：完整记录了怜悯告诉他"你只需要学习 torch.dist"的对话，甚至包含了自己把 `torch.dist` 误认为计算距离的笑话——学习的触发点是一个具体的开发任务（给 OpenRLHF 写 weight_update 接口）。
- `sglang/scheduler/readme.md` 开头："想要系统性学习 SGLang Scheduler 的想法已经有一年了...这个事情必然是悬在我心上的，否则经常看到在 SGLang 的各种技术讨论群里，大家聊的内容有许多我都毫不了解，让人产生了深深的技术焦虑。"——动机来自社区讨论中的知识焦虑。

#### 阶段二：建立系统性认知框架

在进入代码之前，你总是先用大篇幅建立概念性的理解框架。这个框架通常包含：分类对比、设计权衡、历史脉络。

- `rlhf/verl/readme.md`：在分析 verl 代码之前，用整整一个大段落阐释 single-controller vs multi-controller 的设计哲学，从通讯压力、鲁棒性、SPMD 模式层层推进，最后才说"理解了这些概念后，我们可以正式进入 veRL 的 introduction 了"。
- `rlhf/sys-design/readme-2.md`：在分析 FSDP 代码之前，先用大量篇幅从 DP → DDP → DeepSpeed Zero Stage 1/2/3 逐级推演，建立完整的"通讯换显存"谱系，明确标注"总的来说，zero 就是通讯换显存的典型代表"之后，才进入 FSDP 本身。
- `rlhf/slime/fp8/readme.md`：在讨论 FP8 在 RL 中的应用之前，先从 Tensor Core 硬件演进（Volta → Turing → Ampere → Hopper → Blackwell）讲起，再讲 FP8 格式（E4M3 vs E5M2）、scale 选择（FP32 vs E8M0）、量化策略（per-tensor / per-block / per-token），建立完整的低精度计算知识体系后，才进入"FP8 + RL"的主题。

#### 阶段三：代码落地——永远是真实系统的代码

概念框架建立后，转入代码分析。衔接方式是极为自然的"有了这些基础"句式。代码永远来自真实的生产框架（SGLang、verl、slime、Megatron），带有具体的 GitHub commit 链接和文件路径。

- `rlhf/sys-design/readme-1.md`："好了，有了这些基础，我们来速览 slime 在 co-locate 策略下具体的权重更新"——接下来直接贴出 slime 的 `update_weights_from_tensor` 完整代码，逐行分析。
- `sglang/code-walk-through/readme-CN.md`：每个组件（TokenizerManager、Scheduler、TpModelWorker、ModelRunner）都附带精确到行号的 GitHub 链接，如 `TokenizerManager` 链接到 `tokenizer_manager.py#L88`。
- `rlhf/OpenRLHF/readme.md`：Reward、Advantage、Actor Loss、Critic Loss 四个计算步骤，每一步都先给数学公式，再给完整的 Python 代码实现，代码内有逐行注释。
- `rlhf/verl/multi-turn/code-walk-through/readme.md`：明确基于 commit `76f63cffa5` 进行分析，保证读者可以精确对照源码。

**双轨之间的衔接模式：** 概念轨到代码轨的过渡总是用一个明确的转折信号：
- "我们还是回到主线上"（`rlhf/sys-design/readme-1.md`）
- "有了这些认识，我们再来看"（`rlhf/verl/readme.md`）
- "有了这个直观的理解，我们回顾下"（`rlhf/verl/readme.md`）
- "好了，有了这些基础"（`rlhf/sys-design/readme-1.md`）

---

### 维度二：行文习惯 — 叙事逻辑和前后关联

#### 2.1 开篇结构：个人动机 + 致谢 + 路线图

你的文章开头有一套稳定的三段式结构：

1. **个人动机/情感**：总是坦诚自己为什么写这篇文章，经常包含个人感受。
   - `rlhf/partial-rollout/readme.md`："苦于工作繁琐，一直没有空拜读。正好最近去西雅图出差，在 LAX 往返 SEA 的路上终于有时间虔诚拜读这一工作。越读起来，越有一种文人相赏，难逢知音的感觉。"
   - `rlhf/verl/readme.md`："众所周知，我一直在 SGLang team 负责端茶倒水 + RLHF。"
   - `rlhf/rl-walk-through/part-1.md`："出于我个人的学习习惯，我对理论不感兴趣，好在对于 ML SYS researcher 而言，这样的直观已然可以带来巨大的帮助。"

2. **致谢**：几乎每篇重要文章都有详细的致谢列表，标明每个贡献者的学校/公司。
   - `rlhf/sys-design/readme-1.md`："照理，感谢参与本文档讨论和撰写的所有朋友们：zhuoran yin（CMU），changyi yang（CMU）..."，甚至加了"排名按照微信群的成员顺序 😂"。
   - `rlhf/slime/code-walk-through/readme.md`："Mao Cheng @ Meta, Zhuoran Yin @ CMU, Ji Li @ Ant Group..."

3. **文章路线图**：在正文开始前，概述本文将讨论什么。
   - `torch/cuda-graph/readme.md`："所以今天的文档尝试理解这些内容：1. 什么是 CUDA Graph；2. CUDA Graph 在推理中的应用；3. 为什么 CUDA Graph 在训练中很少使用..."
   - `sglang/scheduler/readme.md`："这篇文章会 follow 以上三篇文章的思路，深入理解 SGLang 的调度系统"

#### 2.2 复杂概念的拆解方式

你有一套特征鲜明的概念拆解策略：

1. **先下定义，再举例，最后给代码**：
   - `rlhf/verl/readme.md` 中 single-controller 的讲解：先给出加粗的核心定义（"在一个复杂的工作流程中，single controller 只有一个程序负责管理..."），再用 PPO 的工作流程举例，最后才进入 veRL 的具体实现。

2. **数值化的具体例子**：喜欢用具体数字来让抽象概念可感。
   - `rlhf/sys-design/readme-1.md`："假设这个参数的 size 是 `[1024, 1024]`，FSDP 的 TP size 是 4，而 SGLang 的 TP size 是 2。因此在更新参数开始前，每个 rank 上在 FSDP engine 内有 `[256, 1024]` 大小的 tensor..."
   - `rlhf/sys-design/readme-2.md`："考虑大小为 XB 的模型，这三部分会有 12X 的显存占用。"

3. **自问自答 / 设问句**：
   - `rlhf/sys-design/readme-1.md`："有一个问题非常值得分享：为什么 slime 需要先将 megatron 的 model weights offload 掉再 upload 上来，直接保留在 GPU 上不行吗？"
   - `rlhf/verl/readme.md`："这里思考一个小问题，为什么不能拿着 training engine 得到的 logits 做 sampling 然后 decode，貌似也可以用去 rollout？简单来说，太慢了。"

4. **`<details>` 折叠块**：可选的深入内容用折叠块包裹，保持主线清晰。
   - `rlhf/OpenRLHF/readme.md` 中所有的伪代码都放在 `<details><summary>伪代码</summary>` 中。

#### 2.3 文章间的交叉引用和知识链路

repo 内形成了一张密集的知识图谱，引用方式有三种：

1. **前序依赖**：明确告诉读者需要先读什么。
   - `rlhf/OpenRLHF/readme.md`："实际上，我已经写过一篇前序文章：浅析主流 Alignment 算法与 NeMo-Aligner 框架，但真的理解清楚了 RLHF 的计算流还是在读了这两篇文章之后..."
   - `sglang/scheduler/readme.md`："这篇文章会 follow 以上三篇文章的思路"，引用了 sys-design 系列的三篇。

2. **同系列衔接**：系列文章之间有明确的编号和衔接。
   - `rlhf/sys-design/` 系列：readme-1（权重更新）→ readme-2（FSDP）→ readme-3（Megatron）→ readme-4（MoE EP）→ readme-5（RDMA）。
   - `rlhf/verl/multi-turn/code-walk-through/` 系列：readme（初始化）→ readme-2（Rollout）→ readme-3（Make Experience）...
   - `rlhf/rl-walk-through/` 系列：part-1 到 part-8，系统覆盖 RL 基础理论。

3. **跨主题引用**：不同主题之间的知识桥梁。
   - `torch/nccl/readme.md` 引用 `torch/torch-distributed/readme.md`："在 torch-distributed 的后记中已经介绍过了。"
   - `rlhf/sys-design/readme-1.md` 引用 SGLang 架构图："具体的 SGLang 架构可以参考此图"，链接到 `sglang/code-walk-through/` 的 SVG。
   - `sglang/constraint-decoding/readme.md` 引用 SGLang 官方文档和 X-Grammar 论文。

#### 2.4 行文语气特征

- **谦逊但自信**："非常惭愧，从那之后，我也没有理解到二者的关系"（`sglang/scheduler/readme.md`）、"无所谓，我会出手！"（`torch/torch-distributed/readme.md`）。
- **适度幽默**："排名按照微信群的成员顺序 😂"、"~~便于绕开 sudo 用户直接 kill 僵尸进程~~"（`engineer/how-to-use-docker/readme.md`）。
- **敢于表达主观判断**："让我惊讶的是..."、"我个人觉得..."、"其中的酸甜苦辣，自不必多说"。
- **使用加粗和引用块来标记关键结论**。

---

### 维度三：学习深度的把握 — 在哪里止步

**总结：深度按"与我的工作的距离"分层递减，核心系统学到能修改扩展，相邻系统学到能理解复现，纯理论学到能建立直觉。**

#### 层级一：能够修改和扩展（自己直接参与开发的系统）

对于 SGLang、verl、slime 这些自己直接参与的项目，学习深度达到了源码级别，能够理解设计权衡并提出改进方案。

- `rlhf/sys-design/readme-1.md`：不只分析了 verl 的权重更新，还对比分析了 slime 的桶更新策略，最后横向对比三种权重更新方式，给出自己的判断："知易行难...权重更新接口开始的。RL 系统无非就是需要把 inference engine 接进去...可是其中的心酸滋味，自然只有真的打磨过，才能体会。"
- `sglang/code-walk-through/readme-CN.md`：完整追踪了一个请求从输入到输出的全链路（Server → TokenizerManager → Scheduler → TpModelWorker → ModelRunner → AttentionBackend → DetokenizerManager），每个组件都有初始化和核心函数分析。
- `rlhf/slime/spec/readme.md`：不只理解 speculative decoding 的原理，还设计并实现了 online SFT 训练 draft model 的方案，包含完整的训练目标构造、训练输入构造和训练流设计。

#### 层级二：能够理解和复现（相邻的基础设施）

对于 NCCL、CUDA Graph、FSDP 等相邻基础设施，学到理解其设计思想和使用方式，能复现关键操作，但不深入实现细节。

- `torch/nccl/readme.md`：覆盖了 NCCL 的概念、操作类型、Ring/Tree Algorithm、通信协议，以及两台集群的 `nvidia-smi topo -m` 输出分析。止步于"能读懂拓扑输出并做出正确的开发决策"，没有深入 NCCL 内核实现。
- `torch/cuda-graph/readme.md`：理解了 CUDA Graph 为什么在推理中大量使用（确定性操作流）、为什么在训练中很少使用（动态性），以及 torch-memory-savor 如何保护 CUDA Graph（虚拟地址与物理内存分离管理）。止步于理解原理，没有实现自己的 CUDA Graph 管理器。
- `torch/torch-distributed/readme.md`：通过实际代码练习掌握了 `init_process_group`、`all_reduce`、`broadcast` 等 API，但没有深入 PyTorch 分布式通信的底层实现。

#### 层级三：建立直觉和概念框架（理论和算法）

对于 RL 理论、量化理论等，明确止步于"直觉层面"。

- `rlhf/rl-walk-through/part-1.md`：开篇直说"出于我个人的学习习惯，我对理论不感兴趣，好在对于 ML SYS researcher 而言，这样的直观已然可以带来巨大的帮助"。保留了"平凡到看似愚蠢的定义"，但目标是建立直觉而非证明定理。
- `rlhf/partial-rollout/readme.md`：对 Kimi K1.5 的技术报告做了全面但选择性的解读——RL Recipe、采样策略、Long2Short 训练都有覆盖，但最浓墨重彩的是"RL Infra"部分（Partial Rollout），因为这直接关联作者自己的系统工作。
- `sglang/constraint-decoding/readme.md`：讲清了 constraint decoding 的概念、基本原理、X-Grammar 的算法优化和系统优化，止步于"能使用 SGLang 的 structured output 功能并理解其原理"，没有去实现 PDA。

#### 深度一致性分析

深度并不是在所有 topic 之间一致的，而是沿着一条清晰的"距离梯度"递减：

```
自己开发的系统 (SGLang/verl/slime) → 源码级，能修改扩展
                ↓
依赖的基础设施 (FSDP/NCCL/CUDA Graph) → 原理级，能正确使用
                ↓
相关的算法理论 (PPO/RL Theory/量化理论) → 直觉级，能指导系统设计
                ↓
读到的论文 (Kimi K1.5/SWE-Bench) → 摘要级，提取对系统工作有用的信息
```

这个深度分层是非常合理的——作为 ML SYS researcher，系统实现是核心战场，理论是弹药库，论文是情报来源。

---

## 第二部分：/learn Agent 设计方案

### 1. 核心功能定义

基于以上分析，/learn agent 的核心定位是：**一个理解你学习风格的个人 ML System 学习助手**。它不是通用的知识问答工具，而是基于这个 repo 的知识体系，帮助你高效学习新 topic 并保持风格一致性的专属工具。

#### 功能 F1：学习大纲生成（/learn plan）

**输入**：一个新的学习 topic（例如："我想学习 FlashAttention 的实现"）

**输出**：一份符合你风格的学习大纲，包含：
- **动机定位**：这个 topic 和你现有知识体系（repo 中已有内容）的关系。它填补了哪个知识空白？与哪些已有文章产生关联？
- **前置知识检查**：基于 repo 中已有的文章，列出学习这个 topic 之前应该回顾的内容（带 repo 内链接）。
- **学习路线图**：按照你的"概念框架 → 代码落地"双轨模式，规划学习步骤。每步标注对应的深度层级（修改扩展 / 理解复现 / 建立直觉）。
- **推荐资源**：基于你的历史引用习惯（知乎文章、官方文档、GitHub 代码），推荐最适合的学习资源。

#### 功能 F2：写作辅助（/learn write）

**输入**：一篇你正在撰写的草稿，或一个你希望开始写的 topic。

**输出**：
- **风格检查**：检查草稿是否符合你的行文习惯（开篇三段式、概念拆解方式、过渡句式、深度把握）。
- **交叉引用建议**：基于 repo 中已有内容，建议应该在哪里添加交叉引用。
- **深度校准**：根据 topic 在你的"距离梯度"中的位置，建议当前内容是否过深或过浅。
- **结构模板**：如果从零开始写，生成一个符合你风格的文章结构模板。

#### 功能 F3：知识图谱查询（/learn map）

**输入**：一个问题或 topic。

**输出**：
- 在 repo 知识图谱中的定位：这个 topic 被哪些文章覆盖过？覆盖到什么深度？
- 未覆盖的空白区域：基于已有文章的交叉引用网络，指出哪些被引用但未写的 topic。
- 学习路径推荐：如果某人想从零学习到这个 topic，应该按什么顺序阅读 repo 中的文章。

#### 功能 F4：复习与巩固（/learn review）

**输入**：一个已写过的 topic 或文章路径。

**输出**：
- 基于 repo 内容生成复习问题（概念理解 + 代码理解）。
- 标注这篇文章中哪些内容可能已经过时（基于框架版本演进等信息）。
- 延伸阅读建议：基于这篇文章的 topic，推荐后续可以深入的方向。

### 2. 交互方式

#### 2.1 主入口

在 Claude Code 中通过 skill 调用：

```
/learn plan "FlashAttention 的实现原理和在 SGLang 中的应用"
/learn write ./sglang/flash-attn/readme.md
/learn map "KV Cache"
/learn review ./rlhf/sys-design/readme-1.md
```

#### 2.2 输入形式

- **自然语言 topic 描述**："/learn plan tensor parallelism in MoE models"
- **文件路径**："/learn write ./new-article/readme.md"（对已有草稿做风格检查）
- **关键词**："/learn map NCCL"（查询知识图谱）

#### 2.3 输出形式

- **Markdown 文件**：/learn plan 和 /learn write 的输出为 Markdown 格式，可以直接作为文章骨架。
- **终端交互**：/learn map 和 /learn review 以结构化文本在终端呈现。
- **交互式追问**：所有功能都支持追问，例如 "/learn plan" 输出大纲后，可以继续 "把第三步展开"。

### 3. 上下文管理：`.learn/` 目录结构

```
.learn/
├── readme.md                    # 本 proposal
├── skill.md                     # /learn skill 的 prompt 定义
├── index/
│   ├── knowledge-graph.json     # repo 知识图谱（文章间的引用关系、topic 标签）
│   ├── article-meta.json        # 每篇文章的元信息（topic、深度层级、前置依赖、发布状态）
│   └── style-guide.md           # 从第一部分分析中提炼的风格指南（机器可读版本）
├── templates/
│   ├── code-walkthrough.md      # code walk through 类文章的模板
│   ├── sys-design.md            # 系统设计分析类文章的模板
│   ├── paper-reading.md         # 论文阅读笔记类文章的模板
│   └── tutorial.md              # 教程类文章的模板
└── config.md                    # agent 配置（深度偏好、引用风格等可调参数）
```

#### 3.1 knowledge-graph.json

自动扫描 repo 中所有 `.md` 文件，提取：
- 文章标题、路径、语言（中文/英文/双语）
- 文章间的引用关系（通过 markdown 链接分析）
- topic 标签（通过标题和内容关键词提取）
- 发布状态（已发布 / Pending Review / 未完成）
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

#### 3.2 style-guide.md

将第一部分的分析成果编码为机器可用的风格指南，包含：
- 开篇结构要求（动机 → 致谢 → 路线图）
- 概念拆解的标准流程（定义 → 数值例子 → 代码）
- 过渡句式的模板库
- 深度层级判定规则
- 交叉引用的触发条件

#### 3.3 article-meta.json

存储每篇文章的结构化元信息，供 agent 快速检索。这个文件通过扫描 repo 自动生成，也支持手动补充。

### 4. 风格保持机制

#### 4.1 System Prompt 中注入风格指南

/learn skill 的 system prompt 包含 `.learn/index/style-guide.md` 的完整内容，确保 agent 的每次输出都符合风格约束。

#### 4.2 基于模板的输出生成

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
[贴出关键代码，逐步分析]

## [模块 2]
...
```

#### 4.3 三维度自动检查

/learn write 在生成或检查内容时，自动进行三个维度的验证：

1. **双轨检查**：是否同时包含概念框架和代码分析？代码是否来自真实系统？
2. **叙事检查**：开篇是否有动机？段落间是否有过渡？是否有交叉引用？
3. **深度检查**：当前 topic 在"距离梯度"中的位置是什么？内容深度是否匹配？

### 5. 可扩展性：其他人如何基于这套框架贡献

#### 5.1 个人风格配置化

`.learn/config.md` 中的风格参数设计为可覆盖的，其他人 fork 这个 repo 后，可以：
- 修改 `config.md` 中的深度偏好（例如，理论研究者可能希望在"层级三"投入更多深度）
- 替换 `templates/` 中的模板（适配自己的写作风格）
- 保留 `knowledge-graph.json` 的结构但更新内容（对应自己的笔记体系）

#### 5.2 贡献者角色

当其他人向 repo 贡献文章时，/learn 可以帮助：
- **风格一致性审核**：`/learn write` 检查新文章是否符合 repo 的整体风格。
- **知识图谱更新**：新文章合入后，自动更新 `knowledge-graph.json`。
- **缺口发现**：`/learn map` 可以帮助贡献者找到 repo 中的知识空白，引导他们写最有价值的文章。

#### 5.3 `.learn/` 作为独立可复用模块

`.learn/` 目录随 repo 同步，其他人 clone 后即可使用。目录结构和配置文件的设计遵循"约定优于配置"原则：
- `knowledge-graph.json` 可以通过一个脚本从任意 Markdown repo 自动生成。
- `style-guide.md` 可以通过 /learn agent 对任意 repo 进行分析后自动生成。
- `templates/` 是可选的，没有模板时 agent 会基于 style-guide 动态生成。

### 6. 实现路径建议

为确保落地质量，建议分三期实现：

**Phase 1（MVP）**：
- 实现 `/learn` 作为 Claude Code skill
- 手动编写 `style-guide.md`（基于本 proposal 第一部分）
- 手动编写 `knowledge-graph.json`（通过脚本半自动生成）
- 实现 `/learn plan` 功能

**Phase 2（核心功能）**：
- 实现 `/learn write` 和 `/learn map`
- 编写文章模板
- 自动化 knowledge-graph.json 的生成和更新

**Phase 3（完善与扩展）**：
- 实现 `/learn review`
- 支持多人协作场景
- 支持其他人 fork 后自定义
