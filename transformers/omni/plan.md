# 学习 Omni 模型架构以指导 SGLang-Omni 重构——学习计划

## 动机定位

最近在推动 [SGLang-Omni 的重构](https://github.com/sgl-project/sglang-omni/issues/188)，越看代码越觉得当前框架的抽象层级太多了——一个请求从 HTTP API 到 `torch.forward` 要穿过 8-10 层，其中 Stage → Worker → Executor → Engine 四层的职责高度重叠。但在动刀砍层之前，有一个前置问题必须先回答清楚：**我们到底要支持什么样的模型？它们的架构差异在哪里？哪些计算模式可以统一，哪些必须保持差异化？**

如果不理解模型架构就去设计抽象，要么抽象太厚（当前的问题），要么抽象太薄（砍掉了不该砍的灵活性）。所以这篇笔记的目标很明确：通过深入研究目前需要支持的两个代表模型——Fish Audio S2 Pro 和 Qwen3-Omni——的架构，推导出对推理框架抽象层设计的约束。

### 开篇风格要点

1. **从重构工作切入，表达个人情感**——"很遗憾"对现有代码的不满、"充满期待"对重构挑战的兴奋。不是中性地描述问题，而是以第一人称坦诚自己的判断和期待
2. **开篇亮出 issue #188 的核心矛盾**：8-10 层抽象 vs 目标大量减少抽象层级
3. **概念框架开篇表达谦逊**——"我也是第一次接触 Omni 模型"，对传统语音研究领域表达敬畏。这不是客套，而是真实的学习者心态，让文章基调从一开始就是"学习笔记"而非"教程"
4. **不要 meta-frame 概念章节**——不说"我们需要先建立一套概念工具箱"这类功能性描述，直接进入概念讲解

### 驱动问题

> **当我们要在一个推理框架中同时服务 S2 Pro（Dual AR）和 Qwen3-Omni（Thinker + DiT Talker）这两类架构迥异的 omni 模型时，哪些计算模式可以统一抽象为框架层的通用能力，哪些必须作为模型特异的实现留给各自的底层实现？**

这个问题在读者理解了两个模型的架构差异之后才能有意义地提出——大约在第二步末尾或第三步开头。

### 系列定位

这是 `transformers/omni/` 下的第一篇（也可能是唯一一篇，视后续需要）。与以下已有文章产生关联：

- [再探 CUDA Graph：核心机制、多图复用以及 Dual AR 模型的统一覆盖优化](../../torch/cuda-graph/readme-2.md)：已深入分析了 S2 Pro 的 Dual AR 架构和 CUDA Graph 覆盖方案，本文从**模型架构对框架抽象的影响**这一不同视角重新审视同一模型

本文的深度层级为**修改扩展级**（SGLang-Omni 是自己参与开发的系统），需要达到"能理解设计权衡并提出改进方案"的水平。

## 前置知识检查

- [再探 CUDA Graph（系列第二篇）](../../torch/cuda-graph/readme-2.md)：理解 S2 Pro 的 Dual AR 架构、Slow AR / Fast AR 的计算特征、两种 KV cache 的共存——本文不重复这些内容，但会从不同视角引用
- 基本的 Transformer 架构理解：attention、FFN、KV cache、prefill/decode 二阶段

## 学习路线图

> **核心原则一：概念 → 模型 → 抽象推导。** 先建立 omni 模型的通用概念框架，再分别深入两个代表模型的架构，最后从架构差异推导出对框架抽象的约束。
>
> **核心原则二：递进推导，不是平铺罗列。** 每一步的内容从前一步推导而来。概念不是独立的知识点罗列，而是为后续的模型分析和抽象设计建立推导工具箱。

### 第一步：Omni 模型的通用概念框架

- **深度层级**：理解复现（建立概念工具箱，为后续修改扩展级的分析服务）
- **目标**：让读者理解 omni 模型的通用 pipeline，以及各阶段存在哪些设计自由度——这些自由度正是导致不同模型架构差异的根源
- **写作位置**：开篇之后的第一个正文章节

#### 1.1 Omni Pipeline（四阶段概览）

先给读者全景图：一个 omni 模型要完成的事情就是"听懂语音输入，生成语音回复"，天然分成 Audio Encoding → Understanding（Thinker）→ Speech Synthesis（Talker）→ Audio Decoding（Vocoder）四个阶段。用 mermaid 画出 pipeline。

- 对每个阶段给出一段话的概要
- 强调 Encoding 和 Decoding 跨模型相对稳定，架构分歧集中在中间两步
- Audio Encoding 段为 codec 做前向引用："codec token 是什么、为什么必须是离散的，是下一节的主题"

#### 1.2 Codec Audio Encoding（含 VQ 和 RVQ）

**Codec 和 RVQ 合并为一个完整的流叙述，不拆分为独立小节。** 叙事按四段递进：

1. **连续编码（时域压缩）**：从数据量问题出发（16kHz-48kHz → 50 万标量步），推导出压缩必要性。encoder 把波形压成低频连续帧向量序列（~12.5 帧/秒，128 维）。强调：到这一步仍是连续值，不能走标准 LLM 管线。
2. **向量量化（VQ）**：codebook = 有限代表向量集；最近邻查找 → 整数索引 → codec token id。WHY：连续值 → 离散 id 才能接 LLM 管线。
3. **从单层 VQ 到 RVQ**：经典 trade-off（码本太大 vs 太小）自然引出 RVQ。关键："每一层量化的是上一层的残差"。澄清 RVQ vs text tokenizer 的差异。数值例子。
4. **层间信息不对称性**：前层偏语义，后层偏声学细节。催生"大模型管前几层 + 小模块补后续层"的设计，引出 S2 Pro Dual AR。

#### 1.3 Understanding（Thinker）

**编辑决策：plan 原本无独立小节，写作中发现 Pipeline 展开了 Encoding（Codec），跳过 Understanding 直接到 Speech Synthesis 会让读者困惑，因此增设。**

- 本质：LLM/MLLM 做 prefill + decode，和标准 chat 模型无本质区别
- TTS 场景 vs omni 场景的输入差异（text+audio vs multimodal），但计算模式不变
- 框架视角：**Understanding 是不同 omni 模型间最同质的部分**，SGLang 已有优化直接复用
- 收尾过渡：架构分歧在 Understanding 之后

#### 1.4 Speech Synthesis（合并了原 1.4 设计自由度 + 原 1.5 信息流动机）

**编辑决策：原 plan 的 1.4（设计自由度表）和 1.5（为什么需要 hidden states）合并为一个小节。设问从"为什么需要 hidden states"反转为"为什么需要 text token"——后者更有深度，且引出 Talker 能力不足这个悲观判断。**

内容结构：
1. 设计自由度表（生成方式、Codebook 策略、信息流、时序）
2. 设问 + 展开：**"既然 Talker 已经需要 hidden states，为什么还需要 text token？"**
   - 悲观理解：Talker 能力不足，hidden states 信息纠缠，缺乏显式语义锚点 → 内容偏移
   - Text token 提供离散确定性的语义锚点
3. Qwen3-Omni 信息流细节：Talker 接收 Thinker 输出的 text token + audio/visual hidden states，不接收 text hidden states
4. TTFV trade-off + Qwen3-Omni 异步流水线
5. **S2 Pro 对比**：S2 Pro 不产生 text token，Slow AR 直接生成 semantic codec token，整个信息传递在 codec token 空间内完成

**关键严谨性约束**：
- 不能对 S2 Pro 使用 Thinker-Talker 术语（S2 Pro 没有独立的 Thinker-Talker 分离）
- "text token + hidden states"信息流明确限定为 Thinker-Talker 架构的特征

#### 1.5 Audio Decoding

- Vocoder 把 codec token 还原为音频波形
- 典型实现：causal ConvNet（EVA-GAN、Code2Wav、Vocos、HiFi-GAN）
- causal 的重要性：不需要后续帧即可合成当前帧，支持逐帧流式输出
- Qwen2.5-Omni → Qwen3-Omni 演进：block-wise DiT → causal ConvNet（Code2Wav）
- 框架视角：计算量轻、完全解耦于 LLM 调度，LLM memory-bandwidth bound vs ConvNet compute bound，可 MPS 并行
- Audio Decoding 和 Audio Encoding 一样，跨模型相对稳定，不是框架抽象的核心矛盾

### 第二步：Fish Audio S2 Pro 架构

- **深度层级**：修改扩展（S2 Pro 是 SGLang-Omni 支持的核心模型之一）
- **目标**：让读者理解 S2 Pro 的 Dual AR 架构及其推理特征，为后续与 Qwen3-Omni 的对比建立基础
- **交叉引用策略**：[CUDA Graph 系列第二篇](../../torch/cuda-graph/readme-2.md)已经从 CUDA Graph 优化的视角详细分析了 S2 Pro 的架构。本步不重复那些内容，而是聚焦于**模型架构对框架抽象的影响**

**写作结构原则（从第一轮迭代中总结）：事实先行，分析后置。**
- 先把推理过程讲完讲透（组件 + 完整 decode 流程），读者此时已经知道全貌
- 然后再做 Pipeline 映射、设计考量分析——因为事实已经建立，分析不需要重复解释组件分工
- 这样可以彻底避免旧版本中"分析在前导致后面讲 decode 流程时同样内容重复第二遍"的问题

**写作严谨性约束（从第一轮迭代中总结）：**
- **不混淆编码和生成**：RVQ 是 codec encoder 的量化机制，Slow AR / Fast AR 做的是 token 生成。两者共享 codebook 结构，但作用于不同阶段。不能说"这是 RVQ 中承载语义的那一层"，应说"codebook 的这一结构性质来自 RVQ 训练"
- **输入侧 vs 输出侧 codec token 必须显式区分**：输入侧的 codec token 是参考音频的编码，输出侧的 codec token 是目标语音的编码，两者是完全不同的序列
- **术语精确**："10 个 codec token"而非"10 层 token"（它们是 10 个独立 token，不是一个 token 有 10 层）
- **不用稻草人对比**：不搞"展平 vs 多流 vs 分阶段"的排除法，直接讲设计动机
- **Slow AR 的输入描述要统一本质**：prefill 输入 embedding，decode 输入 MCF 聚合向量——接口一致，构造方式不同

#### 2.1 模型简介

S2 Pro 是纯 TTS 模型，给定参考语音 + 目标文本，生成符合参考音色的语音。

#### 2.2 四个组件

先交代清楚 Slow AR、Fast AR、Codec Decoder、MCF 各自的角色。重点：Slow AR 的输入（prefill: prompt embedding; decode: MCF 聚合向量）必须在此处交代清楚。

#### 2.3 一帧的完整 decode 流程

按时间步 t 的四个阶段逐步展开：Slow AR → Fast AR → Codec Decoder → MCF。用 mermaid 画出单个时间步的 decode 流程。

#### 2.4 S2 Pro 与 Omni Pipeline 的对照

推理流程讲完后，再映射回四阶段 pipeline。核心澄清：S2 Pro 不走 Thinker → Talker 分离路径，Understanding + Speech Synthesis 合并在 Slow AR + Fast AR 中。显式区分输入侧 codec token（RVQ 编码参考音频）和输出侧 codec token（Dual AR 生成目标语音）。

#### 2.5 Dual AR 的设计考量

分析为什么拆成两个模型（Slow AR 的 context window 只装语义 token，Fast AR 固定长度不跨帧），以及逐项映射回 Speech Synthesis 设计自由度矩阵。

#### 2.6 从 serving 的视角看

S2 Pro 对框架友好的根本原因：Slow AR 就是标准 LLM decode loop，唯一特殊处理是 MCF 聚合向量作为输入。Fast AR 是固定后处理，不需要 continuous batching / paged KV cache。

### 第三步：Qwen3-Omni 架构

- **深度层级**：修改扩展
- **目标**：理解 Qwen3-Omni 的 Thinker + Talker 架构及其异步流水线推理模式
- **写作结构**：与第二步保持一致——**事实先行，分析后置**

**plan 修正记录**：原 plan 将 Talker 描述为 DiT-based（Diffusion Transformer），这是基于 Qwen2.5-Omni 的设计。实际 Qwen3-Omni 的 Talker 是 AR-based MoE LLM（3B-A0.3B），配合 MTP Module（80M）补全 codebook + Code2Wav（200M causal ConvNet）合成波形。Qwen3-Omni 已经用 Code2Wav 替换了 Qwen2.5-Omni 的 block-wise DiT。

#### 3.1 模型全貌

Qwen3-Omni 是端到端多模态理解 + 语音合成模型，与 S2 Pro（纯 TTS）定位不同。

#### 3.2 五个组件

Thinker、Talker、MTP Module、Code2Wav、Audio/Vision Encoder。组件描述中交代角色和参数量级，细节（token rate、具体维度）留给推理流程展开。Encoder 描述精简到框架视角需要的信息，训练细节（TM-RoPE 等）放入折叠块或删除。

#### 3.3 完整推理流程

按阶段展开：Encoder 预处理 → Thinker 生成 text → Talker 生成 codec → MTP 补全 → Code2Wav 合成。每个阶段的输入/输出/KV cache 行为写清楚。

**关键**：显式区分 AuT encoder 的输出（连续 audio hidden states，不是 RVQ codec token）和 Talker 生成的输出（离散 codec token）。

#### 3.4 端到端延迟分解

标注数据来源（技术报告 / 自测），给出各阶段延迟数值。

#### 3.5 Qwen3-Omni 与 Omni Pipeline 的对照

推理流程讲完后，映射回四阶段 pipeline，与 S2 Pro 章节结构对称。

#### 3.6 从 serving 的视角看

双 LLM 异步调度、跨 stage tensor relay、标准 LLM serving 能力复用、流式输出。

**注意**：与 S2 Pro 的对比（串行嵌套 vs 异步流水线）不在本节展开，留给第四步的结构化对比。本节只描述 Qwen3-Omni 自身的 serving 需求。

### 第四步：架构对比与框架抽象推导

- **深度层级**：修改扩展
- **目标**：从两个模型的架构差异中，推导出推理框架的抽象设计约束——这是本文的核心交付物
- **方法**：结构化对比 → 推导共性 → 推导差异 → 得出抽象边界

#### 4.1 结构化架构对比表

| 维度 | Fish Audio S2 Pro | Qwen3-Omni |
|------|-------------------|------------|
| **模型用途** | 纯 TTS | 多模态理解 + 语音合成 |
| **理解阶段** | Slow AR（Qwen3-4B） | Thinker（多模态 LLM） |
| **理解阶段本质** | 标准 LLM decode | 多模态 LLM prefill + decode |
| **合成方式** | AR（Fast AR 逐 codebook 9步） | Diffusion（DiT 迭代 denoising） |
| **合成阶段 KV Cache** | Static 预分配 | 无（不是 AR） |
| **合成阶段 batching** | 可 continuous batching | 按 chunk/段落 batch |
| **Thinker→Talker 信息流** | Hidden states（内部传递，同一 forward） | Hidden states + text tokens（跨 stage 传递） |
| **CUDA Graph 适用性** | 全 pipeline 可覆盖 | 仅 Thinker decode 阶段 |
| **Vocoder** | 需要（codec → waveform） | 需要（codec → waveform） |

#### 4.2 从对比推导共性（可统一的部分）

1. **LLM backbone serving**：两者的"理解"阶段都是 LLM（或多模态 LLM），可以共享 SGLang 的核心 serving 能力
2. **Pipeline staging**：都需要多阶段串联（encode → understand → synthesize → decode），需要 stage 概念
3. **Vocoder**：最后一步都是 codec → waveform 的转换，vocoder 可以统一
4. **Tensor relay**：都需要在 stage 之间传递 tensor（hidden states、codec tokens）

#### 4.3 从对比推导差异（必须差异化的部分）

1. **Speech Synthesis 阶段的执行模式完全不同**：AR vs Diffusion，不可能用同一个 ModelRunner 抽象
2. **KV Cache 管理差异**：S2 Pro 需要管理两套 KV cache（paged + static），Qwen3-Omni 的 Talker 不需要 KV cache
3. **Batching 策略差异**：AR-based Talker 可以 continuous batching，Diffusion-based Talker 需要不同的 batching 策略
4. **CUDA Graph 适用范围不同**：S2 Pro 全覆盖，Qwen3-Omni 仅 Thinker

#### 4.4 推导抽象边界（驱动问题的回答）

基于 4.2 和 4.3 的分析，推导出框架抽象应该在哪里画线：

- **统一层（框架提供）**：Pipeline orchestration、LLM serving（SGLang 已有）、Tensor relay、Vocoder execution
- **差异层（模型特异）**：Speech Synthesis 的 ModelRunner 实现、KV cache 管理策略、CUDA Graph 覆盖策略

**对 issue #188 的回应**：

- Stage → Worker → Executor → Engine 中，Worker 和 Executor 的确是多余的间接层——它们的 `getattr` 动态委托只是在翻译接口，没有增加语义
- 但 Stage 和 Engine 各有存在的必要：Stage 负责 pipeline orchestration（管理多个阶段的串联），Engine 负责单个阶段内的调度（continuous batching、CUDA Graph 等）
- **ModelRunner 需要成为核心抽象边界**：框架保证 Engine → ModelRunner 的接口稳定，ModelRunner 以下的实现（AR / Diffusion / Dual AR）由各模型自行决定

### 第五步（可选）：优化方向——跨模态投机采样

- **深度层级**：建立直觉（这是一个尚未验证的优化方向）
- **目标**：记录关于 Thinker-Talker overlap 的思考，作为后续工作的方向性讨论
- **写作风格**：明确标注为"未经验证的想法"，与前四步的确定性分析做区分

#### 5.1 当前串行流程的 latency 分析

当前：Thinker decode 全部完成 → Hidden states + text tokens 传给 Talker → Talker 开始生成

TTFT（Time to First Voice）= Thinker prefill + Thinker decode 全部 + Talker 首段生成

#### 5.2 Overlap 的可能性

设想：Thinker 完成 prefill 后立即将 hidden states 发给 Talker 开始 decode codec token，Thinker 继续 decode text token。两个过程 overlap。

- **可能的收益**：减少 TTFT
- **核心挑战**：文字-语音一致性（如果 Talker 先生成了语音，但 Thinker 后续的 text token 与语音不一致怎么办？）
- **类比**：跨模态的投机采样（Cross-modal Speculative Decoding）

#### 5.3 Drawback 分析

结构化对比：

| 维度 | 当前串行方案 | Overlap 方案 |
|------|-------------|-------------|
| TTFT | Thinker 全部完成后才有语音 | Thinker prefill 后即可有语音 |
| 文字-语音一致性 | 完全一致（Talker 看到全部 text） | 可能不一致，需要回滚机制 |
| 训练难度 | 标准训练 | 需要新的训练范式 |
| 工程复杂度 | 低 | 高（回滚机制、校验逻辑） |
| 用户体验 | 延迟较高但输出稳定 | 延迟低但可能有字幕回滚 |

## 推荐资源

- [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215)（如有）：Qwen3-Omni 的前身，理解 Thinker + Talker 架构
- [Fish Speech / S2 Pro 论文或技术博客](https://fish.audio)：Dual AR + RVQ 的设计动机
- [EnCodec (Meta)](https://arxiv.org/abs/2210.13438)：理解 RVQ codec 的原理
- [SoundStorm (Google)](https://arxiv.org/abs/2305.09636)：非自回归 codec 生成的代表方案，与 AR 方案的对比
- SGLang-Omni 代码库中的关键文件：
  - `sglang_omni/models/fishaudio_s2_pro/sglang_model.py`：S2 Pro 的模型定义
  - `sglang_omni/models/qwen3_omni/`：Qwen3-Omni 的模型定义
  - `sglang_omni/engines/`：当前的抽象层级结构

## 写作注意事项

1. **S2 Pro 的架构分析不要重复 CUDA Graph 文章的内容**——交叉引用即可，本文聚焦于"模型架构如何影响框架抽象"这个新视角
2. **第四步是核心交付物**——前三步都是为第四步的推导提供素材。如果篇幅受限，前三步可以精简，第四步不能精简
3. **Qwen3-Omni 的 Talker 实现细节可能需要进一步调研**——PR #155 的代码可能有更多信息，写作时需要补充具体的代码引用
4. **保持"学习笔记"基调**——这不是一个 RFC 或设计文档，而是通过学习模型架构来为后续的重构设计积累认知

## 从第一轮写作中总结的风格修正（/learn-write 迭代记录）

以下要点来自 chenyang 对第一轮 /learn-write 产出的修正，优先级高于上述通用写法指导：

### 开篇与情感表达
- **表达个人情感和判断**：对现有代码的不满用"很遗憾"直说，对重构挑战的兴奋用"充满期待"表达。不要写中性、克制的问题陈述
- **进入新领域时表达谦逊**：概念章节开篇应坦诚"我也是第一次接触"，对相关领域的前辈表达敬畏。这是真实的学习者心态
- **不要 meta-frame 概念章节**：不说"我们需要先建立一套概念工具箱"、"这套工具箱的作用是..."。直接进入概念讲解，读者自然会明白这些概念的作用

### 概念讲解风格
- **不能把技术概念当黑盒**：codec 不是"一种将音频编码为 token 的神经网络"这一句话就完了——必须显式展开内部的两步过程（连续编码 → codebook 离散化），否则读者无法理解后续的 RVQ
- **类比必须精确且与 LLM 管线相关**："WAV 压缩为 MP3"这类浅层类比不够。应该用 codebook ↔ vocabulary、codec ↔ tokenizer 这样直接对应读者已有知识体系的精确类比
- **章节标题用描述性名词**："Codec Audio Token"，不用问句式"为什么需要 Codec？"
- **解释 WHY 时要接地气**：不是抽象地说"需要压缩"，而是说清楚"连续向量没法用标准 LLM 管线（词表 + 交叉熵 + 自回归采样）处理，所以必须离散化"
