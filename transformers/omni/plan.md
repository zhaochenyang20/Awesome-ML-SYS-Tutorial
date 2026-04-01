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
- **方法**：从"为什么需要 codec"这个最底层的问题出发，逐步推导出完整的 pipeline
- **写作位置**：开篇之后的第一个正文章节

#### 1.1 Codec Audio Token

从数据量问题出发，推导出 codec 的必要性，然后**显式展开 codec 的两步编码过程**（这一点在写作中容易被忽略——不能把 codec 当黑盒）：

- **数据量差异**：语音的原始采样率（16kHz-48kHz）意味着 1 秒音频 = 16000-48000 个采样点。如果直接让 Transformer 处理原始音频波形，序列长度完全不可接受
- **两步编码过程（必须显式展开）**：
  1. **第一步——连续编码**：Audio Encoder 将原始波形编码为低频的连续特征向量序列（例如 12.5 Hz，每帧一个向量，常见 128 维）。这一步完成了时域压缩，但输出仍然是连续的
  2. **第二步——离散化（Codebook Lookup）**：对每个连续特征向量，在 codebook（码本）中找最近的条目（nearest neighbor lookup），用该条目的整数索引作为最终的 codec token。码本是一张预训练好的向量查找表，**角色上等同于 LLM 的 vocabulary**——都是"有限集合里的条目 + 查表用的整数索引"
- **为什么需要离散化**：连续特征向量如果不经过离散化，无法沿用"有限词表 + 交叉熵 + 自回归采样"这条标准 LLM 管线。离散化让 codec token 和 text token 在形式上统一
- **Codec ↔ Tokenizer 的完整类比**：tokenizer 用规则或 BPE 把字符串映射成 vocabulary 里的整数序列；audio codec 用 encoder + VQ 把波形映射成 codebook 里的整数序列。此后将 codec token 当离散序列交给 Transformer，与处理 text token 没有本质区别
- **数值例子**：原始 24kHz 采样 → 编码后约 12.5 token/s → 压缩比约 1920:1。一句 5 秒的话从 120000 个采样点变成约 63 个 codec token

**写作注意**：
- 章节标题用描述性名词（"Codec Audio Token"），不用问句式（"为什么需要 Codec？"）
- 不使用浅层类比（如"WAV 压缩为 MP3"）——应该用 LLM 管线相关的精确类比（codebook ↔ vocabulary, codec ↔ tokenizer）
- 两步编码过程是理解后续 RVQ 的前提，不能省略

#### 1.2 RVQ：多层 Codebook 的信息组织

从 1.1 建立的"连续特征 → codebook lookup → 离散 token"流程推导：单个 codebook 的表达能力是有限的。

- **单 codebook 的信息瓶颈**：一个 codebook 能表达的信息有限。码本大小（vocabulary size）和量化精度之间存在根本性的 trade-off：码本太大训练困难、embedding table 爆炸；码本太小量化误差高，还原出来的音频像机器人
- **RVQ（Residual Vector Quantization）**：用多层 codebook 逐步逼近原始信号。第一层编码粗粒度的语义信息（"说了什么"），后续各层编码残差（"怎么说的"——音色、韵律、情感细节）。这种层级结构暗示了一个关键推论：**不同 codebook 层的信息价值是高度不对称的——第一层携带最核心的语义，损失它等于损失语音内容；后续层携带的是"锦上添花"的声学细节，损失它们只是降低音质**
- **对后续模型分析的意义**：这种不对称性直接催生了"用大模型生成语义层、用小模型补全声学层"的设计思路——S2 Pro 的 Dual AR 正是这一思路的具体实现

#### 1.3 通用 Omni Pipeline：四阶段推导

从 codec 的存在推导出完整的 pipeline：

1. **Audio Encoding**：有了 codec 的概念，第一步自然是把输入音频编码为 codec token（或等价的连续表示）。Audio Encoder 将原始波形 → codec token 序列
2. **Understanding（Thinker）**：有了 codec token，需要一个足够强大的模型来理解它们的含义并生成文本响应。这一步本质上就是一个（可能是多模态的）LLM 做 prefill + decode
3. **Speech Synthesis（Talker）**：理解并生成了文本之后，需要把文本转回语音。**这一步存在最大的设计自由度**——到底用什么方式生成 codec token？AR？Diffusion？Flow matching？这是导致不同模型架构差异的核心分歧点
4. **Audio Decoding**：最后，把生成的 codec token 通过 vocoder 还原为音频波形。这一步通常是一个轻量级的 ConvNet（如 Vocos、HiFi-GAN），计算量远小于前三步

用 mermaid 画出通用 pipeline。

#### 1.4 设计自由度：Speech Synthesis 阶段的分歧

从 1.3 的第三阶段"最大的设计自由度"展开：

| 设计维度 | 选项 A | 选项 B |
|----------|--------|--------|
| 生成方式 | **Autoregressive**：逐 token 生成 codec | **Non-Autoregressive**：Diffusion / Flow Matching，一次性或迭代生成 |
| Codebook 生成策略 | **逐 codebook AR**：先生成第 1 层，再自回归地生成 2-N 层 | **并行生成**：所有 codebook 层同时输出 |
| 与 Thinker 的信息流 | **Hidden States + Text Tokens**：Talker 接收两者 | **仅 Hidden States**：Talker 从 hidden states 直接解码 |
| Thinker-Talker 时序 | **串行**：Thinker decode 完成后 Talker 才开始 | **流式**：Thinker 每生成一个 token，Talker 就开始处理 |

**这张表不是为了穷举所有可能，而是为了建立一个分析框架**：后续看每个具体模型时，只需要确定它在每个维度上的选择，就能快速定位其架构特征和推理约束。

#### 1.5 Thinker 为什么需要向 Talker 传递 Hidden States？

这是一个初看 omni 模型架构时很自然的疑问——"既然 Thinker 已经生成了 text token，为什么 Talker 不直接拿 text token 就好？为什么还需要 hidden states？"

展开分析：
- Text token 只包含"说什么"的信息，不包含"怎么说"的信息（语气、停顿、重音）
- Hidden states 是 Thinker 在理解输入音频后的内部表示，保留了韵律和情感特征
- 同时传递两者是为了"语义一致性 + 声学质量"的双重目标

**但这里存在一个 trade-off**：同时依赖 text token 和 hidden states 意味着 Talker 必须等待 Thinker 的 decode 完成（至少是部分完成），这直接影响了推理的 TTFT（Time to First Voice）。这个 trade-off 将在后续讨论跨模态投机采样时再展开。

### 第二步：Fish Audio S2 Pro 架构

- **深度层级**：修改扩展（S2 Pro 是 SGLang-Omni 支持的核心模型之一）
- **目标**：让读者理解 S2 Pro 的 Dual AR 架构及其推理特征，为后续与 Qwen3-Omni 的对比建立基础
- **方法**：先交代模型全貌，再进入计算特征和推理流程
- **交叉引用策略**：[CUDA Graph 系列第二篇](../../torch/cuda-graph/readme-2.md)已经从 CUDA Graph 优化的视角详细分析了 S2 Pro 的架构。本步不重复那些内容，而是聚焦于**模型架构对框架抽象的影响**——即从推理框架的角度看，S2 Pro 的架构对 ModelRunner、Scheduler、KV cache 管理提出了什么要求

#### 2.1 模型全貌（先行）

- **名称**：Fish Audio S2 Pro
- **来源**：Fish Audio
- **参数量级**：~5B（Slow AR ~4B + Fast AR ~400M + vocoder）
- **用途**：TTS（Text-to-Speech），给定参考语音 + 目标文本，生成符合参考音色的语音
- **链接**：[HuggingFace](https://huggingface.co/fishaudio/s2-pro)
- **在 SGLang-Omni 中的状态**：已合并主分支，含 CUDA Graph + streaming 支持

#### 2.2 Dual AR 推理流程

从 RVQ 的层级结构（第一步 1.2）推导 Dual AR 的设计动机：

- RVQ 的第 1 层编码语义，后 9 层编码声学细节
- 自然的分工：用一个大模型（Slow AR，基于 Qwen3-4B）负责语义理解和生成第 1 层 codebook token，用一个小模型（Fast AR，4 层 Transformer ~400M）逐层生成剩余 9 层
- 每个时间步的完整流程：Slow AR decode 一步 → 取 hidden state → Fast AR 自回归 9 步 → MCF 聚合 10 个 codebook token → 作为 Slow AR 下一步的输入

用 mermaid 画出单个时间步的 decode 流程。

引用 CUDA Graph 文章中的数值特征（不重复推导，直接引用结论并标注来源）：

| 组件 | 参数量 | 层数 | 单步耗时量级 | KV Cache 类型 |
|------|--------|------|-------------|--------------|
| Slow AR | ~4B | 36 | ms 级 | SGLang Paged KV Cache |
| Fast AR | ~400M | 4 | μs 级（×9步） | Static 预分配 KV Cache |

#### 2.3 S2 Pro 对推理框架的需求（核心分析）

从 2.2 的架构推导出框架需求：

1. **标准 LLM serving 能力**：Slow AR 基于 Qwen3-4B，天然继承 continuous batching、paged KV cache、RadixAttention 等 SGLang 已有能力
2. **非标准的嵌套 AR**：Fast AR 嵌套在 Slow AR 的每个 decode step 内部，这不是独立的另一个模型调用，而是 forward 的一部分。这意味着 ModelRunner 的 `forward()` 需要能容纳这种嵌套结构
3. **双 KV Cache 共存**：Paged（Slow AR）+ Static（Fast AR）在同一个 forward 中共存，对 memory management 提出了额外要求
4. **Codec Decoder（Vocoder）**：最后一步的 vocoder 是独立的轻量级网络，需要一种 stage 概念来串联 LLM decode 和 vocoder

**关键推论**：S2 Pro 的 Dual AR 设计使得它在 LLM backbone 层面与标准 LLM 同构（structurally isomorphic），可以最大程度复用 SGLang 的已有基础设施。**这是 S2 Pro 对框架设计"友好"的根本原因。**

### 第三步：Qwen3-Omni 架构

- **深度层级**：修改扩展
- **目标**：理解 Qwen3-Omni 的 Thinker + Talker 架构，特别是 Talker 使用 DiT（Diffusion Transformer）而非 AR 的设计选择及其推理含义
- **方法**：先全貌，再计算特征，最后推导框架需求

#### 3.1 模型全貌（先行）

- **名称**：Qwen3-Omni
- **来源**：阿里通义
- **参数量级**：Thinker（多模态 LLM，具体参数量级待确认）+ Talker（DiT-based 语音合成）
- **用途**：端到端多模态理解 + 语音合成（不仅是 TTS，还支持音频/图像/视频理解）
- **在 SGLang-Omni 中的状态**：Thinker 已合并主分支；Talker 在 PR #155（`talker_jing` 分支），需 rebase
- **关键区别**：Qwen3-Omni 是一个**理解 + 生成**的全能模型，而 S2 Pro 是一个纯 TTS 模型

#### 3.2 Thinker：多模态 LLM

- 本质上是一个多模态 LLM，接受文本、图像、音频、视频输入
- 从推理框架角度：Thinker 的 prefill 阶段需要处理多模态 embedding 注入（`_inject_multimodal_embeds`），decode 阶段与标准 LLM 一致
- **与 S2 Pro Slow AR 的对比**：两者在 decode 阶段都是标准的 autoregressive LLM，区别在于 prefill 阶段——Thinker 需要处理多种模态的输入，而 S2 Pro 只需要交织 text token 和 audio token

#### 3.3 Talker：DiT-based 语音合成

这是 Qwen3-Omni 与 S2 Pro 最大的架构差异所在：

- **S2 Pro 的 Talker 用 AR**（Fast AR 逐 codebook 自回归 9 步）
- **Qwen3-Omni 的 Talker 用 Diffusion（DiT）**：接收 Thinker 的 hidden states + text tokens，通过迭代 denoising 生成 codec token

深入 DiT Talker 的计算模式：
1. 不是逐 token 生成，而是对一段时间窗口的 codec 进行迭代 denoising
2. 每次 denoising step 都是一次完整的 forward pass（不存在 KV cache 的概念，因为不是 AR）
3. Denoising 的步数是超参数（trade-off：步数多 → 质量高但慢；步数少 → 快但可能质量下降）

**关键推论**：DiT Talker 的计算模式与 AR 完全不同——没有 KV cache、没有 continuous batching 的需求（因为不是逐 token 生成）、没有 paged memory 管理。**这意味着 SGLang 的核心优化（paged KV cache、RadixAttention、CUDA Graph for decode）对 Talker 阶段完全不适用。**

#### 3.4 Thinker-Talker 的信息流与时序

- Thinker decode 生成 text token 序列 + 每步的 hidden states
- Talker 接收 hidden states + text tokens 后开始 DiT denoising
- 当前设计是串行的：Thinker 完成（或完成一个 chunk）后 Talker 才开始

**这引出了与第一步 1.5 的对应**：Thinker-Talker 之间的信息流设计直接决定了推理 pipeline 的 latency 特征。

#### 3.5 Qwen3-Omni 对推理框架的需求（核心分析）

1. **标准多模态 LLM serving**：Thinker 阶段可以复用 SGLang 的 LLM serving 能力 + 多模态 embedding 注入
2. **非 LLM 的 Diffusion serving**：Talker 阶段需要一种完全不同的执行模式——固定步数的迭代 forward，没有 KV cache，没有 continuous batching
3. **跨阶段的数据传递**：hidden states 需要从 Thinker 传递给 Talker，这涉及跨 stage 的 tensor relay（shm/nccl/nixl）
4. **流式输出**：用户期望实时听到语音，Talker 可能需要 chunk-by-chunk 地生成

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
