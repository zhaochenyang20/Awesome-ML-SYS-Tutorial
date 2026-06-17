# MOSS-TTS-Local-Transformer-v1.5 在 SGLang-Omni 中的端到端服务化优化

今天，我们正式发布并开源了 MOSS-TTS-Local-Transformer-v1.5 模型，这是我们 MOSS-TTS v1.5 系列的第二款旗舰模型，采用了 `Global transformer + local transformer`  （以下简称 Local）的架构设计，并在多项 benchmark 上取得了优异的表现。同时，我们很高兴地宣布，[SGLang-Omni](https://github.com/sgl-project/sglang-omni) Day0 支持了 MOSS-TTS-Local-Transformer-v1.5 的高性能推理。

在本文中，我们将系统梳理 SGLang-Omni team 与 MOSI Infra team 目前为止在 SGLang-Omni 这个框架上为 MOSS-TTS-Local-Transformer-v1.5 推向生产所做的每一项联合优化：从参考音频缓存、CUDA Graph 化的帧解码，到流式 vocoder 会话管理等。我们希望通过这些分享，促进 TTS 开源社区的进步。

> 有关 SGLang-Omni 针对 TTS 模型的更多推理优化工程经验，推荐阅读： [优化 TTS 推理：SGLang Omni 从性能剖析到流式化的工程经验](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/sglang-omni/tts-optimization.md)

## 模型概览

![MOSS-TTS-Local-Transformer 1.5 模型架构：Text Tokenizer 和 Audio Tokenizer 分别将文本和参考音频编码为 token 序列，由 Decoder-Only LLM（Global Transformer）处理多通道输入，每帧经 Local Transformer 生成 RVQ 码，最终通过 Audio DeTokenizer 还原为波形。](images/moss-local-transformer-arch.png)

MOSS-TTS-Local-Transformer-v1.5 由 MOSI.AI 联合 OpenMOSS 团队和上海创智学院发布，延续了 Audio Tokenizer + LLM 的端到端自回归范式，在语音编解码器、主干模型和训练规模上全面升级，目标是在 48kHz 双声道高保真、低延迟流式生成和零样本音色克隆之间取得平衡。

模型由两部分组成：

- 底层音频接口采用 **MOSS-Audio-Tokenizer-v2**。这是我们自研的高质量音频 tokenizer（encoder + decoder 合计约 **2B 参数**），Audio token 的 frame rate 为 12.5hz, 支持 0.125kbps - 4kbps 可变比特率的压缩，支持 48kHz 立体声重建，内含语义信息建模，并以 RVQ 离散表示为 TTS 建模提供稳定声学空间。评测上，MOSS-Audio-Tokenizer-v2 的重建指标（SIM、STOI、PESQ-NB、PESQ-WB）在多个比特率下领先当前主流离散音频 tokenizer（Encodec、DAC、BigCodec、SpeechTokenizer、Mimi、XCodec2.0、Higgs Audio Tokenizer 等），证明其高保真建模的能力。

- 主干 LLM 基于 **Qwen3-4B**，采用 Global Transformer + Local Transformer 架构：Global Transformer 负责文本语义、跨语种上下文、参考音频音色与韵律建模；Local Transformer 在每个声学帧内快速生成 RVQ token。当前模型使用前 12 层 RVQ、12.5Hz 帧率和 1 层 Local Transformer，形成纯自回归、天然流式的生成路径。

在功能方面，MOSS-TTS-Local-Transformer-v1.5 支持直接 TTS、续写、零样本音色克隆、时长控制与显式停顿控制（`[pause 3.2s]`），并覆盖世界主流的 31 种语言。我们的训练使用了约 **400 万小时**的多语种语音数据，最长推理音频可达 10 分钟，在多个公开评测集上的性能达到了业界领先水平：

| 评测集 | WER | SIM |
|--------|----:|----:|
| Seed-TTS-Eval | 5.10% | 69.23% |
| CV3-Eval | 7.48% | 61.59% |
| MiniMax Multilingual | 6.37% | 75.31% |
| X Voice | 20.48% | 63.00% |

> 以上 WER/SIM 基于模型级离线评测，与后文"Benchmark 汇总"节中的 serving benchmark 使用不同的 ASR 管线和评测协议，数值不可直接比较。

MOSS-TTS-Local-Transformer-v1.5 全程在阿里云自研的国产 PPU-ZW810 集群上进行千卡级训练交付[^ppu]。

[^ppu]: 此处为模型训练侧信息，与本文聚焦的 serving 优化无直接关联，仅供参考。

## 为什么 MOSS Local v1.5 对 serving runtime 构成挑战

标准 LLM serving 引擎围绕一组清晰的隐含假设构建：**单模型、单阶段 decode 循环**——每步生成一个 token，所有计算发生在同一个 Transformer 的同一次 forward pass 中，内存管理只需关注一组 KV cache 的线性增长。MOSS Local v1.5 等 TTS 模型从根本上打破了这一范式。它不是"带有额外特性的 LLM"，而是一个**多阶段解码系统**，其服务化复杂度来自各阶段间计算特性、依赖模式和资源需求的深层异构。

### 从 LLM 到多阶段解码：范式层面的差异

> 关于多阶段解码范式与传统 LLM serving 的更多系统性对比，推荐阅读 [Why SGLang-Omni](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/sglang-omni/why-sglang-omni.md)

MOSS Local v1.5 的推理是一条三阶段异构流水线：**预处理**（参考音频编码）→ **AR 引擎**（backbone + local transformer 帧级自回归）→ **流式 vocoder**（codec decoder 实时解码波形）。三个阶段由不同模型驱动、拥有独立的权重和状态空间，与 LLM prefill/decode 共享同一模型的两阶段结构有本质区别。这带来三层新的系统复杂度：

- **计算范式异构。** Backbone decode 是 memory-bound 的（与标准 LLM 一致）；local transformer 的帧内微循环（12 次 local transformer forward + 13 次顺序采样）计算量极小，瓶颈在 kernel launch 开销而非算力或带宽；codec decoder 则是 compute-heavy 的 ~1B 参数完整 forward pass。三种瓶颈模式并存，面向单一模式的优化（如 PagedAttention、tensor parallelism）无法同时适配所有阶段。
- **依赖模式分化。** AR 引擎到 vocoder 是异步解耦的（帧级流式消费），但 AR 引擎内部 backbone 与 local transformer 是同步紧耦合的（每帧 13 次顺序采样，逐 codebook 反馈）。这种帧内同步 + 帧间异步的混合依赖在 LLM 的线性链式 decode 中并无对应。
- **跨阶段显存争用。** Backbone 权重 + KV cache、codec 约 2B（encoder + decoder 合计）独立权重 + 流式解码状态、local transformer 反馈缓冲区 + 采样状态池，三者可以共享 GPU 显存，但消耗模式不同，无法用单一 `mem_fraction` 参数线性划分。

### MOSS Local v1.5 的模型特异性挑战

在上述多阶段范式差异之上，MOSS Local v1.5 的具体架构设计还带来了额外的工程挑战。

**更大的 codec。** 典型 TTS 系统将重量级 backbone 与轻量级 codec 配对（如 Higgs TTS v3 模型采用 4B 参数的 Qwen3 backbone 搭配小型 DAC decoder）。相比之下，MOSS 的 codec 显著重于典型 TTS 系统：MOSS-Audio-Tokenizer-v2 codec（encoder + decoder 合计约 2B 参数，达到 Qwen3-4B backbone 的一半）承担了参考音频编码和生成码解码为 48 kHz 立体声波形的任务。这意味着在做好 backbone 性能优化的同时，针对 codec encoder 和 decoder 两侧的优化（batching、缓存、CUDA Graph 等）也必不可少。

**多通道帧向量。** 序列中每个位置是一个 13 维向量：1 个 text 通道 + 12 个 RVQ audio codebook 通道。输入 embedding 是 13 个独立 embedding 查表的逐元素求和，而非单一表索引。这打破了 serving 引擎用固定 embedding 表做 CUDA Graph 的标准 `input_ids → embed_tokens → hidden` 路径。

**重量级参考音频编码。** 声音克隆需要将参考音频通过完整 codec encoder 编码后才能开始 AR 生成。每条参考约消耗 ~250 ms，比文本分词贵几个数量级。生产环境中固定说话人池的反复引用（常见场景）使得缓存成为刚需。

## MOSS Local v1.5 在 SGLang-Omni 三阶段流水线下的推理过程

SGLang-Omni 以三阶段异构流水线服务 MOSS Local v1.5。每个阶段由独立的调度器驱动，阶段间通过消息队列异步衔接，形成一条从文本输入到 48 kHz 立体声输出的端到端流式链路。

### 阶段一：预处理

预处理阶段在 CPU 线程池（最多 16 并发）中完成文本分词与参考音频编码，产出 AR 引擎所需的 prompt 张量。

文本侧使用 Qwen3 BPE tokenizer，将用户文本与控制 token（如 `[pause 3.2s]`）编码为 token ID 序列。音频侧将参考音频通过 codec encoder（~1B 参数）编码为 RVQ 码。编码结果与文本 token 共同组装为 `[T, 13]` 的多通道 prompt 张量：通道 0 承载 text token ID，通道 1--12 承载 RVQ codebook token ID（每个词表大小 1024）。纯文本位置将通道 1--12 填充为 `audio_pad_code = 1024`；音频帧位置则在通道 0 填入 slot token，通道 1--12 填入实际 codebook 码。

组装完成的 prompt 张量通过 `SimpleScheduler` 提交给 AR 引擎。

### 阶段二：AR 引擎——逐帧自回归生成

AR 引擎是整条流水线的核心，由 SGLang `OmniScheduler` 调度。它封装了 Qwen3-4B backbone 和 1 层 local transformer（单层因果 Transformer，使用 RoPE 位置编码和 SiLU 激活），以帧为单位执行自回归生成。

**Embedding 融合。** 进入 backbone 前，`[B, T, 13]` 输入的 13 个通道被融合为每个位置一个向量：

```
embed(pos) = text_embed(ch_0) + Σ_{i=0}^{11} audio_embed_i(ch_{i+1}) * [ch_{i+1} ≠ pad]
```

Pad code 对应的 embedding 行在权重加载时被置零，因此 mask 是隐式的。Audio embedding 表与对应的输出头共享权重（`audio_lm_heads[i].weight ≡ audio_embeddings[i].weight`），在帧内 local decode 中形成紧耦合的表示回路。

**逐帧解码。** Prefill 完成后，生成按帧循环进行。`MossTTSLocalModelRunner` 在每帧中编排以下流程：

1. **Global backbone 步。** 当前帧向量的融合 embedding 进入 36 层 Qwen3-4B backbone，执行因果注意力并使用 KV cache。Backbone 最后一层在当前位置的 hidden state 即为该帧的全局上下文向量。Backbone forward 返回的是 hidden states 而非 logits——真正的采样逻辑被推迟到 local decode 阶段。

2. **Local transformer：二元决策。** 全局 hidden state 作为 1 层 local transformer（上下文窗口 = 13 个位置）在位置 0 的输入。一个 2 路线性头输出 continue/stop logits，分别映射到 `audio_assistant_slot_token_id`（继续）和 `audio_end_token_id`（停止）。

3. **Local transformer：12 个 codebook 采样。** 若决定继续，则顺序采样各 codebook。对于 codebook `c` ∈ `[0, 11]`：
   - `audio_lm_heads[c]` 将当前 local hidden state 投影为 codebook logits `[1024]`
   - 采样一个 token（支持 temperature、top-k、top-p、可选 repetition penalty）
   - 若 `c < 11`：通过 `audio_embeddings[c]` 嵌入采样 token，送入 local transformer 的位置 `c + 1`，更新 hidden state 以供下一个 codebook 使用

4. **帧组装与反馈。** 12 个采样码加上 slot text token 构成下一个 `[1, 13]` 行。其融合 embedding 在帧解码 CUDA Graph 内部完成计算，写入 `MossTTSLocalDecodeStatePool` 的反馈缓冲区，作为下一帧 backbone 的直接输入——无需 host 端张量构造。

单帧的计算开销为：1 次 backbone forward，加上 local decode 微循环内的 1 次初始 local transformer step + 1 次二元 head 投影 + 1 次二元采样 + 12 次 codebook head 投影 + 12 次 codebook 采样 + 12 次 code embedding 查表 + 11 次 local transformer step（codebook 0--10 各一次）+ 1 次 slot embedding 查表 + 12 次 feedback 加法。仅 local decode 就涉及约 500 次顺序 kernel launch，计算量极小但全部串行执行。若不使用 CUDA Graph，kernel launch 开销将成为主导。

每帧生成完成后，`MossTTSLocalModelRunner` 将该帧的 `[1, 13]` 行（CPU 张量）作为 `OutgoingMessage` 发送至 vocoder 阶段，实现帧级流式传递。

### 阶段三：流式 Vocoder

流式 vocoder 由 `MossTTSLocalStreamingVocoderScheduler` 驱动，管理一个持久化的批量 codec 流式解码会话（`_CodecStreamSession`）。

Vocoder 为每个活跃请求分配一个流式 slot。到达的帧行被剥离通道 0 的 text token 后，按 slot 累积到 pending 缓冲区。当某个 slot 的 pending 帧数达到阈值时触发解码：首帧阈值较小（默认 5 帧，约 0.4 秒音频），以降低首包延迟；后续帧阈值较大（默认 25 帧），以提升吞吐。多个 slot 的解码请求在同一 `step()` 调用中跨请求合并批处理，共享 codec decoder 的 GPU 算力。

解码产生的音频分块（48 kHz 立体声 float32）通过 SSE 流式推送给客户端。请求结束时（收到 stop 信号），vocoder 刷新 pending 缓冲区中的剩余帧并释放 slot。

## 优化全景

基于上述三阶段流水线，SGLang-Omni team 与 MOSI Infra team 针对每个阶段的瓶颈特性设计了对应的优化方案。下表汇总了各阶段的优化项及其核心收益：

| 流水线阶段 | 优化项 | 核心思路 | 关键收益 |
|-----------|--------|---------|---------|
| **预处理** | 批量编码 | 合并并发 codec 编码请求，最多 8 条凑批 | 提升 GPU 利用率，摊薄编码开销 |
| | LRU 缓存 + Single-Flight | 内容寻址缓存 + 并发去重 | 吞吐 +32%（缓存扩容后） |
| **AR 引擎** | 双 CUDA Graph | Backbone graph + 帧解码 graph 独立捕获 | 消除帧内约 500 次 kernel launch 开销 |
| | State Pool 常驻状态 | 预分配固定地址 GPU buffer 持有逐请求状态 | 支撑 CUDA Graph 重放的地址稳定性 |
| | GPU 原生 Radix Hash | GPU 侧多项式哈希替代 CPU blake2b | 消除逐帧 D2H 同步 |
| | 采样器编译 | `torch.compile` 融合采样 kernel 链 | 吞吐 +12.3%，延迟 -11.1% |
| | 异步解码 | Launch/Resolve 阶段拆分重叠 | 修复 bs=1 正确性 bug，性能中性 |
| | Launch 向量化 | 向量化 launch 阶段的状态准备 | Launch 准备耗时 -45% |
| **流式 Vocoder** | 流式会话 + Slot 管理 | 持久化批量 codec streaming 上下文 | 帧级流式输出，TTFA 降低 5.3 倍 |
| | 双阈值合并步 | 首帧小阈值 + 稳态大阈值，跨请求合并解码 | 兼顾低 TTFA 与高吞吐 |
| | 有状态 Vocoder CUDA Graph | 原地 buffer 更新 + 按帧数捕获 graph | 每步 ~2x 加速 |
| **跨阶段** | 单卡共置内存预算 | 显式 codec_mem_reserve 隔离 AR 与 codec 显存 | 吞吐 +8.9% |

下面逐项展开介绍。

## 参考音频编码器优化

### 批量编码

参考音频编码是预处理阶段的瓶颈。Codec 编码器处理变长波形，单请求编码会导致 GPU 利用率不足。SGLang-Omni 通过 `_BatchedReferenceEncoder` 合并并发编码请求：一个后台工作线程从提交队列中取出最多 8 个条目，等待最长 4ms 凑批。同一批次内相同路径会去重（两个请求引用同一文件时 codec 只编码一次）。批次失败时回退为逐条编码并隔离错误传播 --- 一条编码失败不会影响整个批次。

参考音频时长硬性上限为 100 秒，防止异常输入阻塞编码器线程。值得注意的是，[Higgs TTS 曾尝试在线批量编码但最终移除](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/sglang-omni/tts-optimization.md)，原因是轻量级 DAC 编码器的批量收益有限，反而引起 GPU 利用率抖动和与 AR 引擎的资源争用。MOSS 的 codec encoder 约 1B 参数，单次编码约 250 ms，批量化的摊薄效果更显著；同时 max 8 条、4 ms 凑批窗口将 burst 控制在可接受范围内。

### LRU 缓存与 Single-Flight 去重

生产环境的声音克隆工作负载通常复用少量说话人，缓存编码结果可以消除冗余 codec 计算。`CachedReferenceEncoder` 叠加三层机制：

**内容寻址 key。** 缓存 key 是文件完整内容的 `xxh3_64` 哈希，而非文件路径 --- 因此重命名或移动后内容不变的文件共享同一缓存条目。Key 计算采用三层策略以避免冗余全文件读取：
1. Stat-tuple memo key（`path:size:mtime_ns:ctime_ns`）用于快速短路
2. 哨兵字节读取（从文件头部、中部、尾部各采样 8 KB）以检测 stat 相同但内容不同的覆写
3. 仅在哨兵不匹配或冷启动时才做全文件内容哈希

哨兵层至关重要：`trust_stat=False`（默认设置）确保文件被以相同大小和时间戳覆写但内容不同时不会产生过期命中。

**Single-flight 去重。** 对同一未缓存参考的并发请求共享一次进行中的编码。第一个请求成为 leader 执行编码；follower 等待 leader 的 future 结果。这防止了冷启动时多个请求引用同一说话人导致的惊群效应。

**CPU 常驻 int32 存储。** 缓存码以 `int32` 存储在 CPU 上（对 `[0, 1023]` 范围的 codebook 值无损）。每次检索均返回 `.clone().to(torch.long)`，无论缓存命中与否 --- 调用方无法修改共享条目，下游代码看到统一的 dtype/device 契约（batch-invariance）。

缓存容量经过实验调优：初始 256 条目限制在 SeedTTS 英文评测（1088 请求中有 666 个唯一参考）上导致持续驱逐，吞吐损失 32%。将上限提升至 1024（1.85 MB 主机内存）可容纳完整工作集并恢复吞吐。当前默认 8192 条目、64 MB 字节上限，为更大说话人池预留空间。`MOSS_REF_AUDIO_CACHE=0` 环境变量可完全禁用缓存以便调试。

> **Benchmark（2× H100, c=16, SeedTTS EN 1088 样本）：** 缓存从 256 扩至 1024 后吞吐 +32.0%（6.78 → 8.95 req/s），延迟均值 -24.3%（2.35s → 1.78s）。详见 [PR #778](https://github.com/sgl-project/sglang-omni/pull/778)。

## AR 引擎优化

### 双 CUDA Graph 捕获

SGLang-Omni 在引擎初始化时捕获两组独立的 CUDA Graph：

**Backbone Graph** 由 SGLang 标准基础设施为 Qwen3 backbone 捕获，覆盖带 radix cache 的 KV attention 路径和 batch size 分桶，与 SGLang 对任何因果 LM 的处理方式一致。

**帧解码 Graph** 由模型类自身捕获。每个 graph 覆盖完整的 local decode 微循环：1 次 local transformer step + 二元采样器 + 12 次顺序（head 投影 → 采样 → 嵌入 → local step）迭代 + feedback embedding 求和。Graph 按 batch size 分桶 `[1, 2, 4, 8, 16]` 捕获。不在桶列表中的 batch size 零填充至下一个桶大小，输出再裁剪回实际 batch size。

两组 graph 必须按特定顺序捕获。在创建 SGLang 基础设施期间临时禁用 backbone graph，使帧解码 graph 的捕获在稳定的设备内存状态下进行。Local transformer 的 KV cache buffer 在捕获前以 `max(桶大小)` 预分配并冻结，确保地址在重放间保持稳定。

一个关键设计选择：feedback embedding（刚生成帧的融合表示，作为下一帧的 backbone 输入）在帧 graph *内部*计算。`feedback_embed = embed_text(slot_token) + Σ embed_audio[i](code_i)` 作为 graph 重放的一部分被捕获，因此从 backbone hidden state 到下一帧输入 embedding 的整个帧解码是单次 graph launch，无需 host 端张量构造。

**Eager 回退。** 任何包含 `audio_repetition_penalty ≠ 1.0` 的请求的 batch 将回退至 eager 解码路径，因为 repetition penalty 需要逐请求的历史收集，无法在固定 graph 拓扑中捕获。超过最大 graph 桶大小的 batch 也会回退。

### Pool 常驻解码状态

AR 解码期间的逐请求状态 --- feedback embedding、采样参数、seed、生成步计数器、用于 repetition penalty 的音频 token 历史 --- 必须跨帧存活，且可在固定 GPU 地址处被 CUDA Graph 重放访问。

`MossTTSLocalDecodeStatePool` 在构造时（任何 graph 捕获之前）预分配 `P = max_running_requests + 1` 行 GPU buffer。每个活跃请求获取一个行索引；该行的 buffer 以连续 GPU 张量形式持有所有逐请求状态：

| Buffer | 形状 | dtype | 用途 |
|--------|------|-------|------|
| `feedback_embeds` | `[P, hidden]` | bf16 | 下一帧 backbone 输入 embedding |
| `{text,audio}_{temp,top_p}` | `[P]` | float32 | 采样温度与 top-p |
| `{text,audio}_top_k` | `[P]` | int64 | 采样 top-k |
| `seeds` | `[P]` | int64 | 逐请求确定性 RNG seed |
| `generation_steps` | `[P]` | int64 | 调度器已提交的帧计数器 |
| `sampling_steps` | `[P]` | int64 | Launch 侧帧计数器 |
| `audio_repetition_penalty` | `[P]` | float32 | 逐请求 repetition penalty 系数 |
| `audio_token_presence` | `[P, 12, 1024]` | bool | Repetition penalty 历史 |

行地址在构造后永不变更。Padding 行（`P - 1`）被保留且永不分配给请求 --- 作为未来 in-graph 路由的稳定 no-op 目标。

一个 staging 表（`_decode_input_embedding`）桥接 pool 与 backbone CUDA Graph：每次解码步前，feedback embedding 从 pool 行复制到 staging 表的权重矩阵中，整数行索引作为 `input_ids` 传入。Backbone 的 embedding 查表于是变成简单的表索引，将模型特定的 13 通道融合逻辑完全隔离在 backbone graph 之外。

> **Benchmark（H100, c=16, SeedTTS EN）：** Pool 重构本身对端到端吞吐为中性（噪声内），其价值在于为 CUDA Graph 重放提供地址稳定性基础。详见 [PR #745](https://github.com/sgl-project/sglang-omni/pull/745)。

### GPU 原生 Radix Hash

SGLang 的 radix cache 需要对多通道帧向量做 key 以实现 KV 复用。基线方案 --- 将行传输到 CPU、用 blake2b 哈希、再将 key 传回 --- 每个解码帧增加一次 GPU→CPU 同步。SGLang-Omni 替换为 `gpu_radix_row_hash`：在 GPU 上对每行全部 13 个通道做多项式哈希。哈希值被折叠到 special-token ID 带以下，确保不会与调度器用于 EOS 检测的控制 token 碰撞。Stop 帧行跳过哈希，保留原始 `audio_end_token_id` 供调度器的完成逻辑使用。

这消除了一次逐帧 D2H 传输。实测影响低于测量噪声（~±1.5% 跨会话方差），但移除了一个在更高吞吐下会成为瓶颈的同步点。

### 种子化无分支采样器编译

Local decode 微循环每帧执行 13 次采样操作（1 次二元 + 12 次 codebook）。每次使用 `sample_seeded_branchless`：一个 GPU 原生采样器，其 RNG 状态由 `(request_seed, frame_step, channel_index)` 确定性派生，使采样结果在不同并发度和 batch 组合下保持确定性。采样器无分支设计以保持 CUDA Graph 可捕获性（无 host 侧控制流）。

在帧 graph 捕获前用 `torch.compile(mode="max-autotune-no-cudagraphs")` 编译该采样器可融合采样 kernel 链。编译范围刻意收窄：仅编译采样器，不编译 local transformer 或 logit 投影，因为更广泛的编译在测试中改变了生成的音频码。

> **Benchmark（SeedTTS EN 1088 样本, c=16）：** 吞吐 +12.3%（5.17 → 5.81 req/s），延迟均值 -11.1%（1.54s → 1.37s），RTF -10.5%。详见 [PR #773](https://github.com/sgl-project/sglang-omni/pull/773)。

### 异步解码：基础设施而非免费午餐

SGLang-Omni 为 MOSS Local v1.5 实现了单步前瞻异步解码，将每个解码步拆分为 GPU 侧 **launch** 阶段（帧解码、采样、feedback 提交）和 CPU 侧 **resolve** 阶段（输出收集、EOS 检测、结果分发）。帧 N 的 launch 阶段与帧 N - 1 的 resolve 阶段重叠。

合入该功能的首要原因是**正确性**而非性能。拆分修复了一个具体 bug：在 batch size 1 时，基础调度器将 `next_token_ids` 别名到可变的 `output_ids` buffer 上。下一步的原地覆写在 EOS 检测前破坏了 stop token ID，导致 4096 帧失控生成。异步拆分在 launch 时 clone 设备张量并在 resolve 时恢复，防止数据竞争。

性能影响为中性 --- 所有测量均在跨会话噪声内（c=1 时 ±0.3% RTF）。Resolve 阶段约 0.06ms，相对于 ~10.4ms 的总步时间，可隐藏的尾部不足。作为对比，[Higgs TTS 的异步解码](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/sglang-omni/tts-optimization.md)取得了显著收益，因为其 delay pattern 架构在每步仍有可观的 D2H 同步开销（3 次合并为 1 次后仍需一次 staging 传输）可被重叠隐藏。MOSS Local v1.5 的情况不同：GPU 原生 Radix Hash（上节）已将逐帧 D2H 同步完全消除，异步解码已无可隐藏的 host 端尾部。异步解码默认关闭（`enable_async_decode=False`），且仅在 batch size ≥ 2 且无 audio repetition penalty 时生效。

## 流式 Vocoder

### 原生流式会话与 Slot 管理

MOSS-Audio-Tokenizer-v2 的 decoder 原生支持有状态流式解码：给定一系列 codec 帧，它可以增量解码，产生的音频分块与离线全序列解码 bit 级一致。这与常见的 overlap-add 流式方案有本质区别——例如 [Higgs TTS 的流式 vocoder](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/sglang-omni/tts-optimization.md) 采用窗口化解码（stride / overlap / holdback），需通过交叉淡入淡出（crossfade）拼接相邻分块以避免边界伪影，无法保证与离线解码一致。MOSS codec 的因果卷积和有界上下文注意力机制使其天然支持增量解码，无需重叠或淡入淡出，因此流式输出可做到 bit 级一致，简化了质量验证。SGLang-Omni 通过 `_CodecStreamSession` 利用这一特性 --- 一个持久化的批量 `codec.streaming()` 上下文，存活于引擎的整个生命周期。

会话管理一个固定的 slot 池，分为两个通道：
- **Stream slot**（默认 8 个）：通过 `acquire()`/`release()` 分配给活跃流式请求
- **Offline slot**：保留给非流式请求，防止非流式请求因等待流式 slot 释放而死锁

单次 `step()` 调用中所有 slot 必须提供相同数量的帧（codec 的批量 forward 要求统一 `T`）。执行掩码选择哪些 slot 参与每一步；非参与 slot 在 codec 层被 mask 掉，而非零填充。

### 双阈值合并步

流式解码使用双阈值设计同时优化首包音频时间（TTFA）和稳态吞吐：

**初始阈值**（`initial_chunk_frames`，默认 5 帧 = 12.5 Hz 下 0.4s）：首次解码步触发前的最小累积帧数。较小的初始阈值以较短的首个分块为代价最小化 TTFA。

**稳态阈值**（`stream_chunk_frames`，默认 25 帧 = 2s）：后续所有步使用。较大的分块摊薄每步开销并改善分块边界处的 codec 质量。

每步完成后阈值重置为稳态值。初始阈值在首个分块到达时锁存，可通过 API 参数逐请求覆盖。

**合并步（coalesced steps）**将多个请求合并到单次批量解码中。当任一请求跨越其阈值（变为"到期"），调度器检查其他已分配 slot 的请求能否加入。一个未到期请求可以加入的条件是：已发出至少一个分块且已累积至少 `join_floor` 帧（`max(1, min(initial_chunk_frames, stream_chunk_frames))`，即初始阈值与稳态阈值的较小值，下限为 1 帧）。合并步中所有参与者解码相同数量的帧（所有参与者待处理帧数的 `min`，上限 `max_step_frames = 100`）。这种批量化在请求间摊薄 codec 的每次调用开销，所有活跃 slot 共享一次批量 D2H 传输。

**TTFA 结果。** 并发 1 时流式实现 5.3 倍 TTFA 提升（0.70s → 0.13s）。并发 4 时提升 1.5 倍（0.83s → 0.56s）。更高并发下 TTFA 优势收窄，因为 codec GPU 时间在更多活跃 stream 间共享。并发 16 时流式吞吐较离线下降约 29%（3.81 vs 5.37 req/s），体现了流式 vocoder 步更频繁触发对 AR 引擎 GPU 时间的竞争。流式 WER 2.53%，非流式 2.74%，差异不足 0.3 个百分点，无质量回归。

> **Benchmark（SeedTTS EN 1088 样本, c=16）：** 流式吞吐 3.81 req/s vs 离线 5.37 req/s（-29%）；流式 WER 2.53% vs 离线 WER 2.74%。低并发（c=1）下流式吞吐反而提升 21%（1.42 → 1.72 req/s）。详见 [PR #753](https://github.com/sgl-project/sglang-omni/pull/753)。

吞吐权衡是真实的：在中高并发下，流式模式比离线批量解码更早、更频繁地触发 vocoder 步，消耗原本可用于 AR 生成的 GPU 时间。这是流式 TTS 中延迟与吞吐之间的固有张力。

### 有状态 Vocoder CUDA Graph

> 以下优化已在 [PR #798](https://github.com/sgl-project/sglang-omni/pull/798) 中实现并验证，已合入主分支。

Codec decoder 每步启动数百个 kernel（7 个 Transformer block 共 104 个 attention 层带 KV cache 和 RoPE position offset，加上卷积和激活）。对 vocoder 步做 CUDA Graph 可消除这些 launch 开销。

核心挑战在于有状态性：codec 的流式 decoder 在步间维护 KV cache 和 RoPE position offset。标准 CUDA Graph 在捕获时记录张量地址并在重放时使用这些固定地址。但 codec 的 attention 层原先每步将 KV cache 张量重新赋值为新分配的 buffer --- graph 将在过期地址上重放。

修复方案将张量重新赋值替换为原地 buffer 更新（`scatter_`/`index_copy_`），使 buffer 地址在步间保持稳定。基于此，按 `T` 值捕获 CUDA Graph（每个不同帧数一个 graph，实际约 ~13 个），batch 宽度固定为会话的总 slot 数。

每步解码加速显著：

| 帧数 (T) | Eager | Graphed | 加速比 |
|-----------|-------|---------|--------|
| 5（初始分块） | 65.8 ms | 30.7 ms | 2.14x |
| 4 | 66.3 ms | 30.1 ms | 2.20x |
| 8 | 65.6 ms | 34.0 ms | 1.93x |
| 100（离线） | 222.9 ms | 215.3 ms | 1.04x |

短分块（流式场景的常见情况）受益最大，因为其计算量与 launch 开销之比最低。端到端流式吞吐在并发 8 下提升 16--29%，并发 16 下提升 13--22%，vocoder 耗时在并发 8 下减少 40%。

输出与 eager 解码 bit 级一致（经 1600 帧滑动窗口比对和 15 个单元测试验证）。实现具有安全降级能力：当显存不足（低于 `cuda_graph_min_free_gb`，默认 3 GB）、捕获 OOM 或重放出错时，会话透明回退至 eager 解码。

> **Benchmark（H100, 流式 SeedTTS EN）：** 端到端流式吞吐 c=8 +16--29%，c=16 +13--22%；vocoder 耗时 c=8 -40%。每步数据见上表。详见 [PR #798](https://github.com/sgl-project/sglang-omni/pull/798)。

## 单卡共置内存预算

单 GPU 部署时，codec（预处理编码器和 vocoder 解码器共用）与 AR backbone 共享同一块 GPU 显存。SGLang 默认的内存 profiling 面向独立 LLM 部署设计，不考虑共置的 codec。

SGLang-Omni 引入显式内存预算：
- `total_gpu_memory_fraction = 0.90`（10% 保留给系统/驱动开销）
- `codec_mem_reserve = 0.05`（vocoder codec 运行时分配余量）
- AR 引擎有效占比：`0.90 - 0.05 = 0.85`

预处理 codec 在同一进程中首先加载，其显存占用被 SGLang 的进程级 profiling 自动捕获 --- 无需单独预留。`codec_mem_reserve` 仅覆盖 vocoder 的运行时分配模式（activation buffer、streaming 状态）。

双 GPU 部署时（codec 在 `cuda:1`，AR 在 `cuda:0`），共置预算机制被跳过。AR 引擎的 `mem_fraction_static` 由 pipeline config 直接设为 `0.85`（`stages.py` 中的 fallback 值 `0.6` 会被 config 覆盖）。

仅此项预算优化在单卡共置配置上就带来 8.9% 的吞吐提升，原因是此前 SGLang 过度分配 KV cache 池，为 codec 留下的余量不足，在内存压力下迫使更频繁的请求回退。

> **Benchmark（单卡共置, c=8, SeedTTS EN 1088 样本）：** 吞吐 +8.9%（4.77 → 5.20 req/s），RTF 均值 -8.4%（0.40 → 0.37）。详见 [PR #810](https://github.com/sgl-project/sglang-omni/pull/810)。

## Benchmark 汇总

### 逐项优化增量

以下汇总各优化项在其引入时的单独测量结果。所有测量均基于 SeedTTS 英文评测（1088 样本），除非另行标注。硬件和并发度在不同实验间有差异 --- 每行标明其条件。

| 优化项 | 硬件 | 并发度 | 指标 | 结果 | 来源 |
|--------|------|--------|------|------|------|
| **参考缓存 (256 → 1024)** | 2x H100 | 16 | 吞吐 | +32.0% (6.78 → 8.95 req/s)† | [PR #778](https://github.com/sgl-project/sglang-omni/pull/778) |
| | | | 延迟均值 | -24.3% (2.35s → 1.78s) | [PR #778](https://github.com/sgl-project/sglang-omni/pull/778) |
| **采样器编译** | 未记录 | 16 | 吞吐 | +12.3% (5.17 → 5.81 req/s) | [PR #773](https://github.com/sgl-project/sglang-omni/pull/773) |
| | | | 延迟均值 | -11.1% (1.54s → 1.37s) | [PR #773](https://github.com/sgl-project/sglang-omni/pull/773) |
| | | | RTF | -10.5% (0.364 → 0.326) | [PR #773](https://github.com/sgl-project/sglang-omni/pull/773) |
| **内存预算** | 未记录 | 8 | 吞吐 | +8.9% (4.77 → 5.20 req/s) | [PR #810](https://github.com/sgl-project/sglang-omni/pull/810) |
| | | | RTF 均值 | -8.4% (0.40 → 0.37) | [PR #810](https://github.com/sgl-project/sglang-omni/pull/810) |
| **流式** | 未记录 | 1 | TTFA | 0.13s（较离线 0.70s 降低 5.3 倍） | [PR #753](https://github.com/sgl-project/sglang-omni/pull/753) |
| | | 4 | TTFA | 0.56s（较离线 0.83s 降低 1.5 倍） | [PR #753](https://github.com/sgl-project/sglang-omni/pull/753) |
| | | 16 | WER | 2.53% (离线 2.74%) | [PR #753](https://github.com/sgl-project/sglang-omni/pull/753) |
| | | 16 | 吞吐 | -29% (5.37 → 3.81 req/s) | [PR #753](https://github.com/sgl-project/sglang-omni/pull/753) |
| **Vocoder CUDA Graph** | H100 | 8, 流式 | 吞吐 | +16--29% | [PR #798](https://github.com/sgl-project/sglang-omni/pull/798) |
| | | B=16, T=5 | 步延迟 | 30.7 ms (2.14x) | [PR #798](https://github.com/sgl-project/sglang-omni/pull/798) |
| **Launch 向量化** | 合成测试 | BS=64 | Launch 准备 | -45% (80us → 44us) | [PR #759](https://github.com/sgl-project/sglang-omni/pull/759) |
| **State pool** | H100 | 16 | 端到端吞吐 | 中性（噪声内） | [PR #745](https://github.com/sgl-project/sglang-omni/pull/745) |
| **异步解码** | H100 | 1--4 | 端到端 | 中性（噪声内） | [PR #758](https://github.com/sgl-project/sglang-omni/pull/758) |

> † 参考缓存行的 6.78 req/s 为 cache_size=256（持续驱逐）时的吞吐，8.95 req/s 为扩容至 1024 后的吞吐，二者是同一实验内的对比。
>
> **注意：** 各实验硬件配置不同（2x H100、单卡共置等），部分 PR 未在 body 中记录硬件型号。数据不可直接合并为单一累积加速比。"未记录" 表示 PR body 中未明确标注硬件信息。

### 最终综合性能

以下数据来自 [PR #798](https://github.com/sgl-project/sglang-omni/pull/798) 合入后的 CI 全量评测（SeedTTS 英文评测，1088 样本，2× GPU，并发 16），代表全部优化叠加后的端到端系统性能。ASR 使用 Qwen3-ASR-1.7B，Speaker Similarity 使用 WavLM-Large finetune。CI 运行在阿里云自托管 runner 上，GPU 型号未在日志中记录[^ci-gpu]。

[^ci-gpu]: CI runner 环境为 CUDA 13.0、2× GPU，但 nvidia-smi 输出未包含在 CI 日志中，GPU 型号无法从现有日志确认。逐项优化增量表中标注硬件的条目（如 PR #778 的 2× H100）来自 PR body 的显式标注，已验证为 H100。

**非流式模式（Non-streaming，c=16）**

| 指标 | 值 |
|------|-----|
| 完成请求 / 失败请求 | 1088 / 0 |
| 吞吐 (req/s) | 5.976 |
| 音频吞吐 (s/s) | 26.303 |
| 延迟均值 / 中位数 | 2.669s / 2.593s |
| 延迟 p95 / p99 | 3.384s / 4.149s |
| RTF 均值 / 中位数 | 0.6443 / 0.6029 |
| RTF p95 / p99 | 0.9991 / 1.1972 |
| WER (corpus) | 1.75% |
| WER (逐样本均值) | 1.82% |
| WER (排除 >50% 异常值后 corpus) | 1.67% |
| Speaker Similarity 均值 | 64.18 |
| 音频平均时长 | 4.401s |

**流式模式（Streaming，c=16）**

| 指标 | 值 |
|------|-----|
| 完成请求 / 失败请求 | 1088 / 0 |
| 吞吐 (req/s) | 2.909 |
| 音频吞吐 (s/s) | 12.804 |
| 延迟均值 / 中位数 | 5.474s / 5.488s |
| 延迟 p95 / p99 | 6.166s / 6.417s |
| RTF 均值 / 中位数 | 1.3216 / 1.2526 |
| RTF p95 / p99 | 1.9746 / 2.1675 |
| TTFA 均值 / 中位数 | 4.616s / 4.658s |
| TTFA p95 / p99 | 5.227s / 5.462s |
| Inter-chunk 间隔均值 / p95 | 0.109s / 0.371s |
| WER (corpus) | 2.14% |
| WER (逐样本均值) | 2.23% |
| WER (排除 >50% 异常值后 corpus) | 1.72% |
| 平均音频分块数 | 8.82 |

**Vocoder CUDA Graph 逐步加速（B=16）**

| 帧数 (T) | Eager (ms/步) | Graphed (ms/步) | 加速比 |
|-----------|--------------|----------------|--------|
| 4 | 66.3 | 30.1 | 2.20x |
| 5（初始分块） | 65.8 | 30.7 | 2.14x |
| 8 | 65.6 | 34.0 | 1.93x |
| 13 | 65.4 | 40.4 | 1.62x |
| 25 | 74.8 | 58.3 | 1.28x |
| 100（离线） | 222.9 | 215.3 | 1.04x |

**流式 vs 非流式一致性验证：** 通过（streaming 与 non-streaming 输出 artifact 比对 PASSED）。

> **注意：** 流式模式下 TTFA（首包音频时间）在 c=16 高并发下较高，主要因为请求排队和 vocoder GPU 时间在多个活跃 stream 间共享。低并发场景下 TTFA 显著更低（参见逐项优化增量中 PR #753 的数据）。流式 RTF > 1 表示在 c=16 满载下 vocoder 步更频繁地触发、消耗了 AR 引擎的 GPU 时间，是流式 TTS 中延迟与吞吐之间的固有权衡。


## 生产部署

### 优雅降级与开关

每项优化都有保持正确性的回退路径：

- **CUDA Graph 回退。** Backbone 和帧解码 graph 在不支持的 batch size 或带 repetition penalty 的请求时回退至 eager 执行。Vocoder CUDA Graph 在显存不足、捕获失败或重放出错时回退至 eager。所有回退对客户端透明。
- **参考缓存开关。** `MOSS_REF_AUDIO_CACHE=0` 完全禁用缓存，强制每个请求走批量编码器。适用于排查疑似缓存一致性问题，无需修改配置文件，重启服务即可生效。
- **异步解码默认关闭。** `enable_async_decode=False`。启用时在 batch size 1 或 repetition penalty 生效时回退为同步执行。
- **流式 vocoder slot 耗尽。** 请求到达时若无可用 stream slot，vocoder 累积全部帧并在生成完成后离线解码。客户端仍能收到完整音频，只是没有渐进式投递。
- **KV cache 驱逐恢复。** 当 SGLang radix cache 在内存压力下驱逐 KV 状态时，model runner 从已存储的 output rows 重建音频历史并重新 prefill，防止 KV 丢失导致的生成伪影。
- **Chunked prefill 正确性修复。** PR #745 修复了一个静默正确性 bug：在 `moss_tts_local` 路径下，`output_processor` 在 chunked prefill 时无条件将中间 chunk 的数据传给 model adapter，导致垃圾 feedback embedding 进入队列。当 prompt 超过 chunk 阈值时会静默产生错误音频，无报错无崩溃。修复方案参照 Higgs 路径已有的 `is_chunked` 守卫逻辑，并添加了覆盖该场景的单元测试。

### 可观测性

- **参考缓存指标。** 命中、未命中和 single-flight 合并计数器每 60 秒带速率日志记录。持续低命中率信号工作集溢出（需提升 `ref_audio_cache_max_items`）；高合并率表示并发重复流量（突发模式下正常）。
- **Vocoder 错误隔离。** 合并步失败时记录错误并中止所有参与请求，各自收到独立的错误消息。失败请求被清理（释放 slot、清除 stream 状态）且不影响未参与的请求。
- **帧解码日志。** 每个解码步产生一个 `MossTTSLocalDecodeJournal` 快照（请求 ID、pool 行、生成的行），在异步解码的前瞻窗口内存活。提供逐步审计轨迹以调试生成异常。

### Canary 指标

生产上线时，建议在灰度/canary 阶段重点监控以下指标，以及早发现优化回退或资源异常：

- **WER/CER**：在留出的评测集上与离线解码基线对比。流式解码应在 ±0.5 个百分点以内。
- **RTF（实时因子）**：作为逐请求计算成本的代理指标。回退或显存压力会导致回归。
- **TTFA（首包音频时间）**：流式请求的关键指标。目标：低并发下 sub-200ms，生产并发下 sub-600ms。
- **吞吐（req/s）**：持续负载下与同并发度的非流式基线对比。
- **缓存命中率**：固定说话人工作负载下稳态应超过 90%。
- **p95/p99 延迟**：捕捉因 graph 回退或缓存未命中导致的尾延迟回归。

## 局限性与未来工作

**原生 pool 常驻全帧 graph。** 当前帧解码 CUDA Graph 使用私有静态 buffer，每帧需要 H2D 拷入采样参数和拷出生成码。下一步是直接从 pool 常驻 buffer（地址稳定、已预分配）读取参数，并在捕获区域内将输出写回 pool，完全消除拷贝开销。

**有状态 vocoder CUDA Graph。** 已合入（[PR #798](https://github.com/sgl-project/sglang-omni/pull/798)），将每步约 2 倍加速带入默认 serving 路径，输出 bit 级一致并支持 eager 安全回退。进一步优化方向包括动态 buffer 大小适配和 multi-stream 并行解码。

**流式高并发调度。** 当前并发 4--16 的流式吞吐因更频繁、更小的 vocoder 步而相比离线批量解码有损耗。正在探索自适应分块大小（高负载下使用更大分块）和优先级感知的 slot 调度，以在不牺牲 TTFA 的前提下恢复吞吐。

**torch.compile 应用于编码器和 backbone。** Codec 编码器和 Qwen3 backbone 尚未编译。编码器编译潜力尤其大，考虑到每次参考的编码成本（约 250 ms / H100）。Backbone 编译已评估但推迟，因为 CUDA Graph 已捕获大部分 kernel launch 节省，且 compile 的冷启动惩罚对首个请求影响显著。

**更广泛的 benchmark 覆盖。** 当前测量集中于 SeedTTS 英文。中文和多语言评测、长语音生成（>30s）、多说话人并发工作负载和不同参考音频时长需要系统性覆盖，以验证优化收益的泛化性。

## 总结

通过 SGLang-Omni team 与 MOSI Infra team 的联合优化，MOSS-TTS-Local-Transformer-v1.5 在 SGLang-Omni 三阶段流水线下实现了多项关键性能提升。全部优化叠加后的最终综合基准（PR #798 CI，SeedTTS EN 1088 样本，2× GPU，c=16）：非流式吞吐 5.976 req/s、RTF 均值 0.64、WER 1.75%；流式吞吐 2.909 req/s、inter-chunk 均值 0.109s、WER 2.14%。各项优化的独立贡献包括：参考音频缓存扩容 +32% 吞吐、采样器编译 +12.3% 吞吐、有状态 vocoder CUDA Graph 每步约 2 倍加速、单卡内存预算优化 +8.9% 吞吐。每项优化均保留正确性回退路径，确保生产环境的稳定性。我们将持续推进上述未来工作方向，并期待社区的参与和反馈。
