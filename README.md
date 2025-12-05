# Awesome-ML-SYS-Tutorial

## [English README](./README-eng.md) | [简体中文](./README.md)

My learning notes/codes for ML SYS.

一直以来对 ML + SYS 很感兴趣，苦于本科没有学好 ML，更没学好 SYS，但是读博了觉得自己应该可以在这方面试一试。

有如此打算，一来是我发觉组里很多能力出众的高年级学长们做的是 ML Theory + Application。不过，真的把一个 Theory 落到一个良好的 Application 上，即便是他们这样让我敬佩的 theory researcher，也有着一定挑战。在我入学前，组里有两篇让我眼前一亮的工作 [SPIN](https://github.com/uclaml/SPIN) 和 [SPPO](https://github.com/uclaml/SPPO)。工作本身都有非常棒的价值，但是如果在工程/系统上优化好，想来可以有更好的影响力。

此外，博士入学前的暑假，我和组里同学做了一个 In-context Learning for Agent 的工作 [COPS](https://github.com/uclaml/COPS)，比较符合我的审美。我们就两个人主力干活，一个大哥推理论，而我负责在工程/系统上实现。这种工作模式让我的体感非常舒适，基于此，我甚至得出一个粗糙的结论：

$$
\dfrac{\text{Theory}+\text{System}}{2}=\text{Application}
$$

这就是我想做 ML + SYS 的初衷了。所以从 2024 年的夏季开始，我开始慢慢上手 ML + SYS 这个尚且方兴未艾的领域。需要学习的实在太多了，有的在一些平台（譬如知乎和 HuggingFace Blog）上已经有了很好的资料，但是其他部分仍有所欠缺。所以，这个 repo 主要记载了我自己的一些学习笔记/读后感/思索/参考过的资料 etc，我姑且按照自己的大版图进行分类，也欢迎大家 PR。每一个大的板块，倒叙阅读就是我的学习过程，欢迎大家参考此路径上手。

## 未公开部分

之前的笔记大多写于 2024 年年底，经过了半年时间，我的仓库已略年久失修。一方面我自己更多在项目中负责推动 + delivery，反而自己很少写代码；另一方面，多多少少不少朋友向我们的仓库贡献了笔记，但我完全没有来得及整理。这段时间会不断完成整理并发布。下方文章列表中标记为 [Pending Review] 的文章尚未 review，欢迎大家一起 review！

## RLHF System 开发笔记

### slime 框架

- [Support FSDP2 as A Training Backend for slime](./rlhf/slime/fsdp/readme.md)：在 slime 中新增了 FSDP 作为训练后端，并与 Megatron 完成对齐。FSDP 能够更加灵活支持诸如 Qwen3-Next/gpt-oss 等架构创新的模型，并且有助于我们进一步支持 VLM RL。同样刊载[英文版本](./rlhf/slime/fsdp/readme_en.md)和[知乎](https://zhuanlan.zhihu.com/p/1979141713449742500)。
- [Unified FP8: Moving Beyond Mixed Precision for Stable and Accelerated MoE RL](./rlhf/slime/fp8/readme.md)：在 RL 中完全使用 FP8 进行采样（Rollout）和训练（Training），同样刊载[英文版本](./rlhf/slime/fp8/readme_en.md)和[知乎](https://zhuanlan.zhihu.com/p/1974681194017865986)。
- [Power Up Speculative Decoding In Reinforcement Learning](./rlhf/slime/spec/readme.md)：将 speculative decoding 引入到了 RL 的采样流程中，在 batch size 合适的情况下，采样速度得到了显著提升；并且，draft model 也会在训练过程中更新。相较于冻结 draft model 的做法，accepted length 持续维持在较高水平，产生长期稳定的正收益。同样刊载[英文版本](./rlhf/slime/spec/readme-en.md)。
- [深入浅出 slime RL 框架的优雅设计与源码](./rlhf/slime/code-walk-through/readme.md)：slime 源码赏析，同样刊载于[知乎](https://zhuanlan.zhihu.com/p/1946402397409740613)和[英文版本](./rlhf/slime/code-walk-through/readme_en.md)。
- [Pending Review] [slime FSDP Setup Guide](./rlhf/slime/fsdp/release_log/setup_fsdp.md)：记录如何在 slime 上测试 FSDP，包括 H 卡和 B 卡，以及 Colocate 和 Disaggregated 两种 placement 方式。
- [Pending Review] [PPO 中 GAE 的分 chunk 并行计算（基于 slime 的实现）](./rlhf/slime/batch-GAE/ppo-gae-chunk.md)：将标准 GAE 的后向递推改写为基于 chunk 的并行前缀扫描，在长序列场景下大幅缓解 GAE 计算瓶颈，在 slime 中实现约 100×–300× 加速。同样刊载于[知乎](https://zhuanlan.zhihu.com/p/1975237289425798560)。

### verl 框架

- [通过 Torch Memory Snapshot 分析 VLM RL 训练中的显存泄露问题](./torch/mem-snapshot/readme.md)：分析 SGLang 的显存泄露问题，以及解决方案，同样刊载于[知乎](https://zhuanlan.zhihu.com/p/1943202817247519535)和[英文版本](./torch/mem-snapshot/readme-en.md)。
- [Latency optimization for weight updates](./sglang/latency-accelerate-for-weight-updates/readme.md)：一次对效率的 debug 过程，同样刊载于[记一次对 SGLang weight update latency 的优化](https://zhuanlan.zhihu.com/p/9908228168)。
- [深入浅出理解 verl 源码（初始化）](./rlhf/verl/multi-turn/code-walk-through/readme.md)：同样刊载于[知乎](https://zhuanlan.zhihu.com/p/1920751852749849692)，还有[英文版本](./rlhf/verl/multi-turn/code-walk-through/readme_EN.md)。
- [深入浅出理解 verl 源码（Rollout）](./rlhf/verl/multi-turn/code-walk-through/readme-2.md)：同样刊载于[知乎](https://zhuanlan.zhihu.com/p/1923349757566388159)，还有[英文版本](./rlhf/verl/multi-turn/code-walk-through/readme-2-EN.md)。
- [Pending Review] [深入浅出理解 verl 源码（Make Experience）](./rlhf/verl/multi-turn/code-walk-through/readme-3.md)：分析 verl 中 make experience 部分的逻辑。
- [AgentLoop 源码浅析](./rlhf/verl/multi-turn/code-walk-through/readme-6.md): 分析 verl 中基于 AgentLoop 的 multi-turn RL 的实现。
- [verl 参数速览](./rlhf/verl/multi-turn/code-walk-through/readme-5.md)：verl 参数速览，同样刊载于[知乎](https://zhuanlan.zhihu.com/p/1925041836998783250)，还有[英文版本](./rlhf/verl/multi-turn/code-walk-through/readme-5-EN.md)。
- [从 tokenizer 视角来分析 Agentic 多轮训练的复杂性](./rlhf/verl/multi-turn/fast_tokenization/multiturn_tokenization_and_masking_ZH.md)：同样刊载于[知乎](https://zhuanlan.zhihu.com/p/1917126584806139373)和[英文版本](./rlhf/verl/multi-turn/fast_tokenization/multiturn_tokenization_and_masking.md)。
- [Pending Review] [DAPO Dynamic Filtering 实现与 Batch Size 解析](./rlhf/verl/multi-turn/code-walk-through/dapo.md)：探索通过将 prompt 补齐到更小的 batch size 实现更高的并行度。
- [系统性分析 verl multi-turn training 的时间消耗](./rlhf/verl/multi-turn/tool_examples/profile.md)：verl 多轮交互与工具调用 profile 分析，还有[英文版本](./rlhf/verl/multi-turn/tool_examples/profile_en.md)和[知乎](https://zhuanlan.zhihu.com/p/1929748460212552414)。
- [SGLang, verl, OpenBMB 与清华大学团队联合开源：在主流 RLHF 框架上首次支持多轮交互与工具调用](./rlhf/verl/multi-turn/release_log/verl-multiturn-rollout-Release_ZH.md)：在主流 RLHF 框架上首次支持多轮交互与工具调用，同样刊载于[知乎](https://zhuanlan.zhihu.com/p/1906007821889283171)。
- [Search-R1 & veRL-SGLang: Train LLMs with Multi-Turn RL to Reason and Call a Search Engine](./rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like_ZH.md)：整合 Search-R1 framework 到 verl-sglang 生态，同样刊载于[知乎](https://zhuanlan.zhihu.com/p/1912156329751081620)。
- [SGLang-veRL Server：从 Engine 到 Server，我们需要更灵活的 RLHF rollout 接口](./rlhf/verl/server-based/veRL-server-based-rollout.md)：为了实现更复杂的 RLHF 系统，我们逐步将 veRL 当中的 rollout engine 替代为 rollout server，同样刊载于[知乎：SGLang-veRL Server](https://zhuanlan.zhihu.com/p/1890631652486665464)。
- [HybridFlow veRL 原文浅析](./rlhf/verl/readme.md)：SGLang 的 hybrid engine 的原理与实现，同样刊载于[知乎：HybridFlow veRL 原文浅析](https://zhuanlan.zhihu.com/p/24682036412)。

### OpenRLHF 框架

- [图解大模型RLHF系列之：人人都能看懂的PPO原理与源码解读](https://zhuanlan.zhihu.com/p/677607581)以及[图解OpenRLHF中基于Ray的分布式训练流程](https://zhuanlan.zhihu.com/p/12871616401)：猛猿小姐姐的非常好的 RLHF 入门资料，看了之后会对 RLHF 的计算流以及 OpenRLHF PPO 的框架有很好的理解，我自己也补充了写自己的理解在 [RLHF 的计算流](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/tree/main/rlhf/OpenRLHF#rlhf-%E7%9A%84%E8%AE%A1%E7%AE%97%E6%B5%81)。
- [浅析以 OpenRLHF 为代表的 post-training 系统的计算流程](./rlhf/OpenRLHF/readme.md)：基于猛猿小姐姐的文章再做补充，Github native 渲染的巨烂，甚至看[知乎](https://zhuanlan.zhihu.com/p/16370000391)好了。

### AReal 框架
- [AReal Code Walk Through](./rlhf/areal/code-walk-through_CN.md) AReal 源码赏析，同样刊载于 [英文版本](./rlhf/areal/code-walk-through_CN.md)。

### 系统设计与优化

- [RL 系统深思：深入理解权重更新机制](./rlhf/sys-design/readme-1.md)：半年工作的总结，深入理解权重更新机制，同样刊载于[知乎](https://zhuanlan.zhihu.com/p/1925210722704531547)和[英文版本](./rlhf/sys-design/readme-1-EN.md)。
- [RL 系统深思：FSDP 训练后端](./rlhf/sys-design/readme-2.md)：讨论 FSDP 的原理和实现，以及分析 verl 的 FSDP 使用。同样刊载于[知乎](https://zhuanlan.zhihu.com/p/1929115059113693341)和[英文版本](./rlhf/sys-design/readme-2-en.md)。
- [Pending Review] [RL 系统深思：Megatron](./rlhf/sys-design/readme-3.md)：Megatron 的基本特性浅析，重点分析 Megatron 在 RL 框架中的使用。
- [扩展 OpenRLHF 的推理引擎](./rlhf/OpenRLHF/develop-log.md)：将 SGLang 接入到 OpenRLHF 的开发笔记，整个过程非常痛苦，而且目前还有 nccl hang error，已经直接联系了 deepspeed core contributor 在修复了。
- [Pending Review] [SGLang as rollout engine of GRPO trainer](./rlhf/GRPO/SGLang_GRPO.md)：介绍如何将 SGLang 作为 TRL 中 GRPO Trainer 的推理后端，GRPO 是 PPO 的变体，在优化数学推理能力的同时优化 PPO 的内存使用。

### 算法与理论

- [Pending Review] [Learning to Reason under Off-Policy Guidance](./rlhf/partial-rollout/Learning_to_Reason_under_Off-Policy_Guidance.md)：使用离线策略辅助在线学习的 LUFFY 框架，通过将 off-policy 推理轨迹与 on-policy rollout 结合，动态平衡模仿与探索。
- [Kimi K1.5: Long Context RL 的成功实践](./rlhf/partial-rollout/readme.md)：Long Context RLHF 的工业级实现，一直很喜欢 kimi 团队的技术报告，同样刊载于 [Kimi K1.5: Long Context RL 的成功实践](https://zhuanlan.zhihu.com/p/1894282607325344277)。
- [Rule-based Reward](https://zhuanlan.zhihu.com/p/13211508979)：这篇只有知乎，浅浅写了写，老实说原文写的我并不太喜欢，但是 determined reward 确实 charming。
- [SWE-Bench：如何构造 LLM 时代的优秀 Benchmark](https://zhuanlan.zhihu.com/p/16292266518)，基于 SWE-Bench 的论文阅读笔记，如何构造好的 benchmark 以为 post-training 提供细粒度 reward，是永恒且美妙的话题。
- [浅析主流 Alignment 算法与 NeMo-Aligner 框架](https://zhuanlan.zhihu.com/p/5220718268)


## SGLang 学习笔记

### 核心架构与优化

- [SGLang Code Walk Through](./sglang/code-walk-through/readme.md)：一个请求被 SGLang Engine 处理的全过程，还有一些 part 没有完成，但是大多地方已经 okay，也让很多 SGLang begginer 就此开始。这里还有[中文版本](./sglang/code-walk-through/readme-CN.md)。
- [Walk Through SGLang / VLLM Worker](./sglang/sglang-worker/readme.md)：SGLang 的代码不完全解析，同样刊载于 [Walk Through SGLang / VLLM Worker](https://zhuanlan.zhihu.com/p/6363614076)，这次我们还贴心提供了[英文版本](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/sglang-worker/readme.md)。更详细的解析应该参考 [SGLang Code Walk Through](./sglang/code-walk-through/readme.md)，这个只是辅助看看。
- [Walk Through SGLang Scheduler](./sglang/sglang-scheduler/readme-CN.md)
- [Pending Review] [SGLang Scheduler Evolution](./sglang/scheduler-evolution/SGLang%20Scheduler%20技术变迁.md)：详细介绍了 SGLang Scheduler 从串行到 CPU / GPU overlap 的技术演进及相关组件，对比前代 overlap Scheduler 和当前引入多 CUDA stream 与 FutureMap 的 overlap Scheduler。可到知乎查看[文章](https://zhuanlan.zhihu.com/p/1969077475129688722)
- [Pending Review] [KV Cache Code Walkthrough](./sglang/kvcache-code-walk-through/readme.md)：KV cache 管理实现的概览，从 Scheduler 组件开始，详细说明 prefill 和 decode 阶段中 KV cache 和内存池的更新过程。
- [Pending Review] [SGLang 多模态请求生命周期：以 Qwen2.5-VL 为例的架构级深度解析](./sglang/code-walk-through/multimodal_request_lifecycle.md)：以 Qwen2.5-VL 为参考模型，提供对 SGLang 框架内多模态请求处理流程的详细剖析。
- [Pending Review] [How A Model is Loaded in Hugging Face and SGLang](./sglang/how-model-is-loaded/readme.md)：记录模型在 Hugging Face 和 SGLang 中的加载过程，帮助理解权重加载机制。
- [Pending Review] [Speculative Decoding](./sglang/speculative-decoding/speculative-decoding.md)：介绍 speculative decoding 优化技术，利用较小的 draft model 预测下一个 K 个 token，实现最高 K 倍的加速。
- [Pending Review] [Zero-Overhead Batch Scheduler](./sglang/zero-overhead-scheduler/zero-overhead-batch-scheduler.md)：介绍零开销批处理调度器，解决传统推理系统中 CPU 调度和 GPU 计算串行执行导致的 GPU Bubble 问题。
- [Pending Review] [Data Parallelism Attention](./sglang/dp-attention/readme.md)：详细介绍 DP Attention 的原理与实现，针对 DeepSeek 等使用 MLA 且只有一个 KV head 的模型，避免 tensor parallelism 导致的 KV cache 重复。
- [浅析 SGLang 框架的量化设计与思路](./sglang/quantization/quantization_architecture.md)：同样刊载于[知乎：浅析 SGLang 框架的量化设计与思路](https://zhuanlan.zhihu.com/p/1971183020338832111)还有[英文版本](./sglang/quantization/quantization_architecture_en.md)。
- [Constraint Decoding 的概念、方法与优化](./sglang/constraint-decoding/readme.md)：同样刊载于[知乎：一文理解 Constraint Decoding 的概念、方法与优化](https://zhuanlan.zhihu.com/p/18336995950)。
- [Pending Review] [Online Update Weights](./sglang/online-update-weights/readme.md)：介绍 SGLang 中 `online_update_weights` 接口的实现，区别于从磁盘读取权重的 `update_weights`，该接口从训练 engine 中直接通过 nccl 广播新的权重。
- [Pending Review] [SGLang Verl Engine 优化解析](./sglang/sglang-verl-engine/readme.md)：解析 SGLang 中 verl engine 的优化，包括 `update_weights_from_tensor` 等接口的实现。
- [Latency Accelerate For Weight Updates](./sglang/latency-accelerate-for-weight-updates/readme-CN.md)
- **[🔥相关调试] [通过 Torch Memory Snapshot 分析 VLM RL 训练中的显存泄露问题](./torch/mem-snapshot/readme.md)**：分析 SGLang 的显存泄露问题，以及解决方案，同样刊载于[知乎](https://zhuanlan.zhihu.com/p/1943202817247519535)和[英文版本](./torch/mem-snapshot/readme-en.md)。

### 使用与实践

- [Pending Review] [Qwen3-Coder Usage](./sglang/qwen/coder.md)：介绍如何在 SGLang 中使用 Qwen3-coder，包括 tool-parser 的使用。
- [Pending Review] [NVIDIA Dynamo](./sglang/nvidia-dynamo/dynamo.md)：介绍 NVIDIA Dynamo，一个为多节点分布式环境中的生成式 AI 和推理模型服务设计的高吞吐量低延迟推理框架。
- [查看 HuggingFace 模型结构](https://zhuanlan.zhihu.com/p/9912733791)
- [SGLang 后端原文解析](https://zhuanlan.zhihu.com/p/716543182)
- [Reward / Embed Model Sever Engine 现状浅析](https://zhuanlan.zhihu.com/p/4148050391)
- [小白视角：vllm 迁移到 SGLang 的体验与收获](https://zhuanlan.zhihu.com/p/714833359)
- [小白视角：利用 SGL 来 Serve Embedding Model](https://zhuanlan.zhihu.com/p/715805386)
- [小白视角：利用 vllm serve 新的 Embedding Model](https://zhuanlan.zhihu.com/p/715857723)

## Scheduling and Routing

- [Mooncake：将 P / D 分离进行到底](https://zhuanlan.zhihu.com/p/1711346141)
- [prefill 和 decode 该分离到不同的卡上么？](https://zhuanlan.zhihu.com/p/1280567902)
- [基于 chunked prefill 理解 prefill 和 decode 的计算特性](https://zhuanlan.zhihu.com/p/718715866)
- [ModelServer：基于 SGLang 的前端分发系统](https://zhuanlan.zhihu.com/p/718015016)


## ML System 基本功

### Transformers & Model Architecture

- [Pending Review] [Transformer中的交叉注意力机制](./transformers/attention/cross_attention.md)：介绍 Transformer 中的交叉注意力机制，允许解码器访问和使用编码器的相关信息，同样有[英文版本](./transformers/attention/cross_attention_en.md)。
- [一文理解 special tokens 和 chat template](./transformers/special_tokens/special_tokens.md)：同样记录于知乎 [一文理解 special tokens 和 chat template](https://zhuanlan.zhihu.com/p/17052593700)。

### CUDA & GPU

- [基于 torch-memory-savor 浅析 CUDA Graph](./torch/cuda-graph/readme.md)：同样刊载于[知乎：基于 torch-memory-savor 浅析 CUDA Graph](https://zhuanlan.zhihu.com/p/1921726788574360686)和[英文版](./torch/cuda-graph/readme_en.md)。

### Distributed Training & Communication

- [Pending Review] [手搓 Tensor Parallelism](./torch/tensor-parallelism/readme.md)：关于 Tensor Parallelism 的实现与实践。
- [NCCL 与 NVIDIA TOPO](./torch/nccl/readme.md)：NCCL 的入门与 NVIDIA 显卡的检测，同样刊载于[NCCL 与 NVIDIA TOPO](https://zhuanlan.zhihu.com/p/6160835906)。
- [NCCL and SGLang](./torch/nccl/readme_en.md)：NCCL 在 SGLang 中的应用，其实和中文内容非常接近，但是额外刊载了一些并行策略的内容。我应该不会修缮完成这个笔记，而是单独写笔记来记录并行策略。
- [PyTorch Distributed](./torch/torch-distributed/readme.md)：`torch.distributed` 的通讯实践， GIL 和 `all_reduce` 的细节。这一部分同样刊载在 [知乎：PyTorch 通讯实践](https://zhuanlan.zhihu.com/p/5853094319)。
- [[原创][深度][PyTorch] DDP系列第一篇：入门教程](https://zhuanlan.zhihu.com/p/178402798)：虽然我没学明白 DDP 的内容，我只是借此学习了下 GIL 和 ring all reduce，这一步刊载于 [torch-distributed 的后记](./torch/torch-distributed/readme.md#gil)。
- [nvidia-smi命令详解和一些高阶技巧介绍](https://www.yourmetaverse.cn/deep_learning/199/)：主要是一些网络拓扑，在我本机的结果记录在 [nccl 部分](./torch/nccl/readme.md#nvlink-查询)。

### Quantization

- [Give me BF16 or Give Me Death，当下量化方法的全面评测](https://zhuanlan.zhihu.com/p/5485556270)
- [AWQ：模型量化应当关注激活值](https://zhuanlan.zhihu.com/p/942485319)


## 开发指南

- [How to use docker](./engineer/how-to-use-docker/readme.md)：如何使用 docker 来管理开发环境。请注意，为了共同塑造良好的科研环境，避免有人用 baseline "在我的机器上能跑"来恶心别人，学习 docker 对任何人都是必不可少的。同样我们也有[英文版本](./engineer/how-to-use-docker/readme_en.md)和[知乎](https://zhuanlan.zhihu.com/p/1916764175230801287)。
- [配置清爽的开发环境](./engineer/uv/readme.md)：配置清爽的开发环境，同样刊载于[知乎：配置清爽的开发环境](https://zhuanlan.zhihu.com/p/23440683394)。
- [在 CI 上编译 jupyter notebook 并部署为文档](https://zhuanlan.zhihu.com/p/2382351079)
