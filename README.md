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

## RLHF System 开发笔记

- [AgentLoop 源码探究](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/agent-loop/agentLoop_CN.md): 对 verl 中的 agentLoop 模块进行了分析
- [系统性分析 verl multi-turn training 的时间消耗](./rlhf/verl/multi-turn/tool_examples/profile.md)：verl 多轮交互与工具调用 profile 分析，还有[英文版本](./rlhf/verl/multi-turn/tool_examples/profile_en.md)和[知乎](https://zhuanlan.zhihu.com/p/1929748460212552414)。
- [RL 系统深思：FSDP 训练后端](./rlhf/sys-design/readme-2.md)：讨论 FSDP 的原理和实现，以及分析 verl 的 FSDP 使用。同样刊载于[知乎](https://zhuanlan.zhihu.com/p/1929115059113693341)和[英文版本](./rlhf/sys-design/readme-2-en.md)。
- [RL 系统深思：深入理解权重更新机制](./rlhf/sys-design/readme-1.md)：半年工作的总结，深入理解权重更新机制，同样刊载于[知乎](https://zhuanlan.zhihu.com/p/1925210722704531547)和[英文版本](./rlhf/sys-design/readme-1-EN.md)。
- [verl 参数速览](./rlhf/verl/multi-turn/code-walk-through/readme-5.md)：verl 参数速览，同样刊载于[知乎](https://zhuanlan.zhihu.com/p/1925041836998783250)，还有[英文版本](./rlhf/verl/multi-turn/code-walk-through/readme-5-EN.md)。
- [深入浅出理解 verl 源码（Rollout）](./rlhf/verl/multi-turn/code-walk-through/readme-2.md)：同样刊载于[知乎](https://zhuanlan.zhihu.com/p/1923349757566388159)，还有[英文版本](./rlhf/verl/multi-turn/code-walk-through/readme-2-EN.md)。
- [深入浅出理解 verl 源码（初始化）](./rlhf/verl/multi-turn/code-walk-through/readme.md)：同样刊载于[知乎](https://zhuanlan.zhihu.com/p/1920751852749849692)，还有[英文版本](./rlhf/verl/multi-turn/code-walk-through/readme_EN.md)。
- [从 tokenizer 视角来分析 Agentic 多轮训练的复杂性](rlhf/verl/multi-turn/fast_tokenization/multiturn_tokenization_and_masking_ZH.md)：同样刊载于[知乎](https://zhuanlan.zhihu.com/p/1917126584806139373)和[英文版本](rlhf/verl/multi-turn/fast_tokenization/multiturn_tokenization_and_masking.md)。
- [Search-R1 & veRL-SGLang: Train LLMs with Multi-Turn RL to Reason and Call a Search Engine](rlhf/verl/multi-turn//tool_examples/verl-multiturn-searchR1-like_ZH.md)：整合 Search-R1 framework 到 verl-sglang 生态，同样刊载于[知乎](https://zhuanlan.zhihu.com/p/1912156329751081620)。
- [SGLang, verl, OpenBMB 与清华大学团队联合开源：在主流 RLHF 框架上首次支持多轮交互与工具调用](rlhf/verl/multi-turn/release_log/verl-multiturn-rollout-Release_ZH.md)：在主流 RLHF 框架上首次支持多轮交互与工具调用，同样刊载于[知乎](https://zhuanlan.zhihu.com/p/1906007821889283171)。
- [Kimi K1.5: Long Context RL 的成功实践](./rlhf/partial-rollout/readme.md)：Long Context RLHF 的工业级实现，一直很喜欢 kimi 团队的技术报告，同样刊载于 [Kimi K1.5: Long Context RL 的成功实践](https://zhuanlan.zhihu.com/p/1894282607325344277)。
- [SGLang-veRL Server：从 Engine 到 Server，我们需要更灵活的 RLHF rollout 接口](rlhf/verl/server-based/veRL-server-based-rollout.md)：为了实现更复杂的 RLHF 系统，我们逐步将 veRL 当中的 rollout engine 替代为 rollout server，同样刊载于[知乎：SGLang-veRL Server](https://zhuanlan.zhihu.com/p/1890631652486665464)。
- [Rule-based Reward](https://zhuanlan.zhihu.com/p/13211508979)：这篇只有知乎，浅浅写了写，老实说原文写的我并不太喜欢，但是 determined reward 确实 charming。
- [HybridFlow veRL 原文浅析](./rlhf/verl/readme.md)：SGLang 的 hybrid engine 的原理与实现，同样刊载于[知乎：HybridFlow veRL 原文浅析](https://zhuanlan.zhihu.com/p/24682036412)。
- [扩展 OpenRLHF 的推理引擎](./rlhf/OpenRLHF/develop-log.md)：将 SGLang 接入到 OpenRLHF 的开发笔记，整个过程非常痛苦，而且目前还有 nccl hang error，已经直接联系了 deepspeed core contributor 在修复了。
- [SWE-Bench：如何构造 LLM 时代的优秀 Benchmark](https://zhuanlan.zhihu.com/p/16292266518)，基于 SWE-Bench 的论文阅读笔记，如何构造好的 benchmark 以为 post-training 提供细粒度 reward，是永恒且美妙的话题。
- [浅析以 OpenRLHF 为代表的 post-training 系统的计算流程](./rlhf/OpenRLHF/readme.md)：基于猛猿小姐姐的文章再做补充，Github native 渲染的巨烂，甚至看[知乎](https://zhuanlan.zhihu.com/p/16370000391)好了。
- [图解大模型RLHF系列之：人人都能看懂的PPO原理与源码解读](https://zhuanlan.zhihu.com/p/677607581)以及[图解OpenRLHF中基于Ray的分布式训练流程](https://zhuanlan.zhihu.com/p/12871616401)：猛猿小姐姐的非常好的 RLHF 入门资料，看了之后会对 RLHF 的计算流以及 OpenRLHF PPO 的框架有很好的理解，我自己也补充了写自己的理解在 [RLHF 的计算流](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/tree/main/rlhf/OpenRLHF#rlhf-%E7%9A%84%E8%AE%A1%E7%AE%97%E6%B5%81)。
- [Latency optimization for weight updates](./sglang/latency-accelerte-for-weight-updates/readme.md)：一次对效率的 debug 过程，同样刊载于[记一次对 SGLang weight update latency 的优化](https://zhuanlan.zhihu.com/p/9908228168)。
- [浅析主流 Alignment 算法与 NeMo-Aligner 框架](https://zhuanlan.zhihu.com/p/5220718268)


## SGLang 学习笔记

- [查看 HuggingFace 模型结构](https://zhuanlan.zhihu.com/p/9912733791)
- [Constraint Decoding 的概念、方法与优化](./sglang/constraint-decoding/readme.md)：同样刊载于[知乎：一文理解 Constraint Decoding 的概念、方法与优化](https://zhuanlan.zhihu.com/p/18336995950)。
- [SGLang Code Walk Through](./sglang/code-walk-through/readme.md)：一个请求被 SGLang Engine 处理的全过程，还有一些 part 没有完成，但是大多地方已经 okay，也让很多 SGLang begginer 就此开始。这里还有[中文版本](./sglang/code-walk-through/readme-CN.md)。
- [Walk Through SGLang / VLLM Worker](./sglang/sglang-worker/readme.md)：SGLang 的代码不完全解析，同样刊载于 [Walk Through SGLang / VLLM Worker](https://zhuanlan.zhihu.com/p/6363614076)，这次我们还贴心提供了[英文版本](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/sglang-worker/readme.md)。更详细的解析应该参考 [SGLang Code Walk Through](./sglang/code-walk-through/readme.md)，这个只是辅助看看。
- [Walk Through SGLang Scheduler](./sglang/sglang-scheduler/readme-CN.md)
- [Latency Accelerate For Weight Updates](./sglang/latency-accelerte-for-weight-updates/readme-CN.md)
- [Reward / Embed Model Sever Engine 现状浅析](https://zhuanlan.zhihu.com/p/4148050391)
- [SGLang 后端原文解析](https://zhuanlan.zhihu.com/p/716543182)
- [小白视角：利用 vllm serve 新的 Embedding Model](https://zhuanlan.zhihu.com/p/715857723)
- [小白视角：利用 SGL 来 Serve Embedding Model](https://zhuanlan.zhihu.com/p/715805386)
- [小白视角：vllm 迁移到 SGLang 的体验与收获](https://zhuanlan.zhihu.com/p/714833359)

## Scheduling and Routing

- [Mooncake：将 P / D 分离进行到底](https://zhuanlan.zhihu.com/p/1711346141)
- [prefill 和 decode 该分离到不同的卡上么？](https://zhuanlan.zhihu.com/p/1280567902)
- [基于 chunked prefill 理解 prefill 和 decode 的计算特性](https://zhuanlan.zhihu.com/p/718715866)
- [ModelServer：基于 SGLang 的前端分发系统](https://zhuanlan.zhihu.com/p/718015016)


## ML System 基本功

- [基于 torch-memory-savor 浅析 CUDA Graph](./distributed/cuda-graph/readme.md)：同样刊载于[知乎：基于 torch-memory-savor 浅析 CUDA Graph](https://zhuanlan.zhihu.com/p/1921726788574360686)和[英文版](./distributed/cuda-graph/readme_en.md)。
- [NCCL 与 NVIDIA TOPO](./distributed/nccl/readme.md)：NCCL 的入门与 NVIDIA 显卡的检测，同样刊载于[NCCL 与 NVIDIA TOPO](https://zhuanlan.zhihu.com/p/6160835906)。
- [PyTorch Distributed](./distributed/torch-distributed/readme.md)：`torch.distributed` 的通讯实践， GIL 和 `all_reduce` 的细节。这一部分同样刊载在 [知乎：PyTorch 通讯实践](https://zhuanlan.zhihu.com/p/5853094319)。
- [Give me BF16 or Give Me Death，当下量化方法的全面评测](https://zhuanlan.zhihu.com/p/5485556270)
- [AWQ：模型量化应当关注激活值](https://zhuanlan.zhihu.com/p/942485319)
- [[原创][深度][PyTorch] DDP系列第一篇：入门教程](https://zhuanlan.zhihu.com/p/178402798)：虽然我没学明白 DDP 的内容，我只是借此学习了下 GIL 和 ring all reduce，这一步刊载于 [torch-distributed 的后记](./distributed/torch-distributed/readme.md#gil)。
- [nvidia-smi命令详解和一些高阶技巧介绍](https://www.yourmetaverse.cn/deep_learning/199/)：主要是一些网络拓扑，在我本机的结果记录在 [nccl 部分](./distributed/nccl/readme.md#nvlink-查询)。


## 开发指南

- [How to use docker](./engineer/how-to-use-docker/readme.md)：如何使用 docker 来管理开发环境。请注意，为了共同塑造良好的科研环境，避免有人用 baseline "在我的机器上能跑"来恶心别人，学习 docker 对任何人都是必不可少的。同样我们也有[英文版本](./engineer/how-to-use-docker/readme_en.md)和[知乎](https://zhuanlan.zhihu.com/p/1916764175230801287)。
- [配置清爽的开发环境](./engineer/uv/readme.md)：配置清爽的开发环境，同样刊载于[知乎：配置清爽的开发环境](https://zhuanlan.zhihu.com/p/23440683394)。
- [一文理解 special tokens 和 chat template](./transformers/special_tokens/special_tokens.md)：同样记录于知乎 [一文理解 special tokens 和 chat template](https://zhuanlan.zhihu.com/p/17052593700)。
- [在 CI 上编译 jupyter notebook 并部署为文档](https://zhuanlan.zhihu.com/p/2382351079)

## 未公开部分

之前的笔记大多写于 2024 年年底，经过了半年时间，我的仓库已略年久失修。一方面我自己更多在项目中负责推动 + delivery，反而自己很少写代码；另一方面，多多少少不少朋友向我们的仓库贡献了笔记，但我完全没有来得及整理。这段时间会不断完成整理并发布。这里索性列举下这些尚未完全的笔记，希望大家多多指正。

- [NCCL and SGLang](./distributed/nccl/readme_en.md)：其实和中文内容非常接近，但是额外刊载了一些并行策略的内容。我应该不会修缮完成这个笔记，而是单独写笔记来记录并行策略。

