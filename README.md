# Awesome-ML-SYS-Tutorial

My learning notes/codes for ML SYS.

一直以来对 ML + SYS 很感兴趣，苦于本科没有学好 ML，更没学好 SYS，但是读博了觉得自己应该可以在这方面试一试。

有如此打算，一来是我发觉组里大多同学做的是 ML Theory + Application，但是真的把一个 Theory 落到一个良好的 Application 上都有一定阻碍。譬如组里之前发的两篇工作 [SPIN](https://github.com/uclaml/SPIN) 和 [SPPO](https://github.com/uclaml/SPPO)，都有很好的影响力，但是工程实现不尽理想。虽然这并不妨碍工作的价值，但是如果在工程/系统上优化好，想来可以有更好的影响力。

此外，博士入学前的暑假，我和组里同学做了一个 In-context Learning for Agent 的工作 [COPS](https://github.com/uclaml/COPS)，比较符合我的审美。我们就两个人主力干活，一个大哥推理论，而我负责在工程/系统上实现。这种工作模式让我的体感非常舒适，基于此，我甚至得出一个粗糙的结论：

$$
\dfrac{\text{Theory}+\text{System}}{2}=\text{Application}
$$

这就是我想做 ML + SYS 的初衷了。所以从 2024 年的夏季开始，我开始慢慢上手 ML + SYS 这个尚且方兴未艾的领域。需要学习的实在太多了，有的在一些平台（譬如知乎和 HuggingFace Blog）上已经有了很好的资料，但是其他部分仍有所欠缺。所以，这个 repo 主要记载了我自己的一些学习笔记/读后感/思索 etc

## 个人完成部分

目前我个人完成的部分，按照时间倒叙有：

**刊载于 GitHub**

其实也连载于知乎，但是知乎的 markdown 支持太糟糕了，所以建议直接切原片，看 GitHub Repo。

- [OpenRLHF 开发笔记](./openrlhf/readme.md)：OpenRLHF 的开发笔记，现在还没有更新完全，但是值得研究 post training framework 的朋友参考。

- [SGLang Code Walk Through](./sglang/code-walk-through/readme.md)：一个请求被 SGLang Engine 处理的全过程，工程量太大了，还在和很多朋友一起施工中。此外，这里有一个不一定及时更新了的[中文版本](./sglang/code-walk-through/readme-CN.md)。

- [Latency optimization for weight updates](./sglang/latency-accelerte-for-weight-updates/readme.md)：一次对效率的 debug 过程，同样刊载于[记一次对 SGLang weight update latency 的优化](https://zhuanlan.zhihu.com/p/9908228168)。

- [Walk Through SGLang / VLLM Worker](./sglang/sglang-worker/readme.md)：SGLang 的代码不完全解析，同样刊载于 [Walk Through SGLang / VLLM Worker](https://zhuanlan.zhihu.com/p/6363614076)，这次我们还贴心提供了[英文版本](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/sglang-worker/readme_eng.md)。更详细的解析应该参考 [SGLang Code Walk Through](./sglang/code-walk-through/readme.md)，这个只是辅助看看。

- [NCCL 与 NVIDIA TOPO](./nccl/readme.md)：NCCL 的入门与 NVIDIA 显卡的检测，同样刊载于[NCCL 与 NVIDIA TOPO](https://zhuanlan.zhihu.com/p/6160835906)。

- [PyTorch Distributed](./torch-distributed/readme.md)：`torch.distributed` 的通讯实践， GIL 和 `all_reduce` 的细节。这一部分同样刊载在 [知乎：PyTorch 通讯实践](https://zhuanlan.zhihu.com/p/5853094319)。



**刊载于知乎**

- [Give me BF16 or Give Me Death，当下量化方法的全面评测](https://zhuanlan.zhihu.com/p/5485556270)
- [浅析主流 Alignment 算法与 NeMo-Aligner 框架](https://zhuanlan.zhihu.com/p/5220718268)
- [Reward / Embed Model Sever Engine 现状浅析](https://zhuanlan.zhihu.com/p/4148050391)
- [在 CI 上编译 jupyter notebook 并部署为文档](https://zhuanlan.zhihu.com/p/2382351079)
- [Mooncake：将 P / D 分离进行到底](https://zhuanlan.zhihu.com/p/1711346141)
- [prefill 和 decode 该分离到不同的卡上么？](https://zhuanlan.zhihu.com/p/1280567902)
- [AWQ：模型量化应当关注激活值](https://zhuanlan.zhihu.com/p/942485319)
- [基于 chunked prefill 理解 prefill 和 decode 的计算特性](https://zhuanlan.zhihu.com/p/718715866)
- [ModelServer：基于 SGLang 的前端分发系统](https://zhuanlan.zhihu.com/p/718015016)
- [SGLang 后端原文解析](https://zhuanlan.zhihu.com/p/716543182)
- [小白视角：利用 vllm serve 新的 Embedding Model](https://zhuanlan.zhihu.com/p/715857723)
- [小白视角：利用 SGL 来 Serve Embedding Model](https://zhuanlan.zhihu.com/p/715805386)
- [小白视角：vllm 迁移到 SGLang 的体验与收获](https://zhuanlan.zhihu.com/p/714833359)

## 他山之石部分

如前文所述，其实大量的材料在知乎等等平台都是有的。**他山之石，可以攻玉，~~懒惰是人类进步的阶梯。~~** 我也将自己学习过其他同行的优秀博文记录在下：


- [图解大模型RLHF系列之：人人都能看懂的PPO原理与源码解读](https://zhuanlan.zhihu.com/p/677607581)以及[图解OpenRLHF中基于Ray的分布式训练流程](https://zhuanlan.zhihu.com/p/12871616401)：猛猿小姐姐的非常好的 RLHF 入门资料，看了之后会对 RLHF 的计算流以及 OpenRLHF PPO 的框架有很好的理解，我自己也补充了写自己的理解在 [RLHF 的计算流](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/tree/main/rlhf/OpenRLHF#rlhf-%E7%9A%84%E8%AE%A1%E7%AE%97%E6%B5%81)。
- [[原创][深度][PyTorch] DDP系列第一篇：入门教程](https://zhuanlan.zhihu.com/p/178402798)：虽然我没学明白 DDP 的内容，我只是借此学习了下 GIL 和 ring all reduce，这一步刊载于 [torch-distributed 的后记](./torch-distributed/readme.md#gil)。
- [nvidia-smi命令详解和一些高阶技巧介绍](https://www.yourmetaverse.cn/deep_learning/199/)：主要是一些网络拓扑，在我本机的结果记录在 [nccl 部分](./nccl/nccl.md#nvlink-查询)。
