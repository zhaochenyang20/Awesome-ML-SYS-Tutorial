# HybridFlow veRL 原文浅析

众所周知，我一直在 SGLang team 负责端茶倒水 + RLHF。对于后者，这段时间我一直在学习 veRL 的整体框架，hybrid engine 的想法真令人眼前一亮。其实 SGLang 团队一直有成员在负责这块工作的推动，但是由于 DeepSeek 模型支持的工作强度巨大，加之团队成员的精力有限，还没能将这系列的改动 up stream 到 main branch 上。不过，随着 grok 的发布，我们会将 SGLang 对于 hybrid engine 的支持，快速 up stream 到 main branch 上，这里浅浅预告一番。

回到这篇文章本身，这是 SGLang-veRL 系列工作的开端笔记，也督促自己更加全面地学习 training and inference co-design。

PS：之前也有基于 nemo-aligner 和 OpenRLHF 做一些解析，欢迎大家参考，也感谢这些框架作者的精彩贡献。大家都是写框架的人，其中的酸甜苦辣，自不必多说。

- [浅析主流 Alignment 算法与 NeMo-Aligner 框架](https://zhuanlan.zhihu.com/p/5220718268)
- [浅析以 OpenRLHF 为代表的 post-training 系统的计算流程](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/OpenRLHF/readme.md)

## Single-Controller vs Multi-Controller

在梳理 introduction 之前，我们先来引入 veRL 所依赖的一大重要概念——single controller 和 multi controller。

**在一个复杂的工作流程中，single controller 只有一个程序负责管理，而其他的子模块只负责执行。所有控制逻辑都写在唯一 controller 上，实现简单，便于调试。然而，single controller 所承担的控制压力巨大，一来，通讯强度大而效率堪忧，二来，倘若 single controller 崩溃，整个系统将彻底失效。反过来，multi controller 则有多个控制程序来管理不同的子模块，每个子模块仍旧只负责执行自己的功能。如此以来，单个控制程序的管理压力降低，系统更加鲁棒可扩展。然而，控制逻辑分散在多个程序中，实现复杂，难以调试。**

有了这个直观的理解，我们回顾下 [PPO 的粗略工作流程](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/OpenRLHF/readme.md#%E6%9B%B4%E6%96%B0%E6%B5%81%E7%A8%8B)。注意，actor 在 RLHF 会进行 auto-regressive decoding，而 critic, reward 和 reference 则只会 prefill，不会 decode。所以，我们将 actor 的推理特定称为 rollout，而其他模型的推理称为 inference。

1. 准备一个 batch 的 prompts；
2. 将这个 batch 的 prompts 输入给 Actor，rollout 得到 responses；
3. 将 prompt + responses 输入给 Critic/Reward/Reference，进行 inference，分别计算得得到 values、reward 和 log probs，将这些整合称为 experiences；
4. 根据 experiences 多轮计算 actor loss 和 critic loss 并更新 Actor 和 Critic。

如此以来，一个很自然的想法是用 single controller 来管理整个工作流程，然后每个子模块（Actor、Critic、Reward、Reference）再各自由一个 single controller 来管理。然而，这种 naive 的实现里，两层控制结构都是 single controller，因此系统内的通讯 overhead 非常大。一个可能不太直观的点是——最高层负责总调度的 single controller 承担的通讯压力，反而比每个子模块的 single controller 要小。

可以简单理解，rollout engine 只需要把 prompt + response return 给上层 controller 就好了，上下两层的通讯量最多 M 级别；而 rollout engine 内部的通讯就非常大了，不然怎么会把 NV Link 接近 T 为单位的通讯速度都拉满呢？

因此，每个子部件内的 controller 承担的通讯压力反而是更大的，如果这一层还都是 single controller 的，可以想见效率堪忧。事实也是如此，主流的训练引擎都是 multi controller 的，譬如 FSDP、Megatron 和 DeepSpeed。既然 Actor 的 Training Engine 是 multi controller 的（具体来说是 SPMD 的），那么 Actor 的 rollout engine，是否也该是 multi controller 的？从理论上来讲，是的，直觉告诉我们，SPMD 的 training engine 和 SPMD 的 rollout engine 相互通讯，会比和 single controller 的 rollout engine 通讯要高效得多。这个直觉也禁得起推敲。SPMD 的 training engine 与 rollout engine 都采用了分布式、多控制器的设计，各节点之间可以直接点对点通信，这样可以充分利用高速互联（例如 NVLink 或 InfiniBand），实现数据并行传输和计算。反过来，training engine 采用 SPMD 而 rollout engine 采用 single controller 的话，所有训练节点的数据都必须汇总到某个控制节点上，才可以完成 training engine到 rollout engine 间的参数更新，这就不可避免地引入了通信瓶颈和单点故障风险。

因此，SPMD 的 training engine 和 SPMD 的 rollout engine 简直是绝配。然而，由于历史原因，目前主流的 rollout engine 还是以 single controller 为主的。所以，将 SGLang 由 single controller 改为 SPMD 是我们的一个重要工作目标。事实上，我们已经有了成熟的 PR，可以[参考这个 branch](https://github.com/fzyzcjy/sglang/tree/feat/overall_verl)。

注：上面描述的 single controller 和 SPMD 模式只是经验上的 practice，但实际上他们描述的是：single controller 主要关注于控制流是否单点；而 SPMD 模式更关注的是分布式场景下数据执行流。即使是 single controller，也是可以使用 SPMD 模式进行数据流执行的。

总之，这篇文章的开篇花费了巨大的篇幅来简述 single controller 和 multi controller，以及为什么 SGLang 需要从 single controller 改为 multi controller。理解了这些概念后，我们可以正式进入 veRL 的 introduction 了。

## Introduction

正如前文所提到的，multi-contoller 能够有效的降低通讯压力，提升系统鲁棒性。然而，如果最顶层的 controller 也是 multi-controller 的，其实对用户会非常复杂。在一个 controller 内代码的修改，需要将所有 dependency 都修改一遍。很难想象读 ML researcher 会愿意接受这一点。

因此，veRL 在上层暴露出 single controller 的接口，并进行完善的封装。用户能够基于算法设计，自由组合并行策略（3D 并行、ZeRO 还有 FSDP），直接对子模块进行拼装；而在每个子模块内部，采用 multi-controller，提供强劲的效率。当然，可能更改子模块就会相对麻烦。

**有了 single controller 和 multi-controller 的概念后，这里引入veRL 的第二个核心概念：hybrid engine。在 RLHF 流程中，actor model 的 generation 和 rollout 占据了绝大多数运行时间（在 veRL 是 58.9%）。并且，由于 PPO 是 on-policy 算法，经验（experiences）必须来自于被 train 的模型本身，因此，rollout 和 training 是必须串行的。如果这两者使用不同的资源组，比如 rollout 用 2 张卡，而 training 用 4 张卡，rollout 的时候 training 的资源闲置，training 的时候 rollout 的资源闲置，无论如何都会浪费大量的计算资源。由此，veRL 将 training 和 rollout engine 放置在同一个资源组中串行执行。training 时，将 rollout engine 的显存回收（offload 到 CPU 上 或者直接析构掉），rollout 时，再将 training engine 的显存释放掉。这种将 actor model 的不同 engine 放置在同一个资源组上的方案，就称为 hybrid engine。**

注意到，除开 hybrid engine 之外，类似共用资源组的方法还有 collocate。在讲述 collocate 策略之前，我们回顾下四个子模块分别需要什么 engine：

1. actor model 需要 training engine 和 rollout engine。前者是用现代 training engine，比如 Megatron 或者 FSDP，后者得用现代推理引擎，比如 SGLang 或者 vllm 作为 rollout engine。这里思考一个小问题，为什么不能拿着 training engine 得到的 logits 做 sampling 然后 decode，貌似也可以用去 rollout？简单来说，太慢了，用训练引擎做 decode 的效果自然不如专用的推理引擎。
2. critic model 需要 training engine 和 inference engine。前者还是是现代的训练引擎，但是后者，可以用现代的推理引擎的高效 prefill 来得到 value 么？其实不能，critic model 的 inference 会直接复用 training engine 的 forward 来得到 value，所以 critic 的 inference engine 和 training engine 其实是同一个。其中的原因在此旧事重提：

> 推理引擎的 kernal fusion 和 training engine 差距不小，batch size 不一样时，推理请求 dispatch 到不同的 kernal 上，然后 numerical 误差逐层累计，到了 log probs 这层就到了不可忽视的程度了。这个问题在 bert 时代就有了，training engine 和 inference engine 的精度差异无法规避，而且全心来搞一两个月可能内都没法修复。所以现在推理引擎在 RLHF 中更多是加速 sampling，reward 和 embedding 还得用训练脚本来算，可能得半年后花好几个月研究研究这个问题。

3. reference model 和 reward model 只需要 inference，因为二者不需要训练，但是如同我之前提到的一样，用现代推理引擎得到的 log probs 和 reward 的精度不如用现代训练引擎得到的精度，所以这里选择用 training engine 的 forward 来做 inference，得到 log probs 和 reward。

有了这些认识，我们再来看 collocate 策略。collocate 策略将 actor 的 training engine 和 reference 的 inference engine 放置在同一个资源组上，将 critic 的 training/inference engine 和 reward 的 inference engine 放置在同一个资源组上，最后单独放置 actor 的 rollout engine。

与此相对的是，hybrid engine 单独强调了将 actor model 的 rollout engine 和 training engine 放置在同一个资源组上，而 collate 则强调的是不同子模块之间的。可以见到，hybrid 共用资源组的 engine 都属于 actor，二者区别显著更大，更容易 OOM。当然，collocate 和 hybrid engine 都可以提高 GPU 利用率，不过速度自然会有所损失。

总之，这两个概念是 veRL 最强大的贡献。实际上 veRL 还提供了一套基于贪心搜索的 placement（资源组）分配算法，不过按照作者描述，这个 feature 效果不太诱人，现在比较冷门了。

## Background

这一部分是一些背景补充：

- 现代的分布式训练框架（Megatron-LM，MegaScale，DeepSpeed）都支持了 3D 并行，也即 DP PP TP。LLM serving 也有对应的策略和概念，不过其中只有模型参数和 KV cache 会被 sharded，不存在优化器和梯度的需求。
- Actor model 训练是 compute bounded 的，通常倾向于更高的 TP 或者 PP size。而采用同样的高 TP 或者 PP size 的 rollout engine 则效率不佳，实际上 rollout engine 通常希望加大 DP size。所以，为了提高两个阶段各自的效率，actor model 的 training engine 和 rollout engine 会采用不同的并行策略。然而，不同的并行策略导致两个阶段之间的参数更新需要 resharding，导致通讯和访存开销显著。

## Hybrid Engine Performance

- 为了提供灵活的并行策略供用户组合，veRL 提供了 `3DParallelWorker`，`FSDPWorker` 和 `ZeROWorker` 三个基类，并用子类支持各种并行策略。为了做到 Training Engine 和 Rollout Engine 之间的 parameter update，veRL 提供了 8 种 transfer protocols，包括但不限于 `3D_PROTO`, `DP_PROTO`, `ONE_TO_ALL`  等等。
- 费力构造好的 Hybrid Engine 自然要大显身手，veRL 鼓励用户采用 Hybrid 的方式来控制 actor 的 rollout 和 training。不过目前 Hybrid 使用的 SPMD rollout engine 相比起单独放置的 single controller rollout engine 会有一定的推理速度下降。
- veRL 采用 mixed precision 训练，模型参数采用 BF16，梯度和 Adam 优化器采用 FP32，actor 的 rollout 和 其他模型的 inference 采用 FP16。

这一部分记录一些 placement 的经验。希望以后我能有实力基于机器的物理属性直接分析得到这些结论。主要对比这四种策略：

![](./img/placement.png)

1. fully collocate：所有的子模块都放在同一个资源组上，也即 DeepSpeed-Chat。
2. hybrid：actor 的 rollout engine 和 training engine 放在同一个资源组上，其他子模块进行部分 collocate，这是 veRL 所提出的策略，但是 veRL 原文其实给出了一个搜索方法，能够贪心搜索所有的策略，选择出最佳策略。
3. split collocate：actor 的 training engine 和 reference 的 inference engine 放在同一个资源组上，critic 的 training/inference engine 和 reward 的 inference engine 放在同一个资源组上；最后单独放置 actor 的 rollout engine，这是 OpenRLHF 和 NeMo-Aligner 的默认策略。
4. stand alone：所有子模块都单独放置，早期 OpenRLHF 会这么做，现在自然不会了。

这里直接给结论了：
- 16 ~ 64 GPUs 范围内，fully collocate 效果最好；
- 96 ~ 128 GPUs with 34B models 或者 96 GPUs with 13B models，split collocate 效果最佳。
- 暴力搜索总可以得到最佳策略，但是搜索成本过大了 😂
- 每个子模块都可以高强度利用计算资源时，fully collocate 效果最好。
- 在大规模训练时，actor 和 critic 分开放置效果更佳。

最后，veRL 顶层的 single controller 还带了一个好处，方便利用 rule-based reward，这就是一个新的故事了。
