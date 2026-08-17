# dots3-note Preview：为了挑战 IMO，LLM 选手做了什么

推理这个词，存在两个不同的含义：一个是指为了完成高难度任务，LLM 需要收集环境反馈，进行深度推导，决定下一步行动，直到得出最终答案（Reasoning）；另外一部分则是我大多数时候的工作内容，对给定的 LLM 模型输入一段 prompt，得到模型的采样结果（inference）。比较好玩的是，很长一段时间，inference 对 Reasoning 是不太有感知的。具体来说，大概在 2024 年以前，reasoning 依赖的技术流派，诸如 chain of thought、tree of thought、recursive reasoning，按照现在的视角来看，其输出都是不够长的；而到了 2025 年 1 月，DeepSeek 公开了 long context reasoning 的秘密后，reasoning 的长度如同怪物般增长，也实质性给 inference 带来了压力。在这之后，如同大家见到的那样，SGLang 生态为了 long context reasoning 做出了许多新的技术设计，譬如为了 long context reasoning RL 需要支持 abort / partial rollout，为了存储恐怖的推理中间结果（rational）需要将 KV cache 一层一层向下存储...

2025 年 8 月，我在从家往返 SFO 的飞机上读完了 Kimi K1.5 的技术报告，写了这篇[Kimi K1.5: Long Context RL 的成功实践](../../rlhf/partial-rollout/readme.md)。一年半过去，long context 的量级竟然还能上升。K1.5/R1 时代，大家讨论的 reasoning，需要几分钟到几十分钟的 decode。而到了现在，2026 年的 8 月 14 日，小红书 dots studio 开源了成功挑战 IMO 赛题的 dots3-note Preview 模型。他们给出的推理开销是：ARC-AGI-3 规模的复杂任务需要 Agent 自主进行数千轮交互，完整运行时间可达 40 到 50 小时。

自然，这样的需求，对于 Infra 而言，带来了崭新的挑战。

## IMO 规模的数学问题解题思路

在 2025 年时，google OpenAI 等团队第一次公开挑战 IMO 赛题。让模型保证证明正确性，当时的主流做法是形式化验证：人工先把题面翻译成 Lean 一类的形式化语言，此后让模型在形式系统内求解，正确性由 lean 的完备性来保证。自然，真实世界里大量复杂问题是无法完整形式化的，这条路径的通用性较为受限。(当然，真实世界应该极少存在 IMO 问题 😂)

dots 这次则完全绕开了形式化，模型直接读取组委会提供的原始 LaTeX 输入，以 agentic 的方式端到端解题，结合自然语言推理并调用 Python 解释器完成解题（老实说我觉得如果真的是和人类选手掰腕子，调用 python 解释器似乎也不是完全合理）。既然失去了验证器这一层保障，证明的正确性只能由模型自己承担。dots 团队设计的 harness 为此拆成三个部分：Proof 生成候选证明，Verify 判断已有证明的正误并给出改进建议，Refine 针对 Verify 的意见继续修改；经过多轮并行推理、审查、修正与方案整合之后，形成最终提交评阅的证明。

这套 harness 的负载形状其实对 inference 反而是友好的，至少和 RadixTree 的目标非常契合：同一道题会派生出大量并行分支，而分支之间共享着系统提示、题面、乃至前若干版证明；Verify 环节要把一整份证明读进上下文才能产出一段 critique；Refine 环节又把 proof 和 critique 一起带上继续生成。换言之，这是一个树状展开、前缀高度共享、prefill 占比极高的负载。举一个不严谨的比方，这套 harness 是对 Tree of Thought 的现代复刻。

<div align="center">
  <img src="radix-tree.svg" alt="Radix Tree" width="75%">
</div>


## TEMPO：将自我评价纳入训练目标

Verify 一节如同我们所预料的一样，所有听上去用 prompt 就能控制的环境，经过训练后，效果都会有显著提升。为了强化模型具备评价自身方案的能力，dots 在 RL 阶段就把"评价当前状态"当作与"推进任务"并列的优化目标，方法叫 [TEMPO (Test-Time-Scaled Value Estimation with Macro-Step Policy Optimization)](https://studio.dots.ai/dots/tempo-blog.html)。

超长程 RL 有两点显著的困难。第一，反馈周期过长：最终奖励要等轨迹结束才能观测到，当一次 rollout 持续数十小时，任务本身的物理执行时间就构成了信号获取周期的下限。第二，信用分配困难：horizon 越长，早期决策与最终奖励之间隔着越来越长的行为链，也就更难判断哪些中间状态真正影响了结果。

一个自然的解决方向是，在轨迹尚未结束时提前估计当前状态的未来回报。无需等到最终结果出现，未完成的轨迹也能尽早产生训练信号，这正是 actor-critic 方案希望做到的事情。但传统 critic 的做法，是在模型上再外挂一个只输出分数的 value head：把当前状态输入给 value head，做一次 forward，得到标量估值就结束。value head 的单次预估实际上就是一次 forward，预估的计算量是固定的，当前模型的解题状态再难，预估也不会多想一步、多花一点算力。小红书团队指出，**当 actor 本身依赖 test-time scaling 才能解题时，估计它的中间状态同样是一个复杂推理问题**：critic 需要回顾交互历史、检查已经形成的假设、判断当前搜索方向是否可行、推演后续路径。因此，既然 actor 的能力可以随 test-time compute 增长，用于评价这些状态的 critic 也应具备同样的能力。

### macro-step

有了前文的铺垫，一个非常自然的想法是，将先前作为 critic 的简单 value head 更换为另一个生成模型，比如说用另一个体积更小的 reasoning LLM 来做 value head。如此一来，单次估值要生成一段较长的推理，可能还包含反思与工具调用，成本远高于 scalar head 的一次前向，因此它不可能像传统 critic 那样在每个 token 或每个动作之后密集运行。

TEMPO 的处理方式是把估值放在固定边界上，将连续 k 轮的模型—环境交互定义为一个 macro-step，估值只在终点做一次。作者给的理由是，在 Agent 任务中，有实际意义的状态变化通常发生在推理与工具调用完成、环境返回新观测之后，因此把多轮交互聚合起来，既给 actor 留出了完成一段完整探索的空间，也让 critic 能够在信息更充分的状态上估值。（PS：训练效率当然是另一个重要的考量 😂）

而 macro-step 除了是估值的边界之外，同时也是 rollout 与梯度更新的基本单位。GRPO 必须等待整条轨迹跑完、拿到最终奖励，才能算出 reward、做一次梯度更新；TEMPO 则在 macro-step 结束时就可以进行梯度更新，收集前一段内的环境奖励，加上 critic 对终点状态「后面还有多少 reward」的估值，来组成这一段的完整 reward，不必等任务结束也能更新 actor。

| | 单次 rollout 长度 | 中间估值 | 覆盖完整 horizon 的方式 |
|---|---|---|---|
| GRPO | 完整 T 轮 | 无 | 每条轨迹都跑完 |
| PPO | 完整 T 轮 | scalar value head，沿途估计 | 每条轨迹都跑完 |
| TEMPO | 一个 macro-step，k 轮 | 生成式 critic，段末估一次 | 保存终点状态，跨多次更新逐步推进 |

对 Infra 而言，TEMPO 把一条 40 小时的轨迹拆成了若干个可以独立调度的段，每段都能立刻产出训练数据。这与 partial rollout 是同一条脉络上的东西，区别只在于 partial rollout 缓存的是未完成请求的生成状态，而 TEMPO 保存的是 actor 与环境的联合状态，且这样的中间联合状态是可以用作训练的。

### 生成式 critic 与特权信息

critic 的输入是当前的 macro-step 终点状态，加上这一段以及更早的交互记录；输出则是一整段 reasoning rationale，末尾附一个数值估计。它的 reasoning 过程，尝试回答三个问题：actor 目前已经摸清了环境的哪些规律，它正在尝试的方案是什么，以及还有哪些障碍没有解决。

比较有意思的是，critic 还能读到 actor 完全看不到的东西，譬如环境的内部状态、隐藏规则、测试方案、乃至游戏源码。这些特权信息只进 critic 的上下文，绝不进 actor 的上下文，原因也很直白：一旦 actor 能读源码，它就不再需要探索了，整个训练目标会立刻退化成"抄答案"。注意到，actor 完成训练后直接部署到线上完成最后的 IMO 考试时，critic 是不会参与的，因此让 critic 获得超出 actor 的信息是合理的，它只负责判断 actor 的方向对不对，多一份 ground truth 只会让判断更准。

在我的研究生涯看来，许多工作的评判比起完成这件工作本身还要复杂。客观任务，比如推理系统，评判标准是非常客观的，我们有非常多广为人知的指标，诸如 TTFT、TBT 等等；但是这世界上还存在着大量"不可描述，难以名状"的主观任务，比如一个人的代码写的是否规范，一个人在团队中是否可靠，或者 dots3-note 模型完成 IMO 赛题的证明方法是否科学严谨。很遗憾，越是主观的任务，在我看来往往评估成本越高。但是，主观评测总有一些简化的路径，譬如允许评价侧使用生成侧拿不到的信息。critic 正是利用和 actor 的不对称来完成评测，且这个不对称并非能力上的不对称，而是信息上的不对称。

至于如何训练生成式的 critic 模型，这是 value head 时代无需考虑的问题。value head 的训练是标准回归，给定状态和目标值，算 MSE，梯度回传到 value head 的几层参数上，干净利落。但生成式 critic 输出的是一整段 token，其中只有末尾的数值算是目标，前面几千个 token 的推理过程没有任何监督信号。如果直接对末尾那个数字做回归，梯度基本只作用在最后几个 token 上。这和我们的目标不完全一样，我们希望训练 critic 通过推理得到准确 value 的能力，推理过程的监督信号极为重要。

TEMPO 的解法是把价值拟合本身也视为一个 RL 问题，思路与 GRPO 完全一致。具体来说，先对同一个状态采 m 次独立估值，每次得到一整段推理加一个数字 value；然后按这个数字与目标值 V 的误差给每次采样一个 reward，误差越小 reward 越高；接着在这 m 个 reward 上做组内中心化，得到每条采样的 advantage；最后按 policy gradient 更新，梯度自然就落到了整段推理上。critic 被不断奖励，学习到了更好的思考方式。

### actor 兼任 critic

既然 critic 是一个会推理、会调工具、还能读源码的生成模型，我们是否应该为其单独进行训练？TEMPO 方案中，直接让 actor 兼任了 critic，跑 macro-step 的时候模型是 actor，到了段末边界换一套 prompt 并且输入额外的环境评测信息，同一份权重就扮演起了 critic。

当然，这个设计存在一个潜在的冲突，上一节讨论到 critic 能读到额外的环境信息而 actor 读不到，现在两个角色其实是同一份参数，那 actor 会不会借着共享权重把答案间接学走，考场作弊？我们可以从上下文管理和参数共享两个角度来思考。只要环境信息只出现在 critic 的上下文里，用途是核对 actor 的方向对不对，而轮到模型以 actor 身份行动的时候，它的上下文里并没有这些东西，也就谈不上考场作弊。更进一步，critic 侧的梯度更新确实会落到同一组参数上，规则本身会不会在训练过程中被顺带记进参数里，坦诚说原文并没能做出判断，我们也没有看到能证伪的实验。从结果来看，这共享权重的方案最终产生的模型水平扎实，在真实世界泛化水平过硬，即便存在潜在的泄露，也没有造成实质问题。

> Note：关于 critic 和 actor 的参数共享，会不会某种意义上让 actor 作弊？我和小红书技术团队的乔超老师简单聊了聊，他们的实验大概观察如下：长程任务的 prompt 消耗极慢，2048 轮的任务按 64 步一个 macro-step 来算，一条数据要很久才被用掉，训练很少过完一个 epoch；而特权信息只是 critic 当次推理的 context，数据经过多个 batch 训练的话，模型不太会把 shortcut 记进参数。总结而言，critic 和 actor 的参数共享潜在的负面影响相对可控。

撇开这层顾虑，参数共享的正面理由是两类任务吃的本来就是同一套理解。actor 需要跟踪环境变化、维护自己的假设、规划下一步动作，而 critic 需要判断进展到了哪一步、哪些假设可能是错的、后面还有没有可行路径，这两件事在底层需要的东西本质上重合。共享之后，actor 在交互中积累的环境感知可以直接拿去估值，而 critic 在大量中间状态上习得"此路不通"，这种判断力也会顺着同一组参数回流到 actor。

落到 Infra 上，这个设计有好有坏：好消息是不必再多养一个 critic 模型，rollout engine 服务的仍然是同一份权重，只是请求被分成了两类 prompt；坏消息是这两类请求的负载形状差得相当远，actor 那一侧是长上下文加短输出的多轮交互，每轮只吐几百到几千个 token，而 critic 那一侧是超长上下文加长输出的一次性推理，一次要生成一大段带反思和工具调用的分析。训练过程中，inference scheduler 需要在同一个 batch 里同时照顾这两种形状，continuous batching 的收益会被拉低不少。

> Note：实际上，按照小红书团队的反馈，训练过程是串行的，不会出现 critic 和 actor 混合 batch 的情况。

### macro-step 的策略梯度

注意到，TEMPO 每次只对一个 macro-step 做梯度更新，那它究竟还是不是在优化"整条轨迹的期望回报"这个原始目标？如果不是，那前面所有的设计都只是在优化某个替代目标，效果好也许是 by practice，而不是 by design。

dots 团队在附录里给了论证，思路是把完整轨迹按固定规则唯一地切分成 M 个连续的 macro-step，注意这个切分只是把同一组 state-action 对重新分组，并没有增删任何一项，因此完整轨迹的策略梯度可以原样按 macro-step 重新组织书写，整理之后得到的结论是：

**在切分规则固定、macro-step 数 M 固定的前提下，完整轨迹的策略梯度等价于：均匀选取一个 macro-step，只计算这个 macro-step 的梯度，再乘以 M。**

这个等价成立有一个前提，即被采样的 macro-step 以及它之前的那段前缀，都必须来自当前的 actor。而 TEMPO 的续跑机制恰好违反了这一点，因为被保存下来当作起点的前缀，是若干轮之前由旧版本 actor 生成的，起点的状态分布已经偏离了当前 actor 自己会走到的分布。dots 团队的处理是对前缀做重要性采样修正，也就是给这条样本乘上一个"新 policy 走到这个前缀的概率除以旧 policy 走到这个前缀的概率"的权重，把分布纠正回来。

值得一提的是，K1.5 的 partial rollout 面对的其实是同一个问题：被缓存的那部分轨迹由旧 policy 生成，恢复之后用新 policy 续跑，严格来说已经不是 on-policy 了。当时 Kimi 的处理相对朴素，主要靠"偏离不大"的经验判断（by practice），而 TEMPO 把尺度推到了一条轨迹可能横跨十几次参数更新，偏离已经大到不能再用经验糊弄，也就必须给出正式的修正（by design）。

### 实验结果

dots 团队在 ARC-AGI-3 公开集上取了 25 个游戏，每个模型每个游戏跑 2 次，对比 TEMPO、从同一起点训练的 GRPO、以及未经 RL 的 base checkpoint。这里的 Score 同时衡量任务进度与动作效率，推进到更深的关卡得分更高，而在达到相同进度的前提下，用掉的环境交互越少得分越高。

在最多 2048 轮交互的预算下，GRPO 相比 base 有可观提升，说明短 rollout、不使用价值模型的 RL 本身就有收益；TEMPO 相比 base 提升 31.5%，在 GRPO 的基础上再提升 20.6%；而在把关卡通过率对齐之后，TEMPO 的 Score 仍然更高。最后这一条其实比第二条更有说服力，因为分数更高还可以用"探索得更多"来解释，但相同进度下用更少的步数就没法这么解释了，它说明模型确实更早地判断出了哪条路走不通，也就是 critic 那套判断能力真的泛化回了 actor 身上。

至于绝对位置，用官方评测代码的那套 harness 来看：

| 模型 | ARC-AGI-3（arcagi3 harness） | ARC-AGI-2 |
|---|---|---|
| dots3-note Preview | 6.9 | 81.4 |
| Claude Opus 4.8 | 1.5 | 72.1 |
| GPT-5.5 | 0.4 | 85.0 |

6.9 这个绝对值当然很低，但已经是次优的四倍多，而且需要注意的是，ARC-AGI-3 考察的是能否在陌生环境中自主学习，而 ARC-AGI-2 考察的是静态抽象推理，这是两种完全不同的能力，所以才会出现 dots3 在 ARC-AGI-2 上并不领先 GPT-5.5、在 ARC-AGI-3 上却拉开一个数量级这样的组合。

## IMO 选手的 Infra 方案

开篇提到的那些设计，无论是 abort、partial rollout，还是把 KV cache 一层层向下存储，其实都共享着一个从来没被写下来的隐含前提，即**一条推理轨迹的上下文始终装得进模型的注意力窗口**。R1 时代的 reasoning 再长，几十万 token 的 rational 也还在 context length 之内，工程上要解决的是显存不够用，而不是注意力窗口需要配套做剪裁。到了 IMO 这种规模的问题，40 到 50 小时、数千轮交互的 agent 打破了这一前提。任务长度超过窗口之后，历史就必须被裁剪（在 claude code 和 codex 中称为 compact），裁剪引发的问题与显存不够不完全一样，后文会讨论到，context window 不够的时候，类似 HiCache 这种向更下层存储 KV cache 的思路本身就不存在。为了描述这一过程带来的改动，我们分步展开讨论。

### Dots Note 模型的显存开销

进入到 compact 相关讨论之前，我们来直接计算 dots 模型的显存开销，dots3 的 `max_position_embeddings` 是 524288，也就是 512K，这个窗口在今天的开源模型里已经属于最长的一档。[config.json](https://huggingface.co/dots-studio/dots3-note-prev/blob/main/config.json) 中提到，模型一共 46 层，其中 13 层 full attention、33 层 sliding window。两类层虽然都是 MLA，参数配置并不相同：

| | full 层（13 层） | SWA 层（33 层） |
|---|---|---|
| `kv_lora_rank` | 512 | 1024 |
| `qk_nope_head_dim` | 128 | 192 |
| `num_attention_heads` | 128 | 64 |
| `rope_theta` | 8e7 | 5e4 |
| sliding window | — | 513 |
| DSA indexer | 有，`index_topk=2048` | 无 |

MLA 之下，每 token 每层要存的 KV 字节数是 `(kv_lora_rank + qk_rope_head_dim) × dtype_size`，与 head 数完全无关，这是因为 MLA 落盘的是压缩后的 latent 向量，而把 latent 展开回每个 head 的 K 和 V 的那个矩阵是层的固定权重，每次前向现算，不进 cache。代入数字，BF16 下 full 层是 1152 字节，再加上 DSA index key 及其量化 scale 的 132 字节，合计 1284 字节，而 SWA 层是 2176 字节。把这两个数字和层数、序列长度乘开，一条打满 512K 的请求需要的 KV 是：

| 假想架构 | 单请求 KV | 相对 dots3 |
|---|---|---|
| 标准 MHA（128 head、head_dim 128） | 1472 GiB | 180x |
| 46 层全走 full MLA + DSA | 28.8 GiB | 3.5x |
| dots3 的 hybrid（13 full + 33 SWA） | 8.19 GiB | 1x |

这张表里有两点值得注意：其一，33 层 SWA 合计只占 40.6 MiB，而且是个常数，因为窗口 513 按 page 64 对齐到 576 个 token 之后，就与序列到底有多长彻底无关了，换算一下，8.19 GiB 里有 99.5% 都来自那 13 层 full。其二，这张表基本上就是 512K 上下文何以可行的全部答案，标准 MHA 下单请求要 1.4 TiB，一台 H200 连一条都放不下，而 MLA 把宽度从 head 数的函数压成一个超参、hybrid SWA 又把 33 层的容量从随序列增长变成常数，两者叠加之后才落到 8 GiB 这个可以工程化的量级。

顺着这个数字往下算并发，8 卡 H200 每卡 141 GiB，在 `--mem-fraction-static 0.87` 之下静态池约 122.7 GiB；BF16 权重 537 GiB 按 TP8/EP8 摊到每卡是 67.1 GiB；再给 CUDA graph 与 activation 留 8 到 12 GiB，剩下 43 到 47 GiB 归 KV。考虑到 SGLang 的部署 recipe 推荐开启 `--enable-dp-attention --dp-size 8`，attention 走的是数据并行，一条请求的 KV 完整落在某一个 rank 上而不切分，于是每卡大约能挂 5 条打满 512K 的请求，八个 rank 合计 43 到 46 条。当然，这里甚至没有考虑 KV cache 的前缀复用，是一个不太合理的简化。由此可见，单条请求 512K 的 context length，在 H200 上相当宽裕。当然，模型的 context window 从来不是由 KV cache 能存多少决定的，它是一个相当复杂的参数，取决于 position embedding 的外推范围以及训练阶段究竟见过多长的序列等等。

### 裁剪与 cache 失效

很遗憾，512K 这个窗口对于 ARC-AGI-3 这一量级的任务仍然是不够用的。dots 给出的数字是，复杂任务需要 Agent 自主进行数千轮交互，完整运行时间可达 40 到 50 小时，按每轮几千 token 粗算，累计上下文轻易就是几百万 token 的量级，比 512K 高出的可不止一个数量级。至于 IMO 那套 harness 的上下文究竟如何管理，官方只提到"让 Agent 递归生成 proof，并通过工具调用对生成的 proof 进行自我评估和增强"，具体细节并未公开。不过 blog 附录里 NL2repo 的评测配置可以拿来做个参照，那是他们公开过的、最接近这种形态的长程 agentic 设置：每个任务限 250 轮、10 小时，4 核 CPU、32 GB 内存，推理用 temperature 1.0、top-p 0.95、最大输出 49152 token、384K 上下文窗口；而当上下文超限时，裁掉较早的 reasoning 和过长的工具输入输出，但保留最新的 24 条消息以及完整的 tool-call / result 配对。

> Note：这是 NL2repo 官方 harness 的设置，和 TEMPO 无关。TEMPO 的 macro-step 把 sandbox 当成黑盒，里面怎么截断、怎么压上下文，由黑盒里的 harness 自己定义。上文这种「丢掉较早的 reasoning、只留最近若干轮」本质是截断，不是 Claude Code 那种写成一段 memory 的 summarize，和 dots model 训练过程的 harness 方案也没有直接关系。

在还没触发裁剪的时候，agentic 负载对引擎其实相当友好。第 i 轮的 prompt 完整包含第 i-1 轮的 prompt，而 radix cache 是一棵前缀树，这种只往后长、不动前面的模式命中率接近 100%，每轮真正需要 prefill 的只有新增的那一小段，也就是工具返回加上模型这一轮的输出，量级在几千 token。

而一旦触发裁剪，对 KV cache 复用的影响是相当显著的。裁掉较早的 reasoning，意味着序列中间部分被挖掉了一段，后面所有 token 的位置都要往前移；而 RoPE 编码的是绝对位置，位置一变，先前的 KV Cache 是根本不能用的。裁剪必然伴随整个上下文的重新 prefill，这是任何层级的 cache 都不可能改变的。这也是两代推理模型对 inference 系统的需求差异所在：R1 时代，大量并发的请求会让 KV cache 超出显存极限，我们选择逐级向下 offload 存储；而截断/compact，意味着 KV cache 失效，re-prefill 必然发生。

取 384K context window、每轮新增 3000 token（工具返回加模型输出）、裁剪后保留约 75%，那么首次触发大约在第 130 轮，此后每约 32 轮就要触发一次，每次需要重新 prefill 约 295K token；摊到每一轮上，平均 prefill 从 3000 token 升到约 12200 token，是无裁剪情形的 4.1 倍。这些假设当然未必准确，但结论的量级是稳定的：一条长轨迹的成本曲线并不是平稳的，而是一段一段廉价的增量 decode，被周期性的大规模 re-prefill 反复打断。

> Note：关于 compact 的等待时间，一个显而易见的方向是 looking-forward compact：一旦判断下一次 tool call 很可能把窗口撑爆，server 侧可以提前把 compact 做完，和下一次生成重叠起来，把re-prefill 的等待开销 overlap 起来。

### 模型架构对重算成本的影响

重算虽然躲不掉，但 dots3 的架构其实让它比想象中成本低了不少。我们来参考一次 295K token 的 re-prefill，计算其中 attention 部分的浮点运算量：

| | attention 计算量 | 占比 |
|---|---|---|
| 假想的 46 层全 dense | 58.4 PFLOP | — |
| dots3 实际 | 5.85 PFLOP | 100% |
| 　其中 33 层 SWA | | 2.1% |
| 　其中 13 层 DSA indexer | | 79.2% |
| 　其中 13 层 DSA 主 attention | | 18.7% |

省下这十倍的原因不难看出：33 层 SWA 每个 query 只考虑 513 个 token，复杂度是 `O(L·W)` 而不是 `O(L²)`；13 层 full 的主 attention 又被 DSA 压到了 `O(L·topk)`，每个 query 只看 2048 个 token，同样低廉。

真正值得玩味的是剩下那部分的构成，将近八成的开销落在了 DSA 的 indexer 上。原因也很直接，indexer 要先拿低维 head 把当前 query 和全部历史 token 都算一遍相关性分数，才谈得上从中挑出 top-k，所以这一步仍然是彻头彻尾的 `O(L²)`，只不过用的是 64 头 128 维，比主 attention 的 128 头 192 维便宜约三倍而已。换句话说，DSA 砍掉了主 attention 的平方项，自己却又引入了一个新的平方项，只是系数小了一些；若要为 re-prefill 这类负载做 kernel 层面的优化，该动的显然是 indexer 选 top-k 的这一步，而不是已经很便宜的主 attention。


### 稳定性与部署

一条要跑 40 到 50 小时的推理，对服务端最朴素的要求就是稳定，[SGLang Cookbook](https://github.com/sgl-project/sglang/blob/6ad3f2d8fdc8b0b0411746ef6f77f731b0339541/docs/cookbook/autoregressive/RedNote/Dots3-Note.mdx)里有几处设置很能说明问题。`--watchdog-timeout 1800` 与 `SGLANG_WARMUP_TIMEOUT=1800` 都是默认值的若干倍，一次 295K token 的重算再叠上 DeepEP 的 all-to-all，单步耗时会远超常规负载，watchdog 卡太紧很容易误杀正常请求。`--cuda-graph-backend-prefill disabled` 让 prefill 不进 graph，一方面变长 prefill 本就不适合 graph，另一方面该模型的 SWA 层在 prefill 走的是需要主机侧逐请求切序列尾巴的路径，本身也进不了 graph。`--cuda-graph-backend-decode full` 配 `--cuda-graph-max-bs-decode 32`，这个 32 与前面算出的 43 条并发恰好是同一量级，说明 graph 的覆盖范围是照着真实并发设的，而不是随手填的默认值。此外脚本还会按显存自动降级，低于 120 GiB 的卡如果不开 `--language-only`，就直接把 decode graph 关掉，因为 graph 显存与 KV 池抢的是同一块预算。

顺带一提，已合入的 cookbook 文档里明确写了 H100 跑不动 BF16 版本：8 卡合计 640 GiB，在 `--mem-fraction-static 0.87` 之下静态池约 557 GiB，而 BF16 权重本身就有 537 GiB，524288 上下文下压根没有可用的 KV 池，所以 H100 用户被直接指向 FP8。把一个不 work 的配置留在文档里、并且把账算给你看，这比悄悄删掉要有用得多。关于这类显存账的详细算法，可以参考[《当 SGLang OOM 的时候，究竟在 OOM 什么？》](../kvcache-code-walk-through/mem-fraction-static.md)。

## 参考资料

- [TEMPO: Test-Time-Scaled Value Estimation with Macro-Step Policy Optimization](https://studio.dots.ai/dots/tempo-blog.html)
- [dots 模型取得 IMO 2026 官方认证满分金牌成绩](https://studio.dots.ai/dots/imo-zh.html)
- [dots3-note Preview 官方 blog](https://studio.dots.ai/dots/dots3-zh.html)
- [HuggingFace: dots-studio/dots3-note-prev](https://huggingface.co/dots-studio/dots3-note-prev)
- [SGLang 对 dots.note.omni 的模型支持 PR #33829](https://github.com/sgl-project/sglang/pull/33829)
