# dots3-note Preview：为了挑战 IMO，Infra 该做些什么？

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

Verify 如同我们所料见的一样，所有听上去用 prompt 就能控制的环境，经过训练后，效果都会有显著提升。为了强化模型具备评价自身方案的能力，dots 在 RL 阶段就把"评价当前状态"当作与"推进任务"并列的优化目标，方法叫 [TEMPO (Test-Time-Scaled Value Estimation with Macro-Step Policy Optimization)](https://studio.dots.ai/dots/tempo-blog.html)。

超长程 RL 有两点显著的困难。第一，反馈周期过长：最终奖励要等轨迹结束才能观测到，当一次 rollout 持续数十小时，任务本身的物理执行时间就构成了信号获取周期的下限。第二，信用分配困难：horizon 越长，早期决策与最终奖励之间隔着越来越长的行为链，也就更难判断哪些中间状态真正影响了结果。

一个自然的解决方向是，在轨迹尚未结束时提前估计当前状态的未来回报。无需等到最终结果出现，未完成的轨迹也能尽早产生训练信号，这正是 actor-critic 方案希望做到的事情。但传统 critic 的做法，是在模型上再外挂一个只输出分数的 value head：把当前状态输入给 value head，做一次 forward，得到标量估值就结束。value head 的单次预估实际上就是一次 forward，预估的计算量是固定的，当前模型的解题状态再难，预估也不会多想一步、多花一点算力。小红书团队指出，**当 actor 本身依赖 test-time scaling 才能解题时，估计它的中间状态同样是一个复杂推理问题**：critic 需要回顾交互历史、检查已经形成的假设、判断当前搜索方向是否可行、推演后续路径。因此，既然 actor 的能力可以随 test-time compute 增长，用于评价这些状态的 critic 也应具备同样的能力。

### macro-step

有了前文的铺垫，一个非常自然的想法是，将先前作为 critic 的简单 value head 更换为另一个生成模型，比如说用另一个体积更小的 reasoning LLM 来做 value head。如此一来，单次估值要生成一段较长的推理，可能还包含反思与工具调用，成本远高于 scalar head 的一次前向，因此它不可能像传统 critic 那样在每个 token 或每个动作之后密集运行。

TEMPO 的处理方式是把估值放在固定边界上，将连续 k 轮的模型—环境交互定义为一个 macro-step，估值只在终点做一次。作者给的理由是，在 Agent 任务中，有实际意义的状态变化通常发生在推理与工具调用完成、环境返回新观测之后，因此把多轮交互聚合起来，既给 actor 留出了完成一段完整探索的空间，也让 critic 能够在信息更充分的状态上估值。（PS：训练效率当然是另一个重要的考量 😂）

而 macro-step 除了是估值的边界之外，同时也是 rollout 与梯度更新的基本单位。GRPO 必须等待整条轨迹跑完、拿到最终奖励，才能算出 reward、做一次梯度更新；TEMPO 则在 macro-step 结束时就可以进行梯度更新，收集前一段内的环境奖励，加上 critic 对终点状态「后面还有多少 reward」的估值，来组成成这一段的完整 reward，不必等任务结束也能更新 actor。

| | 单次 rollout 长度 | 中间估值 | 覆盖完整 horizon 的方式 |
|---|---|---|---|
| GRPO | 完整 T 轮 | 无 | 每条轨迹都跑完 |
| PPO | 完整 T 轮 | scalar value head，沿途估计 | 每条轨迹都跑完 |
| TEMPO | 一个 macro-step，k 轮 | 生成式 critic，段末估一次 | 保存终点状态，跨多次更新逐步推进 |

对 Infra 而言，TEMPO 把一条 40 小时的轨迹拆成了若干个可以独立调度的段，每段都能立刻产出训练数据。这与 partial rollout 是同一条脉络上的东西，区别只在于 partial rollout 缓存的是未完成请求的生成状态，而 TEMPO 保存的是 actor 与环境的联合状态，且这样的联合状态是可以用作训练的。

### 生成式 critic 与特权信息

critic 读入当前状态与此前的交互记录，输出一段推理加上最终估值，需要分析的是三件事：actor 已经掌握的环境规律、正在尝试的方案、以及尚未解决的障碍。

比较有意思的是，critic 还可以访问 actor 看不到的特权信息，譬如环境内部状态、隐藏规则、测试方案、乃至游戏源码，这些信息不会进入 actor 的上下文，只作为评价侧的额外证据。这一条实际上把 critic 变成了一个开卷的评价者，它不需要自己解题，只需判断 actor 的方向对不对，而有了源码之后这个判断可以做得非常准。由此也能看出，"评价比生成更简单"这个假设在这里之所以成立，并不是因为评价本身天生简单，而是因为评价侧被允许使用生成侧拿不到的信息。

作者给的"放置骑士"例子很能说明问题：棋盘上已有两枚预置棋子，actor 要再放六枚，要求任意两枚之间不存在马步攻击关系，而规则事先并不告知，只能通过交互推断。从同一个 macro-step 起点采样两条各 64 轮的轨迹，两条都没有拿到新的环境奖励，从环境反馈上看它们完全一样。其中分支 A 延续了一个错误假设，把"棋子之间存在攻击关系"当成了任务目标，它对马步规则的理解其实是对的，只是目标方向反了，此后枚举的所有候选布局都要求盘面存在攻击关系，而 critic 对照游戏源码之后判定这类布局不可能满足真实完成条件，也就是说 actor 已经把全部正解排除在了搜索空间之外；分支 B 则回读了早期记录并推翻了该假设，进一步发现同色格上的棋子不会互相攻击，据此写出的布局正是五个可行解之一。

于是两条轨迹的估值出现了显著差距，而这个差距完全不来自环境奖励。可见，分支 A 犯的是"目标方向反了"这类整体性错误，它的每一步局部推理都自洽，只有把整条轨迹放在一起并对照真实规则才能看出问题，这恰恰是 scalar value head 做不到的判断。

### critic 的训练

既然 critic 是生成模型，训练上的麻烦就在于怎么用一个标量目标去监督一整段生成。

目标本身用 TD 形式构造，对同一起点采样 n 条 macro-step 轨迹，每条的回报由段内环境奖励与终点估值共同构成，取均值得到 value target V。监督方式则绕开了回归损失，因为训练数据只有数值目标、没有推理过程的监督信号，TEMPO 索性把价值拟合本身也转成了 RL 问题：在同一状态上采 m 次独立估值，按每次估值与 V 的误差给奖励，在这 m 个奖励上做组内中心化得到 advantage，再据此更新 critic。此外，估值误差还会按当前状态尚可获得的回报跨度做归一化，否则任务早期（剩余回报空间大）与末期（空间小）的误差存在量级差，梯度会被前者主导。

这样一来 actor 与 critic 就统一到了同一套 GRPO 框架之下，优化方法完全相同，区别只在 prompt 与奖励信号。由此带来的变化是，传统 critic 的能力上限由 value head 的容量决定，而这里的上限由模型愿意为一次估值花多少 token 决定，也就是说 critic 的 test-time scaling 能力本身成了被 RL 激发的对象。

由于 TD target 里含有 critic 对终点状态的自估值，也就是 bootstrap，而训练初期这个估值并不可靠，误差会沿着连续的 macro-step 向更早的状态传播，所以 TEMPO 会先做一轮 value warm-up：从离线采样的完整轨迹中按环境奖励计算 Monte Carlo return，作为不需要 bootstrap 的 value target，让 critic 先具备基本的估值能力，再进入 TD 训练。

### actor 兼任 critic

TEMPO 并不单独维护一套 critic 参数，而是让同一个模型兼任两个角色：执行 macro-step 时它是 actor，到达边界之后切换为 critic，两个角色共享参数，靠 prompt、上下文与奖励信号加以区分。这么做的理由是两类任务所需的能力高度重叠，actor 要理解任务目标、环境变化与交互历史，维护假设并规划下一步，而 critic 同样需要这些理解，用于判断进展、识别错误假设、评估后续可行性；并且参数共享还让两类训练信号可以互相迁移。

回到 IMO 那套 harness，Proof-Verify-Refine 之所以能拿满分，前提正在于此，因为模型并不是"会做事的 actor 加上会打分的 critic"，而是一个既会做事、又会评价自己的模型，Verify 环节做的事情本质上就是 critic 在做的事情，只是评价对象从游戏状态换成了一份证明。

### macro-step 的策略梯度

TEMPO 每次只优化一个 macro-step，但它的目标仍然是完整长程轨迹的期望回报，作者在附录里给了论证：按固定规则把完整轨迹唯一地切分成 M 个连续 macro-step，切分只是把同一组 state-action 对分组，并未改变集合本身，因此完整轨迹的策略梯度可以按 macro-step 重新分组书写，整理之后得到的结论是：

> 在切分规则固定、macro-step 数 M 固定的前提下，完整轨迹的策略梯度等价于：均匀选取一个 macro-step，只计算这个 macro-step 的梯度，再乘以 M。

这个等价成立的前提，是被采样的 macro-step 及其之前的前缀都来自当前 actor，而续跑机制恰好违反了这一点，因为被保存下来作为起点的前缀可能来自旧版本的 actor，起点分布会偏离当前 actor 自身的状态分布，作者的处理是对前缀做重要性采样修正。

值得一提的是，K1.5 的 partial rollout 面临的其实是同一个问题，被缓存的轨迹由旧 policy 生成，恢复之后用新 policy 续跑，严格来说已经不是 on-policy 了；当时 Kimi 的处理相对朴素，主要靠"偏离不大"的经验判断，而 TEMPO 把尺度推到了一条轨迹横跨十几次参数更新，也就必须给出更正式的处理。

### 实验结果

作者在 ARC-AGI-3 公开集上取 25 个游戏，每个模型每个游戏跑 2 次，对比 TEMPO、同起点训练的 GRPO、以及未经 RL 的 base checkpoint；其中 Score 这个指标同时衡量任务进度与动作效率，推进到更深关卡得分更高，而在达到相同进度时交互越少得分越高。

在最多 2048 轮交互的预算下，GRPO 相比 base 有可观提升，说明短 rollout、不使用价值模型的 RL 本身就有收益；TEMPO 相比 base 提升 31.5%，在 GRPO 的基础上再提升 20.6%；而在关卡通过率对齐之后，TEMPO 的 Score 仍然更高。最后这一条比第二条更有说服力，因为分数更高还可以用"探索更多"来解释，但相同进度下用更少的步数，说明模型确实更早地判断出了哪条路走不通，也就是 critic 的能力泛化回了 actor 身上。

至于绝对位置，用官方评测代码的那套 harness 来看：

| 模型 | ARC-AGI-3（arcagi3 harness） | ARC-AGI-2 |
|---|---|---|
| dots3-note Preview | 6.9 | 81.4 |
| Claude Opus 4.8 | 1.5 | 72.1 |
| GPT-5.5 | 0.4 | 85.0 |

6.9 这个绝对值当然很低，但已经是次优的四倍多，而 ARC-AGI-3 考察的是能否在陌生环境中自主学习，与 ARC-AGI-2 那种静态抽象推理是两回事，所以才会出现 dots3 在 ARC-AGI-2 上并不领先 GPT-5.5、在 ARC-AGI-3 上却拉开数量级这样的组合。

## 第二种形状：线性超长与测试时记忆

IMO 的负载是树状的，深度有限而宽度极大，ARC-AGI-3 则是另一种形状，单条轨迹线性跑几千轮，完整运行 40 到 50 小时，任务长度显著超过模型的上下文窗口。关于后者，主 blog 里有一条观察值得单独拿出来：

> 只要任务长度显著超过模型的上下文长度，通过上述强化学习方法训练模型解决问题后，模型就能自行学会生成有助于未来决策的记忆。

也就是说，写记忆这个行为并没有被单独设计奖励，它是被任务长度逼出来的，因为上下文装不下整个任务，而完成任务又需要早期信息。blog 里给的 memory 片段，先是提出假设：

```
Goal hypothesis: MERGE the two blues (overlap same cell).
Shortest path to overlap=10 moves, overlap at (1,5).
```

后来则变成了确认：

```
+- WIN CONDITION (CONFIRMED): MERGE the two blue tokens (get them to
   overlap/combine). When they merge, current_level advances (level solved).
```

坦诚说，在没有看到完整 prompt 与 harness 设计之前，很难判断这里面有多少是模型自己长出来的、多少是 harness 的结构诱导出来的。不过对 Infra 而言，这个机制的意义与它是不是"自发"无关，后文会说明，它直接决定了引擎能不能便宜地扔掉历史。

## 上一代方案的边界

开篇提到的那些设计，无论是 abort、partial rollout，还是把 KV cache 一层层向下存储，其实都共享着一个隐含前提，即**一条轨迹的上下文始终装得进窗口**。partial rollout 缓存的是一条尚未跑完但仍在窗口内的序列；HiCache 把 KV 从 GPU 挪到 host 再挪到远端，挪的也是完整前缀的 KV，换回来还能直接接着用。R1 那一代的 reasoning 再长，几十万 token 的 rational 也还在 context length 之内，工程上要解决的是"放不下显存"，而不是"放不下窗口"。

而 40 到 50 小时、数千轮交互的 agent 直接击穿了这个前提，任务长度超过窗口之后历史必须被裁剪，裁剪引发的问题与"显存不够"完全是另一类。下面按三个层次展开：单请求的 KV 账、双几何给引擎带来的改动、以及超出窗口之后的 cache 失效。

## 一条 512K 请求要多少 KV

先看 [config.json](https://huggingface.co/dots-studio/dots3-note-prev/blob/main/config.json)，模型 46 层中有 13 层 full attention、33 层 sliding window，节律是每四层一个 full，而两类层虽然都是 MLA，几何却并不相同：

| | full 层（13 层） | SWA 层（33 层） |
|---|---|---|
| `kv_lora_rank` | 512 | 1024 |
| `qk_nope_head_dim` | 128 | 192 |
| `num_attention_heads` | 128 | 64 |
| `rope_theta` | 8e7 | 5e4 |
| sliding window | — | 513 |
| DSA indexer | 有，`index_topk=2048` | 无 |

MLA 之下每 token 每层的 KV 字节数是 `(kv_lora_rank + qk_rope_head_dim) × dtype_size`，与 head 数无关；BF16 下 full 层是 1152 字节，再加上 DSA index key 及其量化 scale 的 132 字节，合计 1284 字节，而 SWA 层是 2176 字节。可见，被称作便宜的 SWA 层每 token 反而更贵，它便宜在容量而非宽度。把这两个数字乘开，一条打满 512K 的请求需要的 KV 是：

| 假想架构 | 单请求 KV | 相对 dots3 |
|---|---|---|
| 标准 MHA（128 head、head_dim 128） | 1472 GiB | 180x |
| 46 层全走 full MLA + DSA | 28.8 GiB | 3.5x |
| dots3 的 hybrid（13 full + 33 SWA） | 8.19 GiB | 1x |

其中 33 层 SWA 合计只有 40.6 MiB，而且是个常数，因为窗口 513 按 page 64 对齐到 576 个 token 之后就与序列长度无关了，也就是说 8.19 GiB 里有 99.5% 都来自那 13 层 full。这张表基本上就是 512K 何以可行的全部答案：标准 MHA 下单请求 1.4 TiB，一张卡连一条都放不下；MLA 把宽度从 head 数的函数变成一个超参，压到 28.8 GiB，仍然吃不消；再叠上 hybrid SWA，把 33 层的容量从"随序列增长"变成"常数 576 token"，才落到 8 GiB 这个可以工程化的量级。

顺着这个数字往下算并发，8 卡 H200 每卡 141 GiB，在 `--mem-fraction-static 0.87` 之下静态池约 122.7 GiB，BF16 权重 537 GiB 按 TP8/EP8 摊到每卡是 67.1 GiB，再给 CUDA graph 与 activation 留 8 到 12 GiB，剩下 43 到 47 GiB 归 KV。由于 recipe 开了 `--enable-dp-attention --dp-size 8`，attention 走的是数据并行，一条请求的 KV 完整落在一个 rank 上，于是每卡大约能挂 5 条打满 512K 的请求，八个 rank 合计 43 到 46 条（这里的预留量是我的估计，实际取决于 `--cuda-graph-max-bs-decode` 与 chunked prefill 的 buffer，量级不会差太多）。

43 条并发对应的是 `--max-running-requests 256` 这个设置，也就意味着实际负载中绝大多数请求远未打满窗口。对 IMO 那种树状负载而言这不成问题，分支虽多但每条不长；而对 ARC-AGI-3 那种线性负载，这就是硬约束了，一台机器同时挂不住几百条 40 小时的轨迹。

## 双几何在引擎里的代价

上面那张几何对照表对 SGLang 而言是个不小的麻烦，因为原有的 `SWAKVPool` 假设 full 层与 SWA 层共享同一套 KV 几何、仅仅容量不同，所以两个内部池同类同参，而 dots3 需要的是 full 侧走 `DSATokenToKVPool`、SWA 侧走 `MLATokenToKVPool`，且两侧的 `kv_lora_rank` 还不一样。对应到 [PR #33829](https://github.com/sgl-project/sglang/pull/33829)，一共有三处改动：

1. KV 池要能接受两组独立的 class 与 kwargs，见 [`_build_hybrid_mla_swa_kv_pool`](https://github.com/sgl-project/sglang/blob/4a4746c4a5d43a334abe368319f645634204a36e/python/sglang/srt/mem_cache/kv_cache_configurator.py#L1319)；
2. `pool_configurator` 里基于 `num_kv_heads × head_dim` 的 per-token 字节数公式，对 MLA 模型算出来的是个无关的数，需要按 `attention_arch` 分岔到 latent 公式，并把 DSA 的 index key 与量化 scale 显式计入；
3. attention backend 需要按 `layer.sliding_window_size` 逐层分派，full 层走 DSA 路径、SWA 层走窗口路径，见 [`DotsHybridAttnBackend`](https://github.com/sgl-project/sglang/blob/4a4746c4a5d43a334abe368319f645634204a36e/python/sglang/srt/layers/attention/dots_hybrid_backend.py#L35)。

MTP 那边还有一个连带问题，起因是 dots3 的 MTP 为全共享的一层，部署时 `--speculative-draft-model-path` 直接指向 target 自己，draft 模型由同一份 config 改写而来，引擎在派生 attention shape 之前会把 `swa_*` 那套几何整体搬到无前缀位置（[`model_config.py#L602`](https://github.com/sgl-project/sglang/blob/4a4746c4a5d43a334abe368319f645634204a36e/python/sglang/srt/configs/model_config.py#L602)）。于是 draft 层实际上是一个 SWA 层，而原来的 pool configurator 把所有 EAGLE draft 层都按 full attention 记账，会同时造成 draft 侧超配与 target 侧容量被挤掉，修法是在 spec 配置里记录有几层 draft 属于 SWA，让 configurator 分三类记账。值得注意的是，这个修改里并没有出现 `Dots3` 字符串，它修的是"draft 层的几何与容量是两个独立维度"这条通用事实。关于 MTP 与 EAGLE 家族的更多讨论，可以参考[《slime 的 speculative decoding 支持》](../../rlhf/slime/spec/readme.md)。

## 超出窗口之后：裁剪与 cache 失效

blog 附录里 NL2repo 的评测配置把长程 agentic 的上下文管理写得很清楚，每个任务限 250 轮、10 小时，4 核 CPU、32 GB 内存，推理用 temperature 1.0、top-p 0.95、最大输出 49152 token、384K 上下文窗口，而其中最关键的是这一句：

> 当上下文超限时，裁掉较早的 reasoning 和过长的工具输入输出，但保留最新的 24 条消息以及完整的 tool-call / result 配对。

在未触发裁剪的时候，第 i 轮的 prompt 完整包含第 i-1 轮的 prompt，而 radix cache 是一棵前缀树，这种单调增长的模式命中率接近 100%，每轮只需要 prefill 新增的那部分，也就是工具返回加上模型输出，量级在几千 token，agentic 负载看起来便宜的原因正在于此。

而一旦触发裁剪，情况不仅反转，还比"prefix cache 失效"要严重得多。裁掉较早的 reasoning 意味着序列中间被挖掉一段，剩余 token 的位置随之前移，而 RoPE 编码的是绝对位置，**位置一变，被保留部分的 KV 本身就是错的**，不只是命中不了，而是根本不能用。所以裁剪必然伴随整个上下文的重新 prefill，任何层级的 cache 都救不了，HiCache 把 KV 挪到 host 或远端在这里也没有意义，因为挪回来的是按旧位置编码的 KV。这也正是与上一代问题的分界线所在：R1 时代 KV 是存不下，答案是往下一层存储挪；而这里 KV 是失效而非存不下，只能重算。

这笔代价大致是可以估的，取 384K 窗口、每轮新增 3000 token（工具返回加模型输出，已属保守）、裁剪后保留约 75%，那么首次触发大约在第 130 轮，此后每约 32 轮触发一次，每次需要重新 prefill 约 295K token；摊到每轮，平均 prefill 从 3000 token 升到约 12200 token，**是无裁剪情形的 4.1 倍**。这些假设当然都可以调，但结论的量级是稳定的，即一条长轨迹的成本曲线并不是平的，而是廉价的增量 decode 被周期性的大规模重算反复打断；顺着这个结论往下，大概有三个方向值得想。

**其一，裁剪时不重编号位置。** 如果保留 token 的 position id 维持原值、留出空洞，被保留部分的 KV 就仍然有效，而 SGLang 的 page table 本来就按 block 任意 gather、positions 也是显式传入的，工程上支持非连续位置并不困难。真正的障碍在模型侧，带空洞的 position id 对模型而言是 OOD 的，除非训练时就按这种方式构造过样本，所以这是一个训练与 Infra 需要一起决定的接口，而非引擎单方面能优化的事情。

**其二，memory 机制降低了裁剪的代价。** 前文提到模型自发学会了把关键结论写进 memory，而从服务视角来看，memory 的价值并不止于"帮模型记住东西"，更在于它让扔掉历史这件事变得可以接受，因为关键信息已经被压缩成一段很短的文本，裁掉几十万 token 的原始交互并不会丢失决策依据。可见，训练侧长出来的能力，直接降低了推理侧的上下文管理成本，这是这份材料里我认为最值得注意的一处训练与 Infra 的耦合。

**其三，重算时 chunked prefill 的粒度是个全局问题。** recipe 里给的是 `--chunked-prefill-size 16384`，配合 `SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=8192` 与 `SGLANG_MAX_KV_CHUNK_CAPACITY=8192`；295K token 的 prefill 若不分块会把 decode 请求饿死很久，而在一个只有四十几条并发的场景里，被饿死的恰恰是另外几十条同样已经跑了几十小时的轨迹。

## 重算的代价被架构削掉了大半

重算虽然不可避免，但 dots3 的架构其实让它比想象中便宜不少，按 295K token 的重算来估算 attention 的打分量级，结果是这样：

| | 打分量（相对） | 占比 |
|---|---|---|
| 假想的 46 层全 dense | 7.6x | — |
| dots3 实际 | 1x | 100% |
| 　其中 33 层 SWA | | 1.7% |
| 　其中 13 层 DSA indexer | | 94.4% |
| 　其中 13 层 DSA 主 attention | | 3.9% |

33 层 SWA 的 attention 是 `O(L·W)` 而非 `O(L²)`，且 W 只有 513，这 33 层在重算里几乎是白送的；13 层 full 的主 attention 又被 DSA 压到了 `O(L·topk)`，topk 为 2048，同样很便宜，整体比全 dense 省 7.6 倍。

但剩下的开销里有 94.4% 落在 DSA 的 indexer 上，因为 indexer 要用低维 head 对全部历史打分才能选出 top-k，这一步仍然是 `O(L²)`，只是用的是 64 头 128 维、比主 attention 的 128 头 192 维便宜约三倍而已。换句话说，**在长上下文重算这个场景里，DSA 从优化项变成了主要开销项**，若要为这类负载做 kernel 层面的优化，该动的是 indexer 的打分而不是主 attention。（需要说明，这一节与上一节的分析都是我从公开配置推出来的，dots 的 blog 并没有讨论裁剪对 cache 的影响，如果他们的实现对裁剪点做了特殊处理，结论会变。）

## 小 batch decode 与 MTP

前面算出的四十几条并发，决定了 decode 阶段的性质，即**batch 小，decode 就是彻底的访存瓶颈**，每一步都要把 8 GiB 的 KV 完整读一遍，却只算四十几个 token 的 GEMM，Tensor Core 基本闲着。而这恰恰是 MTP 收益最大的区间，因为投机采样的加速比在大 batch 下会被摊薄（GEMM 已经吃满），在小 batch、访存受限的情况下，一次前向多验证几个 token 近乎白赚。recipe 里开的是 3 步 draft、4 个 draft token、eagle-topk 1，draft 模型就是 target 自己的那一层全共享 MTP，既不需要额外 checkpoint，KV 也用 SWA 几何与 SWA 容量，量级只有几十 MiB。

DSA 在 decode 阶段扮演的角色与 prefill 时并不相同：decode 时 L 已经很大，主 attention 若不做稀疏化就要读全部历史，而 `index_topk=2048` 把这一步的读取量钉死在了常数；至于 33 层 SWA，本来就只看 513 个 token，topk 比整个窗口还大，再叠 DSA 也不会有收益，所以模型只在 full 层挂 indexer 是对的。

## 稳定性与部署

一条要跑 40 到 50 小时的推理，对服务端最朴素的要求就是别崩、别卡死，[cookbook 脚本](https://github.com/sgl-project/sglang/blob/6ad3f2d8fdc8b0b0411746ef6f77f731b0339541/docs/cookbook/autoregressive/RedNote/Dots3-Note.mdx)里有几处设置值得注意。`--watchdog-timeout 1800` 与 `SGLANG_WARMUP_TIMEOUT=1800` 都是默认值的若干倍，因为一次 295K token 的重算叠加 DeepEP 的 all-to-all，单步耗时会远超常规负载，watchdog 太紧会误杀正常请求；`--cuda-graph-backend-prefill disabled` 让 prefill 不进 graph，一方面变长 prefill 本就不适合 graph，另一方面该模型的 SWA 层在 prefill 走的是需要主机侧逐请求切序列尾巴的路径，也进不了 graph；`--cuda-graph-backend-decode full` 配 `--cuda-graph-max-bs-decode 32`，这个 32 与前面算出的 43 条并发是同一量级，说明 graph 的覆盖范围是按真实并发设的。此外脚本还会按显存自动降级，低于 120 GiB 的卡如果不开 `--language-only` 就直接关掉 decode graph，因为 graph 显存与 KV 池抢的是同一块预算。

顺带一提，已合入的 cookbook 文档里明确写了 H100 跑不动 BF16 版本，8 卡 640 GiB，在 `--mem-fraction-static 0.87` 之下静态池约 557 GiB，而 BF16 权重就有 537 GiB，524288 上下文下没有可用的 KV 池，所以 H100 用户被指向 FP8。关于这类显存账的详细算法，可以参考[《当 SGLang OOM 的时候，究竟在 OOM 什么？》](../kvcache-code-walk-through/mem-fraction-static.md)。

## 训练侧：需要保存的不只是 KV

TEMPO 对 RL 系统的要求与上面讨论的推理服务又是另一套，macro-step 的续跑机制要求引擎能保存并恢复 **actor 与环境的联合状态**。actor 那一半是熟悉的，rollout engine 把一个会话的完整交互历史 checkpoint 下来、下一轮从它继续，partial rollout 早就处理过这类问题；环境那一半才是新东西，几千个并发的、各有内部状态机的环境要能被快照、存储、精确恢复到某个时刻，ARC-AGI-3 这类游戏环境尚可，而 dots 同期开源的 VibeLifeBench 那种带模拟时间线、22 个 mock 服务后端、288 个工具接口的环境，快照语义就复杂得多了。

另一个材料里没有给出答案的问题则是方差控制，因为一条轨迹横跨十几次参数更新，意味着同一任务的不同 macro-step 由不同版本的 policy 生成，而重要性权重的方差会随版本差距增大而增大，那么实践中究竟是限制一条轨迹最多跨多少次更新，还是对权重做 clip？这些才是真正落地时最难的部分，也是我最期待在技术报告里看到的内容。

## 结语

回到标题的问题上，为了挑战 IMO，或者更一般地说，为了支撑一条 40 到 50 小时的推理，Infra 需要做的事情大致可以归成三条。

第一是把 512K 上下文的 KV 从放不下变成放得下，而这一条主要由模型架构解决，MLA 把每 token 的宽度从 head 数的函数变成一个超参，hybrid SWA 又把 33 层的容量从随序列增长变成常数，两者叠加是 180 倍的差距，引擎要做的则是让两套不同的 latent 几何在 KV 池、显存记账、attention 分派这三个层面都正确。

第二是接受并发只有四十几条这个事实并围绕它优化，小 batch 意味着 decode 彻底访存受限，MTP 的收益因此被放大，也意味着一次几十万 token 的重算会拖住整台机器上其他所有长轨迹，chunked prefill 的粒度与 watchdog 的阈值都要按这个尺度重设。

第三是正视任务长度超出窗口之后的裁剪问题，这并不是上一代"KV 存不下"的延续，那个问题的答案是把 KV 往下一层存储挪，而裁剪导致的是位置重编号之后 KV 直接失效，只能重算；缓解方向目前看有两个，一是训练侧让模型学会写 memory，把扔掉历史的代价降下来，二是训练与推理一起约定不重编号位置，让引擎能保住被裁剪部分之后的 KV，前者 dots3 已经做到了，后者则还没有人做。

从 R1 到 dots3，一年半时间里 reasoning 对 inference 提出的要求变了两次，先是序列变长，答案是把 KV 往下沉，如今是任务长度超过窗口，而答案还没有定型。

## 参考资料

- [TEMPO: Test-Time-Scaled Value Estimation with Macro-Step Policy Optimization](https://studio.dots.ai/dots/tempo-blog.html)
- [dots 模型取得 IMO 2026 官方认证满分金牌成绩](https://studio.dots.ai/dots/imo-zh.html)
- [dots3-note Preview 官方 blog](https://studio.dots.ai/dots/dots3-zh.html)
- [VibeLifeBench](https://vibebench.github.io/VibeLifeBench_homepage/) 与 [VibeSearchBench](https://vibebench.github.io/VibeSearchBench.github.io/)
- [ARC-AGI-3 评测代码](https://github.com/harbor-framework/harbor/pull/2728)
- [HuggingFace: dots-studio/dots3-note-prev](https://huggingface.co/dots-studio/dots3-note-prev)、[GitHub](https://github.com/studio-dots-ai/dots3-note-prev)、[Transformers PR](https://github.com/huggingface/transformers/pull/47844)
- [SGLang PR #33829](https://github.com/sgl-project/sglang/pull/33829)
