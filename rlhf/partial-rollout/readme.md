# Kimi K1.5: Long Context RL 的成功实践

感谢 kimi 团队的这篇雄文。大概在 DeepSeek 开源 R1 的同一时间，就有许多朋友向我推荐过 K1.5 的技术报告。苦于工作繁琐，一直没有空拜读。正好最近去西雅图出差，在 LAX 往返 SEA 的路上终于有时间虔诚拜读这一工作。越读起来，越有一种文人相赏，难逢知音的感觉。

本科毕业以来，确实有感受到自己在团队协作和个人能力上的长足长进。但是博士入学之后，我一直没有什么高强度投入的工作发表，未免感到焦虑。今天读完这样的雄文，心情大爽。希望自己能在余下的科研生涯中多参与这样具有开源精神的重磅工作。能让自己的名字出现在此番工作的作者名录之上，不比 xxxxx 强？当然，这就又带来了在大项目中，如何证明自己 credit 的问题。虽然如此，我总归相信自己的想法仍是大有裨益的。

絮絮叨叨说了这么多，这篇文章主要复盘自己拜读 K1.5 技术报告的思索。由于是技术报告，这篇扎实的文章涵盖了从数据、训练方法到训练系统的方方面面，读完真是余音绕梁，不绝如缕。

## RL Recipe

K1.5 的训练可以细分为 pretrain，vanila SFT，long-CoT SFT 和 RL 四个阶段。技术报告主要讲述的是 RL 阶段的故事。

### RL prompt 选择

高质量的 RL prompt 需要 diverse, balance and accurate to evaluate。为了决定每个 prompt 的难度，作者采用一个 SFT 模型在较高的 temperature 下生成 10 次答案，以 10 次内的通过率作为难度，来平衡训练时的样本。此外，一些复杂的推理问题通过错误的推导也能猜出正确答案。为了避免此类 reward hacking，作者进一步确保每个 prompt 的 reasoning path 和 final answer 都能被准确验证。作者先排除了容易出现此类错误的题目，例如多选题、判断题和证明题。然后，作者进一步过滤掉一些容易猜测出答案的问题。具体来说，给模型 8 次机会，如果在没有 CoT 的情况下，有超过 1 次可以直接给出答案，就将其移除。

### Long-CoT SFT

作者通过 prompt engineering 构建了一个 multi-modal long-CoT warmup dataset，来让模型初步学会这几种推理能力，evaluation，reflection，exploration 和 planning。

### Length Penalty

在进行 long context RL 训练的过程中，如果不对模型的输出长度做出控制，会很容易观测到 answer length 的显著增加。虽然这带来了更好的性能，但过长的推理过程在训练和推理时成本高昂，而且人类通常不倾向于过度思考。因此，作者引入了一个长度惩罚项，来控制模型的输出长度：

$$ \text{len\_reward}(i) = \begin{cases} \lambda & \text{if } r(x, y_i, y^*) = 1 \\ \min(0, \lambda) & \text{if } r(x, y_i, y^*) = 0 \end{cases} $$

$$ \lambda = 0.5 - \frac{\text{len}(i) - \text{min\_len}}{\text{max\_len} - \text{min\_len}} $$

式中，$r(x, y_i, y^*)$ 是第 $i$ 个推理过程的 reward，可以简单理解为正确性。

分开想想这几种情况：

1. 最长的推理过程，正确的答案。$len\_reward = -0.5$，抑制模型生成此推理过程。
2. 最长的推理过程，错误的答案。$len\_reward = -0.5$，抑制模型生成此推理过程。
3. 最短的推理过程，正确的答案。$len\_reward = 0.5$，鼓励模型生成此推理过程。
4. 最短的推理过程，错误的答案。$len\_reward = 0$，对模型没有影响。

对于长度在 $\frac{\text{max\_len} + \text{min\_len}}{2}$ 以下的的推理过程，如果答案错误，则 length reward 为 0，如果答案正确，则 length reward 大于 0 且随着 length 递减。超过这个长度的 reasoning path，无论答案正误与否，都给一样的负分 length reward。

### 采样策略

虽然 RL 本身具有较好的采样特性（难度更高的问题会提供更大的梯度），但其训练效率仍然有限。一些明确的先验采样方法可能会带来更大的性能提升。作者采用的方案有：

1. 课程学习：从较简单的任务开始训练，逐步过渡到更具挑战性的任务。由于初始 RL 模型的性能有限，在非常困难的问题上花费有限的计算资源往往只能产生很少的正确样本，训练效率低下。同时，用于训练的数据天然就包含难度标签，进一步让难度递增学习策略直观且有效。
2. 优先采样：成功率较低的问题采样更多次。我们跟踪每个问题 $i$ 的成功率 $s_i$，并按照 $1 - s_i$ 的比例采样问题，从而使成功率较低的问题获得更高的采样概率。这将模型的努力集中在它最薄弱的领域，从而加速学习并提升整体性能。

### Code, Math and Visual Data

由于爬虫的合规性限制，很多网上得到的 coding 问题是没有测例的。为此，作者使用 [CYaRon1](https://github.com/luogu-dev/cyaron) 来生成测例以计算 reward。当然，这需要很多严苛的假设，譬如不需要特殊的评判，并且这些问题有可用的 ground truth solution，以便利用这些 solution 生成更高质量的测试用例。

至于数学，评估数学问题的一大挑战是，不同的表达形式可能代表相同的答案。例如，$a^2 - 4$ 和 $(a + 2)(a - 2)$ 可能都是同一个问题的有效解。为了提高奖励模型的评分准确性，作者尝试了两种方法：

1. Classic RM：参考 InstructGPT，作者实现了一个基于 value-head 的奖励模型，并收集了约 80 万个数据点进行微调。该模型最终以 `{question, answer, reference answer}` 作为输入，输出一个标量，指示响应是否正确。
2. Chain-of-Thought RM：最近的一些工作证明，加入了 CoT 的 reward model 在需要细致标准的任务上效果显著更优。作者继续收集了约 80 万个带有 CoT 的数据集来微调 reward model。在与 Classic RM 相同输入的基础上，CoT model 明确生成逐步推理过程，最后以 JSON 格式提供最终的 reward。在作者的 case study 中，Classic RM 的准确率约为 84.4，而 CoT RM 达到了 98.5 的准确率。最终，他们采用了 CoT RM 以确保更正确的反馈。

最后，Vision RL 的数据主要来源于三大类：Real-world data，Synthetic visual reasoning data 以及 Text-rendered data。其中，text-rendered data 的做法颇有意思，将文本内容转换为图片，专门强调了模型处理文本密集图像的能力。

### Long2Short 训练

Long-CoT 模型的推理开销显著更大，作者指出了多种可以将 Long CoT 能力迁移到 Short CoT 上。比如 model merging，shortest rejection sampling，DPO 还有 Long2Short RL。这里首先分享感受上最暴力的方法——model merging。非常简单，直接把 long cot 模型和 short cot 模型的参数取平均值，就得到了更好的 short cot 模型。再者，shortest rejection sampling 就是每次从所有正确的 samples 中，选择最短且正确的答案作为最终的 sample。当然，rejection sampling 的另一面就是 DPO，可以把短的错误答案和正确的长答案都作为 negative sample，构造 pairewise preference data。同样，简单的 rejection sampling 也可以广泛用于 math 和 coding 问题上，因为 rule-based verification 比起人自身还准确。在 SFT 阶段，作者也利用 rejection sampling 来扩充数据集。

## RL Infra

这是我最感兴趣的部分了，作者在他们的 RL 系统中重点讲述了 partial rollout 这一想法，这是最重要的创新了。

### Partial Rollout

【这部分来自我和 Yuzhen Zhou 写的 proposal，就直接搬过来了，写的很具体】

**问题描述**

众所周知，大规模工业级 RLHF 系统中，推理阶段（rollout）占据了整体流程过半数的开销。 考虑一次多任务（mutli-tasks）的 PPO 训练，不同任务在解码长度（ decode length）上存在显著差异；即便是同一任务内，其 decode length 也存在不均衡。

具体来说，目前 rollout 阶段，rollout engine 主要采用 data parlllesim 的方式，每个 rollout worker 负责一部分的采样任务，各自维护并且完成自身的全部 requests。每个 worker 上需要处理的 requests 数量是相近的，但是 requests 之间的 decode length 差距显著。然而，requests 被更上层的 DP Manager 发送到各个 worker 后，worker 之间不会再交换这些 requests。比如将 10 万条 prompt 分给 8 个 worker，每个 worker 平均而言各自处理 1.25 万条。这种完全割裂的结构使得一旦某个 shard 内部存在“慢任务”，整个训练就会被这个 shard 拖住，GPU 资源无法充分利用。如此以来，decode length 的不均衡直接导致了 rollout 阶段可能会出现严重的长尾阻塞问题：在严格 on-policy 的前提下——用于当前 iteration 训练的 tracjories 必须由当前 iteration 的 policy model rollout 得到——一些 requests 完成 rollout 很快，另一些却需要很长时间，导致整个 rollout 阶段的流水线被处理慢任务的 worker 阻塞，资源利用率显著下降。且随着多任务训练的数据量加大，这种阻塞越发显著。

更加严重的是，目前的 dp manager 采用的 routing 策略多以 prefix maximum 为主，也即发送到每个 worker 上的 request 彼此尽可能存在 shared prefix。这种 prefix maximum 的 routing policy 会将具有相似 prefix 的任务分配到同一个 worker 上，而与 long decode 任务相似 prefix 的任务也更大概率是 long decode 任务。prefix maximum 虽然节省了 prefill 开销，但它可能会将大量的 long decode request 都发到了同一个 worker 上，这加剧了我们先前描述的情况。

总结一下，现有的 Rollout 流程的 imbalance 来自于以下几个方面：

- Rollout 同步：因为严格 on-policy 要求，全部的 rollout 请求完成后，才能统一进入训练阶段；
- 长尾任务拖慢整体：部分计算资源完成短任务后空等，资源浪费；
- DP 设置下任务隔离，无法跨 worker 调度等待队列；
- 主流 routing policy 在节约 prefill 开销的同时忽略了 decode-heavy 任务造成的影响。

**可行方案 Partial Rollout**

为了解决上述问题，近期出现了一种广受关注的优化方案——Partial Rollout。这个策略的核心思想是在一定程度上牺牲 on-policy 要求，不再等待所有 prompt 全部完成推理，而是增大采样量，挑选出已经完成 rollout 的部分先进行训练，剩下的未完成的样本延后处理。

举个简单例子：

- 每轮训练仅需 128 个样本，但同时启动 512 个  requests 进行推理；
- 当有 128 个 prompt rollout 完成时，立即进入训练流程；
- 剩余 384 个未完成 rollout 的 prompt：
  - 继续使用当前 policy 异步完成（如果训练和推理互不干扰）；
  - 或中止并缓存当前生成状态，在后续迭代中恢复，继续推理或重头开始。

**策略权衡**

使用 Partial Rollout 需要考虑一个 policy model 的选择问题：这些样本是继续使用它们最初启动时的旧模型（旧 policy）继续推理，还是使用当前已经更新的模型（新 policy）重新开始。无论采取何种具体补全策略，一旦允许先训练已完成的部分、而后处理剩余未完成任务，就不可避免地引入了训练数据的策略不一致性——即训练数据不再严格来源于当前最新的 policy，可能夹杂了部分旧 policy 生成的轨迹。这种非一致性并非偶然，而是 Partial Rollout 设计带来的必然，它是对训练效率做出的主动妥协。我们需要在（非）严格 on-policy、训练效果、推理吞吐量之间进行取舍。

回到 partial rollout 本身，对未在当前 iteration 完成采样的样本，如果使用参数更新后模型继续完成 rollout，那么这部分数据的生成过程会跨越多个 policy 的阶段，不再是完全的“on-policy”训练数据。这节省了时间和算力，提升了整体训练效率，不过效果没有严格保证。如果丢弃 rollout 未完成的样本，坚持使用最新 policy 从头来生成全部训练数据，虽然保证了训练的严谨性，但之前未采用的样本的推理推理开销就被浪费了，影响了资源利用率。

Partial Rollout 的核心问题不可回避，它必然在训练效率和策略一致性之间做出权衡：它以牺牲一部分策略新鲜度为代价，显著提升了训练的吞吐率，并减少了资源浪费。目前虽然缺乏明确的理论证据证明其负面影响，但我们应当正视它对策略学习带来的潜在影响，并在工程效率与算法严谨性之间做出理性取舍。

**如果想维护 strictly on policy 呢？**

其实 gradient accumulation 提供了一个似是而非的参考。假如我们有 1024 个数据，将它分成 1024 / 4=256 个数据一 batch。每个 iteration 都将整个 batch 完整的计算 loss 并且更新梯度，但算完 4 批（256 * 4）后再去更新模型参数。这样确实可以保证 on policy。但是，这个方法乍一听可以做，个人认为实际上意义不大：

很直观，gradient accumlation 解决的是一次性利用大量的【完整】 requests 会 OOM的问题，但是 patial rollout 试图解决是获得【完整】 requests 需要时间特别久的问题。即便用了 gradient accumlation，到底还是会花费大量的时间来在一个 iteration 内，从零开始得到【完整】的 trajcotries。 此外，分多次去组织一个大的 batch size 在情况允许时，效率肯定不如一次性用一个大的 batch size 好，因为同样的数据同样的更新规模（达到效果是相同的），gradient accumlation 还增加了频繁切换 Rollout 和 Training 的开销。

long context RL 还存在别的系统侧问题：在现有 rollout 引擎的部署过程中，context length 需要在 server 启动前指定，因为要直接编译进入计算图中。如果在推理过程中出现了 decode 特别长的请求，超出了当前设置的 context length，就只能重启 engine，重新设置一个更长的 context length。我们想要找寻更好的解决方案，从系统和算法层面减少这种重启的频率，或者降低重启的开销。

**总结**

总的来说，Partial Rollout 提供了一种灵活、高效的多任务 RL 训练方式来降低 long context RL 时 decode 阶段的巨大开销。它提升了资源利用率和训练节奏，是应对多任务推理长尾问题的有效手段。然而，这一策略的核心代价是引入了不可避免的 off-policy 训练现象，这是所有类似异步优化设计在多任务环境中都难以完全避免的。因此，如何缓解策略偏移对最终模型性能的影响，是我们需要研究的问题。

### RL framework

回到 k1.5 上，作者搭建了基于 megatron 和 vllm 的大规模 RL 系统。在这种工业级实践中，他们关注的和我们确实区别很大。比如，他们会关注 rollout stage 和 training stage 之间的启动间隔，从训练到 rollout 需要 1min，反过来需要 10s。此外，还涉及到了我从没考虑过的变量——checkpoint engine。简单来说，随着 rollout 的逐步进行，需要的 trajactoris length 越长，rollout engine 开始设置的 context length 可能就不够大了。目前的做法是反复 kill and relaunch 新的 rollout engine，这需要存下 ckpt 并且尽可能降低重启的开销。

在 code 执行方面，他们采用了 crun 而不是 dokcer，效率直观上快了很多。

## Ablation

- 大模型短推理和小模型长推理：小模型可以用长推理来比肩大模型，但是总体上，大模型的 token efficiency 是更好的。large model with short context 和 small model with long context 目前是效果一致的方案。
- 与 ReST 的方法相反，在训练中引入 negative gradient 能够显著增强模型生成 long cot 的效率。
- 课程学习（由易到难学习）带来了显著的提升。