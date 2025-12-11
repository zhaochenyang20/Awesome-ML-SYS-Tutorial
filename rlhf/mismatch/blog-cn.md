TLDR: 我们分享了解决训练-推理不匹配问题的两种视角：通过完全对齐 SGLang 和 FSDP 后端实现比特级精确的同策略训练来消除不匹配，以及使用 TIS/MIS 方法进行算法缓解。

这里，MIS（掩码重要性采样）包括两个操作：序列/令牌掩码 + 重要性采样。此外，我们探索了不同设置的各种组合，发现启用 MIS/TIS 不会损害 RL 性能——因此，我们建议将它们作为默认设置启用。我们有时观察到在训练后期阶段不匹配显著增加，尽管这并未导致训练崩溃，并且进一步证明了应用重要性采样可以有效抑制这种不匹配的增长。

## 什么是训练-推理不匹配？

<img src="pics/training-inference-mismatch.png" alt="训练-推理不匹配" width="30%">

训练-推理不匹配，在本文中，指的是 rollout 引擎和训练引擎之间的数值不一致。即使两个引擎使用相同的模型权重，它们也可能对相同的令牌序列产生略微不同的对数概率。这是因为 rollout 和训练引擎通常使用不同的内核、不同的批次大小、不同的激活专家和不同的归约顺序。（参考 TML 博客）

广泛认为训练-推理不匹配可能导致 RL 崩溃。但说实话，即使在像 GLM 4.6 这样的前沿模型的后训练中，我们也从未遇到过这种情况。

我们使用 K3 KL 来测量 rollout 中使用的对数概率与训练中使用的对数概率之间的差异（详见附录）。在密集模型中，K3 KL 是 e-5 到 e-3。在 MOE 模型中，K3 KL 通常是 e-3 到 e-1。尽管这种不匹配并不总是显著的，但它仍然引入了微妙的离策略：用于采样的策略与用于计算损失的策略不完全相同。在困难的任务上，例如多轮智能体，据说这种小的差异有时会随着时间的推移累积，最终使整个训练过程不稳定甚至崩溃（参考博客第 3 节）。

从所有这些意义上说，训练-推理不匹配应该被视为 RL 系统的一个不可忽视的问题。用户可以选择完全消除以获得正确性，或者缓解以提高效率。为了支持这两种需求，Slime 提供了两种解决方案，允许用户选择最符合其系统要求的权衡：

在我们的实验中，Slime 上的 RL 训练在实践中非常稳定。我们花费了大量时间试图找到一个崩溃的基线，但未能找到。如果您知道任何开源 RL 任务会在某些步骤后由于不匹配增加而崩溃，并且可以在单个节点上复现，请随时与我们联系。

## 为什么训练和推理可能不同

原因是多种多样的。例如，当批次大小较小时，内核可能使用分割归约优化，这会根据输入大小改变归约顺序。由于浮点运算是非结合的，以不同顺序累加值会引入数值差异。每个张量核心指令也可能在内部以不同的顺序执行归约（参考：TML 博客）。

因此，即使在 SGLang 中，对相同样本使用不同批次大小进行推理也可能产生略微不同的数值输出。此外，rollout 和训练在 RL 中具有根本不同的工作负载：rollout 使用微小的有效矩阵逐个生成令牌，而训练在大批次中处理完整序列。这些截然不同的矩阵形状导致系统选择不同的 GPU 内核，进一步放大了 rollout-训练不匹配。

## 不匹配的缓解

考虑到训练-推理不匹配的存在和部分原因，我们提出了两种解决方案：

1. 我们对齐 rollout 和训练之间的每个算子后端，使 rollout 对数概率和训练对数概率在比特级上完全相同。这实现了训练-推理 KL = 0，为您提供 100% 真正的同策略行为。
2. 与其强制在推理和训练中使用对齐的内核（这会降低效率到不可接受的程度），我们将 rollout 对数概率视为权威的行为策略，并使用重要性采样或拒绝采样来进行离策略 rollout 校正。

我们向社区提供这些选项，并尽力使 RL 训练更加稳定和可调试。

## 真正的同策略

正如我们所揭示的，完全消除不匹配的关键是在训练和 rollout 之间对齐所有算子后端——使训练和推理中的每个操作在比特级上相等。为了实现这一目标，我们仔细选择了用于每个模型组件的内核。

具体来说，我们使用批次不变内核：这是真正同策略的先决条件，我们采用了来自 Thinking Machines 的内核。此实现为 RMSNorm、Matmul 和其他常见算子（包括 log_softmax 和 mean）提供批次不变内核。

基于此实现，我们添加了以下实现和优化：

- FlashAttention-3：我们在训练和推理中都使用 Flash Attention 3 后端，因为它在 prefill 和 decode 操作之间实现了比特级相等，同时与 Triton 版本相比保持高效。它还支持 Radix Cache。
- DeepGEMM：在我们的真正同策略实现中，我们使用 DeepGEMM 的快速矩阵乘法作为确定性后端，这更高效。对于不同的输入大小，DeepGEMM 将使用固定的归约顺序和张量核心指令，这与形状变化无关。
- Torch.compile()：为了在启用真正同策略时提高效率，我们使用 torch.compile 通过避免许多小内核来加速。一些操作，例如 RoPE，也被编译以加速。
- 数值对齐：为了简单起见，我们还在两个系统之间对齐数值操作细节，例如操作数据类型、详细内核等。

## 算法缓解

<img src="pics/algorithmic-mitigation.png" alt="算法缓解" width="30%">

让我们首先从算法角度看看为什么这种不匹配很重要。原始 PPO 算法公式如下，其中 $$\pi_\theta$$ 表示正在优化并用于计算训练损失的当前策略，$$\pi_{\text{old}}
$$ 表示生成 rollout 数据的行为策略，即当前更新步骤之前模型的动作概率。

$$\mathcal{L}_{\text{PPO}}(\theta)
= - \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim \pi_{\textcolor{red}{\text{old}}}} \left[
  \sum_{t=0}^{|y|-1}
  \min \left(
    \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\textcolor{red}{\text{old}}}(y_t \mid x, y_{<t})} A_t,\,
    \text{clip}\left(
      \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\textcolor{red}{\text{old}}}(y_t \mid x, y_{<t})},\,
      1 - \epsilon,\,
      1 + \epsilon
    \right) A_t
  \right)
\right]$$

这是当 SGLang 和 Megatron 的输出不完全匹配时具有训练-推理不匹配问题的基本 PPO 算法。在此公式中，用于采样的策略来自 SGLang，而用于计算损失的策略来自 Megatron。这种不匹配使 PPO 损失成为重要性采样的不正确形式。

$$\mathcal{L}_{\text{PPO}}(\theta)
= - \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim \pi_{\textcolor{red}{\text{SGLang}}}} \left[
  \sum_{t=0}^{|y|-1}
  \min \left(
    \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\textcolor{blue}{\text{Megatron}}}(y_t \mid x, y_{<t})} A_t,\,
    \text{clip}\left(
      \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\textcolor{blue}{\text{Megatron}}}(y_t \mid x, y_{<t})},\,
      1 - \epsilon,\,
      1 + \epsilon
    \right) A_t
  \right)
\right]$$

## 绕过和统一的 PPO 重要性采样

<img src="pics/bypassing-ppo.png" alt="绕过和统一的 PPO 重要性采样" width="30%">

为了实现算法正确性，可以直接使用 rollout 引擎的对数概率作为离线 PPO 重要性采样中的旧策略，而不是从训练引擎重新计算的对数概率。然后它变成正确的数学形式：

$$\mathcal{L}_{\text{PPO}}(\theta)
= - \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim \pi_{\textcolor{red}{\text{SGLang}}}} \left[
  \sum_{t=0}^{|y|-1}
  \min \left(
    \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\textcolor{red}{\text{SGLang}}}(y_t \mid x, y_{<t})} A_t,\,
    \text{clip}\left(
      \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\textcolor{red}{\text{SGLang}}}(y_t \mid x, y_{<t})},\,
      1 - \epsilon,\,
      1 + \epsilon
    \right) A_t
  \right)
\right]$$

这样，训练引擎上的 log_prob 重新计算将被跳过——它将节省所有生成轨迹上的一个前向传播计算。

## 解耦的 3 策略 PPO 重要性采样

<img src="pics/decoupled-ppo.png" alt="解耦的 3 策略 PPO 重要性采样" width="30%">

然而，有时您可能希望将训练-rollout 不匹配与一般的离策略重要性采样解耦。解耦 PPO 通过解耦两个角色来实现批次无关的 PPO：近端策略（PPO 裁剪的锚定策略，控制更新大小）和行为策略（用于重要性采样中的离策略校正）。因此，在此模式中有 3 个角色参与：目标策略 $$\pi_\theta$$、近端策略 $$\pi_{\textcolor{blue}{\text{old}}}$$ 和行为策略 $$\pi_{\textcolor{red}{\text{SGLang}}}$$。$$\pi_{\textcolor{blue}{\text{old}}}$$ 在每个训练步骤开始时使用 Megatron 重新计算。总公式如下：

$$\mathcal{L}_{\text{PPO-decoupled}}(\theta)
= - \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim \pi_{\textcolor{red}{\text{SGLang}}}} \left[
  \sum_{t=0}^{|y|-1}
  \frac{\pi_{\textcolor{blue}{\text{old}}}(y_t \mid x, y_{<t})}{\pi_{\textcolor{red}{\text{SGLang}}}(y_t \mid x, y_{<t})}
  \min \left(
    \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\textcolor{blue}{\text{old}}}(y_t \mid x, y_{<t})} A_t,\,
    \mathrm{clip}\left(
      \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\textcolor{blue}{\text{old}}}(y_t \mid x, y_{<t})},\,
      1 - \epsilon,\,
      1 + \epsilon
    \right) A_t
  \right)
\right].$$

第一个重要性比率 $$\frac{\pi_{\text{old}}(y|x)}{\pi_{\text{SGLang}}(y|x)}$$ 自然地表现为动态学习率缩放项。当 rollout 分布偏离近端策略时，该比率会缩小有效更新（类似于信任域控制）。这直接连接到后面的平滑策略，该策略防止由 rollout-训练不匹配引起的大更新。

## 批次归一化与偏差-方差权衡

虽然第一个重要性比率已经充当每令牌自适应学习率控制器，但控制在批次级别仍然是随机的：从行为策略的"更容易"区域采样的批次倾向于放大有效步长，而稀有或不匹配的样本会急剧缩小它。

因此，我们强烈建议在使用序列或几何级别时启用 --tis-batch-normalize（自归一化重要性采样）。此技术解决了离策略训练中的两个关键问题：学习率稳定性和偏差-方差权衡。

在标准重要性采样中，每个批次的平均权重可能会根据采样数据在行为策略下是"可能的"还是"不可能的"而剧烈变化，这会导致有效学习率振荡并使训练不稳定。自归一化权重使其均值始终为 1，保持跨更新的步长一致，并大幅减少批次间方差。

因为这种归一化已经抑制了方差，我们可以放宽裁剪或掩码阈值，从而减少它们引入的偏差。随着批次大小变大，仅自归一化就可以使估计器既稳定又几乎无偏，而不依赖于激进的截断。

## 掩码/拒绝重要性采样

详见此处。

除了基于裁剪的重要性采样，我们还提供掩码和拒绝采样（RS）作为针对训练-推理不匹配的更强保障。当 rollout 引擎为采样的令牌分配极低概率时，重要性比率可能增长到不安全的幅度。即使被裁剪，这种情况仍然会将不正确的梯度注入训练。RS 通过丢弃这些令牌——或者，如果必要，整个序列——当比率超过预设的信任阈值时，完全避免此问题，防止有害更新生效。

此机制强制执行更有原则的信任域：如果采样的行为偏离近端策略太远，我们根本不从该样本学习。它保证所有有效的训练数据与假定的 rollout 分布保持一致，并在不匹配变得极端的情况下保护优化免于崩溃。

然而，纯拒绝采样可能会减少可用数据的数量并增加方差，特别是在不匹配适中时。因此，我们在 MIS 中将 RS 与重要性采样结合：IS 为大多数令牌保持数学校正，而 RS 仅在差异变得严重时充当安全阀。在我们的实验中，这种混合方法提供了稳定的性能，并在后期不匹配激增期间提高了鲁棒性，而不会牺牲学习效率。

## 实验

在识别一组重要性采样（IS）基线时，我们遇到了一个在大多数先前的 RLHF 或智能体训练基线中不出现的要求：我们必须能够获得模型原始响应的对数概率。

这意味着不允许对模型输出进行任何后处理，因为对响应字符串的任何修改都会破坏采样令牌与我们稍后评估其对数概率的令牌之间的对应关系。

不幸的是，许多现有的智能体基线确实依赖于轻量级后处理，通常用于简单任务，如修剪标签、删除前缀或完成部分响应。这些操作在经典智能体示例中很常见，但它们使 IS 正确 RL 的对数概率评估无效。

例如：
- Search-R1 在响应中执行后处理：
  [链接](https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/llm_agent/generation.py#L54)
- Retool 也这样做：
  [链接](https://github.com/THUDM/slime/blob/main/examples/retool/generate_with_retool.py#L147)

目前，我们还没有找到这些智能体任务需要这种后处理的可靠理论原因。幸运的是，完全删除后处理并使用模型的原始输出仍然产生与原始基线相似的奖励。因此，我们目前采用这种简单的变通方法，尽管下游影响仍然不确定。

此外，由于工程带宽有限，我们选择使用 GRPO 而不是 PPO 来演示 IS 行为。

### 不匹配的存在

我们首先确认，随着训练的进行，K3 KL 会增加。我们的设置是：
- 训练数据集：[链接](https://huggingface.co/datasets/aaabiao/dapo_filter)
- 评估数据集：aime 24 + aime 25
- 基础模型：Qwen3-4b-base（[链接](https://huggingface.co/Qwen/Qwen3-4B-Base)）
- 算法：REINFORCE (Williams et al. 1992)

<img src="pics/mismatch-existence.png" alt="不匹配的存在" width="50%">

您可以看到，在训练的初始步骤中，随着模型学习且困惑度下降，mis k3 kl 实际上下降了。但在 600 步之后，尽管训练和评估奖励保持稳定，mis K3 KL 指标开始急剧增加，表明训练和 rollout 不匹配的存在。

### IS 不会损害性能

参见我们的 weight&bias 博客。

在我们的实验中，我们还验证了启用分布校正——包括几种常用配置——不会降低性能或使训练不稳定。为了证明这一点，我们在训练开始时启用了不同的 IS 相关选项，并将它们与没有 IS 校正的基线进行比较。

下面是我们评估的四种配置：

1. 基线
2. 令牌级重要性采样（IS）
3. 令牌级 IS + 掩码/拒绝采样（RS）[也称为 MIS]
4. 令牌级 IS + 掩码/拒绝采样（RS）+ 批次归一化（BN）[也称为 MIS]

在所有设置中，我们一致观察到稳定的训练曲线。所有四种配置都成功复现了约 100 步后的特征长度增加，表明启用 IS 不会对学习动力学产生负面影响。基于这些结果，我们建议将 IS 作为默认配置启用，因为它提供不匹配校正而不会牺牲性能。

<img src="pics/is-performance.png" alt="IS 不会损害性能" width="50%">

### IS 可以抑制 KL 增加

为了测试 MIS（IS + RS + BN）是否有效，我们在第 650 步继续训练，结果如下。您可以看到，对于基础运行，kl 继续增加，但使用 MIS 后，增加趋势被成功抑制并开始下降。

<img src="pics/is-kl-suppression.png" alt="IS 可以抑制 KL 增加" width="50%">

## 使用方法

有关更多详细信息，我们提供完整的指南和可运行的示例：
- 真正的同策略训练（FSDP）：[链接](https://github.com/THUDM/slime/tree/main/examples/true_on_policy)
- 算法不匹配校正（Megatron）：[链接](https://github.com/THUDM/slime/tree/main/examples/train_infer_mismatch_helper)

如果您的目标是完全消除 rollout-训练不匹配，我们建议使用真正的同策略解决方案。

如果您希望在缓解不匹配的同时保持高性能，诸如 MIS 的算法校正是一个轻量级且有效的选择。

下面是对可用选项的简要概述。

### 真正的同策略

要打开真正的同策略模式，添加参数：

```bash
CUSTOM_ARGS=(
    --true-on-policy-mode
)
```

### 算法缓解

请参考[此链接](https://github.com/THUDM/slime/blob/main/examples/train_infer_mismatch_helper/README.md)以获取每个属性的详细完整说明。

Slime 提供了一个全面的配置系统，允许用户灵活地平衡偏差和方差。要打开重要性采样，您必须在启动脚本中添加以下属性。

```bash
CUSTOM_ARGS=(
   --use-tis
   --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)
```

然后您可以在[此链接](https://github.com/THUDM/slime/main/examples/train_infer_mismatch_helper/mis.yaml)中调整详细配置。

简而言之，您可以在四个关键维度上配置校正策略：

#### 计算级别
这决定了重要性权重如何从令牌聚合到序列。
- 令牌级别
  - 为每个令牌独立计算权重。
  - 特征：计算简单但数学上有偏差。适用于大多数一般场景。
- 序列级别
  - 序列权重是所有令牌权重的乘积。
  - 特征：数学上无偏但遭受极端方差。仅当不匹配非常小或批次大小很大时推荐。
- 几何级别
  - 使用所有令牌权重的几何平均值作为序列权重。
  - 特征：权衡解决方案。它保留序列级信息，同时避免乘积方法的数值不稳定性，在偏差和方差之间取得平衡。它还为长上下文任务提供一些长度不变属性。

#### 拒绝采样和掩码
为了防止极端重要性权重使训练不稳定并强制执行硬信任域，我们对权重应用约束。
- IS 模式（重要性采样）
  - --tis-mode：选项包括 clip 或 truncate。这强制权重保持在 [lower_bound, upper_bound] 范围内。
- RS 模式（拒绝采样）
  - --use-rs：与其限制权重，RS 直接掩码（丢弃）落在阈值之外的令牌或序列。这确保了有效数据的梯度纯度，但减少了有效的训练样本大小。

MIS 的工作引入了 IS 和 RS 在不同级别的组合。

#### 否决机制
这充当独立于 IS/RS 设置的低级安全网。
- 机制：如果序列包含任何在旧策略下概率低于否决阈值（例如，$$p < 10^{-6}$$）的令牌，则丢弃整个序列。
- 为什么需要它：它防止"灾难性更新"。即使被裁剪，分母中接近零概率的令牌也可能引入数值不稳定性或破坏性梯度。

#### 自归一化
--tis-batch-normalize：自归一化。在整个批次上归一化重要性权重，使其均值等于 1.0。这防止权重的大小使训练步长不稳定。

## 关于不匹配解决的更多内容

- Slime 还支持来自 Deepseek V3.2 的无偏 KL 估计：[链接](https://github.com/THUDM/slime/pull/1004)
- Slime 还支持 rollout 路由重放：[链接](https://github.com/THUDM/slime/pull/715)

任何不匹配解决工具都可以在 Slime 中找到！

## 参考文献
- Liu, J., Li, Y., Fu, Y., Wang, J., Liu, Q., & Shen, Y. (2025, September). When Speed Kills Stability: Demystifying RL collapse from the training-inference mismatch. Blog [https://richardli.xyz/rl-collapse]
  - Part 1: Why Off-Policy Breaks RL — An SGA Analysis Framework [https://richardli.xyz/rl-collapse-1]
  - Part 2: Applying the SGA Framework — Token v.s. Sequence-level Correction [https://richardli.xyz/rl-collapse-2]
  - Part 3: Trust Region Optimization via Sequence Masking [https://richardli.xyz/rl-collapse-3]
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3–4), 229–256.
- He, Horace and Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference", Thinking Machines Lab: Connectionism, Sep 2025.

## 附录

### K3 KL 定义

参考：[链接](http://joschu.net/blog/kl-approx.html)
K₃ KL 是基于似然比（$$\frac{p(x)}{q(x)}
$$）的 KL 散度的简单无偏估计器。它定义为：
$$ k_3(x) = \frac{p(x)}{q(x)} - 1 - \log \frac{p(x)}{q(x)}.$$

## 致谢

Chenyang Zhao, Yingru Li, Yueming Yuan, Changyi Yang, Chenxin Xie, Jiajun Li, Banghua Zhu, Yuzhen Zhou, Li Ji, Tom, Jiawei Xu, Hongyu Lu

