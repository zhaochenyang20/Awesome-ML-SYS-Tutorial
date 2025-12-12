# 愿速度与稳定同在：基于 Slime 的训推不一致全栈解决方案

TL;DR：本文探讨了 RLHF 中的“训练-推理不匹配”问题，并介绍了 Slime 框架提供的两种解决方案：通过后端完全对齐实现的**真正同策略（True On-Policy）**训练，以及基于 TIS/MIS 的**算法**层面缓解。我们建议将TIS/MIS作为默认配置，因为它能在不损耗性能的前提下有效稳定训练。

训练-推理不匹配是指 Rollout（推理）引擎与训练引擎之间存在的数值不一致，这可能会破坏R的稳定性。在本文中，我们分析了这种不匹配产生的原因，并介绍了 Slime 提供的两种解决方案。对于追求绝对正确性的用户，我们提供了**true On-Policy**模式，实现了 SGLang 与 FSDP 之间的比特级对齐；对于更看重效率的用户，我们提供了如掩码重要性采样（MIS）等**算法**缓解方案。我们的实验表明，MIS 能有效抑制训练后期的不匹配增长，同时不影响模型performance，推荐作为默认设置开启。

## 什么是训练-推理不匹配？

<img src="pics/training-inference-mismatch.png" alt="Training Inference Mismatch" width="30%">

在本文中，“训练-推理不匹配”指的是 Rollout（生成）引擎与训练引擎之间存在的数值不一致性。即使两个引擎使用完全相同的模型权重，针对相同的 Token 序列，它们计算出的 log-probabilities 也可能存在细微差异。造成这种情况的原因在于，Rollout 和训练引擎通常使用不同的算子、不同的批次大小（Batch Size）、不同的专家激活策略以及不同的归约顺序（Reduction Order）。（Ref Thinking Machine Lab [博客](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)）

> 业界存在观点认为，训练-推理不匹配可能导致 RL 训练崩溃（Collapse）。但坦率地说，即使在像 GLM 4.6 这种前沿模型的Post-training阶段，我们也从未遇到过这种情况。

我们使用 K3 KL 散度来衡量 Rollout 阶段与训练阶段所使用的对数概率之间的差异（详见附录）。在 Dense 模型中，K3 KL 通常在 1e-5 到 1e-3 之间；而在混合专家（MoE）模型中，K3 KL 通常在 1e-3 到 1e-1 之间。尽管这种不匹配看起来并不总是非常显著，但它引入了一种微妙的 'off-policy' 效应：采样时使用的策略与计算损失时使用的策略并不完全相同。在诸如多轮对话 Agent 等复杂任务上，据说这种微小的差异有时会随时间累积，最终破坏整个训练过程的稳定性，甚至导致崩溃（参考博客[第 3 节](https://richardli.xyz/rl-collapse)）。

从这些角度来看，训练-推理不匹配应当被视为 RL 系统中一个不可忽视的问题。用户可以选择彻底消除它以确保正确性，或者采取缓解措施以兼顾效率。为了同时满足这两种需求，Slime 提供了两种解决方案，允许用户根据自身系统的需求进行权衡选择。

在我们的实验中，Slime 上的 RL 训练在实践中表现得非常稳定。我们花了大量时间试图复现出一个因不匹配而崩溃的baseline，但始终未能如愿。如果您知道任何开源的 RL 任务会因为不匹配增加而在若干步后崩溃，并且能在单节点上复现，欢迎随时联系我们。

## 为什么训练和推理结果会不同？

原因多种多样。例如，当批次较小时，算子可能会使用分割归约（Split-reduction）优化，这种优化会根据输入大小改变归约顺序。由于浮点运算不满足结合律，不同的累加顺序会引入数值差异。每个 Tensor Core 指令在内部执行归约时，顺序也可能不同（Ref：Thinking Machine Lab [博客](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)）。

因此，即使在 SGLang 中，使用不同的批次大小对同一样本进行多次推理，也可能产生略微不同的数值输出。此外，Rollout 和训练在 RL 中的负载特征截然不同：Rollout 是decode process, 一次生成一个 Token，涉及的有效矩阵计算极小；而训练则是prefill process, 大批量处理完整序列。这种矩阵形状的巨大差异导致框架之间选择不同的 GPU 算子，进一步放大了 Rollout 与训练之间的不匹配。

## 如何缓解不匹配

鉴于训练-推理不匹配的存在及其成因，我们提出两种解决方案：

1.  **True On-Policy**：我们将推理和训练之间的每一个算子后端都进行对齐，确保推理的对数概率与训练的对数概率在比特级上完全一致。这实现了训练-推理 KL = 0，从而提供 100% 真正的同策略（True On-Policy）行为。
2.  **算法修正**：与其强制在推理和训练中使用完全对齐的算子（这会对efficiency有比较大的影响），不如将 Rollout 的对数概率视为权行为策略（Behavior Policy），并利用重要性采样（Importance Sampling）或拒绝采样（Rejection Sampling）来进行异策略 Rollout 的修正。

## 真正的同策略（True On-Policy）

正如前文所述，彻底消除不匹配的关键在于对齐训练和推理之间的所有算子后端——确保训练和推理中的每一个操作在比特级上相等。为了实现这一目标，我们对模型各组件使用的算子进行了精挑细选。

具体来说，我们使用了**批次无关（Batch-invariant）算子**：这是实现真正同策略的先决条件。我们采用了 Thinking Machines 提供的算子实现，涵盖了 RMSNorm、Matmul 以及其他常用算子（如 log_softmax 和 mean）。

在此基础上，我们还加入了以下实现与优化：

-   **FlashAttention-3**：我们在训练和推理中均使用 Flash Attention 3 后端，因为它在 Prefill和 Decode操作之间能保持比特级一致，且相比 Triton 版本效率比较高。它同时也支持 Radix Cache。
-   **DeepGEMM**：我们使用 DeepGEMM 的快速矩阵乘法作为确定性后端，效率更高。对于不同的输入大小，DeepGEMM 会使用固定的归约顺序和 Tensor Core 指令，不受形状变化的影响。
-   **Torch.compile()**：为了在启用“真正同策略”模式时提升效率，我们使用 `torch.compile` 来避免大量细碎算子的开销。部分操作（例如 RoPE）也经过编译加速。
-   **数值对齐**：为了简化问题，我们还对齐了两个系统间的数值操作细节，例如算子数据类型（dtype）、具体的算子实现等。

## 算法层面的缓解（Algorithmic Mitigation）

<img src="pics/algorithmic-mitigation.png" alt="Algorithmic Mitigation" width="30%">

让我们先从算法角度看看为什么这种不匹配至关重要。原始 PPO 算法的公式如下，其中 $$\pi_\theta$$ 表示当前正在优化并用于计算训练损失的策略，而 $$\pi_{\text{old}}$$ 表示生成 Rollout 数据的行为策略，即当前更新步骤之前的模型动作概率。

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

当 SGLang 和 Megatron 的输出不完全匹配时，这就是带有训练-推理不匹配问题的基础 PPO 算法。在这个公式中，用于采样的策略来自 SGLang，而用于计算损失的策略来自 Megatron。这种不匹配使得 PPO 损失变成了一种不正确的重要性采样形式：

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

## 绕过与统一 PPO 重要性采样

<img src="pics/bypassing-ppo.png" alt="Bypassing and Unified PPO Importance Sampling" width="30%">

为了实现算法上的正确性，我们可以直接使用SGLang输出的对数概率作为离线 PPO 重要性采样中的“旧策略（Old Policy）”，而不是使用Megatron重新计算出的对数概率。这样，公式就回归到了正确的数学形式：

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

通过这种方式，我们可以跳过Megatron上的 log_prob 重计算步骤——这能节省所有生成轨迹的一次前向传播（Forward Pass）计算量。

## 解耦的三策略 PPO 重要性采样

<img src="pics/decoupled-ppo.png" alt="Decoupled, 3-policy PPO Importance Sampling" width="30%">

然而，有时你可能希望将“训推不匹配”与“通用的重要性采样”解耦。解耦 PPO（Decoupled PPO）通过分离两个角色来实现批次无关（Batch-independent）的 PPO：**近端策略（Proximal Policy）**（作为 PPO 裁剪的锚点，控制更新幅度）和**行为策略（Behavior Policy）**（用于重要性采样中的异策略修正）。因此，该模式涉及三个角色：目标策略 $$\pi_\theta$$、近端策略 $$\pi_{\textcolor{blue}{\text{old}}}$$ 以及行为策略 $$\pi_{\textcolor{red}{\text{SGLang}}}$$。其中，$$\pi_{\textcolor{blue}{\text{old}}}$$ 是在每个训练步开始时使用 Megatron 重新计算得到的。总公式如下：

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
\right]$$

第一个 importance ratio $$\frac{\pi_{\text{old}}(y|x)}{\pi_{\text{SGLang}}(y|x)}$$ 自然地充当了一个动态学习率缩放项。当 Rollout 分布偏离近端策略时，该比率会缩小有效更新幅度（类似于Trust Region控制）。这直接关联到后文提到的平滑策略，旨在防止由 Rollout-训练不匹配引发过大的更新。

## 批次归一化与偏差-方差权衡（Bias-Variance Trade-off）

虽然第一个Importance Ratio已经起到了 Token 级自适应学习率控制器的作用，但这种控制在批次（Batch）层面仍是随机的：从行为策略较匹配的区域采样出的批次往往会放大有效步长，而稀有或不匹配的样本则会急剧缩小步长。

因此，我们强烈建议在使用**序列级（Sequence Level）**或**几何级（Geometric Level）**计算时启用 `--tis-batch-normalize`（自归一化重要性采样）。这项技术解决了异策略训练中的两个关键问题：**学习率稳定性**和**偏差-方差权衡**。

在标准重要性采样中，每个批次的平均Importance Ratio可能会因采样数据在行为策略下是“高概率”还是“低概率”而剧烈波动，导致有效学习率震荡，破坏训练稳定性。对权重进行自归一化（使其均值始终为 1）可以保持更新步长的一致性，并大幅降低批次间的方差。

由于这种归一化已经抑制了方差，我们可以放宽clip或mask的阈值，从而减少它们引入的bias。随着批次增大，仅靠自归一化就能使估计器既稳定又近乎无偏，而无需依赖激进的截断操作。

## 掩码/拒绝重要性采样（Masked / Rejection Importance Sampling）

> 完整细节请参阅[此处](https://richardli.xyz/rl-collapse-3)。

除了基于裁剪的重要性采样外，我们还提供了**掩码（Masking）**和**拒绝采样（Rejection Sampling, RS）**，作为应对训推不匹配的更强力保障。当 Rollout 引擎为某个采样 Token 分配的概率极低时，重要性比率可能会增长到不安全的数量级 (比如1e12)。即使经过裁剪，这类情况仍可能向训练中注入错误的梯度。RS 通过在比率超过预设信任阈值时直接丢弃这些 Token（必要时甚至丢弃整个序列）来彻底规避此问题，防止有害更新生效。

这种机制强制执行了一个更有原则的信任域（Trust Region）：如果采样的行为偏离近端策略太远，我们干脆不从该样本中学习。这保证了所有有效的训练数据都与假设的 Rollout 分布保持一致，并在不匹配变得极端时保护优化过程免于崩溃。

然而，纯粹的拒绝采样可能会减少可用数据量并增加方差，尤其是在不匹配程度中等时。因此，我们在 MIS 中将 RS 与重要性采样（IS）相结合：IS 负责对大多数 Token 进行数学修正，而 RS 仅在差异严重时充当“安全阀”。在我们的实验中，这种混合方法在训练后期不匹配激增时提供了稳定的性能，提高了鲁棒性，且未牺牲学习效率。

## 实验

在确定baseline时，我们遇到了一个在大多数过往 RLHF 或 Agent 训练基线中未曾出现的要求：**我们必须能够获取推理引擎产出的对数概率**。

这意味着不允许对模型输出进行任何后处理（Post-processing），因为对响应字符串的任何修改都会破坏“采样 Token”与“后续评估 log-prob 的 Token”之间的对应关系。

遗憾的是，许多现有的 Agent 基线确实依赖一些轻量级的后处理，如修剪ta、移除前缀或补全部分响应。这些操作在经典 Agent 示例中很常见，但它们会使IS无法正确进行。例如：

- Search-R1 在响应中执行后处理：[链接](https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/llm_agent/generation.py#L54)
- Retool 也是如此：[链接](https://github.com/radixark/miles/blob/main/examples/retool/generate_with_retool.py#L147)

目前，我们尚未理解为何这些 Agent 任务必须依赖此类后处理。幸运的是，完全移除后处理并直接使用模型的原始输出，仍然能获得与原始基线相似的reward。因此，我们目前采用了直接去除这种简单的变通方案。

此外，由于有限的时间和资源，我们选择使用 GRPO 而非 PPO 来演示 IS 的行为。

### 不匹配现象的存在

我们首先确认，随着训练的进行，K3 KL 确实会增加。我们的设置如下：
- 训练数据集：[Dapo](https://huggingface.co/datasets/aaabiao/dapo_filter)
- 评估数据集：aime 24 + aime 25
- 基座模型：Qwen3-4b-base ([链接](https://huggingface.co/Qwen/Qwen3-4B-Base))
- 算法：REINFORCE (Williams et al. 1992)

<img src="pics/mismatch-existence.png" alt="Existence of Mismatch" width="50%">

可以看到，在训练初期，随着模型学习且Perplexity下降，mis K3 KL 实际上下降了。但在 600 步之后，尽管训练和评估奖励保持稳定并未下降，mis K3 KL 指标却开始急剧上升，表明训练和 Rollout 之间的不匹配确实存在并且在训练后期会加大。

### IS 不会损害Performance

>  完整的wandb log请看 [此处](https://wandb.ai/ch271828n-team/slime-dapo/reports/IS-Has-No-Harm--VmlldzoxNTE3NTM3MQ?accessToken=vbaw93cjkyi8d6iul7gzvccehf2ugff1cicfcmlaxjv88n875i0ip1ixqfr42s9b)。

我们在实验中还验证了启用TIS/MIS（包括几种常用配置）并不会降低性能或破坏训练稳定性。为了证明这一点，我们在训练开始时启用了不同的 IS 相关选项，并将其与未进行 IS 修正的基线进行了对比。
我们评估了以下四种配置：

1.  Baseline（基线）
2.  Token-level Importance Sampling (IS)
3.  Token-level IS + Masking/Rejection Sampling (RS) [即 MIS]
4.  Token-level IS + Masking/Rejection Sampling (RS) + Batch Normalization (BN) [即 MIS]

在所有设置中，我们均观察到了稳定的训练曲线。所有四种配置都成功复现了约 100 步后的Response Length增加现象，这表明启用 IS 不会对学习动态产生负面影响。基于这些结果，我们建议将 IS 作为默认配置启用，因为它在不牺牲性能的前提下提供了mismatch修正。

<img src="pics/is-performance.png" alt="IS Won't Harm Performance" width="50%">

### IS 可以抑制 KL 增长

为了测试 MIS (IS + RS + BN) 是否有效，我们从第 650 步继续训练，结果如下。可以看到，对于 Base Run（基线运行），KL 继续增加；但使用了 MIS 后，增长趋势被成功抑制并开始下降。

<img src="pics/is-kl-suppression.png" alt="IS Can Supress KL Increase" width="50%">

## 使用指南

如果想要了解更多细节，我们提供了完整的指南和可运行的示例：
- **True On-Policy (FSDP)**：[链接](https://github.com/THUDM/slime/tree/main/examples/true_on_policy)
- **算法不匹配修正 (Megatron)**：[链接](https://github.com/THUDM/slime/tree/main/examples/train_infer_mismatch_helper)

如果您的目标是完全消除训推不匹配，我们推荐使用**True-On Policy**解决方案。

如果您倾向于在缓解不匹配的同时保持高性能，像 MIS 这样的**算法修正**是一个轻量级且有效的选择。

以下是可用选项的简要概述。

### True On Policy

要开启True On Policy模式，请添加以下参数：

```bash
CUSTOM_ARGS=(
    --true-on-policy-mode
)
```

### 算法缓解（Algorithmic Mitigation）

请参阅 [此链接](https://github.com/THUDM/slime/blob/main/examples/train_infer_mismatch_helper/README.md) 获取每个attribute的详细完整解释。

Slime 提供了一个全面的配置系统，允许用户灵活平衡偏差（Bias）与方差（Variance）。要开启重要性采样，必须在启动脚本中添加以下属性：

```bash
CUSTOM_ARGS=(
   --use-tis
   --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)
```

然后，您可以在 [此文件](https://github.com/THUDM/slime/blob/main/examples/train_infer_mismatch_helper/mis.yaml) 中调整详细配置。

简单来说，您可以从四个关键维度配置修正策略：

#### 计算层级（Calculation Levels）
决定如何将重要性权重从 Token 聚合到序列。
- **Token Level（Token 级）**
  - 为每个 Token 独立计算权重。
  - 特点：计算简单，但在数学上是有偏的（biased）。适用于大多数通用场景。
- **Sequence Level（序列级）**
  - 序列权重是所有 Token 权重的乘积。
  - 特点：数学上无偏，但面临极大的方差。仅建议在不匹配非常小或批次很大时使用。
- **Geometric Level（几何级）**
  - 使用所有 Token 权重的几何平均值作为序列权重。
  - 特点：一种折衷方案。它保留了序列级信息，同时避免了乘积方法的数值不稳定性，在偏差和方差之间取得平衡。对于长上下文任务，它还提供了一定的长度不变性（length-invariant）属性。

#### 拒绝采样与掩码（Rejection Sampling & Masking）
为了防止极端的重要性权重破坏训练稳定性并强制执行硬信任域，我们对权重施加约束。
- **IS Mode（重要性采样模式）**
  - `--tis-mode`：选项包括 `clip`（裁剪）或 `truncate`（截断）。这将强制权重保持在 `[lower_bound, upper_bound]` 范围内。
- **RS Mode（拒绝采样模式）**
  - `--use-rs`：RS 不限制权重，而是直接mask（丢弃）落在阈值之外的 Token 或序列。这确保了有效数据的梯度纯度，但会减少有效的训练样本量。

MIS 的工作引入了不同层级的 IS 和 RS 组合。

#### 否决机制（Veto Mechanism）
这是独立于 IS/RS 设置的底层安全网。
- **机制**：如果序列中包含任何在旧策略下概率低于否决阈值（例如 $$p < 10^{-6}$$）的 Token，则丢弃整个序列。
- **必要性**：防止“灾难性更新”。即使经过裁剪，分母中接近零概率的 Token 也可能引入数值不稳定性或破坏性梯度。

#### 自归一化（Self-Normalization）
`--tis-batch-normalize`：自归一化。在整个 Batch 范围内对重要性权重进行归一化，使其均值等于 1.0。这可以防止权重幅度导致训练步长不稳定。

## 更多关于不匹配解决的功能

- Slime 还支持来自 Deepseek V3.2 的无偏 KL 估计：[链接](https://github.com/THUDM/slime/pull/1004)
- Slime 还支持 Rollout 路由回放（Rollout Routing Replay）：[链接](https://github.com/THUDM/slime/pull/715)

任何不匹配解决工具都能在 Slime 中找到！

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
K₃ KL 是基于似然比（$$\frac{p(x)}{q(x)}$$）的 KL 散度的一个简单无偏估计量。定义如下：
$$ k_3(x) = \frac{p(x)}{q(x)} - 1 - \log \frac{p(x)}{q(x)}.$$

## 致谢

Chenyang Zhao, Yingru Li, Yueming Yuan, Changyi Yang, Chenxin Xie, Jiajun Li, Banghua Zhu, Yuzhen Zhou, Li Ji, Tom, Jiawei Xu, Hongyu Lu
