# Learning to Reason under Off-Policy Guidance —— 使用离线策略辅助在线学习

https://arxiv.org/abs/2504.14945

## 核心问题与思考

- **On-Policy vs. Off-Policy**：作者认为，传统的 **On-Policy** 训练倾向于放大模型已有的行为模式，而无法引入全新的认知能力。那么，如何让 LLM 突破其初始认知边界，获得更强的推理能力呢？
- **模仿学习**：一种可能的解决方案是通过模仿学习，例如利用 DeepSeek-R1 等强大的 LRM 生成的推理轨迹对 LLM 进行微调。
- **潜在担忧**：单纯的模仿学习可能使模型陷入表面化、僵化的推理模式，限制其进一步的学习和探索能力。

## 方法

**离线策略知识与在线策略学习的探索有效结合的 LUFFY 框架**：通过将 off-policy 推理轨迹与 on-policy rollout 结合，动态平衡模仿与探索；当模型无法独立生成正确解时，会为 off-policy rollout 分配更高的优先级；而一旦模型开始生成成功的推理轨迹， on-policy rollout 将优先。

- **Mixed-Policy GRPO**

  接下来到了我们最关心的部分：这种混合优势计算会给策略梯度算法的估计引入偏差，因为算法假设 rollouts 是由当前的策略分布生成的。在这里，论文使用 importance sampling 来校准梯度估计，并将这种方法称为 Mixed-Policy GRPO（公式3）

- **Policy Shaping**

  混合策略 GRPO 通过重要性采样成功地整合了离线策略 rollout，但引出了新问题：重要性采样加速收敛，但显著减少了探索。熵的崩溃速度远快于 On Policy RL，表明 rollout 变得越来越确定性，并且探索多样化推理轨迹的能力降低了 → 通过 regularized importance sampling，用转换函数重新加权离线策略分布梯度，增加了模型标准分布中低概率 token 的梯度权重，以增强从低概率 token 学习。

- **No Clip**

  移除了 clipping，以允许在更新不熟悉但有效的动作时具有更大的灵活性，从而更好地整合离线策略推理行为。

## 实验方法

### Zero-RL 方法

针对 Zero-RL 方法，论文考虑了以下几种方法：

1. Simple-RL ：基于规则的奖励，从 Qwen2.5-Math-7B 模型训练；
2. Oat-Zero ：基于规则的奖励，从 Qwen2.5-Math-7B 训练，并提出移除 GRPO 优势计算中的标准差以及策略损失计算中的 token 级归一化；
3. PRIME-Zero ：通过隐式过程奖励，利用策略 rollout 和结果标签进行训练。
4. OpenReasonerZero ：一种最近的 Zero-RL 方法的开源实现。

### 其他方法

除此之外，还考虑了：

1. On-Policy RL：使用 Dr.GRPO 算法，基于相同奖励和数据进行在线策略训练。
2. SFT：使用与 LUFFY 相同的提示和推理轨迹，通过监督微调训练模型。

## Ablation

- **Policy Shaping 和 NoClip** 
  - 对混合策略训练产生了显著的积极贡献 
  - 但是在没有 Offline Policy 指导的情况下（即“纯 On Policy +Shaping/NoClip”），这些功能并不能带来改进 → 强调了外部信号对于获取细致和可泛化推理技能的必要性。

- **Mixed-Policy GRPO**
  - 早期阶段：混合策略显著优于 On-Policy RL，收敛更快。
  - 后期阶段：但随着训练的进行，性能逐渐收敛至与 On-Policy RL 相似的水平 → 直接整合离线策略轨迹虽然加速了收敛，但未能阻止模型陷入局部最优。
  - 而 Policy Shaping 缓解了过早收敛，并在后期训练阶段持续扩大了性能优势。移除 Clipping 后，这种好处进一步放大，从而实现支持更激进和有效探索的参数更新

## Analysis

LUFFY 能够有选择性且真正有效地选择 Offline Policy，避免过拟合离线数据的表面特征，优于 SFT 的形式化模仿。

LUFFY 更具有真正的探索能力：低 temperature 下 SFT 能与 LUFFY 旗鼓相当，但 temperature 升高后 SFT 性能会下降，无法发现**额外**的正确路径。相比之下，LUFFY 有效地利用了外部推理轨迹，而没有牺牲模型发现其他解决方案的能力。