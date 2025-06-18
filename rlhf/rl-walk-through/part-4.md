# 时序差分算法

动态规划算法要求马尔可夫决策过程是已知的，即要求与智能体交互的环境是完全已知的。智能体并不需要和环境真正交互来采样数据，直接用动态规划算法就可以解出最优价值或策略。这就好比对于有监督学习任务，如果直接显式给出了数据的分布公式，那么也可以通过在期望层面上直接最小化模型的泛化误差来更新模型参数，并不需要采样任何数据点。

但这在大部分场景下并不现实，机器学习的主要方法都是在数据分布未知的情况下针对具体的数据点来对模型做出更新的。甚至对于大部分强化学习现实场景，其马尔可夫决策过程的状态转移概率都是是无法写出来的，更何谈动态规划。在这种情况下，智能体只能和环境进行交互，通过采样到的数据来学习，这类学习方法统称为无模型的强化学习（model-free reinforcement learning）。

不同于动态规划算法，无模型的强化学习算法不需要事先知道环境的奖励函数和状态转移函数，而是直接使用和环境交互的过程中采样到的数据来学习。这一部分将要讲解无模型的强化学习中的两大经典算法：Sarsa 和 Q-learning，它们都是基于时序差分（temporal difference，TD）的强化学习算法。同时，本章还会引入一组概念：在线策略学习和离线策略学习。通常来说，在线策略学习要求使用在当前策略下采样得到的样本进行学习，一旦策略被更新，当前的样本就被放弃了；而离线策略学习可以反复利用采集到的经验，能够更好地利用历史数据，并具有更小的样本复杂度（算法达到收敛结果需要在环境中采样的样本数量），这使其被更广泛地应用。

## 时序差分算法

**TD (Temporal Difference) 是一种用来估计一个策略的价值函数的方法，它结合了蒙特卡洛和动态规划算法的思想。时序差分法和蒙特卡洛的相似之处在于可以从样本数据中学习，不需要事先知道环境；和动态规划的相似之处在于根据贝尔曼方程的思想，利用后续状态的价值估计来更新当前状态的价值估计。** 回顾一下在多臂老虎机一节中，蒙特卡洛方法对价值函数的增量更新方式：

- TD 直接从 episodes of experience 中学习，不需要知道模型
- TD 使用贝尔曼方程来更新价值函数
- TD 使用逐步 bootstrapping 来学习，不用知道完整的 episode

$$V(s_t) \leftarrow V(s_t) + \alpha [G_t - V(s_t)]$$

这里我们将 $\frac{1}{N(s)}$ 替换成了 $\alpha$，表示对价值估计更新的步长。可以将$\alpha$取为一个常数，此时更新的步长是固定的；也可以将$\alpha$取为一个衰减序列（例如 $\frac{1}{t}$），此时更新步长会随着时间递增而减小。蒙特卡洛方法必须等到一个序列结束之后才能计算得到一次的回报 $G_t$，而时序差分方法只依赖当前和后续状态的价值估计值，而无需等待整个序列结束。时序差分算法的一个变种是基于贝尔曼方程的预测误差的增量更新方式：

$$V(s_t) \leftarrow V(s_t) + \alpha [R_{t+1} + \gamma V(s_{t+1}) - V(s_t)]$$

其中 $R_{t+1} + \gamma V(s_{t+1})-V(s_t)$ 是当前策略为时序差分的目标值误差（error），时序差分算法将下一步长的乘积作为状态价值的更新量。

## Sarsa 算法

用贪婪算法根据动作价值选取动作来和环境交互，再根据得到的回报和后续状态价值估计来更新当前状态动作价值。公式如下：

$$
\pi(a|s) = \begin{cases} 
\frac{\epsilon/|A| + 1 - \epsilon}{1/|A|} & \text{如果 } a = \arg\max_{a'} Q(s, a') \\
1/|A| & \text{其他动作}
\end{cases}
$$

现在，我们就可以得到一个实际的基于时序差分方法的强化学习算法。这个算法被称为 Sarsa，因为它的动作价值更新用到了当前状态s，当前动作a，获得的奖励r，下一个状态s'和下一个动作a'，将这些符号拼接后就得到了Sarsa的名称。以下是Sarsa的具体算法如下：

- 初始化 $Q(s, a)$
- for 序列e = 1 -> $E$ do:
  - 得到初始状态s
  - 用$\epsilon$-greedy 策略根据Q选择当前状态s下的动作a
    - for 时间步t = 1 -> $T$ do:
      1. 得到环境反馈的r，s'
      2. 用$\epsilon$-greedy 策略根据Q选择当前状态s'下的动作a'
      3. $Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma Q(s', a') - Q(s, a)]$
      4. $s \leftarrow s', a \leftarrow a'$
    - end for
  - end for
- end for

由此，Sarsa 算法现式地更新了策略价值函数，最后的策略如上文所述，被间接更新：

$$
\pi(a|s) = \begin{cases} 
\frac{\epsilon/|A| + 1 - \epsilon}{1/|A|} & \text{如果 } a = \arg\max_{a'} Q(s, a') \\
1/|A| & \text{其他动作}
\end{cases}
$$

蒙特卡洛方法利用当前状态之后每一步的奖励而不使用任何估计的价值函数。时序差分算法只利用一步奖励和下一步的奖励而不用等待一个序列结束，**蒙特卡洛算法的价值函数是无偏的（unbiased）的，但是具有比较大的方差，因为每一步的未来转移都可能有不确定的转移方向；时序差分算法具有较小的方差，但是由于只关注一步转移转移，而用下一个状态的价值估计来代替后续的价值估计，所以 TD 是有偏差的，其价值估计中没有用到真实的价值。** 那有没有什么方法可以结合二者的优势呢？多步时序差分的思想是用折中方式进行更新：

$$G_t = r_t + \gamma Q(s_{t+1}, a_{t+1})$$

替换成：

$$G_t = r_t + \gamma r_{t+1} + \cdots + \gamma^n Q(s_{t+n}, a_{t+n})$$

于是：

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]$$

替换成：

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma r_{t+1} + \cdots + \gamma^n Q(s_{t+n}, a_{t+n}) - Q(s_t, a_t)]$$

## Q-Learning

除了 Sarsa，还有一种非常著名的基于时序差分算法的强化学习算法——Q-learning。Q-Learning 和 Sarsa 的最大区别在于 Q-learning 的时序差分更新方式：

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [R_t + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t)]$$

Q-learning 算法的具体流程如下：

- 初始化 $Q(s, a)$
- for 序列e = 1 -> $E$ do:
  - 得到初始状态s
  - for 时间步t = 1 -> $T$ do:
    1. 用$\epsilon$-greedy 策略根据Q选择当前状态s下的动作a
    2. 得到环境反馈的r，s'
    3. $Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$
    4. $s \leftarrow s'$
  - end for
- end for

我们可以用价值迭代的思想来理解 Q-learning，即 Q-learning 是直接在估计Q*，因为动作价值函数的见尔曼最优方程是：

$$Q^*(s, a) = r(s, a) + \gamma \sum_{s' \in S} P(s'|s, a) \max_{a'} Q^*(s', a')$$

而 Sarsa 估计当前 $\epsilon$-贪婪策略的动作价值函数，需要强调的是，Q-learning 的更新并非必须使用当前贪心策略 $argmax_a Q(s, a)$来得到的动作，而是可以直接根据更新公式来更新状态动作价值，而不依赖当前策略的选择。

更新的核心点在于，我们使用的是一个$\epsilon$-贪婪策略来与环境交互，再根据得到的回报和下一步状态的动作价值函数来更新当前状态动作价值。Sarsa 是 on-policy 算法，必须使用当前$\epsilon$-贪婪策略选择动作，而 Q-learning 是 off-policy 算法，两个概念强化学习中非常重要。

1. 在 Q learning 中，behaviour 和 target policies 都会 improve；target policy 是 greedy 策略，而 behaviour policy 可以是随机策略，也可以是 $\epsilon$-greedy 策略；
2. sarsa 算法会真实地选择 $A_{t+1}$，而 Q learning 算法会选择 $argmax_a Q(s_{t+1}, a')$ 作为 $A_{t+1}$，从而前者是 on-policy 算法，后者是 off-policy 算法；

这两句话，下图解释的非常好：

 
<div align="center">
  <img src="./pics/sarsa-q-learning.png" width="50%" />
</div>