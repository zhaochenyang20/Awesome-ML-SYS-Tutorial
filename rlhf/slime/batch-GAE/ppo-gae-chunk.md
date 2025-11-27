# PPO 中 GAE 的分 chunk 并行计算（基于 slime 的实现）

> 对应知乎原文：《PPO 中 GAE 的分 chunk 并行计算》（https://zhuanlan.zhihu.com/p/1975237289425798560）
> 对应代码：[`THUDM/slime` PR #850 — Chunk-Scan GAE](https://github.com/THUDM/slime/pull/850)

## 1. **TL;DR**

这篇文章中，作者围绕 slime 框架里的 PPO + GAE 做了一次性能改造：

**背景**：在 agentic RL 场景里，在序列超长的时候，slime 原本的 GAE 计算是按 sample 分批串行从尾部到头扫一遍。而这将直接变成训练瓶颈。

**做的事**：

1. slime 先把传统的串行后向递推计算 GAE 的方式，修改为先将 GAE 按时间分为多个 Chunk，之后按时间逆序，用当前遍历的时间点的 Chunk 和上一个 Chunk 计算出的 `lastgaelam` 逐步推出最后的 GAE。
2. 而这篇文章则借鉴了 *linear attention* [**@sonta**](https://www.zhihu.com/people/buhezuobugaoxing) 的思路，通过分块前缀扫描，将多个计算完成的局部 GAE 合并，计算出最终 GAE，每个 Chunk 的计算之间不再有依赖。

**效果**：

1. 在 slime 中，GAE 计算时间得到 **100×–300×** 的加速；
2. 并行度取决于 `chunk_size`，在不 OOM 的前提下，`chunk_size` 越大加速越明显。

## 2. 技术背景：为什么要搞 GAE 的 Chunk-Scan？

### 2.1 为什么现在 GAE 会变成瓶颈？

在 RLHF / Agentic RL 里，PPO 仍是一个非常常用、表现稳定的算法。我们需要在每个 token 上计算 advantage，最常见的就是 GAE（Generalized Advantage Estimation）。而 GAE 的标准写法是一个从后往前的递推公式，对序列长度 T 来说，是 O(T) 串行依赖。

在 slime 中，GAE 的算法实现如下：

```python
lastgaelam = torch.zeros(B, device=device, dtype=dtype)
adv_rev = []

for t in reversed(range(max_len)):
    next_value = full_values[:, t + 1] if t < max_len - 1 else 0.0
    delta = full_rewards[:, t] + gamma * next_value - full_values[:, t]
    lastgaelam = delta + gamma * lambd * lastgaelam
    adv_rev.append(lastgaelam)

full_advantages = torch.stack(adv_rev[::-1], dim=1)  # [B, max_len]
```

slime 在一开始的实现里，追求的是“支持变长序列”，优点是在模型计算时，不需要 padding 到所有序列的 max_len， 避免浪费无效的计算，所以在计算 GAE 时，是一个序列一个序列计算而不是拼成 batch 计算，造成了性能瓶颈，我们很快改成常见的 “padding 到 max_len，再按 batch 计算 GAE” 的写法。但是可惜的是，这并不足以达到可能的最佳性能，它在时间维度仍是串行，而这导致了在长序列场景下依然很吃力。

在此基础上，作者结合 [**@sonta**](https://www.zhihu.com/people/buhezuobugaoxing) 在讲 *linear attention* 时提到的 “分 chunk 并行 + chunk 间轻量递推” 的思路，尝试把 GAE 也改造成一个 chunk 级别可并行的“前缀扫描（scan）”问题。

### 2.2 完全矩阵化计算下的爆显存问题

想对 GAE 并行，其实有一个非常优雅的方案，直接写成矩阵乘法：

- 把 GAE 写成 $A_t = \sum_{k=t}^{T-1} w^{k-t} \delta_k$（其中 $w = \gamma \lambda$）

- 构造一个 T×T 的上三角权重矩阵 W，然后做 $A = \delta W^\top$

这是完全可以并行的，但是这直接导致了时间复杂度和空间复杂度都是 O(T²)。一旦 T 达到了 64K、128K 的级别，会直接 OOM。

> torchrl 中有[使用 conv1d 将时间复杂度降到 O(T)](https://github.com/pytorch/rl/blob/8570c25a745da54ca647b8a70231112f063d1421/torchrl/objectives/value/utils.py#L13) 的方案，但是空间复杂度依然是 O(T²)，因此还是会有上面这个 OOM 问题。

因此，我们希望找到一个能同时兼顾并行度，又能保证显存可控的 GAE 计算方式。

## 3. 架构设计：从串行 GAE 到 Chunk-Scan GAE

### 3.1 GAE

在开始前，我们先回顾一下标准的 GAE：

我们记 delta 为 
$$
\delta_t = r_t + \gamma V_{t+1} - V_t
$$
则 GAE 的 advantage 为
$$
A_t = \sum_{k=t}^{T-1} (\gamma \lambda)^{k-t} \delta_k
$$
也可以写成后向递推的形式：
$$
A_t = \delta_t + \gamma \lambda A_{t+1}, \quad t = T-1, T-2, \dots, 0
$$

### 3.2 方案一：串行解法

slime 目前的版本其实已经给出了答案：

```python
lastgaelam = torch.zeros(B, device=device, dtype=dtype)
adv_rev = []

for t in reversed(range(max_len)):
    next_value = full_values[:, t + 1] if t < max_len - 1 else 0.0
    delta = full_rewards[:, t] + gamma * next_value - full_values[:, t]
    lastgaelam = delta + gamma * lambd * lastgaelam
    adv_rev.append(lastgaelam)

full_advantages = torch.stack(adv_rev[::-1], dim=1)  # [B, max_len]
```

- 优点：实现简单，数值稳定；

- 缺点：这个版本在时间维度完全串行，长序列下性能不行。

### 3.3 方案二：纯矩阵解法

利用前向展开式：
$$
A_t = \sum_{k=t}^{T-1} w^{k-t} \delta_k,\quad w = \gamma \lambda
$$
我们可以构造一个 T×T 的权重矩阵 W：
$$
W_{t,k} =
\begin{cases}
w^{k-t}, & k \ge t \\
0,       & k < t
\end{cases}
$$
于是有：
$$
A = \delta W^\top
$$

- 优点：矩阵乘法可以在 GPU 上高度并行；
- 缺点：非常容易 OOM

### 3.4 方案三：Chunk-Scan

我们可以把整条序列拆成若干个长度为 C 的 chunk：

```
第一个 chunk：0 - C-1
第二个 chunk：C - 2C-1
...
第 c 个 chunk：cC - (cC + L_c - 1)
```

在反向序列上定义 GAE 递推：
$$
S_i = \widetilde{\delta}_i + w S_{i-1}, \quad w = \gamma \lambda, \quad S_{-1} = 0
$$
对于第 c 个 chunk，定义“跨 chunk 状态”：
$$
s_{\text{prev}} = S_{cC - 1}
$$
c = 0 时，有 $s_{\text{prev}} = S_{-1} = 0$；

现在考虑 chunk c 内部的第 t 个元素（局部索引 t = 0..L_c-1）：

- 全局索引 $i = cC + t$；
- 展开递推关系得到：

$$
\begin{aligned}
S_{cC + t}
&= \widetilde{\delta}_{cC + t}
 + w \widetilde{\delta}_{cC + t - 1}
 + \cdots
 + w^t \widetilde{\delta}_{cC}
 + w^{t+1} S_{cC - 1}
\end{aligned}
$$

把“当前 chunk 内”的部分单独拿出来：
$$
s^{(c)}_t = \sum_{k=0}^{t} w^{t-k} \widetilde{\delta}_{cC + k}
$$
于是最终公式可以写成：
$$
\boxed{
S_{cC + t} = s^{(c)}_t + w^{t+1} \, s_{\text{prev}}, \quad t = 0, \dots, L_c - 1
}
$$
这意味着：

- 局部部分 $s^{(c)}_t$ 可以在 chunk 内用矩阵/conv 并行算；
- 跨 chunk 只需要维护一个标量状态 `s_prev`，串行递推即可。

时间复杂度：O(T·C)
空间复杂度：O(T + C²)

上面是非常严谨的公式推导过程，其实简单来说，Chunk-scan 的核心想法就是：

1. 把长序列切成若干小 chunk
2. 让 GPU 并行计算每个 chunk 内的递推，上面的公式则是得出了可以并行计算的部分 $s^{(c)}_t$ 
3. 再把这些 chunk 的结果组合起来

### 3.5 [`THUDM/slime` PR #850 — Chunk-Scan GAE](https://github.com/THUDM/slime/pull/850) 实现的伪代码

以下是一个展示如何把 Chunk-Scan GAE 写成批量计算函数的伪代码，代码原型来自：[`THUDM/slime` PR #850 — Chunk-Scan GAE](https://github.com/THUDM/slime/pull/850) 

```pseudocode
function chunked_gae(rewards, values, gamma, lambda, chunk_size):

    w = gamma * lambda

    # 1. 计算每一步的 δ_t
    deltas = compute_deltas(rewards, values)   # δ_t = r_t + γV_{t+1} - V_t

    # 2. 反向时间顺序（从后往前的递推 -> 在反向序列上从左往右）
    deltas_rev = reverse_time(deltas)

    # 3. pad 到 chunk_size 的整数倍，并拆成若干个 chunks
    deltas_chunks = split_into_chunks(deltas_rev, chunk_size)

    # 4. 为“每个 chunk 内部”的扫描预计算一个小核：
    #    给定一段 Δ[0..C-1]，算出 s_local[t] = Σ_{k≤t} w^(t-k) * Δ[k]
    kernel = build_chunk_kernel(chunk_size, w)         # C×C 的上三角矩阵
    pow_vec = build_power_vector(chunk_size, w)        # [w^1, w^2, ..., w^C]

    # 5. 所有 chunk 内部并行做局部 scan
    #    local_scan[c, t] = s_local^(c)[t]
    local_scans = []
    for each chunk in deltas_chunks in parallel:
        s_local = chunk @ kernel         # 这里用任意并行实现都行
        local_scans.append(s_local)

    # 6. 在 chunk 之间串行传播“前缀状态” s_prev
    s_prev = 0
    full_scan_rev = empty_like(deltas_rev)

    for c from 0 to num_chunks-1:
        s_local = local_scans[c]         # 当前 chunk 内部的结果，长度 L_c

        # 注入跨 chunk 的状态：
        # S_global[t] = s_local[t] + w^(t+1) * s_prev
        S_global = s_local + s_prev * pow_vec[0:L_c]

        write_into(full_scan_rev, chunk_index=c, values=S_global)

        # 下一个 chunk 的起点状态 = 当前 chunk 最后一个位置
        s_prev = S_global[L_c - 1]

    # 7. 去掉 padding，反向回正向时间
    advantages = reverse_time(remove_padding(full_scan_rev))

    # 8. returns 一般就是 V_t + A_t
    returns = values + advantages

    return advantages, returns

```

## 4. 实现效果

根据原文的实验结果，实现效果非常可观：

| No chunk        | chunk size = 64 | chunk size = 128 | chunk size = 256 |                    |
| --------------- | --------------- | ---------------- | ---------------- | ------------------ |
| B=256, T=131072 | 5.935994s       | 0.070122s        | 0.034059s        | 0.018390s ( x317 ) |
| B=128, T=65536  | 2.902570s       | 0.232986s        | 0.017645s        | 0.009134s          |

可以看到：

1. 在 T=131072、`chunk_size=256` 时，加速比约 **317×**；
2. 在 T=65536、`chunk_size=256` 时，加速比也非常可观；

只要有足够显存能用来提升 chunk size，并行度就能大幅度增加，GAE 的计算时间也能相当可观地被缩减。

## 5. 具体使用方法：在 slime 里怎么用 Chunk-Scan GAE？

Chunk-Scan 已被作为默认的训练行为，因此对于用户的安装或迁移，仅需更新镜像即可。

### 5.1 安装

1. 拉取当前最新版本 docker 镜像，确保其包含 [`THUDM/slime` PR #850 — Chunk-Scan GAE](https://github.com/THUDM/slime/pull/850) 的改动。截止至 11/24，官方还未更新 docker 镜像。
2. 根据官方指引部署服务：https://github.com/THUDM/slime/blob/main/docs/en/get_started/quick_start.md
3. 使用默认参数即可使用 Chunk-Scan 训练 PPO。

### 5.2 迁移

1. 升级至当前最新版本 docker 镜像，确保其包含 [`THUDM/slime` PR #850 — Chunk-Scan GAE](https://github.com/THUDM/slime/pull/850) 的改动。截止至 11/24，官方还未更新 docker 镜像。
2. 恢复训练即可

## 6. 未来计划

1. 更系统的 benchmark & 可视化工具

   提供一键脚本，方便用户评估自己任务是否值得开启 Chunk-Scan。

2. 更全面地测试整体框架的性能，更细粒度地测量各个部分的耗时情况，找出类似的潜在的问题。

3. 检查其他部分的代码是否也存在可以通过修改算法提升并发度的情况，如果有，需要探索优化的可能性。

## 7. 工程附录：踩过的坑 & 学到的东西

1. GAE 变成瓶颈这件事本身，就是一个“用实验结果纠正工程直觉”的例子。在实验数据真正跑出之前，很难想到 GAE 计算会成为 PPO 流水线的瓶颈。因此对于一个成熟的框架来说，应该在测试性能时，把性能测试的粒度划分得足够细，从而发现一些设计之初可能会忽视的问题。
