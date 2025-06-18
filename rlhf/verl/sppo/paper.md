

# SPIN 原文解析

```math
\theta_{t+1} \leftarrow \arg\min_{\theta} \, \mathbb{E}_{(\mathbf{x}, \mathbf{y}, \widehat{P}(y \succ \pi_t \mid \mathbf{x})) \sim \mathcal{D}_t} \left( \log \left( \frac{\pi_\theta(\mathbf{y} \mid \mathbf{x})}{\pi_t(\mathbf{y} \mid \mathbf{x})} \right) - \eta \left( \widehat{P}(y \succ \pi_t \mid \mathbf{x}) - \frac{1}{2} \right) \right)^2
```

SPO 原文中的 loss，由于 \( \widehat{P}(y \succ \pi_t \mid \mathbf{x}) \) 计算效率偏低，可以使用 Kimi K1.5 中的 loss：

```math
L(\theta) = \mathbb{E}_{(x, y^*) \sim \mathcal{D}} \left[ 
    \mathbb{E}_{(y, z) \sim \pi_{\theta_i}} \left[ 
        \left( r(x, y, y^*) - \tau \log Z - \tau \log \frac{\pi_\theta(y, z \mid x)}{\pi_{\theta_i}(y, z \mid x)} \right)^2 
    \right] 
\right]
```

```math
\tau \log Z \approx \tau \log \frac{1}{k} \sum_{j=1}^{k} \exp\left( \frac{r(x, y_j, y^*)}{\tau} \right)
\quad \bar{r} = \text{mean}(r(x, y_1, y^*), \ldots, r(x, y_k, y^*))
```

> To approximate $\tau \log Z$, we use samples $(y_1, z_1), \ldots, (y_k, z_k) \sim \pi_{\theta_i}$:
> ```math
> \tau \log Z \approx \tau \log \frac{1}{k} \sum_{j=1}^{k} \exp(r(x, y_j, y^*) /\tau)
> ```
> We also find that using empirical mean of sampled rewards  
> ```math
> \bar{r} = \text{mean}(r(x, y_1, y^*), \ldots, r(x, y_k, y^*))
> ```
> yields effective practical results.  
> This is reasonable since $\tau \log Z$ approaches the expected reward under $\pi_{\theta_i}$ as $\tau \to \infty$.

最终我们用的公式为：

```math
\theta_{t+1} \leftarrow \arg\min_{\theta} \, \mathbb{E}_{(\mathbf{x}, \mathbf{y}, \widehat{P}(y \succ \pi_t \mid \mathbf{x})) \sim \mathcal{D}_t} \left( 
\log \left( \frac{\pi_\theta(\mathbf{y} \mid \mathbf{x})}{\pi_t(\mathbf{y} \mid \mathbf{x})} \right) 
- \eta \left( r(x, y, y^*) - \frac{1}{2} \right)
\right)^2
```

对应代码：

```python
def compute_sppo_loss(
    old_log_prob: torch.Tensor,      # (bs, seq_len)
    log_prob: torch.Tensor,          # (bs, seq_len)
    rewards: torch.Tensor,           # (bs,)
    response_mask: torch.Tensor,     # (bs, seq_len)
    eta: float = 1.0,
    loss_agg_mode: str = "seq-mean-token-sum"
):
    """
    SPPO Loss computation.
    """
    # Compute log-ratios over masked tokens
    log_prob_sum = (log_prob * response_mask).sum(dim=1)  # (bs,)
    old_log_prob_sum = (old_log_prob * response_mask).sum(dim=1)  # (bs,)
    
    log_ratios = log_prob_sum - old_log_prob_sum  # (bs,)

    scaled_rewards = eta * (rewards - 0.5)
    loss_vec = (log_ratios - scaled_rewards) ** 2  # (bs,)
    
    if loss_agg_mode == "seq-mean-token-sum":
        loss = loss_vec.mean()
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_lengths = response_mask.sum(dim=1)  # (bs,)
        token_mean_loss = loss_vec / seq_lengths.clamp(min=1)
        loss = token_mean_loss.mean()
    elif loss_agg_mode == "token-mean":
        sample_mask = response_mask.any(dim=1).float()  # (bs,)
        loss = verl_F.masked_mean(loss_vec, sample_mask)       
    else:
        raise ValueError(f"Unsupported loss_agg_mode: {loss_agg_mode}")

    return loss, log_ratios, scaled_rewards
```
