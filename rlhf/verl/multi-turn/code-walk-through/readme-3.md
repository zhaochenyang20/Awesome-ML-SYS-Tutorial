## Make Experience

经过了漫长的战线，我们终于分析完了 rollout 部分的逻辑。我们接着分析 make experience 部分的逻辑。

<details>
<summary>Make Experience 源码</summary>

```python
with marked_timer("reward", timing_raw, color="yellow"):
    # compute reward model score
    if self.use_rm:
        reward_tensor = self.rm_wg.compute_rm_score(batch)
        batch = batch.union(reward_tensor)

    if self.config.reward_model.launch_reward_fn_async:
        future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
    else:
        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

# recompute old_log_probs
with marked_timer("old_log_prob", timing_raw, color="blue"):
    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
    entropys = old_log_prob.batch["entropys"]
    response_masks = batch.batch["response_mask"]
    loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
    entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
    old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
    metrics.update(old_log_prob_metrics)
    old_log_prob.batch.pop("entropys")
    batch = batch.union(old_log_prob)

    if "rollout_log_probs" in batch.batch.keys():
        # TODO: we may want to add diff of probs too.
        rollout_old_log_probs = batch.batch["rollout_log_probs"]
        actor_old_log_probs = batch.batch["old_log_probs"]
        attention_mask = batch.batch["attention_mask"]
        responses = batch.batch["responses"]
        response_length = responses.size(1)
        response_mask = attention_mask[:, -response_length:]

        rollout_probs = torch.exp(rollout_old_log_probs)
        actor_probs = torch.exp(actor_old_log_probs)
        rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
        rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
        rollout_probs_diff_max = torch.max(rollout_probs_diff)
        rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
        rollout_probs_diff_std = torch.std(rollout_probs_diff)
        metrics.update(
            {
                "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
            }
        )

if self.use_reference_policy:
    # compute reference log_prob
    with marked_timer("ref", timing_raw, color="olive"):
        if not self.ref_in_actor:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
        else:
            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
        batch = batch.union(ref_log_prob)

# compute values
if self.use_critic:
    with marked_timer("values", timing_raw, color="cyan"):
        values = self.critic_wg.compute_values(batch)
        batch = batch.union(values)

with marked_timer("adv", timing_raw, color="brown"):
    # we combine with rule-based rm
    reward_extra_infos_dict: dict[str, list]
    if self.config.reward_model.launch_reward_fn_async:
        reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
    batch.batch["token_level_scores"] = reward_tensor

    if reward_extra_infos_dict:
        batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

    # compute rewards. apply_kl_penalty if available
    if self.config.algorithm.use_kl_in_reward:
        batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
        metrics.update(kl_metrics)
    else:
        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

    # compute advantages, executed on the driver process

    norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

    batch = compute_advantage(
        batch,
        adv_estimator=self.config.algorithm.adv_estimator,
        gamma=self.config.algorithm.gamma,
        lam=self.config.algorithm.lam,
        num_repeat=self.config.actor_rollout_ref.rollout.n,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
        config=self.config.algorithm,
    )
```

</details>

这一部分的操作还是很好读懂了，非常 standard：

1. 通过 `self.reward_fn` 或 `self.rm_wg.compute_rm_score` 计算 trajectory 的 reward。verl 支持各式各样的 reward，不单单是 reward model。
2. 重算 behaviour policy 的 log probabilities: 使用 `self.actor_rollout_wg.compute_log_prob(batch)` 来重算 log probs。这里原因在 part 1 讲述 [importance sampling](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/code-walk-through/readme.md#actorrolloutrefworker__init__) 的部分也阐述过了。这里非常让我想吐槽的是，verl 里面 `old_log_prob` 就是用 training engine 重算的 behaviour policy 的 log probs，用 old 来描述让我比较费解。
3. 计算 reference policy 的 log probabilities: 如果使用了 reference policy，则计算 reference policy 的 log probs，用于 KL divergence 约束。
4. 计算 Critic 的 value: 如果使用了 Critic model，则通过 `self.critic_wg.compute_values(batch)` 预测当前 state 的 value。
5. 估算 Advantage: 调用 `compute_advantage` 函数，根据配置的advantage estimator、折扣因子 (gamma)、GALA 因子 (lam) 等参数，利用 reward 和 value 估计计算优势函数。

## Training

非常标准：

```python
# update critic
if self.use_critic:
    with marked_timer("update_critic", timing_raw, color="pink"):
        critic_output = self.critic_wg.update_critic(batch)
    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
    metrics.update(critic_output_metrics)

# implement critic warmup
if self.config.trainer.critic_warmup <= self.global_steps:
    # update actor
    with marked_timer("update_actor", timing_raw, color="red"):
        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
        actor_output = self.actor_rollout_wg.update_actor(batch)
    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
    metrics.update(actor_output_metrics)
