# DAPO Dynamic Filtering 实现与 Batch Size 解析


`dynamic filtering` 相关逻辑已经在代码中得到了很好的实现，我们现在想要探索，能否通过将prompt补齐到更小的batch size，实现更高的并行度。比如补齐到 `mini_batch_size`。这需要我们对补齐后的batch的后续处理做更全面的分析。而且，更多的batch意味着更多的check-> back fill，是否会对性能造成影响，也需要考虑。


## 1 Filter Groups 配置入口

`FilterGroupsConfig` 定义了三项参数：是否启用、过滤指标、以及动态补采上限。

```46:67:verl/trainer/config/algorithm.py
@dataclass(frozen=True)
class FilterGroupsConfig(BaseConfig):
    """Configuration for filter groups (used in DAPO and Entropy)."""

    enable: bool = False                # 开关
    metric: Optional[str] = None        # acc / score / seq_reward / seq_final_reward
    max_num_gen_batches: int = 0        # ≤0 代表无上限
```




## 2 Batch Size 相关参数

| 字段 | 作用 | 在`test_dapo_7b.sh`中的设置 |
|------|------|----------|
| `data.train_batch_size` | 过滤后、一次参数更新使用的 **Prompt 数** | **512** |
| `data.gen_batch_size`   | 每轮初始生成 Prompt 数 | **1536** |
| `rollout.n`             | 每个 Prompt 采样 Response 数 | **16** |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | 实际上也是单个GPU的batch size | **32** |
| `ppo_micro_batch_size_per_gpu` | 单个GPU的barch_size TODO：与 mini size关系 | None |


## 3 核心实现（`dapo_ray_trainer.py`）

以下代码⽚段展示了一次迭代内的 **动态过滤循环**：

```178:230:recipe/dapo/dapo_ray_trainer.py
else:  # NOTE: 过滤后若 Prompt 数不足 train_batch_size，将跳到下一轮生成
        metric_name = self.config.algorithm.filter_groups.metric
        # 准备用于多样性检查的序列级指标
        if metric_name == "seq_final_reward":
            # 为方便计算 std，将 tensor 转为 numpy
            new_batch.non_tensor_batch["seq_final_reward"] = (
                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
            )
        elif metric_name == "seq_reward":
            new_batch.non_tensor_batch["seq_reward"] = (
                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
            )

        # 按 Prompt UID 收集指标值
        prompt_uid2metric_vals = defaultdict(list)
        for uid, metric_val in zip(
            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name]
        ):
            prompt_uid2metric_vals[uid].append(metric_val)

        # 计算组内标准差并确定保留的 Prompt
        prompt_uid2metric_std = {}
        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

        kept_prompt_uids = [
            uid
            for uid, std in prompt_uid2metric_std.items()
            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1  # 若多样/单采样则保留
        ]
        num_prompt_in_batch += len(kept_prompt_uids)

        # 根据保留的 UID 映射到轨迹索引
        kept_traj_idxs = []
        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
            if traj_from_prompt_uid in kept_prompt_uids:
                kept_traj_idxs.append(idx)

        # 仅保留筛选后的轨迹
        new_batch = new_batch[kept_traj_idxs]
        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

        # 判断是否继续生成或对齐批次大小
        prompt_bsz = self.config.data.train_batch_size
        if num_prompt_in_batch < prompt_bsz:
            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                print(f"{num_gen_batches=}. Keep generating...")
                progress_bar.update(1)
                continue
            else:
                raise ValueError(
                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                )
        else:
            # 对齐到 train_batch_size × rollout.n 的轨迹数
            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
            batch = batch[:traj_bsz]
```

流程要点：
1. 方差计算方式由`metric_value`决定。
2. 若组内指标标准差为 0 且组内有大于一个prompt，整组 Prompt 被丢弃。 
3. 丢弃后有效 Prompt 不足 `train_batch_size`，继续采集直至满足数量或超出 `max_num_gen_batches`。
4. 一旦达到目标 Prompt 数，仍需裁剪至 `train_batch_size × rollout.n` 的整数倍，以便后续对齐。





## 4 mini-batch 相关逻辑



在 `RayPPOTrainer._validate_config`中，有对batch size的校验逻辑：

```430:462:verl/trainer/ppo/ray_trainer.py
real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
assert real_train_batch_size % minimal_bsz == 0  # 可被 DP 均分

# 没开启 dynamic_bsz 时的 Actor 检查
assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
    assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
    assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sequence_parallel_size >= n_gpus
```

TO BE CONTINUE...
