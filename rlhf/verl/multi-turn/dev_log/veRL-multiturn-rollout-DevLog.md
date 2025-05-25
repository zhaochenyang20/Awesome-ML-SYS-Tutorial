# veRL-multiturn-rollout-DevLog

原本的 `veRL-multiturn-rollout` 只实现了 `dr` 和 `swedev` 任务上的 multiturn，而计算 reward 的服务在智谱内网无法访问，训练效果无法得到验证。因此，希望引入 multiturn 对 `gsm8k` 任务的支持。

## 关键代码

### 1. 奖励计算函数

`veRL` 中实现了三种 `RewardManager`

```python
# verl/trainer/main_ppo.py 
# in main_task()
reward_manager_name = config.reward_model.get("reward_manager", "naive")
    if reward_manager_name == 'naive':
        from verl.workers.reward_manager import NaiveRewardManager
        reward_manager_cls = NaiveRewardManager
    elif reward_manager_name == 'prime':
        from verl.workers.reward_manager import PrimeRewardManager
        reward_manager_cls = PrimeRewardManager
    elif reward_manager_name == "swedev":
        from verl.workers.reward_manager import SWEDevRewardManager
        reward_manager_cls = SWEDevRewardManager
    else:
        raise NotImplementedError
    reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=compute_score)
```

这里希望用最简单的本地计算，故采用默认的 `naive`。`gsm8k` 任务作为 `data_source` 在 `_default_compute_score` 里分支判断，已经实现（这里是不是该改成多态）。

- **`NaiveRewardManager`**

  - 在本地计算奖励，不依赖于外部API。它主要通过 `compute_score` 函数来评估生成的文本。

  - 工作流程
    1. 从 `DataProto` 中提取 prompt 和 response。
    2. 将 prompt 和 response 拼接成完整的序列。
    3. 使用 `self.compute_score` 函数（默认 `_default_compute_score`）计算奖励。这个函数会比较生成的文本和 ground truth。
    4. 将奖励值放入 `reward_tensor` 中相应的位置。

- **`PrimeRewardManager`**

  - 并行计算来加速奖励计算过程。通过 `parallel_compute_score_async` 函数并行地评估多个 completion。

  - 工作流程
    1. 从 `DataProto` 中提取 completions, references, 和 tasks。
    2. 使用 `asyncio.run(parallel_compute_score_async(...))` 并行地计算每个 completion 的奖励。`parallel_compute_score_async` 内部使用 `ProcessPoolExecutor` 来实现并行。
    3. 将奖励值放入 `reward_tensor` 中相应的位置。

- **`SWEDevRewardManager`**

  - 目前是为 `swedev` 任务定制的（可以无痛泛化），依赖于外部 API 来计算奖励。它通过发送 API 请求到远程服务器来获取奖励分数，适合奖励计算任务复杂的 task。
  - 工作流程
    1. 从 `DataProto` 中提取 instance IDs，转成 sandbox 中的 `sids` 用于后续标识每个任务。
    2. 异步向 API 发送请求，获取每个 instance 的奖励分数。
    3. 将奖励值放入 `reward_tensor` 中相应的位置。

### 2. 根据 `task_type` 获取关键函数

这是第一个扩展点，"gsm8k" 这行是我额外加的，然后对应的实现 `gsm8k_start`，`gsm8k_obs`，`gsm8k_end` 来支持 multiturn。

```python
# verl/workers/agentic/async_rollout.py
# in AsyncRollout.gengerate_sequence()
loop_fn, start_fn, gen_fn, obs_fn, end_fn = {
            "dr": (ids_agent_loop, dr_start, gen_id, partial(dr_obs, tokenizer=tokenizer), dr_end),
            "swedev": (ids_agent_loop, swedev_start, gen_id, partial(swe_dev_obs, tokenizer=tokenizer), swe_dev_end),
            "gen_chat": (openai_chat_agent_loop, partial(openai_chat_start, url=url), gen_chat, partial(openai_chat_obs, url=url), partial(openai_chat_end, url=url)),
            "gsm8k": (ids_agent_loop, gsm8k_start, gen_id, partial(gsm8k_obs, tokenizer=tokenizer), gsm8k_end),
        }[self.task_type]
```

### 3. 生成 prompt 和数据集

第二个扩展点，为 `gsm8k` 提供获取 prompt 和处理数据集的工具方法。需要实现 `generate_gsm8k_prompt` 和 `preprocess_gsm8k_dataset`。

```python
# verl/utils/agentic_utils.py
PROMPT_GENERATOR = {
    "swedev": swedev_prompt_generator,
    "default": default_prompt_generator,
    "gsm8k": generate_gsm8k_prompt
}

PREPROCESS_DATASET = {
    "swedev": swedev_preprocess_dataset,
    "default": default_preprocess_dataset,
    "gsm8k": preprocess_gsm8k_dataset
}

# 好像没用到过
SPECIFIC_TENSOR_LIST = {
    "swedev": ["instance_id"],
    "default": [],
    "gsm8k": [] 
}
```
