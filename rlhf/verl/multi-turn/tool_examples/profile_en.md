# Systemic Profiling of Time Consumption in verl Multi-Turn Training

Multi-turn training systems are inherently complex and resource-intensive. While this document won't directly propose solutions to optimize multi-turn training, the first step in problem-solving is always to identify the problem at the finest granularity. Simply observing an anomaly in a certain step's rollout might be far from the root cause. This document aims to find the true issues at a more granular level.

We are sharing a set of systematic analysis methods here; WandB can only show the time consumed by each training step, but our analysis granularity goes far beyond that. Based on our analysis method, we can specifically observe the **time consumed by each event in each turn of each request of each rollout DP worker in the rollout phase of each step.**

Acknowledgements: Zhuoran Yinn (CMU), Changyi Yang (CMU), Chengxi Li (CMU), Huapeng Zhou (UW), Hongyu Lu (TikTok), Chenyang Zeng (Amazon)

Additionally, verl is testing features like agent loop on multi-turn, which we expect to provide stable acceleration for multi-turn training. This content will be shared later.

## Profiling Principle

Intuitively, we've added no more than 50 lines of profiling code to [sglang\_rollout.py](https://github.com/PrinsYin/verl/blob/multiturn_profile_log/verl/workers/rollout/sglang_rollout/sglang_rollout.py) to record timestamps at the end of each event. When actually profiling, please add similar logic to your workflow and run your training. After training, use the scripts in [profile-multiturn-scripts](https://github.com/PrinsYin/verl/tree/multiturn_profile_log/profile-multiturn-scripts) for analysis.

Specifically, Ray's logs are automatically compressed and optimized, which speeds up Ray's logging. However, different workers, requests, and turns are mixed together, making it difficult to find the truly slow workers and requests. Our method is simple and effective: by forcing synchronization, all requests for each worker in each step are written to `{step_id}/{worker_id}.jsonl`. Then, our [visualization script](https://github.com/PrinsYin/verl/tree/multiturn_profile_log/profile-multiturn-scripts) reads the relevant data and visualizes it.

## Specific Usage

In your `sglang_rollout.py`, define [`SGLangLogManager`](https://github.com/PrinsYin/verl/blob/f1c6ee60ae701789875b00616e45bd0ae5cb171c/verl/workers/rollout/sglang_rollout/sglang_rollout.py#L93). Its usage is very simple.

<details>
<summary>SGLangLogManager</summary>

```python
# logging tool for sglang multi-turn rollout
class SGLangLogManager:
    def __init__(self):
        self.file_handles = {}
        atexit.register(self.close_all)
    
    def get_handle(self, log_path):
        if log_path not in self.file_handles:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self.file_handles[log_path] = open(log_path, 'a', buffering=1)
        return self.file_handles[log_path]
    
    def log(self, log_path, event, duration=None, extra=None, workid=None, step=None,**extra_keys):
        handle = self.get_handle(log_path)
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
        }
        if duration is not None:
            log_entry["duration_sec"] = duration
        if extra is not None:
            log_entry["extra"] = extra
        if workid is not None:
            log_entry["workid"] = workid
        if step is not None:
            log_entry["step"] = step
        if extra_keys is not None:
            for key in extra_keys:
                log_entry[key] = extra_keys[key]
        ordered_keys = ["timestamp", "event", "duration_sec"] + [k for k in log_entry.keys() if k not in ("timestamp", "event", "duration_sec")]
        ordered_entry = {k: log_entry[k] for k in ordered_keys if k in log_entry}
        handle.write(json.dumps(ordered_entry) + '\n')
        handle.flush()
    
    def close_all(self):
        for handle in self.file_handles.values():
            handle.close()
```

</details>

Subsequently, in the `SGLangRollout` class, define [`log_manager`](https://github.com/PrinsYin/verl/blob/f1c6ee60ae701789875b00616e45bd0ae5cb171c/verl/workers/rollout/sglang_rollout/sglang_rollout.py#L330), [`log_dir`](https://github.com/PrinsYin/verl/blob/f1c6ee60ae701789875b00616e45bd0ae5cb171c/verl/workers/rollout/sglang_rollout/sglang_rollout.py#L331), and [`step`](https://github.com/PrinsYin/verl/blob/f1c6ee60ae701789875b00616e45bd0ae5cb171c/verl/workers/rollout/sglang_rollout/sglang_rollout.py#L326), and initialize them:

<details>
<summary>SGLangRollout</summary>

```python
class SGLangRollout(BaseRollout):
    def __init__(
        self,
        actor_module: str,
        config: DictConfig,
        processing_class: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin],
        model_hf_config,
        port=None,
        trust_remote_code: bool = False,
        device_mesh: DeviceMesh | None = None,
        **kwargs,
    ):
        """Synchronized SGLang rollout engine.

        Args:
            actor_module: Huggingface model name or path to the model. The
                model should be supported by SGLang.
            config: A DictConfig object containing SGLang-specific operational
                parameters and rollout settings.
                Refer to https://docs.sglang.ai/backend/server_arguments.html
            processing_class: The tokenizer or processor instance compatible with the actor_module.
            model_hf_config: The Hugging Face model's configuration (e.g.,
                `transformers.PretrainedConfig`). It provides architectural
                details and hyperparameters like `max_position_embeddings`,
                used by SGLang for correct model initialization. This is
                the model's inherent design, not SGLang's runtime behavior.
            port: Optional port for multi-node initialization when nnodes > 1.
            trust_remote_code: Whether or not to allow for custom models
                defined on the Hub in their own modeling files.
            device_mesh: Optional `DeviceMesh` object for distributed setup.
            **kwargs: Additional keyword arguments, primarily `train_tp` for
                Megatron Backend integration to initialize hybrid engine
                process groups.
        """
        super().__init__()
        self.step = 0
        self.config = config
        self._device_mesh_cpu = device_mesh
        os.environ.setdefault("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")
        # Added log_manager
        self.log_manager = SGLangLogManager()
        # Added log_dir. If your experiment does not set EXPERIMENT_NAME, it defaults to multiturn_log_dir.
        self.log_dir = "logs/"+os.getenv("EXPERIMENT_NAME", "multiturn_log_dir")
```

</details>

Add [`self.step += 1`](https://github.com/PrinsYin/verl/blob/f1c6ee60ae701789875b00616e45bd0ae5cb171c/verl/workers/rollout/sglang_rollout/sglang_rollout.py#L619) in the `generate_sequences` function.

Then, call `self.log_manager.log` before and after the specific event you want to profile. For example:

<details>
<summary>Timing for engine_async_generate</summary>

```python
torch.cuda.synchronize()
generate_start_time = time.time()

### Event start
if self._tp_rank == 0:
    loop = asyncio.get_event_loop()
    output = loop.run_until_complete(
        self._engine.async_generate(
            prompt=None,  # because we have already convert it to prompt token id
            sampling_params=request_sampling_params,
            return_logprob=True,
            input_ids=idx_list,
            image_data=image_list,
        )
    )
    torch.cuda.synchronize()
    generate_end_time = time.time()
    log_path = os.path.join(
        self.log_dir,
        f"step_{self.step}",
        f"worker_{self._rank}.jsonl"
    )

    ### Event end

    self.log_manager.log(
        log_path,
        event="engine_async_generate",
        duration=generate_end_time - generate_start_time,
        workid=self._rank,
        step=self.step
    )
```

</details>

Note that when using `time.time()` for timing, `torch.cuda.synchronize()` is required to ensure synchronization between GPU and CPU instructions. The code above also shows the definition of the specific log path:

```python
log_path = os.path.join(
    self.log_dir,
    f"step_{self.step}",
    f"worker_{self._rank}.jsonl"
)
```

In the example [sglang\_rollout.py](https://github.com/PrinsYin/verl/blob/f1c6ee60ae701789875b00616e45bd0ae5cb171c/verl/workers/rollout/sglang_rollout/sglang_rollout.py), we called `self.log_manager.log` 36 times for extremely fine-grained profiling. You are welcome to decide which events you want to profile, discover the events that genuinely slow down multi-turn training, and optimize them specifically.

## Existing Conclusions

Next, we'll share the visualization results we obtained from profiling [GSM8K multi-turn training](https://github.com/PrinsYin/verl/blob/multiturn_profile_log/profile-multiturn-scripts/README.md#a-quick-reproduction-of-the-profile). We believe that readers can conduct more thorough analyses using our tools.

### Long-Tail Effect

<div style="display: flex; align-items: center;">
  <img src="https://github.com/PrinsYin/verl/blob/multiturn_profile_log/profile-multiturn-scripts/script-examples/cdf_per_step.png?raw=true" style="width: 40%;"/>
  <img src="./pics/wandb.png" style="width: 40%;"/>
</div>

1.  We first set the `max response length` for rollout to 1000 and created a CDF graph of the relative times of requests for all DP workers in a specific step (top right image). We observed that the long-tail effect of rollout is very significant; in the rollout process of a step's requests, 80% of requests complete in the first 40% of the time, with the rest solving the remaining long-tail problems.
2.  We then combined this with WandB's visualization (red line in the top left image) and found that during the stable increase of reward, the mean response length does not change significantly, or even decreases, remaining around 500. However, in any given step, we found that the maximum response length of the model directly reached our set `max rollout length`.

Based on the observation from the top right image, we initially planned to implement an oversampling strategy, for example, oversampling 20% of requests in each round. After a specified number of requests were collected into the data buffer, the oversampled and unfinished rollout data would be directly discarded. This differs from partial rollout, where these oversampled but unfinished requests would be saved and continued in the next step based on the previous step.

However, after further observing the red curve in the left image, we chose a more direct solution. Instead of oversampling, we directly adjusted the `max response length` downwards. For example, we tried setting the `max response length` to 600, expecting a significant optimization of the rollout time per step, with the reward not being significantly affected. **In fact, we believe that the requests with very long tails tend to have lower rewards, as the response length sharply increases after the reward collapses.**

Fortunately, we achieved considerable results, as shown by the green line in the top left image, with equally good convergence. Furthermore, when we observed the throughput and the time consumed per step, we found that a `max length` of 600 was consistently faster than 1000.

<div align="center">
<img src="./pics/wandb_600.png" style="width: 40%;"/>
</div>

1.  **Response length min** was not affected.
2.  **Response length mean** initially decreased slightly (because the outlier response lengths dropped from 1000 to 600); as time progressed, the response length mean remained consistently lower, but the reward was equally good, and even consistently better after 80 steps.
3.  **Response length max** was not affected, always reaching the highest length. This is also the core of our strategy—we believe that long-tailed requests severely slow down rollout speed and do not contribute positively to the reward. Directly reducing `max response length` can increase rollout speed while maintaining the reward.
4.  Clearly, **time per step** consistently decreased, and the relative advantage became even more significant after the sharp increase in response length.
5.  In addition, **perf/throughput** was consistently higher.

Overall, we believe that a direct, large-scale reduction of `max response length` is a very effective and simple solution. In fact, we further verified that 500, 600, 700, and 800 still converged very well. More surprisingly, we directly reduced the `max response length` to 400 (note that when `max length` was 1000, the mean response length at step 0 already exceeded 420), and surprisingly, the final convergence effect was still equally good. On one hand, GSM8K itself is very simple and doesn't require that many tokens. On the other hand, using 400 as the max would obviously affect the reward at step 0, dropping it by 0.1. But it still converged to 0.95 in the end, and the mean response length never reached 400 in between. However, after reducing from 600 to 400, there was no significant improvement in time per step.

### Analyzing a specific slow request in an abnormally slow Rollout step

CDF is just the basic use of our analysis tool. We can further analyze the time consumption of each event for each worker. Based on the case where max length was 600, in 7 repeated experiments, we found that step 67 of one experiment was a severe peak, spending significantly more time on rollout.

Further examining each rollout worker in step 67, we found:

<div style="display: flex; align-items: center;">
<img src="https://github.com/PrinsYin/verl/blob/multiturn_profile_log/profile-multiturn-scripts/script-examples/worker_0.png?raw=true" style="width: 40%;"/>
<img src="https://github.com/PrinsYin/verl/blob/multiturn_profile_log/profile-multiturn-scripts/script-examples/worker_1.png?raw=true" style="width: 40%;"/>
</div>

Worker 0's rollout took a very long time, and the other 7 workers were barraged here, waiting for worker 0 to finish. Looking further at the requests in worker 0, we noticed that worker 0 had a total of 512 requests (note that we enabled the [repetitive sampling](https://github.com/volcengine/verl/pull/2258) feature):

```python
data.train_batch_size=256
actor_rollout_ref.rollout.n=16
```

We further obtained the CDF graphs for worker 0 and other workers:

<div style="display: flex; align-items: center;">
<img src="https://github.com/PrinsYin/verl/blob/multiturn_profile_log/profile-multiturn-scripts/script-examples/request_analysis_and_cdf1.png?raw=true" style="width: 40%;"/>
<img src="https://github.com/PrinsYin/verl/blob/multiturn_profile_log/profile-multiturn-scripts/script-examples/request_analysis_and_cdf2.png?raw=true" style="width: 40%;"/>
</div>

We found that the request CDF for worker 0 was very strange: about 400 requests returned in the first 25 seconds, followed by 150 seconds where no requests returned; then the remaining requests returned. Additionally, the slopes at 175s and 15s were consistent, which suggests that the GPU was idle during the middle blank period. We observed this peak once and then conducted 6 repeated experiments, but the same situation never occurred again.

We then looked into those 100 abnormal requests and found two typical ones:

<div style="display: flex; align-items: center;">
<img src="https://github.com/PrinsYin/verl/blob/multiturn_profile_log/profile-multiturn-scripts/script-examples/outlier_req_1.png?raw=true" style="width: 40%;"/>
<img src="https://github.com/PrinsYin/verl/blob/multiturn_profile_log/profile-multiturn-scripts/script-examples/outlier_req_2.png?raw=true" style="width: 40%;"/>
</div>

Both are very abnormal: the **reward calculation** on the left is very slow; the **engine generation** on the right is very slow. We further analyze:

1.  Why is the reward calculation in the left image so slow? I remember GSM8K's reward calculation is just a string comparison, a pure CPU function, it shouldn't be that slow; this is a very suspicious point; and it's worth noting: I'm not sure if the reward calculated during the rollout phase is used during the training phase.

> The reward calculation during the rollout phase is for calculating the end-to-end reward, including tool calling and interaction reward. The reason for the high proportion of `reward_cal` might be that the full response was input. We submitted [a PR](https://github.com/volcengine/verl/pull/2568) to only use the last 300 characters of the last turn to see.

3.  Why is the engine call in the right image so slow?

> Note that based on our profiling tool, we can actually get the time consumption for turn 1 and turn 2. The time for turn 1 is very normal, only 20 seconds, but turn 2 took 150 seconds, and we observed significantly more engine generation than normal. Unfortunately, we did not synchronously capture GPU behavior with nsys at that time. We then conducted 6 repeated experiments, and no peak occurred again. However, no peak might be a good thing, lol.

### Long Preprocessing Time

Next, we'll demonstrate what insights we can gain through the profiling tool. We further analyzed the percentage of time consumed by each request. It can be seen that only three events are significant:

  - `async_generate_duration`: the actual rollout
  - `preprocessing_duration`: preprocessing
  - `barrier_wait_duration`: waiting for all workers to finish rollout

| Event | Avg % of Total Time |
|-------------------------|----------|
| async_generate_duration | 74.05%   |
| preprocessing_duration  | 18.57%   |
| barrier_wait_duration   | 6.39%    |
| broadcast_duration      | 0.52%    |
| data_extraction_duration| 0.35%    |
| cache_flush_duration    | 0.08%    |
| padding_duration        | 0.04%    |
| sorting_duration        | 0.00%    |
| final_construction_duration| 0.00% |
| batch_construction_duration| 0.00% |
| concatenation_duration  | 0.00%    |

We further analyzed the curve of the average percentage of each event in each step. We found that as the response length increased, the proportion of preprocessing also decreased. We believe that the time consumption of preprocessing is almost constant and has little relation to the response length. Furthermore, preprocessing is independent of the step status and can be done during dataset construction. Additionally, the agent loop feature further optimized the time consumption of preprocessing. We will analyze this in a future article.

<div style="display: flex; align-items: center;">
<img src="https://github.com/PrinsYin/verl/blob/multiturn_profile_log/profile-multiturn-scripts/script-examples/event_percentage_step.png?raw=true" style="width: 40%;"/>
</div>

### Turn Analysis

Finally, we analyze the distribution of turns:

<div style="display: flex; align-items: center;">
<img src="https://github.com/PrinsYin/verl/blob/multiturn_profile_log/profile-multiturn-scripts/script-examples/turn_difference.png?raw=true" style="width: 25%;"/>
<img src="https://github.com/PrinsYin/verl/blob/multiturn_profile_log/profile-multiturn-scripts/script-examples/turn_distribution.png?raw=true" style="width: 33.33%;"/>
<img src="https://github.com/PrinsYin/verl/blob/multiturn_profile_log/profile-multiturn-scripts/script-examples/turns_num.png?raw=true" style="width: 33.33%;"/>
</div>

From left to right, these are the average engine generation time for different turns in each step for the first 80 steps; the average time consumed by requests with different numbers of turns; and the average proportion of requests with different numbers of turns in each step.

Intuitively, the actual time spent on engine generation is not much, and most requests have only one or two turns.