# verl sglang multi-turn over sample

## 快速复现

1. 创建新的 docker（如果熟悉这套安装，可以跳过）：

使用前需要配置好 `WANDB_API_KEY`，参考[这个过程](https://community.wandb.ai/t/where-can-i-find-the-api-token-for-my-project/7914)。

```bash
# 如果你的系统没有配置过 HF_TOKEN 和 WANDB_API_KEY，请先配置好
docker run -it --name h100_verl_{your_name} --gpus all \
    --shm-size 32g \
    -v {your_cache_path}:/root/.cache \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env "WANDB_API_KEY=$WANDB_API_KEY" \
    --ipc=host \
    lmsysorg/sglang:latest \
    /bin/bash
```

进入 docker 后，可以查看被映射的环境变量：

```bash
echo $HF_TOKEN
echo $WANDB_API_KEY
```

以后每次从 docker 里面 exit 出来，再用这个指令可以重启：

```bash
docker start -i h100_verl_{your_name}
```

2. 基于源码安装 verl-sglang

配置 python 环境：

```bash
mkdir -p /tmp
chmod 1777 /tmp
apt update
apt install -y python3.10 python3.10-venv
python3 -m ensurepip --upgrade
python3 -m venv ~/.python/verl-sglang
source ~/.python/verl-sglang/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade uv
```

3. 安装 verl-sglang：

```bash
cd ~
git clone -b over_sample https://github.com/zhaochenyang20/verl.git
cd verl

python -m uv pip install wheel setuptools
python3 -m uv pip install -e ".[sglang]" --prerelease=allow
python3 -m uv pip install -r ./requirements.txt --no-build-isolation
python3 -m uv pip install torch_memory_saver
```

4. 测试 gsm8k：

```bash
cd ~/verl
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 拉取并预处理 gsm8k 数据集
python examples/data_preprocess/gsm8k_multiturn_w_tool.py

# 启动 8 卡训练
bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh
```

5. 测试 dapo：

```bash
cd ~/verl
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash examples/sglang_multiturn/run_qwen3_4b_dapo_multiturn.sh
```

## 设计思路和具体实现

基于这个 commit：[b979a73e358313afafab5db512cd5ae0009ccac0](https://github.com/zhaochenyang20/verl/tree/b979a73e358313afafab5db512cd5ae0009ccac0)

设计思路已经讨论了非常多次了，为了解决 long tail 问题，采用 over sample 是非常常见的策略。相比于 partial rollout，此处设计的策略更粗暴。没有完成的 reqs 将会直接被丢弃。

具体通过 `monitor_and_cancel`，`process_request_with_monitoring` 和 `run_with_cancellation` 三个函数来实现。`monitor_and_cancel` 负责监控完成数量，一旦达到目标立即行动，取消剩余任务，并向 engine 发送 abort 信号。`process_request_with_monitoring` 负责处理单个请求，并根据完成情况返回真实结果或 padding 数据。`run_with_cancellation` 同时启动 `monitor_and_cancel` 和 `process_request_with_monitoring`。

- `process_request_with_monitoring`

```python
async def process_request_with_monitoring(req):
    nonlocal completed_count
    try:
        result = await self._async_rollout_a_request(req, do_sample, is_validate, **kwargs)

        async with completion_lock:
            if completed_count < target_completion:
                completed_count += 1
                print(f"✅ Request {req.request_id} completed ({completed_count}/{total_requests})")
                return result  # 返回真实结果
            else:
                # 超过目标，返回padding
                logger.info(f"Request {req.request_id} finished after target met, creating padding")
                return self._create_padding_request(req)
```

1. 每个 request 会独立启动自身的 `process_request_with_monitoring` 中，通过 `await` 阻塞式执行 `_async_rollout_a_request`。
2. 对于那些较早完成的 request，result 得到了真实结果，对 `completed_count` 计数器递增。注意这里 `completed_count` 是全局变量，需要使用 `completion_lock` 确保计数操作的原子性，读写不会冲突。
3. 对于那些较晚完成的 request，`monitor_and_cancel` 检测到 `completed_count` 达到 `target_completion`，会取消这些任务，并向 sglang engine 发送 `abort_requests` 请求。

- `monitor_and_cancel`

```python
async def monitor_and_cancel():
    nonlocal completed_count
    while completed_count < target_completion:
        await asyncio.sleep(0.1)  # 每0.1秒检查一次

    print(f"🎯 Target reached: {completed_count}/{total_requests} completed!")
    print("🚫 Cancelling remaining requests and sending abort to engine...")

    # 取消剩余的任务
    cancelled_count = 0
    for task in all_tasks:
        if not task.done():
            task.cancel()
            cancelled_count += 1

    # 向engine发送abort信号
    try:
        abort_result = await self._engine.abort_request(abort_all=True)
        print(f"✅ Abort signal sent to engine: {abort_result}")
    except Exception as e:
        print(f"❌ Failed to send abort signal to engine: {e}")
```

持续监控完成数量，一旦达到目标立即行动，取消剩余任务，并向 sglang engine 发送 `abort_requests` 信号。注意这里的 engine abort 实际上写法在 `sglang_rollout.py` 里面的 `AsyncEngine` 类：

```python
    async def abort_request(self, rid: str = "", abort_all: bool = False):
        """Abort a specific request or all requests.

        Args:
            rid: The request ID to abort. If empty and abort_all is False, no action is taken.
            abort_all: If True, abort all running requests regardless of rid.
        """
        try:
            result = self.tokenizer_manager.abort_request(rid=rid, abort_all=abort_all)
            print(f"🔍 Abort result: {result}")
            return result if result is not None else {"status": "aborted"}
        except Exception as e:
            logger.error(f"Failed to abort requests: {e}")
            raise
```

这里有几点值得玩味：

1. 其实 verl 的 `AsyncEngine` 继承并且重写了 sglang Engine 的很多方法，比如 `update_weights_from_tensor` 和 `resume_memory_occupation`。按理说其实 sglang Engine 不实现这些方法也不影响 verl，当然影响其他框架。一开始我以为必须要现在 sglang 中实现对 Engine 的 `abort_request`，因为起初只有 server 有而 engine 没有。但是考虑到 `AsyncEngine` 重写了 `abort_request`，所以其实 sglang Engine 不需要实现这一功能，我们也无需为此发版。毕竟，在 verl 上更新 SGLang 版本确实太痛苦了。
2. 【现在我们只是 abort 了 engine，tool 是否 abort 影响如何？】
3. 【和 `update_weights_from_tensor` 不一样，`abort_request` 内部是不能通过 await 去调用 `self.tokenizer_manager.abort_request`，得直接调用。感谢 jiajun 和 yuzhen 的提醒。这里得去查 sglang tokenizer_manager 内部的实现。如果某个函数在 tokenizer_manager 中是异步实现的，那么外部调用才可以有 await 的语法。坦诚说这里是因为我对异步语法并不熟悉，而且我也不理解为什么 `resume_memory_occupation` 和 `abort_request` 在 tokenizer_manager 中，前者是异步的，后者是同步的。此外，如果我们要在一个异步函数中只写一行 await 某个异步函数，这就意味着其实在等待内部的异步函数执行完成么？这么写有什么意义呢？ 具体来说：】

```python
# sglang_rollout.py

    async def resume_memory_occupation(self, tags: Optional[list[str]] = None):
        """Resume GPU occupation."""
        # because __init__ is a sync method, it can not call the async release_memory_occupation
        # have to move release_memory_occupation from __init__ to here
        # For multi-stage awake, we run release weight and kv_cache when we resume weights for the first time.
        if self._need_reload:
            await self.release_memory_occupation()
            self._need_reload = False

        if tags is None:
            obj = ResumeMemoryOccupationReqInput()
        else:
            obj = ResumeMemoryOccupationReqInput(tags=tags)
        return await self.tokenizer_manager.resume_memory_occupation(obj, None)
```

内层的 `self.tokenizer_manager.resume_memory_occupation` 是一个异步函数，所以在外层的 `resume_memory_occupation` 函数在等待内层的完成。如果这样，为什么外层的 `resume_memory_occupation` 函数不能是同步的呢？目前外层的函数是异步的，所以调用外层函数并等待也得 await。比如说：

```python
# fsdp_sglang.py

async def release_memory(self):
    if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.rollout_config.free_cache_engine:
        if self.multi_stage_wake_up:
            await self.inference_engine.release_memory_occupation(tags=["kv_cache", "weights"])
        else:
            await self.inference_engine.release_memory_occupation()
        log_gpu_memory_usage("After release memory occupation in sharding manager", logger=logger)
```


- `run_with_cancellation`


```python
async def run_with_cancellation():
    nonlocal all_tasks

    # 创建所有任务
    all_tasks = [asyncio.create_task(process_request_with_monitoring(req)) for req in req_list]

    # 启动监控任务
    monitor_task = asyncio.create_task(monitor_and_cancel())

    try:
        # 等待所有任务完成（包括被取消的）
        results = await asyncio.gather(*all_tasks, return_exceptions=True)

        # 处理结果，将异常转换为padding
        output_req_list = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # 异常转换为padding
                logger.warning(f"Task {i} resulted in exception: {result}")
                output_req_list.append(self._create_padding_request(req_list[i]))
            else:
                output_req_list.append(result)

        return output_req_list
    finally:
        # 清理监控任务
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
```

有了对前两者的理解，最后 `run_with_cancellation` 就非常清晰了。注意，`all_tasks` 和读写锁 `completion_lock` 是这三个函数的全局变量。这里同时启动所有 reqs 的 `process_request_with_monitoring` 并且创造 `monitor_task` 来监视。注意，虽然每个 req 的 `_async_rollout_a_request` 未必会完成，但是这个函数上层的 `process_request_with_monitoring` 是一定会结束的，所以 `results = await asyncio.gather(*all_tasks, return_exceptions=True)` 一定会返回，然后逐个处理 `results`，这时候里面存在三种情况：`COMPLETED`，`Exception` 和 `PADDING`。将 `Exception` 转换为 `PADDING` 后，返回 `output_req_list`。

整体读完，我觉得设计的还算清晰，实现可能未必好，还要大改。

这里列几个我觉得必须要检查的地方：

1. 【TODO：直接打成 padding 的实现是否正确，至少一定不能有 loss。我理想的设计中，这个 reqs 就是被丢弃了，让 GRPO 的 group size 减小了，这个请求就是不存在，而且还会省下一些训练时间。这里得仔细检查，是否除开 response_loss_mask 设为 0 之外还有别的要修改的地方。我一开始修改了 `agg_loss` 函数，但是后来问了 claude，可能并不需要，需要额外确认。此外，reward 函数是否需要更改，我也并不确定，对于 FSDP 来说，是这个函数 `def _expand_to_token_level`。最后，假如我们实现了完美丢弃，这样不同的 GRPO group 的 requests 数量不一致，按理来说这会影响 GRPO group 的 variance，可能会让训练更加不稳定。能否刻画这一影响？】还有一种设计，直接打成 padding 实际上把 partial rollout 得到的 trajs 直接丢了。可以考虑保留这些 tracjs，但是 loss mask 为 0，然后 reward 也是 0，听龙老师说可能好些。
2. 在异步的部分，我提到了这个问题：

> 坦诚说这里是因为我对异步语法并不熟悉，而且我也不理解为什么 `resume_memory_occupation` 和 `abort_request` 在 tokenizer_manager 中，前者是异步的，后者是同步的。此外，如果我们要在一个异步函数中只写一行 await 某个异步函数，这就意味着其实在等待内部的异步函数执行完成么？这么写有什么意义呢？ 具体来说：

3. 整个实现能否更清晰一些，能够让 verl team 同意这个简单的 feature。目前我需要更改 sglang_rollout.py 和 metric_utils.py 的 compute_data_metrics 函数。前者已经讲述很多了，后者其实很 tricky。我目前将那些被 abort 掉的请求在每次 reward 算均值的时候排除在分母之外。有几个问题：【先是我们需要确认，在 validation 的时候，没有 abort 掉的请求，理论上 validation step 不会受到 `aborted_mask = (response_length == 0).bool() ` 的影响，这个得做实验验证下，validation step 就不会有 aborted reqs。】此外，如果把这些被 abort 的请求记录在 metric 中，这一部分的 reward 全是 0，所以相比不做 over sample 的 baseline，over sample 的 reward 会低很多。如果又不考虑这些被 abort 的请求，实际上这些被 abort 的请求往往是更多 turn 更难的 example，这部分 reward 相比那些没有被 abort 的请求更低。所以无视这部分的 reward 会导致 reward 虚高。目前我没有办法避免这种虚高，但是我认为 validation step 的 reward 是准确的，目前并无大碍。【我们直接让 loss mask = 0 就避免了 padding 的 reqs 影响 loss，但是我问 GPT，貌似只对特定的 agg loss mode 是有效的，这里得去研究下。】

整体上表现符合预期，training step 的 reward 虚高，validation step 的 reward 能够和 baseline align 上，而 rollout time 有明显改善。
