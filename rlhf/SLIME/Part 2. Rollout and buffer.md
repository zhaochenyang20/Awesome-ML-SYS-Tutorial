### Rollout and Buffer

本节用尽量直白的方式讲清楚两件事：
- 整体链路：从拿到一个 prompt，到拿到可训练的 `train_data` 字典的完整链路。
- 每个函数干什么：`Buffer`、`RolloutDataSource`、`RolloutManager/Engine` 的关键方法职责与输入输出。


---

## High Level Concepts

- RolloutManager（统一调度）
  - 启动/连接 sglang-router；创建 `Buffer` actor；创建并初始化一组推理引擎；提供异步 API（生成、重置前缀缓存、offload/onload）。
- RolloutRayActor（推理引擎）
  - 基于 `SglangEngine` 构建多机/多卡推理能力；提供权重更新、前缀缓存重置、暂停/继续、显存 offload 等。
- Buffer（数据缓冲 + 转换）
  - 管理本地缓冲区与数据源；调度 rollout/评测生成函数；把推理结果转为标准训练数据格式。
- RolloutDataSource（数据源）
  - 从真实 `Dataset` 顺序/循环取样，或生成“占位样本”；维护 epoch/offset/index；可保存/恢复进度并支持shuffle。

---

## E2E Flow

1) 训练侧发起一次 rollout：调用 `RolloutManager.async_generate(rollout_id)`。
2) `RolloutManager` 将请求转给 `Buffer.generate(rollout_id)`（Ray 远程调用）。
3) `Buffer` 取样：先用本地缓冲区，缺的由 `RolloutDataSource.get_samples()` 从数据集/占位生成补齐。
4) `Buffer` 调用生成函数：`generate_rollout(args, rollout_id, buffer)`，驱动推理引擎并把结果写回 `Sample`。
5) 非评测路径下，`Buffer` 可选择把原始样本 dump 到磁盘（debug/复现）。
6) `Buffer._convert_samples_to_train_data(samples)` 把 `Sample` 列表转成训练端统一使用的dict（tokens、loss_masks、rewards 等）。
7) `Buffer` 返回 Ray 引用（`Box(ray.put(train_data))`）。
8) 训练侧拿到引用后，进入本轮训练/更新。

---

## Buffer：取样、生成、格式转换

[`Buffer`](https://github.com/THUDM/slime/blob/main/slime/ray/buffer.py) 功能与要点：
- 取样：`get_samples(self, num)` 先消耗本地 `buffer`，不足则用 `RolloutDataSource.get_samples(num)` 补齐。
- 生成：`generate(self, rollout_id, evaluation=False)` 按需选择生成函数并转化output为 `train_data`。
- 转换：`_convert_samples_to_train_data(self, samples)` 产出训练端所需的张量列表字典，并补齐/校验 `loss_masks`。
- Box(ray.put(data))：为避免大对象拷贝，返回的是 Ray 对象引用的轻量包装。
- loss_masks：若未提供，会在转换阶段按 response_length 自动补 1，并做长度校验。

<details><summary>Buffer 关键实现</summary>

```python
@ray.remote
class Buffer:
    def __init__(self, args, wandb_run_id):
        self.data_source = RolloutDataSource(args)
        self.buffer: list[list[Sample]] = []
        self.buffer_filter = pop_first if args.buffer_filter_path is None else load_function(args.buffer_filter_path)
        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)
        if num_samples:
            samples += self.data_source.get_samples(num_samples=num_samples)
        return samples

    def generate(self, rollout_id, evaluation=False):
        self.rollout_id = rollout_id
        if not evaluation and self.args.load_debug_rollout_data:
            data = torch.load(open(self.args.load_debug_rollout_data.format(rollout_id=rollout_id), "rb"))["samples"]
            data = [Sample.from_dict(sample) for sample in data]
        else:
            fn = self.eval_generate_rollout if evaluation else self.generate_rollout
            data = fn(self.args, rollout_id, self, evaluation=evaluation)
            if not evaluation and isinstance(data[0], list):
                data = sum(data, [])

        if not evaluation:
            if (tmpl := self.args.save_debug_rollout_data) is not None:
                path = Path(tmpl.format(rollout_id=self.rollout_id))
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dict(rollout_id=self.rollout_id, samples=[s.to_dict() for s in data]), path)
            data = self._convert_samples_to_train_data(data)
        return Box(ray.put(data))

    def _convert_samples_to_train_data(self, samples: list[Sample]):
        train_data = {
            "tokens": [s.tokens for s in samples],
            "response_lengths": [s.response_length for s in samples],
            "rewards": [s.get_reward_value(self.args) for s in samples],
            "truncated": [1 if s.status == Sample.Status.TRUNCATED else 0 for s in samples],
            "sample_indices": [s.index for s in samples],
        }
        loss_masks = []
        for s in samples:
            if s.loss_mask is None:
                s.loss_mask = [1] * s.response_length
            assert len(s.loss_mask) == s.response_length
            loss_masks.append(s.loss_mask)
        train_data["loss_masks"] = loss_masks
        if samples and samples[0].metadata and "raw_reward" in samples[0].metadata:
            train_data["raw_reward"] = [s.metadata["raw_reward"] for s in samples]
        if samples and samples[0].metadata and "round_number" in samples[0].metadata:
            train_data["round_number"] = [s.metadata["round_number"] for s in samples]
        return train_data
```

</details>

使用与扩展：
- Plug and play：把你自己的 `generate_rollout(args, rollout_id, buffer, evaluation=False)` 函数路径写进 `args.rollout_function_path` 即可替换。
- 自定义取样策略：实现 `buffer_filter(args, rollout_id, buffer, num_samples)` 并在 `args.buffer_filter_path` 指定。
- Debug/复现：通过 `args.save_debug_rollout_data` 导出、`args.load_debug_rollout_data` 导入。

---

## RolloutDataSource：数据从哪来，怎么分组

核心语义：
- 两种来源：
  - 真实数据集：使用 `Dataset` + `AutoTokenizer`，按 `rollout_max_prompt_len` 截断。
  - 占位样本：无真实数据时，仍然能跑完整流程（只负责 index/分组）。
- 分组输出：返回值是二维列表 `list[list[Sample]]`，每个内层列表对应同一个 prompt 的多条采样，长度为 `n_samples_per_prompt`。
- 进度管理：维护 `epoch_id / sample_offset / sample_index`，支持 `save()/load()`。

<details><summary>RolloutDataSource 核心代码</summary>

```python
class RolloutDataSource:
    def __init__(self, args):
        self.epoch_id = 0; self.sample_index = 0; self.sample_offset = 0
        if args.rollout_global_dataset:
            tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
            self.dataset = Dataset(
                args.prompt_data, tokenizer=tokenizer, max_length=args.rollout_max_prompt_len,
                prompt_key=args.input_key, label_key=args.label_key, metadata_key=args.metadata_key,
                tool_key=args.tool_key, apply_chat_template=args.apply_chat_template, seed=args.rollout_seed,
            )
            if args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)
        else:
            self.dataset = None

    def get_samples(self, num_samples):
        samples = []
        if self.dataset is not None:
            # 顺序/循环取样，并为每个 prompt 复制 n_samples_per_prompt 份
            if self.sample_offset + num_samples <= len(self.dataset):
                prompt_samples = self.dataset.samples[self.sample_offset : self.sample_offset + num_samples]
                self.sample_offset += num_samples
            else:
                prompt_samples = self.dataset.samples[self.sample_offset :]
                num_samples -= len(prompt_samples)
                self.epoch_id += 1
                if self.args.rollout_shuffle:
                    self.dataset.shuffle(self.epoch_id)
                prompt_samples += self.dataset.samples[:num_samples]
                self.sample_offset = num_samples
            for prompt_sample in prompt_samples:
                group = []
                for _ in range(self.args.n_samples_per_prompt):
                    s = copy.deepcopy(prompt_sample)
                    s.index = self.sample_index; self.sample_index += 1
                    group.append(s)
                samples.append(group)
        else:
            # 生成占位 Sample
            for _ in range(num_samples):
                group = []
                for _ in range(self.args.n_samples_per_prompt):
                    s = Sample(index=self.sample_index); self.sample_index += 1
                    group.append(s)
                samples.append(group)
        return samples
```

</details>

---

## RolloutManager 与推理引擎：怎么把“生成”组织起来

职责与关键点：
- 创建 Buffer actor 与一组 `RolloutRayActor` 推理引擎，按 Placement Group 绑定到固定 GPU/节点。
- 多节点引擎只对“第 0 节点”发请求，内部通过进程组同步。
- 管理路由器（sglang-router），以及常用控制原语（前缀缓存重置、offload/onload）。

<details><summary>RolloutManager/Engine（节选）</summary>

```python
@ray.remote
class RolloutRayActor(RayActor):
    def init(self, dist_init_addr, port, nccl_port):
        self.infer_engine = SglangEngine(args=self.args, rank=self.rank,
                                         dist_init_addr=dist_init_addr, port=port, nccl_port=nccl_port)
        if self.args.offload:
            self.infer_engine.sleep()

class RolloutManager:
    def __init__(self, args, pg, wandb_run_id):
        _start_router(args)
        self.data_buffer = Buffer.options(num_cpus=1, num_gpus=0).remote(args, wandb_run_id=wandb_run_id)
        self.all_rollout_engines = create_rollout_engines(args, pg)
        nodes_per_engine = max(1, args.rollout_num_gpus_per_engine // args.rollout_num_gpus_per_node)
        self.rollout_engines = self.all_rollout_engines[::nodes_per_engine]
        self.rollout_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()

    def async_generate(self, rollout_id, evaluation=False):
        return self.data_buffer.generate.remote(rollout_id, evaluation=evaluation)
```

</details>

<details><summary>Start Router</summary>

```python
def _start_router(args):
    if args.sglang_router_ip is not None:
        return
    from sglang_router.launch_router import RouterArgs
    args.sglang_router_ip = get_host_info()[1]
    args.sglang_router_port = find_available_port(random.randint(3000, 4000))
    router_args = RouterArgs(host=args.sglang_router_ip, port=args.sglang_router_port, balance_abs_threshold=0)
    process = multiprocessing.Process(target=run_router, args=(router_args,))
    process.daemon = True; process.start(); time.sleep(3); assert process.is_alive()
```

</details>

---


## 自定义拓展

- 更换生成逻辑：实现 `generate_rollout(args, rollout_id, buffer, evaluation=False)` 并在 `args.rollout_function_path` 指定路径。
- 自定义取样策略：实现 `buffer_filter(args, rollout_id, buffer, num_samples)` 并在 `args.buffer_filter_path` 指定。
- 多路奖励/远程 RM：把数值放进 `Sample.metadata["raw_reward"]`，转换阶段会同步到 `train_data["raw_reward"]`。
- 进度管理：用 `RolloutDataSource.save()/load()` 进行断点续跑，并配合 `rollout_shuffle`来shuffle。

---


