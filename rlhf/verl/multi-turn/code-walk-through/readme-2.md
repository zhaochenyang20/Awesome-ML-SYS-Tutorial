# 深入浅出理解 verl 源码

在 [Part 1](readme.md) 中，我们介绍了 verl 的初始化过程，我们进一步介绍 verl 的训练过程。

在 GRPO 中，单个 step 包含四个阶段：load data -> rollout -> make experience -> update model。区别于前一节的详述，本节会使用伪代码结合源码的方式进行阐述。

```mermaid
flowchart LR
subgraph W2["Initialize"]
WP[Process Data] --> A
direction TB D1[Data Prepare] --> A
A[TaskRunner] --> B1[RayPPOTrainer]
B1 --> Workers

    subgraph Workers["Workers"]
        direction TB
                WA[ActorRolloutWorker] --> WD[FSDP Engine]
        WB[CriticWorker] --> WD
        WC[RewardModelWorker] --> WD
        WD --> WE[SGLang Engine]
    end
    
    Workers --> C1[Hybrid Engine]
end

subgraph W3["Train Loop"]
    direction TB
    E[DataLoader] --> RolloutBox
    
    subgraph RolloutBox["Rollout"]
        F1[Prepare Data] --> F2[SGLang Async Rollout]
        F2 --> F3[Multi-turn Chat Process]
    end
    
    RolloutBox --> ExpBox
    
    subgraph ExpBox["Make Experience"]
        G1[Recompute Log Probs] --> G2[Compute Reward]
        G2 --> G3[Compute Advantage]
    end
    
    ExpBox --> UpdateBox
    
    subgraph UpdateBox["Train The Model"]
        H1[Load FSDP Model Weight] --> H2[Compute Gradient]
        H2 --> H3[Weights Update]
        H3 --> H4[Sync Weights]
    end
    
    UpdateBox --> E
end

W2 --> W3

```

## 数据加载与预处理

verl 通过 `DataProto` 和 `RLHFDataset` 来实现数据处理。具体来说，在 [`main_ppo.py`](https://github.com/volcengine/verl/blob/76f63cffa5081564d8fea93a1cb3ce8bd5bdcc39/verl/trainer/main_ppo.py#L193) 中，我们观察这个函数：

<details>
<summary>create_rl_dataset 源码</summary>

```python
def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    # Check if a custom dataset class is specified in the data configuration
    # and if the path to the custom class is provided
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type

        # Dynamically load the custom dataset class
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        # Verify that the custom dataset class inherits from torch.utils.data.Dataset
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
    else:
        # Use the default RLHFDataset class if no custom class is specified
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    # Instantiate the dataset using the determined dataset class
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset
```

</details>

非常典型，创造一个了 `RLHFDataset` 实例，并返回。而具体的 [`RLHFDataset`](https://github.com/volcengine/verl/blob/76f63cffa5081564d8fea93a1cb3ce8bd5bdcc39/verl/utils/dataset/rl_dataset.py#68) 实现如下：

<details>
<summary>RLHFDataset 实现</summary>

```python
class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    messages = self._build_messages(doc)
                    raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    images = [process_image(image) for image in messages.pop(image_key)] if image_key in messages else None
                    videos = [process_video(video) for video in messages.pop(video_key)] if video_key in messages else None

                    return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])

            else:

                def doc2len(doc) -> int:
                    return len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))

            self.dataframe = self.dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(self.dataframe)}")

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}

            images = None
            if self.image_key in row_dict and row_dict.get(self.image_key, None) is not None:
                images = [process_image(image) for image in row_dict.pop(self.image_key)]
                multi_modal_data["image"] = images

            videos = None
            if self.video_key in row_dict and row_dict.get(self.video_key, None) is not None:
                videos = [process_video(video) for video in row_dict.pop(self.video_key)]
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(model_inputs)

            # second_per_grid_ts isn't used for training, just for mrope
            row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()

```

</details>

1. 支持从远程存储（如 HDFS）下载 Parquet 文件到本地缓存，支持共享内存（shared memory）加速文件访问，自动管理文件路径，支持检查点恢复。
2. 使用 HuggingFace `datasets` 库读取 Parquet 文件，支持多个数据文件的合并，自动处理数据格式转换。
3. 根据最大长度过滤过长的 prompts，支持多进程并行处理，可配置的过滤策略。
4. 支持图像和视频的多模态输入，解析 `<image>` 和 `<video>` 标签，将多模态内容转换为结构化格式。
5. 添加 chat template 来格式化对话，将文本转换为 token IDs，生成注意力掩码和位置编码。
6. padding 到指定长度，支持多种截断策略（left, right, middle, error），生成位置编码。
7. 支持训练中断后的恢复，可以从原始文件重新构建数据集，兼容序列化/反序列化。
8. 返回包含以下关键字段的字典：`input_ids`, `attention_mask`, `position_ids`, `raw_prompt_ids`, `multi_modal_data`, `multi_modal_inputs`, `index`, `tools_kwargs`。

这里最重要的一个参数是 `tools_kwargs`，用于为不同的 tools 提供配置参数。它的结构如下：

```python
tools_kwargs = {
    "tool_name": {
        "create_kwargs": {...},      # 工具创建时的参数
        "execute_kwargs": {...},     # 工具执行时的参数（可选）
        "calc_reward_kwargs": {...}, # 计算奖励时的参数（可选）
        "release_kwargs": {...},     # 释放资源时的参数（可选）
    }
}
```

比如 Search-R1 的 `tools_kwargs` 如下：

```python
tools_kwargs = {
    "search-r1": {
        "create_kwargs": {
            "ground_truth": ground_truth,
            "question": question, 
            "data_source": data_source_tagged
        }
    }
}
```

具体这些参数是如何调用了一个 tool，我们会留在后续部分继续介绍。

## 训练入口 [`RayPPOTrainer.fit()`](https://github.com/volcengine/verl/blob/76f63cffa5081564d8fea93a1cb3ce8bd5bdcc39/verl/trainer/ppo/ray_trainer.py#L903)

1. 创建 Tracking 日志记录器，设置全局步数，加载检查点，并在训练前进行验证。
2. 使用 tqdm 创建进度条，显示训练进度，并设置初始步数。
3. 遍历配置的总 epoch 数和数据加载器，每个 train batch 更新多步。
4. 从 batch 中分离出用于 rollout 的数据（`input_ids`, `attention_mask`, `position_ids` 等），保留其他数据用于后续处理。
5. 调用 `ActorRolloutWorker` 生成序列，并记录生成时间。
6. 处理 REMAX 基线（如果使用）：生成确定性基线序列，计算基线奖励，用于 REMAX 优势估计器。
7. 为每个样本分配唯一 ID，重复数据以对齐多次采样，计算响应掩码，并可选地进行批次平衡。
8. 根据配置使用奖励模型或自定义奖励函数计算 token 级别的奖励分数，支持同步和异步计算。
9. 使用 megatron 基于训练开始前的 policy 重新计算 behaviour policy 的 log probabilities，用于重要性采样，同时计算熵值。（原因在 [part 1](./readme.md#actorrolloutrefworker__init__) 讲过）
10. 使用 reference policy 计算 log probs，用于 KL 散度计算。
11. 使用 Critic 网络计算状态价值，用于优势函数估计。
12. 根据配置的优势估计器（GAE、GRPO、REMAX 等）计算优势函数，支持 KL 惩罚。
13. 使用计算出的优势函数更新 Critic 网络参数。
14. 在 Critic 预热完成后，使用 PPO 损失函数更新 Actor 网络参数。
15. 将生成的序列、输入、输出和分数保存到指定目录。
16. 根据配置的频率执行验证，计算验证指标并记录。
17. 根据配置的频率保存模型检查点。
18. 收集训练指标、时序指标和吞吐量指标，并记录到日志系统。
19. 更新进度条，递增全局步数，并在达到总训练步数时结束训练。
20. 根据配置在特定步数启用/禁用性能分析，用于调试和优化。

<details>
<summary>RayPPOTrainer.fit() 源码</summary>

```python
def fit(self):
    """
    The training loop of PPO.
    The driver process only need to call the compute functions of the worker group through RPC
    to construct the PPO dataflow.
    The light-weight advantage computation is done on the driver process.
    """
    from omegaconf import OmegaConf

    from verl.utils.tracking import Tracking

    logger = Tracking(
        project_name=self.config.trainer.project_name,
        experiment_name=self.config.trainer.experiment_name,
        default_backend=self.config.trainer.logger,
        config=OmegaConf.to_container(self.config, resolve=True),
    )

    self.global_steps = 0

    # load checkpoint before doing anything
    self._load_checkpoint()

    # perform validation before training
    # currently, we only support validation using the reward_function.
    if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
        val_metrics = self._validate()
        assert val_metrics, f"{val_metrics=}"
        pprint(f"Initial validation metrics: {val_metrics}")
        logger.log(data=val_metrics, step=self.global_steps)
        if self.config.trainer.get("val_only", False):
            return

    # add tqdm
    progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

    # we start from step 1
    self.global_steps += 1
    last_val_metrics = None

    for epoch in range(self.config.trainer.total_epochs):
        for batch_dict in self.train_dataloader:
            do_profile = self.global_steps in self.config.trainer.profile_steps if self.config.trainer.profile_steps is not None else False
            if do_profile:
                self.actor_rollout_wg.start_profile()
                if self.use_reference_policy:
                    self.ref_policy_wg.start_profile()
                if self.use_critic:
                    self.critic_wg.start_profile()
                if self.use_rm:
                    self.rm_wg.start_profile()

            metrics = {}
            timing_raw = {}
            batch: DataProto = DataProto.from_single_dict(batch_dict)

            # pop those keys for generation
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            gen_batch = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            is_last_step = self.global_steps >= self.total_training_steps

            with marked_timer("step", timing_raw):
                # generate a batch
                with marked_timer("gen", timing_raw, color="red"):
                    if not self.async_rollout_mode:
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    else:
                        self.async_rollout_manager.wake_up()
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        self.async_rollout_manager.sleep()
                    timing_raw.update(gen_batch_output.meta_info["timing"])
                    gen_batch_output.meta_info.pop("timing", None)

                if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    with marked_timer("gen_max", timing_raw, color="purple"):
                        gen_baseline_batch = deepcopy(gen_batch)
                        gen_baseline_batch.meta_info["do_sample"] = False
                        gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                        batch = batch.union(gen_baseline_output)
                        reward_baseline_tensor = self.reward_fn(batch)
                        reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                        batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                        batch.batch["reward_baselines"] = reward_baseline_tensor

                        del gen_baseline_batch, gen_baseline_output

                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                # repeat to align with repeated responses in rollout
                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                batch = batch.union(gen_batch_output)

                batch.batch["response_mask"] = compute_response_mask(batch)
                # Balance the number of valid tokens across DP ranks.
                # NOTE: This usually changes the order of data in the `batch`,
                # which won't affect the advantage calculation (since it's based on uid),
                # but might affect the loss calculation (due to the change of mini-batching).
                # TODO: Decouple the DP balancing and mini-batching.
                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)

                # compute global_valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

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

                # Log rollout generations if enabled
                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                if rollout_data_dir:
                    with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                        print(batch.batch.keys())
                        inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                        outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                        scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                        self._dump_generations(
                            inputs=inputs,
                            outputs=outputs,
                            scores=scores,
                            reward_extra_infos_dict=reward_extra_infos_dict,
                            dump_path=rollout_data_dir,
                        )

                # validate
                if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

            # training metrics
            metrics.update(
                {
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                }
            )
            # collect metrics
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            # TODO: implement actual tflpo and theoretical tflpo
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

            progress_bar.update(1)
            self.global_steps += 1

            if do_profile:
                self.actor_rollout_wg.stop_profile()
                if self.use_reference_policy:
                    self.ref_policy_wg.stop_profile()
                if self.use_critic:
                    self.critic_wg.stop_profile()
                if self.use_rm:
                    self.rm_wg.stop_profile()

            if is_last_step:
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return
```

这里很值得分享一个核心问题，对 SGLang 而言，或者对现在的 RL 而言，我们每天说来说去的 async 究竟是什么意思？和 PD 分离一样，async 也有非常多的层面：

1. Async RL 代表的是在 training rollout 分离的系统上，rollout 只在 update weights 的时候被打断，其余时刻永远 rollout，哪怕 target policy 正在被 training engine 更新。这方面是 [AreaL](https://github.com/inclusionAI/AReaL) 和 [SLIME](https://github.com/THUDM/slime)。

2. Async Rollout 这个词是特指在 rollout 的时候，把一个 batch requests 拆为单个 request，然后逐个调用 `SGLangEngine.generate()`。

乍一听，这没有什么特别的，似乎还会更慢些。但是考虑到 tool call 的问题，这就非常严肃了。假设我们把一整个 batch 的 requests 作为一个 batch 塞给 sglang 似乎还要快些，毕竟对 SGLang 的 scheduler 而言，更好组 batch。但是，一整个 batch 进去，得一整个 batch 出来。这些 batch 里面的 requests 同时返回，同时被 paser 解析查看是否有 tool call 的 parameter，然后发送请求给 tool。如此以来，整个 tool 的调用大概率会拥堵，甚至在我们考虑到如果要加入多个 tool（虽然目前没有）的话，用一个状态机去管理每个 request 的 tool call 状态会成一场噩梦，何况有的 requests 会在多轮里面多次调用 tool。因此，为了方便管理每个 request tool call 的状态机和让 tool 被调度的更加均匀。SGLang 采取了 Async Rollout 策略，也即把一个 batch 的 requests 拆为单个 request，然后逐个异步调用 `SGLangEngine.generate()`。这样每个 reqeuest 自己管理自己的状态机，方便维护并且 tool call 效率更高。

理解了这一层，我们可以来看看代码实现：

<details>
<summary>generate_sequences 源码</summary>

```python

@GPUMemoryLogger(role="sglang rollout", logger=logger)
@torch.no_grad()
def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
    if self.config.multi_turn.enable:
        return self._req_level_generate_sequences(prompts, **kwargs)
    return self._batch_level_generate_sequences(prompts, **kwargs)

```
</details>

这里明确指出，如果是用了 mutli-turn 训练，则将 batch 的 requests 拆为单个 request，调用 `_req_level_generate_sequences`；而不调用 tool 的单轮 RL，仍旧组 batch 直接发送。

我们只观察 `_req_level_generate_sequences` 的部分源码：

<details>
<summary>_req_level_generate_sequences 部分源码</summary>


```python
@GPUMemoryLogger(role="sglang rollout", logger=logger)
@torch.no_grad()
def _req_level_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
    # Async rollout with tools support
    do_sample = prompts.meta_info.get("do_sample", True)
    is_validate = prompts.meta_info.get("validate", False)
    tgt_device = prompts.batch["input_ids"].device
    if self._tp_rank == 0:
        req_list = self._preprocess_prompt_to_async_rollout_requests(
            prompts,
            n=1 if is_validate else self.config.n,
        )
        loop = asyncio.get_event_loop()
        output_req_list = loop.run_until_complete(
            asyncio.gather(
                *[self._async_rollout_a_request(req, do_sample, is_validate, **kwargs) for req in req_list],
            )
        )
        sorted_output_req_list = sorted(output_req_list, key=lambda x: (x.batch_data_id, x.rollout_offset))
    else:
        sorted_output_req_list = None
```

<details>

现在来看，`asyncio.gather(*[self._async_rollout_a_request(req, do_sample, is_validate, **kwargs) for req in req_list],)` 就显得无比清晰了。



#### 2.1.2 数据管理与分离

数据分离的设计原理在于将用于生成的数据（如`input_ids`、`attention_mask`、`position_ids`等）从主批次中弹出，同时动态检测并处理多模态数据（`multi_modal_data`），支持工具调用的配置参数（`tools_kwargs`）和交互式训练的配置参数（`interaction_kwargs`），并保留原始提示数据（`raw_prompt_ids`、`raw_prompt`）用于生成。这种数据分离的必要性源于RL训练中不同阶段对数据格式的差异化需求：在生成阶段，模型只需要prompts用于推理，不包含回答；而在训练阶段，则需要完整的prompt+response序列用于策略更新。通过这种数据分离设计，veRL能够灵活处理不同长度的序列，同时保持数据的一致性和完整性，为后续的序列生成和模型训练提供高质量的数据基础。

数据加载阶段的核心实现在 [`RayPPOTrainer.fit()`](https://github.com/volcengine/verl/blob/76f63cffa5081564d8fea93a1cb3ce8bd5bdcc39/verl/trainer/ppo/ray_trainer.py#L957) 方法中：

```python
# 从DataLoader获取原始批次数据
batch: DataProto = DataProto.from_single_dict(batch_dict)

# 弹出用于生成的数据键
batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
if "multi_modal_data" in batch.non_tensor_batch:
    non_tensor_batch_keys_to_pop.append("multi_modal_data")
if "raw_prompt" in batch.non_tensor_batch:
    non_tensor_batch_keys_to_pop.append("raw_prompt")
if "tools_kwargs" in batch.non_tensor_batch:
    non_tensor_batch_keys_to_pop.append("tools_kwargs")
if "interaction_kwargs" in batch.non_tensor_batch:
    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
gen_batch = batch.pop(
    batch_keys=batch_keys_to_pop,
    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
)
```


`DataProto` 是veRL中用于统一处理数据的核心数据结构，定义在 [`protocol.py`](https://github.com/volcengine/verl/blob/76f63cffa5081564d8fea93a1cb3ce8bd5bdcc39/verl/protocol.py#L201)：

```python
@dataclass
class DataProto:
    """
    A DataProto is a data structure that aims to provide a standard protocol for data exchange between functions.
    It contains a batch (TensorDict) and a meta_info (Dict). The batch is a TensorDict https://pytorch.org/tensordict/.
    TensorDict allows you to manipulate a dictionary of Tensors like a single Tensor. Ideally, the tensors with the
    same batch size should be put inside batch.
    """

    batch: TensorDict = None
    non_tensor_batch: Dict = field(default_factory=dict)
    meta_info: Dict = field(default_factory=dict)

    def __post_init__(self):
        # perform necessary checking
        self.check_consistency()
```

`DataProto`提供标准化的数据交换协议，基于PyTorch的TensorDict，支持张量的批量操作，同时通过`non_tensor_batch`字典来处理NumPy数组等非张量数据。`meta_info`存储额外的元信息。其中几个关键都是关于`DataProto`的创建，包括:

**[`from_single_dict()`](https://github.com/volcengine/verl/blob/76f63cffa5081564d8fea93a1cb3ce8bd5bdcc39/verl/protocol.py#L331) - 从字典创建DataProto**：
```python
@classmethod
def from_single_dict(cls, data: Dict[str, Union[torch.Tensor, np.ndarray]], meta_info=None, auto_padding=False):
    """Create a DataProto from a dict of tensors and non_tensors"""
    tensors = {}
    non_tensors = {}

    for key, val in data.items():
        if isinstance(val, torch.Tensor):
            tensors[key] = val
        elif isinstance(val, np.ndarray):
            non_tensors[key] = val
        else:
            raise ValueError(f"Unsupported type in data {type(val)}")

    return cls.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info, auto_padding=auto_padding)
```

**[`pop()`](https://github.com/volcengine/verl/blob/76f63cffa5081564d8fea93a1cb3ce8bd5bdcc39/verl/protocol.py#L517) - 弹出数据子集**：
```python
def pop(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None) -> "DataProto":
    """Pop a subset of the DataProto via `batch_keys` and `meta_info_keys`

    Args:
        batch_keys (list, optional): a list of strings indicating the keys in batch to pop
        meta_info_keys (list, optional): a list of keys indicating the meta info to pop

    Returns:
        DataProto: the DataProto with the poped batch_keys and meta_info_keys
    """
    if batch_keys is None:
        batch_keys = []
    if meta_info_keys is None:
        meta_info_keys = []
    if non_tensor_batch_keys is None:
        non_tensor_batch_keys = []

    tensors = {}
    # tensor batch
    for key in batch_keys:
        assert key in self.batch.keys()
        tensors[key] = self.batch.pop(key)
    non_tensors = {}
    # non tensor batch
    for key in non_tensor_batch_keys:
        assert key in self.non_tensor_batch.keys()
        non_tensors[key] = self.non_tensor_batch.pop(key)
    meta_info = {}
    for key in meta_info_keys:
        assert key in self.meta_info.keys()
        meta_info[key] = self.meta_info.pop(key)
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)
```

**`union()` - 合并数据**：
```python
def union(self, other: "DataProto") -> "DataProto":
    """Union with another DataProto. Union batch and meta_info separately.
    Throw an error if

    - there are conflict keys in batch and they are not equal
    - the batch size of two data batch is not the same
    - there are conflict keys in meta_info and they are not the same.

    Args:
        other (DataProto): another DataProto to union

    Returns:
        DataProto: the DataProto after union
    """
    self.batch = union_tensor_dict(self.batch, other.batch)
    self.non_tensor_batch = union_numpy_dict(self.non_tensor_batch, other.non_tensor_batch)
    self.meta_info = union_two_dict(self.meta_info, other.meta_info)
    return self
```
除此之外，`DataProto`还能通过数据验证`check_consistency()`确保在数据分离和合并过程中不会出现不一致的情况，同时支持在CPU和GPU之间高效移动数据（`to()`），这对于分布式训练中的参数卸载和加载至关重要，索引操作和批处理操作则提供了灵活的数据处理能力，使得veRL能够高效地处理不同长度序列的批处理、多模态数据的组合以及复杂的训练数据流。这些特性共同构成了veRL数据流处理的技术基础，为后续的序列生成、经验处理和模型更新阶段提供了可靠的数据管理保障。

#### 2.1.3 `RLHFDataset` 数据集类：高效的数据加载与预处理

[`RLHFDataset`]((https://github.com/volcengine/verl/blob/76f63cffa5081564d8fea93a1cb3ce8bd5bdcc39/verl/utils/dataset/rl_dataset.py#L68)) 是veRL中专门用于RLHF数据加载的数据集类：

```python
class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()
```

这一块可以讲的东西不多，和算法相关的主要就是数据的下载和tokenize：

```python
def _download(self, use_origin_parquet=False):
    from verl.utils.fs import copy_to_local

    data_files = self.data_files if not use_origin_parquet else self.original_data_files
    for i, parquet_file in enumerate(data_files):
        self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

# tokenize
def _read_files_and_tokenize(self):
    dataframes = []
    for parquet_file in self.data_files:
        # read parquet files and cache
        dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
        dataframes.append(dataframe)
    self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

    print(f"dataset len: {len(self.dataframe)}")

    # filter out too long prompts
    if self.filter_overlong_prompts:
        tokenizer = self.tokenizer
        processor = self.processor
        prompt_key = self.prompt_key
        image_key = self.image_key
        video_key = self.video_key

        if processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            def doc2len(doc) -> int:
                messages = self._build_messages(doc)
                raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                images = [process_image(image) for image in messages.pop(image_key)] if image_key in messages else None
                videos = [process_video(video) for video in messages.pop(video_key)] if video_key in messages else None

                return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])

        else:

            def doc2len(doc) -> int:
                return len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))

        self.dataframe = self.dataframe.filter(
            lambda doc: doc2len(doc) <= self.max_prompt_length,
            num_proc=self.num_workers,
            desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
        )

        print(f"filter dataset len: {len(self.dataframe)}")

# 数据获取与预处理
def __getitem__(self, item):
    """
    Note that we also return the raw_input_ids so that it can be combined with other chat template
    """
    row_dict: dict = self.dataframe[item]
    messages = self._build_messages(row_dict)
    model_inputs = {}

    if self.processor is not None:
        from verl.utils.dataset.vision_utils import process_image, process_video

        raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        multi_modal_data = {}

        images = None
        if self.image_key in row_dict and row_dict.get(self.image_key, None) is not None:
            images = [process_image(image) for image in row_dict.pop(self.image_key)]
            multi_modal_data["image"] = images

        videos = None
        if self.video_key in row_dict and row_dict.get(self.video_key, None) is not None:
            videos = [process_video(video) for video in row_dict.pop(self.video_key)]
            multi_modal_data["video"] = [video.numpy() for video in videos]

        model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        if "second_per_grid_ts" in model_inputs:
            model_inputs.pop("second_per_grid_ts")

        # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
        row_dict["multi_modal_data"] = multi_modal_data
        row_dict["multi_modal_inputs"] = dict(model_inputs)

        # second_per_grid_ts isn't used for training, just for mrope
        row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

    else:
        raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

    input_ids, attention_mask = verl_F.postprocess_data(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=self.max_prompt_length,
        pad_token_id=self.tokenizer.pad_token_id,
        left_pad=True,
        truncation=self.truncation,
    )

    if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
        from verl.models.transformers.qwen2_vl import get_rope_index

        position_ids = [
            get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )
        ]  # (1, 3, seq_len)

    else:
        position_ids = compute_position_id_with_mask(attention_mask)

    row_dict["input_ids"] = input_ids[0]
    row_dict["attention_mask"] = attention_mask[0]
    row_dict["position_ids"] = position_ids[0]

    raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
    if len(raw_prompt_ids) > self.max_prompt_length:
        if self.truncation == "left":
            raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
        elif self.truncation == "right":
            raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
        elif self.truncation == "middle":
            left_half = self.max_prompt_length // 2
            right_half = self.max_prompt_length - left_half
            raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
        elif self.truncation == "error":
            raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

    row_dict["raw_prompt_ids"] = raw_prompt_ids
    # encode prompts without chat template
    if self.return_raw_chat:
        row_dict["raw_prompt"] = messages

    # get prompts with chat template
    if self.return_full_prompt:
        row_dict["full_prompts"] = raw_prompt  # array of strings

    # add index for each prompt
    index = row_dict.get("extra_info", {}).get("index", 0)
    tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
    interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
    need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
    if need_tools_kwargs and not tools_kwargs:
        logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
    row_dict["index"] = index
    row_dict["tools_kwargs"] = tools_kwargs
    row_dict["interaction_kwargs"] = interaction_kwargs
    return row_dict
```

[`collate_fn`](https://github.com/volcengine/verl/blob/76f63cffa5081564d8fea93a1cb3ce8bd5bdcc39/verl/utils/dataset/rl_dataset.py#L37) 可以将张量数据和非张量数据分别收集并且使用`torch.stack`将张量数据堆叠成批次。这里统一将将非张量数据转换为NumPy数组。具体的实现如下：

```python
def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, *dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}
```


#### 2.1.4 FAQ

**Q1: 为什么需要将数据分离为生成数据和训练数据？**

A1: 在RLHF训练中，不同阶段需要不同的数据格式：
- **生成阶段**：只需要输入prompt，用于模型推理生成响应
- **训练阶段**：需要完整的prompt+response序列，用于计算损失和更新策略

这种分离设计避免了数据冗余，提高了内存效率，同时保持了数据的一致性。

**Q2: DataProto和普通的字典有什么区别？**

A2: DataProto提供了更强大的功能：
- **类型安全**：自动区分张量数据和非张量数据
- **批量操作**：支持索引、切片、连接等批量操作
- **设备管理**：支持设备间数据移动
- **一致性检查**：自动验证数据的一致性

**Q3: 如何处理超长序列？**

A3: veRL提供了多种截断策略：
- [**left**](https://github.com/volcengine/verl/blob/76f63cffa5081564d8fea93a1cb3ce8bd5bdcc39/verl/utils/dataset/rl_dataset.py#L280)：从左侧截断，保留序列末尾
- [**right**](https://github.com/volcengine/verl/blob/76f63cffa5081564d8fea93a1cb3ce8bd5bdcc39/verl/utils/dataset/rl_dataset.py#L282)：从右侧截断，保留序列开头
- [**middle**](https://github.com/volcengine/verl/blob/76f63cffa5081564d8fea93a1cb3ce8bd5bdcc39/verl/utils/dataset/rl_dataset.py#L284)：从中间截断，保留开头和结尾
- **error**：抛出错误，要求用户处理

**Q4: 多模态数据是如何处理的？**

A4: veRL通过ProcessorMixin支持多模态数据：
- **图像处理**：支持图像编码和网格化处理
- **视频处理**：支持视频帧提取和时间编码 ([images/video](https://github.com/volcengine/verl/blob/76f63cffa5081564d8fea93a1cb3ce8bd5bdcc39/verl/utils/dataset/rl_dataset.py#L90))
- **位置编码**：为多模态数据生成特殊的位置编码
- **注意力掩码**：处理多模态数据的注意力机制


**Q5: tool use data是如何处理的？**

A7: veRL支持工具调用的数据处理：
- **工具参数**：通过`tools_kwargs`传递工具调用参数
- **交互参数**：通过`interaction_kwargs`传递交互式训练参数
- [SGLang Rollout的tool use kwargs 处理](https://github.com/volcengine/verl/blob/76f63cffa5081564d8fea93a1cb3ce8bd5bdcc39/verl/workers/rollout/sglang_rollout/sglang_rollout.py#L1082-1099) - 工具参数处理
- [veRL自己的base tool类](https://github.com/volcengine/verl/blob/76f63cffa5081564d8fea93a1cb3ce8bd5bdcc39/verl/tools/base_tool.py) - 基础工具类实现

### 2.2 Rollout

数据准备完成后，进入序列生成阶段。这是RL训练中的"环境交互"环节，让当前策略生成响应序列，为后续的经验收集提供基础数据。

#### 2.2.1 核心类架构

veRL的rollout系统采用了模块化设计，主要包含以下几个重要类：

**类层次结构：**
```
BaseRollout (抽象基类)
├── SGLangRollout (SGLang推理引擎)
├── HFRollout (HuggingFace推理引擎)
├── VLLMRollout (vLLM推理引擎)
└── NaiveRollout (简单推理引擎)
```

#### 2.2.2 `BaseRollout` 抽象基类

**文件**: `verl/workers/rollout/base.py`

```python
class BaseRollout(ABC):
    """Rollout系统的抽象基类，定义了所有rollout引擎必须实现的接口"""
    
    @abstractmethod
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """生成序列的核心抽象方法
        
        Args:
            prompts: 包含输入提示的DataProto对象
            
        Returns:
            DataProto: 包含生成序列的DataProto对象
        """
        pass
```

**设计理念：**
- **统一接口**：所有rollout引擎都实现相同的`generate_sequences`接口
- **可扩展性**：支持不同的推理后端（SGLang、vLLM、HF等）
- **数据一致性**：使用`DataProto`作为统一的数据容器

#### 2.2.3 `SGLangRollout` 核心实现

**文件**: `verl/workers/rollout/sglang_rollout/sglang_rollout.py`

`SGLangRollout`是veRL中最核心的rollout实现，支持高效的多轮对话和工具调用。

**初始化流程**：
```python
class SGLangRollout(BaseRollout):
    def __init__(self, actor_module, config, tokenizer, model_hf_config, 
                 port=None, trust_remote_code=False, device_mesh=None, **kwargs):
        """SGLang Rollout引擎的初始化
        
        核心组件：
        1. 工具系统初始化 - 支持多轮对话中的工具调用
        2. 分布式环境设置 - 支持多节点推理
        3. 推理引擎初始化 - SGLang AsyncEngine
        4. 采样参数配置 - 控制生成行为
        """
        super().__init__()
        self.config = config
        self._device_mesh_cpu = device_mesh
        
        # 1. 工具系统初始化
        (self._tool_schemas, self._tool_map, self._tool_call_parser_type,
         self._sgl_tools, self._function_call_parser) = self._initialize_tools(config, tokenizer)
        
        # 2. 分布式环境初始化
        self._init_distributed_env(device_mesh_cpu=device_mesh, **kwargs)
        
        # 3. 推理引擎初始化
        self._init_inference_engine(trust_remote_code, actor_module, port)
        
        # 4. 采样参数初始化
        self._init_sampling_params(**kwargs)
```

**关键方法实现**：

**`generate_sequences()` - 主入口方法**：
```python
def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
    """序列生成的主入口方法
    
    根据配置选择不同的生成模式：
    - 多轮对话模式：使用异步请求级生成
    - 单轮对话模式：使用批处理级生成
    """
    if self.config.multi_turn.enable:
        return self._req_level_generate_sequences(prompts, **kwargs)
    else:
        return self._batch_level_generate_sequences(prompts, **kwargs)
```

**`_batch_level_generate_sequences()` - 批处理生成**：
```python
def _batch_level_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
    """批处理级序列生成 - 适用于单轮对话
    
    核心步骤：
    1. 数据预处理：提取token IDs和图像数据
    2. 采样参数配置：根据验证/训练模式设置不同参数
    3. SGLang引擎调用：异步生成序列
    4. 结果后处理：构建完整的序列数据
    5. 内存管理：清理KV缓存
    """
    # 1. 数据预处理
    idx = prompts.batch["input_ids"]  # (bs, prompt_length)
    attention_mask = prompts.batch["attention_mask"]
    position_ids = prompts.batch["position_ids"]
    
    # 2. 多模态数据处理
    if "multi_modal_data" in prompts.non_tensor_batch:
        sglang_inputs = []
        for raw_prompt_ids, multi_modal_data in zip(
            prompts.non_tensor_batch.pop("raw_prompt_ids"),
            prompts.non_tensor_batch.pop("multi_modal_data"),
        ):
            sglang_inputs.append({
                "prompt_token_ids": raw_prompt_ids,
                "multi_modal_data": multi_modal_data,
                "image_data": multi_modal_data.get("image", None),
            })
    
    # 3. 采样参数配置
    do_sample = prompts.meta_info.get("do_sample", True)
    is_validate = prompts.meta_info.get("validate", False)
    
    if not do_sample:
        # 确定性生成（贪婪解码）
        kwargs = dict(
            n=1, temperature=0, top_p=1, top_k=-1,
            max_new_tokens=self.config.response_length,
        )
    elif is_validate:
        # 验证模式参数
        kwargs = dict(
            top_k=self.config.val_kwargs.top_k,
            top_p=self.config.val_kwargs.top_p,
            temperature=self.config.val_kwargs.temperature,
            n=1,
        )
    
    # 4. SGLang引擎调用
    with self.update_sampling_params(**kwargs):
        if self._tp_rank == 0:  # 只在主进程中调用引擎
            loop = asyncio.get_event_loop()
            output = loop.run_until_complete(
                self._engine.async_generate(
                    prompt=None,
                    sampling_params=self.sampling_params,
                    return_logprob=True,
                    input_ids=idx_list,
                    image_data=image_list,
                )
            )
        else:
            output = None
        
        # 5. 结果广播到所有TP rank
        [output] = broadcast_pyobj(
            data=[output],
            rank=self._rank,
            dist_group=self._device_mesh_cpu["tp"].get_group(),
            src=self._device_mesh_cpu["tp"].mesh[0].item(),
        )
    
    # 6. 结果后处理
    out = _post_process_outputs(self.tokenizer, output)
    response = out[0].to(idx.device)
    rollout_log_probs = out[1].to(idx.device)
    
    # 7. 序列构建
    seq = torch.cat([idx, response], dim=-1)
    response_length = response.size(1)
    
    # 8. 位置ID和注意力掩码更新
    delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
    delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
    response_position_ids = position_ids[:, -1:] + delta_position_id
    position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
    
    # 9. 构建返回数据
    batch = TensorDict({
        "prompts": idx,
        "responses": response,
        "input_ids": seq,
        "rollout_log_probs": rollout_log_probs,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }, batch_size=batch_size)
    
    # 10. 内存清理
    if self.config.free_cache_engine and self._engine is not None:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._engine.flush_cache())
    
    return DataProto(batch=batch, non_tensor_batch=_non_tensor_batch)
```

#### 2.2.4 `AsyncRolloutRequest` 异步请求模型

**文件**: `verl/workers/rollout/schemas.py`

`AsyncRolloutRequest`是多轮对话中的核心数据结构，管理单个对话请求的完整生命周期。

**核心属性**：
```python
class AsyncRolloutRequest(BaseModel):
    """异步rollout请求的数据模型"""
    
    # 基础标识
    batch_data_id: int = 0          # 批次中的数据索引
    rollout_offset: int = 0         # 同一数据的多次采样偏移
    request_id: str                 # 唯一请求ID
    state: AsyncRolloutRequestStateEnum  # 请求状态
    
    # 对话内容
    messages: List[Message]         # 对话消息列表
    tool_schemas: Optional[List[OpenAIFunctionToolSchema]] = None  # 工具模式
    tools_kwargs: Dict[str, Any] = {}  # 工具参数
    
    # Token序列
    input_ids: List[int]            # 完整输入token序列
    prompt_ids: List[int]           # 提示部分token序列
    response_ids: List[int]         # 响应部分token序列
    
    # 掩码和位置信息
    attention_mask: List[int]       # 注意力掩码
    position_ids: List[int]         # 位置ID
    loss_mask: List[int]            # 损失计算掩码
    
    # 奖励和指标
    reward_scores: Dict[str, float] # 工具奖励分数
    metrics: Dict[str, List[Any]]   # 性能指标
    
    # 配置参数
    max_prompt_len: int             # 最大提示长度
    max_response_len: int = 8192    # 最大响应长度
    max_model_len: int = 32768      # 最大模型长度
```

**状态管理**：
```python
class AsyncRolloutRequestStateEnum(str, Enum):
    """异步rollout请求的状态枚举"""
    PENDING = "pending"      # 等待处理
    RUNNING = "running"      # 正在运行
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"        # 失败
    TOOL_CALLING = "tool_calling"  # 工具调用中
```

**关键方法**：

**`add_assistant_message()` - 添加助手消息**：
```python
def add_assistant_message(self, tokenizer: PreTrainedTokenizer, content: str, 
                         tool_calls: Optional[List[ToolCall]] = None) -> None:
    """添加助手的回复消息
    
    Args:
        tokenizer: 分词器
        content: 文本内容
        tool_calls: 工具调用列表（可选）
    """
    # 创建助手消息
    assistant_message = Message(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
    )
    self.messages.append(assistant_message)
    
    # 计算新消息的token
    if tool_calls is not None:
        # 包含工具调用的消息
        content = tokenizer.apply_chat_template(
            [*BASE_CHAT_HISTORY, *self.messages[-2:]],  # 用户消息 + 助手消息
            tools=[tool.model_dump() for tool in self.tool_schemas],
            add_generation_prompt=False,
            tokenize=False
        )
    else:
        # 普通文本消息
        content = tokenizer.apply_chat_template(
            [*BASE_CHAT_HISTORY, *self.messages[-2:]],
            tools=None,
            add_generation_prompt=False,
            tokenize=False
        )
    
    # 计算新增的token
    content_ids = tokenizer.encode(
        content[self.base_conv_wo_gen_prompt_end_pos:],
        add_special_tokens=False
    )
    
    # 更新输入序列
    self._update_input_ids(content_ids, attention_mask=True, loss_mask=True)
```

**`add_tool_response_messages()` - 添加工具响应**：
```python
def add_tool_response_messages(self, tokenizer: PreTrainedTokenizer, 
                              contents: list[str]) -> None:
    """添加工具响应消息
    
    Args:
        tokenizer: 分词器
        contents: 工具响应内容列表
    """
    if not contents:
        return
    
    # 添加工具响应消息
    self.messages.extend([
        Message(role="tool", content=content) 
        for content in contents
    ])
    
    # 计算工具响应的token
    content = tokenizer.apply_chat_template(
        [*BASE_CHAT_HISTORY, *self.messages[-len(contents):]],
        tools=[tool.model_dump() for tool in self.tool_schemas],
        add_generation_prompt=False,
        tokenize=False
    )
    content_ids = tokenizer.encode(
        content[self.base_conv_wo_gen_prompt_end_pos:],
        add_special_tokens=False
    )
    
    # 更新输入序列（工具响应不参与损失计算）
    self._update_input_ids(content_ids, attention_mask=True, loss_mask=False)
```

#### 2.2.5 `_req_level_generate_sequences()` 请求级生成

**多轮对话的核心实现**：
```python
def _req_level_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
    """请求级序列生成 - 适用于多轮对话和工具调用
    
    核心特点：
    1. 每个对话独立处理，支持不同的轮数
    2. 支持工具调用和异步执行
    3. 动态长度管理，避免等待最慢的对话
    4. 完整的对话状态跟踪
    """
    # 1. 预处理：将批处理数据转换为独立请求
    if self._tp_rank == 0:
        req_list = self._preprocess_prompt_to_async_rollout_requests(
            prompts,
            n=1 if is_validate else self.config.n,  # 验证时只生成1个候选，训练时生成n个
        )
        
        # 2. 并发处理所有请求
        loop = asyncio.get_event_loop()
        output_req_list = loop.run_until_complete(
            asyncio.gather(*[
                self._async_rollout_a_request(req, do_sample, is_validate, **kwargs) 
                for req in req_list
            ])
        ) # 最后生成候选的response list
        
def _batch_level_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
    """批量生成序列"""
    
    # 获取输入数据
    idx = prompts.batch["input_ids"]
    attention_mask = prompts.batch["attention_mask"]
    position_ids = prompts.batch["position_ids"]
    eos_token_id = prompts.meta_info["eos_token_id"]
    
    batch_size = idx.size(0)
    
    # 处理非张量数据
    non_tensor_batch = prompts.non_tensor_batch
    if "raw_prompt_ids" not in non_tensor_batch:
        non_tensor_batch["raw_prompt_ids"] = [self._pre_process_inputs(idx[i]) for i in range(batch_size)]
    
    # 准备输入数据
    sglang_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch["raw_prompt_ids"]]
    
    # 调用生成引擎
    output = self._engine.async_generate(input_ids=sglang_inputs, **kwargs)
    
    # 处理生成的输出
    batched_output_token_ids, batched_logprobs = self._post_process_outputs(output)
    
    # 拼接生成的输出与原始输入
    response = torch.cat([idx, batched_output_token_ids], dim=-1)
    
    # 更新注意力掩码和位置编码
    response_attention_mask = self.get_response_mask(response, eos_token_id)
    attention_mask = torch.cat([attention_mask, response_attention_mask], dim=-1)
    position_ids = self.update_position_ids(position_ids, response)
    
    # 返回生成的批次数据
    batch = TensorDict({
        "prompts": idx,
        "responses": response,
        "input_ids": response,
        "attention_mask": attention_mask,
        "position_ids": position_ids
    }, batch_size=batch_size)
    
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

```

**并发处理的优势：**
- **资源利用率高**: 不需要等待最慢的请求
- **支持复杂交互**: 每个对话可以有不同的工具调用序列
- **内存效率**: 避免为所有可能的轮数预分配内存

### 2.3 经验处理阶段 (Make Experience)

序列生成完成后，进入经验处理阶段。这一阶段将生成的序列转换为强化学习算法所需的训练数据，包括重新计算概率、计算奖励、评估参考策略等关键步骤。

**什么是Experience？**
在强化学习中，experience指的是智能体与环境交互产生的数据，包括状态、动作、奖励、下一状态等。在LLM训练中，experience包括输入prompt、生成的response、获得的reward，以及用于策略更新的各种概率和优势值。

**重新计算 log_prob**
这是对应的重新计算log_prob代码，在 `verl/trainer/ppo/ray_trainer.py:995-1030`：

**为什么要重新计算log_prob？**
在生成阶段，我们得到的log_prob是用于采样的，不够精确（通常是fp16/bf16），这是因为在rollout我们的目标是"快"，快速生成old_log_prob。而在训练阶段，我们需要更精确的概率计算用于PPO等RL算法，如果ratio = exp(new_log_prob - old_log_prob)差别太大，就会导致clip和KL约束失真，从而破坏训练的稳定性。此外，生成和训练可能使用不同的数值精度或计算图，重新计算确保一致性。

```python
# 重新计算 old_log_probs - 这是PPO算法的关键输入
with _timer("old_log_prob", timing_raw):
    # 使用当前策略重新计算生成序列的概率
    # 这个概率将作为PPO中的"old policy"概率
    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
    batch = batch.union(old_log_prob)  # 将结果合并到batch中
```

**Reward compute**
veRL 支持灵活的奖励计算策略，可以结合 model-based reward 和 rule-based reward，适应不同类型的任务需求。对应的代码，在 `verl/trainer/ppo/ray_trainer.py:980-993`。

```python
with _timer("reward", timing_raw):
    # 方式1：奖励模型计算（可选）- 使用训练好的奖励模型评估响应质量
    if self.use_rm:
        reward_tensor = self.rm_wg.compute_rm_score(batch)
        batch = batch.union(reward_tensor)
    
    # 方式2：规则/函数奖励计算 - 使用任务特定的评估函数
    if self.config.reward_model.launch_reward_fn_async:
        # 异步计算：适合耗时的奖励函数（如代码执行、数学验证）
        future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
    else:
        # 同步计算：适合简单快速的奖励函数
        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
```

veRL提供了多种reward计算方式，可以根据任务需求灵活选择：

1. **内置Reward函数**

veRL内置了多种预定义的reward函数，支持常见的数据集和任务类型：

#### 数学推理任务
```python
# 在配置中指定数据源
data:
  reward_fn_key: "data_source"  # 用于标识数据源类型的键

# 支持的数据源包括：
# - "openai/gsm8k": GSM8K数学问题
# - "hendrycks_math": 数学推理数据集
# - "hiyouga/geometry3k": 几何问题
```

#### 代码生成任务
```python
# 代码执行reward
data:
  reward_fn_key: "data_source"
  
# 支持的数据源：
# - "mbpp": Python代码执行
# - "humaneval": HumanEval代码评估
```

#### 多模态任务
```python
# 视觉-语言任务
data:
  reward_fn_key: "data_source"
  
# 支持的数据源：
# - "llava_v1_5_mix": LLaVA多模态对话
```

* **自定义Reward函数**

veRL支持完全自定义的reward函数，这是最灵活的方式：

#### 配置文件设置
```yaml
# 在配置文件中指定自定义reward函数
custom_reward_function:
  # 自定义reward函数文件的路径
  path: "/path/to/your/custom_reward.py"
  
  # reward函数在文件中的名称（默认为compute_score）
  name: "my_custom_reward"
  
  # 传递给reward函数的额外参数
  reward_kwargs:
    threshold: 0.8
    penalty_factor: 0.5
```

#### 自定义Reward函数实现

**基本格式**：
```python
# /path/to/your/custom_reward.py
def my_custom_reward(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    自定义reward函数
    
    Args:
        data_source (str): 数据源标识符
        solution_str (str): 模型生成的响应文本
        ground_truth (str): 标准答案
        extra_info (dict, optional): 额外的信息
        **kwargs: 从配置文件中传递的额外参数
    
    Returns:
        float: reward分数 (0.0-1.0)
    """
    # 你的reward计算逻辑
    score = compute_custom_score(solution_str, ground_truth)
    return score
```

**复杂Reward函数示例**：
```python
def advanced_reward_function(data_source, solution_str, ground_truth, extra_info=None, 
                           threshold=0.8, penalty_factor=0.5):
    """
    高级自定义reward函数示例
    
    功能：
    1. 基础准确性评分
    2. 格式规范性检查
    3. 逻辑一致性验证
    4. 可配置的惩罚机制
    """
    
    # 1. 基础准确性评分
    accuracy_score = compute_accuracy(solution_str, ground_truth)
    
    # 2. 格式规范性检查
    format_score = check_format_correctness(solution_str, data_source)
    
    # 3. 逻辑一致性验证
    logic_score = verify_logic_consistency(solution_str, extra_info)
    
    # 4. 综合评分
    final_score = (accuracy_score * 0.6 + 
                   format_score * 0.2 + 
                   logic_score * 0.2)
    
    # 5. 应用惩罚机制
    if final_score < threshold:
        final_score *= penalty_factor
    
    return final_score

def compute_accuracy(solution, ground_truth):
    """计算答案准确性"""
    # 实现你的准确性计算逻辑
    if solution.strip().lower() == ground_truth.strip().lower():
        return 1.0
    return 0.0

def check_format_correctness(solution, data_source):
    """检查格式正确性"""
    if data_source == "openai/gsm8k":
        # GSM8K特定的格式检查
        if "\\boxed" in solution:
            return 1.0
        return 0.5
    return 1.0

def verify_logic_consistency(solution, extra_info):
    """验证逻辑一致性"""
    # 实现逻辑一致性检查
    return 1.0
```

**返回字典格式的Reward函数**：
```python
def detailed_reward_function(data_source, solution_str, ground_truth, extra_info=None):
    """
    返回详细信息的reward函数
    
    Returns:
        dict: 包含多个评分维度的字典
    """
    accuracy = compute_accuracy(solution_str, ground_truth)
    format_score = check_format(solution_str)
    logic_score = check_logic(solution_str)
    
    return {
        "score": accuracy * 0.7 + format_score * 0.2 + logic_score * 0.1,
        "accuracy": accuracy,
        "format_score": format_score,
        "logic_score": logic_score,
        "detailed_feedback": generate_feedback(solution_str, ground_truth)
    }
```

* **Reward Manager类型**

veRL提供了多种Reward Manager来处理不同的计算需求：

#### NaiveRewardManager（默认）
```yaml
reward_model:
  reward_manager: "naive"  # 默认的reward管理器
```

#### BatchRewardManager
```yaml
reward_model:
  reward_manager: "batch"  # 批处理reward管理器
```

#### PrimeRewardManager
```yaml
reward_model:
  reward_manager: "prime"  # Prime算法专用reward管理器
```

#### DAPORewardManager
```yaml
reward_model:
  reward_manager: "dapo"  # DAPO算法专用reward管理器
```

* **异步Reward计算**

对于耗时的reward计算（如代码执行、数学验证），veRL支持异步计算：

```yaml
reward_model:
  # 启用异步reward计算
  launch_reward_fn_async: True
  
  # 沙箱融合配置（用于代码执行等）
  sandbox_fusion:
    url: "http://localhost:8000"  # 沙箱服务URL
    max_concurrent: 64  # 最大并发数
```

**异步Reward函数示例**：
```python
def async_code_execution_reward(data_source, solution_str, ground_truth, extra_info=None):
    """
    异步代码执行reward函数
    
    适用于需要执行代码验证的任务
    """
    import asyncio
    import aiohttp
    
    async def execute_code(code):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8000/execute",
                json={"code": code, "test_cases": extra_info.get("test_cases", [])}
            ) as response:
                result = await response.json()
                return result["passed_tests"] / result["total_tests"]
    
    # 提取代码部分
    code = extract_code_from_solution(solution_str)
    
    # 异步执行代码
    loop = asyncio.get_event_loop()
    score = loop.run_until_complete(execute_code(code))
    
    return score
```

* **混合Reward策略**

veRL支持结合多种reward来源的混合策略：

```python
def hybrid_reward_function(data_source, solution_str, ground_truth, extra_info=None):
    """
    混合reward函数示例
    
    结合：
    1. 规则基础评分
    2. 模型评分（如果可用）
    3. 人工标注评分（如果可用）
    """
    
    # 1. 规则基础评分
    rule_score = compute_rule_based_score(solution_str, ground_truth)
    
    # 2. 模型评分（从extra_info中获取）
    model_score = extra_info.get("model_score", 0.0)
    
    # 3. 人工标注评分（从extra_info中获取）
    human_score = extra_info.get("human_score", 0.0)
    
    # 4. 加权组合
    weights = {
        "rule": 0.4,
        "model": 0.4,
        "human": 0.2
    }
    
    final_score = (rule_score * weights["rule"] + 
                   model_score * weights["model"] + 
                   human_score * weights["human"])
    
    return final_score
```

* **Reward函数调试和监控**

#### 调试输出
```python
def debug_reward_function(data_source, solution_str, ground_truth, extra_info=None):
    """带调试信息的reward函数"""
    
    print(f"[DEBUG] Data source: {data_source}")
    print(f"[DEBUG] Solution: {solution_str}")
    print(f"[DEBUG] Ground truth: {ground_truth}")
    print(f"[DEBUG] Extra info: {extra_info}")
    
    score = compute_score(solution_str, ground_truth)
    print(f"[DEBUG] Computed score: {score}")
    
    return score
```

#### 性能监控
```python
import time

def monitored_reward_function(data_source, solution_str, ground_truth, extra_info=None):
    """带性能监控的reward函数"""
    
    start_time = time.time()
    
    # 执行reward计算
    score = compute_complex_reward(solution_str, ground_truth)
    
    end_time = time.time()
    computation_time = end_time - start_time
    
    # 记录性能指标
    print(f"[PERF] Reward computation time: {computation_time:.4f}s")
    
    return score
```

* **Reward最佳实践**

#### 配置管理
```yaml
# 推荐的项目结构
project/
├── configs/
│   ├── reward_functions/
│   │   ├── math_reward.py
│   │   ├── code_reward.py
│   │   └── custom_reward.py
│   └── training_config.yaml
├── scripts/
│   └── run_training.sh
└── README.md
```

#### 错误处理
```python
def robust_reward_function(data_source, solution_str, ground_truth, extra_info=None):
    """健壮的reward函数，包含错误处理"""
    
    try:
        # 输入验证
        if not solution_str or not ground_truth:
            print(f"[WARNING] Invalid input: solution='{solution_str}', ground_truth='{ground_truth}'")
            return 0.0
        
        # 执行reward计算
        score = compute_score(solution_str, ground_truth)
        
        # 输出验证
        if not isinstance(score, (int, float)) or score < 0 or score > 1:
            print(f"[WARNING] Invalid score: {score}")
            return 0.0
        
        return score
        
    except Exception as e:
        print(f"[ERROR] Reward computation failed: {e}")
        return 0.0  # 返回默认分数
```

#### 测试验证
```python
def test_reward_function():
    """测试reward函数"""
    
    # 测试用例
    test_cases = [
        {
            "data_source": "openai/gsm8k",
            "solution": "The answer is \\boxed{42}",
            "ground_truth": "42",
            "expected_score": 1.0
        },
        {
            "data_source": "openai/gsm8k", 
            "solution": "The answer is 42",
            "ground_truth": "42",
            "expected_score": 0.5  # 格式不正确
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        score = my_custom_reward(**test_case)
        assert abs(score - test_case["expected_score"]) < 0.01, \
            f"Test case {i} failed: expected {test_case['expected_score']}, got {score}"
    
    print("All tests passed!")
```

通过以上方式，你可以根据具体任务需求灵活选择和实现reward函数，充分利用veRL提供的reward计算框架。

**奖励设计的考虑：**
- **模型奖励**: 通用但可能有偏差，适合对话、摘要等主观任务
- **规则奖励**: 准确但覆盖有限，适合数学、代码等客观任务
- **混合奖励**: 结合两者优势，在实际应用中很常见

**reference策略评估**
这是对应的reference策略评估代码，在 `verl/trainer/ppo/ray_trainer.py:1031-1038`：

**为什么需要参考策略？**
参考策略（Reference Policy）通常是训练前的原始模型，用于计算KL散度约束，防止新策略偏离原始行为太远。这在RLHF中特别重要，确保模型在获得高奖励的同时保持合理的语言行为。

```python
if self.use_reference_policy:
    with _timer("ref", timing_raw):
        if not self.ref_in_actor:
            # 独立的参考策略模型 - 需要额外的GPU内存，但更精确
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch) # 计算reference model的log_prob
        else:
            # 使用Actor中的LoRA基模型作为参考 - 内存高效，适合LoRA微调
            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
        batch = batch.union(ref_log_prob)
```


**Advantage函数计算**
Advantage函数衡量某个动作相对于平均水平的好坏程度。正值表示该动作比平均好，负值表示比平均差。这是策略梯度算法的核心，帮助模型学习更好的行为。这是对应的advantage函数计算代码，在 `verl/trainer/ppo/ray_trainer.py:1062-1076`：

```python
batch = compute_advantage(
    batch,
    adv_estimator=self.config.algorithm.adv_estimator,    # 优势估计方法：GAE、GRPO等
    gamma=self.config.algorithm.gamma,                    # 折扣因子：未来奖励的权重
    lam=self.config.algorithm.lam,                        # GAE参数：偏差vs方差的权衡
    num_repeat=self.config.actor_rollout_ref.rollout.n,   # 每个prompt的采样数
    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,     # GRPO特殊归一化
    multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable, # 多轮对话支持
    config=self.config.algorithm,
)
```

**不同优势估计方法：**
- **GAE**: 使用价值函数估计，偏差小但需要额外的Critic网络
- **GRPO**: 使用群组统计，无需价值函数，适合数学推理任务
- **REINFORCE**: 直接使用奖励，简单但方差大

### 2.4 模型更新阶段 (Train)

经验处理完成后，进入最终的模型更新阶段。这一阶段使用收集到的经验数据来改进策略，是强化学习训练的核心环节。

**训练阶段的设计理念：**
veRL 采用分阶段训练策略。在训练初期，可以只训练 Critic 来稳定价值估计，然后再开始 Actor 训练。这种预热策略在复杂任务中特别有效。

#### 2.4.1 Critic 更新 (仅GAE算法)

**为什么需要Critic？**
Critic网络负责估计状态价值，这个估计的准确性直接影响优势函数的质量。通过独立更新Critic，可以使用更多的训练步骤和不同的学习率，提高价值估计的准确性。

**主控制流程**
这是对应的Critic更新代码，在 `verl/trainer/ppo/ray_trainer.py:1078-1083`：

```python
if self.use_critic:  # 只有GAE算法需要Critic
    with _timer("update_critic", timing_raw):
        # Critic更新使用回归损失，目标是准确预测累积奖励
        critic_output = self.critic_wg.update_critic(batch)
    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
    metrics.update(critic_output_metrics)
```

**Critic训练细节：**
- **损失函数**: 通常使用MSE回归损失
- **目标值**: 实际获得的累积奖励
- **更新频率**: 可能比Actor更频繁，以提高估计准确性

#### 2.4.2 Actor 更新

**Critic warmup机制：**
在训练初期，Critic的价值估计可能不准确，导致Actor更新的方向错误。通过设置预热期，让Critic先稳定下来，再开始Actor训练。

**主控制流程**
这是对应的Actor更新代码，在 `verl/trainer/ppo/ray_trainer.py:1085-1092`：

```python
if self.config.trainer.critic_warmup <= self.global_steps:  # 预热期检查
    with _timer("update_actor", timing_raw):
        # 添加多轮对话的元信息，用于特殊处理
        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
        # 执行PPO策略更新
        actor_output = self.actor_rollout_wg.update_actor(batch)
    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
    metrics.update(actor_output_metrics)
```

**PPO更新过程：**
1. 计算新旧策略的概率比值
2. 使用advantage加权策略梯度
3. 应用PPO的clip机制防止更新过大
4. 可选地添加KL散度惩罚

#### 2.4.3 FSDP Actor 更新实现

**内存管理的复杂性：**
大模型训练的一个挑战是内存管理。veRL支持参数卸载（offloading），将暂时不用的参数移到CPU或存储，需要时再加载回GPU。这个过程需要精心协调以避免性能损失。

**FSDP Actor 更新实现**
这是对应的FSDP Actor更新实现代码，在 `verl/workers/fsdp_workers.py:575-625`：

```python
def update_actor(self, data: DataProto):
    # 数据预处理：移动到CPU以支持各种硬件配置
    data = data.to('cpu')
    
    # 内存优化：按需加载模型参数
    if self._is_offload_param:
        # 将FSDP模型参数从CPU/存储加载到GPU
        load_fsdp_model_to_gpu(self.actor_module_fsdp)
    if self._is_offload_optimizer:
        # 将优化器状态加载到GPU
        load_fsdp_optimizer(optimizer=self.actor_optimizer, 
                           device_id=get_torch_device().current_device())
    
    # 执行实际的策略更新
    output = self.actor.update_policy(data=data)
```

**FSDP的优势：**
- **内存效率**: 参数分片减少单GPU内存需求
- **计算效率**: 避免不必要的参数复制
- **可扩展性**: 支持任意数量的GPU

## 多轮对话特殊处理

### 配置文件

**文件**: `examples/sglang_multiturn/config/tool_config/`

多轮对话需要特殊的工具配置和状态管理：

1. **工具集成**: 支持计算器、搜索等工具调用
2. **状态保持**: SGLang 引擎维护多轮对话状态
3. **掩码处理**: 使用 `loss_mask` 进行多轮损失计算
4. **异步处理**: 支持并发的多轮对话生成

### 关键实现

**文件**: `verl/trainer/ppo/ray_trainer.py:129-178`

```python
def apply_kl_penalty(data: DataProto, kl_ctrl, kl_penalty="kl", multi_turn=False):
    if multi_turn:
        # 多轮对话使用 loss_mask 而不是 response_mask
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

```

## 性能优化点

### 1. 异步Rollout

**文件**: `verl/workers/rollout/async_server.py`

- 支持异步推理，提高GPU利用率
- 并发处理多个生成请求

### 2. 权重分片管理

**文件**: `verl/workers/sharding_manager/`

- 在推理和训练引擎间高效转换模型权重
- 减少内存占用和传输开销

### 3. 序列长度平衡

**文件**: `verl/trainer/ppo/ray_trainer.py:862-875`

- 自动平衡不同GPU间的token数量
- 提高训练效率

---

## GRPO 算法示例: 无价值函数的群组相对策略优化

### GRPO 工作原理

**文件**: `verl/trainer/ppo/core_algos.py:166-196`

GRPO (Group Relative Policy Optimization) 是一种无需价值函数的强化学习算法，特别适用于数学推理等需要多次采样的任务：

1. **群组采样**: 对每个prompt生成多个响应（如n=5）
2. **相对奖励**: 计算每组内响应的平均奖励作为基线
3. **优势计算**: `advantage = reward - group_mean`
4. **无Critic**: 不需要训练独立的价值网络

### 核心算法实现

**GRPO优势计算**: `verl/trainer/ppo/core_algos.py:166-196`

```python
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
):
    """GRPO优势计算 - 基于群组相对奖励"""
    scores = token_level_rewards.sum(dim=-1)  # 序列级奖励

    # 按prompt索引分组计算统计量
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    for i in range(scores.shape[0]):
        id2score[index[i]].append(scores[i])

    # 计算每组的均值和标准差
    for idx in id2score:
        if len(id2score[idx]) > 1:
            id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            id2std[idx] = torch.std(torch.tensor(id2score[idx]))
        else:
            id2mean[idx] = torch.tensor(0.0)
            id2std[idx] = torch.tensor(1.0)

    # 计算相对优势
    for i in range(scores.shape[0]):
        if norm_adv_by_std_in_grpo:
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        else:
            scores[i] = scores[i] - id2mean[index[i]]

    scores = scores.unsqueeze(-1) * response_mask
    return scores, scores

```

### 配置差异对比

### PPO vs GRPO 关键配置

| 配置项 | PPO | GRPO |
| --- | --- | --- |
| `algorithm.adv_estimator` | `gae` | `grpo` |
| `actor_rollout_ref.rollout.n` | `1` | `5` (或更大) |
| `actor_rollout_ref.actor.use_kl_loss` | `false` | `true` |
| `algorithm.use_kl_in_reward` | `true` | `false` |
| Critic 组件 | **需要** | **不需要** |

### 算法选择逻辑

**文件**: `verl/trainer/ppo/ray_trainer.py:323-346`

```python
if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
    self.use_critic = True  # 使用价值函数
elif self.config.algorithm.adv_estimator in [
    AdvantageEstimator.GRPO,
    AdvantageEstimator.GRPO_PASSK,
    AdvantageEstimator.REINFORCE_PLUS_PLUS,
    # ...
]:
    self.use_critic = False  # 无价值函数

```

### GRPO 性能优势

1. **内存节省**: 无需训练Critic网络
2. **训练稳定**: 基于群组统计，减少方差
3. **适用场景**: 特别适合数学推理、代码生成等需要多次采样的任务
4. **实现简单**: 避免value function的复杂调参

### 高级扩展: DrGRPO

**文件**: `docs/algo/grpo.md`

DrGRPO 解决了GRPO的长度偏差问题：

```bash
# DrGRPO配置差异
actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm  # 关闭序列维度平均
actor_rollout_ref.actor.use_kl_loss=false                      # 不使用KL loss
algorithm.norm_adv_by_std_in_grpo=false                        # 关闭标准差归一化

```

## 总结

veRL 的 VLM multi-turn RL 工作流程通过模块化设计实现了：

1. **高效推理**: SGLang 提供快速的多轮对话生成
2. **分布式训练**: FSDP 支持大模型的并行训练
3. **灵活奖励**: 支持多种奖励函数组合
4. **异步处理**: 提高GPU利用率和训练效率
5. **多模态支持**: 原生支持视觉-语言模型
6. **算法多样性**: 支持PPO、GRPO、RLOO等多种RL算法

整个系统设计既保证了性能，又维持了良好的可扩展性和易用性。

flowchart TB
subgraph Init["初始化阶段"]
direction TB
A[启动RayPPOTrainer] --> B1[创建资源池]
B1 --> B2[初始化分布式Workers]
B2 --> C1[构建Hybrid Engine]
C1 --> D1[数据预处理]
end

```
subgraph TrainLoop["训练循环"]
    direction TB
    E[数据加载] --> F[序列生成Rollout]
    F --> G[经验处理]
    G --> H[模型更新]
    H --> E
end

Init --> TrainLoop

subgraph Detail["详细流程"]
    direction LR
    subgraph Workers["Workers初始化详情"]
        direction TB
        W1[ActorRolloutWorker] --> W4[FSDP训练引擎]
        W2[CriticWorker] --> W4
        W3[RewardModelWorker] --> W4
        W4 --> W5[SGLang推理引擎]
    end

    subgraph Rollout["序列生成详情"]
        direction TB
        F1[准备生成数据] --> F2[SGLang异步生成]
        F2 --> F3[多轮对话处理]
    end

    subgraph Experience["经验处理详情"]
        direction TB
        G1[重新计算Log Prob] --> G2[计算Reward]
        G2 --> G3[GRPO优势计算]
    end

    subgraph Update["模型更新详情"]
        direction TB
        H1[加载FSDP模型参数] --> H2[计算策略梯度]
        H2 --> H3[应用GRPO更新]
        H3 --> H4[参数分片保存]
    end
end

B2 --> Workers
F --> Rollout
G --> Experience
H --> Update

```

flowchart TB
subgraph Init["初始化阶段"]
direction TB
A[启动RayPPOTrainer] --> B1[创建资源池]
B1 --> B2[初始化分布式Workers]
B2 --> C1[构建Hybrid Engine]
C1 --> D1[数据预处理]
end

```
subgraph TrainLoop["训练循环"]
    direction TB
    E[数据加载] --> F1[准备生成数据]
    F1 --> F2[SGLang异步生成]
    F2 --> F3[多轮对话处理]
    F3 --> G1[重新计算Log Prob]
    G1 --> G2[计算Reward]
    G2 --> G3[GRPO优势计算]
    G3 --> H1[加载FSDP模型参数]
    H1 --> H2[计算策略梯度]
    H2 --> H3[应用GRPO更新]
    H3 --> H4[参数分片保存]
    H4 --> E
end

Init --> TrainLoop

subgraph Workers["Workers初始化详情"]
    direction TB
    W1[ActorRolloutWorker] --> W4[FSDP训练引擎]
    W2[CriticWorker] --> W4
    W3[RewardModelWorker] --> W4
    W4 --> W5[SGLang推理引擎]
end

B2 --> Workers

```