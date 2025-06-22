# Verl Multi-turn RL Code Walk Through

承蒙社区厚爱，基于 SGLang 的 Agentic RL 如火如荼。考虑到各大 RL 框架的代码更新频率极高而社区二次开发需求巨大，我们选择以 verl 出发，分析其 end to end mutli-turn RL 训练的全过程。整体上，我们希望覆盖所有重要的 class 以及函数，更细粒度的代码不再展开。我们的写作风格希望能够 follow SGLang 的 code-walk-through：

[SGLang Code Walk Through](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/code-walk-through/readme-CN.md)


## 数据预处理

### 预处理入口

以 GSM8K 为例，预处理脚本是 `examples/data_preprocess/gsm8k_multiturn_w_tool.py`。整个脚本只做了经典的 huggingface datasets mapping，核心逻辑如下：

1. 加载 openai/gsm8k 原始数据集（train/test）。
2. 对每条原始数据，生成带有工具调用要求的 prompt（比如在 user turn 强调模型可以调用 `calc_gsm8k_reward` 工具）。
3. 同样对于每条原始数据，解析答案；将 ground truth 写入 extra_info 字段。
4. 存储为 parquet 文件，分别保留为 train.parquet 和 test.parquet，默认路径为 `~/data/gsm8k/`。

## 启动训练

一个典型的启动命令如下：

```bash
# now 用于生成实验启动的时间尾缀，避免重复启动实验时覆盖已有 wandb log
function now() {
    date '+%Y-%m-%d-%H-%M'
}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nohup bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh trainer.experiment_name=qwen2.5-3b_rm-gsm8k-sgl-multiturn-$(now) > logs/gsm8k-$(now).log 2>&1 &
```

### 脚本配置

verl 的各项参数实属复杂，我们会单独编写文档来分享 SGLang RL 小组对 verl 各类参数的理解。在这部分，我们想要格外强调的是 verl 各类 config 的覆盖关系。

## 配置文件分析

### 基础配置结构

这个配置文件采用了**分层配置**的设计模式：
- **基础配置**：`ppo_trainer.yaml` 提供PPO训练的默认配置
- **任务配置**：`gsm8k_multiturn_grpo.yaml` 继承基础配置并添加GSM8K特定设置
- **工具配置**：`gsm8k_tool_config.yaml` 定义数学计算工具
- **运行时覆盖**：shell脚本中的参数会覆盖yaml中的默认值

### 2. 核心训练参数

#### 2.1 算法配置
```bash
algorithm.adv_estimator=grpo  # 使用GRPO优势估计器
algorithm.use_kl_in_reward=False  # 不在奖励中应用KL惩罚
```

#### 2.2 数据配置
```bash
data.train_batch_size=256  # 训练批次大小
data.max_prompt_length=1024  # 最大prompt长度
data.max_response_length=1024  # 最大响应长度
data.filter_overlong_prompts=True  # 过滤过长prompt
data.truncation='error'  # 长度超限时报错
data.return_raw_chat=True  # 返回原始对话格式
```

#### 2.3 模型配置
```bash
actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct  # 使用Qwen2.5-3B模型
actor_rollout_ref.model.use_remove_padding=True  # 移除padding token
actor_rollout_ref.model.enable_gradient_checkpointing=True  # 启用梯度检查点
```

### 3. 多轮对话配置

#### 3.1 多轮设置
```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      enable: True  # 启用多轮对话
      max_turns: 5  # 最大对话轮次
```

#### 3.2 工具集成
```bash
actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml"
```

工具配置定义了`calc_gsm8k_reward`函数，用于：
- 验证模型答案的正确性
- 返回1.0（正确）或0.0（错误）的奖励

### 4. 训练优化配置

#### 4.1 批次大小优化
```bash
actor_rollout_ref.actor.ppo_mini_batch_size=256  # PPO微批次大小
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32  # 每GPU微批次大小
actor_rollout_ref.rollout.n=16  # 每个prompt生成16个响应
```

#### 4.2 内存管理
```bash
actor_rollout_ref.actor.fsdp_config.param_offload=False  # Actor参数不卸载
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False  # Actor优化器不卸载
actor_rollout_ref.ref.fsdp_config.param_offload=True  # Reference模型参数卸载
actor_rollout_ref.rollout.gpu_memory_utilization=0.5  # GPU内存利用率50%
```

#### 4.3 并行策略
```bash
actor_rollout_ref.rollout.tensor_model_parallel_size=2  # 张量并行度
actor_rollout_ref.rollout.name=sglang  # 使用SGLang推理引擎
```

### 5. 损失函数配置

#### 5.1 KL损失
```bash
actor_rollout_ref.actor.use_kl_loss=True  # 启用KL损失
actor_rollout_ref.actor.kl_loss_coef=0.001  # KL损失系数
actor_rollout_ref.actor.kl_loss_type=low_var_kl  # 低方差KL损失
```

#### 5.2 其他损失
```bash
actor_rollout_ref.actor.entropy_coeff=0  # 熵损失系数为0
actor_rollout_ref.actor.optim.lr=1e-6  # 学习率
```

### 6. 训练监控配置

#### 6.1 日志记录
```bash
trainer.logger=['console','wandb']  # 同时输出到控制台和WandB
trainer.project_name='gsm8k_async_rl'  # WandB项目名
trainer.experiment_name='qwen2.5-3b_function_rm-gsm8k-sgl-multi-w-tool-verify-n16'  # 实验名
```

#### 6.2 检查点和验证
```bash
trainer.save_freq=-1  # 不保存检查点
trainer.test_freq=20  # 每20步验证一次
trainer.total_epochs=15  # 总训练轮次
trainer.critic_warmup=0  # 无Critic预热
```

### 7. 硬件资源配置

```bash
trainer.n_gpus_per_node=8  # 每节点8个GPU
trainer.nnodes=1  # 1个节点
```

### 8. 配置特点总结

1. **多轮对话优化**：专门针对数学问题的多轮推理设计
2. **工具集成**：通过`calc_gsm8k_reward`工具实现答案验证
3. **内存效率**：通过参数卸载和梯度检查点优化内存使用
4. **分布式训练**：支持FSDP和SGLang的分布式推理
5. **GRPO算法**：使用GRPO优势估计器，适合多轮对话场景
6. **高吞吐量**：通过微批次和并行策略实现高效训练

这个配置体现了Verl框架在**多轮工具调用RLHF**方面的先进设计，特别适合需要多步推理的数学问题训练。

### 1.3 工具配置

- `examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml`
  - 定义了 `calc_gsm8k_reward` 工具的 schema，供多轮 RL 训练时自动调用。

---

## 2. 配置文件结构

- `gsm8k_multiturn_grpo.yaml` 继承自 `ppo_trainer.yaml`，并开启 multi_turn、指定工具配置路径。
- 训练相关参数（如 batch size、epoch、KL loss 等）可通过 shell 脚本直接覆盖。

---

## 3. 数据加载与 RL 训练主流程

- 训练主入口：`verl.trainer.main_ppo.py` 的 `main()` 和 `run_ppo()`。
- 数据加载：`create_rl_dataset` 负责读取 parquet，构造 RLHFDataset。
- 工具参数、ground truth、reward 机制等均在数据预处理阶段写入，训练时自动解析。

---

**这样梳理后，用户可以一目了然地从数据预处理、配置、到训练启动的全流程，便于理解和复现。**  
如需更细致的代码级流程，可继续补充每个函数的调用链和关键实现。

## 系统启动与初始化 Dataflow

### 1.1 主入口函数调用链路

主入口函数 `run_ppo` 负责整个 RL 训练流程的启动和协调：

```python
# 1. 主入口点
def run_ppo(config) -> None:
    # 1.1 初始化 Ray 集群，配置 CPU 资源和运行时环境变量。
    ray.init(
        runtime_env={"env_vars": {...}},
        num_cpus=config.ray_init.num_cpus,
    )
    
    # 1.2 创建远程 TaskRunner 实例。TaskRunner 是 Ray 中的一个远程 actor，用于执行主要的训练任务。
    runner = TaskRunner.remote()
    # 1.3 异步执行远程任务 runner.run()，并等待其完成。
    ray.get(runner.run.remote(config))
```

### 1.2 `TaskRunner.run()` 函数详细流程

`TaskRunner.run()` 函数是系统初始化的核心，它负责加载配置、初始化模型、Tokenizer、奖励函数、数据集以及训练器，并最终启动训练：

```python
def run(self, config):
    # 2.1 配置解析和验证：使用 OmegaConf 解析并验证传入的配置对象。
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)
    
    # 2.2 模型下载：将模型文件从远程路径复制到本地。
    local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))
    
    # 2.3 Tokenizer 初始化：根据下载的模型路径初始化 Hugging Face Tokenizer 和 Processor。
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
    
    # 2.4 Worker 类型选择：根据配置中的 actor 策略（FSDP 或 Megatron）选择相应的 Worker 类。
    if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker
        actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup
    elif config.actor_rollout_ref.actor.strategy == "megatron":
        from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker
        actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
        ray_worker_group_cls = NVMegatronRayWorkerGroup
    
    # 2.5 角色映射配置：定义不同角色（ActorRollout, Critic）到 Ray 远程 Worker 类的映射。
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(actor_rollout_cls),
        Role.Critic: ray.remote(CriticWorker),
    }
    
    # 2.6 资源池配置：定义 Ray 资源池的规格和角色到资源池的映射。
    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
    }
    
    # 2.7 奖励函数初始化：加载用于训练和验证的奖励模型。
    reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
    val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {}))
    
    # 2.8 数据集创建：创建训练和验证数据集，以及训练数据采样器。
    train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
    val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
    train_sampler = create_rl_sampler(config.data, train_dataset)
    
    # 2.9 资源池管理器创建：实例化 ResourcePoolManager 管理 Ray 资源。
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    
    # 2.10 训练器创建和初始化：创建 RayPPOTrainer 实例，传入所有必要的配置和组件。
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=collate_fn,
        train_sampler=train_sampler,
        device_name=config.trainer.device,
    )
    
    # 2.11 初始化 Workers：通过 trainer 调用初始化 Ray Workers。
    trainer.init_workers()
    
    # 2.12 开始训练：启动 PPO 训练循环。
    trainer.fit()
```

-----

## 2\. `RayPPOTrainer` 初始化详细流程

### 2.1 `RayPPOTrainer.__init__()` 函数

`RayPPOTrainer` 的构造函数主要负责设置训练器的基本配置、功能标志、数据加载器以及进行配置验证：

```python
def __init__(self, config, tokenizer, role_worker_mapping, resource_pool_manager, ...):
    # 3.1 基础配置设置：保存传入的配置和组件实例。
    self.config = config
    self.tokenizer = tokenizer
    self.role_worker_mapping = role_worker_mapping
    self.resource_pool_manager = resource_pool_manager
    self.ray_worker_group_cls = ray_worker_group_cls
    
    # 3.2 功能标志设置：根据配置启用或禁用 Critic、Reference Policy、Reward Model 和 Hybrid Engine。
    self.use_critic = config.critic.enable
    self.use_reference_policy = config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss
    self.use_rm = config.reward_model.enable
    self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
    
    # 3.3 配置验证：调用内部函数验证配置的合理性。
    self._validate_config()
    
    # 3.4 数据加载器创建：创建训练和验证数据加载器。
    self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
```

### 2.2 `_validate_config()` 函数

此函数在训练器初始化时执行配置的完整性检查，确保资源、批次大小和 Multi-turn 设置符合要求：

```python
def _validate_config(self):
    # 4.1 GPU 数量验证：计算并检查可用 GPU 总数。
    n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
    
    # 4.2 批次大小验证：根据 Megatron 或其他策略计算最小批次大小。
    if config.actor_rollout_ref.actor.strategy == "megatron":
        model_parallel_size = config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
        assert n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
        megatron_dp = n_gpus // (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size)
        minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
    else:
        minimal_bsz = n_gpus
    
    # 4.3 实际批次大小验证：确保实际训练批次大小是最小批次大小的整数倍。
    real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
    assert real_train_batch_size % minimal_bsz == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size ({minimal_bsz})"
    
    # 4.4 Multi-turn 配置验证：如果启用 Multi-turn，则检查工具配置路径和优势估计器类型。
    if config.actor_rollout_ref.rollout.multi_turn.enable:
        assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None
        assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO]
```

-----

## 3\. Worker 初始化详细流程

### 3.1 `init_workers()` 函数

`init_workers()` 负责在 Ray 集群上实例化和初始化 ActorRollout、Critic、Reference Policy 和 Reward Model Workers。它通过资源池管理和 WorkerGroup 机制实现：

```python
def init_workers(self):
    # 5.1 创建资源池：通过 ResourcePoolManager 创建 Ray 资源池。
    self.resource_pool_manager.create_resource_pool()
    
    # 5.2 初始化资源池到类的映射：为每个资源池创建一个字典，用于存储不同角色 Worker 的类。
    self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}
    
    # 5.3 创建 ActorRollout Worker：如果 Hybrid Engine 启用，则为 ActorRollout 创建 RayClassWithInitArgs 包装器。
    if self.hybrid_engine:
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout],
            config=self.config.actor_rollout_ref,
            role="actor_rollout",
        )
        self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
    
    # 5.4 创建 Critic Worker：如果 Critic 启用，则创建 Critic Worker 的 RayClassWithInitArgs 包装器。
    if self.use_critic:
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
        critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
        self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls
    
    # 5.5 创建 Reference Policy Worker：如果 Reference Policy 启用，则创建其 RayClassWithInitArgs 包装器。
    if self.use_reference_policy:
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
        ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
        self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls
    
    # 5.6 创建 Reward Model Worker：如果 Reward Model 启用，则创建其 RayClassWithInitArgs 包装器。
    if self.use_rm:
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
        self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls
    
    # 5.7 初始化 WorkerGroup：遍历所有资源池，创建并生成 WorkerGroup。
    all_wg = {}
    for resource_pool, class_dict in self.resource_pool_to_cls.items():
        # 5.7.1 创建共置 Worker 类：将多个 Worker 类组合成一个共置类。
        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        
        # 5.7.2 创建 WorkerGroup：实例化 RayWorkerGroup。
        wg_dict = self.ray_worker_group_cls(
            resource_pool=resource_pool, 
            ray_cls_with_init=worker_dict_cls, 
            device_name=self.device_name, 
            **wg_kwargs
        )
        
        # 5.7.3 生成 Worker：在 Ray 中实际创建 Worker 实例。
        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
        all_wg.update(spawn_wg)
    
    # 5.8 初始化各个 Worker：根据角色从 all_wg 中获取 WorkerGroup 并调用其 init_model 方法。
    if self.use_critic:
        self.critic_wg = all_wg["critic"]
        self.critic_wg.init_model()
    
    if self.use_reference_policy and not self.ref_in_actor:
        self.ref_policy_wg = all_wg["ref"]
        self.ref_policy_wg.init_model()
    
    if self.use_rm:
        self.rm_wg = all_wg["rm"]
        self.rm_wg.init_model()
    
    # 5.9 初始化 ActorRollout Worker（最后初始化以优化内存）：ActorRollout Worker 通常占用大量内存，在其他 Worker 初始化后再初始化。
    self.actor_rollout_wg = all_wg["actor_rollout"]
    self.actor_rollout_wg.init_model()
```

### 3.2 `ResourcePoolManager.create_resource_pool()` 函数

此函数负责根据配置为 Ray 集群创建和管理资源池：

```python
def create_resource_pool(self):
    # 6.1 为每个资源池创建 RayResourcePool：实例化 RayResourcePool 对象。
    for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
        resource_pool = RayResourcePool(
            process_on_nodes=process_on_nodes, 
            use_gpu=True, 
            max_colocate_count=1, 
            name_prefix=resource_pool_name
        )
        self.resource_pool_dict[resource_pool_name] = resource_pool
    
    # 6.2 检查资源可用性：验证 Ray 集群是否有足够的资源来满足资源池的需求。
    self._check_resource_available()
```

### 3.3 `RayWorkerGroup.spawn()` 函数

`spawn()` 方法在 Ray 中实际创建 Worker 实例，并将其组织成 WorkerGroup：

```python
def spawn(self, prefix_set=None):
    # 7.1 获取放置组：从资源池获取 Ray 放置组，用于控制 Worker 的物理位置。
    pgs = self.resource_pool.get_placement_groups(strategy=strategy, device_name=self.device_name)
    world_size = self.resource_pool.world_size
    
    # 7.2 创建 Worker 实例：遍历每个 rank，创建 Worker 实例并添加到内部列表中。
    for rank in range(world_size):
        # 7.2.1 获取放置组和本地 rank。
        pg = pgs[rank // local_world_size]
        local_rank = rank % local_world_size
        
        # 7.2.2 创建 Worker：使用 RayClassWithInitArgs 包装器创建远程 Worker。
        worker = self.ray_cls_with_init(
            placement_group=pg, 
            placement_group_bundle_idx=local_rank, 
            use_gpu=use_gpu, 
            num_gpus=num_gpus, 
            device_name=self.device_name
        )
        self._workers.append(worker)
        self._worker_names.append(name)
        
        # 7.2.3 等待注册中心就绪（仅在 rank 0）：确保注册中心 actor 已经启动。
        if rank == 0:
            register_center_actor = None
            actor_name = f"{self.name_prefix}_register_center"
            start_time = time.time()
            
            while time.time() - start_time < self._ray_wait_register_center_timeout:
                if actor_name in list_named_actors():
                    register_center_actor = ray.get_actor(actor_name)
                    break
                time.sleep(1)
    
    # 7.3 绑定 Worker 方法：将 Ray 远程方法绑定到 WorkerGroup 对象。
    self._bind_worker_method(self.ray_cls_with_init.cls, func_generator)
    
    return {prefix: self for prefix in prefix_set}
```

-----

## 4\. `ActorRolloutRefWorker` 初始化详细流程

### 4.1 `ActorRolloutRefWorker.__init__()` 函数

`ActorRolloutRefWorker` 负责 Actor 和 Rollout 模型的初始化和管理，它处理分布式环境、设备网格和 profiler 配置：

```python
def __init__(self, config: DictConfig, role: str):
    # 8.1 基础初始化：调用 Worker 基类的构造函数。
    Worker.__init__(self)
    self.config = config
    
    # 8.2 分布式初始化：如果 PyTorch 分布式环境未初始化，则进行初始化。
    if not torch.distributed.is_initialized():
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.distributed.init_process_group(
            backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}", 
            rank=rank, 
            world_size=world_size
        )
    
    # 8.3 设备网格创建：为 FSDP 创建设备网格。
    world_size = torch.distributed.get_world_size()
    self.device_mesh = create_device_mesh(
        world_size=world_size, 
        fsdp_size=self.config.actor.fsdp_config.fsdp_size
    )
    
    # 8.4 Ulysses 序列并行设置：如果启用 Ulysses 序列并行，则初始化其设备网格。
    self.ulysses_sequence_parallel_size = self.config.actor.get("ulysses_sequence_parallel_size", 1)
    if self.ulysses_sequence_parallel_size > 1:
        dp = world_size // self.ulysses_sequence_parallel_size
        self.ulysses_device_mesh = init_device_mesh(
            device_name, 
            mesh_shape=(dp, self.ulysses_sequence_parallel_size), 
            mesh_dim_names=["dp", "sp"]
        )
    
    # 8.5 角色设置：根据传入的 role 参数设置 Worker 的具体角色（actor, rollout, ref）。
    self.role = role
    assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]
    
    self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
    self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
    self._is_ref = self.role in ["ref", "actor_rollout_ref"]
    
    # 8.6 分析器配置：根据 Worker 角色配置 profiler。
    profiler_config = ProfilerConfig()
    if self._is_actor:
        profiler_config = profiler_config.union(ProfilerConfig(**OmegaConf.to_object(config.actor.get("profiler", DictConfig({})))))
    if self._is_rollout:
        profiler_config = profiler_config.union(ProfilerConfig(**OmegaConf.to_object(config.rollout.get("profiler", DictConfig({})))))
    if self._is_ref:
        profiler_config = profiler_config.union(ProfilerConfig(**OmegaConf.to_object(config.ref.get("profiler", DictConfig({})))))
    
    WorkerProfilerExtension.__init__(self, WorkerProfiler(rank=self.rank, config=profiler_config))
```

### 4.2 `ActorRolloutRefWorker.init_model()` 函数

此函数是模型初始化的关键点，负责构建 Actor、Rollout 和 Reference Policy 模型，并配置优化器和检查点管理器：

```python
@register(dispatch_mode=Dispatch.ONE_TO_ALL)
def init_model(self):
    # 9.1 外部库导入：如果配置中指定了外部库，则进行导入。
    if self.config.model.get("external_lib", None) is not None:
        importlib.import_module(self.config.model.external_lib)
    
    # 9.2 配置解析：解析模型和 Transformer 的覆盖配置。
    override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
    if self._is_actor:
        override_transformer_config = OmegaConf.to_container(self.config.actor.megatron.get("override_transformer_config", OmegaConf.create()), resolve=True)
    elif self._is_ref:
        override_transformer_config = OmegaConf.to_container(self.config.ref.megatron.get("override_transformer_config", OmegaConf.create()), resolve=True)
    else:
        override_transformer_config = None
    
    # 9.3 数据类型设置：设置模型参数的数据类型。
    self.param_dtype = torch.bfloat16
    self.dtype = PrecisionType.to_dtype(self.param_dtype)
    
    # 9.4 Actor 和 Rollout 模型构建：根据 Worker 角色构建 Actor 模块、优化器和调度器。
    if self._is_actor or self._is_rollout:
        optim_config = self.config.actor.optim if self._is_actor else None
        self.actor_module, self.actor_optimizer, self.actor_optimizer_scheduler, self.actor_model_config, self.actor_optim_config = self._build_model_optimizer(
            model_path=self.config.model.path,
            optim_config=optim_config,
            override_model_config=override_model_config,
            override_transformer_config=override_transformer_config,
        )
        
        # 9.4.1 参数卸载：如果配置了参数卸载，则将 Megatron 模型和优化器卸载到 CPU。
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)
    
    # 9.5 Actor 初始化：如果当前 Worker 是 Actor 角色，则实例化 MegatronPPOActor。
    if self._is_actor:
        self.actor = MegatronPPOActor(
            config=self.config.actor,
            model_config=self.actor_model_config,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            actor_module=self.actor_module,
            actor_optimizer=self.actor_optimizer,
        )
    
    # 9.6 Rollout 初始化：如果当前 Worker 是 Rollout 角色，则构建 Rollout 模块和分片管理器。
    if self._is_rollout:
        self.rollout, self.sharding_manager = self._build_rollout(
            trust_remote_code=self.config.model.get("trust_remote_code", False)
        )
        self.rollout.sharding_manager = self.sharding_manager
    
    # 9.7 Reference Policy 初始化：如果当前 Worker 是 Reference Policy 角色，则构建其模块。
    if self._is_ref:
        self.ref_module, self.ref_model_config = self._build_model_optimizer(
            model_path=self.config.model.path,
            optim_config=None,
            override_model_config=override_model_config,
            override_transformer_config=override_transformer_config,
        )
        self.ref_policy = MegatronPPOActor(
            config=self.config.ref,
            model_config=self.ref_model_config,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            actor_module=self.ref_module,
            actor_optimizer=None,
        )
        if self._ref_is_offload_param:
            offload_megatron_model_to_cpu(self.ref_module)
    
    # 9.8 检查点管理器初始化：如果当前 Worker 是 Actor 角色，则初始化 FlopsCounter 和 MegatronCheckpointManager。
    if self._is_actor:
        self.flops_counter = FlopsCounter(self.actor_model_config)
        self.checkpoint_mananager = MegatronCheckpointManager(...)
    
    get_torch_device().empty_cache() # 清理 PyTorch 缓存以释放内存。
```

### 4.3 `_build_model_optimizer()` 函数

此函数负责根据模型路径和配置构建 Megatron 模型和优化器：

```python
def _build_model_optimizer(self, model_path, optim_config, override_model_config, override_transformer_config):
    # 10.1 配置初始化：初始化 Hugging Face 和 Transformer 配置，并获取 Generation Config。
    self._init_hf_config_and_tf_config(model_path, model_path, self.dtype, override_model_config, override_transformer_config, self.config.model.get("trust_remote_code", False))
    self.generation_config = get_generation_config(self.local_path)
    
    # 10.2 模型提供者函数：定义一个内部函数，用于初始化 Megatron 模型。
    def megatron_actor_model_provider(pre_process, post_process):
        from verl.models.mcore import init_mcore_model
        parallel_model = init_mcore_model(
            self.tf_config, 
            self.hf_config, 
            pre_process, 
            post_process, 
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights, 
            value=False, 
            freeze_moe_router=override_model_config.get("moe_config", {}).get("freeze_moe_router", False)
        )
        parallel_model.to(get_device_name()) # 将模型移动到设备上。
        return parallel_model
    
    # 10.3 Actor 和 Rollout 模型构建：如果当前 Worker 既是 Actor 又是 Rollout 角色，则获取模型并加载权重。
    if self._is_actor and self._is_rollout:
        actor_module = get_model(
            megatron_actor_model_provider,
            wrap_with_ddp=True, # 使用 DistributedDataParallel 包装。
            use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer,
        )
        
        # 10.3.1 权重加载：根据配置加载模型权重。
        if self.config.actor.load_weight:
            if self.config.actor.megatron.use_dist_checkpointing:
                load_mcore_dist_weights(actor_module, self.config.actor.megatron.dist_checkpointing_path, is_value_model=False)
            else:
                load_megatron_gptmodel_weights(self.config, self.hf_config, actor_module, params_dtype=self.dtype, is_value_model=False)
        
        if self.rank == 0:
            print_model_size(actor_module[0])
    
    # 10.4 Reference Policy 模型构建：如果当前 Worker 是 Reference Policy 角色，则获取模型并加载权重。
    elif self._is_ref:
        ref_module = get_model(
            model_provider_func=megatron_actor_model_provider,
            model_type=ModelType.encoder_or_decoder,
            wrap_with_ddp=False,
            use_distributed_optimizer=self.config.ref.megatron.use_distributed_optimizer,
        )
        
        if self.config.ref.load_weight:
            if self.config.ref.megatron.use_dist_checkpointing:
                load_mcore_dist_weights(ref_module, self.config.ref.megatron.dist_checkpointing_path, is_value_model=False)
            else:
                load_megatron_gptmodel_weights(self.config, self.hf_config, ref_module, params_dtype=self.dtype, is_value_model=False)
        
        return ref_module, self.hf_config
    
    # 10.5 优化器初始化：如果当前 Worker 是 Actor 角色，则初始化 Megatron 优化器和调度器。
    if self._is_actor:
        optim_config_megatron = init_megatron_optim_config(optim_config)
        actor_optimizer = get_megatron_optimizer(model=actor_module, config=optim_config_megatron)
        actor_optimizer_scheduler = get_megatron_optimizer_param_scheduler(optimizer=actor_optimizer, config=optim_config)
    else:
        optim_config = None
        actor_optimizer = None
        actor_optimizer_scheduler = None
    
    return actor_module, actor_optimizer, actor_optimizer_scheduler, self.hf_config, optim_config
```

### 4.4 `_build_rollout()` 函数

该函数专门用于构建 Rollout 模块，特别是 SGLang Rollout，并创建相应的分片管理器：

```python
def _build_rollout(self, trust_remote_code=False):
    # 11.1 设备网格创建：为 Rollout 创建推理张量并行（infer_tp）设备网格。
    infer_tp = self.config.rollout.tensor_model_parallel_size
    dp = self.world_size // infer_tp
    assert self.world_size % infer_tp == 0
    rollout_device_mesh = init_device_mesh(
        get_device_name(), 
        mesh_shape=(dp, infer_tp), 
        mesh_dim_names=["dp", "infer_tp"]
    )
    
    # 11.2 SGLang Rollout 构建：如果 Rollout 名称是 "sglang"，则导入并实例化 SGLangRollout 和 MegatronSGLangShardingManager。
    if self.config.rollout.name == "sglang":
        from verl.workers.rollout.sglang_rollout import SGLangRollout
        from verl.workers.sharding_manager.megatron_sglang import MegatronSGLangShardingManager
        
        rollout = SGLangRollout(
            actor_module=local_path,
            config=self.config.rollout,
            tokenizer=self.tokenizer,
            model_hf_config=self.actor_model_config,
            trust_remote_code=trust_remote_code,
            device_mesh=rollout_device_mesh,
        )
        
        # 11.2.1 权重转换器创建：获取 Megatron 权重的转换器。
        from verl.models.mcore import get_mcore_weight_converter
        weight_converter = get_mcore_weight_converter(self.actor_model_config, self.dtype)
        
        # 11.2.2 分片管理器创建：实例化 MegatronSGLangShardingManager。
        sharding_manager = MegatronSGLangShardingManager(
            actor_module=self.actor.actor_module,
            inference_engine=rollout._engine,
            model_config=self.actor_model_config,
            transformer_config=self.tf_config,
            layer_name_mapping=layer_name_mapping,
            weight_converter=weight_converter,
            device_mesh=rollout_device_mesh,
        )
    
    return rollout, sharding_manager
```

-----

## 5\. SGLang Rollout 初始化详细流程

### 5.1 `SGLangRollout.__init__()` 函数

`SGLangRollout` 负责管理 SGLang 推理引擎，包括工具系统和分布式环境的初始化：

```python
def __init__(self, actor_module, config, tokenizer, model_hf_config, port=None, trust_remote_code=False, device_mesh=None, **kwargs):
    # 12.1 基础初始化：调用父类构造函数并设置配置和设备网格。
    super().__init__()
    self.config = config
    self._device_mesh_cpu = device_mesh
    os.environ.setdefault("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")
    
    # 12.2 工具系统初始化：如果配置了工具，则初始化工具 schemas、map 和解析器。
    (
        self._tool_schemas,
        self._tool_map,
        self._tool_call_parser_type,
        self._sgl_tools,
        self._function_call_parser,
    ) = self._initialize_tools(config, tokenizer)
    
    # 12.3 分布式环境初始化：初始化 SGLang 推理所需的分布式环境。
    self._init_distributed_env(device_mesh_cpu=device_mesh, **kwargs)
    
    # 12.4 配置验证：验证模型配置。
    self._verify_config(model_hf_config=model_hf_config)
    
    # 12.5 推理引擎初始化：初始化 SGLang 推理引擎。
    self._init_inference_engine(trust_remote_code, actor_module, port)
    
    # 12.6 采样参数初始化：初始化生成序列的采样参数。
    self._init_sampling_params(**kwargs)
    
    # 12.7 Tokenizer 设置：设置 Tokenizer 和 padding token ID。
    self.tokenizer = tokenizer
    self.pad_token_id = tokenizer.pad_token_id
```

### 5.2 `_initialize_tools()` 函数

此函数根据配置文件初始化 Multi-turn 对话中的工具，包括工具的 schema、映射和函数调用解析器：

```python
def _initialize_tools(self, config, tokenizer):
    # 13.1 检查工具配置：如果没有工具配置路径，则返回空列表和字典。
    if config.multi_turn.tool_config_path is None:
        return [], {}, None, [], None
    
    # 13.2 加载工具配置：从配置文件加载工具并初始化工具列表。
    tools_config_file = config.multi_turn.tool_config_path
    tools_config = OmegaConf.load(tools_config_file)
    tool_list = initialize_tools_from_config(tools_config)
    
    # 13.3 工具 schema 创建：创建 OpenAI 格式的工具 schema 和工具名称到工具对象的映射。
    tool_schemas = [tool.get_openai_tool_schema().model_dump() for tool in tool_list]
    tool_map = {tool.name: tool for tool in tool_list}
    
    # 13.4 工具调用解析器类型确定：根据 Tokenizer 类型确定工具调用解析器。
    tool_call_parser_type = get_tool_call_parser_type(tokenizer)
    
    # 13.5 SGLang 工具创建：为 SGLang 创建 Tool 对象。
    sgl_tools = [Tool.model_validate(tool_schema) for tool_schema in tool_schemas]
    
    # 13.6 函数调用解析器创建：实例化 FunctionCallParser。
    function_call_parser = FunctionCallParser(
        sgl_tools,
        tool_call_parser_type,
    )
    
    return tool_schemas, tool_map, tool_call_parser_type, sgl_tools, function_call_parser
```

### 5.3 `_init_inference_engine()` 函数

该函数负责初始化 SGLang 推理引擎，处理模型路径和分布式环境设置：

```python
def _init_inference_engine(self, trust_remote_code, actor_module, port):
    # 14.1 本地路径处理：确保模型路径是本地可访问的。
    local_path = copy_to_local(actor_module, use_shm=self.config.model.get("use_shm", False))
    
    # 14.2 分布式环境设置：根据设备网格设置 rank 和 tensor parallel size。
    if self._device_mesh_cpu is not None:
        self._rank = self._device_mesh_cpu["tp"].mesh[0].item()
        self._tp_rank = self._device_mesh_cpu["tp"].mesh[1].item()
        self._tp_size = self._device_mesh_cpu["tp"].mesh.shape[1]
    else:
        self._rank = 0
        self._tp_rank = 0
        self._tp_size = 1
    
    # 14.3 SGLang 引擎创建：仅在 tensor parallel rank 为 0 的进程上创建 EngineManager 实例。
    if self._tp_rank == 0:
        from sglang.srt.managers import EngineManager
        
        self._engine = EngineManager(
            model_path=local_path,
            trust_remote_code=trust_remote_code,
            **self.config.engine_kwargs
        )
    else:
        self._engine = None # 其他进程不直接持有 EngineManager 实例。
```

-----

## 6\. 训练主循环详细流程

### 6.1 `RayPPOTrainer.fit()` 函数

`fit()` 函数是 PPO 训练的主循环，它协调序列生成、奖励计算、优势估计和模型更新：

```python
def fit(self):
    # 15.1 训练循环初始化：遍历每个 epoch 和每个训练批次。
    for epoch in range(self.config.trainer.total_epochs):
        for batch_dict in self.train_dataloader:
            metrics = {}
            timing_raw = {}
            
            # 15.2 数据预处理：将原始批次数据转换为 DataProto 对象。
            batch: DataProto = DataProto.from_single_dict(batch_dict)
            
            # 15.3 生成批次准备：从批次中弹出生成相关的键。
            gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
            
            # 15.4 序列生成：通过 actor_rollout_wg 调用生成序列。
            with marked_timer("gen", timing_raw, color="green"):
                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                timing_raw.update(gen_batch_output.meta_info["timing"])
                gen_batch_output.meta_info.pop("timing", None)
            
            # 15.5 批次合并：将生成输出与原始批次合并。
            batch = batch.union(gen_batch_output)
            
            # 15.6 旧 log 概率计算：通过 actor_rollout_wg 计算旧策略的 log 概率。
            with marked_timer("old_log_prob", timing_raw, color="blue"):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                batch = batch.union(old_log_prob)
            
            # 15.7 参考策略计算：如果使用参考策略，则计算其 log 概率。
            if self.use_reference_policy:
                with marked_timer("ref", timing_raw, color="olive"):
                    if not self.ref_in_actor:
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    else:
                        ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)
            
            # 15.8 价值函数计算：如果使用 Critic，则计算价值。
            if self.use_critic:
                with marked_timer("values", timing_raw, color="cyan"):
                    values = self.critic_wg.compute_values(batch)
                    batch = batch.union(values)
            
            # 15.9 奖励计算和优势估计：计算奖励并应用 KL 惩罚（如果启用），然后计算优势。
            with marked_timer("adv", timing_raw, color="brown"):
                # 15.9.1 奖励计算：获取奖励模型计算的 token 级别分数。
                reward_extra_infos_dict: dict[str, list]
                if self.config.reward_model.launch_reward_fn_async:
                    reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                batch.batch["token_level_scores"] = reward_tensor
                
                if reward_extra_infos_dict:
                    batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                
                # 15.9.2 KL 惩罚应用：如果配置了 KL 惩罚，则进行应用。
                if self.config.algorithm.use_kl_in_reward:
                    batch, kl_metrics = apply_kl_penalty(
                        batch, 
                        kl_ctrl=self.kl_ctrl_in_reward, 
                        kl_penalty=self.config.algorithm.kl_penalty,
                        multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable
                    )
                    metrics.update(kl_metrics)
                else:
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                
                # 15.9.3 优势计算：根据配置的优势估计器（GAE, GRPO 等）计算优势。
                norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
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
            
            # 15.10 Critic 更新：如果使用 Critic，则更新 Critic 模型。
            if self.use_critic:
                with marked_timer("update_critic", timing_raw, color="pink"):
                    critic_output = self.critic_wg.update_critic(batch)
                critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                metrics.update(critic_output_metrics)
            
            # 15.11 Actor 更新：在 Critic warmup 结束后更新 Actor 模型。
            if self.config.trainer.critic_warmup <= self.global_steps:
                with marked_timer("update_actor", timing_raw, color="red"):
                    batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(actor_output_metrics)
            
            # 15.12 指标收集和日志：计算数据指标并更新总指标。
            data_metrics = compute_data_metrics(batch=batch, use_critic=self.use_critic)
            metrics.update(data_metrics)
            
            # 15.13 验证：如果满足验证频率，则执行验证。
            if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                with marked_timer("testing", timing_raw, color="yellow"):
                    val_metrics: dict = self._validate()
                    if is_last_step:
                        last_val_metrics = val_metrics
                metrics.update(val_metrics)
            
            # 15.14 日志记录：记录当前步骤的指标。
            logger.log(data=metrics, step=self.global_steps)
            
            # 15.15 检查点保存：如果满足保存频率，则保存模型检查点。
            if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                self._save_checkpoint()
            
            self.global_steps += 1
```

-----

## 7\. 序列生成详细流程

### 7.1 `ActorRolloutRefWorker.generate_sequences()` 函数

此函数是 ActorRolloutRefWorker 中用于生成序列的主要方法，它协调数据传输、模型卸载和 Rollout 模块的调用：

```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
@WorkerProfiler.annotate(color="red")
def generate_sequences(self, prompts: DataProto):
    # 16.1 设备转移：将 prompts 数据转移到当前设备。
    prompts = prompts.to(get_device_id())
    assert self._is_rollout # 确保当前 Worker 是 Rollout 角色。
    
    # 16.2 元信息设置：设置生成序列所需的 EOS token ID 和 Pad token ID。
    meta_info = {
        "eos_token_id": self.generation_config.eos_token_id if self.generation_config is not None else self.tokenizer.eos_token_id,
        "pad_token_id": self.generation_config.pad_token_id if self.generation_config is not None else self.tokenizer.pad_token_id,
    }
    prompts.meta_info.update(meta_info)
    
    # 16.3 参数卸载：如果配置了优化器卸载，则将其卸载到 CPU。
    if self._is_offload_optimizer:
        offload_megatron_optimizer(self.actor_optimizer)
    
    # 16.4 序列生成：通过 sharding_manager 上下文执行序列生成。
    timing_generate = {}
    with self.sharding_manager:
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module) # 如果配置了模型卸载，则将其卸载到 CPU。
        
        # 16.4.1 数据预处理：通过 sharding_manager 对数据进行预处理。
        prompts = self.sharding_manager.preprocess_data(prompts)
        
        # 16.4.2 生成执行：调用 rollout 模块的 generate_sequences 方法进行序列生成。
        with simple_timer("generate_sequences", timing_generate):
            output = self.rollout.generate_sequences(prompts=prompts)
        
        # 16.4.3 数据后处理：通过 sharding_manager 对生成结果进行后处理。
        output = self.sharding_manager.postprocess_data(output)
    
    # 16.5 时序信息处理：收集并减少生成过程中的时序信息。
    timing_generate.update(self.sharding_manager.timing)
    timing_generate = reduce_timing(timing_generate)
    output.meta_info["timing"] = timing_generate
    
    # 16.6 设备转移和缓存清理：将输出转移到 CPU 并清理 GPU 缓存。
    output = output.to("cpu")
    get_torch_device().empty_cache()
    
    return output
```

### 7.2 `SGLangRollout.generate_sequences()` 函数

此函数根据是否启用 Multi-turn 调用相应的序列生成方法：

```python
def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
    # 17.1 Multi-turn 检查：如果启用了 Multi-turn，则调用请求级别生成序列。
    if self.config.multi_turn.enable:
        return self._req_level_generate_sequences(prompts, **kwargs)
    
    # 17.2 批量级别生成：否则，调用批次级别生成序列。
    return self._batch_level_generate_sequences(prompts, **kwargs)
```

### 7.3 `_req_level_generate_sequences()` 函数

该函数处理 Multi-turn 场景下的请求级序列生成，涉及异步请求处理、工具调用和结果聚合：

```python
def _req_level_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
    # 18.1 参数提取：提取采样参数和目标设备。
    do_sample = prompts.meta_info.get("do_sample", True)
    is_validate = prompts.meta_info.get("validate", False)
    tgt_device = prompts.batch["input_ids"].device
    
    # 18.2 请求预处理：仅在 tensor parallel rank 为 0 的进程上预处理 prompts 为异步请求。
    if self._tp_rank == 0:
        req_list = self._preprocess_prompt_to_async_rollout_requests(
            prompts,
            n=1 if is_validate else self.config.n,
        )
        
        # 18.3 异步请求处理：使用 asyncio 异步处理每个请求。
        loop = asyncio.get_event_loop()
        output_req_list = loop.run_until_complete(
            asyncio.gather(
                *[self._async_rollout_a_request(req, do_sample, is_validate, **kwargs) for req in req_list],
            )
        )
        
        # 18.4 结果排序：根据 batch_data_id 和 rollout_offset 对输出请求列表进行排序。
        sorted_output_req_list = sorted(output_req_list, key=lambda x: (x.batch_data_id, x.rollout_offset))
    else:
        sorted_output_req_list = None
    
    # 18.5 结果广播：将 rank 0 的结果广播到所有 tensor parallel 组中的进程。
    dist.barrier()
    [sorted_output_req_list] = broadcast_pyobj(
        data=[sorted_output_req_list],
        rank=self._rank,
        dist_group=self._device_mesh_cpu["tp"].get_group(),
        src=self._device_mesh_cpu["tp"].mesh[0].item(),
        force_cpu_device=False,
    )
    
    # 18.6 批次数据构建：从 sorted_output_req_list 中提取并聚合 prompt 和 response 相关的数据。
    prompt_ids, response_ids = [], []
    prompt_attention_mask, response_attention_mask = [], []
    prompt_position_ids, response_position_ids = [], []
    prompt_loss_mask, response_loss_mask = [], []
    messages = []
    reward_scores = []
    
    for req in sorted_output_req_list:
        assert req.state == AsyncRolloutRequestStateEnum.COMPLETED # 确保请求已完成。
        
        # 18.6.1 数据提取和验证：将请求数据转换为 PyTorch 张量并添加到列表中。
        prompt_ids.append(torch.tensor(req.prompt_ids, dtype=torch.int, device=tgt_device))
        response_ids.append(torch.tensor(req.response_ids, dtype=torch.int, device=tgt_device))
        prompt_attention_mask.append(torch.tensor(req.prompt_attention_mask, dtype=torch.int, device=tgt_device))
        response_attention_mask.append(torch.tensor(req.response_attention_mask, dtype=torch.int, device=tgt_device))
        prompt_position_ids.append(torch.tensor(req.prompt_position_ids, dtype=torch.int, device=tgt_device))
        response_position_ids.append(torch.tensor(req.response_position_ids, dtype=torch.int, device=tgt_device))
        prompt_loss_mask.append(torch.tensor(req.prompt_loss_mask, dtype=torch.int, device=tgt_device))
        response_loss_mask.append(torch.tensor(req.response_loss_mask, dtype=torch.int, device=tgt_device))
        messages.append({"messages": req.messages})
        reward_scores.append(req.reward_scores)
    
    # 18.7 序列填充：对 prompt 和 response ID 进行填充，以确保批次中的序列长度一致。
    prompt_ids = pad_sequence(prompt_ids, batch_first=True, padding_value=self.pad_token_id, padding_side="left")
    if prompt_ids.shape[1] < self.config.prompt_length:
        prompt_ids = pad_sequence_to_length(prompt_ids, self.config.prompt_length, self.pad_token_id, left_pad=True)
    
    response_ids = pad_sequence(response_ids, batch_first=True, padding_value=self.pad_token_id)
    if response_ids.shape[1] < self.config.response_length:
        response_ids = pad_sequence_to_length(response_ids, self.config.response_length, self.pad_token_id)
    
    # 18.8 其他张量填充：对 attention_mask 和 loss_mask 进行填充。
    prompt_attention_mask = pad_sequence(prompt_attention_mask, batch_first=True, padding_value=0, padding_side="left")
    if prompt_attention_mask.shape[1] < self.config.prompt_length:
        prompt_attention_mask = pad_sequence_to_length(prompt_attention_mask, self.config.prompt_length, 0, left_pad=True)
    
    response_attention_mask = pad_sequence(response_attention_mask, batch_first=True, padding_value=0)
    if response_attention_mask.shape[1] < self.config.response_length:
        response_attention_mask = pad_sequence_to_length(response_attention_mask, self.config.response_length, 0)
    
    # 18.9 Position IDs 处理：填充 prompt_position_ids，并根据 prompt 长度和 response 长度计算 response_position_ids。
    prompt_position_ids = pad_sequence(prompt_position_ids, batch_first=True, padding_value=0, padding_side="left")
    if prompt_position_ids.shape[1] < self.config.prompt_length:
        prompt_position_ids = pad_sequence_to_length(prompt_position_ids, self.config.prompt_length, 0, left_pad=True)
    
    response_length = response_ids.size(1)
    delta_position_id = torch.arange(1, response_length + 1, device=response_ids.device)
    delta_position_id = delta_position_id.unsqueeze(0).repeat(len(sorted_output_req_list), 1)
    response_position_ids = prompt_position_ids[:, -1:] + delta_position_id
    
    # 18.10 Loss Mask 处理：填充 prompt_loss_mask 和 response_loss_mask。
    prompt_loss_mask = pad_sequence(prompt_loss_mask, batch_first=True, padding_value=0, padding_side="left")
    if prompt_loss_mask.shape[1] < self.config.prompt_length:
        prompt_loss_mask = pad_sequence_to_length(prompt_loss_mask, self.config.prompt_length, 0, left_pad=True)
    
    response_loss_mask = pad_sequence(response_loss_mask, batch_first=True, padding_value=0)
    if response_loss_mask.shape[1] < self.config.response_length:
        response_loss_mask = pad_sequence_to_length(response_loss_mask, self.config.response_length, 0)
    
    # 18.11 最终张量拼接：将 prompt 和 response 相关的张量拼接成完整的 input_ids, attention_mask, position_ids, loss_mask。
    input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
    attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
    position_ids = torch.cat((prompt_position_ids, response_position_ids), dim=-1)
    loss_mask = torch.cat((prompt_loss_mask, response_loss_mask), dim=-1)
    
    # 18.12 批次数据构建：创建 TensorDict 存储所有张量数据。
    batch = TensorDict(
        {
            "prompts": prompt_ids,
            "responses": response_ids,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        },
        batch_size=len(sorted_output_req_list),
    )
    
    # 18.13 缓存清理：如果配置了 free_cache_engine 且当前进程是 master，则刷新引擎缓存。
    if self.config.free_cache_engine and self._engine is not None and self._tp_rank == 0:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._engine.flush_cache())
    
    # 18.14 返回结果：返回包含张量和非张量数据的 DataProto 对象。
    return DataProto(
        batch=batch,
        non_tensor_batch={
            "messages": np.array(messages),
            "reward_scores": np.array(reward_scores),
        },
    )
```

### 7.4 `_async_rollout_a_request()` 函数

此函数处理单个异步 Rollout 请求，包括状态转换、工具调用和与推理引擎的交互：

```python
async def _async_rollout_a_request(self, req: AsyncRolloutRequest, do_sample=True, is_validate=False, **kwargs):
    # 19.1 初始化：复制请求对象，并初始化相关变量。
    assert self._tp_rank == 0, "only the master process can call this function"
    _req = deepcopy(req)
    finish_reason_type = None
    output = None
    current_turns = 0
    
    # 19.2 Multi-turn 循环：在一个请求的 Multi-turn 对话中进行迭代。
    while current_turns < self.config.multi_turn.max_turns:
        # 19.2.1 PENDING 状态处理：处理请求的初始 PENDING 状态。
        if _req.state == AsyncRolloutRequestStateEnum.PENDING:
            await self._handle_pending_state(_req)
            _req.state = AsyncRolloutRequestStateEnum.RUNNING # 转换为 RUNNING 状态。
            
        # 19.2.2 TOOL_CALLING 状态处理：处理工具调用。
        elif _req.state == AsyncRolloutRequestStateEnum.TOOL_CALLING:
            if _req.messages[-1].tool_calls is not None:
                # 19.2.2.1 工具调用执行：异步执行解析出的工具调用。
                parsed_tool_calls = _req.messages[-1].tool_calls
                tool_call_results = await asyncio.gather(
                    *[
                        self._tool_map[tool_call.function.name].execute(
                            _req.request_id,
                            tool_call.function.arguments,
                            **_req.tools_kwargs[tool_call.function.name].get("execute_kwargs", {}),
                        )
                        for tool_call in parsed_tool_calls
                    ]
                )
                
                # 19.2.2.2 工具响应添加：将工具响应作为消息添加到请求中。
                _req.add_tool_response_messages(self.tokenizer, [resp for resp, _, _ in tool_call_results])
                
                # 19.2.2.3 指标更新：更新工具相关的指标。
                for tool_call, (resp, reward, metrics) in zip(parsed_tool_calls, tool_call_results):
                    _req.update_metrics(metrics, tool_call.function.name)
                
                # 19.2.2.4 长度检查：如果输入长度超过最大模型长度，则终止。
                if len(_req.input_ids) >= self.config.max_model_len:
                    finish_reason_type = FinishReasonTypeEnum.STOP
                    break
                
                _req.state = AsyncRolloutRequestStateEnum.RUNNING # 转换回 RUNNING 状态。
            else:
                raise ValueError(f"Unexpected tool calling last message state: {_req.messages[-1]}")
                
        # 19.2.3 RUNNING 状态处理：处理正常的生成请求。
        elif _req.state == AsyncRolloutRequestStateEnum.RUNNING:
            # 19.2.3.1 长度检查：在生成前检查 prompt 长度。
            if len(_req.get_generation_prompt_ids(self.tokenizer)) + 1 >= self.config.max_model_len:
                finish_reason_type = FinishReasonTypeEnum.LENGTH
                break
            
            # 19.2.3.2 引擎调用：调用 SGLang 推理引擎进行生成。
            output = await self._handle_engine_call(_req, do_sample, is_validate, **kwargs)
            content = output["text"]
            finish_reason_type = FinishReasonTypeEnum.from_str(output["meta_info"]["finish_reason"]["type"])
            current_turns += 1
            
            # 19.2.3.3 长度限制处理：如果因为长度限制而停止，则添加助理消息并终止。
            if finish_reason_type == FinishReasonTypeEnum.LENGTH:
                _req.add_assistant_message(self.tokenizer, content)
                break
            else:
                # 19.2.3.4 工具调用检查：检查生成内容是否包含工具调用。
                if self._function_call_parser and self._function_call_parser.has_tool_call(content):
                    finish_reason_type = FinishReasonTypeEnum.TOOL_CALL
                    _req.state = AsyncRolloutRequestStateEnum.TOOL_CALLING # 转换为 TOOL_CALLING 状态。
                    
                    # 19.2.3.4.1 工具调用解析：解析生成内容中的工具调用。
                    try:
                        normed_content, tool_calls = self._function_call_parser.parse_non_stream(content)
                    except (JSONDecodeError, AttributeError):
                        normed_content = content
                        tool_calls = []
                    
                    # 19.2.3.4.2 工具调用处理：为每个解析出的工具调用创建 OpenAIFunctionToolCall 对象。
                    parsed_tool_calls = []
                    for tool_call in tool_calls:
                        function, has_decode_error = OpenAIFunctionCallSchema.from_openai_function_parsed_schema(
                            OpenAIFunctionParsedSchema(
                                name=tool_call.name,
                                arguments=tool_call.parameters,
                            )
                        )
                        
                        if has_decode_error:
                            continue
                            
                        parsed_tool_calls.append(
                            OpenAIFunctionToolCall(
                                id=str(tool_call.tool_index),
                                function=function,
                            )
                        )
                    
                    # 19.2.3.4.3 消息添加：添加包含工具调用的助理消息，或在没有有效工具调用时终止。
                    if len(parsed_tool_calls) > 0:
                        _req.add_assistant_message(self.tokenizer, normed_content, tool_calls=parsed_tool_calls)
                    else:
                        _req.add_assistant_message(self.tokenizer, content)
                        finish_reason_type = FinishReasonTypeEnum.STOP
                        _req.state = AsyncRolloutRequestStateEnum.COMPLETED
                        break
                else:
                    # 19.2.3.5 普通消息添加：如果没有工具调用，则添加普通助理消息并终止。
                    _req.add_assistant_message(self.tokenizer, content)
                    break
    
    # 19.3 轮次限制检查：如果达到最大轮次，则设置终止原因。
    if current_turns >= self.config.multi_turn.max_turns:
        finish_reason_type = FinishReasonTypeEnum.STOP
    
    # 19.4 请求最终化：最终化请求，设置奖励分数和终止原因。
    _req.finalize(
        self.tokenizer,
        reward_scores={},
        finish_reason_type=finish_reason_type,
    )
    
    return _req
```

### 7.5 `_handle_engine_call()` 函数

该函数封装了对 SGLang 推理引擎的异步调用，用于实际的文本生成：

```python
async def _handle_engine_call(self, req: AsyncRolloutRequest, do_sample: bool, is_validate: bool, **kwargs):
    # 20.1 生成 prompt IDs：获取用于生成的 prompt token IDs。
    generation_prompt_ids = req.get_generation_prompt_ids(self.tokenizer)
    
    # 20.2 采样参数更新：在采样参数上下文中调用引擎。
    with self.update_sampling_params(**kwargs):
        # 20.3 引擎调用：异步调用 SGLang 引擎进行文本生成。
        output = await self._engine.async_generate(
            prompt=None, # 因为已经转换为 prompt token id
            sampling_params=self.sampling_params,
            return_logprob=True,
            input_ids=[generation_prompt_ids],
            image_data=None,
        )
    
    return output
```

-----

## 8\. 训练计算详细流程

### 8.1 `ActorRolloutRefWorker.compute_log_prob()` 函数

此函数计算 Actor 模型的 log 概率，并处理数据传输、模型加载和分片：

```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
@WorkerProfiler.annotate(color="blue")
def compute_log_prob(self, data: DataProto):
    # 21.1 设备转移：将数据转移到当前设备。
    data = data.to(get_device_id())
    assert self._is_actor # 确保当前 Worker 是 Actor 角色。
    
    # 21.2 参数加载：如果配置了参数卸载，则将 Megatron 模型加载到 GPU。
    if self._is_offload_param:
        load_megatron_model_to_gpu(self.actor_module)
    
    # 21.3 微批次大小设置：设置 log 概率计算的微批次大小和最大 token 长度。
    micro_batch_size = self.config.actor.log_prob_micro_batch_size_per_gpu
    data.meta_info["micro_batch_size"] = micro_batch_size
    data.meta_info["max_token_len"] = self.config.actor.log_prob_max_token_len_per_gpu
    data.meta_info["use_dynamic_bsz"] = self.config.actor.log_prob_use_dynamic_bsz
    data.meta_info["temperature"] = self.config.rollout.temperature
    
    # 21.4 分片管理器处理：通过 Ulysses 分片管理器进行数据预处理和后处理。
    with self.ulysses_sharding_manager:
        data = self.ulysses_sharding_manager.preprocess_data(data=data)
        
        # 21.5 Log 概率计算：调用 Actor 模块的 compute_log_prob 方法。
        output, _ = self.actor.compute_log_prob(data=data, calculate_entropy=True)
        
        output = self.ulysses_sharding_manager.postprocess_data(data=output)
    
    # 21.6 设备转移和参数卸载：将输出转移到 CPU 并卸载模型参数。
    output = output.to("cpu")
    if self._is_offload_param:
        offload_megatron_model_to_cpu(self.actor_module)
    
    get_torch_device().empty_cache()
    return output
```

### 8.2 `MegatronPPOActor.compute_log_prob()` 函数

此函数在 MegatronPPOActor 中计算 log 概率，处理数据准备、前向传播和管道并行同步：

```python
def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
    # 22.1 数据准备：将数据转移到设备并处理微批次大小。
    data.to(get_device_id())
    data.batch = data.batch.contiguous()
    use_dynamic_bsz = data.meta_info.get("use_dynamic_bsz", False)
    micro_batch_size = data.meta_info.get("micro_batch_size", None)
    max_token_len = data.meta_info.get("max_token_len", None)
    
    # 22.2 Log 概率计算函数：定义一个内部函数用于从模型输出中提取 log 概率。
    def compute_logprobs_fn(output, data, use_dynamic_bsz=False, indices=None):
        response = data["responses"]
        response_length = response.size(1)
        log_probs = output["log_probs"][:, -response_length - 1 : -1].contiguous()
        return {"log_probs": log_probs}
    
    # 22.3 前向计算：在无梯度模式下执行模型的前向传播。
    with torch.no_grad():
        output = self.forward_backward_batch(
            data=data, 
            forward_only=True, 
            use_dynamic_bsz=use_dynamic_bsz, 
            micro_batch_size=micro_batch_size, 
            max_token_len=max_token_len, 
            mini_batch_size=None
        )
        
        # 22.4 结果处理：在管道并行最后一阶段聚合 log 概率，并处理动态批次大小的索引。
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            log_probs = [o["log_probs"] for o in output["output"]]
            log_probs = torch.cat(log_probs, dim=0).to(torch.float32)
            
            if use_dynamic_bsz:
                indices = output["indices"]
                indices = list(itertools.chain.from_iterable(indices))
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                log_probs = log_probs[revert_indices]
        else:
            log_probs = torch.empty_like(data.batch["attention_mask"], dtype=torch.float32)
        
        # 22.5 管道并行同步：在管道并行组中广播 log 概率。
        torch.distributed.broadcast(
            tensor=log_probs,
            src=mpu.get_pipeline_model_parallel_last_rank(),
            group=mpu.get_pipeline_model_parallel_group(),
        )
    
    # 22.6 熵计算（如果启用）：如果需要，计算熵。
    if calculate_entropy:
        entropy = -torch.sum(torch.exp(log_probs) * log_probs, dim=-1)
        return log_probs, entropy
    
    return log_probs
```

### 8.3 `CriticWorker.compute_values()` 函数

此函数在 CriticWorker 中计算价值函数，处理数据传输、模型加载和分片：

```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
@WorkerProfiler.annotate(color="cyan")
def compute_values(self, data: DataProto):
    # 23.1 设备转移：将数据转移到当前设备。
    data = data.to(get_device_id())
    
    # 23.2 参数加载：如果配置了参数卸载，则将 FSDP 模型加载到 GPU。
    if self._is_offload_param:
        load_fsdp_model_to_gpu(self.critic_module)
    
    # 23.3 微批次设置：设置前向传播的微批次大小和最大 token 长度。
    micro_batch_size = self.config.forward_micro_batch_size_per_gpu
    data.meta_info["micro_batch_size"] = micro_batch_size
    data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
    data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
    
    # 23.4 分片管理器处理：通过 Ulysses 分片管理器进行数据预处理和后处理。
    with self.ulysses_sharding_manager:
        data = self.ulysses_sharding_manager.preprocess_data(data=data)
        
        # 23.5 价值计算：调用 Critic 模块的 compute_values 方法。
        values = self.critic.compute_values(data=data)
        output = DataProto.from_dict(tensors={"values": values})
        
        output = self.ulysses_sharding_manager.postprocess_data(data=output)
    
    # 23.6 设备转移和参数卸载：将输出转移到 CPU 并卸载模型参数。
    output = output.to("cpu")
    if self._is_offload_param:
        offload_fsdp_model_to_cpu(self.critic_module)
    
    return output
```

### 8.4 `DataParallelPPOCritic.compute_values()` 函数

此函数在 DataParallelPPOCritic 中计算价值，处理数据分割、前向传播和价值掩码：

```python
def compute_values(self, data: DataProto) -> torch.Tensor:
    # 24.1 评估模式设置：将 Critic 模型设置为评估模式。
    self.critic_module.eval()
    micro_batch_size = data.meta_info["micro_batch_size"]
    select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
    batch = data.select(batch_keys=select_keys).batch
    use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
    
    # 24.2 微批次分割：根据批次大小和动态批次大小设置分割数据。
    if has_multi_modal_inputs:
        num_micro_batches = data.batch.batch_size[0] // micro_batch_size
        non_tensor_select_keys = ["multi_modal_inputs"]
        micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
    elif use_dynamic_bsz:
        max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
        micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
    else:
        micro_batches = batch.split(micro_batch_size)
    
    # 24.3 价值计算：遍历每个微批次，计算价值并聚合结果。
    values_lst = []
    for micro_batch in micro_batches:
        if isinstance(micro_batch, DataProto):
            micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
        
        with torch.no_grad(): # 在无梯度模式下计算价值。
            values = self._forward_micro_batch(micro_batch)
        values_lst.append(values)
    
    values = torch.concat(values_lst, dim=0)
    
    # 24.4 动态批次大小处理：如果使用动态批次大小，则恢复原始顺序。
    if use_dynamic_bsz:
        indices = list(itertools.chain.from_iterable(indices))
        revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
        values = values[revert_indices]
    
    # 24.5 响应掩码应用：仅对响应 token 应用价值掩码。
    responses = data.batch["responses"]
    attention_mask = data.batch["attention_mask"]
    response_length = responses.size(1)
    response_mask = attention_mask[:, -response_length:]
    values = values * response_mask # 只有动作 token 有价值
    
    return values
```

-----

## 9\. 模型更新详细流程

### 9.1 `ActorRolloutRefWorker.update_actor()` 函数

此函数在 ActorRolloutRefWorker 中更新 Actor 模型，处理数据传输、参数加载和分片：

```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
@WorkerProfiler.annotate(color="red")
def update_actor(self, data: DataProto):
    # 25.1 设备转移：将数据转移到 CPU（在每个微批次中会转移到设备）。
    data = data.to("cpu") 
    assert self._is_actor # 确保当前 Worker 是 Actor 角色。
    
    # 25.2 参数加载：如果配置了参数或优化器卸载，则将其加载到 GPU。
    if self._is_offload_param:
        load_fsdp_model_to_gpu(self.actor_module_fsdp)
    if self._is_offload_optimizer:
        load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())
    
    # 25.3 分片管理器处理：通过 Ulysses 分片管理器进行数据预处理和后处理。
    with self.ulysses_sharding_manager:
        data = self.ulysses_sharding_manager.preprocess_data(data=data)
        
        # 25.4 Actor 更新：调用 Actor 模块的 update_policy 方法。
        output = self.actor.update_policy(data=data)
        
        output = self.ulysses_sharding_manager.postprocess_data(data=output)
    
    # 25.5 设备转移和参数卸载：将输出转移到 CPU 并卸载模型和优化器参数。
    output = output.to("cpu")
    if self._is_offload_param:
        offload_fsdp_model_to_cpu(self.actor_module_fsdp)
    if self._is_offload_optimizer:
        offload_fsdp_optimizer(self.actor_optimizer)
    
    get_torch_device().empty_cache()
    return output
```

### 9.2 `DataParallelPPOActor.update_policy()` 函数

此函数在 DataParallelPPOActor 中更新策略，包括策略损失、熵损失、KL 损失的计算和反向传播：

```python
def update_policy(self, data: DataProto):
    # 26.1 训练模式设置：将 Actor 模型设置为训练模式。
    self.actor_module.train()
    
    # 26.2 参数提取：提取温度和 Multi-turn 标志。
    temperature = data.meta_info["temperature"]
    multi_turn = data.meta_info.get("multi_turn", False)
    
    select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
    if multi_turn:
        select_keys.append("loss_mask")
    if self.config.use_kl_loss:
        select_keys.append("ref_log_prob")
    
    batch = data.select(batch_keys=select_keys).batch
    has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
    
    # 26.3 微批次分割：根据批次大小和多模态输入分割数据。
    if has_multi_modal_inputs:
        num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
        non_tensor_select_keys = ["multi_modal_inputs"]
        dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
    else:
        dataloader = batch.split(self.config.ppo_mini_batch_size)
    
    # 26.4 指标初始化：初始化用于收集指标的字典。
    metrics = {}
    
    # 26.5 微批次循环：遍历每个微批次，执行策略更新。
    for micro_batch in dataloader:
        # 26.5.1 设备转移：将微批次数据转移到当前设备。
        if isinstance(micro_batch, DataProto):
            micro_batch = {**micro_batch.batch.to(get_device_id()), **micro_batch.non_tensor_batch}
        else:
            micro_batch = micro_batch.to(get_device_id())
        
        # 26.5.2 数据提取：提取响应、attention_mask、旧 log 概率和优势。
        responses = micro_batch["responses"]
        response_length = responses.size(1)
        attention_mask = micro_batch["attention_mask"]
        
        if multi_turn:
            response_mask = micro_batch["loss_mask"][:, -response_length:]
        else:
            response_mask = attention_mask[:, -response_length:]
        
        old_log_prob = micro_batch["old_log_probs"]
        advantages = micro_batch["advantages"]
        
        # 26.5.3 前向计算：执行模型前向传播，计算熵和 log 概率。
        entropy, log_prob = self._forward_micro_batch(
            micro_batch=micro_batch, 
            temperature=temperature, 
            calculate_entropy=True
        )
        
        # 26.5.4 策略损失计算：计算 PPO 策略损失，包括 clipfrac。
        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            cliprange=self.config.clip_ratio,
            cliprange_low=self.config.clip_ratio_low,
            cliprange_high=self.config.clip_ratio_high,
            clip_ratio_c=self.config.clip_ratio_c,
            loss_agg_mode=self.config.loss_agg_mode,
        )
        
        # 26.5.5 熵损失计算：如果配置了熵系数，则计算熵损失并加到策略损失中。
        if self.config.entropy_coeff != 0:
            entropy_loss = agg_loss(
                loss_mat=entropy, 
                loss_mask=response_mask, 
                loss_agg_mode=self.config.loss_agg_mode
            )
            policy_loss = pg_loss - entropy_loss * self.config.entropy_coeff
        else:
            policy_loss = pg_loss
        
        # 26.5.6 KL 损失计算：如果使用 KL 损失，则计算 KL 散度并加到策略损失中。
        if self.config.use_kl_loss:
            ref_log_prob = micro_batch["ref_log_prob"]
            kld = kl_penalty(
                logprob=log_prob, 
                ref_logprob=ref_log_prob, 
                kl_penalty=self.config.kl_loss_type
            )
            kl_loss = agg_loss(
                loss_mat=kld, 
                loss_mask=response_mask, 
                loss_agg_mode=self.config.loss_agg_mode
            )
            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
            metrics["actor/kl_loss"] = kl_loss.detach().item()
            metrics["actor/kl_coef"] = self.config.kl_loss_coef
            
        # 26.5.7 反向传播：计算策略损失的梯度。
        policy_loss.backward()
        
        # 26.5.8 指标收集：收集策略损失、clipfrac、熵等指标。
        metrics.update({
            "actor/policy_loss": pg_loss.detach().item(),
            "actor/policy_clipfrac": pg_clipfrac.detach().item(),
            "actor/ppo_kl": ppo_kl.detach().item(),
            "actor/policy_clipfrac_lower": pg_clipfrac_lower.detach().item(),
            "actor/entropy": entropy.detach().item(),
        })
    
    # 26.6 优化器步骤：执行优化器步骤并计算梯度范数。
    grad_norm = self._optimizer_step()
    metrics["actor/grad_norm"] = grad_norm.detach().item()
    
    return DataProto.from_dict(tensors={}, meta_info={"metrics": metrics})
```

### 9.3 `CriticWorker.update_critic()` 函数

此函数在 CriticWorker 中更新 Critic 模型，包括价值损失的计算和反向传播：

```python
def update_critic(self, data: DataProto):
    # 27.1 训练模式设置：将 Critic 模型设置为训练模式。
    self.critic_module.train()
    metrics = {}
    
    # 27.2 数据选择：选择更新 Critic 所需的键。
    select_keys = ["input_ids", "responses", "attention_mask", "position_ids", "values", "returns"]
    batch = data.select(batch_keys=select_keys).batch
    has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
    
    # 27.3 微批次分割：根据批次大小和多模态输入分割数据。
    if has_multi_modal_inputs:
        num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
        non_tensor_select_keys = ["multi_modal_inputs"]
        dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
    else:
        dataloader = batch.split(self.config.ppo_mini_batch_size)
    
    # 27.4 微批次循环：遍历每个微批次，执行 Critic 更新。
    for micro_batch in dataloader:
        # 27.4.1 设备转移：将微批次数据转移到当前设备。
        if isinstance(micro_batch, DataProto):
            micro_batch = {**micro_batch.batch.to(get_device_id()), **micro_batch.non_tensor_batch}
        else:
            micro_batch = micro_batch.to(get_device_id())
        
        # 27.4.2 数据提取：提取响应、attention_mask、values 和 returns。
        responses = micro_batch["responses"]
        attention_mask = micro_batch["attention_mask"]
        values = micro_batch["values"]
        returns = micro_batch["returns"]
        response_length = responses.size(1)
        response_mask = attention_mask[:, -response_length:]
        
        # 27.4.3 前向计算：执行 Critic 模型前向传播，预测价值。
        vpreds = self._forward_micro_batch(micro_batch)
        
        # 27.4.4 价值损失计算：计算 Critic 的价值损失。
        vf_loss, vf_clipfrac = core_algos.compute_value_loss(
            vpreds=vpreds,
            values=values,
            returns=returns,
            response_mask=response_mask,
            cliprange_value=self.config.cliprange_value,
            loss_agg_mode=self.config.loss_agg_mode,
        )
        
        # 27.4.5 动态批次大小处理：根据动态批次大小调整损失。
        if self.config.use_dynamic_bsz:
            loss = vf_loss * (len(micro_batch) / self.config.ppo_mini_batch_size)
        else:
            loss = vf_loss / self.gradient_accumulation
        
        # 27.4.6 反向传播：计算价值损失的梯度。
        loss.backward()
        
        # 27.4.7 指标收集：收集价值损失、clipfrac 和预测价值的均值。
        metrics.update({
            "critic/vf_loss": vf_loss.detach().item(),
            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
        })
    
    # 27.5 优化器步骤：执行优化器步骤并计算梯度范数。
    grad_norm = self._optimizer_step()
    metrics["critic/grad_norm"] = grad_norm.detach().item()
    
    return DataProto.from_dict(tensors={}, meta_info={"metrics": metrics})
```

-----

## 10\. 奖励计算和优势估计详细流程

### 10.1 `apply_kl_penalty()` 函数

该函数负责在奖励中应用 KL 惩罚，并更新自适应 KL 控制器：

```python
def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    # 28.1 数据提取：提取响应、token 级别分数和批次大小。
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    
    # 28.2 响应掩码确定：根据是否 Multi-turn 确定响应掩码。
    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
    
    # 28.3 KL 散度计算：计算旧 log 概率和参考 log 概率之间的 KL 散度，并应用掩码。
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], 
        data.batch["ref_log_prob"], 
        kl_penalty=kl_penalty
    )
    kld = kld * response_mask
    beta = kl_ctrl.value # 获取当前的 KL 惩罚系数。
    
    # 28.4 奖励调整：将 KL 惩罚从 token 级别分数中减去，得到调整后的奖励。
    token_level_rewards = token_level_scores - beta * kld
    
    # 28.5 当前 KL 计算：计算当前批次的平均 KL 散度。
    current_kl = masked_mean(kld, mask=response_mask, axis=-1)
    current_kl = torch.mean(current_kl, dim=0).item()
    
    # 28.6 KL 控制器更新：根据当前 KL 更新自适应 KL 控制器。
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards
    
    # 28.7 指标返回：返回 KL 惩罚和惩罚系数作为指标。
    metrics = {
        "actor/reward_kl_penalty": current_kl, 
        "actor/reward_kl_penalty_coeff": beta
    }
    
    return data, metrics
```

### 10.2 `compute_advantage()` 函数

该函数根据配置的优势估计器（GAE 或 GRPO）计算优势和回报：

```python
def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, config=None):
    # 29.1 响应掩码计算：如果数据中没有 response_mask，则计算它。
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    
    # 29.2 GAE 优势估计器：如果使用广义优势估计器 (GAE)。
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        
        # 29.2.1 PF-PPO 重加权（如果启用）：如果配置了 PF-PPO，则对数据进行重加权。
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.get("pf_ppo_reweight_method", "pow"),
                config.get("pf_ppo_weight_pow", 2.0),
            )
    
    # 29.3 GRPO 优势估计器：如果使用 GRPO 优势估计器。
    elif adv_estimator == AdvantageEstimator.GRPO:
        # 29.3.1 掩码准备：根据 Multi-turn 调整 GRPO 计算掩码。
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            response_length = grpo_calculation_mask.size(1)
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
        
        # 29.3.2 GRPO 计算：计算 GRPO 优势。
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    
    # 29.4 其他优势估计器：处理其他类型的优势估计器。
    else:
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        
        if "uid" in data.non_tensor_batch:
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    
    return data
```

### 10.3 `compute_gae_advantage_return()` 函数

此函数在无梯度模式下计算广义优势估计（GAE）和回报：

```python
def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor, lam: torch.Tensor):
    # 30.1 初始化：初始化 lastgaelam 和 advantages_reversed 列表。
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]
        
        # 30.2 反向计算 GAE：从后向前迭代，计算每个时间步的 GAE。
        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        
        # 30.3 优势计算：反转 advantages_reversed 并堆叠成张量，然后计算 returns。
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        
        # 30.4 优势标准化：对优势进行掩码白化（标准化）。
        advantages = verl_F.masked_whiten(advantages, response_mask)
    
    return advantages, returns
```

-----

## 11\. 关键数据结构

### 11.1 `DataProto` 结构

`DataProto` 是系统中用于封装批次数据的核心结构，它将张量数据、非张量数据和元信息组织在一起：

```python
class DataProto:
    def __init__(self, batch: TensorDict, non_tensor_batch: dict = None, meta_info: dict = None):
        self.batch = batch # 包含所有张量数据，通常是一个 TensorDict。
        self.non_tensor_batch = non_tensor_batch or {} # 包含非张量数据（如字符串、列表等）。
        self.meta_info = meta_info or {} # 包含元信息，如批次大小、设备 ID 等。
    
    def union(self, other: DataProto) -> DataProto:
        # 合并两个 DataProto：合并张量批次、非张量批次和元信息。
        new_batch = self.batch.union(other.batch)
        new_non_tensor_batch = {**self.non_tensor_batch, **other.non_tensor_batch}
        new_meta_info = {**self.meta_info, **other.meta_info}
        return DataProto(new_batch, new_non_tensor_batch, new_meta_info)
    
    def select(self, batch_keys: List[str], non_tensor_batch_keys: List[str] = None) -> DataProto:
        # 选择特定的键：从当前 DataProto 中选择指定的张量键和非张量键，返回新的 DataProto。
        selected_batch = self.batch.select(batch_keys)
        selected_non_tensor_batch = {}
        if non_tensor_batch_keys:
            for key in non_tensor_batch_keys:
                if key in self.non_tensor_batch:
                    selected_non_tensor_batch[key] = self.non_tensor_batch[key]
        return DataProto(selected_batch, selected_non_tensor_batch, self.meta_info)
```

### 11.2 `AsyncRolloutRequest` 结构

`AsyncRolloutRequest` 用于管理 Multi-turn Rollout 过程中的异步请求，它包含了请求的状态、消息历史、工具相关信息和生成所需的各种 token IDs：

```python
class AsyncRolloutRequest(BaseModel):
    # 31.1 基础字段：请求的唯一标识符、批次 ID、Rollout 偏移量、当前状态和消息历史。
    batch_data_id: int = 0
    rollout_offset: int = 0
    request_id: str
    state: AsyncRolloutRequestStateEnum
    messages: List[Message]
    
    # 31.2 工具相关字段：工具 schemas 和工具执行所需的 kwargs。
    tool_schemas: Optional[List[OpenAIFunctionToolSchema]] = None
    tools_kwargs: Dict[str, Any] = {}
    
    # 31.3 Token 相关字段：各种类型的 token ID 列表，用于模型输入和输出。
    input_ids: List[int]
    prompt_ids: List[int]
    response_ids: List[int]
    attention_mask: List[int]
    prompt_attention_mask: List[int]
    response_attention_mask: List[int]
    position_ids: List[int]
    prompt_position_ids: List[int]
    response_position_ids: List[int]
    loss_mask: List[int]
    prompt_loss_mask: List[int]
    response_loss_mask: List[int]
    
    # 31.4 其他字段：奖励分数、最大 prompt/response/模型长度和指标。
    reward_scores: Dict[str, float]
    max_prompt_len: int
    max_response_len: int = 8192
    max_model_len: int = 32768
    metrics: Dict[str, List[Any]] = {}
    
    # 31.5 配置字段：是否使用推理聊天模板、是否启用 tokenization 健全性检查以及生成相关的 prompt 位置信息。
    use_inference_chat_template: bool
    enable_tokenization_sanity_check: bool
    generation_prompt_ids: List[int]
    base_conv_wo_gen_prompt_end_pos: int
    base_conv_with_gen_prompt_end_pos: int
```

-----

## 12\. 性能优化关键点

为了确保 Verl Multi-turn RL 系统的高效运行，采用了多种性能优化策略，主要集中在内存管理、通信优化和异步处理方面。

### 12.1 内存管理

大规模模型训练对内存的需求极高，以下技术用于优化内存使用：

```python
# 32.1 参数卸载：在不需要时将模型参数从 GPU 卸载到 CPU，以释放 GPU 内存。
if self._is_offload_param:
    offload_fsdp_model_to_cpu(self.actor_module_fsdp)

# 32.2 缓存清理：显式清理 PyTorch 的 CUDA 缓存，有助于回收未使用的 GPU 内存。
get_torch_device().empty_cache()

# 32.3 动态批次大小：根据可用的 token 长度动态调整批次大小，从而更有效地利用 GPU 内存，减少填充开销。
if use_dynamic_bsz:
    max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
    micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
```

### 12.2 通信优化

分布式训练中，Worker 间的通信是性能瓶颈之一。系统采用分片管理和高效广播来减少通信开销：

```python
# 33.1 分片管理器：通过 Ulysses 分片管理器，将数据自动分发到不同的设备进行并行计算，减少手动数据传输的复杂性和开销。
with self.ulysses_sharding_manager:
    data = self.ulysses_sharding_manager.preprocess_data(data=data)
    # 计算
    output = self.actor.compute_log_prob(data=data, calculate_entropy=True)
    output = self.ulysses_sharding_manager.postprocess_data(data=output)

# 33.2 广播优化：使用 PyTorch 分布式通信原语高效地广播 Python 对象（例如排序后的请求列表），确保所有 Worker 都能获取到必要的信息，同时避免重复计算。
[sorted_output_req_list] = broadcast_pyobj(
    data=[sorted_output_req_list],
    rank=self._rank,
    dist_group=self._device_mesh_cpu["tp"].get_group(),
    src=self._device_mesh_cpu["tp"].mesh[0].item(),
    force_cpu_device=False,
)
```

### 12.3 异步处理

异步处理是实现 Multi-turn 对话和提高系统吞吐量的关键。通过非阻塞操作，系统可以并发执行多个任务：

```python
# 34.1 异步工具调用：通过 asyncio.gather 并发执行多个工具调用，而不是串行执行，显著减少等待时间。
tool_call_results = await asyncio.gather(
    *[
        self._tool_map[tool_call.function.name].execute(
            _req.request_id,
            tool_call.function.arguments,
            **_req.tools_kwargs[tool_call.function.name].get("execute_kwargs", {}),
        )
        for tool_call in parsed_tool_calls
    ]
)

# 34.2 异步引擎调用：使用 SGLang 引擎的异步生成接口，允许在生成过程中进行其他操作，提高资源利用率。
output = await self._engine.async_generate(
    prompt=None,
    sampling_params=self.sampling_params,
    return_logprob=True,
    input_ids=[generation_prompt_ids],
    image_data=None,
)
```

-----

## 13\. 错误处理和调试

系统的健壮性通过严格的配置验证和运行时状态检查来保证，这些措施有助于早期发现和解决潜在问题。

### 13.1 配置验证

在系统启动和初始化阶段，会进行多项配置检查，以确保资源分配和参数设置的合理性：

```python
# 35.1 资源可用性检查：在创建资源池时，检查 Ray 集群是否有足够的 GPU 资源来满足需求，避免资源不足导致的崩溃。
def _check_resource_available(self):
    node_available_resources = ray.state.available_resources_per_node()
    node_available_gpus = {node: node_info.get("GPU", 0) for node, node_info in node_available_resources.items()}
    
    total_available_gpus = sum(node_available_gpus.values())
    total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
    
    if total_available_gpus < total_required_gpus:
        raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

# 35.2 批次大小验证：确保实际训练批次大小能被最小可能批次大小整除，这对于 Megatron 等并行策略至关重要。
real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
assert real_train_batch_size % minimal_bsz == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size ({minimal_bsz})"
```

### 13.2 状态检查

在运行时，对关键数据结构和流程状态进行断言检查，以捕获非预期行为：

```python
# 36.1 请求状态验证：确保异步 Rollout 请求在处理时处于预期的完成状态。
assert req.state == AsyncRolloutRequestStateEnum.COMPLETED, f"Request {req.request_id} is not completed"

# 36.2 长度一致性检查：验证输入序列的各种张量（input_ids, attention_mask, position_ids, loss_mask）的长度是否一致，避免维度不匹配的问题。
assert len(req.input_ids) == len(req.attention_mask) == len(req.position_ids) == len(req.loss_mask), f"""Request {req.request_id} has different length of {len(req.input_ids)=}, {len(req.attention_mask)=}, {len(req.position_ids)=}, {len(req.loss_mask)=}"""

# 36.3 长度限制检查：确保输入序列的长度不超过模型能够处理的最大长度，防止 OOM 错误或模型截断。
assert len(req.input_ids) <= self.config.max_model_len, f"Request {req.request_id} has input_ids length {len(req.input_ids)} greater than max_model_len {self.config.max_model_len}"
```

-----

## 14\. 总结

这份超详细的函数级别流程分析清晰地展示了 **Verl Multi-turn RL** 系统的完整技术实现。其关键特点在于：

1.  **分层架构设计**: 从 Ray 集群管理到具体的模型计算，每一层都有明确的职责分工，使得系统结构清晰，易于维护和扩展。
2.  **异步处理能力**: 引入了对工具调用的异步 Multi-turn 对话处理机制，显著提升了系统的并发能力和响应速度。
3.  **内存优化**: 通过**参数卸载**、**动态批次大小**和**缓存清理**等技术，有效管理大规模模型训练中的内存消耗，使其能在有限资源下运行。
4.  **分布式计算**: 全面支持 **FSDP** 和 **Megatron** 两种主流的并行策略，确保系统能够高效地利用多 GPU 和多节点资源。
5.  **灵活的工具集成**: 支持自定义工具的动态加载和执行，极大地增强了模型在实际应用中的功能性和交互性。
6.  **完整的训练流程**: 实现了从数据加载、序列生成、奖励计算、优势估计到模型更新的完整闭环，构成了一个端到端的强化学习框架。

整个系统通过精心设计的函数调用链、模块化组件和多重优化策略，共同构建了一个高效、可扩展且功能强大的 Multi-turn RLHF 训练框架。