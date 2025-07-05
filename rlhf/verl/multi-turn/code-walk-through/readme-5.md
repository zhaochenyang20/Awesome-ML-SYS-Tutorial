# verl 参数速览

由于 [Hydra 的使用](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/code-walk-through/readme.md#%E8%84%9A%E6%9C%AC%E9%85%8D%E7%BD%AE)，verl 的参数分散在整个框架各地，而且官方的参数文档缺乏更新。因此，我们 SGLang RL 小组联合 Amazon AGI SF Lab 整理了一份 verl 参数速览。由于参数众多，我们很难保证每个参数的理解都是绝对正确的，但是都是经过我们反复检查。思考再三，我们决定将这份参数速览分享给社区，希望对大家有所帮助。整个手册的参与者有：

Ji Li（蚂蚁），Zhuoran Yin（CMU），Changyi Yang（CMU），Chengxi Li（CMU），Xinpeng Wei（Amazon），Chenyang Zhao（Amazon）


## Batch Size

| 参数名称 | 详细解释 |
| --- | --- |
| `data.train_batch_size` | **作用**：定义了单次训练发送给 Rollout Engine 的样本数量，也即这是在每个 PPO 迭代开始时，从训练数据集中采样的提示 （Prompt）数量。<br><br>**详细解释**：这个值是 RL 训练中的基本样本数量。例如，设置为 1024 意味着在一次迭代中会：<br>1. 从数据集中随机抽取 1024 个 prompt。<br> 2. 将这 1024 个 prompt 发送给当前的 Rollout Engine 中，从而得到 1024 组完整的 trajectories（prompt, response）。<br>3. 接下来，这 1024 个 trajectories 进行经验计算（make experience），后续用于 Actor 和 Critic 模型的更新。<br><br>**影响与权衡**：影响总共训练的样本量。 |
| `data.val_batch_size` （Deprecated) | **作用**：在 Validation 阶段使用的批次大小。<br><br>**详细解释**：这与 `train_batch_size` 类似，但仅用于评估模型性能，不参与训练。如果设置为 `null`，会使用验证集的大小作为默认值。Note: 已经deprecated，推荐设置为 null。此时，整个 validation dataset 一次性发给 SGLang engines，自行进行内存管理。|
| `actor_rollout_ref.actor.ppo_mini_batch_size` <br> `critic.ppo_mini_batch_size` | **作用**：定义了 PPO 训练更新中的 ini-batch 大小。<br><br>**详细解释**：`data.train_batch_size` 收集到的全部经验数据将被分割成多个 mini-batch，每块的大小就是 `ppo_mini_batch_size`。模型每处理完一个 mini-batch，才会进行一次参数更新。<br>例如，如果 `train_batch_size = 1024`，`ppo_mini_batch_size = 256`，那么在一个 PPO Epoch 中，模型会进行 `1024 / 256 = 4` 次参数更新。<br><br>**影响与权衡**：增大 mini-batch，单次更新的梯度更稳定，但更新频率更低，更新次数减少。|
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` <br> `critic.ppo_micro_batch_size_per_gpu` | **作用**：定义了在单个 GPU 上进行一次 forward/backward 的数据大小。<br><br>**详细解释**：这是实现梯度累积的核心参数。mini-batch 会被再次切分为若干个 micro-batch。例如，在单卡上，`ppo_mini_batch_size = 256`，`ppo_micro_batch_size_per_gpu = 32`，那么梯度累积的步数就是 `256 / 32 = 8`。这意味着模型会运行 8 次 forward 得到 loss，然后 backward 的到 gradient。每次处理 32 个样本，直到累积完整个 mini-batch 计算出的梯度。此时，使用累积的总梯度，对模型参数进行一次更新（`optimizer.step()`）。这个值必须根据显存大小来严格调整，是防止 OOM 的关键。<br><br>**影响与权衡**：增大此值，减少了梯度累积的次数，可以提高训练的吞吐量，增大显存消耗。|
| `actor_rollout_ref.actor.ppo_micro_batch_size` <br> `critic.ppo_micro_batch_size`（Deprecated) | **作用**：已弃用，被 `per_gpu` 版本取代，因为它能更好地适应分布式训练环境。 |

## Dynamic Batch Size

当样本长度差异很大时，按样本数量划分批次可能导致不同批次的计算量极不均衡，而基于 token 总数来控制 batch size 是一种平衡每个 batch 训练时间的方案。

| 参数名称 | 详细解释 |
| --- | --- |
| `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` <br> `critic.ppo_max_token_len_per_gpu` | **作用**：定义了一个 PPO micro batch size 中，单个 GPU 能处理的最大 Token 总数。<br><br>**详细解释**：这是 `ppo_micro_batch_size_per_gpu` 的替代方案，与 `use_dynamic_bsz` 配合使用。系统会自动打包样本，直到总 Token 量（`prompt_len + response_len`）接近这个阈值，形成一个动态的 micro batch size，从而稳定计算效率；无论长短样本，每个微批次的计算量都相对恒定。<br>例如，设置为 `actor_rollout_ref.actor.ppo_max_token_len_per_gpu = 16384`，系统可能会打包 16 个长度为 1024 的样本（16 * 1024 = 16384）或者 64个长度为 256 的样本（64 * 256 = 16384）。<br><br>**影响与权衡**：通常比固定样本数的微批次效率更高，能更好地利用计算资源，减少 GPU 不稳定性。通常设置为 `n * ({data.max_prompt_length} + {data.max_response_length})` |
| `reward_model.forward_max_token_len_per_gpu` <br> `critic.forward_max_token_len_per_gpu` <br> `actor_rollout_ref.ref.log_prob_max_token_len_per_gpu` | **作用**：只进行 forward 计算的 Model 的一个 micro-batch 的 token 最大数量。<br><br>**详细解释**：一些模型（Reward Model, Critic 求 value, Reference Model 求 log probs）在 make experience 阶段只有 forward 计算，此时 rollout engine 已经 offload 了，而 training engine 还没启动，显存占用是很少的。因此，可以为它们设置一个更大的 batch size 以加速计算。这些参数同样是 `use_dynamic_bsz` 的一部分，用于优化这些特定任务的执行效率。 |
| `critic.forward_micro_batch_size_per_gpu` <br> `reward_model.micro_batch_size_per_gpu` <br> `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu` | **作用**：同样为只进行 forward 计算的 model 设置 micro-batch size。<br><br>**详细解释**：同上一行参数。 |
| `actor_rollout_ref.actor.use_dynamic_bsz` <br> `critic.use_dynamic_bsz` <br> `reward_model.use_dynamic_bsz` | **作用**：是否启用 Dynamic Batch Size。<br><br>**详细解释**：当此项为 `True` 时，系统会忽略基于样本数的 `micro_batch_size_per_gpu` 参数，转而使用基于 Token 数的 `max_token_len_per_gpu` 参数来构建 batch。 |
| `trainer.balance_batch` | **作用**：是否在分布式训练的各个 dp rank 间平衡 batch size。<br><br>**详细解释**：在 single controller 上将 data 重新排序使得每个 dp rank 获得相似数目的 token。 |

## Rollout Sampling Parameters

| 参数名称 | 作用与解释 |
| --- | --- |
| `actor_rollout_ref.rollout.temperature` | temperature 值越高，概率分布越平滑，生成结果更多样、更随机；值越低，分布越尖锐，生成结果更倾向于高概率词元，更确定、更保守。`temperature=0` 通常等同于 Greedy Decoding。 |
| `actor_rollout_ref.rollout.top_k` | 在每一步生成时，只考虑概率最高的 K 个 token 进行采样。例如，`top_k=50` 表示只从概率前 50 的 token 中选择。<br>- 禁用时：在 Hugging Face 中设置为 `0` 或 `None`，在 SGLang 中设置为 `-1`（此时从整个词汇表采样）。|
| `actor_rollout_ref.rollout.top_p` | 从概率最高的 token 开始累加，直到它们的总概率达到 P，然后从这个 nucleus token 集合中进行采样。是一种动态选择采样范围的方法。`top_p=1.0` 表示不限制。 |
| `actor_rollout_ref.rollout.use_fire_sampling` | 是否使用 Fire Sampling，来自字节的[论文](https://arxiv.org/abs/2410.21236)。 |
| `actor_rollout_ref.rollout.n` | 为每个 prompt 生成的 response 数量，也即 GRPO 中的 group size。|
| `actor_rollout_ref.rollout.ignore_eos` | 是否忽略 EOS (End-of-Sentence) 标记。如果为 `True`，即使模型生成了 EOS 标记，也会继续生成直到达到 `max_response_length`。 |

## Performance and Resource Management

| 参数名称 | 作用与解释 |
| --- | --- |
| `actor_rollout_ref.rollout.prompt_length` | 最大的 prompt 长度，过长则被截断。 |
| `actor_rollout_ref.rollout.response_length` | 最大的 response 长度，到达最大长度时 SGLang engine 会直接返回。 |
| `actor_rollout_ref.rollout.dtype` | 模型数据类型。例如 `bfloat16`, `float16`，需要与训练阶段的模型类型对齐，否则更新模型参数的时候还需要做量化。 |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | SGLang 中模型参数和 KV Cache占显存的比例，如果使用 0.4.8.post1 以上版本 SGLang，则可以设置到 0.85，使用以下版本的 SGLang 则需要设置到 0.5 左右。|
| `actor_rollout_ref.rollout.free_cache_engine` | Rollout 后是否释放引擎缓存；SGLang 中启用此选项将触发 `flush_cache()` 操作：清空 kv cache pool，将所有 slots 标记为可用。通过释放 KV Cache 的逻辑占用，但是不释放物理显存。为什么需要 flush kv cache 可以参考[此处](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/code-walk-through/readme.md#sglangrolloutasyncengine)。|
| `actor_rollout_ref.rollout.load_format` | 模型权重加载模式。例如 `dummy_dtensor`（随机初始化权重，用于快速调试）、`hf`、`safetensors`（推荐，安全且高效）。 |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` (TP_SIZE) | 张量并行大小，表示用多少个 GPU 来共同运行一个 SGLang engine。例如，`TP_SIZE=4` 表示将一个大模型的权重切成 4 份，由 4 个 GPU 协同完成推理。 |
| `actor_rollout_ref.rollout.max_model_len` | 模型能处理的最大总长度（prompt + response）；如果未设置，通常由模型配置决定。 |
| `actor_rollout_ref.rollout.max_num_seqs` | 引擎能同时处理的最大请求量，或者说同时推理的最多 prompts 数量。 |
| `actor_rollout_ref.rollout.enable_chunked_prefill` | 是否启用 Chunked Prefill，对于非常长的 Prompt，可以将其分块处理，减少显存峰值，但是降低吞吐量。 |
| `actor_rollout_ref.rollout.disable_log_stats` | 是否禁用推理引擎的统计日志，以减少控制台输出。 |

---

### SGLang 配置

| 参数名称 | 作用与解释 |
| --- | --- |
| `actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend` | **SGLang 使用的注意力后端**。可以选择如 `flashinfer`, `triton`, `flashmla`, `null`  几种实现，以适应自身显卡。 |

---

### multi-turn tool calling

这部分参数主要用于需要多轮交互的场景，如工具调用、连续对话等，由 SGLang Engine 支持。

| 参数名称 | 作用与解释 |
| --- | --- |
| `actor_rollout_ref.rollout.multi_turn.enable` | 是否启用多轮对话模式。 |
| `actor_rollout_ref.rollout.multi_turn.max_turns` | 最多进行 tool calling 的轮次，null 时会默认设置成 `max_model_len // 3` 来避免无限对话。|
| `actor_rollout_ref.rollout.multi_turn.tool_config_path` | 工具配置文件路径，定义模型可以调用的外部工具。 |
| `actor_rollout_ref.rollout.multi_turn.completion_callback` | 自定义 callback function，在每轮生成后可以执行自定义逻辑。 |
| `actor_rollout_ref.rollout.multi_turn.use_inference_chat_template` | 是否使用模型在 inference 阶段的 chat template。`True` 表示遵循 inference 阶段的模板格式。`False` 表示使用预训练中的模板，可能包含额外思考过程的完整 Token 序列。对于任何模型，一定要保证在 post training 和后续 inference 进行测试的阶段采用一致的模板。 |
| `actor_rollout_ref.rollout.multi_turn.enable_tokenization_sanity_check` | 是否进行 tokenization 安全性检查，检查逐轮 tokenize 的结果与一次 tokenize 整个 chat history 的结果一致。 |

### 验证阶段配置

| 参数名称 | 作用与解释 |
| --- | --- |
| `actor_rollout_ref.rollout.val_kwargs.*` | 验证阶段的 sampling parameters，这允许我们在 post training 和 validation 时使用不同的 sampling parameters。例如，验证时通常设置 `temperature=0` 和 `do_sample=False` 来进行贪心解码，以获得更稳定的评估结果。 |

### Dataset

| 参数名称 | 作用与解释 |
| --- | --- |
| `data.tokenizer` | Tokenizer 的类或路径。如果为 null，将从模型中自动推断。 |
| `data.use_shm` | 是否使用共享内存（shared memory）来加载数据。 |
| `data.train_files` | 训练集 parquet 文件。可以是列表或单个文件；路径可以是本地路径或 HDFS 路径。 |
| `data.val_files` | 验证集 parquet 文件。可以是列表或单个文件。 |
| `data.prompt_key` | 数据集中 prompt 的字段。默认为 `prompt`。 |
| `data.reward_fn_key` | 用于选择奖励函数（如果每个样本使用不同奖励函数）的字段。 |
| `data.max_prompt_length` | 最大提示长度。所有提示将向左填充到此长度。 |
| `data.return_raw_input_ids` | 是否返回未添加聊天模板的原始 `input_ids`;当 reward model 的 chat template 与 policy model 不同时使用。 |
| `data.return_raw_chat` | 是否返回未应用聊天模板的原始 response。 |
| `data.return_full_prompt` | 是否返回带有聊天模板的完整 prompt。 |
| `data.shuffle` | 是否在 DataLoader 中打乱数据。 |
| `data.validation_shuffle` | 是否打乱验证集。 |
| `data.filter_overlong_prompts` | 是否过滤超长的 prompt。 |
| `data.filter_overlong_prompts_workers` | 过滤超长 prompt 的工作进程数。对于大型数据集，使用多进程加速。默认为 1。 |
| `data.truncation` | 如果 `input_ids` 或 `prompt` 超过最大长度，则进行截断。 |
| `data.image_key` | 多模态数据集中表示图像的字段。默认为 `images`。 |
| `data.video_key` | 多模态数据集中表示视频的字段。 |
| `data.trust_remote_code` | 是否信任本地的的 huggingface cache；注意，这个 remote 是相对 huggingface 而言的，所以这个参数考虑的是“是否信任本地”。 |
| `data.custom_cls.path` | 包含自定义数据集类的文件路径。如果未指定，将使用预实现的默认数据集。 |
| `data.custom_cls.name` | 指定文件中的数据集类名。 |

### Actor, Rollout & Reference Worker 配置

Critic 和 Actor 的参数是非常一致的，不再赘述。

| 参数名称 | 描述 |
| ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `actor_rollout_ref.hybrid_engine`                                            | 目前只支持 hybird engine，将 actor 和 rollout 模型放在同一资源组上。                                                         |
| `actor_rollout_ref.model.path`                                               | Huggingface 模型路径。可以是本地路径或 HDFS 路径。                                                             |
| `actor_rollout_ref.model.use_shm`                                            | 是否使用共享内存（SHM）来加速模型权重的加载。                                                                                          |
| `actor_rollout_ref.model.external_lib`                                       | 用于注册 Huggingface 模型/分词器的额外 Python 包。                                                                                                 |
| `actor_rollout_ref.model.override_config`                                    | 用于覆盖模型原始配置，主要用于 dropout。                                                                                                               |
| `actor_rollout_ref.model.enable_gradient_checkpointing`                      | actor 训练过程是否重算梯度，以时间换空间。                                                              |
| `actor_rollout_ref.model.enable_activation_offload`                          | actor 训练是否将 activation offload 到 CPU。                                                                         |
| `actor_rollout_ref.model.use_remove_padding`                                 | 训练期间是否移除输入中的 padding元。                                                                                                                |
| `actor_rollout_ref.model.use_liger`                                          | 是否使用 Liger kernel 进行线性层融合。                                                                            |
| `actor_rollout_ref.model.use_fused_kernels`                                  | 是否使用自定义 fused kernel（如 FlashAttention, fused MLP）。                      |
| `actor_rollout_ref.model.fused_kernel_options.impl_backend`                  | 融合核的实现后端，triton 或 torch。需要和 `use_fused_kernels` 配合使用                                                                 |
| `actor_rollout_ref.model.trust_remote_code`                                  | 是否信任本地的的 huggingface cache；注意，这个 remote 是相对 huggingface 而言的，所以这个参数考虑的是“是否信任本地”。                                     |
| `actor_rollout_ref.actor.strategy`                                           | 训练 backend fsdp, fsdp2 或 megatron。        |
| `actor_rollout_ref.actor.grad_clip`                                          | Actor 更新的梯度裁剪。                                                                                                                                                              |
| `actor_rollout_ref.actor.clip_ratio`                                         | PPO 裁剪比率。                                                                                                                                                                                           |
| `actor_rollout_ref.actor.clip_ratio_low`                                     | 非对称裁剪的下界（用于 dual-clip PPO）。                                                                                                                    |
| `actor_rollout_ref.actor.clip_ratio_high`                                    | 非对称裁剪的上界（用于 dual-clip PPO）。                                                                                                                    |
| `actor_rollout_ref.actor.clip_ratio_c`                                       | Dual-clip PPO 中的常数 C；当优势 < -C 时进行裁剪。                                                                                                               |
| `actor_rollout_ref.actor.loss_agg_mode`                                      | 损失聚合模式：`token-mean`, `seq-mean-token-sum`, 或 `seq-mean-token-mean`。                                                                                                                        |
| `actor_rollout_ref.actor.entropy_coeff`                                      | PPO 损失中的熵正则化系数。                                                                                                                                               |
| `actor_rollout_ref.actor.use_kl_loss`                                        | 是否使用 KL 损失代替 KL 奖励惩罚。对于 GRPO 为 True。                                                                                                |
| `actor_rollout_ref.actor.use_torch_compile`                                  | 是否使用 `torch.compile()`。                                                                   |
| `actor_rollout_ref.actor.kl_loss_coef`                                       | 启用 `use_kl_loss` 时的 KL 损失系数，用于 GRPO。                                                                                                                |
| `actor_rollout_ref.actor.kl_loss_type`                                       | KL 散度损失的类型。选项：`kl`, `abs`, `mse`, `low_var_kl`, `full`。                                                                                                                           |
| `actor_rollout_ref.actor.ppo_epochs`                                         | PPO 轮数。                                                                                                                                     |
| `actor_rollout_ref.actor.shuffle`                                            | 打乱训练数据。                                                                                                  |
| `actor_rollout_ref.actor.ulysses_sequence_parallel_size`                     | Ulysses 类的 sequence parallel 大小。                                            |
| `actor_rollout_ref.actor.entropy_from_logits_with_chunking`                  | 通过分块计算熵以减少显存峰值。                                                                                                      |
| `actor_rollout_ref.actor.entropy_checkpointing`                              | 是否将 entropy 通过 checkpoint 存下来。                               |
| `actor_rollout_ref.actor.checkpoint.save_contents`                           | 保存的检查点中包含的内容。                                                                                                         |
| `actor_rollout_ref.actor.checkpoint.load_contents`                           | 从检查点加载时指定的内容。                                                                                                         |
| `actor_rollout_ref.actor.optim.lr`                                           | 学习率。                                                                                                             |
| `actor_rollout_ref.actor.optim.lr_warmup_steps`                              | 预热步数；负值则由 `lr_warmup_steps_ratio` 决定。        |
| `actor_rollout_ref.actor.optim.lr_warmup_steps_ratio`                        | 预热步数比例（当 `lr_warmup_steps` 为负时使用）。                                                                                                                |
| `actor_rollout_ref.actor.optim.min_lr_ratio`                                 | 余弦调度器的最小学习率比例。                                                                                                                |
| `actor_rollout_ref.actor.optim.num_cycles`                                   | 学习率调度中的余弦周期数。                                                                                                                |
| `actor_rollout_ref.actor.optim.warmup_style`                                 | 学习率预热风格：`constant` 或 `cosine`。                                                      |
| `actor_rollout_ref.actor.optim.total_training_steps`                         | 总训练步数。                                                                                                                |
| `actor_rollout_ref.actor.optim.weight_decay`                                 | 权重衰减系数，控制训练过程中对权重施加的 L2 正则化的强度。                                                                                                                |
| `actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params`             | 触发 FSDP 包装一个层的最小参数数量。                                                                                                                |
| `actor_rollout_ref.actor.fsdp_config.param_offload`                          | 是否将模型参数卸载到 CPU（以速度换内存）。                                                                                                         |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload`                      | 是否将优化器状态卸载到 CPU。                                                                                                                |
| `actor_rollout_ref.actor.fsdp_config.offload_policy`                         | 仅用于 FSDP2：训练期间卸载参数/梯度/优化器。                                                                                                                  |
| `actor_rollout_ref.actor.fsdp_config.reshard_after_forward`                  | 仅用于 FSDP2：前向传播后重新分片以减少内存占用。                                                                                                  |
| `actor_rollout_ref.actor.fsdp_config.fsdp_size`                              | 每个 FSDP 分片组中的 GPU 数量；-1 表示自动。                                                                                                                     |
| `actor_rollout_ref.actor.fsdp_config.forward_prefetch`                       | 仅用于 FSDP1：在前向计算完成前预取下一次前向传播的 all-gather。                                                                                                   |
| `actor_rollout_ref.actor.profiler.discrete`                                  | True 表示每个任务有自己的数据库，False 表示所有任务共享一个。                                                                                                                      |
| `actor_rollout_ref.actor.profiler.all_ranks`                                 | 是否对所有 rank 进行性能分析。                                                                                                                      |
| `actor_rollout_ref.actor.profiler.ranks`                                     | 将被分析的 rank。null 或 [0,1,...]。                                                                                                                      |
| `actor_rollout_ref.ref.strategy`                                             | Reference 模型的 FSDP 配置，与 actor 相同。                                                                                                                      |
| `actor_rollout_ref.ref.fsdp_config.param_offload`                            | FSDP 中是否卸载参数。                                                                                                                      |
| `actor_rollout_ref.ref.fsdp_config.reshard_after_forward`                    | 仅用于 FSDP2：是否在模型前向传播后重新分片以节省内存。                                                                                                                  |
| `actor_rollout_ref.ref.fsdp_config.forward_prefetch`                         | 仅用于 FSDP1：在前向计算完成前预取下一次前向传播的 all-gather。                                                                                                   |
| `actor_rollout_ref.ref.fsdp_config.wrap_policy.min_num_params`               | FSDP 包装模块中的最小参数量。                                                                                                                      |
| `actor_rollout_ref.ref.profiler.discrete`                                    | True 表示每个任务有自己的数据库，False 表示所有任务共享一个。                                                                                                                      |
| `actor_rollout_ref.ref.profiler.all_ranks`                                   | 是否对所有 rank 进行性能分析。                                                                                                                      |
| `actor_rollout_ref.ref.profiler.ranks`                                       | 将被分析的 rank。null 或 [0,1,...]。                                                                                                                      |

### Reward Model

| 参数名称 | 描述 |
| --- | --- |
| `reward_model.enable` | 是否启用奖励模型。如果为 False，则仅使用用户定义的奖励函数计算奖励。 |
| `reward_model.strategy` | FSDP 策略：`fsdp` 或 `fsdp2`或`megatron`。 |
| `reward_model.model.input_tokenizer` | 输入分词器。如果奖励模型的聊天模板与策略不一致，则需要此项。 |
| `reward_model.model.path` | RM 的 HDFS 路径或本地路径。仅支持 AutoModelForSequenceClassification。 |
| `reward_model.model.use_shm` | 是否使用共享内存加载模型。 |
| `reward_model.model.external_lib` | 外部模型实现（可选）。 |
| `reward_model.model.use_remove_padding` | 使用移除填充优化（节省计算）。 |
| `reward_model.model.use_fused_kernels` | 是否使用融合的奖励核以加速。 |
| `reward_model.model.trust_remote_code` | 是否允许加载远程代码模型，默认为 False。 |
| `reward_model.model.fsdp_config.wrap_policy.min_num_params` | 触发 FSDP 包装的最小参数数量。 |
| `reward_model.model.fsdp_config.param_offload` | 是否将模型参数卸载到 CPU。 |
| `reward_model.model.fsdp_config.reshard_after_forward` | 仅用于 FSDP2：前向传播后重新分片以减少内存占用。 |
| `reward_model.model.fsdp_config.fsdp_size` | 每个 FSDP 分片组中的 GPU 数量；-1 表示自动。 |
| `reward_model.model.fsdp_config.forward_prefetch` | 仅用于 FSDP1：在前向计算完成前预取下一次前向传播的 all-gather。 |
| `reward_model.reward_manager` | 定义计算基于规则的奖励和处理不同奖励源的机制。 |
| `reward_model.launch_reward_fn_async` | 是否在 log_prob 期间异步启动自定义奖励函数。 |
| `reward_model.sandbox_fusion.url` | 用于远程 reward 函数的 URL。 |
| `reward_model.sandbox_fusion.max_concurrent` | 允许到沙箱的最大并发请求数。 |
| `reward_model.profiler.discrete` | True 表示每个任务有自己的数据库，False 表示所有任务共享一个。 |

### Custom Reward Function

| 参数名称 | 描述 |
| --- | --- |
| `custom_reward_function.path` | 包含自定义奖励函数的文件路径。 |
| `custom_reward_function.name` | 指定文件中的奖励函数名称。默认为 `compute_score`。 |

### Algorithm

| 参数名称 | 描述 |
| --- | --- |
| `algorithm.gamma` | 未来奖励的折扣因子。 |
| `algorithm.lam` | GAE 估计器中偏差和方差的权衡。 |
| `algorithm.adv_estimator` | 优势估计器类型：`gae`, `grpo`, `reinforce_plus_plus` 等。 |
| `algorithm.norm_adv_by_std_in_grpo` | 是否在 GRPO 中按标准差归一化优势。 |
| `algorithm.use_kl_in_reward` | 是否在奖励中启用 KL 惩罚。 |
| `algorithm.kl_penalty` | 如何估计 KL 散度：`kl`, `abs`, `mse`, `low_var_kl`, 或 `full`。 |
| `algorithm.kl_ctrl.type` | KL 控制类型：`fixed` 或 `adaptive`。 |
| `algorithm.kl_ctrl.kl_coef` | KL 惩罚的初始系数。 |
| `algorithm.kl_ctrl.horizon` | 自适应控制器的 horizon 值（如果启用）。 |
| `algorithm.kl_ctrl.target_kl` | 目标 KL 散度（用于自适应控制器）。 |
| `algorithm.use_pf_ppo` | 是否启用偏好反馈 PPO。 |
| `algorithm.pf_ppo.reweight_method` | 样本重加权方法：`pow`, `max_min`, 或 `max_random`。 |
| `algorithm.pf_ppo.weight_pow` | `pow` 方法中用于权重缩放的幂。 |

### Trainer

| 参数名称 | 描述 |
| --- | --- |
| `trainer.balance_batch` | 是否在分布式工作节点间平衡批次大小。 |
| `trainer.total_epochs` | 训练的总轮数。 |
| `trainer.total_training_steps` | 总训练步数（可显式设置或从轮数派生）。 |
| `trainer.profile_steps` | 将被分析的步骤。null 表示不进行分析。 |
| `trainer.controller_nsight_options.trace` | 对于controller进程，选择要追踪的 API（比如cuda，nvtx，cublas，etc）。 |
| `trainer.controller_nsight_options.cuda-memory-usage` | 对于controller进程，是否profile CUDA 内存使用情况。必须是字符串 `"true"` 或 `"false"`。 |
| `trainer.controller_nsight_options.cuda-graph-trace` | 对于controller进程，是否将CUDA graphs 将被作为一个整体进行追踪。 |
| `trainer.worker_nsight_options.trace` | 对于worker进程，选择要追踪的 API。 |
| `trainer.worker_nsight_options.cuda-memory-usage` | 对于worker进程，是否profile CUDA 内存使用情况。必须是字符串 `"true"` 或 `"false"`。 |
| `trainer.worker_nsight_options.cuda-graph-trace` | 对于worker进程，是否CUDA graphs 将被作为一个整体进行追踪。 |
| `trainer.worker_nsight_options.capture-range` | 仅在 torch.cuda.profiler.start 和 stop 范围内进行分析。默认值为cudaProfilerApi，不要更改此配置。 |
| `trainer.worker_nsight_options.capture-range-end` | 指定捕获范围结束时的期望行为。 |
| `trainer.worker_nsight_options.kill` | 向目标应用程序的进程组发送信号。我们让程序自行退出。 |
| `trainer.project_name` | 用于实验跟踪（如 wandb）的项目名称。 |
| `trainer.experiment_name` | 用于在跟踪工具中识别运行的实验名称。 |
| `trainer.logger` | 使用的日志后端：`console`, `wandb` 等。 |
| `trainer.log_val_generations` | 验证期间要记录的生成数量。 |
| `trainer.rollout_data_dir` | 用于记录 rollout 数据的目录；如果为 null 则不转储。 |
| `trainer.validation_data_dir` | 用于记录验证数据的目录；如果为 null 则不转储。 |
| `trainer.nnodes` | 训练中使用的节点数。 |
| `trainer.n_gpus_per_node` | 每个节点的 GPU 数量。 |
| `trainer.save_freq` | 模型检查点的保存频率（按迭代次数）。 |
| `trainer.resume_mode` | 恢复模式：`auto`, `disable`, 或 `resume_path`。 |
| `trainer.resume_from_path` | 从该路径恢复训练（仅当 resume_mode 为 `resume_path` 时使用）。 |
| `trainer.val_before_train` | 是否在训练开始前运行验证。 |
| `trainer.val_only` | 是否只运行验证。 |
| `trainer.test_freq` | 验证频率（以训练迭代次数计）。 |
| `trainer.critic_warmup` | 在更新策略之前预热 critic 的迭代次数。 |
| `trainer.default_hdfs_dir` | 用于保存检查点的默认分布式文件系统路径。 |
| `trainer.del_local_ckpt_after_load` | 加载后是否删除本地检查点。 |
| `trainer.default_local_dir` | 用于保存检查点的默认本地目录。 |
| `trainer.max_actor_ckpt_to_keep` | 保留的 actor 检查点的最大数量。 |
| `trainer.max_critic_ckpt_to_keep` | 保留的 critic 检查点的最大数量。 |
| `trainer.ray_wait_register_center_timeout` | Ray worker 等待注册的超时时间（秒）。 |
| `trainer.device` | 运行训练的设备（如 `cuda`, `cpu`）。 |

### Ray Init

| 参数名称 | 描述 |
| --- | --- |
| `ray_init.num_cpus` | Ray 使用的 CPU 数量。使用 SLURM 时应使用固定数字而不是 null。 |
| `ray_init.timeline_json_file` | 保存 Ray 时间线 JSON 文件以进行性能分析的路径。 |