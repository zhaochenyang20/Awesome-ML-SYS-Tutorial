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

### Actor, Rollout & Reference 模型

| 参数名称 (Parameter Name)                                                    | 描述 (Description)                                                                                                                                                                                                         |
| ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `actor_rollout_ref.hybrid_engine`                                            | 是否为混合引擎，目前仅支持混合引擎。 (Whether it's a hybrid engine, currently only supports hybrid engine.)                                                                                                                |
| `actor_rollout_ref.model.path`                                               | Huggingface 模型路径。可以是本地路径或 HDFS 路径。 (Huggingface model path. This can be either local path or HDFS path.)                                                                                                   |
| `actor_rollout_ref.model.use_shm`                                            | 是否使用共享内存（SHM）来加速模型权重的加载。 (Whether to use shared memory (SHM) for accelerating the loading of model weights.)                                                                                          |
| `actor_rollout_ref.model.external_lib`                                       | 用于注册 Huggingface 模型/分词器的额外 Python 包。 (Additional Python packages to register huggingface models/tokenizers.)                                                                                                 |
| `actor_rollout_ref.model.override_config`                                    | 用于覆盖模型原始配置，主要用于 dropout。 (Used to override model's original configurations, mainly dropout.)                                                                                                               |
| `actor_rollout_ref.model.enable_gradient_checkpointing`                      | 为 actor 启用梯度检查点。 (Enable gradient checkpointing for actor.)                                                                                                                                                       |
| `actor_rollout_ref.model.enable_activation_offload`                          | 为 actor 启用激活卸载。 (Enable activation offloading for actor.)                                                                                                                                                          |
| `actor_rollout_ref.model.use_remove_padding`                                 | 训练期间是否移除输入中的填充（padding）词元。 (Whether to remove padding tokens in inputs during training.)                                                                                                                |
| `actor_rollout_ref.model.lora_rank`                                          | 设置为正值以启用 LoRA（例如，32）。 (Set to positive value to enable LoRA (e.g., 32).)                                                                                                                                     |
| `actor_rollout_ref.model.lora_alpha`                                         | LoRA 的缩放因子。 (LoRA scaling factor.)                                                                                                                                                                                   |
| `actor_rollout_ref.model.target_modules`                                     | 应用 LoRA 的目标模块。选项："all-linear" 或线性层列表。 (Target modules to apply LoRA. Options: "all-linear" or list of linear layers.)                                                                                    |
| `actor_rollout_ref.model.use_liger`                                          | 是否使用 Liger 进行线性层融合。 (Whether to use Liger for linear layer fusion.)                                                                                                                                            |
| `actor_rollout_ref.model.use_fused_kernels`                                  | 是否使用自定义融合核（如 FlashAttention, fused MLP）。 (Whether to use custom fused kernels (e.g., FlashAttention, fused MLP).)                                                                                            |
| `actor_rollout_ref.model.fused_kernel_options.impl_backend`                  | 融合核的实现后端。选项："triton" 或 "torch"。需要和 `use_fused_kernels` 配合使用 (Implementation backend for fused kernels. Options: "triton" or "torch".)                                                                 |
| `actor_rollout_ref.model.trust_remote_code`                                  | 是否允许加载远程代码模型。 (Whether to enable loading a remote code model.)                                                                                                                                                |
| `actor_rollout_ref.actor.strategy`                                           | 训练策略：fsdp, fsdp2 或 megatron。这里使用 fsdp。 (fsdp, fsdp2 or megatron. fsdp backend used here.)                                                                                                                      |
| `actor_rollout_ref.actor.ppo_mini_batch_size`                                | PPO 中每个样本拆分成的子批次大小。 (Split each sample into sub-batches of this size for PPO.)                                                                                                                              |
| `actor_rollout_ref.actor.ppo_micro_batch_size`                               | [已弃用] 全局微批次大小。 ([Deprecated] Global micro batch size.)                                                                                                                                                          |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`                       | 每个 GPU 的本地微批次大小。 (Local per-GPU micro batch size.)                                                                                                                                                              |
| `actor_rollout_ref.actor.use_dynamic_bsz`                                    | 是否在运行时动态调整批次大小。 (Whether to automatically adjust batch size at runtime.)                                                                                                                                    |
| `actor_rollout_ref.actor.ppo_max_token_len_per_gpu`                          | 单个 PPO 批次中每个 GPU 的最大词元数；影响梯度累积。 (Max tokens per GPU in one PPO batch; affects gradient accumulation.)                                                                                                 |
| `actor_rollout_ref.actor.grad_clip`                                          | Actor 更新的梯度裁剪。 (Gradient clipping for actor updates.)                                                                                                                                                              |
| `actor_rollout_ref.actor.clip_ratio`                                         | PPO 裁剪比率。 (PPO clip ratio.)                                                                                                                                                                                           |
| `actor_rollout_ref.actor.clip_ratio_low`                                     | 非对称裁剪的下界（用于 dual-clip PPO）。 (Lower bound for asymmetric clipping (used in dual-clip PPO).)                                                                                                                    |
| `actor_rollout_ref.actor.clip_ratio_high`                                    | 非对称裁剪的上界（用于 dual-clip PPO）。 (Upper bound for asymmetric clipping (used in dual-clip PPO).)                                                                                                                    |
| `actor_rollout_ref.actor.clip_ratio_c`                                       | Dual-clip PPO 中的常数 C；当优势 < -C 时进行裁剪。 (Constant C in Dual-clip PPO; clips when advantage < -C.)                                                                                                               |
| `actor_rollout_ref.actor.loss_agg_mode`                                      | 损失聚合模式："token-mean", "seq-mean-token-sum", 或 "seq-mean-token-mean"。 (Loss aggregation mode: "token-mean", "seq-mean-token-sum", or "seq-mean-token-mean".)                                                        |
| `actor_rollout_ref.actor.entropy_coeff`                                      | PPO 损失中的熵正则化系数。 (Entropy regularization coefficient in PPO loss.)                                                                                                                                               |
| `actor_rollout_ref.actor.use_kl_loss`                                        | 是否使用 KL 损失代替 KL 奖励惩罚。对于 GRPO 为 True。 (Whether to use KL loss instead of KL reward penalty. True for GRPO.)                                                                                                |
| `actor_rollout_ref.actor.use_torch_compile`                                  | 是否使用 torch.compile()。 (Whether to use torch.compile().)                                                                                                                                                               |
| `actor_rollout_ref.actor.kl_loss_coef`                                       | 启用 use_kl_loss 时的 KL 损失系数。用于 GRPO。 (KL loss coefficient when use_kl_loss is enabled. For GRPO.)                                                                                                                |
| `actor_rollout_ref.actor.kl_loss_type`                                       | KL 散度损失的类型。选项："kl", "abs", "mse", "low_var_kl", "full"。 (Type of KL divergence loss. Options: "kl"(k1), "abs", "mse"(k2), "low_var_kl"(k3), "full".)                                                           |
| `actor_rollout_ref.actor.ppo_epochs`                                         | 每个批次的 PPO 轮数。 (Number of PPO epochs per batch.)                                                                                                                                                                    |
| `actor_rollout_ref.actor.shuffle`                                            | 在 PPO 轮次之间打乱训练数据。 (Shuffle training data across PPO epochs.)                                                                                                                                                   |
| `actor_rollout_ref.actor.ulysses_sequence_parallel_size`                     | Ulysses-style 模型并行的序列并行大小。 (Sequence parallelism size for Ulysses-style model parallelism.)                                                                                                                    |
| `actor_rollout_ref.actor.entropy_from_logits_with_chunking`                  | 通过分块计算熵以减少内存峰值。 (calculate entropy with chunking to reduce memory peak.)                                                                                                                                    |
| `actor_rollout_ref.actor.entropy_checkpointing`                              | 重新计算熵。 (recompute entropy.)                                                                                                                                                                                          |
| `actor_rollout_ref.actor.checkpoint.save_contents`                           | 保存的检查点中包含的内容。 (What to include in saved checkpoints.)                                                                                                                                                         |
| `actor_rollout_ref.actor.checkpoint.load_contents`                           | 从检查点加载时指定的内容。 (For more flexibility, you can specify the contents to load from the checkpoint.)                                                                                                               |
| `actor_rollout_ref.actor.optim.lr`                                           | 学习率。 (Learning rate.)                                                                                                                                                                                                  |
| `actor_rollout_ref.actor.optim.lr_warmup_steps`                              | 预热步数；负值则由 lr_warmup_steps_ratio 决定。 (Warmup steps; negative value delegates to lr_warmup_steps_ratio.)                                                                                                         |
| `actor_rollout_ref.actor.optim.lr_warmup_steps_ratio`                        | 预热步数比例（当 lr_warmup_steps 为负时使用）。 (Warmup steps ratio (used if lr_warmup_steps is negative).)                                                                                                                |
| `actor_rollout_ref.actor.optim.min_lr_ratio`                                 | 余弦调度器的最小学习率比例。 (Minimum LR ratio for cosine schedule.)                                                                                                                                                       |
| `actor_rollout_ref.actor.optim.num_cycles`                                   | 学习率调度中的余弦周期数。 (Number of cosine cycles in LR schedule.)                                                                                                                                                       |
| `actor_rollout_ref.actor.optim.warmup_style`                                 | 学习率预热风格："constant" 或 "cosine"。 (LR warmup style: "constant" or "cosine".)                                                                                                                                        |
| `actor_rollout_ref.actor.optim.total_training_steps`                         | 总训练步数（必须在运行时覆盖）。 (Total training steps (must be overridden at runtime).)                                                                                                                                   |
| `actor_rollout_ref.actor.optim.weight_decay`                                 | 权重衰减。 (Weight decay.)                                                                                                                                                                                                 |
| `actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params`             | 触发 FSDP 包装一个层的最小参数数量。 (Minimum number of parameters to trigger wrapping a layer with FSDP.)                                                                                                                 |
| `actor_rollout_ref.actor.fsdp_config.param_offload`                          | 是否将模型参数卸载到 CPU（以速度换内存）。 (Whether to offload model parameters to CPU (trades speed for memory).)                                                                                                         |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload`                      | 是否将优化器状态卸载到 CPU。 (Whether to offload optimizer state to CPU.)                                                                                                                                                  |
| `actor_rollout_ref.actor.fsdp_config.offload_policy`                         | 仅用于 FSDP2：训练期间卸载参数/梯度/优化器。 (Only for FSDP2: offload param/grad/optimizer during train.)                                                                                                                  |
| `actor_rollout_ref.actor.fsdp_config.reshard_after_forward`                  | 仅用于 FSDP2：前向传播后重新分片以减少内存占用。 (Only for FSDP2: Reshard after forward pass to reduce memory footprint.)                                                                                                  |
| `actor_rollout_ref.actor.fsdp_config.fsdp_size`                              | 每个 FSDP 分片组中的 GPU 数量；-1 表示自动。 (Number of GPUs in each FSDP shard group; -1 means auto.)                                                                                                                     |
| `actor_rollout_ref.actor.fsdp_config.forward_prefetch`                       | 仅用于 FSDP1：在前向计算完成前预取下一次前向传播的 all-gather。 (Only for FSDP1: FSDP1 configuration, prefetch the next forward-pass all-gather before the current forward computation.)                                   |
| `actor_rollout_ref.actor.profiler.discrete`                                  | True 表示每个任务有自己的数据库，False 表示所有任务共享一个。 (True for each task has its own database, False for all tasks in one training step share one database.)                                                      |
| `actor_rollout_ref.actor.profiler.all_ranks`                                 | 是否对所有 rank 进行性能分析。 (Whether to profile all ranks.)                                                                                                                                                             |
| `actor_rollout_ref.actor.profiler.ranks`                                     | 将被分析的 rank。null 或 [0,1,...]。 (The ranks that will be profiled. null or [0,1,...].)                                                                                                                                 |
| `actor_rollout_ref.ref.strategy`                                             | Reference 模型的 FSDP 配置，与 actor 相同。 (FSDP config same as actor. For models larger than 7B, it’s recommended to turn on offload for ref by default.)                                                                |
| `actor_rollout_ref.ref.fsdp_config.param_offload`                            | FSDP 中是否卸载参数。 (whether to offload parameters in FSDP.)                                                                                                                                                             |
| `actor_rollout_ref.ref.fsdp_config.reshard_after_forward`                    | 仅用于 FSDP2：是否在模型前向传播后重新分片以节省内存。 (whether to perform reshard after model forward to save memory.)                                                                                                                  |
| `actor_rollout_ref.ref.fsdp_config.forward_prefetch`                         | 仅用于 FSDP1：在前向计算完成前预取下一次前向传播的 all-gather。 (Only for FSDP1: FSDP1 configuration, prefetch the next forward-pass all-gather before the current forward computation.)                                   |
| `actor_rollout_ref.ref.fsdp_config.wrap_policy.min_num_params`               | FSDP 包装模块中的最小参数量。 (minimum number of params in a wrapped module.)                                                                                                                                              |
| `actor_rollout_ref.ref.use_torch_compile`                                    | 是否启用 torch.compile。 (whether to enable torch.compile.)                                                                                                                                                                |
| `actor_rollout_ref.ref.log_prob_micro_batch_size`                            | [将被弃用] 计算 log_prob 时单次前向传播的批次大小（全局）。 ([Will be deprecated] The batch size for one forward pass in the computation of log_prob. Global batch size.)                                                  |
| `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu`                    | 计算 log_prob 时单次前向传播的批次大小（每个 GPU 的本地大小）。 (The batch size for one forward pass in the computation of log_prob. Local batch size per GPU.)                                                            |
| `actor_rollout_ref.ref.log_prob_use_dynamic_bsz`                             | 为 log_prob 计算启用动态批次大小（序列打包）。 (enable dynamic batch size (sequence packing) for log_prob computation.)                                                                                                    |
| `actor_rollout_ref.ref.log_prob_max_token_len_per_gpu`                       | 每个 GPU 的最大词元长度。 (the max token length per GPU.)                                                                                                                                                                  |
| `actor_rollout_ref.ref.ulysses_sequence_parallel_size`                       | 序列并行大小。 (sequence parallel size.)                                                                                                                                                                                   |
| `actor_rollout_ref.ref.entropy_from_logits_with_chunking`                    | 通过分块计算熵以减少内存峰值。 (calculate entropy with chunking to reduce memory peak.)                                                                                                                                    |
| `actor_rollout_ref.ref.entropy_checkpointing`                                | 重新计算熵。 (recompute entropy.)                                                                                                                                                                                          |
| `actor_rollout_ref.ref.profiler.discrete`                                    | True 表示每个任务有自己的数据库，False 表示所有任务共享一个。 (True for each task has its own database, False for all tasks in one training step share one database.)                                                      |
| `actor_rollout_ref.ref.profiler.all_ranks`                                   | 是否对所有 rank 进行性能分析。 (Whether to profile all ranks.)                                                                                                                                                             |
| `actor_rollout_ref.ref.profiler.ranks`                                       | 将被分析的 rank。null 或 [0,1,...]。 (The ranks that will be profiled. null or [0,1,...].)                                                                                                                                 |
| `actor_rollout_ref.rollout.name`                                             | Rollout 模型类型：hf/vllm/sglang。 ([rollout.name](http://rollout.name/): hf/vllm/sglang.)                                                                                                                                 |
| `actor_rollout_ref.rollout.mode`                                             | 同步：LLM，异步：AsyncLLM。 (sync: LLM, async: AsyncLLM.)                                                                                                                                                                  |
| `actor_rollout_ref.rollout.temperature`                                      | Rollout 的采样温度。 (Sampling temperature for rollout.)                                                                                                                                                                   |
| `actor_rollout_ref.rollout.top_k`                                            | Top-k 采样参数。vLLM rollout 为 -1，HF rollout 为 0。 (Top-k sampling parameter. -1 for vLLM rollout, 0 for HF rollout.)                                                                                                   |
| `actor_rollout_ref.rollout.top_p`                                            | Top-p 采样参数。默认为 1.0。 (Top-p sampling parameter. Default 1.0.)                                                                                                                                                      |
| `actor_rollout_ref.rollout.use_fire_sampling`                                | 是否使用 Fire Sampling (https://arxiv.org/abs/2410.21236)。 (Whether to use Fire Sampling.)                                                                                                                                |
| `actor_rollout_ref.rollout.prompt_length`                                    | 通常与 data.max_prompt_length 相同。 (typically the same as data max prompt length.)                                                                                                                                       |
| `actor_rollout_ref.rollout.response_length`                                  | 通常与 data.max_response_length 相同。 (typically the same as data max response length.)                                                                                                                                   |
| `actor_rollout_ref.rollout.dtype`                                            | Rollout 模型参数类型。与 actor 模型的 FSDP/Megatron 类型对齐。 (Rollout model parameters type. Align with actor model's FSDP/Megatron type.)                                                                               |
| `actor_rollout_ref.rollout.gpu_memory_utilization`                           | vLLM/SGLang 用于 KV 缓存的 GPU 内存比例。 (Fraction of GPU memory used by vLLM/SGLang for KV cache.)                                                                                                                       |
| `actor_rollout_ref.rollout.ignore_eos`                                       | 是否在遇到 EOS 后忽略它并继续生成。 (Whether to ignore EOS and continue generating after EOS is hit.)                                                                                                                      |
| `actor_rollout_ref.rollout.enforce_eager`                                    | 是否禁用 CUDA graph。默认为 True 以允许缓存释放。 (Whether to disable CUDA graph. Default True to allow cache freeing.)                                                                                                    |
| `actor_rollout_ref.rollout.free_cache_engine`                                | 生成后是否释放引擎 KVCache。启用时需设置 enforce_eager=True。 (Whether to free engine KVCache after generation. Set enforce_eager=True when enabled.)                                                                      |
| `actor_rollout_ref.rollout.load_format`                                      | Rollout 模型权重的加载器：dummy_dtensor, hf, megatron 等。 (Which loader to use for rollout model weights: dummy_dtensor, hf, megatron, etc.)                                                                              |
| `actor_rollout_ref.rollout.layered_summon`                                   | 对于大模型，分层加载可以节省内存但会变慢。 (for huge model, layered summon can save memory (prevent OOM) but make it slower.)                                                                                              |
| `actor_rollout_ref.rollout.tensor_model_parallel_size`                       | Rollout 的张量并行大小。仅对 vLLM 有效。 (TP size for rollout. Only effective for vLLM.)                                                                                                                                   |
| `actor_rollout_ref.rollout.max_num_batched_tokens`                           | 一个批次中的最大词元数。 (max number of tokens in a batch.)                                                                                                                                                                |
| `actor_rollout_ref.rollout.max_model_len`                                    | Rollout 的最大长度。 (max length for rollout.)                                                                                                                                                                             |
| `actor_rollout_ref.rollout.max_num_seqs`                                     | 序列的最大数量。 (max length of sequences.)                                                                                                                                                                                |
| `actor_rollout_ref.rollout.log_prob_micro_batch_size`                        | [将被弃用] 计算 log_prob 时单次前向传播的批次大小（全局）。 ([Will be deprecated] The batch size for one forward pass in the computation of log_prob. Global batch size.)                                                  |
| `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu`                | 计算 log_prob 时单次前向传播的批次大小（每个 GPU 的本地大小）。 (The batch size for one forward pass in the computation of log_prob. Local batch size per GPU.)                                                            |
| `actor_rollout_ref.rollout.log_prob_use_dynamic_bsz`                         | 为 log_prob 计算启用动态批次大小（序列打包）。 (enable dynamic batch size (sequence packing) for log_prob computation.)                                                                                                    |
| `actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu`                   | log_prob 计算的最大词元长度。 (max token length for log_prob computation.)                                                                                                                                                 |
| `actor_rollout_ref.rollout.disable_log_stats`                                | 禁用日志统计。 (disable logging statistics.)                                                                                                                                                                               |
| `actor_rollout_ref.rollout.enable_chunked_prefill`                           | 设置为 True 可能获得更高吞吐量。激活时请增加 max_num_batched_tokens 或减少 max_model_len。 (may get higher throughput when set to True. When activated, Please increase max_num_batched_tokens or decrease max_model_len.) |
| `actor_rollout_ref.rollout.do_sample`                                        | 仅适用于HF rollout，训练 rollout 期间是否采样。False 使用贪婪采样。 (Whether to sample during training rollout. False uses greedy sampling.)                                                                               |
| `actor_rollout_ref.rollout.n`                                                | 响应数量（即采样次数）。> 1 用于 grpo。 (number of responses (i.e. num sample times). > 1 for grpo.)                                                                                                                       |
| `actor_rollout_ref.rollout.engine_kwargs.vllm.swap_space`                    | 推理引擎使用的交换空间（GB）。null 使用默认值（如 4 GB）。 (Swap space (in GB) used by inference engine. null uses default (e.g., 4 GB).)                                                                                  |
| `actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache` | 是否为多模态模型禁用预处理器缓存。 (Whether to disable the preprocessor cache for multimodel models.)                                                                                                                      |
| `actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend`           | sglang 引擎的注意力后端。选项：flashinfer, triton, flashmla, null 为默认。 (The attention backend for sglang engine. Options: flashinfer, triton, flashmla, null for default.)                                             |
| `actor_rollout_ref.rollout.val_kwargs.top_k`                                 | 验证时的 Top-k 采样参数。 (Top-k sampling parameter for validation.)                                                                                                                                                       |
| `actor_rollout_ref.rollout.val_kwargs.top_p`                                 | 验证时的 Top-p 采样参数。 (Top-p sampling parameter for validation.)                                                                                                                                                       |
| `actor_rollout_ref.rollout.val_kwargs.temperature`                           | 验证时的采样温度。 (Sampling temperature for validation.)                                                                                                                                                                  |
| `actor_rollout_ref.rollout.val_kwargs.n`                                     | 是否为验证重复 n 次。 (whether to repeat n times for validation.)                                                                                                                                                          |
| `actor_rollout_ref.rollout.val_kwargs.do_sample`                             | 验证时是否采样。False 使用贪婪采样。 (Whether to sample during validation. False uses greedy sampling.)                                                                                                                    |
| `actor_rollout_ref.rollout.multi_turn.enable`                                | 对于多轮工具交互任务设为 True；也应将 [rollout.name](http://rollout.name/) 设为 sglang。 (set to True for multi-turn tool interaction tasks; should set [rollout.name](http://rollout.name/) to sglang as well.)           |
| `actor_rollout_ref.rollout.multi_turn.max_turns`                             | 最大轮数，null 表示无限制。 (null for no limit (default max_length // 3).)                                                                                                                                                 |
| `actor_rollout_ref.rollout.multi_turn.tool_config_path`                      | 工具配置文件路径，null 表示无工具。 (null for no tool.)                                                                                                                                                                    |
| `actor_rollout_ref.rollout.multi_turn.completion_callback`                   | 完成回调函数，null 表示默认回调。 (null for default callback.)                                                                                                                                                             |
| `actor_rollout_ref.rollout.multi_turn.use_inference_chat_template`           | True: 使用模型默认聊天模板；False: 使用训练记录的 token id。 (True: use model's default chat template; False: use recorded token ids for training.)                                                                        |
| `actor_rollout_ref.rollout.multi_turn.enable_tokenization_sanity_check`      | 是否启用分词一致性检查，以确保逐轮分词与一次性分词结果相同。 (Whether to enable tokenization sanity check for multi-turn rollout.)                                                                                         |
| `actor_rollout_ref.rollout.profiler.discrete`                                | True 表示每个任务有自己的数据库，False 表示所有任务共享一个。 (True for each task has its own database, False for all tasks in one training step share one database.)                                                      |
| `actor_rollout_ref.rollout.profiler.all_ranks`                               | 是否对所有 rank 进行性能分析。 (Whether to profile all ranks.)                                                                                                                                                             |
| `actor_rollout_ref.rollout.profiler.ranks`                                   | 将被分析的 rank。null 或 [0,1,...]。 (The ranks that will be profiled. null or [0,1,...].)                                                                                                                                 |

### 3. Critic (`critic`)

| 参数名称 (Parameter Name) | 描述 (Description) |
| :--- | :--- |
| `critic.rollout_n` | 每次更新的 rollout 数量（与 actor 的 rollout.n 保持一致）。 (Number of rollouts per update (mirrors actor rollout_n).) |
| `critic.strategy` | 用于 Critic 模型训练的 fsdp 或 fsdp2 策略。 (fsdp or fsdp2 strategy used for critic model training.) |
| `critic.optim.lr` | 学习率。 (Learning rate.) |
| `critic.optim.lr_warmup_steps_ratio` | 预热步数比例；总步数将在运行时注入。 (Warmup steps ratio; total steps will be injected at runtime.) |
| `critic.optim.min_lr_ratio` | 余弦调度器的最小学习率比例。 (Minimum LR ratio for cosine schedule.) |
| `critic.optim.warmup_style` | 学习率预热风格："constant" 或 "cosine"。 (LR warmup style: "constant" or "cosine".) |
| `critic.optim.total_training_steps` | 总训练步数（必须在运行时覆盖）。 (Total training steps (must be overridden at runtime).) |
| `critic.optim.weight_decay` | 权重衰减。 (Weight decay.) |
| `critic.model.path` | 预训练模型权重的路径。 (Path to pretrained model weights.) |
| `critic.model.use_shm` | 是否使用共享内存加载模型。 (Whether to use shared memory for loading the model.) |
| `critic.model.tokenizer_path` | 分词器路径（默认为 actor 的模型路径）。 (Tokenizer path (defaults to actor's model path).) |
| `critic.model.override_config` | Hugging Face 配置覆盖。 (Hugging Face config override.) |
| `critic.model.external_lib` | 外部模型实现（可选）。 (External model implementation (optional).) |
| `critic.model.enable_gradient_checkpointing` | 启用梯度检查点以节省内存。 (Enable gradient checkpointing to save memory.) |
| `critic.model.enable_activation_offload` | 将激活卸载到 CPU 以减少 GPU 内存使用。 (Offload activations to CPU to reduce GPU memory usage.) |
| `critic.model.use_remove_padding` | 使用移除填充优化（节省计算）。 (Use remove padding optimization (saves compute).) |
| `critic.model.trust_remote_code` | 是否信任来自 Hugging Face 模型的远程代码。 (Whether to trust remote code from Hugging Face models.) |
| `critic.model.fsdp_config.param_offload` | 是否将模型参数卸载到 CPU。 (Whether to offload model parameters to CPU.) |
| `critic.model.fsdp_config.optimizer_offload` | 是否将优化器状态卸载到 CPU。 (Whether to offload optimizer state to CPU.) |
| `critic.model.fsdp_config.offload_policy` | 仅用于 FSDP2：训练期间卸载参数/梯度/优化器。 (Only for FSDP2: offload param/grad/optimizer during train.) |
| `critic.model.fsdp_config.reshard_after_forward` | 仅用于 FSDP2：前向传播后重新分片以减少内存占用。 (Only for FSDP2: Reshard after forward pass to reduce memory footprint.) |
| `critic.model.fsdp_config.wrap_policy.min_num_params` | 触发 FSDP 包装的最小参数数量。 (Minimum number of parameters to trigger wrapping.) |
| `critic.model.fsdp_config.fsdp_size` | 每个 FSDP 分片组中的 GPU 数量；-1 表示自动。 (Number of GPUs in each FSDP shard group; -1 means auto.) |
| `critic.model.fsdp_config.forward_prefetch` | 仅用于 FSDP1：在前向计算完成前预取下一次前向传播的 all-gather。 (Only for FSDP1: FSDP1 configuration, prefetch the next forward-pass all-gather before the current forward computation.) |
| `critic.model.lora_rank` | 设置为正值以启用 LoRA（例如，32）。 (Set to positive value to enable LoRA (e.g., 32).) |
| `critic.model.lora_alpha` | LoRA 缩放因子。 (LoRA scaling factor.) |
| `critic.model.target_modules` | LoRA 目标模块："all-linear" 或线性投影层列表。 (LoRA target modules: "all-linear" or list of linear projection layers.) |
| `critic.ppo_mini_batch_size` | 每次更新的 PPO 小批量大小。 (PPO mini-batch size per update.) |
| `critic.ppo_micro_batch_size` | [已弃用] 全局微批次大小。 ([Deprecated] Global micro batch size.) |
| `critic.ppo_micro_batch_size_per_gpu` | 每个 GPU 的本地微批次大小。 (Local per-GPU micro batch size.) |
| `critic.forward_micro_batch_size` | 仅前向传播的批次大小（全局）。 (Forward-only batch size (global).) |
| `critic.forward_micro_batch_size_per_gpu` | 仅前向传播的批次大小（每个 GPU）。 (Forward-only batch size (per GPU).) |
| `critic.use_dynamic_bsz` | 是否在运行时动态调整批次大小。 (Whether to automatically adjust batch size at runtime.) |
| `critic.ppo_max_token_len_per_gpu` | 单个 PPO 批次中每个 GPU 的最大词元数（对 critic 加倍）。 (Max tokens per GPU in one PPO batch (doubled for critic).) |
| `critic.forward_max_token_len_per_gpu` | 前向传播中每个 GPU 的最大词元长度。 (Max token length per GPU in forward pass.) |
| `critic.ulysses_sequence_parallel_size` | Ulysses-style 模型并行的序列并行大小。 (Sequence parallelism size for Ulysses-style model parallelism.) |
| `critic.ppo_epochs` | 每个批次的 PPO 轮数。 (Number of PPO epochs per batch.) |
| `critic.shuffle` | 在 PPO 轮次之间打乱训练数据。 (Shuffle training data across PPO epochs.) |
| `critic.grad_clip` | Critic 更新的梯度裁剪。 (Gradient clipping for critic updates.) |
| `critic.cliprange_value` | PPO 值函数裁剪范围。 (PPO value function clipping range.) |
| `critic.loss_agg_mode` | 损失聚合模式。 (Loss aggregation mode: "token-mean", "seq-mean-token-sum", or "seq-mean-token-mean".) |
| `critic.checkpoint.save_contents` | 保存的检查点中包含的内容。 (What to include in saved checkpoints.) |
| `critic.checkpoint.load_contents` | 加载检查点时包含的内容。 (What to include when loading checkpoints.) |
| `critic.profiler.discrete` | True 表示每个任务有自己的数据库，False 表示所有任务共享一个。 (True for each task has its own database, False for all tasks in one training step share one database.) |
| `critic.profiler.profile_all_ranks` | 是否对所有 rank 进行性能分析。 (Whether to profile all ranks.) |

### 4. Reward 模型 (`reward_model`)

| 参数名称 (Parameter Name) | 描述 (Description) |
| --- | --- |
| `reward_model.enable` | 是否启用奖励模型。如果为 False，则仅使用用户定义的奖励函数计算奖励。 (Whether to enable reward model. If False, we compute the reward only with the user-defined reward functions.) |
| `reward_model.strategy` | FSDP 策略："fsdp" 或 "fsdp2"或"megatron"。 (FSDP strategy: "fsdp" or "fsdp2" or "megatron".) |
| `reward_model.model.input_tokenizer` | 输入分词器。如果奖励模型的聊天模板与策略不一致，则需要此项。 (Input tokenizer. If the reward model’s chat template is inconsistent with the policy, we need to first decode to plaintext, then apply the rm’s chat_template.) |
| `reward_model.model.path` | RM 的 HDFS 路径或本地路径。仅支持 AutoModelForSequenceClassification。 (RM’s HDFS path or local path. Note that RM only supports AutoModelForSequenceClassification.) |
| `reward_model.model.use_shm` | 是否使用共享内存加载模型。 (Whether to use shared memory for loading the model.) |
| `reward_model.model.external_lib` | 外部模型实现（可选）。 (External model implementation (optional).) |
| `reward_model.model.use_remove_padding` | 使用移除填充优化（节省计算）。 (Use remove padding optimization (saves compute).) |
| `reward_model.model.use_fused_kernels` | 是否使用融合的奖励核以加速。 (Whether to use fused reward kernels for speedup.) |
| `reward_model.model.trust_remote_code` | 是否允许加载远程代码模型，默认为 False。 (Whether to enable loading a remote code model, default to False.) |
| `reward_model.model.fsdp_config.wrap_policy.min_num_params` | 触发 FSDP 包装的最小参数数量。 (Minimum number of parameters to trigger wrapping.) |
| `reward_model.model.fsdp_config.param_offload` | 是否将模型参数卸载到 CPU。 (Whether to offload model parameters to CPU.) |
| `reward_model.model.fsdp_config.reshard_after_forward` | 仅用于 FSDP2：前向传播后重新分片以减少内存占用。 (Only for FSDP2: Reshard after forward pass to reduce memory footprint.) |
| `reward_model.model.fsdp_config.fsdp_size` | 每个 FSDP 分片组中的 GPU 数量；-1 表示自动。 (Number of GPUs in each FSDP shard group; -1 means auto.) |
| `reward_model.model.fsdp_config.forward_prefetch` | 仅用于 FSDP1：在前向计算完成前预取下一次前向传播的 all-gather。 (Only for FSDP1: FSDP1 configuration, prefetch the next forward-pass all-gather before the current forward computation.) |
| `reward_model.micro_batch_size` | [已弃用] 全局微批次大小。 ([Deprecated] Global micro batch size.) |
| `reward_model.micro_batch_size_per_gpu` | 每个 GPU 的本地微批次大小。 (Local per-GPU micro batch size.) |
| `reward_model.max_length` | 用于评分处理的最大序列长度。 (Maximum sequence length to process for scoring.) |
| `reward_model.ulysses_sequence_parallel_size` | Ulysses-style 模型并行的序列并行大小。 (Sequence parallelism size for Ulysses-style model parallelism.) |
| `reward_model.use_dynamic_bsz` | 是否在运行时动态调整批次大小。 (Whether to dynamically adjust batch size at runtime.) |
| `reward_model.forward_max_token_len_per_gpu` | 单次前向传播中每个 GPU 的最大词元数。 (Maximum number of tokens per GPU in one forward pass.) |
| `reward_model.reward_manager` | 定义计算基于规则的奖励和处理不同奖励源的机制。 (This defines the mechanism of computing rule-based reward and handling different reward sources.) |
| `reward_model.launch_reward_fn_async` | 是否在 log_prob 期间异步启动自定义奖励函数。 (Whether to launch custom reward function asynchronously during log_prob.) |
| `reward_model.sandbox_fusion.url` | 用于沙箱执行的云/本地函数 URL。 (Cloud/local function URL for sandbox execution.) |
| `reward_model.sandbox_fusion.max_concurrent` | 允许到沙箱的最大并发请求数。 (Max concurrent requests allowed to sandbox.) |
| `reward_model.profiler.discrete` | True 表示每个任务有自己的数据库，False 表示所有任务共享一个。 (True for each task has its own database, False for all tasks in one training step share one database.) |
| `reward_model.profiler.all_ranks` | 是否对所有 rank 进行性能分析。 (Whether to profile all ranks.) |
| `reward_model.profiler.ranks` | 将被分析的 rank。null 或 [0,1,...]。 (The ranks that will be profiled. null or [0,1,...].) |

### 5. 自定义奖励函数 (`custom_reward_function`)

| 参数名称 (Parameter Name) | 描述 (Description) |
| --- | --- |
| `custom_reward_function.path` | 包含自定义奖励函数的文件路径。 (The path to the file containing your customized reward function.) |
| `custom_reward_function.name` | 指定文件中的奖励函数名称。默认为 'compute_score'。 (The name of the reward function within the specified file. Default is 'compute_score'.) |

### 6. 算法 (`algorithm`)

| 参数名称 (Parameter Name) | 描述 (Description) |
| --- | --- |
| `algorithm.gamma` | 未来奖励的折扣因子。 (Discount factor for future rewards.) |
| `algorithm.lam` | GAE 估计器中偏差和方差的权衡。 (Trade-off between bias and variance in the GAE estimator.) |
| `algorithm.adv_estimator` | 优势估计器类型："gae", "grpo", "reinforce_plus_plus" 等。 (Advantage estimator type: "gae", "grpo", "reinforce_plus_plus", etc.) |
| `algorithm.norm_adv_by_std_in_grpo` | 是否在 GRPO 中按标准差归一化优势。 (Whether to normalize advantages by std (specific to GRPO).) |
| `algorithm.use_kl_in_reward` | 是否在奖励中启用 KL 惩罚。 (Whether to enable in-reward KL penalty.) |
| `algorithm.kl_penalty` | 如何估计 KL 散度："kl", "abs", "mse", "low_var_kl", 或 "full"。 (How to estimate KL divergence: "kl", "abs", "mse", "low_var_kl", or "full".) |
| `algorithm.kl_ctrl.type` | KL 控制类型："fixed" 或 "adaptive"。 (KL control type: "fixed" or "adaptive".) |
| `algorithm.kl_ctrl.kl_coef` | KL 惩罚的初始系数。 (Initial coefficient for KL penalty.) |
| `algorithm.kl_ctrl.horizon` | 自适应控制器的 horizon 值（如果启用）。 (Horizon value for adaptive controller (if enabled).) |
| `algorithm.kl_ctrl.target_kl` | 目标 KL 散度（用于自适应控制器）。 (Target KL divergence (used for adaptive controller).) |
| `algorithm.use_pf_ppo` | 是否启用偏好反馈 PPO。 (Whether to enable preference feedback PPO.) |
| `algorithm.pf_ppo.reweight_method` | 样本重加权方法："pow", "max_min", 或 "max_random"。 (Method for reweighting samples: "pow", "max_min", or "max_random".) |
| `algorithm.pf_ppo.weight_pow` | "pow" 方法中用于权重缩放的幂。 (Power used for weight scaling in "pow" method.) |

### 7. 训练器 (`trainer`)

| 参数名称 (Parameter Name) | 描述 (Description) |
| --- | --- |
| `trainer.balance_batch` | 是否在分布式工作节点间平衡批次大小。 (Whether to balance batch sizes across distributed workers.) |
| `trainer.total_epochs` | 训练的总轮数。 (Number of epochs in training.) |
| `trainer.total_training_steps` | 总训练步数（可显式设置或从轮数派生）。 (Total training steps (can be set explicitly or derived from epochs).) |
| `trainer.profile_steps` | 将被分析的步骤。null 表示不进行分析。 (The steps that will be profiled. null means no profiling. null or [1,2,5,...].) |
| `trainer.controller_nsight_options.trace` | 对于controller进程，选择要追踪的 API（比如cuda，nvtx，cublas，etc）。 (Select the API(s) to be traced.) |
| `trainer.controller_nsight_options.cuda-memory-usage` | 对于controller进程，是否profile CUDA 内存使用情况。必须是字符串 "true" 或 "false"。 (Track the GPU memory usage by CUDA kernels. Must be string type "true" or "false".) |
| `trainer.controller_nsight_options.cuda-graph-trace` | 对于controller进程，是否将CUDA graphs 将被作为一个整体进行追踪。 (CUDA graphs will be traced as a whole.) |
| `trainer.worker_nsight_options.trace` | 对于worker进程，选择要追踪的 API。 (Select the API(s) to be traced.) |
| `trainer.worker_nsight_options.cuda-memory-usage` | 对于worker进程，是否profile CUDA 内存使用情况。必须是字符串 "true" 或 "false"。 (Track the GPU memory usage by CUDA kernels. Must be string type "true" or "false".) |
| `trainer.worker_nsight_options.cuda-graph-trace` | 对于worker进程，是否CUDA graphs 将被作为一个整体进行追踪。 (CUDA graphs will be traced as a whole.) |
| `trainer.worker_nsight_options.capture-range` | 仅在 torch.cuda.profiler.start 和 stop 范围内进行分析。默认值为cudaProfilerApi，不要更改此配置。 (Profiling only in a range of torch.cuda.profiler.start and stop. Do not change this config.) |
| `trainer.worker_nsight_options.capture-range-end` | 指定捕获范围结束时的期望行为。 (Specify the desired behavior when a capture range ends.) |
| `trainer.worker_nsight_options.kill` | 向目标应用程序的进程组发送信号。我们让程序自行退出。 (Send signal to the target application's process group. We let the program to exit by itself.) |
| `trainer.project_name` | 用于实验跟踪（如 wandb）的项目名称。 (Project name for experiment tracking (e.g., wandb).) |
| `trainer.experiment_name` | 用于在跟踪工具中识别运行的实验名称。 (Experiment name for run identification in tracking tools.) |
| `trainer.logger` | 使用的日志后端："console", "wandb" 等。 (Logging backends to use: "console", "wandb", etc.) |
| `trainer.log_val_generations` | 验证期间要记录的生成数量。 (Number of generations to log during validation.) |
| `trainer.rollout_data_dir` | 用于记录 rollout 数据的目录；如果为 null 则不转储。 (Directory for logging rollout data; no dump if null.) |
| `trainer.validation_data_dir` | 用于记录验证数据的目录；如果为 null 则不转储。 (Directory for logging validation data; no dump if null.) |
| `trainer.nnodes` | 训练中使用的节点数。 (Number of nodes used in the training.) |
| `trainer.n_gpus_per_node` | 每个节点的 GPU 数量。 (Number of GPUs per node.) |
| `trainer.save_freq` | 模型检查点的保存频率（按迭代次数）。 (Save frequency (by iteration) for model checkpoints.) |
| `trainer.resume_mode` | 恢复模式："auto", "disable", 或 "resume_path"。 (Resume mode: "auto", "disable", or "resume_path".) |
| `trainer.resume_from_path` | 从该路径恢复训练（仅当 resume_mode 为 "resume_path" 时使用）。 (Path to resume training from (only used when resume_mode is "resume_path").) |
| `trainer.val_before_train` | 是否在训练开始前运行验证。 (Whether to run validation before training begins.) |
| `trainer.val_only` | 是否只运行验证。 (Whether to run validation only.) |
| `trainer.test_freq` | 验证频率（以训练迭代次数计）。 (Validation frequency (in training iterations).) |
| `trainer.critic_warmup` | 在更新策略之前预热 critic 的迭代次数。 (Number of iterations to warm up the critic before updating policy.) |
| `trainer.default_hdfs_dir` | 用于保存检查点的默认分布式文件系统路径。 (Default path to distributed filesystem for saving checkpoints.) |
| `trainer.del_local_ckpt_after_load` | 加载后是否删除本地检查点。 (Whether to delete local checkpoints after loading.) |
| `trainer.default_local_dir` | 用于保存检查点的默认本地目录。 (Default local directory for saving checkpoints.) |
| `trainer.max_actor_ckpt_to_keep` | 保留的 actor 检查点的最大数量。 (Maximum number of actor checkpoints to keep.) |
| `trainer.max_critic_ckpt_to_keep` | 保留的 critic 检查点的最大数量。 (Maximum number of critic checkpoints to keep.) |
| `trainer.ray_wait_register_center_timeout` | Ray worker 等待注册的超时时间（秒）。 (Timeout (in seconds) for Ray worker to wait for registration.) |
| `trainer.device` | 运行训练的设备（如 "cuda", "cpu"）。 (Device to run training on (e.g., "cuda", "cpu").) |

### 8. Ray 初始化 (`ray_init`)

| 参数名称 (Parameter Name) | 描述 (Description) |
| --- | --- |
| `ray_init.num_cpus` | Ray 使用的 CPU 数量。使用 SLURM 时应使用固定数字而不是 null。 (Number of CPUs for Ray. Use a fixed number instead of null when using SLURM.) |
| `ray_init.timeline_json_file` | 保存 Ray 时间线 JSON 文件以进行性能分析的路径。 (Path to save Ray timeline JSON for performance profiling.) |