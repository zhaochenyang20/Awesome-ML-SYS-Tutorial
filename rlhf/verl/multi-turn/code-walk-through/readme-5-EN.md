# verl Parameter Handbook

Due to the use of [Hydra](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/code-walk-through/readme.md#%E8%84%9A%E6%9C%AC%E9%85%8D%E7%BD%AE), verl's parameters are scattered throughout the framework, and the official parameter documentation lacks updates. Therefore, our SGLang RL team, in collaboration with the Amazon AGI SF Lab, has compiled a quick look at verl's parameters. Given the large number of parameters, it is difficult for us to guarantee that every interpretation is absolutely correct, but each has been repeatedly reviewed by us. After careful consideration, we decided to share this parameter quick look with the community, hoping it will be helpful to everyone. The contributors to this guide are:

Ji Li (Ant), Zhuoran Yin (CMU), Changyi Yang (CMU), Chengxi Li (CMU), Xinpeng Wei (Amazon), Chenyang Zhao (Amazon)

## Batch Size

| Parameter Name | Detailed Explanation |
| --- | --- |
| `data.train_batch_size` | **Function**: Defines the number of samples sent to the Rollout Engine in a single training step. This is also the number of prompts sampled from the training dataset at the beginning of each PPO iteration.\<br\>\<br\>**Detailed Explanation**: This value is the fundamental sample count in RL training. For example, setting it to 1024 means that in one iteration:\<br\>1. 1024 prompts are randomly drawn from the dataset.\<br\> 2. These 1024 prompts are sent to the current Rollout Engine to generate 1024 complete trajectories (prompt, response).\<br\>3. Next, these 1024 trajectories undergo experience calculation ("make experience"), which is subsequently used for updating the Actor and Critic models.\<br\>\<br\>**Impact and Trade-offs**: Affects the total number of samples trained. |
| `data.val_batch_size` (Deprecated) | **Function**: The batch size used during the validation phase.\<br\>\<br\>**Detailed Explanation**: Similar to `train_batch_size`, but only used for evaluating model performance and not for training. If set to `null`, the size of the validation set is used as the default. Note: This has been deprecated. It is recommended to set it to null. In this case, the entire validation dataset is sent to the SGLang engines at once, which will manage memory themselves.|
| `actor_rollout_ref.actor.ppo_mini_batch_size` \<br\> `critic.ppo_mini_batch_size` | **Function**: Defines the mini-batch size for PPO training updates.\<br\>\<br\>**Detailed Explanation**: All the experience data collected with `data.train_batch_size` will be split into multiple mini-batches, with the size of each being `ppo_mini_batch_size`. The model performs a parameter update only after processing one mini-batch.\<br\>For example, if `train_batch_size = 1024` and `ppo_mini_batch_size = 256`, the model will perform `1024 / 256 = 4` parameter updates in one PPO Epoch.\<br\>\<br\>**Impact and Trade-offs**: Increasing the mini-batch size makes the gradients for a single update more stable, but it reduces the update frequency and the total number of updates. |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` \<br\> `critic.ppo_micro_batch_size_per_gpu` | **Function**: Defines the size of data for a single forward/backward pass on a single GPU.\<br\>\<br\>**Detailed Explanation**: This is the core parameter for implementing gradient accumulation. A mini-batch is further divided into several micro-batches. For example, on a single GPU, if `ppo_mini_batch_size = 256` and `ppo_micro_batch_size_per_gpu = 32`, the number of gradient accumulation steps is `256 / 32 = 8`. This means the model will run 8 forward passes to get the loss, then backward passes to get the gradient, processing 32 samples each time, until the gradients for the entire mini-batch are accumulated. Then, the accumulated total gradient is used to update the model parameters once (`optimizer.step()`). This value must be strictly adjusted based on GPU memory size and is key to preventing OOM (Out of Memory) errors.\<br\>\<br\>**Impact and Trade-offs**: Increasing this value reduces the number of gradient accumulation steps, which can improve training throughput but increases GPU memory consumption. |
| `actor_rollout_ref.actor.ppo_micro_batch_size` \<br\> `critic.ppo_micro_batch_size` (Deprecated) | **Function**: Deprecated. Replaced by the `per_gpu` version as it better accommodates distributed training environments. |

## Dynamic Batch Size

When sample lengths vary significantly, batching by the number of samples can lead to highly imbalanced computational loads across different batches. Controlling the batch size based on the total number of tokens is a solution to balance the training time for each batch.

| Parameter Name | Detailed Explanation |
| --- | --- |
| `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` \<br\> `critic.ppo_max_token_len_per_gpu` | **Function**: Defines the maximum total number of tokens that a single GPU can process in one PPO micro-batch.\<br\>\<br\>**Detailed Explanation**: This is an alternative to `ppo_micro_batch_size_per_gpu` and is used in conjunction with `use_dynamic_bsz`. The system automatically packs samples until the total token count (`prompt_len + response_len`) approaches this threshold, forming a dynamic micro-batch size. This helps stabilize computational efficiency; the computational load for each micro-batch remains relatively constant, regardless of sample length.\<br\>For example, if `actor_rollout_ref.actor.ppo_max_token_len_per_gpu = 16384`, the system might pack 16 samples of length 1024 (16 \* 1024 = 16384) or 64 samples of length 256 (64 \* 256 = 16384).\<br\>\<br\>**Impact and Trade-offs**: Generally more efficient than fixed-sample-count micro-batches, leading to better utilization of computational resources and reducing GPU instability. Typically set to `n * ({data.max_prompt_length} + {data.max_response_length})`. |
| `reward_model.forward_max_token_len_per_gpu` \<br\> `critic.forward_max_token_len_per_gpu` \<br\> `actor_rollout_ref.ref.log_prob_max_token_len_per_gpu` | **Function**: The maximum number of tokens in a micro-batch for models that only perform forward computations.\<br\>\<br\>**Detailed Explanation**: Some models (Reward Model, Critic for value calculation, Reference Model for log probs) only perform forward passes during the "make experience" phase. At this point, the rollout engine has been offloaded, and the training engine has not yet started, resulting in very low GPU memory usage. Therefore, a larger batch size can be set for them to accelerate computation. These parameters are also part of `use_dynamic_bsz` and are used to optimize the execution efficiency of these specific tasks. |
| `critic.forward_micro_batch_size_per_gpu` \<br\> `reward_model.micro_batch_size_per_gpu` \<br\> `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu` | **Function**: Similarly, sets the micro-batch size for models that only perform forward computations.\<br\>\<br\>**Detailed Explanation**: Same as the parameter in the row above. |
| `actor_rollout_ref.actor.use_dynamic_bsz` \<br\> `critic.use_dynamic_bsz` \<br\> `reward_model.use_dynamic_bsz` | **Function**: Whether to enable Dynamic Batch Size.\<br\>\<br\>**Detailed Explanation**: When this is `True`, the system ignores the sample-based `micro_batch_size_per_gpu` parameter and instead uses the token-based `max_token_len_per_gpu` parameter to construct batches. |
| `trainer.balance_batch` | **Function**: Whether to balance the batch size across different data parallel (dp) ranks in distributed training.\<br\>\<br\>**Detailed Explanation**: Reorders data on a single controller to ensure that each dp rank receives a similar number of tokens. |

## Rollout Sampling Parameters

| Parameter Name | Function and Explanation |
| --- | --- |
| `actor_rollout_ref.rollout.temperature` | A higher temperature value smooths the probability distribution, leading to more diverse and random generated results. A lower value sharpens the distribution, making the output more deterministic and conservative, favoring high-probability tokens. `temperature=0` is usually equivalent to Greedy Decoding. |
| `actor_rollout_ref.rollout.top_k` | At each generation step, only the K most probable tokens are considered for sampling. For example, `top_k=50` means selecting only from the top 50 most likely tokens.\<br\>- To disable: Set to `0` or `None` in Hugging Face, or `-1` in SGLang (which samples from the entire vocabulary).|
| `actor_rollout_ref.rollout.top_p` | Cumulatively sums the probabilities of the most likely tokens until the total probability reaches P, then samples from this nucleus set of tokens. It is a dynamic method for selecting the sampling range. `top_p=1.0` means no restriction. |
| `actor_rollout_ref.rollout.use_fire_sampling` | Whether to use Fire Sampling, from a [paper](https://arxiv.org/abs/2410.21236) by ByteDance. |
| `actor_rollout_ref.rollout.n` | The number of responses generated for each prompt, also known as the group size in GRPO.|
| `actor_rollout_ref.rollout.ignore_eos` | Whether to ignore the EOS (End-of-Sentence) token. If `True`, generation continues until `max_response_length` is reached, even if the model produces an EOS token. |

## Performance and Resource Management

| Parameter Name | Function and Explanation |
| --- | --- |
| `actor_rollout_ref.rollout.prompt_length` | The maximum prompt length. Prompts longer than this are truncated. |
| `actor_rollout_ref.rollout.response_length` | The maximum response length. The SGLang engine will return immediately upon reaching this length. |
| `actor_rollout_ref.rollout.dtype` | Model data type, e.g., `bfloat16`, `float16`. This needs to be aligned with the model type used in the training phase; otherwise, quantization will be required when updating model parameters. |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | In SGLang, this is the proportion of GPU memory occupied by model parameters and the KV Cache. If using SGLang version 0.4.8.post1 or higher, this can be set to around 0.85. For older versions, it should be set to around 0.5. |
| `actor_rollout_ref.rollout.free_cache_engine` | Whether to free the engine cache after a rollout. Enabling this option in SGLang triggers the `flush_cache()` operation, which clears the KV cache pool and marks all slots as available. This releases the logical occupation of the KV Cache without freeing the physical GPU memory. For why flushing the KV cache is needed, see [here](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/code-walk-through/readme.md#sglangrolloutasyncengine).|
| `actor_rollout_ref.rollout.load_format` | Model weight loading mode. E.g., `dummy_dtensor` (randomly initialized weights for quick debugging), `hf`, `safetensors` (recommended for safety and efficiency). |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` (TP\_SIZE) | Tensor model parallel size, indicating how many GPUs are used to run a single SGLang engine. For example, `TP_SIZE=4` means splitting a large model's weights into 4 parts, with 4 GPUs collaborating on inference. |
| `actor_rollout_ref.rollout.max_model_len` | The maximum total length (prompt + response) the model can handle. If not set, it is usually determined by the model's configuration. |
| `actor_rollout_ref.rollout.max_num_seqs` | The maximum number of requests the engine can process concurrently, or the maximum number of prompts being inferred simultaneously. |
| `actor_rollout_ref.rollout.enable_chunked_prefill` | Whether to enable Chunked Prefill. For very long prompts, this can split them into chunks for processing, which reduces peak memory usage at the cost of lower throughput. |
| `actor_rollout_ref.rollout.disable_log_stats` | Whether to disable the inference engine's statistical logs to reduce console output. |

-----

### SGLang Configuration

| Parameter Name | Function and Explanation |
| --- | --- |
| `actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend` | **The attention backend used by SGLang**. Options like `flashinfer`, `triton`, `flashmla`, `null` are available to suit different graphics cards. |

-----

### multi-turn tool calling

These parameters are primarily for scenarios requiring multi-turn interactions, such as tool calling or continuous dialogue, supported by the SGLang Engine.

| Parameter Name | Function and Explanation |
| --- | --- |
| `actor_rollout_ref.rollout.multi_turn.enable` | Whether to enable multi-turn dialogue mode. |
| `actor_rollout_ref.rollout.multi_turn.max_turns` | The maximum number of tool calling rounds. If null, it defaults to `max_model_len // 3` to prevent infinite dialogues.|
| `actor_rollout_ref.rollout.multi_turn.tool_config_path` | The path to the tool configuration file, which defines the external tools the model can call. |
| `actor_rollout_ref.rollout.multi_turn.completion_callback` | A custom callback function that can execute custom logic after each generation round. |
| `actor_rollout_ref.rollout.multi_turn.use_inference_chat_template` | Whether to use the model's chat template from the inference phase. `True` means following the inference-stage template format. `False` means using the template from pre-training, which may contain a complete token sequence with an additional thought process. For any model, it is crucial to ensure consistent templates are used during post-training and subsequent inference testing stages. |
| `actor_rollout_ref.rollout.multi_turn.enable_tokenization_sanity_check` | Whether to perform a tokenization sanity check, verifying that the result of tokenizing turn-by-turn is consistent with tokenizing the entire chat history at once. |

### Validation Stage Configuration

| Parameter Name | Function and Explanation |
| --- | --- |
| `actor_rollout_ref.rollout.val_kwargs.*` | Sampling parameters for the validation phase. This allows us to use different sampling parameters during post-training and validation. For example, during validation, it is common to set `temperature=0` and `do_sample=False` for greedy decoding to obtain more stable evaluation results. |

### Dataset

| Parameter Name | Function and Explanation |
| --- | --- |
| `data.tokenizer` | The class or path of the tokenizer. If null, it will be automatically inferred from the model. |
| `data.use_shm` | Whether to use shared memory (SHM) to load data. |
| `data.train_files` | Training set Parquet files. Can be a list or a single file; paths can be local or HDFS paths. |
| `data.val_files` | Validation set Parquet files. Can be a list or a single file. |
| `data.prompt_key` | The field for the prompt in the dataset. Defaults to `prompt`. |
| `data.reward_fn_key` | The field used to select the reward function (if different reward functions are used for each sample). |
| `data.max_prompt_length` | Maximum prompt length. All prompts will be left-padded to this length. |
| `data.return_raw_input_ids` | Whether to return the raw `input_ids` without the chat template applied; used when the reward model's chat template differs from the policy model's. |
| `data.return_raw_chat` | Whether to return the raw response without the chat template applied. |
| `data.return_full_prompt` | Whether to return the full prompt with the chat template applied. |
| `data.shuffle` | Whether to shuffle the data in the DataLoader. |
| `data.validation_shuffle` | Whether to shuffle the validation set. |
| `data.filter_overlong_prompts` | Whether to filter out overly long prompts. |
| `data.filter_overlong_prompts_workers` | The number of worker processes for filtering overly long prompts. Use multiple processes for large datasets to speed up. Defaults to 1. |
| `data.truncation` | Truncate if `input_ids` or `prompt` exceeds the maximum length. |
| `data.image_key` | The field representing images in a multi-modal dataset. Defaults to `images`. |
| `data.video_key` | The field representing videos in a multi-modal dataset. |
| `data.trust_remote_code` | Whether to trust the local Hugging Face cache; note, this 'remote' is relative to Hugging Face, so this parameter considers "whether to trust local." |
| `data.custom_cls.path` | The file path containing the custom dataset class. If not specified, a pre-implemented default dataset will be used. |
| `data.custom_cls.name` | The name of the dataset class in the specified file. |

### Actor, Rollout & Reference Worker Configuration

The parameters for Critic and Actor are very consistent and will not be repeated.

| Parameter Name | Description |
| ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `actor_rollout_ref.hybrid_engine`                                            | Currently only supports hybrid engine, which places the actor and rollout models on the same resource group. |
| `actor_rollout_ref.model.path`                                               | Hugging Face model path. Can be a local path or an HDFS path. |
| `actor_rollout_ref.model.use_shm`                                            | Whether to use shared memory (SHM) to accelerate model weight loading. |
| `actor_rollout_ref.model.external_lib`                                       | Additional Python packages for registering Hugging Face models/tokenizers. |
| `actor_rollout_ref.model.override_config`                                    | Used to override the model's original configuration, mainly for dropout. |
| `actor_rollout_ref.model.enable_gradient_checkpointing`                      | Whether to recompute gradients during actor training, trading time for space. |
| `actor_rollout_ref.model.enable_activation_offload`                          | Whether to offload activations to the CPU during actor training. |
| `actor_rollout_ref.model.use_remove_padding`                                 | Whether to remove padding tokens from the input during training. |
| `actor_rollout_ref.model.use_liger`                                          | Whether to use the Liger kernel for linear layer fusion. |
| `actor_rollout_ref.model.use_fused_kernels`                                  | Whether to use custom fused kernels (e.g., FlashAttention, fused MLP). |
| `actor_rollout_ref.model.fused_kernel_options.impl_backend`                  | The implementation backend for fused kernels, either `triton` or `torch`. Must be used with `use_fused_kernels`. |
| `actor_rollout_ref.model.trust_remote_code`                                  | Whether to trust the local Hugging Face cache; note, this 'remote' is relative to Hugging Face, so this parameter considers "whether to trust local." |
| `actor_rollout_ref.actor.strategy`                                           | Training backend: `fsdp`, `fsdp2`, or `megatron`. |
| `actor_rollout_ref.actor.grad_clip`                                          | Gradient clipping for Actor updates. |
| `actor_rollout_ref.actor.clip_ratio`                                         | PPO clipping ratio. |
| `actor_rollout_ref.actor.clip_ratio_low`                                     | The lower bound for asymmetric clipping (for dual-clip PPO). |
| `actor_rollout_ref.actor.clip_ratio_high`                                    | The upper bound for asymmetric clipping (for dual-clip PPO). |
| `actor_rollout_ref.actor.clip_ratio_c`                                       | The constant C in dual-clip PPO; clipping occurs when advantage \< -C. |
| `actor_rollout_ref.actor.loss_agg_mode`                                      | Loss aggregation mode: `token-mean`, `seq-mean-token-sum`, or `seq-mean-token-mean`. |
| `actor_rollout_ref.actor.entropy_coeff`                                      | The entropy regularization coefficient in the PPO loss. |
| `actor_rollout_ref.actor.use_kl_loss`                                        | Whether to use KL loss instead of a KL reward penalty. True for GRPO. |
| `actor_rollout_ref.actor.use_torch_compile`                                  | Whether to use `torch.compile()`. |
| `actor_rollout_ref.actor.kl_loss_coef`                                       | The KL loss coefficient when `use_kl_loss` is enabled, used for GRPO. |
| `actor_rollout_ref.actor.kl_loss_type`                                       | The type of KL divergence loss. Options: `kl`, `abs`, `mse`, `low_var_kl`, `full`. |
| `actor_rollout_ref.actor.ppo_epochs`                                         | The number of PPO epochs. |
| `actor_rollout_ref.actor.shuffle`                                            | Shuffle the training data. |
| `actor_rollout_ref.actor.ulysses_sequence_parallel_size`                     | The sequence parallel size for Ulysses-style parallelism. |
| `actor_rollout_ref.actor.entropy_from_logits_with_chunking`                  | Compute entropy via chunking to reduce peak memory usage. |
| `actor_rollout_ref.actor.entropy_checkpointing`                              | Whether to save entropy via checkpointing. |
| `actor_rollout_ref.actor.checkpoint.save_contents`                           | The contents to be included in the saved checkpoint. |
| `actor_rollout_ref.actor.checkpoint.load_contents`                           | The specific contents to load from a checkpoint. |
| `actor_rollout_ref.actor.optim.lr`                                           | Learning rate. |
| `actor_rollout_ref.actor.optim.lr_warmup_steps`                              | Number of warmup steps; a negative value means it's determined by `lr_warmup_steps_ratio`. |
| `actor_rollout_ref.actor.optim.lr_warmup_steps_ratio`                        | The ratio of warmup steps (used when `lr_warmup_steps` is negative). |
| `actor_rollout_ref.actor.optim.min_lr_ratio`                                 | The minimum learning rate ratio for the cosine scheduler. |
| `actor_rollout_ref.actor.optim.num_cycles`                                   | The number of cosine cycles in the learning rate schedule. |
| `actor_rollout_ref.actor.optim.warmup_style`                                 | Learning rate warmup style: `constant` or `cosine`. |
| `actor_rollout_ref.actor.optim.total_training_steps`                         | Total number of training steps. |
| `actor_rollout_ref.actor.optim.weight_decay`                                 | Weight decay coefficient, controlling the strength of L2 regularization applied to weights during training. |
| `actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params`             | The minimum number of parameters to trigger FSDP wrapping for a layer. |
| `actor_rollout_ref.actor.fsdp_config.param_offload`                          | Whether to offload model parameters to the CPU (trading speed for memory). |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload`                      | Whether to offload optimizer states to the CPU. |
| `actor_rollout_ref.actor.fsdp_config.offload_policy`                         | For FSDP2 only: Offload parameters/gradients/optimizer during training. |
| `actor_rollout_ref.actor.fsdp_config.reshard_after_forward`                  | For FSDP2 only: Reshard after the forward pass to reduce memory usage. |
| `actor_rollout_ref.actor.fsdp_config.fsdp_size`                              | The number of GPUs in each FSDP sharding group; -1 means automatic. |
| `actor_rollout_ref.actor.fsdp_config.forward_prefetch`                       | For FSDP1 only: Prefetch the all-gather for the next forward pass before the current one completes. |
| `actor_rollout_ref.actor.profiler.discrete`                                  | `True` means each task has its own database; `False` means all tasks share one. |
| `actor_rollout_ref.actor.profiler.all_ranks`                                 | Whether to profile all ranks. |
| `actor_rollout_ref.actor.profiler.ranks`                                     | The ranks to be profiled. `null` or `[0,1,...]`. |
| `actor_rollout_ref.ref.strategy`                                             | FSDP configuration for the Reference model, same as the actor. |
| `actor_rollout_ref.ref.fsdp_config.param_offload`                            | Whether to offload parameters in FSDP. |
| `actor_rollout_ref.ref.fsdp_config.reshard_after_forward`                    | For FSDP2 only: Whether to reshard after the model's forward pass to save memory. |
| `actor_rollout_ref.ref.fsdp_config.forward_prefetch`                         | For FSDP1 only: Prefetch the all-gather for the next forward pass before the current one completes. |
| `actor_rollout_ref.ref.fsdp_config.wrap_policy.min_num_params`               | The minimum number of parameters in an FSDP-wrapped module. |
| `actor_rollout_ref.ref.profiler.discrete`                                    | `True` means each task has its own database; `False` means all tasks share one. |
| `actor_rollout_ref.ref.profiler.all_ranks`                                   | Whether to profile all ranks. |
| `actor_rollout_ref.ref.profiler.ranks`                                       | The ranks to be profiled. `null` or `[0,1,...]`. |

### Reward Model

| Parameter Name | Description |
| --- | --- |
| `reward_model.enable` | Whether to enable the reward model. If `False`, rewards are calculated only using user-defined reward functions. |
| `reward_model.strategy` | FSDP strategy: `fsdp`, `fsdp2`, or `megatron`. |
| `reward_model.model.input_tokenizer` | Input tokenizer. Required if the reward model's chat template is inconsistent with the policy's. |
| `reward_model.model.path` | The HDFS or local path to the RM. Only `AutoModelForSequenceClassification` is supported. |
| `reward_model.model.use_shm` | Whether to use shared memory to load the model. |
| `reward_model.model.external_lib` | External model implementation (optional). |
| `reward_model.model.use_remove_padding` | Use remove padding optimization (saves computation). |
| `reward_model.model.use_fused_kernels` | Whether to use fused reward kernels for acceleration. |
| `reward_model.model.trust_remote_code` | Whether to allow loading models with remote code, defaults to `False`. |
| `reward_model.model.fsdp_config.wrap_policy.min_num_params` | The minimum number of parameters to trigger FSDP wrapping. |
| `reward_model.model.fsdp_config.param_offload` | Whether to offload model parameters to the CPU. |
| `reward_model.model.fsdp_config.reshard_after_forward` | For FSDP2 only: Reshard after the forward pass to reduce memory usage. |
| `reward_model.model.fsdp_config.fsdp_size` | The number of GPUs in each FSDP sharding group; -1 means automatic. |
| `reward_model.model.fsdp_config.forward_prefetch` | For FSDP1 only: Prefetch the all-gather for the next forward pass before the current one completes. |
| `reward_model.reward_manager` | Defines the mechanism for calculating rule-based rewards and handling different reward sources. |
| `reward_model.launch_reward_fn_async` | Whether to launch custom reward functions asynchronously during the log\_prob phase. |
| `reward_model.sandbox_fusion.url` | The URL for remote reward functions. |
| `reward_model.sandbox_fusion.max_concurrent` | The maximum number of concurrent requests allowed to the sandbox. |
| `reward_model.profiler.discrete` | `True` means each task has its own database; `False` means all tasks share one. |

### Custom Reward Function

| Parameter Name | Description |
| --- | --- |
| `custom_reward_function.path` | The file path containing the custom reward function. |
| `custom_reward_function.name` | The name of the reward function in the specified file. Defaults to `compute_score`. |

### Algorithm

| Parameter Name | Description |
| --- | --- |
| `algorithm.gamma` | Discount factor for future rewards. |
| `algorithm.lam` | The trade-off between bias and variance in the GAE estimator. |
| `algorithm.adv_estimator` | The type of advantage estimator: `gae`, `grpo`, `reinforce_plus_plus`, etc. |
| `algorithm.norm_adv_by_std_in_grpo` | Whether to normalize advantage by its standard deviation in GRPO. |
| `algorithm.use_kl_in_reward` | Whether to enable KL penalty in the reward. |
| `algorithm.kl_penalty` | How to estimate KL divergence: `kl`, `abs`, `mse`, `low_var_kl`, or `full`. |
| `algorithm.kl_ctrl.type` | KL control type: `fixed` or `adaptive`. |
| `algorithm.kl_ctrl.kl_coef` | The initial coefficient for the KL penalty. |
| `algorithm.kl_ctrl.horizon` | The horizon value for the adaptive controller (if enabled). |
| `algorithm.kl_ctrl.target_kl` | The target KL divergence (for the adaptive controller). |
| `algorithm.use_pf_ppo` | Whether to enable Preference-Feedback PPO. |
| `algorithm.pf_ppo.reweight_method` | Sample re-weighting method: `pow`, `max_min`, or `max_random`. |
| `algorithm.pf_ppo.weight_pow` | The power used for weight scaling in the `pow` method. |

### Trainer

| Parameter Name | Description |
| --- | --- |
| `trainer.balance_batch` | Whether to balance batch sizes across distributed worker nodes. |
| `trainer.total_epochs` | The total number of training epochs. |
| `trainer.total_training_steps` | Total training steps (can be set explicitly or derived from epochs). |
| `trainer.profile_steps` | The steps to be profiled. `null` means no profiling. |
| `trainer.controller_nsight_options.trace` | For the controller process, selects the APIs to trace (e.g., cuda, nvtx, cublas, etc.). |
| `trainer.controller_nsight_options.cuda-memory-usage` | For the controller process, whether to profile CUDA memory usage. Must be the string `"true"` or `"false"`. |
| `trainer.controller_nsight_options.cuda-graph-trace` | For the controller process, whether CUDA graphs will be traced as a whole. |
| `trainer.worker_nsight_options.trace` | For worker processes, selects the APIs to trace. |
| `trainer.worker_nsight_options.cuda-memory-usage` | For worker processes, whether to profile CUDA memory usage. Must be the string `"true"` or `"false"`. |
| `trainer.worker_nsight_options.cuda-graph-trace` | For worker processes, whether CUDA graphs will be traced as a whole. |
| `trainer.worker_nsight_options.capture-range` | Profile only within the `torch.cuda.profiler.start` and `stop` range. Default is `cudaProfilerApi`, do not change this setting. |
| `trainer.worker_nsight_options.capture-range-end` | Specifies the desired behavior when the capture range ends. |
| `trainer.worker_nsight_options.kill` | Sends a signal to the target application's process group. We let the program exit on its own. |
| `trainer.project_name` | The project name for experiment tracking (e.g., wandb). |
| `trainer.experiment_name` | The experiment name to identify the run in tracking tools. |
| `trainer.logger` | The logging backend to use: `console`, `wandb`, etc. |
| `trainer.log_val_generations` | The number of generations to log during validation. |
| `trainer.rollout_data_dir` | The directory to log rollout data; if `null`, data is not dumped. |
| `trainer.validation_data_dir` | The directory to log validation data; if `null`, data is not dumped. |
| `trainer.nnodes` | The number of nodes used in training. |
| `trainer.n_gpus_per_node` | The number of GPUs per node. |
| `trainer.save_freq` | The frequency of saving model checkpoints (in number of iterations). |
| `trainer.resume_mode` | Resume mode: `auto`, `disable`, or `resume_path`. |
| `trainer.resume_from_path` | Resume training from this path (used only if `resume_mode` is `resume_path`). |
| `trainer.val_before_train` | Whether to run validation before training starts. |
| `trainer.val_only` | Whether to run only validation. |
| `trainer.test_freq` | The validation frequency (in number of training iterations). |
| `trainer.critic_warmup` | The number of iterations to pre-warm the critic before updating the policy. |
| `trainer.default_hdfs_dir` | The default distributed file system path for saving checkpoints. |
| `trainer.del_local_ckpt_after_load` | Whether to delete local checkpoints after loading. |
| `trainer.default_local_dir` | The default local directory for saving checkpoints. |
| `trainer.max_actor_ckpt_to_keep` | The maximum number of actor checkpoints to keep. |
| `trainer.max_critic_ckpt_to_keep` | The maximum number of critic checkpoints to keep. |
| `trainer.ray_wait_register_center_timeout` | The timeout (in seconds) for Ray workers to wait for registration. |
| `trainer.device` | The device to run training on (e.g., `cuda`, `cpu`). |

### Ray Init

| Parameter Name | Description |
| --- | --- |
| `ray_init.num_cpus` | The number of CPUs for Ray to use. A fixed number should be used instead of `null` when using SLURM. |
| `ray_init.timeline_json_file` | The path to save the Ray timeline JSON file for performance analysis. |
