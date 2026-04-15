# Support FSDP2 as A Training Backend for slime


> **TL;DR:**
> 
> **我们在 slime 中新增了 FSDP 作为训练后端，并与 Megatron 完成对齐。FSDP 能够更加灵活支持诸如 Qwen3-Next/gpt-oss 等架构创新的模型，并且有助于我们进一步支持 VLM RL。**

## 背景

### 什么是 FSDP？

**FSDP (Fully Sharded Data Parallel)** 继承了 [DeepSpeed ZeRO Stage 3](https://www.deepspeed.ai/2021/03/07/zero3-offload.html) 的设计哲学，可以被视为是对传统 [DDP (Distributed Data Parallel)](https://docs.pytorch.org/tutorials/beginner/ddp_series_theory.html) 的强力优化。

**从 Replicate 到 Shard**

在传统的 DDP 中，每个 GPU 都维护一份完整的模型权重、梯度和优化器状态（Replication），通过 `all-reduce` 同步梯度。而在 FSDP 中，我们转向了 **Sharding（切分）** 模式：上述所有数据都被切分并分布在不同的 GPU rank 上。

- **前向传播**：需要计算某层时，通过 `all-gather` 临时收集完整参数，计算完立即释放。
- **反向传播**：梯度计算完成后，立即进行 `reduce-scatter` 同步并切分，随即释放完整梯度。

**FSDP1 vs FSDP2**

相比于 FSDP1 将所有参数摊平成一个巨大的 `FlatParameter`，FSDP2 引入了 **DTensor (Distributed Tensor)**。它能够在保持 Tensor 原始结构（如 shape, stride）的前提下，在指定的并行维度上进行更优的切分。这不仅解决了 FSDP1 中元数据易失和 padding 复杂的痛点，更为 MixedPrecision Training 和 LoRA 提供了开箱即用的支持；本文中提到的 FSDP 均指 PyTorch 原生支持的 **FSDP2**。


> ✅ 关于 FSDP 的更多内容可以查阅 SGLang RL team 以往的博客：[**RL System Deep Dive: FSDP Training Backend**](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/sys-design/readme-2-en.md)


### 为什么 slime 需要 FSDP？

熟悉 slime 的朋友都知道，我们已经拥有了基于 Megatron-LM 的成熟训练后端。考虑到引入新后端会带来显著的维护成本，为什么我们还要坚定地支持 FSDP？

1. **VLM 架构适配**：VLM 的模态交互架构复杂，FSDP 的灵活性使其在适配上远比 Megatron 轻松。因此，我们选择 FSDP 作为 VLM RL 训练的首选路径（当然，Megatron 版本的适配也在计划中）。
2. **架构创新的敏捷性**：对于 Qwen3-Next/gpt-oss 这类处于快速迭代中的新架构，FSDP 能让我们以最快速度支持 RL 流程。
3. **低门槛与高易用性**：作为 PyTorch native 的训练后端，FSDP 没有复杂的环境依赖和安装过程。无论是学习曲线还是 debug 成本都显著低于 Megatron。
4. **无缝生态兼容**：FSDP 能直接兼容 HuggingFace Model 格式。这意味着我们无需像使用 Megatron 那样通过 `mbridge` 进行繁琐的权重转换，社区模型开盒即用。

> ⚠️ 部分模型 Megatron 现在也无需手动权重转换了，可参考 [PR link](https://github.com/THUDM/slime/pull/889)。

## FSDP in slime：架构设计

要在 slime 中同时支持 Megatron 和 FSDP 两种截然不同的分布式后端，应该如何避免底层冲突并保持代码整洁？我们采用了 "接口标准化 + 物理隔离" 的顶层设计, 也就是说向外只暴露 FSDP 的核心函数：`init, save, sleep, wake_up, train`。其他不对外暴露的函数用下划线约定，类似 `_train_core`。

此外，我们利用 Ray Actor 机制将不同后端封装在独立的进程空间中，向上层调度器暴露统一的训练原语（如 `train`），从而使上层算法逻辑无需关注底层的梯度同步细节。这种设计大幅消除了全局变量冲突并降低了条件分支复杂度，允许我们针对 FSDP2 的 Sharding 机制和 DTensor 结构进行深度优化。核心实现位于 `slime/backends/fsdp_utils/actor.py`，我们在保持对外逻辑（如 Data Packing、Context Parallel）与 Megatron 高度一致的同时，在内核实现上重构了数据流转路径，确保在享受 FSDP 灵活性的同时，最大化训练效率并维持数值精度。

完善的 FSDP 设计让顶层架构未受影响，整体流程仍旧是标准的 RLHF 循环：Rollout → Data Sharding → Packing → Forward/LogProb → Loss → Backward → Update。在此基础上，我们针对 FSDP 做了多项优化，包括 Data Packing、True On-Policy 模式、CUDA Graph Aware Weight Wake Up 以及 Training-Inference Mismatch 的众多缓解机制。我们接着讨论最顶层的 `init` 和 `train` 函数入口。

### 初始化

在 `init` 阶段，主要完成以下工作：

<p align="center">
  <img src="./pic/1_fsdp_init.png" alt="FSDP actor init 流程" width="50%" />
</p>

- **模型与优化器**：初始化 Actor Model 和 Reference Model，并支持从 Checkpoint 恢复；设置 `true_on_policy_mode` 和 Optimizer。
- Weight Updater：支持 Colocate（训练任务和推理任务放在同一组 GPU）和 Disaggregated（训练任务和推理任务放在不同 GPU）两种模式，用于将训练后的权重同步回 Inference Engine。
- Device Mesh：基于 `DeviceMesh` 构建 DP + CP 的通信拓扑，并调用 `fully_shard` 对参数进行切分。
- **算子优化**：
    - 可进一步通过 `enable_batch_invariant_mode` 强制训练端采用与 SGLang 一致的算子，消除 batch size 对计算结果的影响。
    - 利用 `torch.compile` 固化 RoPE 实现，底层消除算子行为差异，确保 True On-Policy 的对齐。

### 训练流程

`train` 函数作为训练主入口：

<p align="center">
  <img src="./pic/2_fsdp_train.png" alt="FSDP actor train 流程" width="50%" />
</p>

1. **wake up**：将之前被 Offload 的 Actor Model 加载回到 GPU。
2. **data preparation**：
    - 通过 `process_rollout_data` 获取当前 DP rank 所需数据。
    - 调用 `_pack_rollout_data` 将数据打包成 `packed_batches`（详见附录 Data Packing），消除 Padding 带来的性能损耗。
3. **forward & log prob**：
    - 计算 Actor 和 Ref 的 log_prob 与 entropy。
4. **loss calculation**：
    - 计算 PPO/GRPO loss，以及 importance ratio/clip/KL penalty/entropy bonus 等。
    - **mismatch feature**：实时计算 `train_rollout_logprob_abs_diff` 监控训练与推理的数值偏差。
    - 可选择启用 TIS (Truncated Importance Sampling)[Source](https://fengyao.notion.site/off-policy-rl#245721e3f6c48025aaeadec35aa6da9f)，对 policy gradient loss 进行重加权，减缓因为训推差异带来的 off policyness 对模型训练稳定性的影响。
5. **update & offload**：
    - 进行梯度累积和参数更新。
    - **offload 策略**：训练结束后调用 `sleep` 将模型和优化器 offload 到 CPU（colocated 模式）；Ref model 仅在计算 log prob 时加载，用完即 offload。

> ⚠️ slime 提供了完整的 training inference mismatch 解决方案，但是我们从未观察到在大规模训练中，slime 因为所谓的 training inference mismatch 而训练崩溃。具体可参考 [ref link](https://x.com/lmsysorg/status/1989181434180038797)。

## FSDP in slime 特性 & 优化

在架构设计基础上，我们进一步分享目前做出的优化。

### Data Prepare And Packing

每一轮训练开始时，FSDP actor 首先从 rollout 侧拿到一批 balanced rollout sequence，然后按 DP rank 做简单的样本拆分，这一步和常规实现没有差别。为了极致效率，我们接着实现了 [data packing](https://github.com/THUDM/slime/pull/321)。简单来说，在 `slime/backends/fsdp_utils/data_packing.py` 中处理全部的 `pack_sequences`，对于输入的一批序列，根据每条的长度和 `max_tokens_per_gpu` 估算需要多少个 pack（即 `micro-batch` 的数量）。接下来把长短不一的 sequence 分到不同 pack 中，使每个 pack 内 token 总数尽量接近。在每个 pack 内，将多条序列摊平成一条长的 tokens 向量，并构建 `cu_seqlens` 记录各条序列的起止位置。这种策略确保了每个 Pack 的 Token 总量高度一致，消除了传统 Padding 带来的算力浪费。具体细节可以参考附录。

### 严格训推一致

完成 Data Packing 后，actor 会对 packed micro‑batch 计算 ref/actor 的 log‑prob 和 entropy。我们在 FSDP 上实现了 True On Policy。也即对于近期非常火爆的 training inference mismatch 问题，我们给出了最为严格的答案，实现了同一个 policy model 在 training backend 和 inference backend 的 logp rob 绝对一致，从系统层面上解决了 training-infer mismatch。

> ✅ 简单说一下 training-infer kl = 0 的实现和思想如下:
> - Training 和 Inference 都使用 FlashAttn3 当作 backend，来实现bitwise equal
> - 使用 DeepGEMM进行矩阵乘法, Batch-invariant Kernels 实现批次不变性
> 具体细节在 slime 的 Doc 里有更详细的记载, 主要实现的 PR 是 [PR link1](https://github.com/THUDM/slime/pull/566), [PR link2](https://github.com/sgl-project/sglang/pull/12058)

<p align="center">
  <img src="./pic/3_kl_0.png" alt="training-rollout logprob diff = 0" width="50%" />
</p>


我们更进一步优化 true on policy 情况下的性能。`get_logprob_and_entropy_with_cp` 直接复用了 Rollout 传入的 temperature，并关闭了可能引入偏差的 `allow_compile` , [disable compile](https://github.com/THUDM/slime/pull/599) 会禁止 `selective_log_softmax_raw` 的编译，防止因为编译带来的偏差，确保训练端重算的 `log‑prob` 能精准还原 Rollout 时的数值表现。

> ⚠️ 在这里我们发现并解决了一个难以察觉的 Bug 导致了 use-kl-loss 的时候 on policy kl ≠ 0，详见本文末尾附录中的 PPO KL 精度误差。

### Algorithms Mitigation For Mismatch

主流算法实现并不会启用 true on policy 特性（会损失 30% 左右的训练效率），仍旧会有 training-inference mismatch。为了叙述准确，我们将 rollout 阶段记录的 rollout policy log probs，称为 `rollout_log_probs`；进入训练循环后，重计算 policy model 在 training backend 的 log probs，记录为 `old_log_probs`。

在不考虑 training-infer mismatch 的情况下，actor 会在 `_train_step` 里按常规 GRPO/GSPO/PPO 的方式构造 loss。具体来说，每一个 training step 会基于当前的 policy model 计算当前 training data batch 的 log probs，直接记为 `log_probs`。用 `old_log_probs` 和 `log_probs` 构造 importance ratio，叠加 clip、KL norm 和 entropy bonus得到 loss，再做梯度累积和优化器 backward。

而考虑到 mismatch，则 `rollout_log_probs, old_log_probs, log_probs` 都会参与到 loss 的构造中：

- 在 `actor.py` 的 `_train_step` 中，计算 `old_log_probs` 与 `rollout_log_probs` 的绝对差 `train_rollout_logprob_abs_diff`，实时量化训练与推理之间的数值偏差。
- 启用 TIS ([Truncated Importance Sampling](https://fengyao.notion.site/off-policy-rl#245721e3f6c48025aaeadec35aa6da9f))。计算 importance weight，也即 `tis = torch.exp(old_log_probs - rollout_log_probs)`，并对其进行截断（Clipping），用这个权重对 Policy Gradient Loss (`pg_loss`) 进行重加权。这种方法确保模型即使在并非完美的 on-policy 环境下，依然能减缓模型训练崩溃。感谢 [MIS](https://www.notion.so/271211a558b7808d8b12d403fd15edda?pvs=21) 和 [TIS](https://fengyao.notion.site/off-policy-rl#245721e3f6c48025aaeadec35aa6da9f) 的作者团队。

以 GRPO 为例, 最后的 loss 函数便为:

$$
\begin{aligned}
\mathcal{L}(\theta) &= \frac{1}{L} \sum_{t=1}^L \left[ \bar{w}_t \cdot \mathcal{L}^{\text{clip}}_t(\theta) - \beta \text{KL}_t + \lambda H_t \right] \\
\text{where } \mathcal{L}^{\text{clip}}_t &= \min \left( r_t(\theta) A_t, \ \text{clip}(r_t(\theta), 1\pm\epsilon) A_t \right) \\
r_t(\theta) &= \frac{\pi_{\theta}}{\pi_{\text{old}}}, \quad \bar{w}_t = \text{min}\left( \frac{\pi_{\text{old}}}{\pi_{\text{rollout}}}, C \right)
\end{aligned}
$$

### 权重更新优化：权重更新与共卡 colocated 模式

训练结束后，最新的权重会被同步回到 Inference Engine（这是 refit 一词的最佳定义）。在 `update_weight_utis.py` 中，我们完整支持所有模式：`colocated` 和 `distributed` ，前者 train / rollout 交替占用同一批 GPU，后者将 train / rollout 分散在不同 GPU 上。对于这两种方式，我们都采用了分桶异步更新的策略[Reference](https://hebiao064.github.io/rl-weight-sync)，逐个将 chunked 权重同步到 inference engine，尽量减小 peak memory usage。

<p align="center">
  <img src="./pic/4_fsdp_refit.png" alt="Update weights from training to inference with async tensor handle and bucket" width="50%" />
</p>

> ✅ 关于权重更新的具体机制，欢迎查阅 SGLang RL group 以往的博客：[**RL System Deep Thinking: Weight Update Mechanisms**](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/sys-design/readme-1-EN.md)


### 显存优化：卸载策略

在FSDP训练流程中，我们在以下几个场景中会通过 offload weight 来节省内存：

- Train offload：在 colocated 场景下，训练完成后调用 `sleep` 将模型 weight 与 optimizer offload 到 CPU，避免在 rollout 阶段占用内存。
- Ref model: 在使用 KL penalty 时，reference model 只会在 `compute_log_prob` 时被 load 到 GPU，计算完成后立刻 offload 回 CPU，避免 GPU 占用。
- Optimizer offload：在训练阶段，将 model parameter 在不参与计算时都 offload 到 CPU，并且 gradient 也 offload 到 CPU；这显著节省了训练的显存消耗，不过 optimizer step 会在 CPU 上进行，训练时间会明显上升。

## FSDP/Megatron 训练精度对齐

经过我们详细的验证，FSDP 和 Megatron 在训练精度上实现了对齐，详见此 [PR](https://github.com/THUDM/slime/pull/788)。实验采用单机 H100，sglang 0.5.5post1，实验脚本见 [script](https://github.com/THUDM/slime/blob/main/scripts/run-qwen3-4B-fsdp.sh)。图注三条曲线分别对应 Megatron, FSDP colocated w ref model, FSDP colocated w/o ref model。结果符合预期，并且收敛效果相近。

<p align="center">
  <img src="./pic/5_fsdp_mcore_match.png" alt="Raw reward match" width="50%" />
</p>

### Context Parallelism

我们额外考虑 context parallelism 场景，我们想保证 Megatron 和 FSDP 在同样的 Context Parallelism 程度下，能够支持的 response length 相近：

> ✅ 理论上 `max_reponse_length_with_cp = max_reponse_length_without_cp * cp_size` [ref link](https://arxiv.org/pdf/2310.01889)

我们在同样的实验配置: 4 张 B200，global_batch_size = 64 下进行了验证，结果如下：

|  | response_length = 8k | response_length = 16k |
| --- | --- | --- |
| FSDP, cp = 1 | work | **OOM** |
| FSDP, cp = 2 | work | work |
| Megatron(TP = 1), cp = 1 | work | **OOM** |
| Megatron(TP = 1), cp = 2 | work | work |

实验结果符合预期，并且收敛效果相近。

## 快速上手 FSDP Backend

### FSDP 一键启动

```bash
# 如果需要使用 WANDB，需要提前设置好环境变量 WANDB_API_KEY
# 下载模型权重 (Qwen3-4B)
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B

# 下载训练数据集 (dapo-math-17k)
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k

# 下载评估数据集 (aime-2024)
hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /root/aime-2024
  
# clone 代码并安装依赖
git clone https://github.com/THUDM/slime.git
cd slime
pip install -e .


# FSDP不用进行权重转换，native 支持 huggingface 格式
# 开启 reference model，在 colocated 模式下训练 Qwen3-4B
bash /root/slime/scripts/run-qwen3-4B-fsdp.sh
```

> ⚠️ FSDP 通过 `AutoModelForCausalLM.from_pretrained()` 自动读取所有架构信息，无需手动指定。Megatron 需要手动配置参数读取 model 架构信息，或者通过 `--use-hf-config-for-megatron` 实现自动推断， FSDP可以全部从 `config.json` 自动读取，可以直接避免权重格式转换步骤。

### Megatron 与 FSDP 参数对比表

| 配置类别 | Megatron 参数 | FSDP 参数 | 说明 |
| --- | --- | --- | --- |
| **模型加载** | `--load` (Megatron checkpoint) + 架构参数 (`--num-layers`, `--hidden-size` 等) 或 `--use-hf-config-for-megatron` | `--hf-checkpoint` (必需) | **FSDP**: 直接使用 HuggingFace 格式，无需转换权重，通过 `AutoConfig` 自动推断架构 |
| **张量并行** | `--tensor-model-parallel-size` | Coming Soon |  |
| **流水线并行** | `--pipeline-model-parallel-size` | Coming Soon |  |
| **专家并行** | `--expert-model-parallel-size` | Coming Soon |  |
| **上下文并行** | `--context-parallel-size` | `--context-parallel-size` | 两者都支持 CP |
| **初始学习率** | `--lr` | `--lr` | 参数相同 |
| **学习率衰减** | `--lr-decay-style` (linear/cosine) | `--lr-decay-style` (仅 constant) |  |
| **Warmup** | `--lr-warmup-iters` (步数) | Coming Soon |  |
| **最小学习率** | `--min-lr` | Coming Soon |  |
| **优化器类型** | `--optimizer` (adam/sgd 等) | `--optimizer` (默认 adam) | 基本相同 |
| **分布式优化器** | `--use-distributed-optimizer` | 内置于 FSDP | FSDP 默认使用分布式优化器 |
| **梯度检查点** | `--recompute-granularity`, `--recompute-method` | `--gradient-checkpointing` | **FSDP**: 简化为布尔开关 |
| **CPU Offload** | 通过分布式优化器实现 | `--fsdp-cpu-offload` | **FSDP**: 将参数/梯度/优化器状态卸载到 CPU |
| **Attention 后端** | 由 Megatron Core 决定 | `--attn-implementation` (flash_attention_2/sdpa/eager) | **FSDP**: 直接透传给 HuggingFace |
| **混合精度** | `--fp16` 或 `--bf16` | `--fp16` (bf16 自动推断) | 基本相同 |
| **保存时 Offload** | - | `--fsdp-state-dict-cpu-offload` (默认 True) | **FSDP**: 保存 checkpoint 时 offload 到 CPU |
| **训练后端** | 默认或 `--train-backend megatron` | `--train-backend fsdp` (必需) | 用于切换后端 |

### FSDP 目前不支持/不完善的功能

FSDP 目前仅支持 **DP + CP**，不支持 **TP, EP , PP** 。FSDP 的 CP 实现方式与 Megatron Core 不同，Megatron Core 有原生实现（与 TP/PP 深度集成）, FSDP则是通过 Ring Flash Attention 外部库实现。除此之外，Megatron 的 `--recompute-granularity` (full/selective)、`--recompute-method` (uniform/block)、`--recompute-num-layers` 并不支持，FSDP 只有简单的 `--gradient-checkpointing` 开关。最后，FSDP 优化器的学习率目前只支持设置为constant，并且没有 warmup 策略。

## 未来计划

- 维持代码的干净与整洁的同时实现 TP 和 EP
- 支持 FSDP2 上 vision + language 联合训练/部分冻结
- 支持 Qwen3-next/gpt-oss 等混合模型的训练以及优化

## 致谢

Z.ai: Zilin Zhu, Chengxing Xie, Haoran Wang, Lei Li

SGlang RL team: Huapeng Zhou, Chengxi Li, Yusheng Su, Zhuohao Li, Ji Li, Jiahui Wang, Jin Pan, William Ren, Tom, Qisheng Liu, Yuzhen Zhou, Jiajun Li, Yuqi Xiang, Mao Cheng, Chenyang Zhao

Linkedin: Lancert

## 附录

### Context Parallel

[PR link](https://github.com/THUDM/slime/pull/467)

FSDP 的 CP 直接通过 [ring flash attention](https://github.com/zhuzilin/ring-flash-attention) 库实现。相比于 Megatron 复杂的 chunk 机制，FSDP只需要实现简单的连续chunk，负载均衡部分交给 ring flash attn 实现。我们可以只关注输入数据的切分与结果的聚合。

**具体实现流程如下：**

1. **Device Mesh Setup：**在 `setup_device_mesh` 中建立（DP, CP）二维通信组，并使用 `substitute_hf_flash_attn` 将 HuggingFace 模型原本的 Flash Attention 算子替换为支持 CP 的 Ring Flash Attention 实现。
2. **Input Slicing：**在 forward 之前的 `_get_model_inputs_args` 阶段，我们将 Data Packing 后的 `input_ids` 和 `position_ids` 直接在序列维度上使用 `torch.chunk`切分为 `cp_size` 份，当前 rank 仅加载属于自己的那一份数据。同时，调用 `update_ring_flash_attn_params` 将全局的 `cu_seqlens` 信息传递给底层 Attention 算子。
3. **Result Gathering**: 在计算 Log Probs 时 (`get_logprob_and_entropy_with_cp`)，每个 rank 并行计算本地分片的 log_probs 和 entropy。最后通过 `all_gather` 将分布在不同 rank 上的结果拼接回完整序列，并移除为了满足 CP 对齐要求而填充的 Padding。

### 数据打包

[PR link](https://github.com/THUDM/slime/pull/321)

为了避免直接 padding 造成每个 CP rank 上都存在大量的 padding 造成浪费，我们将长序列拼接成连续向量，并用 `cu_seqlens` 记录边界。我们首先复用了megatron的 `process_rollout_data()` 按 DP rank 拆分rollout，随后 `packed_data` 根据 rollout token 数量，DP size 来估算需要多少个 `micro_batch` 来完成一个 `global_batch`。slime中 `global_batch` 和 `micro_batch` 的关系见 Batch & Sample

- 在开启 `use_dynamic_batch_size` 的情况下，需要根据实际的序列长度动态计算 micro-batch 数量：通过 `get_minimum_num_micro_batch_size()` 使用 First-Fit 算法，根据每条序列的长度和 `max_tokens_per_gpu` 限制，估算最少需要多少个 micro-batch 才能容纳所有数据。该数量会在所有 DP  rank 间进行 `all_reduce(MAX)` 同步，确保各 rank 的梯度累积步数一致。
- 若未开启动态 batch size，则直接使用静态公式 global_batch_size // (micro_batch_size * dp_size) 计算固定的 micro-batch 数量。

接下来在 `pack_sequences()` 中执行实际的打包操作：

- 计算分区数 `k_partitions = ceil(total_tokens / max_tokens_per_gpu)`
- 调用 `get_seqlen_balanced_partitions()` 使用 [Karmarkar-Karp](https://en.wikipedia.org/wiki/Largest_differencing_method) 算法（最大差分法）进行负载均衡分配，该算法通过优先队列维护分区状态，每次合并 token 总数差距最大的两个分区，使最终各 pack 的 token 数高度均衡
- 对每个pack，将分配的序列拼接成连续的 `flat_tokens` 向量，同时构建 `cu_seqlens` 数组记录各序列边界，如 `[0, 128, 384, 512]` 表示3条序列长度分别为128、256、128

在Context Parallel模式下（`cp_size > 1`），`pad_packed_sequence_with_cp()` 会对拼接后的序列做最小对齐padding（最多 cp_size-1 个 token），确保总长度能被 cp_size 整除以便跨 rank 切分。虽然这里还是朴素的直接 padding，但是由于padding ≤ cp_size -1，不会导致可见的 overhead。

训练时，`cu_seqlens` 直接传递给Flash Attention处理变长序列；计算 loss 时，`unpack_sequences()` 根据边界信息精确还原每条序列的 log_probs、advantages 等指标。这种方法基本避免了朴素 padding 造成的overhead。

### PPO KL 精度误差

在 PPO 训练流程中涉及三个批次相关的参数, Batch, Micro batch size & Sample 

在理想情况下，当 `sample` 数量 × `micro_batch_size`  = `global_batch_size` 时，意味着一次 rollout 生成的所有样本（sample数量 × 每批次处理的prompt数）恰好等于一个完整的训练批次。此时 rollout 阶段和训练阶段使用的是**同一个未更新的 actor 权重版本**

- Rollout 时用权重 `W_t` 生成 responses
- 训练时仍用权重 `W_t` 计算 log probabilities

因此理论上 PPO KL 散度应该为 0。然而实际运行中（仅在开启 reference model 时），从第一个 micro batch 开始 KL 散度就维持在微小正值而非 0，说明存在数值漂移问题。

该问题由权重交换逻辑中的精度误差引起。原实现参考 Megatron 的方式，通过手动在 CPU 和 GPU 之间交换 ref 和 actor 的 tensors。为兼容 FSDP2 的 DTensor，我们手动创建 DTensor 进行 swap。然而，手动权重交换会导致权重加载过程中产生细微的数值偏差。Megatron 采用这种手动的交换是因为 distributed optimizer 的offload过程很复杂，索性直接交换权重。

最终我们改用了更简洁的方案：将 reference model 作为独立的 FSDP 模型，使用FSDP原生的 CPU Offload，进行管理，仅在 forward 时被加载到GPU中。这种方式完全避免了手动权重交换，充分利用 FSDP 原生的 CPU/GPU 转移机制，从根源上消除了数值漂移，使 PPO KL 收敛到理论值 0，同时不引入额外的 GPU 内存开销。[PR link](https://github.com/THUDM/slime/pull/780)

### **True on policy**

在 CP 的PR 合进去之后 main branch 的 true on policy 居然失效了 [issue link](https://github.com/THUDM/slime/issues/830), 经过排查后发现是精度在缩进之后被 autocast 成了 bf16, 修复之后 training-infer mismatch 成功恢复到0。[PR link](https://github.com/THUDM/slime/pull/833)

为了避免auto cast应用不当导致的精度问题，我们最终选择了FSDP2新支持的 [Mixed Precision](https://docs.pytorch.org/tutorials/intermediate/FSDP_advanced_tutorial.html#mixed-precision)，实现了更加清晰干净的精度管理。

### Batch & Sample

- **Sample 数量（n-samples-per-prompt）**：每个 prompt 生成的候选回复数量
- **Micro batch size**：训练时每次前向/反向传播处理的样本数量（受 GPU 显存限制）
- **Global batch size**：一个完整训练迭代的总样本数，通常由多个 micro batch 累积梯度完成
