# Data Parallelism attention

本文将详细介绍 DP Attention 的原理与实现。如果您已了解 SGLang 的张量并行（TP）、数据并行（DP）机制及其执行链路，那么理解本文内容将会相对容易。若不熟悉相关背景，建议先行阅读《Sglang 源码学习笔记（三）- 分布式和并行（以deepseek 为例）（WIP） - 进击的Bruce的文章 - 知乎》以建立基础。https://zhuanlan.zhihu.com/p/1890082781461207006

关于 SGLang DP Attention 的官方介绍，以及优化效果可参考 ：https://lmsys.org/blog/2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models

官方的描述是：

> The most common parallelism strategy for inference is tensor parallelism. However, it might not be the most efficient strategy for certain models. For example, DeepSeek models use MLA and only have one KV head. If we use tensor parallelism on 8 GPUs, it will lead to duplicated KV cache and unwanted memory usage.
> 
> 
> To overcome this, we've implemented data parallelism (DP) for the multi-head latent attention (MLA) mechanism to improve throughput for DeepSeek models. By adopting DP for the attention component, the KV cache is significantly reduced, allowing for larger batch sizes. In our DP attention implementation, each DP worker handles different types of batches (prefill, decode, idle) independently. The attention-processed data will be all-gathered among all workers before entering the Mixture-of-Experts (MoE) layer, and after processing through the MoE, the data will be redistributed back to each worker. The figure below illustrates this idea.

优化效果：With data parallelism attention enabled, we have achieved up to 1.9x decoding throughput improvement compared to the previous version.

注意：这一优化提升了在大批量场景下服务器受限于 KV 缓存容量时的峰值吞吐量，但不建议用于低延迟、小批量的场景。Ref: https://github.com/sgl-project/sglang/blob/main/docs/references/deepseek.md

**请注意：为便于理解 DP Attention 的核心思想，本文将主要解析其初版实现代码（见 PR #1970），该版本要求 DP_SIZE 与 TP_SIZE 相等。：https://github.com/sgl-project/sglang/pull/1970**

本文撰写于 2025 年 5 月。彼时，SGLang 的 DP Attention 已发展为支持 `1 < DP_SIZE ≤ TP_SIZE` 以及 `MOE-DENSE-TP-SIZE=[1, None]` 等更灵活的配置，以适应更多应用场景。由于这些新特性增加了实现的复杂性，本文将不作深入探讨，重点关注初版设计以便于理解。

# 为什么需要 DP Attention

SGLang v0.4 的发布说明中提到：

> The most common parallelism strategy for inference is tensor parallelism. However, it might not be the most efficient strategy for certain models. For example, DeepSeek models use MLA and only have one KV head. If we use tensor parallelism on 8 GPUs, it will lead to duplicated KV cache and unwanted memory usage.
> 

如发布说明所述，DeepSeek模型中的多头潜在注意力（MLA）机制的 num_kv_heads (KV头数量) 为 1 （减少 KV Cache 来压缩显存占用，从而优化推理速度）。

在 SGLang （以及其他推理引擎）的 QKVParallelLinear（用于注意力机制QKV变换的线性层）实现中，其并行策略如下方代码片段所示：

```cpp
if tp_size >= self.total_num_kv_heads:
    self.num_kv_heads = 1
    self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
else:
    self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
    self.num_kv_head_replicas = 1
```

当张量并行（TP）进程数 (tp_size) 大于或等于总 KV 头数 (total_num_kv_heads) 时，KV 头将被复制。具体而言，原始的每个 KV 头组会被复制 tp_size / total_num_kv_heads 次，每个 TP_Rank 负责处理其中一个复制的 KV 头。

当张量并行（TP）进程数 (tp_size) 少于总 KV 头数 (total_num_kv_heads) 时，KV 头将被分割。此时，每个原始的 KV 头组仅存在一份，由不同的 TP_Rank 分别处理 total_num_kv_heads / tp_size 个 KV 头。

鉴于 MLA 的 num_kv_heads 为1，若采用上述张量并行切分策略，当 tp_size 大于1时，KV Cache 将被复制 tp_size 次。这种复制显著增加了不必要的显存占用，因此传统的张量并行切分方式并不适用于 num_kv_heads 较小的模型（如MLA）。

因此，SGLang 提出了采用数据并行（DP）方式来优化 MLA 处理效率的方案。

值得注意的是，类似的问题也存在于其他模型中。例如，Qwen/Qwen3-235B-A22B 的 num_key_value_heads 为4，在进行大规模部署时，若 tp_size 较大，同样可能导致 KV Cache 的冗余。因此，SGLang 也为 Qwen3-MOE 模型引入了 DP Attention 支持，详情可参见 PR #6121：https://github.com/sgl-project/sglang/pull/6121

注意：这一优化提升了在大批量场景下服务器受限于 KV 缓存容量时的峰值吞吐量，但不建议用于低延迟、小批量的场景。Ref: https://github.com/sgl-project/sglang/blob/main/docs/references/deepseek.md



# 如何实现 DP Attention？

> To overcome this, we've implemented data parallelism (DP) for the multi-head latent attention (MLA) mechanism to improve throughput for DeepSeek models. By adopting DP for the attention component, the KV cache is significantly reduced, allowing for larger batch sizes. In our DP attention implementation, each DP worker handles different types of batches (prefill, decode, idle) independently. The attention-processed data will be all-gathered among all workers before entering the Mixture-of-Experts (MoE) layer, and after processing through the MoE, the data will be redistributed back to each worker. The figure below illustrates this idea.
> 

如前述 SGLang 的博客所解释，DP Attention 的核心思想主要包括以下几点：

1. 通过对注意力组件采用数据并行，KV 缓存得以显著减少
2. 每个 DP 工作单元独立地处理不同类型的批次（例如预填充、解码、空闲）
3. 在进入混合专家 (MoE) 层之前，会在所有工作单元间进行全局汇聚 (all-gathered)
4. 而在通过 MoE 层处理之后，这些数据会再次被分发回各个工作单元

结合下图，我们可以更直观地理解这一机制的实现方式：

![image](https://github.com/user-attachments/assets/83d2b68c-8436-4c56-8828-ee336f98241f)


在数据并行（DP）的基础上，其核心设计如下：

1. 对于模型中除 MLP 层以外的部分（如 Embedding, Self-Attention），每个数据并行单元（DP_Rank）内部的张量并行规模（TP_Size）设置为1，即每个 DP_Rank 独立计算这些部分。
2. 对于 MLP 层，所有 DP_Rank 则共同组成一个大的张量并行组（TP_Group），该组的大小等于数据并行的规模（DP_Size）。

在执行 Embedding 和 Self-Attention 计算时，每个 `DP_Rank` 独立处理其负责的数据分片。

计算流程进行到 MLP 层时，所有 `DP_Rank` 会将其各自批次（batch）的 `hidden_states` 通过 `all_gather` 操作进行汇聚。汇聚后的完整 `hidden_states` 张量，再由所有 `DP_Rank` 构成的 `TP_Group` 以张量并行的方式共同送入 MLP 层进行计算。

MLP 计算完成后，其输出结果会根据原始各 `DP_Rank` 的数据边界进行切片（slice），并将相应部分分发回各个 `DP_Rank`，这一过程与 SGLang 博客中图示的 slice 阶段相对应。

![image](https://github.com/user-attachments/assets/1f94d8f7-30a9-4fd6-9869-1f30c7a3f066)


对应代码：

```cpp
hidden_states, start_idx, end_idx = all_gather(
    hidden_states, forward_batch, self.tp_rank, self.tp_size, self.tp_group
)
hidden_states = self.mlp(hidden_states)
hidden_states = hidden_states[start_idx:end_idx]
```



# SGLang 新版本的 DP Attention

SGLang 后续版本的 DP Attention 进一步增强，已支持 `1 < dp-size <= tp-size` 的灵活配置。

此外，针对 MoE 模型中的 Dense FFNs，SGLang 也支持 `moe_dense_tp_size=[1, None]` 的配置选项。特别地，当该参数设置为 `1` 时（即对这些 Dense FFN 层采用数据并行），可以有效避免高张量并行度下常见的计算单元碎片化问题，同时优化内存使用效率、减少通信开销，进而提升整体系统的扩展性和性能。关于此配置的更详细说明，请参阅：https://lmsys.org/blog/2025-05-05-large-scale-ep/#dense-ffns



# 代码讲解：

https://github.com/sgl-project/sglang/pull/1970

`python/sglang/srt/managers/data_parallel_controller.py`

```cpp
class DataParallelController:
    """A controller that dispatches requests to multiple data parallel workers."""

    def __init__(self, server_args, port_args) -> None:
        # Parse args
        self.server_args = server_args
        self.port_args = port_args
        self.load_balance_method = LoadBalanceMethod.from_str(
            server_args.load_balance_method
        )

        # Init inter-process communication
        self.context = zmq.Context(1 + server_args.dp_size)
        self.recv_from_tokenizer = get_zmq_socket(
            self.context, zmq.PULL, port_args.scheduler_input_ipc_name
        )

        # Dispatch method
        self.round_robin_counter = 0
        dispatch_lookup = {
            LoadBalanceMethod.ROUND_ROBIN: self.round_robin_scheduler,
            LoadBalanceMethod.SHORTEST_QUEUE: self.shortest_queue_scheduler,
        }
        self.dispatching = dispatch_lookup[self.load_balance_method]

        # Start data parallel workers
        base_gpu_id = 0
        self.workers = []
        scheduler_pipe_readers = []
        for dp_rank in range(server_args.dp_size):
            tmp_port_args = PortArgs.init_new(server_args)
            tmp_port_args.tokenizer_ipc_name = port_args.tokenizer_ipc_name
            tmp_port_args.detokenizer_ipc_name = port_args.detokenizer_ipc_name

            if server_args.enable_dp_attention:
                # Share workers for DP and TP
                send_to, reader = self.launch_tensor_parallel_process(
                    server_args,
                    tmp_port_args,
                    base_gpu_id,
                    dp_rank,
                )
                base_gpu_id += 1
                scheduler_pipe_readers.append(reader)
            else:
                send_to = self.launch_tensor_parallel_group(
                    server_args,
                    tmp_port_args,
                    base_gpu_id,
                    dp_rank,
                )
                base_gpu_id += server_args.tp_size
            self.workers.append(send_to)
```

当 `server_args.enable_dp_attention` 为 `True` 时，控制器会调用 `launch_tensor_parallel_process` 方法来启动每个数据并行工作单元。

```cpp
def launch_tensor_parallel_process(
    self,
    server_args: ServerArgs,
    port_args: PortArgs,
    base_gpu_id: int,
    dp_rank: int,
):
    reader, writer = mp.Pipe(duplex=False)
    gpu_id = base_gpu_id
    tp_rank = dp_rank
    proc = mp.Process(
        target=run_scheduler_process,
        args=(server_args, port_args, gpu_id, tp_rank, dp_rank, writer),
    )
    proc.start()
    send_to = get_zmq_socket(
        self.context, zmq.PUSH, port_args.scheduler_input_ipc_name
    )

    return send_to, reader
```

在此函数中，可以看到它为每个 `dp_rank` 启动了一个独立的调度器进程（`run_scheduler_process`）。关键在于，传递给该进程的 `tp_rank` 参数被设置为当前的 `dp_rank`，而由于这里是为每个 DP worker 单独启动进程（而不是一个 TP group 内的多个 rank），可以理解为此时该调度器进程视角下的 `tp_size` 默认为1（或者说，它自己构成一个大小为1的TP组，直到MLP层前）。

`python/sglang/srt/managers/schedule_batch.py`

`def prepare_for_idle`

```cpp
def prepare_for_idle(self):
    self.forward_mode = ForwardMode.IDLE
    self.input_ids = torch.empty(0, dtype=torch.int32).to(
        self.device, non_blocking=True
    )
    self.seq_lens = torch.empty(0, dtype=torch.int32).to(
        self.device, non_blocking=True
    )
    self.extend_num_tokens = 0
```

该函数用于准备一个处于 `IDLE`（空闲）模式的批次。这种情况发生在某个 `dp_rank` 并未分配到实际的请求（requests）时。然而，由于所有 `dp_rank` 都需要参与后续 MLP 层的 `all_gather` 和计算，因此即使是空闲的 `dp_rank` 也需要构造一个空批次并执行相应的前向传播流程（尽管其输入 token 为空）。

`python/sglang/srt/managers/scheduler.py` 

`def prepare_dp_attn_batch`

```cpp
def prepare_dp_attn_batch(self, local_batch: ScheduleBatch):
    # Check if other DP workers have running batches
    if local_batch is None:
        num_tokens = 0
    elif local_batch.forward_mode.is_decode():
        num_tokens = local_batch.batch_size()
    else:
        num_tokens = local_batch.extend_num_tokens

    local_num_tokens = torch.tensor(
        num_tokens, dtype=torch.int64, device=self.device
    )
    global_num_tokens = torch.empty(
        self.tp_size, dtype=torch.int64, device=self.device
    )
    torch.distributed.all_gather_into_tensor(
        global_num_tokens,
        local_num_tokens,
        group=self.tp_worker.get_tp_device_group(),
    )

    if local_batch is None and global_num_tokens.max().item() > 0:
        local_batch = self.get_idle_batch()

    if local_batch is not None:
        local_batch.global_num_tokens = global_num_tokens.tolist()

    return local_batch
```

当启用 DP Attention 时，此函数 (`prepare_dp_attn_batch`) 负责为 `ScheduleBatch` 对象准备额外所需的数据。

`num_tokens`: 当前 `dp_rank` 在此批次中实际需要处理的 token 数量。

`local_num_tokens`: 将 `num_tokens` 转换为张量形式，表示当前 `dp_rank` 本地需要处理的 token 数量。

`global_num_tokens`: 一个张量，用于存储通过 `all_gather` 从所有 `dp_rank`收集到的各自需要处理的 token 数量。其形状为 `torch.Size([self.tp_size])`（在此初版代码中，`dp_size` 等于 `tp_size`，所以这里的 `self.tp_size` 实际上指的是 DP 的 world_size）。

通过 `torch.distributed.all_gather_into_tensor` 操作，每个 `dp_rank` 的 `local_num_tokens` 会被收集到 `global_num_tokens` 张量中。此操作完成后，每个 `dp_rank` 都会拥有包含所有其他 `dp_rank` 的 token 数量信息的 `global_num_tokens` 张量。

`python/sglang/srt/model_executor/forward_batch_info.py`

`def init_new`

```cpp
if ret.global_num_tokens is not None:
    max_len = max(ret.global_num_tokens)
    ret.gathered_buffer = torch.zeros(
        (max_len * model_runner.tp_size, model_runner.model_config.hidden_size),
        dtype=model_runner.dtype,
        device=device,
    )
```

此处为 `ForwardBatchInfo` 对象初始化一个名为 `gathered_buffer` 的缓冲区。该缓冲区专用于 MLP 层计算之前的 `all_gather` 操作，用于暂存从所有 `dp_rank` 汇聚而来的 `hidden_states`。其大小预设为 `max(ret.global_num_tokens) * model_runner.tp_size`，确保能够容纳所有 `dp_rank` 中 token 数量最多的那个 `dp_rank` 所贡献的数据，并乘以 `tp_size` (即 `dp_size`) 来覆盖所有 `dp_rank` 的数据。

`python/sglang/srt/models/deepseek_v2.py` 

`def __init__`

```cpp
if use_dp:
    # For data parallel attention
    if self.q_lora_rank is not None:
        self.q_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.q_lora_rank,
            bias=False,
            quant_config=quant_config,
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ReplicatedLinear(
            q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
        )
    else:
        self.q_proj = ReplicatedLinear(
            self.hidden_size,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
        )
    self.kv_b_proj = ReplicatedLinear(
        self.kv_lora_rank,
        self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        bias=False,
        quant_config=quant_config,
    )
    # O projection.
    self.o_proj = ReplicatedLinear(
        self.num_heads * self.v_head_dim,
        self.hidden_size,
        bias=False,
        quant_config=quant_config,
    )
```

当启用 DP Attention (`use_dp` 为 `True`) 时，注意力（Attention）模块内的线性层（如Q投影、KV投影、O投影等）不再采用传统的张量并行切分方式。取而代之的是，每个 `dp_rank` 会拥有这些层（如 `q_proj`、`kv_b_proj`、`o_proj`）的完整副本（通过 `ReplicatedLinear` 实现），并独立完成这些线性变换的计算。

`def all_gather`

```cpp
def all_gather(
    input_tensor: torch.Tensor, forward_batch: ForwardBatch, rank, world_size, group
):
    if world_size == 1:
        return input_tensor

    all_lens = forward_batch.global_num_tokens
    max_len = max(forward_batch.global_num_tokens)

    padded_tensor = torch.nn.functional.pad(
        input_tensor, (0, 0, 0, max_len - input_tensor.shape[0])
    )

    torch.distributed.all_gather_into_tensor(
        forward_batch.gathered_buffer, padded_tensor, group=group
    )

    gathered_tensors = torch.concat(
        [
            forward_batch.gathered_buffer[i * max_len : i * max_len + all_lens[i]]
            for i in range(world_size)
        ]
    )

    start_index = 0 if rank == 0 else sum(all_lens[:rank])
    end_index = start_index + all_lens[rank]

    return gathered_tensors, start_index, end_index
```

此辅助函数 `all_gather` 封装了在进入 MLP 层之前对 `hidden_states` 进行全局汇聚的核心逻辑。它接收当前 `dp_rank` 的 `input_tensor`（即部分 `hidden_states`），并利用 `forward_batch.global_num_tokens`（记录了所有 `dp_rank` 的 token 数量）和预分配的 `forward_batch.gathered_buffer` 来完成操作。

最终，该函数返回汇聚了所有 `dp_rank` 数据的 `gathered_tensors`，以及当前 `rank` 在这个汇聚后张量中所对应数据的起始 (`start_index`) 和结束 (`end_index`) 索引，以便后续 MLP 计算完成后进行切片分发。
