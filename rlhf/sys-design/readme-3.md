## MEGATRON 概念

> Megatron 就像威震天一样力量惊人但又难以驾驭。

megatron最初的核心理念就是实现 TP DP PP 的 3D 并行，以支持在数千 GPU 上训练超大规模模型。

Megatron-LM 的系列论文

- [Tensor parallelism](https://arxiv.org/pdf/1909.08053)
- [3D 并行](https://arxiv.org/pdf/2104.04473)
- [activation recomputation](https://arxiv.org/pdf/2205.05198)

1. Tensor Parallelism（TP）

将单层 Transformer 内部的线性层参数在维度上切分（如 MLP 权重、注意力头），分布到多个 GPU 上执行，从而避免单卡参数爆炸。适用于单层参数过大的场景。

Megatron 默认通过 `--tensor-model-parallel-size` 启用 TP，执行过程中配合 all-gather、reduce-scatter 进行跨 GPU 通信。

2. Pipeline Parallelism（PP）

将模型的不同层切分为多个 stage，每个 GPU 或 GPU group 负责一部分层，借助流水线调度执行多个 micro-batch，提升计算利用率。

使用 `--pipeline-model-parallel-size` 配置，同时可启用 interleaved virtual pipeline（虚拟流水线）进一步提升 overlap 能力。

3. Data Parallelism（DP）

在多个 worker 上复制模型副本、分发不同样本，通过 gradient all-reduce 同步梯度。Megatron 的 DP 也支持 ZeRO-like 的 distributed optimizer。

使用 `--use-distributed-optimizer`、`--overlap-grad-reduce` 等控制是否采用 ZeRO。

## Megatron-Core

在 Megatron-LM 的基础上，NVIDIA 推出了 Megatron-Core，将核心模块抽象为可复用的 PyTorch 原语与训练组件，支持更复杂的模型结构与用例：

1. Context Parallelism（CP）

在 token 序列维度上进一步切分模型，适用于长上下文训练任务（如 Mamba、Llama 3 128k token）。

2. MoE，EP

支持 Token-Level MoE 路由、专家负载均衡、GroupedGEMM 和 Token Drop 等优化机制，在 8x7B 模型上训练达到 47% MFU。

3. Checkpoint 格式转换

支持 legacy/core/llama_mixtral 等多种模型格式互转，并通过 torch_dist 格式实现跨 TP/PP 维度的 checkpoint 加载与转换。

4. 模块化 API 设计

Megatron-Core 提供基于模块化构建的 GPTModel、TransformerConfig 等组件，一定程度上优化了原本的与模型耦合痛点

下面的表格对比了Megatron Core vs Deepspped的特性支持

| 功能项                                     | MCore MoE     | DeepSpeed       |
|------------------------------------------|----------------|------------------|
| **架构 (Arch)**                           |                |                  |
| Token dropless MoE (dMoE)               | 支持           | 不支持           |
| Token drop MoE                          | 即将支持       | 部分支持（Top-1/2） |
|                                          |                |                  |
| **MoE 路由器 (MoE Router)**              |                |                  |
| Top-K                                   | 支持           | 部分支持（Top-1/2） |
|                                          |                |                  |
| **MoE 负载均衡 (MoE Load balancing)**     |                |                  |
| Z-loss                                  | 支持           | 不支持           |
| Load balancing loss                     | 支持           | 支持             |
| Sinkhorn                                | 支持           | 不支持           |
|                                          |                |                  |
| **并行方式 (Parallelism)**               |                |                  |
| EP（Expert Parallel）                   | 支持           | 支持             |
| TP & SP（张量并行 + 序列并行）           | 支持           | 部分支持（仅 TP） |
| DP（数据并行）                           | 支持           | 支持             |
| PP（流水线并行）                         | 支持           | 不支持           |
| CP（上下文并行）                         | 即将支持       | 不支持           |
| 复杂混合并行支持（如 TP+EP+DP+PP）        | 支持           | 不支持           |
| 分布式 MoE 优化器                        | 支持           | 不支持           |
|                                          |                |                  |
| **训练工具 (Training Utils)**            |                |                  |
| ZeRO-3                                   | 支持           | 支持             |
| 通用 Checkpoint 转换工具（支持 HF 格式） | 支持           | 不支持           |
| MoE 分布式 checkpoint                    | 支持           | 不支持           |
|                                          |                |                  |
| **内核融合 (Kernel Fusion)**             |                |                  |
| GroupedGEMM                              | 支持           | 不支持           |
| Token (un)permutation                    | 支持           | 不支持           |
| Sinkhorn                                 | 支持           | 不支持           |
|                                          |                |                  |
| **训练精度 (Training Dtype)**            |                |                  |
| BF16                                     | 支持           | 支持             |
| FP16                                     | 支持           | 支持             |

## Megatron 性能

Megatron-Core 支持对百亿至千亿参数规模的大语言模型进行高效训练，采用模型并行与数据并行的多重组合策略。在实验中，其训练对象为参数量从 2B 到 462B 不等的 GPT 模型。

实验在最多 6144 张 H100 GPU 上进行，开启以下优化选项以增强通信与计算的重叠：

- `--overlap-grad-reduce` 和 `--overlap-param-gather`：实现数据并行通信与计算的重叠；
- `--tp-comm-overlap`：启用张量并行通信的计算重叠；
- 流水线并行通信重叠为默认开启。

实际上目前在 100B 以下的 dense 模型，FSDP 仍旧能打；而 100B 以上模型以及 MOE，Megatron 是独一份的。

### 弱扩展（Weak-Scaling）实验结果（H100 集群）

### Megatron-Core 弱扩展（Weak-Scaling）实验结果（H100 集群）

弱扩展（weak scaling）指当计算资源（如 GPU 数量）增加的同时，问题规模也按比例增大。

| 模型参数规模 | 张量并行 TP | 流水线并行 PP | 数据并行 DP | GPU 数 | 全局 Batch | 单卡 TFLOP/s | MFU* |
|-------------:|:-----------:|:-------------:|:-----------:|:------:|:----------:|:------------:|:----:|
| **2.1 B** | 1 | 1 | 16 / 64 | 16 / 64 | 256 | 441 / 412 | 45 % / 42 % |
| **8.3 B** | 4 | 1 | 16 / 64 | 64 / 256 | 256 | 457 / 426 | 46 % / 43 % |
| **78 B** | 8 | 2 | 32 / 96 | 512 / 1536 | 960 | 446 / 418 | 45 % / 42 % |
| **314 B** | 8 | 8 | 16 / 48 | 1024 / 3072 | 1152 | 490 / 464 | 50 % / 47 % |
| **509 B**   | 8 | 20| 8 / 24 | 1280 / 3840| 1440| 473 / 426 | 48 % / 43 % |

> ** MFU**：Model FLOPs Utilization
> 

端到端训练吞吐:

![image](https://hackmd.io/_uploads/Sk2l8ywHge.png)

在弱扩展实验中，Megatron-Core 展现出超线性扩展性。模型利用率（MFU）从最小模型的 41% 提升至最大模型的 47%-48%。其主要原因在于：更大的矩阵乘法操作（GEMM）具有更高的算术强度，执行效率更高，从而提升整体硬件利用率。
![weak_scaling](https://developer.download.nvidia.com/images/megatron/megatron-weak-scaling-800w.svg)




<!-- 先贴几个链接 https://zhuanlan.zhihu.com/p/366906920 -->

<!-- ![Pasted image 20250704165412](https://hackmd.io/_uploads/BkZaggUrle.png) -->
### 并行原理


#### TP
[todo]
在张量并行 (TP) 中，每个 GPU 仅处理张量的一部分，并且仅当某些算子需要完整的张量时才触发聚合操作。

**这里的具体内容我原本想写一下 但是研究TP公式和实现细节过于复杂 想要详细弄懂读者请阅读下面这篇文章 这里我们对每种parallel方法仅作概述
https://zhuanlan.zhihu.com/p/622212228**

文章中把张量模型并行的计算架构说完了。张量将矩阵乘法转换为分块矩阵乘法，减少了单卡的内存使用，内容存效率很高。
在一个TP group里 第一个进程读数据broadcast到其他进程中，为了保证数据一致性，只有第一个进程会从磁盘加载模型参数，其余进程通过广播操作获取相同的数据，从而确保组内参数一致性。

由于张量并行在执行期间涉及频繁的数据通信，若并行进程分布在不同节点之间，将会带来较高的通信开销。因此，为了提升计算效率，实际部署时应尽可能将同一个 TP group 的进程安排在单机内，以利用 NVLink 降低通信延迟。


在张量并行中，由于参数被分块计算，每个进程仅生成输出的一部分状态。当这些中间结果需要作为后续层的输入或输出时，通常需要进行 gather或 reduce 操作以合并结果。


![image](https://hackmd.io/_uploads/B16YjQLBel.png)





<details>
<summary>
MLP的TP实现（列张亮并行）
</summary>
class ColumnParallelLinear(torch.nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        *,
        ...

    ):
        super(ColumnParallelLinear, self).__init__()
        ...
        # 获取张量组大小
        world_size = get_tensor_model_parallel_world_size()
        # 当前进程在，张量组中的位置
        rank = get_tensor_model_parallel_rank()
        # 从列维度进行分片大小
        self.output_size_per_partition = divide(output_size, world_size)

        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # 构建参数， 这里nn.liner 矩阵乘法 XA^T + b， 所以对参数进行转置 
        self.weight = Parameter(
                    torch.empty(
                        self.output_size_per_partition, # 输出特征分片，减少单卡参数
                        self.input_size,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )

        ... 

        
    def forward(
        self,
        input_: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
    ):
        ...
        # 反向过程中，梯度需要进行求和all_reduce, 函数定义重写 torch.autograd.Function 类 backward函数
        input_parallel = copy_to_tensor_model_parallel_region(input_)
        ...
        # 重写前向和反向函数， 
        # 详细实现，在megatron 梯度计算章节分析
        output_parallel = self._forward_impl(
            input=input_parallel,
            weight=weight,
            bias=bias,
            ...
            ),
        )

        ...
        # 张量并行输前向出的部分结果，输出参数判断是否需要进行拼接gather操作， 
        # 若后续有行张量并行全连接层， 则不需要gather，减少一次通信 
        if gather_output:
            assert not self.sequence_parallel
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        ... 
        return output, output_bias
</details>

<details>
<summary>
在attention的张量分片操作中， 是对注意力头进行分片，不需要拼接结果就可以进行后续计算，这部分在转换megatron模型到HF格式需要注意，容易出错。
</summary>
class ParallelAttention(MegatronModule):
...

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, inference_params=None,
                rotary_pos_emb=None):

        ...

        if self.attention_type == AttnType.self_attn:
          
            # qkv 全连接层输出 [sq, b, h]
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
            # [句长， 批大小， 维度] --> [句长, 批大小, 注意力头[分片],  头维度]
            # num_query_groups_per_partition 把头按张量并行进行分片， num_attention_heads/word_size

            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                 # 注意头大小已经按张量大小进行分片
                self.num_query_groups_per_partition,
                (
                    (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                    * self.hidden_size_per_attention_head
                ),
            )
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            # 将QKV进行分割, 按最后一个维度进行分割
            (query_layer, key_layer, value_layer) = torch.split(
                mixed_x_layer,
                [
                    (
                        self.num_attention_heads_per_partition // self.num_query_groups_per_partition
                        * self.hidden_size_per_attention_head
                    ),
                    self.hidden_size_per_attention_head,
                    self.hidden_size_per_attention_head
                ],
                dim=3)

            # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn] 
            query_layer = query_layer.view(query_layer.size(0),
                                           query_layer.size(1), -1, self.hidden_size_per_attention_head)

</details>


#### PP

流水线并行就比较通俗易懂。一个模型的各层会在多个GPU上做切分。一个批次（batch）被分割成较小的microbatches，并在这些微批上进行流水线式执行。


管道并行的数据通信量相对较少，只有管道组内相邻的进程间进行通信，故相邻的两个进程之间通信量较少，可将PP group，安排到不同节点，优先安排TP在同一个节点上。
流水线并行是序列化的计算过程，通信类型是P2P通信，单词通信数据量较少但是比较频繁，而且因为流水线的特点，会产生流水线bubble



##### 普通流水线并行

在普通流水线中，模型被拆分成多个阶段（stage），每个 GPU 负责其中的一部分层。数据则被拆分成若干个微批次，每个微批依次经过不同的阶段。比如：

![image](https://hackmd.io/_uploads/rJrlRrvSle.png)

前向传播和反向传播是串行执行的，即前向传播完成后才开始反向传播。这样会导致 GPU 之间的资源利用率不高，因为在前向传播阶段，反向传播阶段的 GPU 会处于空闲状态。而且由于前向传播和反向传播是串行执行的，会导致反向传播的需要的中间变量无法释放，从而导致显存占用过高，甚至 OOM。

##### 1F1B


为了解决 FThenB 的问题，引入了 1F1B 调度策略。1F1B 的全称是 1 Forward 1 Backward，即一边前向传播，一边反向传播


![image](https://hackmd.io/_uploads/HkS_CrvSgg.png)
在 1F1B 中，前向传播和反向传播交替进行，因此在计算一个微批次的反向传播时，前向传播结果可以立刻释放。这种交替执行的模式减少了显存占用。


##### Interleaved Pipelining
普通流水线并行虽然在一定程度上提升了并行计算的效率，但当模型参数继续增大时，依然存在问题：某些 GPU 可能在等待其他阶段完成时会闲置。为了进一步优化这一点，Megatron-LM 提出了交错流水线并行，它在普通流水线的基础上引入了更细粒度的模型分块

将模型进一步划分为多个小块（num_model_chunks），每个小块都有自己的微批
微批和小块交替执行，让每个 GPU 在同一时刻可以处理不同层的计算
![image](https://hackmd.io/_uploads/rksJkIvSee.png)



#### SP

序列并行是 Megatron-LM 中一项重要的优化技术，旨在进一步减少训练长序列模型时的显存占用，特别是激活值的显存。它与张量并行（TP）协同工作，通常在张量并行组内部署。

在 Transformer 模型中，有许多操作在序列维度上是独立的。例如：
Layer Normalization：对每个 token 的 embedding 独立进行归一化。
Dropout：独立地对每个元素应用 dropout。

对于这些操作，我们不需要在每个 GPU 上都拥有完整的序列数据。序列并行利用了这一点，将输入张量的序列维度 (sequence length) 切分到张量并行组内的各个 GPU 上。

##### 工作方式

序列并行是在一个已经定义好的张量并行组（TP group）内进行的。假设 TP group 的大小为 tp_size。一个典型的输入张量到 Transformer 层的形状可能是 (sequence_length, batch_size, hidden_size)。在进行序列并行时，这个张量的 sequence_length 维度会被切分成 tp_size 份。 每个 TP rank（TP组内的 GPU）只处理序列的一部分，即 (sequence_length / tp_size, batch_size, hidden_size)。

在 Transformer 块中，那些在序列维度上独立的操作（如上面提到的 LayerNorm, Dropout, element-wise operations）就可以直接在切分后的序列数据上并行执行。例如，对于 LayerNorm，每个 GPU 只对自己持有的 sequence_length / tp_size 这么长的序列片段进行 LayerNorm 操作。


然而，Transformer 中的核心组件attention是序列依赖的。计算 Query, Key, Value 矩阵乘法时，每个 Query 需要与序列中的所有 Key 进行交互。
因此，在进入自注意力计算之前，必须将各个 GPU 上切分的序列片段收集起来 (gather)，恢复成完整的序列。
##### 通信

All-Gather: 在需要完整序列信息的操作之前，TP 组内的 GPU 会执行一个 All-Gather 操作。每个 GPU 将自己持有的序列片段广播给组内的其他 GPU，同时接收其他 GPU 的片段。
Reduce-Scatter / Slice: 在自注意力计算完成之后，如果后续的操作可以进行序列并行，那么注意力机制的输出需要再次被切分。
在前向传播中，这通常是一个简单的切片 (slice) 操作，每个 GPU 取出属于自己的那部分序列。 在反向传播中，对应的操作通常是 Reduce-Scatter。梯度的计算会先在完整序列上进行，然后通过 Reduce-Scatter 将梯度分发并累加（如果需要）到各个 GPU 对应的序列片段上。


#### CP


megatron中的context并行与sequence并行不同点在于，SP只针对Layernorm和Dropout输出的activation在sequence维度上进行切分，CP则是对所有的input输入和所有的输出activation在sequence维度上进行切分，可以看成是增强版的SP。开启CP就会覆盖SP的效果。除了Attention模块以外，其他的模块(Layernorm、Dropout)由于没有多token的处理，在CP并行时都不用任何修改（指和SP相同）。


Attention计算过程中每个token的Q要跟同一个sequence中其他token的K和V一起进行计算，所以通过CP并行后，在计算Attention前要通过allgather通信拿到所有token的KV向量，在反向计算时对应需要通过reduce_scatter分发gradient梯度。
为了减少显存占用，在前向时每个gpu只用保存一部分KV块，反向时通过allgather通信拿到所有的KV数据。

LLM经常由于sequence长度过长导致显存OOM，通过CP可以更好解决OOM的问题，每个GPU只用处理一部分的sequence, 同时减少CP倍的通信和计算，但保持TP不变，同时activation也会减少CP倍。CP优化的性能参考如下图，在Megatron中通过指定--context-parallel-size可以进行使用。world_size=CP * PP * DP * TP

![image](https://hackmd.io/_uploads/Hk-R5SDSxg.png)



#### EP


在并行化方面，Megatron-Core 还支持专家并行。对于超大规模MoE 模型，它能够灵活地将专家并行与其他并行策略有机结合。

在 token 分发机制上，Megatron-Core MoE 采用了 dropless MoE 操作，即不丢弃任何 token。在路由和负载均衡优化层面，它支持多种路由策略，如通用的top-k，并在负载均衡算法上支持z-loss 以及 load balancing loss 等多种方案。
此外，为解决多个专家接收变长输入问题，Megatron-Core MoE 引入了 GroupedGEMM 技术，并优化效率较低的操作，将其替换为优化的CUDA kernel。
同时，在模型迁移与适配上，Megatron-Core MoE 提供了丰富的模型 checkpoint 转换功能，允许用户导入HuggingFace 模型，并自由调整 TP（tensor parallelism）、PP（pipeline parallelism）和 EP（expert parallelism）等结构，随后利用 Megatron-Core 高效启动模型训练任务。



#### 组合

结合多种并行技术会导致复杂的相互作用，应该如何组合并行技术，以便在保留严格的优化器语义的同时，在给定的batch size下最大限度地提高大型模型的训练吞吐量？

Megatron-LM 提出了PTD-P，利用跨多GPU服务器的流水线并行、多GPU服务器内的张量并行和数据并行的组合，在同一服务器和跨服务器的GPU之间具有高带宽链接的优化集群环境中训练具有万亿参数的模型，并且容易扩展

![image](https://hackmd.io/_uploads/HJqFPNUBgg.png)





##### 通信

- TP：每层前向使用 all-gather 拼接输出，反向使用 reduce-scatter 合并梯度；通信频繁，推荐同机组网（NVLink）。
- PP：前后阶段通过 Send/Recv 传递激活和梯度；通信量小但频繁，通常跨节点部署（Infiniband）。
- DP：每层反向后通过 all-reduce 同步梯度；适合横向扩展增加吞吐。







#### Megatron 源码阅读


[megatron code walk through](https://space.keter.top/docs/high_performance/Megatron%E6%BA%90%E7%A0%81%E9%98%85%E8%AF%BB/pretrain_process)

## Megatron  RL框架支持

- verl

早期版本 Verl 基于 Megatron-LM 0.4，并通过一些变通方式适配 HuggingFace 的模型类。为了使用新版 Megatron 的最新特性并提升训练速度，verl正迁移到 Megatron-Core，并在所有语言模型中采用官方推荐的 GPTModel 类。借助 mcore 的 GPTModel，即可启用 context parallel 等最新功能。

- Areal

Areal原生支持megatron backend 
默认使用及以上的Megatron-LM v0.11.0


- Slime

Slime使用megatron作为主要的training backend 可参考
https://github.com/THUDM/slime/blob/main/README_zh.md

<!-- ## Megatron core 中的通信优化 -->

<!-- #### DP -->

<!-- 
数据并行，DeepSpeed中的ZeRO系列可以在数据并行的维度上对模型、梯度、和优化器参数进行切分。其中，ZeRO-1将原本数据并行中的all-reduce梯度操作切分成reduce-scatter梯度+all-gather参数，这样做的好处是优化器更新可以在切分后的参数量上进行，从而减少了内存开销。

Megatron-Core支持ZeRO-1形式的数据并行，即在DDP中实现reduce-scatter反向传递得到的梯度，在distributed optimizer中实现all-gather优化器更新后的模型参数。一般来说，降低数据并行的通信开销有两个常用的手段。首先，我们可以通过梯度累加，比如说流水线并行中的micro-batching，来降低数据并行中通信的比例，这一点对ZeRO-1依旧适用。

此外，我们可以将通信和计算进行隐藏。和传统的DDP相比，ZeRO-1允许将reduce-scatter和反向传递进行隐藏，将all-gather和前向传递进行隐藏。同时，为了提高通信效率，我们需要将小参数进行合并（即tensor fusion[7]）。我之前有篇论文做的就是这方面的工作，其中通信优化部分的实现方式和现在Megatron-Core里的实现基本一样。当然，类似的技巧对于ZeRO-3来说依旧适用，例如PyTorch中的FSDP也实现了类似的通信隐藏[8]。



其次是张量并行，前面提到，Megatron-LM对于all-gather+线性层的反向传递进行了通信优化。为了节省内存开销，Megatron-LM只存了all-gather之前的输入，所以在反向传递阶段，我们需要all-gather保存的输入用来计算权重的梯度，另外我们还需要对于计算得到的输入的梯度进行reduce-scatter。于是，我们可以将all-gather和计算输入的梯度进行隐藏，然后将reduce-scatter和计算权重的梯度进行隐藏。这种通信和计算无依赖关系的隐藏，又叫做bulk overlap。


除了以上例子，Megatron-LM并没有对其他操作进行通信优化，包括前向传递中的all-gather+矩阵乘，和矩阵乘+reduce-scatter，因为这两个操作中的计算和通信存在依赖关系，无法直接进行隐藏。针对这种情况，我们可以使用tensor partitioning的技术，将一个大的矩阵乘法和集合通信操作，拆分成一系列小的矩阵乘法和集合通信操作，然后对更加细粒度的计算和通信进行流水线式的隐藏。当然，将一个tensor切分得太小，反而会影响实际性能，一般来说切成4份是比较常见的配置。

除了直接切分张量以外，我们还可以将集合通信操作拆分成一系列的p2p通信操作，例如all-gather操作可以拆分成ring-based send/recv通信[9]，其中拆分后的通信和计算同样可以进行隐藏。

具体实现上，Megatron-Core调用了Transformer Engine中的线性层，支持bulk overlap通信隐藏，以及张量切分或者p2p切分方式的通信隐藏。同时，为了降低通信和计算之间存在的干扰，TE使用userbuffer进行张量并行的进程间通信。

最后是流水线并行，流水线并行中需要用到大量的send/recv操作，实现起来非常繁琐。为此，Megatron-LM设计了一系列的p2p通信接口，用来打包send-next, recv-prev, send-prev, recv-next操作，防止p2p通信因为执行顺序不同导致的死锁问题。

Megatron-Core支持1F1B和interleaved 1F1B这两种流水线并行方案，并针对interleaved 1F1B进行了通信隐藏优化。一方面，因为interleaved 1F1B在大模型训练中更为常用，同时其通信开销要远远大于普通的1F1B方案。另一方面，对于1F1B而言，哪怕使用异步的send/recv操作，其实也没有太多的通信优化空间[10]。而对于interleaved 1F1B来说，在steady阶段，我们可以将forward-send-forward-recv通信和反向传递的计算隐藏，然后将backward-send-backward-recv通信和前向传递的计算隐藏。
 -->

## RL选型与模型迁移

### 选型

选择 Megatron-Core 的场景
	•	模型数量少+长期维护 只服务 1-2 个闭源 LLM，可投入时间写 LayerSpec 和权重映射，后期不需要改动很多
	•	追求极限速度 & 显存 需要 3-D 并行（TP × PP × ZeRO-DP）、Flash-Attn v2、FP8 等最高硬件利用率。
	•	GPU 配置较高 
	•	有专业 AI Infra engineer （打工人落泪）

---

选择其它方案（AutoTP / FSDP）的场景
	•	模型多、版本更新快 频繁换模型或魔改结构，重写 Megatron 适配成本太高。
	•	规模较小 可以用 PyTorch FSDP直接包模型，无须改源码。
	•	跨节点带宽有限 Lacking NVLink，TP 效益降低


### 从Megatron plus verl看一眼模型迁移

Megatron 作为框架 模型跟框架无法解耦， 需要手动切割模型并且适配， 兼容性很差。

作为例子 我们来简单看下megatron和verl的兼容部分

*AI
模型管理: registry.py提供统一的模型注册和管理
配置转换: config_converter.py处理不同模型的配置转换
模型创建: model_initializer.py负责模型初始化
前向传播: model_forward.py定义模型推理逻辑
权重处理: weight_converter.py和saver.py处理权重转换和保存
模型加载: loader.py负责权重加载
工具支持: util.py提供序列处理工具

---
verl/models/mcore部分构成了Verl项目中完整的Megatron Core体系， 只适合主流模型，搞了mcore版本没什么大的区别 历史模型又需要重写迁移
class SupportedModel(Enum):  
•    LLAMA = "LlamaForCausalLM"  # tested  
•    QWEN2 = "Qwen2ForCausalLM"  # tested  
•    QWEN2_MOE = "Qwen2MoeForCausalLM"  # pending  
•    DEEPSEEK_V3 = "DeepseekV3ForCausalLM"  # not tested  
•    MIXTRAL = "MixtralForCausalLM"  # tested  
•    QWEN2_5_VL = "Qwen2_5_VLForConditionalGeneration"  # not supported  
•    LLAMA4 = "Llama4ForConditionalGeneration"  # not tested  
•    QWEN3 = "Qwen3ForCausalLM"  # tested  
•    QWEN3_MOE = "Qwen3MoeForCausalLM"  # not tested


#### 支持megatron的方式
1. 使用 mcore 的 GPTModel 构建 HuggingFace 模型（建模阶段）
	a. 将 HuggingFace 的 config 转换为 mcore 的 TransformerConfig
	例如把 LLaMA/Qwen 的配置（层数、维度、注意力头数等）转成 mcore 认可的字段格式。
	 b. 用这个 TransformerConfig 初始化 mcore 的 GPTModel
	c. 将 HuggingFace 的权重加载进 GPTModel 中
	要做权重格式转换和维度 reshape，使得 HuggingFace 的 tensor 能正确装载进 mcore 模型。
2. 将 mcore 模型权重转换成 HuggingFace 格式用来rollout
	a. 解决 mcore 和 HuggingFace 权重结构/命名上的差异
    比如 transformer.layers.0.attn.query.weight 和 model.layers.0.q_proj.weight 的映射关系。
	b. 在线做权重resharding 给 rollout 引擎使用
    这是一个复杂过程，mcore 用了 tensor parallel / pipeline parallel / expert parallel，rollout 引擎可能没有这些并行结构, 所以要动态处理切片方式、通信策略、加载顺序等。
3. 支持 mcore 的并行与加速特性
mcore 最大的优势是其高性能并行训练支持，这些特性需要在 verl 中接好：
	a. 支持各类并行策略：
	b. 支持 recompute、KV缓存等加速技巧
    比如激活重计算、KV cache fusion、RMSNorm fusion 等，在训练或推理中提升效率。
4. Checkpoint 支持
	a. 使用 mcore 的 dist_checkpointing 格式，确保 crash 后可以恢复训练。
	b. 支持将 mcore checkpoint 导出为 HuggingFace 格式
    用于后续推理部署，比如在 SGlang 中加载。
---
### Relevant code

![Untitled diagram _ Mermaid Chart-2025-07-05-220333](https://hackmd.io/_uploads/r1gUZ7DBle.png)






在verl/models/mcore/registry.py下注册了
MODEL_CONFIG_CONVERTER_REGISTRY 
MODEL_INITIALIZER_REGISTRY  
MODEL_FORWARD_REGISTRY 
MODEL_WEIGHT_CONVERTER_REGISTRY 

构建一个megatron training model的流程从megatron_worker.py中的
class ActorRolloutRefWorker(MegatronWorker, DistProfilerExtension)开始

#### `ActorRolloutRefWorker.build_model_optimizer`
（这里要吐槽的一个点是 我在读到这种函数名的时候总会觉得是build model's optimizer， 然而实际上是build model and optimizer 和`ActorRefRolloutworker`同样的槽点。后面部分基本都是在描写build_model_optimizer中构造世纪模型的流程

#### 1.加载config
首先`build_model_optimizer`会调用`verl/single_controller/base/megatron/worker.py`中的`MegatronWorker._init_hf_config_and_tf_config`：
先读取预训练模型中的 `hf config` 再调用`registry.hf_to_mcore_config` 会根据注册表中的相应转换函数将hf的model cofig转换成megatron所需要的config (megatron.core.transformer.TransformerConfig，即tfconfig)格式。 在这一部分获得了hfconfig与转化后的tfconfig 

#### 2.加载模型


<details>
<summary>
TLDR
</summary>
get_model 是 分布式训练中构建模型主流程的调度器，它根据当前进程的并行并行位置，调用 model_provider_func 多次去构造每个子模块。
而 init_mcore_model 就是传入的的 model_provider_func，它负责根据 pre_process 和 post_process 位置、模型类型、MoE/VLM 等特性，实际返回当前 rank 应该持有的子模型结构。
</details>

获得了config之后接下来会调用get_model，get_model是 Megatron/Megatron-Core 的模型构建核心函数
```
verl/utils/megatron_utils.py
def get_model(
    model_provider_func,               # 用户提供的模型构造函数
    model_type=ModelType.encoder_or_decoder,
    wrap_with_ddp=True,           
    use_distributed_optimizer=True,
    transformer_config=None,
) 
 关键输入参数
•	model_provider_func: 一个函数，接受 pre_process / post_process 等参数，返回一个 nn.Module。你提供模型的创建方式
•	model_type: 是 decoder-only、encoder-only，还是 encoder-decoder？会影响 PP 拆分方式
•	wrap_with_ddp: 是否用 DDP 包裹模型
•	transformer_config: Megatron 配置对象，用于判断是否启用 CPU init 等
```

这里面的model_provider_func传入的就是init_mcore_model的一层包装
我们来看一眼`registry.init_mcore_model` 
```
def init_mcore_model(
    tfconfig: TransformerConfig,
    hf_config: PretrainedConfig,
    pre_process: bool = True,
    post_process: bool = None,
    *,
    share_embeddings_and_output_weights: bool = False,
    value: bool = False,
    **extra_kwargs,  # may be used for vlm and moe
) -> GPTModel:
```
`registry.init_mcore_model`是一个模型工厂函数，它根据传入的 transformer 配置（tf_config）、HuggingFace 配置（hf_config）、以及一些控制参数，实例化出一个 mcore（Megatron Core）风格的 transformer 神经网络模型 (`megatron.core.models.gpt.gpt_model.GPTModel`)
内部逻辑：
会根据模型类型（例如 qwen2, qwen2-moe, mixtral, deepseek-v3 等选择`verl/models/mcore/model_initializer.py`中的initializer进行初始化

DenseModel 为例，其 initialize() 方法流程如下：
1.	获取 Transformer LayerSpec
	调用 get_transformer_layer_spec()，内部通过 get_gpt_decoder_block_spec(tfconfig) （mcore自动的Layerspec工具函数） 生成每层的结构定义。
	•	对于 MoE 模型，LayerSpec 会被额外 patch，例如为每层 MLP 的 expert 配置路由门控参数。
2.	提取 Rope 缩放参数
	•	如果 HuggingFace 配置中有 rope_scaling（如 Qwen 中的 linear scaling），将其转换为 Megatron-Core 识别的 seq_len_interpolation_factor。
3.	实例化 
	•	调用 GPTModel(...) 或 Qwen2_5VLModel(...) 构造具体模型，传入上一步生成的 LayerSpec、配置、embedding 和 output 层开关等参数。
	•	对于 Vision-Language 模型，还会额外构建 vision encoder、vision projection 模块
4.	可选设置输出层
	•	若设置 value=True 且 post_process=True，则将输出层替换为一个 Linear 头（
5.	对于 MoE 模型（如 Qwen2 MoE、Mixtral），会进行额外设置



```
class BaseModelInitializer(ABC):
    """Base class for model initializers."""

    def __init__(self, tfconfig: TransformerConfig, hf_config: PretrainedConfig):
        self.tfconfig = tfconfig
        self.hf_config = hf_config

    @abstractmethod
    def get_transformer_layer_spec(self):
        """Get the transformer layer specification.
        https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/models/gpt/gpt_layer_specs.py"""
        pass

    def get_rope_scaling_args(self) -> dict:
        """Get rope scaling args."""
        rope_scaling_args = {}
        if "rope_scaling" in self.hf_config:
            if self.hf_config.rope_scaling is not None:
                # assert self.hf_config.rope_scaling["type"] == "linear", "only linear scaling is supported for now"
                rope_scaling_args["seq_len_interpolation_factor"] = self.hf_config.rope_scaling["factor"]
        return rope_scaling_args

    def initialize(
        self,
        pre_process: bool = True,
        post_process: bool = True,
        share_embeddings_and_output_weights: bool = False,
        value: bool = False,
        **extra_kwargs,
    ) -> GPTModel:
        """Initialize a GPT model with the given configuration.
        https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/models/gpt/gpt_model.py

        Args:
            pre_process (bool): include embedding layer.
            post_process (bool): including an output layer.
            share_embeddings_and_output_weights (bool): input embeddings and output logit weights are shared.
            value (bool): add an extra linear layer for classification or regression.

        Returns:
            GPTModel: An initialized GPT model instance
        """
        transformer_layer_spec = self.get_transformer_layer_spec()
        rope_scaling_args = self.get_rope_scaling_args()
        mtp_block_spec = extra_kwargs.get("mtp_block_spec", None)
        model = GPTModel(
            config=self.tfconfig,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=self.hf_config.vocab_size,
            max_sequence_length=self.hf_config.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type="rope",
            rotary_base=self.hf_config.rope_theta,
            **rope_scaling_args,
            mtp_block_spec=mtp_block_spec,
        )

        if post_process and value:
            from verl.models.llama.megatron.layers.parallel_linear import LinearForLastLayer

            model.output_layer = LinearForLastLayer(input_size=self.tfconfig.hidden_size, output_size=1, config=self.tfconfig)

        return model

class Qwen2MoEModel(BaseModelInitializer):
    """Initializer for Qwen2 MoE models."""

    def get_transformer_layer_spec(self):
        assert self.tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for now"
        transformer_layer_spec = get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True)

        # Patch layer spec for shared experts
        for i in range(len(transformer_layer_spec.layer_specs)):
            transformer_layer_spec.layer_specs[i].submodules.mlp.submodules.shared_experts.params["gate"] = True

        return transformer_layer_spec

    def initialize(self, **kwargs):
        # Qwen default freeze_moe_router: true
        model = super().initialize(**kwargs)
        freeze_moe_router = kwargs.get("freeze_moe_router", True)
        if freeze_moe_router:
            for layer in model.decoder.layers:
                layer.mlp.router.weight.requires_grad = False
        return model
```
#### 3.加载权重
模型初始化后调用 [load_mcore_dist_weights](https://github.com/volcengine/verl/blob/281ecd4cc167afe676dcbaf1612009b5b81555c1/verl/utils/model.py#L536) 使用megatron.core.dist_checkpointing 加载 Megatron 的分布式权重。
若使用 HuggingFace 格式的权重，还可通过 load_megatron_gptmodel_weights 加载，它会进一步调用 `_load_hf_model(...)` 加载 HF 权重 

```
model = AutoModelForCausalLM.from_pretrained(local_model_path, torch_dtype="auto")
state_dict = model.state_dict()
```

Megatron 无自动 key mapping，因此必须在 [load_state_dict_to_megatron_gptmodel](https://github.com/volcengine/verl/blob/281ecd4cc167afe676dcbaf1612009b5b81555c1/verl/models/mcore/loader.py#L56)() 内部手动 根据 key 名称硬编码匹配 并 广播到目标参数上，并加载到megatron

1. 计算全局 rank 和 layer 映射
```
src_rank = _megatron_calc_global_rank(tp_rank=0, dp_rank=0, pp_rank=0, cp_rank=cp_rank)
layer_map = _megatron_calc_layer_map(config)
```

计算广播源 rank（src_rank）— 用于从指定 rank 广播 tensor 给其他 rank。
layer_map 构建全局层号 → (pp_rank, virtual_pp_rank, local_layer_idx) 的映射，用于找到每一层在哪个模型分片中。

2. unwrap模型封装
 
 ```
 from megatron.core import DistributedDataParallel as LocalDDP
from megatron.core.transformer.module import Float16Module
from torch.nn.parallel import DistributedDataParallel as torchDDP

models[i] = unwrap_model(wrapped_model, (torchDDP, LocalDDP, Float16Module))
 ```
 
3. 分布式广播权重张量
 从 src_rank 将张量数据 broadcast() 到其他 rank。
 
 
- 普通张量广播（不切片）
```
def _broadcast_tensor(tensor, name)
```
用于 LayerNorm、FinalNorm、output_layer（value 模型中也用到 lm_head.weight）。
所有 rank 都先广播张量 shape，再构造空 tensor，在 src_rank 中拷贝值并广播。

- Tensor Parallel 切片广播（按 chunk_dim 切）

```
def _broadcast_tp_shard_tensor(tensor, name, chunk_dim=0)
```

主要用于 linear weights，如 self_attn.o_proj.weight, mlp.down_proj.weight
在 src_rank 中对 tensor 进行切片（按 TP rank），每次广播一个切片，目标 TP rank 收到后赋值到对应参数。

- QKV 聚合广播
```
def _broadcast_tp_shard_tensor_qkv(tensor, q_name, k_name, v_name)
```

将 Q/K/V 拼接为 QKV 格式，用于 Megatron 的 fused QKV 权重结构。
区分 num_key_value_heads >= tp_size 与 num_key_value_heads < tp_size 两种情况。


4. 参数结构映射与加载逻辑
遍历每一层 model.layers.{i}，根据 layer_map 映射为 (pp_rank, vpp_rank, local_layer_idx)
如果当前 rank 负责该层，则调用上述广播函数将 state_dict 中的权重加载进模型。如
```
sync_layer.self_attention.linear_qkv.weight <-broadcast_tp_shard_tensor_qkv()
```

对Embedding / FinalNorm / LMHead 进行特殊处理

5. 参数广播到 DP Group

```
def broadcast_params(module):
    for param in module.parameters():
        dist.broadcast(param.data, src=dp_src_rank, group=dp_group)
```


当前流程中明确依赖了 hf 模型的参数命名规则，如：
"model.layers.{i}.self_attn.q_proj.weight" → 对应 Megatron 的 linear_qkv.weight 的一部分
"model.embed_tokens.weight" → 对应 word_embeddings
因此，如果模型结构发生变更需要添加一定的适配逻辑 确保statedict规则和模型正确匹配










---



#### 4.checkpoint


Megatron 对模型结构是高度绑定的，模型构建依赖 GPTModel 类和 gpt_layer_specs，所有模型都必须适配其构造方式 遵循 Megatron 的 BlockSpec / TransformerLayerSpec 模板进行搭建，较难以使用任意结构拼接模型；verl 提供了函数自动帮你完成这个过程，当然前提是你遵守它的结构规范
对于复杂的模型结构，权重加载方式和name mapping都需要一些工作来支持到megatron中

### 那么该如何迁移一个新的模型呢

- 用 mcore 的 GPTModel 来建模 HuggingFace 模型
  - https://github.com/alibaba/Pai-Megatron-Patch/tree/main 项目中有非常多的模型适配代码示例。
  - 主要步骤如下：
    1. 将 HuggingFace 的配置（config）转换为 mcore 的 `TransformerConfig` ，把层数、隐藏维度、注意力头数量等字段映射到 mcore 所用格式。
    2. 使用转换后的 `TransformerConfig` 初始化 mcore 的 `GPTModel`  ，构造出结构化的 `LayerSpec transformer`。
    3. 将 HuggingFace 的权重加载到 `GPTModel` 中，需要做维度转换、命名映射、sharding 等适配。
    4. 对于 VLM，接口可能不同 可以新写一个 class，把 `GPTModel` 作为子模块

- 将 HuggingFace 权重离线/在线转换为 mcore 的 dist_checkpointing 格式 
- 支持 mcore 在线转换权重为 HuggingFace 格式供 rollout 使用






## 使用megatron
```
根据上面对verl使用sglang的walkthrough 使用megatron也大概清楚了
需要准备好 TransformerConfig 和 transformer_layer_spec，这些在 Megatron-LM 的 config 和 spec 文件里有定义。

from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_spec

config = TransformerConfig( # 你需要根据实际情况填写参数
    num_layers=24,
    hidden_size=1024,
    num_attention_heads=16,
    # ... 其他参数
)
layer_spec = get_gpt_layer_spec(config)
model = GPTModel(
    config=config,
    transformer_layer_spec=layer_spec,
    vocab_size=32000,
    max_sequence_length=2048,
)
```
### 参数

---

#### 1. **模型结构参数**
- `--num-layers`：Transformer 层数
- `--hidden-size`：隐藏层维度
- `--ffn-hidden-size`：前馈网络隐藏层维度（通常为 hidden-size 的 4 倍）
- `--num-attention-heads`：注意力头数
- `--seq-length` / `--max-position-embeddings`：最大序列长度
- `--position-embedding-type`：位置编码类型（如 rope、learned_absolute、none）
- `--normalization`：归一化类型（如 RMSNorm、LayerNorm）
- `--swiglu`：是否使用 SwiGLU 激活
- `--group-query-attention`：是否使用 GQA
- `--num-query-groups`：GQA 分组数
- `--disable-bias-linear`：禁用线性层 bias
- `--untie-embeddings-and-output-weights`：输入输出 embedding 是否解耦

---

#### 2. **训练参数**
- `--micro-batch-size`：每卡每步的 batch size
- `--global-batch-size`：全局 batch size（所有卡累加）
- `--train-iters` / `--train-samples`：训练步数/样本数
- `--eval-iters` / `--eval-interval`：评估步数/间隔
- `--save-interval`：模型保存间隔
- `--log-interval`：日志打印间隔

---

#### 3. **优化器与学习率参数**
- `--lr`：初始学习率
- `--min-lr`：最小学习率
- `--lr-decay-style`：学习率衰减方式（如 cosine、linear）
- `--lr-warmup-iters` / `--lr-warmup-fraction`：学习率预热步数/比例
- `--weight-decay`：权重衰减
- `--clip-grad`：梯度裁剪阈值
- `--adam-beta1` / `--adam-beta2`：Adam 优化器参数
- `--bf16` / `--fp16`：是否使用混合精度

---

#### 4. **并行/分布式参数**
- `--tensor-model-parallel-size`：张量并行规模
- `--pipeline-model-parallel-size`：流水线并行规模
- `--expert-model-parallel-size`：MoE 专家并行规模
- `--use-distributed-optimizer`：是否使用分布式优化器
- `--overlap-grad-reduce`：反向传播时重叠梯度归约
- `--overlap-param-gather`：重叠参数聚合

---

#### 5. **数据与分词参数**
- `--data-path`：训练数据路径
- `--tokenizer-type`：分词器类型（如 GPTSentencePieceTokenizer、BertWordPieceLowerCase）
- `--tokenizer-model` / `--vocab-file` / `--merge-file`：分词器模型或词表文件
- `--split`：数据集划分比例（如 969,30,1）

---

#### 6. **MoE/专家模型参数**
- `--num-experts`：专家数
- `--moe-router-load-balancing-type`：MoE 路由负载均衡类型
- `--moe-router-topk`：每 token 路由到的专家数
- `--moe-aux-loss-coeff`：MoE 辅助损失系数

---

#### 7. **日志与保存参数**
- `--tensorboard-dir`：TensorBoard 日志目录
- `--save` / `--load`：模型保存/加载路径

---

#### 8. **特殊模型参数（如 Mamba、Llama3 等）**
- `--mamba-head-dim`、`--mamba-num-groups`、`--mamba-num-heads` 等
- `--language-model-type`：指定模型类型（如 llama3.1_8b）

---

#### 9. **更多参数**
所有参数都可以在 `megatron/training/arguments.py` 文件中找到详细定义和注释。  
你可以直接查找 `add_megatron_arguments` 及其子函数，或者在 example 脚本里查看常用参数组合。


#### 并行配置



| 瓶颈 | 描述 |
|------|------|
| 显存不足 | 模型太大，单卡放不下 |
| 计算过慢 | Transformer 计算量大 |
| 通信拥堵 | 多卡 AllReduce 慢 |
| Batch size 受限 | 无法充分利用吞吐 |
| Layer 不均衡 | 造成 PP 空洞（bubble）|


- **TP**：减少单卡参数量（矩阵切分）
- **PP**：跨层分布计算（层切分）
- **DP**：增加吞吐（多卡同步训练）

---
#####  Megatron 中的 DP 特性
- 使用 `allreduce_params()` 同步参数
- 自定义 DDP 封装以兼容 TP
- `data_parallel_size = world_size // (TP × PP)`

##### 配置时：
- 提升吞吐：增大 DP
- 显存不足：减少 DP，提高 TP/PP
- 多机部署：使用 `--nnodes` 配置 DP 分组

---

##### Megatron TP

- 将线性层等大矩阵分片
- 每个 GPU 仅负责部分列/行
- 前向：all-gather 拼接输出  
  反向：reduce-scatter 合并梯度

配置时：
- 显存紧张：开启 TP（如 TP=2,4）
- 导出 HuggingFace 权重需“合并权重”
- 需使用 Megatron 的特定线性层实现

---

##### Megatron PP

- 可配合 recompute 减少激活显存
- 存在 bubble 空洞 → 增大 micro-batch数量 可缓解
- 层数必须能整除 PP 数（否则报错）

配置时
- 层数大：推荐 PP=4~8
- 提高吞吐：增大 micro-batch
- 多任务模型：每 stage 可绑定不同任务头

---

| 并行机制 | 功能 | 通信 |
|----------|------|------|
| DP       | 多样本并行 | AllReduce |
| TP       | 参数分片并行 | ReduceScatter / AllGather |
| PP       | 层级并行 | Send / Recv |

---

| 参数 | 含义 |
|------|------|
| `--tensor-model-parallel-size` | TP 大小 |
| `--pipeline-model-parallel-size` | PP 大小 |
| `--num-layers` | 总层数（必须能整除 PP） |
| `--micro-batch-size` | 每个 worker 的批大小 |
| `--global-batch-size` | 总 batch = micro × DP × grad_accum |


## REF
[https://developer.nvidia.com/megatron-core#section-get-started](https://developer.nvidia.com/megatron-core#section-get-started)

[https://zhuanlan.zhihu.com/p/694877232](https://zhuanlan.zhihu.com/p/694877232)

https://huggingface.co/blog/bloom-megatron-deepspeed
