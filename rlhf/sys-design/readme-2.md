# RL 系统深思：浅析 FSDP 训练后端

熟悉本系列的读者必然理解，我们绝大多数篇目都在讲述 RL 系统的 rollout engine，对训练后端的讨论并不多。我们会对现在火爆的 DeepSpeed AutoTP，FSDP 和 Megatron-LM 训练后端进行浅析。除此之外，torchtitan 也是被寄予厚望的训练后端，但是我听说作者离职了也不再维护，所以按下不表。本文只讨论 FSDP 的原理和实现。

## DP，DDP 和 FSDP

在提到 FSDP 之前，我们不妨顺着 data parallel 的历史思路，来引出 FSDP 的思想。

推理和训练的 DP 略有不同，训练阶段的 DP 指的是 PyTorch 的 `torch.nn.DataParallel`。在最 naive 的 DP 中，每个 GPU 上都保留有完整的模型参数，主 GPU 负责将数据拆分为多个子集，并将输入数据分发给多个从属 GPU。每个从属 GPU 使用自己的数据子集进行前向和反向传播，并计算梯度。随后，所有从属 GPU 计算出的梯度都会被收集回主 GPU 上。主 GPU 收到所有梯度后，进行平均并更新模型权重，然后再将更新后的完整模型权重广播回每个从属 GPU。这种模式存在明显的性能瓶颈，所有梯度和权重更新都必须经过主 GPU，这会造成主 GPU 的显存和计算资源负担过重，效率低下。

DDP 相比拥有主 GPU 的 DP 而言，做出了一步改进，backward 更新参数会在每个 GPU 上都进行。具体来说，在反向传播过程中，当每个 GPU 都计算完自己的梯度后，触发一次 All-Reduce 操作，所有 GPU 的梯度被平均并返回，每个 GPU 独立地使用相同的平均梯度来更新自己的模型权重。

FSDP 本质上也是 DP，只是原先的单个 GPU 无法塞下完整的模型参数，需要分解到多个 GPU 上，构成一个 DP replica。具体的拆分方式就是 [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)。

ZeRO 设定了三种阶段的训练内存优化模式：

1. Stage 1：shard optimizer states

- 做法：将 optimizer states（如 Adam 的一阶矩 `m` 和二阶矩 `v`）进行分片，每张 GPU 仅保存自己负责的 shard。
- 通信/聚合时机： 每个gpu计算出所有梯度之后，通过reduce-scatter把梯度聚合发送到对应shard的gpu上，每个gpu更新自己shard的对应参数，通过all-gather 更新所有gpu的参数
  - 每个gpu计算出所有梯度之后，通过reduce-scatter把梯度聚合发送到对应shard的gpu上
  - 每个gpu更新自己shard的对应参数
  - 通过all-gather 更新所有gpu的参数
- 优点：节省大量优化器状态内存（通常是参数大小的 2~4 倍），适合中等规模模型，通信量相对可控。
- 限制：参数和梯度仍然是全量复制，每张卡仍需保留它们的完整副本。

 **Stage 2：分片优化器状态 + 梯度**

- 做法：除了优化器状态，每张 GPU 也仅保留部分梯度。
- 通信/聚合时机：
  - 梯度计算结束后，所有 GPU 需要通过 reduce-scatter 将本地的梯度分片合并成全局梯度，每张卡只保留自己负责的 shard。
  - 参数更新时的流程和 Stage 0 相同： 每个gpu各自更新参数 → all-gather更新参数到所有卡。
- 优点：进一步减少显存占用，尤其是在反向传播阶段。
- 限制：通信频率更高，梯度同步变得更复杂，部分框架会配合使用梯度累积（gradient accumulation）来减少通信次数。

**Stage 3：分片优化器状态 + 梯度 + 参数**

- 做法：三者全部分片。每张 GPU 只保留自己负责的一部分参数、梯度和优化器状态。
- 通信/聚合时机：
  - 前向传播：需要用到整个模型参数，因此每个gpu在forward的时候都需要从对应shard的gpu收集参数(all-gather)，再进行前向计算。
  - 反向传播后：梯度通过 reduce-scatter 聚合，每张卡负责自己shard的梯度。同时需要再次收集参数，因为forward收集的参数在使用之后就会被释放。
  - 参数更新：每张卡只更新自己的那部分参数，更新完不需要再广播，因为参数本身就是分布式的。
- 优点：最大限度地降低显存占用，使得训练百亿参数以上的模型成为可能。
- 限制：通信频繁且复杂，all-gather 和 reduce-scatter 成为性能瓶颈，适合搭配重通信优化（如 overlap、压缩、CUDA Graph 等）使用。

![1](https://hackmd.io/_uploads/B1Qp80THlx.png)


随着stage的增加，内存更低，通信更频繁

### 关于通信量

上面这句话并不完全准确，尤其在 ZeRO Stage 1 和 Stage 2 时，通信量并不会增加，反而保持不变。解释如下：

在传统的 Data Parallel（DDP）中，反向传播后需要对每张 GPU 的梯度执行 all-reduce 操作来同步梯度，通信总量为 2Ψ（其中 Ψ 是模型参数总量)。（关于为什么all-reduce是2Ψ，可以看[这里](https://youtu.be/CQ-cOykh0sY)。简单来说all-reduce是通过一次reduce-scatter+一次all-gather实现的）

而在 ZeRO Stage 1 和 Stage 2 中，通信方式虽有变化，但总通信量仍为 2Ψ：

- Stage 1（优化器状态分片【需要check，zero论文里并没有直接分析】}：
  - 反向传播后，每张 GPU 仍持有完整梯度，通过 reduce-scatter 聚合梯度到对应的gpu上（通信量 Ψ）(这里因为每个gpu只需要拿到自己对应参数的梯度 所以我们可以不使用all-reduce)。
  - 接着，每张 GPU 更新自己负责的参数（基于其本地 optimizer state）。
  - 参数更新后，使用 all-gather 将更新的参数 shard 同步回每张卡（通信量 Ψ），以便下一轮前向使用。
- Stage 2（梯度 + 优化器状态分片）：
  - 梯度计算完毕后，直接通过 reduce-scatter 聚合 + 分发梯度（通信量 Ψ），每张卡仅保留其 shard 的梯度。
  - 每张 GPU 使用对应shard 的梯度与 optimizer state 更新对应参数。
  - 然后同样通过 all-gather 收集参数 shard，得到完整参数副本（通信量 Ψ）。

这里顺便分析下stage3的通信开销

- 每次forward到对应参数的时候，负责的shard需要把参数广播到所有gpu，总通信量为Ψ，等同与一个all-gather。
- 因为forward时，临时gather来的参数用完就会被释放以节省显存，所以backward的时候又需要重新获取一遍，总通信量为Ψ，等同与一个all-gather。
- 同时backward的时候需要通过reduce-scatter把梯度聚合到对应的shard上，总通信量为Ψ。

所以stage3总通信量为3Ψ。

微软做了一个视频来演示stage3的training 过程，可以看[这里](https://www.microsoft.com/en-us/research/wp-content/uploads/2020/02/Turing-Animation.mp4?_=1)

## FSDP1 和 FSDP 2
### 一个场景

我们讨论一个场景：有一个包含 3 个 `Linear` 层的 `Layer`，使用 `FSDP` 包装并在 2 个 GPU 之间进行分片。

![Layer](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/accelerate/layer.png)


首先，我们需要了解原始的 `FSDP1` 及其带来的局限性。它将每个 `FSDP` 模块表示为单个 `FlatParameter`，这是一个包含模块所有参数的一维张量，然后在各个rank之间进行分片。也就是说，如果你用 `FSDP1` 包装 `Layer`，会得到如下结果：


![FSDP1](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/accelerate/fsdp1.png)


这种方式带来了一个显著的问题：所有参数被合并为一个张量之后，原始的参数信息（如 dtype、requires_grad 等）无法被逐个记录。由于 FlatParameter 是单个对象，要想保留每个原始参数的元数据，就需要借助一些笨拙的方法。这也是 FSDP1 的一大限制。


与 FSDP1 使用 FlatParameter 的方式不同，FSDP2 使用 DTensor（即 Distributed Tensor，分布式张量）来进行参数管理。 DTensor 是 torch.Tensor 的分布式版本，它不仅支持按指定维度在多个 rank 间进行分片，还携带了关于原始张量的元数据，例如其 dtype、requires_grad、分片方式、[placement 类型](https://pytorch.org/docs/stable/distributed.tensor.html#module-torch.distributed.tensor.placement_types) 等信息。

  
如下图所示，每个 Linear 层的参数都会被单独表示为一个 DTensor，并在第 0 维上在两个 GPU 之间进行分片：


![FSDP2](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/accelerate/fsdp2.png)



### 区别

在 FSDP2 中，每个 Linear 层的参数都会被单独包装为一个 DTensor，这样每个参数的元数据（如数据类型、梯度状态、分片信息等）都能被明确地保留和管理。DTensor 以 dim=0 维度对每个参数单独进行分片，这种 逐参数分片的方式更加直观，也避免了参数展平、拼接、再整体分块的繁琐流程。

相比之下，FSDP1 会将所有参数展平成一个大张量（FlatParameter）并整体切分，这不仅增加了实现复杂度，还引入了额外的开销，例如需要推断每个进程应保存哪些参数片段以及如何还原原始结构。

更重要的是，这种简化并未牺牲性能，FSDP2 在保持与 FSDP1 相当的吞吐量前提下，显著提升了可用性与扩展性。

FSDP2 相比 FSDP1 的核心改进还包括

- 更清晰的内部实现：每个 Parameter 都是一个独立的 DTensor，避免了参数展平、拼接的复杂处理。
- 原生支持部分参数冻结：这使得诸如 [LoRA](https://arxiv.org/abs/2106.09685) 等方法可以**开箱即用**，无需额外修改。
- 支持混合精度参数类型：通过 DTensor，FSDP2 可在同一模型中混合使用 fp8、fp16、bf16 等不同精度的数据类型。
- 更高效、简化的checkpoint机制：
    - 基于 SHARDED_STATE_DICT 和 [torch.distributed.checkpoint](https://pytorch.org/docs/stable/distributed.checkpoint.html)
    - 无需跨 rank 通信，每个 rank 仅保存本地参数分片和对应的元数据
- 支持直接加载分片参数：使用分片的 state_dict 即可完成模型加载，避免额外还原和拼接操作。
- 异步检查点保存：参数会先复制到 CPU，然后主线程继续训练，同时另一个线程将数据写入磁盘，提高训练吞吐。
- 更可控、稳定的内存管理：不再依赖 recordStream，而是通过 CUDA 多流同步实现稳定内存行为。




### FSDP in verl

#### summary

总结来说，verl全面支持FSDP2  FSDP2提供更好的吞吐量和内存使用，并且与其他特性适配。要启用FSDP2，只需要设置：

```
actor_rollout_ref.ref.strategy=fsdp2
actor_rollout_ref.actor.strategy=fsdp2
critic.strategy=fsdp2
reward_model.strategy=fsdp2
```

此外，FSDP2 的 CPU offloading 与梯度累积兼容。可以通过设置 `actor_rollout_ref.actor.fsdp_config.offload_policy=True` 来开启它以节省内存。详见 [https://github.com/volcengine/verl/pull/1026/](https://github.com/volcengine/verl/pull/1026/)

#### code
https://github.com/volcengine/verl/blob/ab11fff33dcaa2409e388ce2f19aff440a5b703f/verl/workers/fsdp_workers.py#L377


##### In ActorRolloutRefWorker
FSDP1： 直接调用torch.distributed.fsdp.FullyShardedDataParallel 包装模型，支持 cpu_offload、auto_wrap_policy、sharding_strategy、mixed_precision、forward_prefetch 等参数，采用FSDPv1的参数体系。
FSDP1：
对于FSDP1 reference policy强制使用CPU offload来节省内存，actor强制关闭 offload因为开启会导致梯度累积结果错误，梯度的累积和同步机制存在时序问题
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

	# We force reference policy to use CPUOffload to save memory.
	# We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
	cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
	fsdp_strategy = self.config.actor.strategy
	if fsdp_strategy == "fsdp":
		actor_module_fsdp = FSDP(
			actor_module,
			cpu_offload=cpu_offload,
			param_init_fn=init_fn,
			use_orig_params=False,
			auto_wrap_policy=auto_wrap_policy,
			device_id=get_device_id(),
			sharding_strategy=sharding_strategy,  # zero3
			mixed_precision=mixed_precision,
			sync_module_states=True,
			device_mesh=self.device_mesh,
			forward_prefetch=self.config.actor.fsdp_config.forward_prefetch,
		)
```

FSDP2 
观察代码可以看到fsdp2是直接in place修改actor_module并返回，而不是fsdp1 返回新的包装对象。此外 fsdp2 需要先保存 state_dict，actor_module在经过apply_fsdp2包装后改变参数分片方式，再通过fsdp2_load_full_state_dict加载权重。

FSDP2下 Actor通过设置`fsdp_config.offload_policy=True`，Actor可以安全开启CPU offload而不影响梯度累积（上文提到的FSDP2 cpu offload与梯度累积兼容）
```python
from verl.utils.fsdp_utils import apply_fsdp2, fsdp2_load_full_state_dict
	mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True)
	if role == "actor" and fsdp_config.offload_policy:
		cpu_offload = CPUOffloadPolicy(pin_memory=True)
		self._is_offload_param = False
		self._is_offload_optimizer = False
	else:
		cpu_offload = None if role == "actor" else CPUOffloadPolicy(pin_memory=True)

	fsdp_kwargs = {
		"mesh": fsdp_mesh,
		"mp_policy": mp_policy,
		"offload_policy": cpu_offload,
		"reshard_after_forward": fsdp_config.reshard_after_forward,
	}
	
	full_state = actor_module.state_dict() 
	apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config) 
	fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)

	actor_module_fsdp = actor_module
```

##### offload

```python
@torch.no_grad()
def offload_fsdp_model_to_cpu(model: FSDP, empty_cache: bool = True):
    # lazy初始化
    _lazy_init(model, model)
    # 通过handles处理flat_param
    for handle in model._all_handles:
        if handle._offload_params:
            continue
        flat_param = handle.flat_param
        handle.flat_param_to(torch.device("cpu"), non_blocking=True)
        flat_param._local_shard = flat_param.data
```

```python
@torch.no_grad()
def offload_fsdp2_model_to_cpu(model, empty_cache: bool = True):
    # 简单处理所有参数
    for param in model.parameters():
        param.data = param.data.to(torch.device("cpu"), non_blocking=True)
```

##### 模型加载

```python
@torch.no_grad()
def load_fsdp_model_to_gpu(model: FSDP):
    _lazy_init(model, model)
    device_id = get_device_id()
    for handle in model._all_handles:
        if handle._offload_params:
            continue
        flat_param = handle.flat_param
        handle.flat_param_to(torch.device(f"{get_device_name()}:{device_id}"), non_blocking=True)
        flat_param._local_shard = flat_param.data
```

```python
@torch.no_grad()
def load_fsdp2_model_to_gpu(model):
    device = get_device_id()
    for param in model.parameters():
        param.data = param.data.to(device, non_blocking=True)
```




## FSDP实现介绍

本节基本参考meta关于fsdp的论文：https://arxiv.org/pdf/2304.11277，这篇论文用13页的内容详细介绍了fsdp的设计选型和哲学，十分推荐想要了解fsdp的朋友直接阅读。

本节默认讨论的都是stage3，fsdp1的情况。fsdp2在本节会略作介绍。

本节并不是一个源码解析，所以不会涉及到很详细的源码解释。

### 核心思想

![2](https://hackmd.io/_uploads/HJDxPCTSxe.png)


首先，fsdp会把参数切分成许多unit。比如，在 MLA 中，一个 attention 层和一个 FFN 层通常组成一个 Transformer block，可做作一个 unit。unit是fsdp管理参数的最小粒度单元。

在前向和反向传播的时候，FSDP只会materialize 一个unit的参数和梯度。除了这个unit之外，其他的参数和梯度都保持shard状态。优化器状态全程保持shard。所以，FSDP的peak memory可如此计算

$$GPU_{i} Peak Memory  = ShardModelSize_{i} + ShardOptimizerSize_{i} + ShardGradientSize_{i} + MAX(fullyMaterializedFSDPUnit)$$



### 模型初始化

在使用 DDP 时，PyTorch 会直接在单个设备上 materialize 整个模型。然而，在 FSDP 中，这种方式不可行。因此，需要解决两个核心挑战：

- 如何在不 materialize 任意 tensor 的情况下构造模型实例，并将实际的初始化延迟到模型被移动到某个具体设备、为 tensor 分配物理内存之后；
- 如何基于用户提供的初始化函数正确地构造模型，即使模型整体无法放入单张 GPU。

为了解决上述问题，FSDP 引入了 deferred initialization 机制。该机制首先在一个 fake device 上初始化所有 tensor。在这一过程中，tensor 并不会分配实际物理内存，我们只会记录 tensor 的构造逻辑及其操作。当 tensor 被移动到真实设备上时，这些操作会被 replay，从而完成 tensor 的真正构建。

理想情况下，每张 GPU 只需初始化其本地的参数 shard。然而，由于无法准确判断用户在模型初始化函数中是否对 unsharded tensor 进行了依赖操作，我们采用与前向和反向传播相同的执行策略：每次仅初始化一个 FSDP unit，完成 shard 操作后再继续下一个 unit 的初始化。

综合来看，FSDP 的初始化流程如下：首先在 fake device 上构造模型实例，并记录所有 tensor 的构造逻辑；随后将模型划分为多个 FSDP unit，每次将一个 unit 移动到实际 GPU 上，对其中的所有 tensor replay 构造逻辑，从而 materialize 出完整的 unit；接着对该 unit 执行shard操作；最后继续初始化下一个 unit，直到整个模型完成初始化和分片。

具体的代码可以参考 https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_init_utils.py#L558

**更多问题**

以上策略在不同fsdp unit之间有dependency的时候都有可能失效，此时无法通过replay正确的构建tensor。FSDP提供了两个代替策略

1. 直接在单GPU上初始化unshard model。这个的intuition是training时候，模型参数其实只占了比较小的一部分。因此，虽然模型可能无法在单卡上训练，但是有可能可以在单卡上初始化。因此，可以先在单卡上初始化模型之后再shard
2. 直接在CPU上初始化unshard model，因为CPU的内存大小往往比gpu大得多。之后，unit by unit的把每个unit转移到单个gpu上，再执行shard。但是因为cpu的并行能力非常差，初始化model可能会变得非常耗时。

### 切分策略

切分策略决定了参数，梯度和优化器状态是否被切分。在此只介绍一些比较重要的sharding strategy。更详细的参数配置请看 **如何使用FSDP** 节

#### **Fully Sharding**

对应zero的stage3，参数、梯度、优化器都被切分。通信开销为ddp的1.5倍

![3](https://hackmd.io/_uploads/BkPGPRpHxl.png)


Note: F为sharing facor，即一个unit会被shard成几份储存

具体实现上，fsdp1会把整个fsdp unit的所有参数flatten成一个巨大的`FlatParameter`。在padding之后再进行shard。


#### Hybrid Sharding

在多节点环境中，节点间通信带宽和延迟通常显著劣于同节点内的 GPU 通信。因此，在参数切分（sharding）时，将单个模型单元（unit）直接分布到跨节点的所有 GPU 上会带来较大的通信开销，尤其在参数同步阶段。

为此，我们采用一种融合数据并行（DP）和张量并行（TP）的混合策略，如图所示。16 个 GPU 被划分为两个 **sharding group**，每组包含 8 个 GPU。每个 sharding group 内部对参数进行切分与重建，因此每组内部可以完整运行模型的前向与反向传播。这 8 个 GPU 在逻辑上共同构成一个模型副本。

同时，这两个 sharding group 彼此之间构成数据并行的副本。在训练过程中，仅需在 sharding group 之间同步梯度，而非参数本身，从而显著减少了跨节点的通信压力。梯度的同步通过一次组内的reduce-scatter和一次组间的all-reduce完成。

这种混合并行策略兼顾了参数切分的内存优势与跨节点通信的效率。

![6](https://hackmd.io/_uploads/rkzUvAaHgg.png)


### 通信优化

FSDP采用了一系列的通信优化技巧来尽量让通信和计算重合。一个示意图如下

首先最简单直观的是forward prefeching，在执行unit i forward的时候，提前all gather unit i+1的参数。

![7](https://hackmd.io/_uploads/rk-DP0THxe.png)


**Backward Prefeching**

在进行 backward prefetching 时，我们不仅需要提前执行参数的 All-Gather 操作，还需在适当时机进行梯度的 Reduce-Scatter 操作。

在原生执行逻辑中，我们通常先对第 *i* 个 unit 执行 Reduce-Scatter，再对第 *i−1* 个 unit 执行 All-Gather。但由于 FSDP 强制所有通信操作都在同一个 NCCL stream 上顺序执行，当前 unit 的 Reduce-Scatter 会阻塞后续 unit 的 All-Gather，从而阻碍下一步梯度计算，影响训练性能。

为避免梯度计算被阻塞，我们调整执行顺序：**优先执行 All-Gather，再进行 Reduce-Scatter**。这样可以确保梯度计算能够及时开始，而 Reduce-Scatter 本身不会成为关键路径上的瓶颈，因此其延后执行是可以接受的。

不过，这种 prefetch 策略需要提前知道在反向传播过程中下一个即将激活的 unit，以便正确地选择 All-Gather 的目标。为此，FSDP 在每次前向传播阶段记录各 unit 的激活顺序，并在反向传播中按其反序作为激活顺序的近似依据，从而实现动态、准确的 backward prefetching。

## 如何使用FSDP

本文基本参考torch官方文档：https://docs.pytorch.org/docs/stable/fsdp.html

FSDP的使用相当开箱即用，一个例子如下

```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

torch.cuda.set_device(device_id)
sharded_module = FSDP(my_module)
# 注意优化器要在FSDP初始化之后再初始化
optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)
x = sharded_module(x, y=3, z=torch.Tensor([1]))
loss = x.sum()
loss.backward()
optim.step()
```

下面介绍FSDP里的一些关键参数

**process_group** (*Optional[Union[ProcessGroup, Tuple[ProcessGroup, ProcessGroup]]]*)

`process_group` 是用于 FSDP 进行 all-gather 和 reduce-scatter 通信的进程组。

- 若设为 `None`，则使用默认的全局进程组。
- 对于混合并行（如 `HYBRID_SHARD`），可以传入一个二元组：
  - 第一个是用于sharding group内的进程组（intra-node sharding），
  - 第二个是用于dp之间交流的进程组（inter-node replication）。
- 如果未指定，FSDP 会自动构建合适的进程组。

**sharding_strategy** (*Optional[**[ShardingStrategy](https://docs.pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.ShardingStrategy)**]*)

切分的策略，基本对应zero的不同stage

- FULL_SHARD: zero的stage3。参数，梯度，优化器都被切分。
- SHARD_GRAD_OP: zero的stage2。梯度，优化器都被切分。
- NO_SHARD: 退化为DDP
- HYBRID_SHARD: 上文已做介绍。相当于node内FULL_SHARD，node之间做DP
- _HYBRID_SHARD_ZERO2: 相当于在node内SHARD_GRAD_OP，node之间做DP

**auto_wrap_policy** (*Optional[Union[Callable[[**[nn.Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)**,* *[bool](https://docs.python.org/3/library/functions.html#bool)**,* *[int](https://docs.python.org/3/library/functions.html#int)**],* *[bool](https://docs.python.org/3/library/functions.html#bool)**], ModuleWrapPolicy, CustomPolicy]]*)

`auto_wrap_policy` 参数用于指定一种策略，自动决定模型中的哪些子模块应被包装为独立的 FSDP 单元（即 fsdp unit）。fsdp unit 是参数切分和通信调度的基本单元，其划分方式直接影响训练时的性能、通信并发性和内存利用率。

当设置了 `auto_wrap_policy`，FSDP 会在构造过程中对模块树进行深度优先遍历，并依据该策略判断是否对某个模块应用 FSDP 包裹。若设置为 `None`，则仅最外层模块会被包裹，所有子模块需要用户手动包裹。

支持以下输入

| 类型                                   | 描述                                                         |
| -------------------------------------- | ------------------------------------------------------------ |
| Callable[[nn.Module, bool, int], bool] | 用户自定义的函数。接收三个参数： （1）当前模块； （2）是否递归模式； （3）当前模块中尚未被包裹的参数总数（numel） |
| ModuleWrapPolicy                       | 内置策略类之一。接收一组模块类型，对所有属于这些类型的模块进行包裹。 |
| CustomPolicy                           | 接收一个 lambda 函数，允许对特定模块返回布尔值或自定义参数配置，从而灵活控制每个模块的包裹行为。 |

FSDP提供了一些默认的策略

| 策略名称                    | 类型     | 包裹逻辑描述                                                 | 适用场景                             |
| --------------------------- | -------- | ------------------------------------------------------------ | ------------------------------------ |
| always_wrap_policy          | Callable | 总是返回 True，所有模块均包裹                                | 测试用，或需要极细粒度分片的场景     |
| size_based_auto_wrap_policy | Callable | 若模块中未被包裹的参数量 ≥ 设定阈值（如 1e8），则包裹        | 大模型分段包裹，按规模划分 FSDP 单元 |
| ModuleWrapPolicy            | 类实例   | 对指定类型的模块统一包裹，如 TransformerBlock                | 结构清晰的模型，如 transformer       |
| CustomPolicy                | 类实例   | 使用 lambda 函数，按模块名称或类型分别指定是否包裹及附加配置 | 精细化控制，如为部分模块指定不同策略 |

Note:

- 共享参数必须在同一个 FSDP 实例内包裹，否则可能引发错误；

**backward_prefetch** (*Optional[**[BackwardPrefetch](https://docs.pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.BackwardPrefetch)**]*)

控制反向传播过程中通信与计算的重叠程度，以提升训练吞吐。

不同选项对应不同的 all-gather 与梯度计算的调度顺序

- BACKWARD_PRE：最大重叠，显存开销最大。在当前梯度计算前就预取下一组参数（all-gather），相当于同时持有当前参数、下一组参数和当前梯度。[此方法在上文通信优化篇已介绍]
- BACKWARD_POST：中等重叠，显存开销适中。在当前梯度计算后再预取下一组参数，只需同时持有当前梯度和下一组参数。
- None：无重叠，显存最省但性能最差。通信与计算完全串行。

**ignored_modules** (*Optional[Iterable[**[torch.nn.Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)**]]*)

指定哪些module不应当被FSDP管理进行shard

**param_init_fn** (*Optional[Callable[[**[nn.Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)**], None]]*)

指定模型的初始化方式。用于正确的初始化参数。默认为None，fsdp会使用上文提到的replay方法。

**forward_prefetch** (*[bool](https://docs.python.org/3/library/functions.html#bool)*) 

是否在当前unit forward开始前all-gather下一个unit的参数。只对CPU-bound的情况有效，因为正常情况下CPU在当前unit foward开始之后之后就会invoke下一个unit的all-gather。此时model必须为static-graph model，因为prefetch依赖于unit的激活顺序。

**limit_all_gathers** (*[bool](https://docs.python.org/3/library/functions.html#bool)*)

有时候CPU的执行会比GPU快太多。例如GPU还在进行第i层的forward的时候，cpu已经发送了第i+1层的all-gather，第i+1层的forward，第i+2层的all-gather，导致过早all-gather了之后层的参数，增加了显存开销。（下图可以看到cpu的指令发出会比GPU上的执行快不少）。

因此我们可以限制CPU每次最多只能执行到下一层的all-gather，这就是为什么这个参数被叫做rate-limiter。

True（默认值）指启动限制，False指不加限制。

![8](https://hackmd.io/_uploads/Bk9uD0argx.png)


**总结**

FSDP还是相当开箱即用的，最复杂的部分就在于如何切分unit。这里需要考虑到切分的粒度，共享参数等等情况。

## RL框架适配

**Verl**

支持FSDP

**Slime**

支持sglang，sglang支持fsdp

**Areal**

支持自己插拔backend，自己写个fsdp backend就行

**Roll**

支持sglang

但是fsdp还是空的https://github.com/alibaba/ROLL/blob/main/roll/distributed/strategy/fsdp_strategy.py

## Appedix

### DeepSpeed和FSDP对印

![9](https://hackmd.io/_uploads/r1FYw0pBee.png)


### Ref

- https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/
- https://huggingface.co/docs/accelerate/usage_guides/fsdp
- https://huggingface.co/docs/accelerate/concept_guides/fsdp_and_deepspeed
- https://www.youtube.com/watch?v=By_O0k102PY&ab_channel=AhmedTaha
- https://www.youtube.com/watch?v=8_k76AHu__s&list=PL_lsbAsL_o2BT6aerEKgIoufVD_fodnuT&ab_channel=PyTorch
- https://huggingface.co/docs/accelerate/concept_guides/fsdp1_vs_fsdp2
- https://dev-discuss.pytorch.org/t/fsdp-cudacachingallocator-an-outsider-newb-perspective/1486
- https://arxiv.org/pdf/2304.11277