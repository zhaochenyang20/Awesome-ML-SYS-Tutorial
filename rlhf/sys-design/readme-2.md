# RL 系统深思：FSDP 训练后端

熟悉本系列的读者必然理解，我们绝大多数篇目都在讲述 RL 系统的 rollout engine，对训练后端的讨论并不多。我们会对现在火爆的 DeepSpeed AutoTP，FSDP 和 Megatron-LM 训练后端进行浅析。除此之外，torchtitan 也是被寄予厚望的训练后端，但是我听说作者离职了也不再维护，所以按下不表。本文只讨论 FSDP 的原理和实现，以及分析 verl 内的 FSDP 使用。

acknowledge:

Qiujiang Chen（SUSTech），Zhuoran Yin & Changyi Yang（CMU），Chenyang Zhao（Amazon）以及猛猿的原文。 

## DP，DDP 和 DeepSpeed Zero

这一部分其实猛猿的[数据并行上篇](https://zhuanlan.zhihu.com/p/617133971)与[数据并行下篇](https://zhuanlan.zhihu.com/p/618865052)已经讲得非常清楚了，非常建议去阅读原文，我也是学到非常多。

我在这里总结下[数据并行上篇](https://zhuanlan.zhihu.com/p/617133971)中我个人印象非常深的地方：

1. 最 naive 的 DP 可以用异步的方式进行优化：所有 worker 不用等到 weights server 把 weights 完全传回来，可以每次只更新部分 worker，其他 worker 继续用旧的参数计算 gradient。

2. All-reduce，reduce-scatter 和 all-gather 的通讯量：这是我收益最深的地方，首先，all-reduce 按照 ring 的方式来实现的话，实际上可以分为 reduce-scatter 和 all-gather 两个阶段，而且这两个阶段在 deepspeed 的三个阶段广泛地使用。在原文的图解中，已经很清晰地写明白了从每个 worker 上都有完整的 list 希望聚合开始，reduce scatter 和 all-gather 希望达成的效果，以及具体的通讯量。我们可以将两个步骤的通讯量各自近视为 $\phi$，其中 $\phi$ 是每个 worker 上需要聚合的列表的大小。很多时候一些讲述通讯量的文章会实际写出通讯量占据的存储大小，这里讨论的只是数据量的多少。

接着是第二篇文章[数据并行下篇](https://zhuanlan.zhihu.com/p/618865052)：


1. 拥有主副 weights 的混合参数训练往往有如下的流程：

<div style="text-align: center;">
  <img src="./pics/mixed.png" alt="mixed-precision" style="width:20%;">
</div>

我们在指 optimizer states 的时候，有些人会把 fp32 的主 model weights 也算在其中，有些人只考虑率 fp32 的 momentum 和 variance；但是显然，这三部分会占据相当大的显存。考虑大小为 XB 的模型，这三部分会有 12X 的显存占用。

接着，上图中蓝色部分的 FP16 的 model weights 永远是从主 weights 中 cast 得到的，不会直接用 fp16 的 gradient 来更新得到。

最后，activation 是一个比较大的变量，一般不会纳入显存占用的考虑。因为 activation 的显存占用不单单取决于模型大小，还取决于 batch size。此外，activation 也可以不存储，每次在做 backward 的时候重算即可。

2. 虽然 kv cache 对 inference 而言是核心，但我们在训练过程中极少考虑 kv cache：

- 训练 LLM 的本质是通过反向传播和梯度下降来更新模型的权重。这意味着我们需要计算损失函数相对于模型中每一个参数的梯度。如果引入 KV cache，Attention 机制的计算图会变得更加复杂。优化器需要跟踪哪些 K、V 是从缓存中读取的，哪些是当前计算的，以及它们是如何组合起来影响后续计算的。这会使得链式法则的实现变得异常复杂，因为需要确保梯度能够正确地回溯到所有参与计算的 K、V 值的原始来源，无论它们是缓存的还是新计算的。
- 引入 KV cache 会引入额外的显存占用，这部分开销显然是非常可观的。
- 模型的参数每次更新后，KV cache 都会完全失效。

3. zero 的三个 stage：

有了精度混合训练的背景，我们可以开始梳理 zero 的三个 stage 了。我很喜欢猛猿原文的叙述逻辑，但是我也向她反馈，捉了几个虫。最核心的问题是，在做 forward 和 backward 时依赖的 model weights 永远是从 optimizer 内存储的主参数 cast 下来的，不会直接拿着 gradient 来更新。三个 zero stage 对应的优化可以记为:

1. stage 1：切割整个 optimizer（optimizer state 和 fp32 的主 weights），保留 gradient 和 fp16 的副本参数。好处非常显而易见，整个 optimizer 存储在 fp32 下，是最占据显存的部分。经过切分后，在每一轮训练中，首先正常 forward 拿到 loss，backward 拿到 gradient，接着，假设一共有 N 个 dp worker，模型的参数量为 X；optimizer 只存储了 1/N 的参数和更新这部分参数所需的 optimizer state。因此，这里 gradient 只需要做一次 reduce-scatter，拿到更新自身 1/N 参数需要的 gradient 即可。对比于 DDP 的情况下，gradient 是需要做 all-reduce 的。如此以来，stage 1 在 gradient 上的通讯量反而节省了不少，近似为 X。而 optimizer 维护主 weights 更新后，向下 cast 到自身管理的那部分 fp16 附 weights，做一次 all-gather 就可以拿到整个 fp16 的附 weights。

2. stage 2：切割 optimizer 和 gradient。在 forward 和 backward 后，拿到完整的 gradient。每个 dp worker 只管理 1/N 的 gradient，因此还是进行一次 reduce-scatter 即可，然后将多余的 gradient 丢弃，余下的部分和 stage 1 一致。甚至这么分析，连通讯量都一致。听上去 deepspeed stage 2 真是完美的设计，不仅比起 stage 1 减少了显存，还在通讯量上没有增加。但是考虑到 gradient accumlation 后，情况就不同了。为了存储下每轮需要累加的 gradient，必须要每次 backward 后在所有 dp worker 上都对 gradient 进行 reduce-scatter，累计起自身管理的那部分 gradient，然后丢弃其余 gradient。如果直接丢弃不归自己管理的 gradient，就无法完成 gradient accumlation。所以 stage 2 的通讯量在开启 gradient accumlation 后，反而比 stage 1 要大。

3. stage 3：进一步完全切割所有部分，包括 optimizer，gradient，以及 fp16 的附 weights。这样一来，在 forward 的时候要将附 weights 进行 all-gather，在 backward 的时候同样要 all-gather weights，然后拿到完整的 gradient 后进行 reduce-scatter，留下自身管理的 gradient。这部分 gradient 更新了自身管理的 optimizer 后，向下将主 weights cast 到 fp16 的附 weights，即可。通讯量相比 stage 2 进一步增大。

总的来说，zero 就是通讯换显存的典型代表，且随着 stage 的增加，通讯量会越来越大。此外，虽然 zero 也会对 model weights 进行拆分，但是我们可以看到在进行 forward 和 backward 时，依赖的 model weights 是需要通过 all-gather 完整聚合起来才能进行计算的。所以这样还是 DP 的思路。与之对比的 TP 是直接进行 forward 后拿到部分 activation，对 activation 进行聚合即可。这也是 FSDP 名字的由来，Fully Sharded Data Parallel。

## Zero 和 FSDP

FSDP 高度继承了 zero 的思想，但是比起 zero 更加激进，默认采用 stage 3 的策略，所有能够 shard 的部分都能拆则拆。

1. 按层（layer-wise）甚至按参数组（parameter group-wise）进行分片。
2. 按需激活：在正向传播时，当某个层需要计算时，它只 all-gather 这一层的参数，计算完成后再丢弃这一层，反向传播也是类似的过程。这与 zero Stage 3 每次都 all-gather 完整 FP16 weights 有所不同。
3. FSDP 在反向传播过程中，当特定层的梯度计算完成后，会立即进行 reduce-scatter 操作来同步和累积这些分片后的梯度，而不是等到所有梯度都计算完再进行全局 reduce-scatter。
4. 激活重计算（Activation Checkpointin）：虽然这不是 FSDP 独有的特性，但 FSDP 经常与激活重计算（`torch.utils.checkpoint`）结合使用。激活重计算通过在反向传播时重新计算激活值来节省显存，而不是在 forward 时存储它们。

最后，FSDP 是 torch native 的，至少我知道埋怨的声音比起 deepspeed 要少很多。

-----

## FSDP1 与 FSDP2

我们通过一个例子来观察 FSDP 1 的局限性：假设有一个包含 3 个 `Linear` 层的 Layer，使用 FSDP 在 2 个 GPU 之间进行分片。

在 FSDP1 中，它会将每个模块表示为一个单独的 `FlatParameter`。这个 `FlatParameter` 是一个巨大的一维张量，包含了该模块所有参数的扁平化表示，然后这个扁平化的张量才在各个 Rank 之间进行分片。

<div style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/accelerate/fsdp1.png" alt="fsdp1" style="width:30%;">
</div>

这种设计带来了一个显著的问题：当所有参数被合并成一个巨大的 `FlatParameter` 后，一些 meta data 无法得到很好的管理。由于 `FlatParameter` 是一个单一的、同质化的对象，要保留每个原始参数的独立元数据，就需要借助一些复杂且笨拙的方法。为了优化这个情况，与 FSDP1 相比，FSDP2 引入了 `DTensor`（Distributed Tensor）来进行参数管理。`DTensor` 是 `torch.Tensor` 的分布式版本，支持按指定维度在多个 Rank 间进行分片，而且原生携带了关于原始张量的所有元数据，例如其 `dtype`、`requires_grad`、具体的分片方式，还有 `placement types` 等信息。

如下图所示，在 FSDP2 中，每个 `Linear` 层的参数都会被单独表示为一个 `DTensor`，并在第 0 维上在两个 GPU 之间进行分片：

<div style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/accelerate/fsdp2.png" alt="fsdp2" style="width:30%;">
</div>

这种逐参数分片的方式更加直观和高效。它避免了 FSDP1 中参数先展平、拼接成一个大张量，再进行整体切分，最后还需要推断每个进程保存哪些片段以及如何还原原始结构的繁琐流程。

除此之外，FSDP2 在保持了 FSDP1 相当的吞吐量的前提下，进一步提升了可用性、扩展性和功能性：

1. 简化实现：将 parameter 转为独立的 `DTensor`。这种设计消除了 FSDP1 中将所有参数展平为一个大型 `FlatParameter` 的复杂逻辑，避免了参数拼接和管理内部偏移量。
2. 原生支持部分参数冻结：FSDP2 对 `DTensor` 的原生支持使得像 LoRA 这样的微调方法能够开箱即用，无需额外的复杂封装或修改。
3. 优化 checkpoint 管理：它默认使用 `SHARDED_STATE_DICT` 来存储检查点，每个 Rank 只保存部分模型参数和 meta data；而且支持直接读取分片的 `state_dict` 来完成模型加载。因此，在存储 checkpoint 时，不用先 all gather，存储效率有所提高。此外，FSDP2 支持异步 checkpoint 保存，先将参数先复制到 CPU，主训练线程继续运行，而读写进程将数据写入磁盘，减少 I/O 阻塞。
4. 稳定的内存管理：FSDP2 不再依赖 `torch.Tensor.record_stream` 这种有时会导致非确定性行为的机制。相反，它通过更底层的 CUDA 多流同步来实现稳定且可预测的内存行为。这种改进减少了因内存管理问题导致的训练中断或性能波动，从而变相提升了训练的总有效时间。
5. 通信与计算重叠：FSDP2 默认采用了 FSDP1 中需要手动配置的 `BACKWARD_PRE` 选项，并且提供了隐式预取（implicit prefetching）来更好地重叠 all-gather 通信与计算。这种优化能够更有效地隐藏通信延迟，提升 GPU 利用率，从而在理论上提高训练吞吐量。
6. 更好兼容：FSDP2 更高的与 FP8 等混合精度类型以及 2D/3D 并行策略结合。


## FSDP 的理论峰值与初始化

本节基于[fsdp 原文](https://arxiv.org/pdf/2304.11277)，讨论 stage3 fsdp1 的情况。

<div style="text-align: center;">
  <img src="./pics/fsdp-algorithm.png" alt="fsdp-arch" style="width:30%;">
</div>

首先，fsdp 会把参数切分成许多 unit。比如，在 MHA 中，一个 attention 层和一个 FFN 层通常组成一个 Transformer block，可做作一个 unit。unit 是 fsdp 管理参数的最小粒度单元。

在前向和反向传播的时候，FSDP 只会 materialize/gather 一个 unit 的参数和梯度。除了这个 unit 之外，其他的参数和梯度都保持 shard 状态。计算完成后，立刻丢弃（de-materialize），所以，FSDP 在单个 GPU 上的 peak memory 可如此计算：

```python
GPU Peak Memory = ShardModelSize + ShardOptimizerSize + ShardGradientSize + MAX(fullyMaterializedFSDPUnit)
```

DDP 初始化模型时，PyTorch 会在单个 GPU 上完全实例化（materialize）整个模型，所有参数都会在内存中一次性分配。然而，对于 FSDP 而言，这种一次性完全加载的方式是不可行的，因为整个模型可能根本无法放入单张 GPU 的内存中。FSDP 通过延迟初始化（Deferred Initialization）解决这个上述问题。FSDP 首先在一个特殊的 “fake device” 上初始化所有张量。在这个阶段，张量并不会分配任何实际的物理存储。相反，系统只会记录下每个张量的构造逻辑及其所涉及的操作（例如，张量的形状、数据类型、初始化方法等）。只有当张量被显式地移动到 GPU 上时，之前记录的构造操作才会被重放（replay），开始分配物理内存，并完成其真正的构建和数值初始化。此外，为了最大限度地节省内存，每张 GPU 只需初始化它本地负责的那部分参数分片。然而，由于模型初始化函数可能包含复杂的逻辑，无法精确判断用户是否对未分片的张量进行了依赖操作（这可能导致需要在初始化阶段临时聚合一些参数），FSDP 实际上的策略是：

1.  **fake init：** 模型在 fake device 上被构造，只记录下所有张量的构造逻辑，不分配物理内存。随后被划分为多个 FSDP unit。
2.  **materialize 单个 Unit：** 每次移动一个 unit 到实际的 GPU 上，这一 unit 内的所有张量的构造逻辑被 replay，从而完全实例化出这个 unit 的完整参数。
3.  **shard 单个 Unit：** 紧接着，这个已实例化的 unit 会立即被分片，将其参数分散存储到不同的 GPU 上。

以上策略在不同 fsdp unit 之间有 dependency 的时候都有可能失效，此时无法通过 replay 正确的构建 tensor。FSDP 提供了两个代替策略：

1. 直接在单 GPU 上初始化 unshard model：虽然模型可能无法在单卡上训练，但是有可能可以在单卡上初始化。因此，可以先在单卡上初始化模型之后再 shard。
2. 在 CPU 上初始化 unshard model：因为 CPU 的内存大小往往比 GPU 大得多。之后，unit by unit 的把每个 unit 转移到单个 gpu 上，再执行 shard。但是初始化 model 可能会变得非常耗时。

## FSDP in verl

verl 默认使用 FSDP1，而且非常流畅地支持 FSDP2，只需要设置：

```bash
actor_rollout_ref.ref.strategy=fsdp2
actor_rollout_ref.actor.strategy=fsdp2
critic.strategy=fsdp2
reward_model.strategy=fsdp2
```

此外，FSDP2 的 CPU offloading 与梯度累积兼容。可以通过设置 `actor_rollout_ref.actor.fsdp_config.offload_policy=True` 来启动。


这里具体举出 verl 当中的实现，代码见 [fsdp_workers.py](https://github.com/volcengine/verl/blob/ab11fff33dcaa2409e388ce2f19aff440a5b703f/verl/workers/fsdp_workers.py#L377)：


对 fsdp1 而言，直接调用 `torch.distributed.fsdp.FullyShardedDataParallel` 来包装 `actor_module`（load 起来的 huggingface 模型），返回一个新的 FSDP 对象来管理分布式训练。此外，支持对 FSDP1 的细粒度配置，如 `cpu_offload`、`auto_wrap_policy`、`sharding_strategy` (默认选择 zero3)、`mixed_precision` 和 `forward_prefetch`。此外，对于 reference policy，强制使用 CPU offload 来节省内存；但是 actor 强制关闭 offload，开启会导致梯度累积结果错误，同步机制存在时序问题。

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

FSDP2 的写法看着比起 FSDP1 要复杂些：

1. 原地修改（In-place Modification）： FSDP2 不再创建新的包装对象，而是通过 `apply_fsdp2` 直接修改原始的 `actor_module`。在 `apply_fsdp2` 调用之后，`actor_module` 的内部参数已经变成了 DTensor，并按照 FSDP2 的策略进行了分片。
2. 由于 `apply_fsdp2` 会改变参数的分片方式，因此需要一个特定的流程来加载权重。首先，保存原始模型的完整 `state_dict` (`full_state = actor_module.state_dict()`)。接着，通过 `apply_fsdp2` 原地修改 `actor_module`，使其参数变为 DTensor 并分片。此时 `actor_module` 的参数结构已经改变，但参数值尚未正确加载。最后，使用 `fsdp2_load_full_state_dict` 函数将之前保存的 `full_state` 加载回被 FSDP2 处理过的 `actor_module`。
3. CPU Offload 与梯度累积兼容： 和 FSDP 1 相对的，通过设置 `fsdp_config.offload_policy=True`，FSDP2 允许 actor 模型安全地开启 CPU Offload，而不会影响梯度累积的正确性。解决了 FSDP1 中存在的时序问题，使得 FSDP2 在内存受限且需要使用梯度累积的场景下更具优势。

```python
from verl.utils.fsdp_utils import apply_fsdp2, fsdp2_load_full_state_dict

elif fsdp_strategy == "fsdp2":
	assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
	mp_policy = MixedPrecisionPolicy(
		param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
	)
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