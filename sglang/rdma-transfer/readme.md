# 训推秒传万亿参数的开源实现

去年开始参与RL infra方面的工作。当时的系统，想要更新推理测的参数需要经过硬盘 — 可想而知更新速度惨不忍睹，各个推理单元争夺硬盘带宽，直到系统的其他部分崩溃罢工。那时候拜读了乐群的"[跨机秒传RL模型参数更新的一些探索](https://abcdabcd987.com/2025/09/07/rl-weight-transfer/)"，惊为天人。既然自己不擅长造轮子（手搓不了新的rdma通讯库），觉得能把轮子装上，给开源社区助助力也在作出贡献。

半年之后，我们的最终成果是能在有IB的H100上把512卡，1T的Kimi fp8模型7秒传完，以及744B bf16的GLM5模型8.5秒传完，比起之前开源的解决方案快了7倍左右；并且能够支持所有主流开源模型和两侧并行逻辑。注意这7秒包括了从推理引擎暂停（pause）到恢复（continue）的所有时间消耗，也就是整个RL训练暂停的时间。本文回溯这半年的开发历程，分享和小伙伴们踩过的各种坑。核心代码和使用可见[Miles上的实现](https://github.com/radixark/miles/blob/main/miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p.py)，[Sglang的介绍](https://github.com/sgl-project/sglang/issues/17311)，和[在sglang-miles上的实现](https://github.com/sgl-project/sglang/pull/21278)。

## 设计传输逻辑

[乐群的系统设计](https://abcdabcd987.com/2025/09/07/rl-weight-transfer/)是追求极致性能 — 通过实现新的底层rdma传输库，寻求权重层面上最优一对一映射来适配fsdp和推理引擎之间的传输策略，并支持了qwen3和Kimi。我们希望可以尽量复用现有的底层库，支持多种训练后端（megatron/fsdp），尽可能支持所有开源模型，并行和量化设置。为此，我们可以接受一些效率上的损失。

回到目前在 slime/miles 上的开源解决方案。现在的UpdateWeightFromDistributed实现依赖于每个pp group的head rank和所有推理rank组建的 NCCL broadcast 原语。在源端，所有节点首先参与 TP 和 EP 维度的 all gather，在每个 PP rank 的 head rank 处生成汇总权重。这个权重会在此转换到hf格式。随后，head rank 参与分布式更新组，通过 update_weight_from_distributed API 将完整hf权重 broadcast 到每个 engine rank，由 local rank 通过 load_weight API 加载其对应的 shard。此过程在每个 PP rank上重复，并分桶循环运行来降低GPU内存压力。

![NCCL update flow](./pics/im0.png)

可以看到，目前开源基于NCCL broadcast的解决方案有以下问题：

- **冗余 (Redundancy)：** 相同的数据在网络中被多次重复发送。
- **闲置 (Inactivity)：** 大多数训练侧 rank 在传输期间处于空闲状态，只有少数参与 broadcast。
- **僵化 (Rigidity)：** NCCL 通信群组在定义后无法被更新，无法支持新加入的引擎。

另一方面，这个更新范式有自己的优势：通过hugging face full 权重作为训推交互的唯一接口，训推两侧具体的并行模式可以被完全抽象出来。原则上它可支持任何模型，并行设计，以及量化操作。

我们这时候想到了sglang上近期支持的权重 [remote fork](https://www.lmsys.org/blog/2025-12-10-rfork/)，一种权重远程加载的机制。通过读取现有推理引擎的权重注册地址，新的推理引擎可以绕过硬盘读取直接快速启动。它的传输层用了[Transfer Engine](https://kvcache-ai.github.io/Mooncake/python-api-reference/transfer-engine.html)来支持RDMA通信，并会自动检测网络拓扑和传输优化。它也是目前[Sglang PD分离功能](https://www.lmsys.org/blog/2025-07-20-k2-large-scale-ep/)的后端之一。在transfer engine 注册过的内存/显存地址可以通过RDMA协议进行网络直传，绕过kernel缓存和CPU媒介。所以轮子已经有了（RDMA 通讯库），地址也可以通过[get_remote_instance_transfer_engine_info](https://github.com/sgl-project/sglang/blob/v0.5.10/python/sglang/srt/entrypoints/http_server.py#L1008)从sglang上拿到了，要怎么装上这个轮子呢？

## 设计方案1：用rdma代替nccl

![P2P update Proposal 1](./pics/im2.png)

最简单的策略是替代传输策略，完全不改变任何训练和推理段的现有逻辑。在每次分桶传输时，在all gather之后，将整个hugging face 权重传输到推理端。由于采用P2P的传输机制，这时候可以利用更多的训练侧rank来传输，看起来已经可以解决闲置（inactivity）和僵化（rigidity）的问题。

但这个设计并没能解决冗余，因为我们依旧需要传输整个模型权重到推理侧。另外我们了解到，注册显存是个开销很大的过程，需要尽可能避免 – 但在这个方案里，针对每一个权重，在每一次传输时，我们都需要重新注册并销毁地址注册。那有没有更优雅的解决方案呢？

## 设计方案2：在训练侧建一个推理引擎副本

![P2P update through source replica](./pics/im3.png)

如果我们复用remote instance weight info的接口，就无需重新注册推理侧的权重。相应的，我们可以在训练侧也预留一个推理引擎的模型副本，并进行注册。由于这时候训练和推理侧的权重格式是完全一样的，我们只需要传输推理引擎真正需要的权重；只需要注册一次；如果有新的引擎加入，改动也非常容易。这看上去解决了冗余；闲置；僵化这所有三个问题！

为了实现这样一个引擎副本，我们在Sglang上添加了一个接口来导出每一个rank上的并行定义。调用的时候可以直接：

```python
model_parallelism_info = engine.get_parallelism_config(rank)
with ParallelismContext(RankParallelismConfig.from_dict(model_parallelism_info)):
    model_replica = get_model(
        model_config=model_config,
        load_config=load_config,
        device_config=device_config,
    )
```

## 训推映射关系

有了新的设计，我们只需要加上新的p2p映射关系即可。逻辑本身比乐群当时的设计更简单 --- 由于我们创建了引擎副本，我们不需要考虑权重层面的传输逻辑，只需要设计训练rank和推理rank之间的映射关系即可。

![Mapping diagram](./pics/im4.png)

已知：每一个推理引擎必须收到所有权重。

求：如何能在每个训练端用最少的引擎副本来满足？

答：将每个训练 rank 映射到其目标推理侧引擎 rank。使用带负载均衡的轮询（round-robin）分配：前几个推理 rank 获得 1:1 映射，剩余目标均匀分布。

**练习题：** 假设训练 pp=4，有 32 个训练 rank 和 2 个 Sglang 引擎实例（每个实例 16 个 rank）。每个训练端需要几个引擎副本？

**答案：** 在 all-gather 之后，每个 PP 组中的每个 rank 都包含了该特定 PP rank 的完整汇总权重。每个目标 rank 需要从每个 PP rank 接收权重。我们从 pp_rank=0 开始：需要将 8 个训练 rank 映射到所有 32 个目标 rank。

1. 轮询映射：src_rank 0 -> tgt_rank 0, ..., src_rank 7 -> tgt_rank 7。
2. 所有现有源端负载相同。进行另一轮轮询分配：src_rank 0 -> tgt_rank 8, ..., src_rank 7 -> tgt_rank 15。
3. 最后，注意 tgt_rank 16 与 tgt_rank 0 是完全等价的（属于不同实例的相同tp rank）。回顾现有分配，将相同的引擎 rank 添加到其现有源端。最终形成 src_rank 0 -> [tgt_rank 0, 16]; [tgt_rank 8, 24] 等。
4. 以此类推到所有其他pp rank — 类似的，src_rank 8 -> [tgt_rank 0, 16] ; [tgt_rank 8, 24]

所以每个训练端需要生成两个副本 --- 在训练rank0上，一个副本传到0，16，另一个副本传到1，17。

## 权重映射关系和流水线更新

为了能实现分桶流水线更新，我们还需要找到hugging face权重和sglang权重之间的映射关系。这个环节比较繁琐，本质上是剖析每个模型在sglang load_weight的过程中到底做了什么，以及什么时候一个sglang的权重彻底更新完成了。这里我们在sglang构建了一个新的类：[ParameterMapper](https://github.com/sgl-project/sglang/pull/20907)来独立解析映射关系：

```python
sglang_name, shard_id, num_shards, expert_id, num_local_experts = parameter_mapper.map(hf_tensor)
```

只有当所有 num_shards 以及所有的 num_local_experts 都更新完毕后，我们才能用transfer engine发送该 Sglang 权重。

每当一个副本里的权重更新完成，我们把任务递交到一个Threadpool里，由transfer engine解除GIL并负责传输。因此在主线程上，只需要做all gather和load weight：

![Pipeline diagram](./pics/im5.png)

写成伪代码：

```python
for hf_tensors in all_gather(self.bucketed_update()):

    ready_tensors = []

    for hf_tensor in hf_tensors:
        sglang_name, shard_id, num_shards, expert_id, num_local_experts = parameter_mapper.map(hf_tensor)
        ready_tensor.append(self.is_tensor_ready(sglang_name, shard_id, expert_id))

    for engine in local_engine_replicas:

        engine.load_weight(hf_tensors)

        for target_rank in self.get_target_ranks(engine):

            submit_transfer(ready_tensor, target_rank, self.thread_pool)
```

我们用slime/miles自带的check weight equal flag来验证传输得到的权重和预先准备的硬盘上的权重能做到一一对应，以证明正确性。

这里有一个有趣的插曲：最初我们考虑到大规模 EP/TP 场景下 P2P 连接数可能较多，专门为每个传输线程分配了独立的 transfer engine 实例，以 engine pool 的形式来分散高负载下的并发压力。然而后续实验表明，transfer engine 内部已经具备了相当成熟的高并发调度与连接复用机制，多个传输线程完全可以共享同一个 transfer engine 完成传输。收敛到单一实例后，不仅减少了 NIC 等硬件资源的重复初始化开销，还规避了多个 transfer engine 各自注册的 memory region 无法跨实例共享的问题。

## 初战告捷 235B

在Qwen3-4B上跑通之后，我们马上在Qwen235B上也成功跑通了bf16的64卡传64卡，3.5秒左右，对比原先的nccl快了三倍多。但和miles的maintainer聊过之后，我们意识到这个方案有致命缺陷：额外占用的显存会极大得影响训练效率 — 无论是迫使batch size变小，开启cpu offload或者checkpointing，都会大幅降低训练速度。我们必须想办法在训练期间想办法释放这部分显存。

我们于是开始尝试在训练期间把引擎副本放上CPU，等到更新完成需要传输之前再挪回GPU。

### 显存优化尝试1：torch memory saver

第一个优化思路，是尝试用torch memory saver确保虚拟显存地址（virtual address）不变，以此来允许同样的地址注册被复用。Sglang用torch memory saver实现了offload功能，允许编译好的CUDA graph复用同样的虚拟地址，不同的物理地址。

可惜的是，我们尝试之后才发现transfer engine 目前并不能直接支持虚拟地址。RDMA传输的本质是绕开OS kernel管理，由显卡驱动（Driver）直接通过linux 内核（kernel）peermem在NIC网卡上注册静态映射表（Memory Translation Table）。目前peermem的注册并不支持新的CUDA 内存管理接口（VMM API）。要改动网卡上的静态注册（MTT），需要繁琐的硬件锁和动态地址追踪才能保证正确性，因此目前没有广泛支持。这也是为什么miles里只能有限支持VMM API，在权重传输和DeepEP的场景之下必须换回cudaMalloc来分配显存。

### 显存优化尝试2：流水线优化

另一个思路是，通过流水线优化尽可能把注册和注销显存的操作藏在ep，tp allgather之外。在多次torch profiler之后我们发现，235B模型的显存注册事件本身就能达到6秒左右，远远超过了传输本身 — 这意味着这个方案也是不可行的。

### 显存优化尝试3：不用显存

忽然我们意识到，为什么要用显存做传输发起方？Transfer Engine同时支持内存和显存 — 我们完全不需要把它重新onload到GPU上。测试后我们发现，传输效率本身并没有受到影响！唯一的性能影响是all gather聚合后的权重需要先d2h传到CPU上。并且，由于注册发生在内存而不是缓存上不需要通过CUDA，注册时间也更快了。

### 未来显存优化尝试：Huge Page

从马腾老师这里我们了解到GPU注册时间长的本质原因是OS的默认页尺寸（page size）太小（4kb），造成注册一个常见的模型需要大量的页，造成页表项（Page Table Entries）数量过载。虽然 GPU 数据流不经过 OS，但 RDMA 注册（控制流）必须经过内核驱动。若以默认 4KB 粒度注册 80GB 显存，会产生约 2000 万个页表项，内核在锁定物理地址并同步到网卡 MTT 表时，巨大的 CPU 循环导致了秒级延迟。如果自己建一个可以支持Huge Page的MemPool给pytorch调用，完全是可以将注册时间控制到一个合理范围之内的。马腾老师的实验里，用32MB的页尺寸注册时间可以压缩到2秒之内。这也是将来一个可以考虑的优化方向。

## 内存OOM！

在接下来的实验里，我们开始攻克GLM4.5（335B（32B active）），可是却在64 → 64卡的实验里遇到了内存OOM。在训练端，我们有TP=8, EP=8, PP=8, CP=2, ETP=1, 8 节点，推理端TP=32, EP=32, DP_ATTN=4, 8 节点，两侧都是bf16，我们现在的设计占据了多少内存？

**答：** 训练端8个节点，并且pp=8，意味着每个节点（0-8 rank）需要传权重给所有的推理侧rank，也就是每个节点必须存有一整个推理模型的权重！实际情况是，由于开启了dp attention，非专家部分的权重甚至需要存储4倍，也就是（340B + 15B*4）*（2 bit/tensor）→ 每个节点上需要占据整整800G的内存，难怪它立刻就OOM了。每个rank需要4个推理副本，总共32个副本，都存在这个节点的内存里。常见的H100节点内存一般在1-4T区间，这个消耗无法接受了。

## 共享副本和流水线更新

这时候我们注意到，sglang引擎里的每个rank，本质上模型结构是完全同质的（除了最新的[EPD](https://www.lmsys.org/blog/2026-01-12-epd/)）。也就是说，每个rank的每个tensor，本质上是同样的尺寸和格式，只是数值有区别。因此，我们可以让每个rank上所有的引擎副本，都共享同一个物理内存（shared replica）。

在具体实现上，我们维护每一个副本的weight loader（他们会自动抽选自己需要的shard），但是让物理内存被分享。在每个rank上，我们只需要维护一份权重，和一个ParameterMapper。

```python
with ParallelismContext(parallelism_config):
    model = get_model(
        model_config=ModelConfig(model_path),
        load_config=load_config,
        device_config=DeviceConfig(device="cpu"),
    )

if first_engine_rank:
    for param in model.parameters():
        param.data = param.data.pin_memory()
    self._shared_params_dict = dict(model.named_parameters())
    self._shared_param_mapper = ParameterMapper.from_model(model)
else:
    for name, param in model.named_parameters():
        param.data = self._shared_params_dict[name]
```

这时候我们需要注意传输的正确性。同样的引擎副本会被传输到不同的目标引擎上，必须保证旧的传输完成之后才能载入新的权重。我们依然复用之前的Threadpool，但发送任务前要确认是否需要等待任务完成才能继续下一步的all gather和load_weight操作。还是以之前的传输为例子，训练侧的rank 0 要用两个副本传给4个目标rank：

src_rank 0 -> [tgt_rank 0, 16]; [tgt_rank 8, 24]

在第一组权重在内存更新完成后（load_weight），我们可以同步发送[0, 16] 的传输任务，因为他们背后的权重是完全一样的。但这里我们需要等待传输任务彻底结束之后，才能更新第二个副本，因为这两个副本背后使用的同样的内存地址。用第二个副本发送[8, 24] 的时候，就无需等待，可以像之前一样直接进入下一个all gather的步骤。

![Shared replica pipeline](./pics/im6.png)

可以看到传输完第一个副本（Engine 0）现在是流程继续的必要步骤。

另一个改变是对权重分片（weight shard）的处理，例如qkv proj。一个副本上只会存储hugging face权重的一部分信息（选择对应的tp，ep），所以load_weight会丢弃其他多余的部分。但由于权重分片的存在，我们需要通过ParameterMapper才能知道那些权重已经全部更新完成。这时候，如果我们过早的进行load_weight，就会直接损失一些还需要的信息。举个例子：

bucket 0: [q_proj, k_proj] —> load_weight(tp=0); load_weight(tp=1) → qkv 现在只有tp=1的信息，不做任何传送因为ParameterMapper告诉我们权重还不ready

bucket 1: [v_proj] —> load_weight(tp=0); → 传输qkv_proj报错！因为这时候的q和k的两部分已经在上一步被更新到了对应tp=1的数值。

因此，我们需要额外维护这些hugging face权重本身，直到对应的sglang权重集齐了所有的分片。化成图的话，差不多是这样：

![Buffered weight shard flow](./pics/im7.png)

训练端需要额外维护还未准备好的权重，更新副本，再依次发送。send1必须在update2之前完成，send2不会阻塞下一步的all gather。

最后，共用共享副本的传输效率代价是怎么样呢？我们在各个模型尺寸上做了比较，结果是在多节点（node >8）之后对传送速度影响很小，属于在统计误差之内的区别。一个合理的解释是在多节点，多专家的情况下，对比跨机all gather所有的专家权重，传输本身所占用的时间实际上非常小，所以这是一个合理的优化。

## 估算传输优势

回看我们的新设计，最后我们用额外的存储（内存上的引擎副本）来换取来传输效率。我们通过这样的场景来量化：

训练侧有 M 个 rank，Sglang 推理侧有 N 个目标 rank；训练 rank 的 pp_size 为 pp，目标 rank 的 ep_size 为 ep；每个引擎 rank 拥有 P 个参数。我们还分配了 K 作为分桶 all gather 的内存缓冲区。如果假设模型仅包含专家权重：

| 指标 | NCCL Broadcast | RDMA P2P |
|------|---------------|----------|
| 训练端参与传输的 Rank 数量 | pp | M |
| 每个推理端 rank 接收的参数量 | ep x P | P |
| 源端额外分配的缓存 Buffer（用于allgather） | K | K* + P |
| 目标端额外分配的缓存 Buffer | K | 0 |

新的P2P很好的解决了之前的问题：

1. **冗余** → 只传输模型真正需要的参数
2. **闲置** → 所有训练侧都参与传输
3. **灵活** → P2P设计，新开一个推理侧只需在相应的训练侧增加一个目标。

为什么额外分配的缓存在P2P是K*？这是由于我们需要额外维护还未更新完毕的hugging face分片。理论上最坏情况下，我们需要维护3*K（q，k，v）；但实际上他们在model_parameter的出现是按照模型层数有序的，所以额外占用的显存非常有限。我们的传输范式本身是write-only，sglang并不知道自己的权重正在被更新。

## 后量化处理，和完整更新步骤

在GLM5和Kimi K2的实验里，我们遇到了新的复杂情况 — check weight equal 没问题，但是logprobs对不上；或者flashinfer的权重会有错。在阅读sglang对于DeepSeek系列模型load weight部分之后我们发现，常见的load分两个部分：

- `load_weight()`：读取并载入checkpoint里的权重到model parameters中
- `post_load_weights()`：五花八门的后处理，主要是改变权重格式（layout），精度（quantization），和硬件对应的优化。比如DeepSeek的MLA，会生成新的w_kc, w_vc权重。

第二个部分通常需要在GPU上完成，更棘手的是这些后处理生成的权重并不被model parameters所包括，所以无法被 check weight equal检查到。

但既然模型本身的权重拥有所有的信息，我们可以完整保留这些逻辑，让他在推理侧重新运行。因此在我们的实现中，我们在训练侧强制跳过post_load_weights，并通过新的[sglang API](https://github.com/sgl-project/sglang/pull/15245)，在推理侧权重更新后，重新调用post_load_weights实现。

我们的P2P更新逻辑大量复用现有的UpdateWeightFromDistributed范式，仅仅改变对all gather后的hf tensor的分桶更新实现。最后，我们需要在训练侧实现的步骤如下：

### 正式运行之前

| 步骤 | 说明 |
|------|------|
| get_remote_instance_transfer_engine_info | 得到推理侧权重注册信息 |
| get_parallelism_info | 得到推理侧并行信息（tp，pp，etc） |
| build_transfer_plan | 得到训推一一对应关系 |
| create_engine_replica | 生成训练侧的引擎副本 |

### 更新过程中

| 步骤 | 说明 |
|------|------|
| pause_and_register_engine | 调用sglang API暂停推理，注册副本权重内存（第一次） |
| update_weight(non-expert and expert) | 分桶更新权重，分专家和非专家依次进行 |
| post_process_weights | 调用sglang API进行权重后处理 |
| update_weight_version | 调用sglang API更新权重版本 |
| continue_generation | 调用sglang API恢复推理 |

## 实验结果

接下来我们在配有 Infiniband 连接的 H100 8 GPU 主机上测试了常见开源模型的传输速度。这些是除去第一步更新后，十次更新的平均值；这个数字包括了以上的整个更新流程。

| 模型家族 | 模型名称 | 总参数量 | 训练配置 | 推理配置 | NCCL (毫秒) | RDMA (毫秒) | 对比 |
|---------|---------|---------|---------|---------|-----------|-----------|------|
| GLM4 | GLM-Z1-9B-0414 | 9B | TP=2, PP=1, CP=2, EP=1, ETP=1, 1 节点 | TP=4, EP=1, 1 节点 | 694.6 | 707.1 | +1.8% |
| DeepSeek-V2 | Moonlight-16B-A3B | 16B(3B) | TP=2, PP=1, CP=1, EP=8, ETP=1, 1 节点 | TP=8, EP=8, 1 节点 | 1,482.0 | 1,073.3 | -27.6% |
| GLM4-MoE | GLM-4.7-9B-Flash | 30B(3B) | TP=4, PP=1, CP=1, EP=8, ETP=1, 1 节点 | TP=4, EP=4, 1 节点 | 2,508.6 | 4,229.0 | +68.6% |
| Qwen3-MoE | Qwen3-30B-A3B | 30B(3B) | TP=4, PP=1, CP=1, EP=8, ETP=1, 2 节点 | TP=8, EP=8, 2 节点 | 2,670.0 | 2,160.2 | -19.1% |
| GLM4-MoE | GLM-4.5-Air | 106B(12B) | TP=1, PP=4, CP=1, EP=8, ETP=1, 4 节点 | TP=8, EP=8, 4 节点 | 5,001.1 | 2,637.2 | -47.3% |
| Qwen3-MoE | Qwen3-235B-A22B | 235B(22B) | TP=4, PP=4, CP=2, EP=16, ETP=1, 8 节点 | TP=32, EP=32, 8 节点 | 10,753.6 | 3,162.0 | -70.6% |
| DeepSeek-V3p2 | GLM-5 | 744B(40B) | TP=4, PP=8, CP=2, EP=16, ETP=1, 16 节点 | TP=64, EP=64, 16 节点 | 58,301.5 | 8,479.7 | -85.5% |
| DeepSeek-V3 | Kimi-K2-fp8 64-block-quantized | 1T(64B) | TP=8, PP=8, CP=4, EP=32, ETP=1, 32 节点 | TP=32, EP=32, 32 节点 | 53,279.1 | 7,227.3 | -86.4% |

从这张表来看，P2P传输在推理侧具有高ep的大型 MoE（混合专家）架构中，性能提升最为显著。在上述 GLM4-MoE 示例的小规模节点配置下，当 EP（专家并行）较小时，在本地将权重加载到 CPU 推理引擎副本的开销超过了 P2P 传输带来的收益。

**注意：** Kimi-K2 特殊处理：在所有模型中，Kimi 是唯一使用 FP8 block quant（块量化）的模型。我们设置了训练节点 = 推理节点 = 32 以确保有足够的内存。原始的 Kimi-K2 权重使用 block_quant 进行fp8量化，窗口大小[128, 128]，这在 sglang-tp-size = 32 时会触发错误。为了解决这个问题，我们将窗口大小改为[64, 64]，并相应地更新了 checkpoint 中所有受影响的 scale 权重（缩放张量）。

成果配图。有意思的是GLM5 32节点的速度甚至比Kimi更慢 — 应该是因为Kimi用了fp8，所以总数据量反而更多。从这个表看来我们的解决方案应该是可以继续scale下去的，而且是越稀疏，效果越好。也欢迎大家分享各自的痛点和使用体验，在不久后也会加入B卡的支持。

---

*核心代码：[Miles P2P实现](https://github.com/radixark/miles/blob/main/miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p.py) | [SGLang PR](https://github.com/sgl-project/sglang/pull/21278) | [SGLang Issue](https://github.com/sgl-project/sglang/issues/17311) | [Slime文档](https://github.com/THUDM/slime/blob/main/docs/zh/blogs/release_v0.1.0.md)*
