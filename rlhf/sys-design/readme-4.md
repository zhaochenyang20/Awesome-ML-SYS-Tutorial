# 深入浅出 DeepSeek MoE，EP 与 FSDP 经典二次开发

这段时间在研究如何在 FSDP 上二次开发 TP 和 EP。这个需求听上去非常神奇，事实上 FSDP 本身只支持 DeepSpeed Zero 类型的 DP，而其他任何并行方式，TP PP EP 都没有官方实现，需要二次开发。就我所知，一些大厂内部有无数前赴后继的工程师，基于 HuggingFace Transformers 和 FSDP 进行了大量的二次开发，希望能够弥补原生 Transformers 和 FSDP 在 MOE 模型上的性能短板。感谢各位大佬的不懈努力，颇有一种为开源社区 xxx 的美感 😂 以上内容就是本文的写作背景，这段时间 SGLang RL 小组的朋友们一起浅浅研究了 EP，并且分析了一些经典开源框架对 FSDP 的二次开发。

【感谢 Xinyi Song 和 Zhuoran Yin 对本文的贡献，为了山姆奥特曼的海边大别野，还是再苦苦员工吧】

## DeepSeek MoE

在 Dense 模型中，每一层的所有参数都会参与每个 Token 的计算。而 MoE 架构将原本巨大的全连接层 FFN 拆分为多个规模较小且结构相同的独立单元，即专家 Experts。MoE 进一步引入稀疏激活（Sparse Activation）机制：对于输入的每一个 Token，只有一小部分专家（如 Top-k）会被选中参与计算。这使得模型可以在保持计算量（FLOPs）基本不变的情况下，通过增加专家数量极大地扩张参数总量，某种意义上兼备了更高的模型能力的上线和更低的计算开销。在这些基本的介绍之外，强烈推荐读者去学习 [DeepSeek MOE](https://arxiv.org/abs/2401.06066) 原文，其中进一步引入了一些新颖的 MOE 优化：

1. 在任意 forward 过程中，每层会有少量的 shared experts，永远被激活。某种意义上，这些 shared experts 存储着常识。
2. 相较于在 DeepSeek MOE 之前的“传统 MOE”，将专家拆的更细，例如之前一层 FFN 拆成 8 个专家，现在拆成 64 个专家。（在 DeepSeek V3 中甚至是 256 个 experts）

这篇文章还有个让我印象深刻的事情，就是如何在 MOE 和 dense 模型之间做公平的比较。这是个很有意思的事情，我们当然不能拿着 LLama 3.1 405B 和 DeepSeek V3 做比较，说 DeepSeek 碾压 Llama，所以 MOE 比起 Dense 强，因为这不 make sense，二者的变量差距远远不止 MOE/Dense 这一项。想要严格控制变量，比较 MOE 和 Dense 二者架构的优劣，必须得从 pretrain 时候就开始，做非常严格的控制，类似朱泽圆老师的《Physics of Language Model》，为了得到最严谨的科学结论，需要非常严格地控制变量。非常遗憾，在学术界，很难有这种从 pretrain 开始控制变量的机会，这也算是我一般不会认同各种学界对 Diffusion LLM，Dense Model，MOE，Hybird Model 孰优孰劣的争论。

回到 DeepSeek MOE 这篇文章本身，我非常喜欢这篇文章，他们为了比较各种架构，也从 pretrain 的 tokens 数量开始做了控制变量。然后提出，对于总参数量为 X 而激活参数为 Y 的 MOE 模型，其模型表现能够比起总参数为 Y 的 Dense 模型高，而计算开销比起总参数为 X 的 Dense 模型低，甚至模型表现有接近并超越总参数为 X 的 Dense 模型的可能性。注意到，这篇文章发布于 2024 年的年初，到现在已经过去两年了，这几乎是这个 MOE 统治时代的开端之作。

当然，如果 MOE 模型没有训好的话，一些 experts 在推理的过程中一直不激活，就会出现下图这种尴尬的情况：

<div style="text-align: center;">
<img src="./pics/MOE.png" alt="MoE Model" width="500">
</div>

对于这种情况，甚至可以选择剪枝掉那些一直没有用过的 experts，总参数量下降了，反而能力没有降低，又可以发一篇 ACL 了。

## Expert Parallelism

让我们回顾朴素的 MoE forward 流程：

1. Gate（路由）计算：输入 Token 经过一个 Gate 网络，计算该 Token 与各个专家的相关性得分。
2. 专家选择：Gate 根据得分选择出需要参与计算的专家。
3. Token 分发：Token 被发送至选中的专家。
4. 专家并行计算：选中的专家各自独立完成矩阵乘法运算。
5. 结果合并：将各专家的输出按 Gate 的权重进行加权求和，传入下一层。


在没有 EP 的情况下，每张 GPU 都必须存储该层所有专家的完整权重。对于参数量动辄上千亿的 MoE 模型（如数百个专家），单卡显存完全无法承载。因此，EP 的核心逻辑是将 Experts 集合在第 0 维度（专家维度）进行拆分，让不同的 Rank 维护不同的专家子集。由于专家被物理隔离在不同的显卡上，原本 gate 所做的逻辑分发变成了真实跨越 GPU rank 的物理分发：

1. Dispatch (All-to-All)

在训练阶段，虽然 Transformer 是 auto regressive 的，但 Causal Mask 实现了全序列的并行 Forward。此时，每个 Rank 都同时持有大量的 Token，且每个 token 都持有不同专家的 gate 分数。于是，在每个 Rank 独立运行 Gate 算法，计算本地 Token 所需的目标专家及其所在的 Rank。由于每个 Rank 都有 Token 需要发往其他 Rank，同时也要接收来自其他所有 Rank 发来的 Token，这构成了典型的 All-to-All 通信，完成了分布式转置（Distributed Transpose）过程，将数据分布从“按序列位置对齐”重组为“按 experts 索引对齐”。

2. Expert Compute

各 Rank 并行执行本地持有的专家计算（FFN）。不同 Rank 之间的专家计算是完全独立的，无需通信同步。但是，此时的计算效率极大地取决于路由分布。如果大量 Token 涌向同一个专家（Hotspot Expert），会导致严重的负载不均衡（Load Imbalance），计算慢的 Rank 会拖累整个集群的同步速度。

3. Combine (All-to-All)

计算完成后，专家输出的局部结果（Expert Latents）需要再次通过 All-to-All 通信进行回传每个专家 Rank 将计算结果发回给该 Token 原始出发的 Rank。这确保了数据的物理分布恢复到进入 MoE 层之前的状态，以便进行后续的加权求和（Combine）、残差连接以及下一层 Attention 的并行计算。

注意到，虽然在 inference 的 Decoding 阶段，每次仅处理一个 Token，但在 Training 和 Pre-filling 阶段，高吞吐的并行计算使得 All-to-All 成为最有效、最常用的通信抽象，因此我们一般认为 EP 需要的是两次 All-to-All 通信。

## EP vs TP

为了降低单个 rank 上的显存负载，TP 也是常见的方案。为什么有了 TP 还需要 EP？对 MOE 而言，二者有什么区别？至少这个问题困扰了我一段时间。现在想来，虽然 TP 也可以切分专家权重（将每个专家的参数分到不同的 rank 上）。

### 通讯量

我们试图先从通讯的角度来分析 TP 和 EP 的优劣。我们定义以下变量：

- $N$：并行组内的 GPU 数量（TP 组或 EP 组大小）。
- $B \times L$：总 Token 数量（Batch Size $\times$ Sequence Length）。
- $H$：Hidden Size（每个 Token 的向量维度）。
$k$：MoE 的 Top-$k$ 激活数（每个 Token 选择的专家数）。
- $S = B \times L \times H$：该层输入数据的总激活量（Activation Size）。


我们基于主流的 Ring All-Reduce 和 Standard Exchange All-to-All 计算每个 GPU 发送的数据量：

TP 将每个 expert 的 FFN 矩阵切分，每经过一个 expert 层，需要在 $W_{down}$ 之后做一次 All-Reduce。在 Ring 算法下，单卡通讯量为 $2 \times \frac{N-1}{N} \times S$，即 $\text{Comm}_{TP} \approx 2S$。注意，TP 的通讯量与 $k$（激活 expert 数）完全无关，哪怕你只激活 1 个 expert，TP 也要雷打不动地同步全量激活值。

EP 在专家维度进行显式拆分，在 Dispatch 阶段，每个 Rank 最初持有的数据量仅为 $S/N$。由于每个 Token 需要分发给 $k$ 个专家，单卡发出的数据量平均为 $k \times (S/N)$。算上 Combine 阶段的回传，单卡通讯量为 $\text{Comm}_{EP} = 2 \times \frac{N-1}{N} \times \frac{kS}{N} \approx \frac{2k}{N}S$。

在大规模并行时，$N$ 很大，如 $N=64$ 或 $256$，只要 $k < N$（DeepSeek每层激活 $k=8$，而专家总数为 $256$），EP 的单卡通讯字节数实际上远小于 TP。

虽然 EP 的通讯量看上去更少，但是 EP 遇到的通讯瓶颈更严重：

1. TP 通常死磕在单机 8 卡的 NVLink 域内，带宽起步 900GB/s；而 EP 往往要跨越节点走 RDMA，带宽通常只有 50GB/s ~ 100GB/s。带宽差了一个数量级。
2. All-to-All 本质上是 $N^2$ 个小连接。在大规模集群下，握手开销、长尾延迟以及负载不均衡（Load Imbalance）带来的等待时间，可能比实际通讯耗时还要大。

### 计算效率

虽然以 DeepSeek MoE 举例，$k=16$，EP 和 TP 在通讯上各有问题，但 EP 在算子效率和**硬件利用率（MFU）**上具有压倒性优势。

TP 的核心逻辑是将矩阵“横着切”或“竖着切”。在 MoE 场景下，单个专家的参数量通常较小。如果使用 TP，每个专家原本就不大的 $H \times \text{Hidden\_Size}$ 矩阵会被进一步切分成 $1/N$。这导致在 GPU 上执行的是极其“瘦长”的矩阵乘法（GEMM）。对于 NVIDIA Tensor Cores 而言，过小的维度无法充分填充计算流水线，实际算力利用率大幅下降。这其实也是 TP 一般不做跨机的一大原因：机器之间的通讯远比机内更慢，这对于 TP 几乎恒定的通讯量来说是个灾难；以及，TP 切到更多机器上，让每个 rank 的形状更加瘦长，GEMM 效率也会大幅下降。

而 EP 能保持 expert 矩阵的完整性。尽管 Token 需要跨卡搬运，但一旦到达目标 GPU，它面对的是一个形状规整、足以触发高效计算内核的完整矩阵。对于 DeepSeek 这种**细粒度专家（Fine-grained Experts）**设计，每个专家极其微小，如果再套用 TP，计算效率将退化到难以忍受的地步。

此外，对于 DeepSeek MoE 而言，其选择 EP 还有更多的 infra 创新，首先是 shared experts 被隔离，不需要参与 EP 通信；其次是 DeepEP 为 large $k$ 带来的极致通讯隐藏：

1. DeepSeek 实现了计算与通讯的极致重叠（Stream-K）。当 Dispatch 的第一批 Token 到达 GPU 时，计算内核立即启动，而不是等待 16 个专家的所有数据全部到齐。

2. RDMA 直驱：DeepEP 绕过了传统的 NCCL 协议栈，利用 PTX 级别优化实现低延迟的跨节点数据交换。在 $k=16$ 产生的巨大吞吐下，DeepEP 依然能维持极高的带宽利用率，使得“通讯时间”几乎完全被“计算时间”掩盖。

基于此，对于主流的 MoE 模型而言，我们如下对比 TP 和 EP：

| 维度 | TP 方案 | EP 方案 (DeepSeek 为例) |
| --- | --- | --- |
| 通讯量 (Bytes) | 固定 (≈2S) | 随 k/N 缩放 (≈2kS/N) |
| 通讯延迟 (Latency) | 极高（高频 All-Reduce 同步锁） | 可控（粗粒度 All-to-All 异步掩盖） |
| 计算效率 (MFU) | 低（矩阵切分导致算子不饱满） | 高（算子完整，易于硬件加速） |
| 集群扩展性 | 局限于单机 NVLink 域 | 支持万卡集群 RDMA 扩展 |

当然，事后想想，其实 MoE 会考虑 EP 优于 TP，还是因为 DeepSeek MoE 引导的这波“小而多”的 MoE 设计。因为单个 expert 小，倘若 TP 切的更细，则每个 rank 上的 GEMM 效率会暴跌；而以往 GShard 那种“大而少”的 MoE 设计，单个 expert 大，TP 切分后，每个 rank 上的 GEMM 效率还是可以保证，所以那个时代对 MoE 采用 TP 才是主流。说到底，EP 也是被算法驱动的新并行策略啊。


### ETP

讨论了这么多 EP 和 TP 的区别，从中我们可以看出，对于 MoE 模型而言，EP 确实是更佳的选择。不过，如果你有留意过 SGLang 启动 DeepSeek R1 的指令的话，会有些诧异；比如说，在 2026 年 1 月 1 日的 SGLang cookbook 中，我们可以同时打开 DeepSeek R1 的 EP 和 TP，得到如下的启动指令：

```bash
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-R1-0528 \
  --tp 8 \
  --ep 8 \
  --enable-symm-mem # Optional: improves performance, but may be unstable
```

这就让人有些费解，TP 和 EP 是如何同时开启的。

实际上，从我们上述的讨论中，可以发觉，EP 将不同的 experts 物理分到不同的 rank 后，这些 experts 仍旧可以做 TP。比如说，EP 2 TP4，先把所有的专家分成两组，比如 0～3 卡负担前 1/2 的专家。接着，单个专家还会拆分到 4 个组内的 rank 上。这种策略确实存在，但是有个术语叫做 ETP，先做 EP，再对每个 expert 做 TP。

很遗憾，这种策略并没有在开源社区被广泛采用，目前 SGLang 的上述指令中，TP 8 和 EP 8 是指 experts 会被分到 8 个 rank 上；而对非 MOE 的部分，比如 linear 层，则会按照 TP 分到 8 个 rank 上，执行的并不是 ETP 模式。

⚠️ 最后，给 SGLang cookbook 打个小小的广告，以往的 [SGLang 文档](https://docs.sglang.io/)是按照 feature 纵向编写的，比如说分析 EP TP DPA 等等各种并行策略，但是很难系统性知道一个给定的 LLM，究竟要横向组合哪些并行策略，才能达到最佳配置。现在我们有了 [SGLang cookbook](https://cookbook.sglang.io/docs/intro)，在模型的维度进行编写，丰富并且详细展开了主流模型的各种配置组合。

## 经典的 FSDP 二次开发：EP

回到本文痛苦的出发点，SGLang RL 小组在一段时间内多次讨论是否要在 slime 已经支持的 FSDP 基础上，进一步支持 EP。如我所说的：

> 这个需求听上去非常神奇，事实上 FSDP 本身只支持 DeepSpeed Zero 类型的 DP，而其他任何并行方式，TP PP EP 都没有官方实现，需要二次开发。就我所知，一些大厂内部有无数前赴后继的工程师，基于 HuggingFace Transformers 和 FSDP 进行了大量的二次开发，希望能够弥补原生 Transformers 和 FSDP 在 MOE 模型上的性能短板。感谢各位大佬的不懈努力，颇有一种为开源社区 xxx 的美感

这里只分享我们的调研流程，我们研究学习了社区其他开源项目基于 FSDP 的 EP 实现，并在此分享。

我们先来看看有了 EP 后的 forward 和 backward 流程：

| forward | backward |
| --- | --- |
| gate<br><br>All-to-All Dispatch<br><br>expert compute (FSDP2)<br><br>all gather<br><br>Expert FFN compute<br><br><br><br>release<br><br>All-to-All Return<br><br>merge | gate<br><br>All-to-All Combine<br><br>expert compute (FSDP2)<br><br>all gather<br><br>Expert FFN compute<br><br>reduce-scatter<br><br>release<br><br>All-to-All Return<br><br>merge |

其实没有什么显著区别，就是加入了 EP 的 all-2-all 通讯。注意到，Backward 中的 Reduce-Scatter 是 FSDP 的标准动作，和 EP 无关。将计算出的完整梯度需要在 DP 组内聚合（Reduce）并重新切分（Scatter）回各个 Rank，在数学上完成了梯度的平均（Reduce）和分发（Scatter）。在 MoE EP 场景下，只有在专家进一步被 FSDP 切分时，才会有对应的 FSDP 级别 Reduce-Scatter。而 experts 本身是不需要在 EP 组内做梯度聚合的，各自优化各自的梯度即可。

遍览各大框架，读下来大家做出的优化一般有：

1. EP 切分 dim 0 experts，而 FSDP 切分 dim 1 hidden size （见 VeOmni）

2. prefetch：在计算第 n 层的时候，预先把第 n+1 层的参数 gather 起来。只用 FSDP 的话可以很直白的做 prefetch。但是因为 EP 计算开始之前有通信，需要手动做一些操作来保证前向和反向的 prefetch （见 VeOmni）

3. DeepEP：苦 NCCL 久矣，使用 RDMA 直驱 （见 Automodel）

4. EPLB：通过专家冗余来解决专家计算负载不均衡

5. fused MoE：其实和 EP 关系不大，单个 GPU 负责多个专家，用 Fused MoE kernel 来加速这些专家的计算

背后的技术有一系列可以参考的优秀文章：

[Deepseek 的 All-to-all 通信: DeepEP 代码解读](https://www.cnblogs.com/CQzhangyu/p/18741625)

[一点浅见：DeepEP 为什么快？](https://zhuanlan.zhihu.com/p/28867733102)

[DeepSeek AI Infra(3) - DeepEP的原理与代码剖析](https://zhuanlan.zhihu.com/p/27777601573)

[MoE 并行负载均衡：EPLB 的深度解析与可视化](https://zhuanlan.zhihu.com/p/29963005584)


## 实现对比

我们这里对比三个社区的高光项目：[VeOmni](https://github.com/ByteDance-Seed/VeOmni), [TorchTitan](https://github.com/pytorch/torchtitan), [Automodel](https://github.com/NVIDIA-NeMo/Automodel)，它们都是先 EP，然后对 EP 完的每个块做 FSDP。


### VeOmni

我们学习时的代码结构如下：

```text
VeOmni/veomni/
├── distributed/
│   ├── parallel_state.py       ← 全局并行状态（ep_fsdp_device_mesh, ep_size）
│   ├── parallel_plan.py        ← EP 切分计划（ParallelPlan.apply()）
│   ├── torch_parallelize.py    ← EP + FSDP 整合入口
│   │   ├── parallelize_model_fsdp2()  ← 主入口
│   │   └── 手动 prefetch 配置
│   ├── fsdp/
│   │   ├── clip_grad_norm.py   ← FSDP1 EP 感知梯度裁剪
│   │   └── extension.py        ← Checkpoint 扩展
│   └── fsdp2/
│       └── clip_grad_norm.py   ← FSDP2 EP 感知梯度裁剪
├── models/
│   └── transformers/
│       └── qwen3_moe/
│           └── parallel_plan.py  ← 模型特定的 EP 参数定义
└── sequence_parallel/
    ├── async_ulysses.py        ← 异步序列并行（与 EP 无关）
    └── ulysses.py              ← 标准 Ulysses
```


整个逻辑看下来还是比较清晰的，就是一个 EP + FSDP 的结构，对于专家先 apply EP，在第 0 个维度（expert）上切分，然后再 FSDP，非专家部分直接按照常规 FSDP 即可。这里有个比较有意思的点，注意到 FSDP 采用的 `fully_shard` 是隐式切分（有人叫做动态逻辑切分），希望对上层是一种无感的并行优化。以 `shape=[128, H, I]` 的一组专家为例，在模型的 forward 开始前，FSDP 内部会偷偷发起一次 all-gather，临时把 8 张卡上的碎片拼回完整的 `[128, H, I]`。而对于上层而言，看到的还是一个完整的 Tensor，不需要关心分布式通信。而 EP 是一个显式切分，或者说静态物理切分，原本 `shape=[128, H, I]` 的一组专家，在每个 Rank 上物理上变成了 `[32, H, I]`。模型代码必须感知到这个变化，比如 MoE 层代码一定要知道，“我这台机器上只有 32 个专家”，并根据这个数量去计算。


```text
Applies EP (when enabled) + FSDP2 parallel strategy to the model.

Flow:
1. Apply EP: Expert tensors [128,H,I] -> [32,H,I] local tensors per EP rank
2. Apply FSDP2 to expert modules: Shard expert tensors along dim-1 (hidden dim)
3. Apply FSDP2 to regular modules: Standard dim-0 sharding
4. Result: Expert params [32, H/fsdp_size, I], regular params use standard FSDP2
```

下面是这一实现的关键函数 `parallelize_model_fsdp2` 节选，[完整代码](https://github.com/ByteDance-Seed/VeOmni/blob/3bd8e6e48c2d741b2b8b4898f90645145bf4287b/veomni/distributed/torch_parallelize.py#L228)：


```python

def parallelize_model_fsdp2(model, enable_mixed_precision=True, basic_modules=None, **kwargs):
    # 【1】专家 128 -> 32 (EP)
    if parallel_state.ep_enabled:
        parallel_plan = model.get_parallel_plan()
        parallel_plan.apply(model, parallel_state.ep_fsdp_device_mesh)
        experts_map = parallel_plan.get_fsdp_no_shard_info(model)

    # 【2. 循环分片】由内而外切分每一层
    layer_pairs = []
    for layer_fqn, layer_mod in decoder_blocks:
        experts_mod = next((exp_mod for exp_fqn, exp_mod in experts_map.items() if ...), None)
        layer_mod._fsdp_modules = []

        if experts_mod:
            fully_shard(experts_mod, **expert_fsdp_kwargs) # 切专家
            layer_mod._fsdp_modules.append(experts_mod)
        
        fully_shard(layer_mod, **fsdp_kwargs) # 切整层
        layer_mod._fsdp_modules.append(layer_mod)
        layer_pairs.append(layer_mod)

    # 【3. 切root model】
    fully_shard(model, **fsdp_kwargs)

    # 【4. 配置prefetch】
    # 正向
    for cur, nxt in zip(layer_pairs, layer_pairs[1:] + [None]):
        if nxt:
            cur.set_modules_to_forward_prefetch(list(reversed(nxt._fsdp_modules)))

    # 反向
    rev_blocks = list(reversed(layer_pairs))
    for cur, prev in zip(rev_blocks, rev_blocks[1:] + [None]):
        if prev:
            cur.set_modules_to_backward_prefetch(list(reversed(prev._fsdp_modules)))

    return model
```

讨论完切分逻辑，我们接着考虑通讯逻辑。All 2 All 通讯的复杂度不低，相关通讯可见[代码](https://github.com/ByteDance-Seed/VeOmni/blob/e4e431d0/veomni/distributed/moe/moe_layer.py)： 

1. Preprocess：在传输重数据之前，通过 all_gather 交换元数据，计算出 Input Splits 和 Output Splits；

2. Dispatch：根据路由索引在本地进行 Permute，利用 dist.all_to_all 完成传输，收到数据后需再次 Sort；

3. Combine：计算完成后，执行逆向的通信和 Unpermute 操作，将 Token 还原回原始序列顺序；

```python

def preprocess(expert_mask, num_experts, ep_group):
    # expert_mask: [Batch, Tokens, Num_Experts] (哪些 token 去哪些专家)
    
    # 1. 算出本地要发给每个 rank 的 token 数量 (Input Splits)
    ep_size = ep_group.size()
    num_local_tokens_per_expert = expert_mask.sum(dim=(1, 2)) 
    input_splits = num_local_tokens_per_expert.reshape(ep_size, -1).sum(dim=1).tolist()

    # 2. dist.all_gather: 收集所有卡上的 num_local_tokens_per_expert
    num_global_tokens_per_expert = torch.zeros(...)
    dist.all_gather_into_tensor(num_global_tokens_per_expert, num_local_tokens_per_expert, group=ep_group)

    # 3. 算出本地将从每个 rank 接收多少 token (Output Splits)
    rank = dist.get_rank(ep_group)
    my_experts_range = slice(rank * num_local_experts, (rank + 1) * num_local_experts)
    tokens_sent_to_me = num_global_tokens_per_expert[:, my_experts_range]
    output_splits = tokens_sent_to_me.sum(dim=1).tolist()

    return input_splits, output_splits, tokens_sent_to_me

def token_pre_all2all(hidden_states, expert_mask, input_splits, output_splits, ...):
    # 1. 本地重排 (Permute)
    # local_permuted: [Token1_to_Exp1, Token2_to_Exp1, ..., TokenN_to_Exp99]
    local_permuted, _ = permute(hidden_states, expert_mask.sum(dim=1))

    # 2. All-to-All
    # 发送：input_splits, 接收：output_splits
    global_permuted = all_to_all(ep_group, local_permuted, output_splits, input_splits)

    # 3. Sort by Expert
    global_permuted = sort_chunks_by_idxs(global_permuted, ...)

    return global_permuted # 准备好喂给 Group GEMM 了

def tokens_post_all2all(expert_outputs, input_splits, output_splits, ...):
    # 1. 算完的数据是按 Expert 排列的，要发回去得按来源 Rank 重排
    expert_outputs = sort_chunks_by_idxs(expert_outputs, ...)

    # 2. All-to-All Return
    unpermute_outputs = all_to_all(ep_group, expert_outputs, input_splits, output_splits)

    # 3. Unpermute
    final_output = unpermute(unpermute_outputs, ...)

    return final_output
```

### Automodel

我们学习时的代码结构如下，比较逆天的是，这个 repo 目前仍旧只有 200 多 star...

```text
Automodel/nemo_automodel/
├── components/
│   ├── distributed/
│   │   └── fsdp2.py            ← FSDP2Manager（moe_mesh 定义）
│   └── moe/
│       ├── parallelizer.py     ← EP + FSDP 整合入口
│       │   ├── ExpertParallel      ← EP 类定义
│       │   ├── apply_ep()          ← EP 切分
│       │   ├── apply_fsdp()        ← FSDP 切分
│       │   └── parallelize_model() ← 主入口
│       ├── layers.py           ← MoE 层实现
│       ├── fsdp_mixin.py       ← MoE FSDP 同步 Mixin（PP 相关）
│       └── megatron/
│           ├── token_dispatcher.py  ← Token 调度（_DeepepManager）
│           ├── fused_a2a.py         ← DeepEP 封装（FusedDispatch/Combine）
│           └── moe_utils.py         ← permute/unpermute 工具
```

Automodel 对 DeepEP 的使用可圈可点。Automodel 通过 `_DeepepManager` 集成了 DeepEP，利用 Fused Dispatch/Combine 算子替代了 NCCL All-to-All：`token_dispatcher.py -> MoEFlexTokenDispatcher -> _DeepepManager -> fused_dispatch`

**DeepepManager from token_dispatcher.py**

有状态的通信上下文管理器，它封装了 DeepEP 库与上层模型逻辑之间的交互。在 dispatch 阶段，DeepEP 底层返回一个 handle 对象，包含通信布局信息。在 combine 阶段，直接取出 self.handle 传给底层。

```python
# token_dispatcher.py 第 90-191 行

class _DeepepManager(_DispatchManager):
    """DeepEP backend for token dispatch/combine"""

    def __init__(self, group, router_topk, num_experts, num_local_experts, ...):
        self.group = group
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts  # 本 EP 组的 expert 数量

        if fused_dispatch is None:
            raise ImportError("DeepEP is not installed.")

    def setup_metadata(self, num_local_tokens, probs):
        """处理 routing map"""
        probs = probs.reshape(num_local_tokens, self.num_experts)
        self.token_probs, self.token_indices = torch.topk(probs, self.router_topk, dim=-1)

    def dispatch(self, hidden_states, async_finish=False, allocate_on_comm_stream=False):
        """Dispatch tokens to experts"""
        # DeepEP 要求 float32
        self.token_probs = self.token_probs.float()

        # 调用 DeepEP 的 fused_dispatch
        (hidden_states, dispatched_indices, dispatched_probs,
         num_tokens_per_expert, handle) = fused_dispatch(
            hidden_states,
            self.token_indices,
            self.token_probs,
            self.num_experts,
            self.group,
            async_finish=async_finish,
        )
        self.handle = handle  # 保存用于 combine
        return hidden_states

    def combine(self, hidden_states, async_finish=False, allocate_on_comm_stream=False):
        """Combine expert outputs"""
        hidden_states, _ = fused_combine(
            hidden_states,
            self.group,
            self.handle,  # 使用 dispatch 时保存的 handle
            async_finish=async_finish,
        )
        self.handle = None
        return hidden_states
```

**FusedDispatch from fused_a2a.py**

```python
# fused_a2a.py 第 80-148 行

class FusedDispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, token_indices, token_probs, num_experts, group, async_finish, ...):
        # 获取 DeepEP Buffer
        buffer = get_buffer(group, get_hidden_bytes(x))

        # 计算 dispatch layout
        (num_tokens_per_rank, num_tokens_per_rdma_rank,
         num_tokens_per_expert, is_token_in_rank, event) = buffer.get_dispatch_layout(
            token_indices, num_experts, ...
        )

        # 调用 DeepEP 核心 dispatch
        (recv_x, recv_token_indices, recv_token_probs,
         num_recv_tokens_per_expert_list, handle, after_event) = buffer.dispatch(
            x,
            topk_idx=token_indices,
            topk_weights=token_probs,  # 必须 float32
            num_tokens_per_rank=num_tokens_per_rank,
            ...
            async_finish=async_finish,
        )

        # 异步同步
        if async_finish:
            after_event.current_stream_wait()

        ctx.handle = handle  # 保存用于 backward
        return recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle

    @staticmethod
    def backward(ctx, ...):
        # backward 调用 combine
        grad_x, grad_token_probs, after_event = buffer.combine(
            grad_output.contiguous(),
            ctx.handle,
            ...
        )
        return grad_x, ...
```


### TorchTitan

```text
torchtitan/
├── distributed/
│   ├── expert_parallel.py      ← 核心！EP 类定义
│   ├── parallel_dims.py        ← Device Mesh 管理
│   └── deepep.py               ← DeepEP 封装（可选）
├── models/
│   ├── moe/
│   │   ├── moe.py              ← MoE 层实现
│   │   └── moe_deepep.py       ← DeepEP MoE 变体
│   └── llama4/infra/
│       └── parallelize.py      ← EP + FSDP 整合入口
```

完整逻辑可见[此处](https://github.com/pytorch/torchtitan/blob/7e4ab85998576c68902603058adada28fb0ed226/torchtitan/models/llama4/infra/parallelize.py#L494)。TorchTitan 实现了全链路的 pre-fetch：

1. MoE 感知预取：大多框架可能只预取下一层 Block，但 TorchTitan 在前向传播时，会显式地同时预取下一层 Block 及其内部的 Experts（[next_transformer_block, next_transformer_block.moe.experts]）；
2. 最大限度计算覆盖：从最开始的 Embedding 层到最后的 Output 层，甚至在反向传播（Backward）过程中，都有对应的 set_modules_to_backward_prefetch 逻辑。这种密不透风的预取最大限度地让计算掩盖通信。

**Parallelize from parallelize.py**

```python
for layer_id, transformer_block in model.layers.items():
    if transformer_block.moe_enabled and ep_degree > 1:
        fsdp_mod_ep_config = fsdp_config.copy()
        fsdp_mod_ep_config["mesh"] = edp_mesh
        _experts_shard_placement_fn = None
        assert edp_mesh is not None
        assert hasattr(transformer_block, "moe")
        if (
            edp_mesh["efsdp"].size() * ep_degree
            > transformer_block.moe.experts.num_experts
        ):
            _experts_shard_placement_fn = lambda param: Shard(1)

        fully_shard(
            transformer_block.moe.experts,
            **fsdp_mod_ep_config,
            reshard_after_forward=reshard_after_forward,
            shard_placement_fn=_experts_shard_placement_fn,
        )
   
        transformer_block.moe.experts.set_gradient_divide_factor(
            gradient_divide_factor,
        )

    fully_shard(
        transformer_block,
        **fsdp_config,
        reshard_after_forward=reshard_after_forward,
    )
```

**Prefetch from parallelize.py**

```python
transformer_blocks = list(model.layers.values())
next_transformer_blocks = transformer_blocks[1:] + [None]

# pyrefly: ignore [bad-argument-type]
if model.tok_embeddings is not None and len(model.layers) > 0:
    # pyrefly: ignore [missing-attribute]
    model.tok_embeddings.set_modules_to_forward_prefetch([transformer_blocks[0]])

for transformer_block, next_transformer_block in zip(
    transformer_blocks, next_transformer_blocks
):
    if next_transformer_block is not None:
        # pyrefly: ignore [missing-attribute]
        if next_transformer_block.moe_enabled:
            # pyrefly: ignore [missing-attribute]
            transformer_block.set_modules_to_forward_prefetch(
                # pyrefly: ignore [missing-attribute]
                [next_transformer_block, next_transformer_block.moe.experts]
            )
        else:
            # pyrefly: ignore [missing-attribute]
            transformer_block.set_modules_to_forward_prefetch(
                [next_transformer_block]
            )
    elif model.norm is not None and model.output is not None:
        # pyrefly: ignore [missing-attribute]
        transformer_block.set_modules_to_forward_prefetch(
            [model.norm, model.output]
        )

# backward
# pyrefly: ignore [not-callable]
reversed_transformer_blocks = list(reversed(model.layers.values()))
prev_transformer_blocks = reversed_transformer_blocks[1:] + [None]

# pyrefly: ignore [bad-argument-type]
if model.norm is not None and model.output is not None and len(model.layers) > 0:
    # pyrefly: ignore [missing-attribute]
    model.output.set_modules_to_backward_prefetch([reversed_transformer_blocks[0]])

for transformer_block, prev_transformer_block in zip(
    reversed_transformer_blocks, prev_transformer_blocks
):
    if prev_transformer_block is not None:
        # pyrefly: ignore [missing-attribute]
        if prev_transformer_block.moe_enabled:
            # pyrefly: ignore [missing-attribute]
            transformer_block.set_modules_to_backward_prefetch(
                # pyrefly: ignore [missing-attribute]
                [prev_transformer_block, prev_transformer_block.moe.experts]
            )
        else:
            # pyrefly: ignore [missing-attribute]
            transformer_block.set_modules_to_backward_prefetch(
                [prev_transformer_block]
            )
    elif model.tok_embeddings is not None:
        # pyrefly: ignore [missing-attribute]
        transformer_block.set_modules_to_backward_prefetch([model.tok_embeddings])
```
