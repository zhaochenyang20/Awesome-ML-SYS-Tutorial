# NCCL 与 NVIDIA TOPO

## What is NCCL

NCCL (NVIDIA Collective Communications Library) 是 NVIDIA 推出的一个用于 GPU 之间高性能通信的库。随着深度学习模型规模的增长（如 GPT-3 的 1750 亿参数），单个 GPU 已无法满足训练需求。这就需要将模型或数据分割到多个 GPU 上进行并行训练，而 GPU 之间必然需要进行数据交换。NCCL 就是为了解决这个场景而生的。它主要解决以下问题：

- 在多 GPU 训练场景下实现高效的数据交换
- 自动识别并优化 GPU 间的通信拓扑
- 提供标准化的集合通信接口
- 支持机内（单机多卡）和机间（多机多卡）通信

## NCCL 重要概念

- Collective Operations：NCCL 支持各种集体通信操作，如广播（Broadcast）、规约（Reduction）、聚合（Aggregation）、AllReduce、AllGather 等。这些操作可以在多个 GPU 或节点之间进行数据同步和合并。
- Processes and Groups：NCCL 中的进程（Processes）表示参与通信的计算节点。进程可以组织成组（Groups），以便在组内进行集体通信。
- Communicators：Communicators 是 NCCL 中用于定义进程之间通信关系的对象。它指定了参与通信的进程组和通信模式（如点对点、广播等）。
- Devices and Streams：NCCL 与 GPU 设备密切相关，它可以在多个 GPU 设备之间进行通信。同时，NCCL 还支持与 CUDA 流（Streams）的集成，以实现异步通信和并行计算。
- Synchronization：在分布式计算中，同步是至关重要的。NCCL 提供了各种同步原语，如 barrier 同步，以确保进程在执行集体通信操作时达到一致的状态。
- Performance Optimization：NCCL 注重性能优化，它提供了一些技术来提高通信效率，如集体通信的合并、数据传输的批量处理、通信与计算的重叠等。
- Fault Tolerance：NCCL 还考虑了容错性，支持在部分节点故障或网络不稳定的情况下进行可靠的通信。

## NCCL 支持的操作类型

NCCL 支持以下几种主要的集合通信操作：

1. **AllReduce**：所有 GPU 的数据先进行规约（如求和），然后广播到所有 GPU
2. **Broadcast**：从一个源 GPU 向其他所有 GPU 广播数据
3. **Reduce**：将所有 GPU 的数据规约到一个目标 GPU
4. **AllGather**: 收集所有 GPU 的数据并分发给所有 GPU
5. **ReduceScatter**: 规约后将结果分散到所有 GPU
6. **Send/Recv**: 点对点通信
7. **AllToAll**: 将数据分发到所有 GPU

## NCCL 的设备要求

- NVIDIA GPU (支持 CUDA)
- 推荐使用支持 NVLink 的 GPU 以获得最佳性能
- 对于多机通信，建议使用 InfiniBand 或 RoCE 网络
- 需要安装对应版本的 CUDA 和 GPU 驱动

## NCCL in PyTorch

PyTorch 内置 NCCL 后端支持，使用非常简单：

```python
import torch.distributed as dist

# 初始化进程组
dist.init_process_group(backend='nccl')

# 创建分布式模型
model = torch.nn.parallel.DistributedDataParallel(model)
```

## NCCL 的工作方式

**Ring Algorithm**

NCCL 在机内通信时主要使用 Ring Algorithm。其核心思想是：

1. 将所有 GPU 组织成一个环
2. 每个 GPU 只与其相邻的 GPU 通信
3. 通过多轮次传递实现全局数据交换
4. 数据被分成多个 chunk 进行并行传输

优点：
- 充分利用 GPU 间的带宽
- 通信负载均衡
- 易于扩展

## Tree Algorithm

在[torch-distributed 的后记](../torch-distributed/readme.md#ring-all-reduce-and-tree-all-reduce)中已经介绍过了。

## 通信协议

NCCL 实现了三种通信协议：

1. **Simple**: 基础协议
2. **LL(Low Latency)**: 低延迟协议，适用于小数据量
3. **LL128**: 在 NVLink 环境下的优化协议，可达到 93.75% 的有效带宽


## NCCL 与其他通信库的对比

1. **与 MPI 的区别**:
   - NCCL 专注于 GPU 通信优化
   - MPI 更通用但性能可能较低
   - 可以结合使用：MPI 管理进程，NCCL 负责 GPU 通信

2. **与 Gloo 的区别**:
   - Gloo 支持 CPU 和 GPU
   - NCCL 在 GPU 通信性能上更优

# 常见 NIVIDA 指令

这里参考了 [WeLearnNLP 的指南](https://www.yourmetaverse.cn/deep_learning/199/)。

## `nvidia-smi topo -m`

最典型的当然有 `nvidia-smi` 和 `nvidia-smi topo -m`。前者都非常熟悉了，这里我对比下两台集群的 `nvidia-smi topo -m` 的输出：

```bash
	GPU0	GPU1	GPU2	GPU3	GPU4	GPU5	GPU6	GPU7	CPU Affinity	NUMA Affinity	GPU NUMA ID
GPU0	 X 	SYS	SYS	SYS	SYS	SYS	SYS	SYS	0-15,32-47	0		N/A
GPU1	SYS	 X 	SYS	SYS	SYS	SYS	SYS	SYS	0-15,32-47	0		N/A
GPU2	SYS	SYS	 X 	SYS	SYS	SYS	SYS	SYS	0-15,32-47	0		N/A
GPU3	SYS	SYS	SYS	 X 	SYS	SYS	SYS	SYS	0-15,32-47	0		N/A
GPU4	SYS	SYS	SYS	SYS	 X 	SYS	SYS	SYS	16-31,48-63	1		N/A
GPU5	SYS	SYS	SYS	SYS	SYS	 X 	SYS	SYS	16-31,48-63	1		N/A
GPU6	SYS	SYS	SYS	SYS	SYS	SYS	 X 	SYS	16-31,48-63	1		N/A
GPU7	SYS	SYS	SYS	SYS	SYS	SYS	SYS	 X 	16-31,48-63	1		N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```

```bash
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      NV18    NV18    NV18    NV18    NV18    NV18    NV18    0-47,96-143     0               N/A
GPU1    NV18     X      NV18    NV18    NV18    NV18    NV18    NV18    0-47,96-143     0               N/A
GPU2    NV18    NV18     X      NV18    NV18    NV18    NV18    NV18    0-47,96-143     0               N/A
GPU3    NV18    NV18    NV18     X      NV18    NV18    NV18    NV18    0-47,96-143     0               N/A
GPU4    NV18    NV18    NV18    NV18     X      NV18    NV18    NV18    48-95,144-191   1               N/A
GPU5    NV18    NV18    NV18    NV18    NV18     X      NV18    NV18    48-95,144-191   1               N/A
GPU6    NV18    NV18    NV18    NV18    NV18    NV18     X      NV18    48-95,144-191   1               N/A
GPU7    NV18    NV18    NV18    NV18    NV18    NV18    NV18     X      48-95,144-191   1               N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```

可以读出很多有趣的信息：

通过对比这两个集群的拓扑信息，我可以得出以下几个重要结论：

1. **互联方式**

- 第一个集群：所有 GPU 之间通过 PCIe 和 NUMA 节点间的 SMP 互联（标记为 SYS）
- 第二个集群：所有 GPU 之间通过 18 条 NVLink 连接（标记为 NV18）

- **性能影响**：第二个集群的 GPU 间通信性能显著优于第一个集群，因为 NVLink 的带宽和延迟都优于 PCIe+SMP 方案

2. **NUMA 架构**

- 两个集群都采用双 NUMA 节点设计：
  - GPU 0-3 属于 NUMA 节点 0
  - GPU 4-7 属于 NUMA 节点 1

- **GPU 通信**：应尽量将相关任务分配到同一 NUMA 节点内的 GPU，以避免跨 NUMA 节点的频繁数据传输

- **CPU 核心分配**：
  - 第一个集群：每个 NUMA 节点分配 32 个核心（如 0-15,32-47）
  - 第二个集群：每个 NUMA 节点分配 96 个核心（如 0-47,96-143）

3. **系统规模**

- GPU 数量：两个集群都是 8 GPU 配置

- CPU 核心总数：
  - 第一个集群：64 核心
  - 第二个集群：192 核心

4. **拓扑完整性**

- 每个 GPU 都与其他所有 GPU 直接相连

## NVLINK 查询

```bash
nvidia-smi nvlink --status -i 0
nvidia-smi nvlink --capabilities -i 0
```

<details>
<summary>nvlink 查询结果</summary>

```bash
GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-5a10e6e5-95f7-2785-ed63-6f6147f304f7)
         Link 0: 26.562 GB/s
         Link 1: 26.562 GB/s
         Link 2: 26.562 GB/s
         Link 3: 26.562 GB/s
         Link 4: 26.562 GB/s
         Link 5: 26.562 GB/s
         Link 6: 26.562 GB/s
         Link 7: 26.562 GB/s
         Link 8: 26.562 GB/s
         Link 9: 26.562 GB/s
         Link 10: 26.562 GB/s
         Link 11: 26.562 GB/s
         Link 12: 26.562 GB/s
         Link 13: 26.562 GB/s
         Link 14: 26.562 GB/s
         Link 15: 26.562 GB/s
         Link 16: 26.562 GB/s
         Link 17: 26.562 GB/s
GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-5a10e6e5-95f7-2785-ed63-6f6147f304f7)
         Link 0, P2P is supported: true
         Link 0, Access to system memory supported: true
         Link 0, P2P atomics supported: true
         Link 0, System memory atomics supported: true
         Link 0, SLI is supported: true
         Link 0, Link is supported: true
         Link 1, P2P is supported: true
         Link 1, Access to system memory supported: true
         Link 1, P2P atomics supported: true
         Link 1, System memory atomics supported: true
         Link 1, SLI is supported: true
         Link 1, Link is supported: true
         Link 2, P2P is supported: true
         Link 2, Access to system memory supported: true
         Link 2, P2P atomics supported: true
         Link 2, System memory atomics supported: true
         Link 2, SLI is supported: true
         Link 2, Link is supported: true
         Link 3, P2P is supported: true
         Link 3, Access to system memory supported: true
         Link 3, P2P atomics supported: true
         Link 3, System memory atomics supported: true
         Link 3, SLI is supported: true
         Link 3, Link is supported: true
         Link 4, P2P is supported: true
         Link 4, Access to system memory supported: true
         Link 4, P2P atomics supported: true
         Link 4, System memory atomics supported: true
         Link 4, SLI is supported: true
         Link 4, Link is supported: true
         Link 5, P2P is supported: true
         Link 5, Access to system memory supported: true
         Link 5, P2P atomics supported: true
         Link 5, System memory atomics supported: true
         Link 5, SLI is supported: true
         Link 5, Link is supported: true
         Link 6, P2P is supported: true
         Link 6, Access to system memory supported: true
         Link 6, P2P atomics supported: true
         Link 6, System memory atomics supported: true
         Link 6, SLI is supported: true
         Link 6, Link is supported: true
         Link 7, P2P is supported: true
         Link 7, Access to system memory supported: true
         Link 7, P2P atomics supported: true
         Link 7, System memory atomics supported: true
         Link 7, SLI is supported: true
         Link 7, Link is supported: true
         Link 8, P2P is supported: true
         Link 8, Access to system memory supported: true
         Link 8, P2P atomics supported: true
         Link 8, System memory atomics supported: true
         Link 8, SLI is supported: true
         Link 8, Link is supported: true
         Link 9, P2P is supported: true
         Link 9, Access to system memory supported: true
         Link 9, P2P atomics supported: true
         Link 9, System memory atomics supported: true
         Link 9, SLI is supported: true
         Link 9, Link is supported: true
         Link 10, P2P is supported: true
         Link 10, Access to system memory supported: true
         Link 10, P2P atomics supported: true
         Link 10, System memory atomics supported: true
         Link 10, SLI is supported: true
         Link 10, Link is supported: true
         Link 11, P2P is supported: true
         Link 11, Access to system memory supported: true
         Link 11, P2P atomics supported: true
         Link 11, System memory atomics supported: true
         Link 11, SLI is supported: true
         Link 11, Link is supported: true
         Link 12, P2P is supported: true
         Link 12, Access to system memory supported: true
         Link 12, P2P atomics supported: true
         Link 12, System memory atomics supported: true
         Link 12, SLI is supported: true
         Link 12, Link is supported: true
         Link 13, P2P is supported: true
         Link 13, Access to system memory supported: true
         Link 13, P2P atomics supported: true
         Link 13, System memory atomics supported: true
         Link 13, SLI is supported: true
         Link 13, Link is supported: true
         Link 14, P2P is supported: true
         Link 14, Access to system memory supported: true
         Link 14, P2P atomics supported: true
         Link 14, System memory atomics supported: true
         Link 14, SLI is supported: true
         Link 14, Link is supported: true
         Link 15, P2P is supported: true
         Link 15, Access to system memory supported: true
         Link 15, P2P atomics supported: true
         Link 15, System memory atomics supported: true
         Link 15, SLI is supported: true
         Link 15, Link is supported: true
         Link 16, P2P is supported: true
         Link 16, Access to system memory supported: true
         Link 16, P2P atomics supported: true
         Link 16, System memory atomics supported: true
         Link 16, SLI is supported: true
         Link 16, Link is supported: true
         Link 17, P2P is supported: true
         Link 17, Access to system memory supported: true
         Link 17, P2P atomics supported: true
         Link 17, System memory atomics supported: true
         Link 17, SLI is supported: true
         Link 17, Link is supported: true
```

</details>

可以分析看到一些对开发实用的特性：

- P2P（点对点）通信
- 系统内存访问
- P2P原子操作
- 系统内存原子操作
- SLI（多GPU并行）
- 完整的链路支持

## GPU 监控

可以监控 GPU 的方式很多，这里推荐 [nvitop](https://github.com/Syllo/nvtop)，非常方便，pip 安装即可，看着最赏心悦目。
