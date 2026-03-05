# 一文速览 RL 场景下基于 RDMA 方案的权重传输设计

## 从 NCCL 到 RDMA

NCCL 是面向用户的集合通信库，而 RDMA 是 nccl 底层的传输协议之一。当你在 InfiniBand 集群上通过 NCCL 执行 `torch.distributed.broadcast(tensor, src=0)` 时，实际字节传输是在底层通过 RDMA verbs 完成的。**RDMA**（Remote Direct Memory Access，远程直接内存访问）允许一台机器在不经过对端 CPU 或操作系统内核参与数据路径的情况下，直接读写另一台机器的内存。

| 关注点 | TCP/Socket | NCCL（基于 RDMA） | 原生 RDMA（Mooncake） |
|--------|-----------|-------------------|----------------------|
| 每次传输的 CPU 开销 | 高（内核缓冲区间 memcpy） | 低（NCCL 内部处理） | 接近零（NIC 执行 DMA） |
| 内核旁路 | 无 | 部分（NCCL 管理） | 完全旁路 |
| 集合通信语义 | 手动实现 | 内建（broadcast、allreduce） | 仅点对点 |
| 细粒度控制 | 高 | 低（不透明的 group） | 高 |
| 建立复杂度 | 低 | 中等（process group） | 高（MR、QP、握手） |

内核旁路（Kernel Bypass）是指数据传输时是否绕过操作系统内核。在传统的 TCP/Socket 通信中，数据发送的路径是：`用户空间 → 系统调用 → 内核协议栈 → 内核缓冲区 → 网卡驱动 → NIC`，每次传输都要经过内核态/用户态切换、内核缓冲区拷贝，开销很大。**内核旁路**的意思是让应用程序直接和网卡（NIC）通信，跳过内核协议栈：`用户空间 → NIC（通过 RDMA/verbs API 直接操作）`。完全内核旁路是 RDMA 高性能的核心原因之一，因为省掉了内核态切换和内存拷贝的开销。

Miles 默认的权重传输方案是基于 NCCL 的 `broadcast`，既然 NCCL 已经在使用 RDMA 了，为什么还要考虑使用原生 RDMA？

1. **点对点零拷贝**：NCCL 的 `broadcast` 要求所有 rank 参与集合操作。使用原生 RDMA，训练端可以直接写入某个推理引擎的 GPU 显存，无需协调整个 group——没有 barrier，没有 collective。
2. **解耦生命周期**：NCCL process group 必须在通信前创建、通信后销毁。在 RL 场景中，训练和推理引擎可能独立启停（弹性伸缩、故障恢复）。基于 Transfer Engine 的方案避免了 process group 生命周期的紧耦合。

对于第二点，具体来说，在 disaggregate training 的情况下， 如果用 nccl 做通讯，要 scale up training 和 Inference 的任何一端（比如加入一个新的 Inference node），都需要先 destroy 已有的 process group，然后建立包含新加入的 inference node 的 new group。直接用原生 RDMA 方案是不需要这些 group level 的操作的。因为 NCCL 的 process group 是静态的——创建时就固定了参与者集合（world size + ranks）。要加一个新节点，必须先销毁已有的 process group，用新的 world size 重新初始化 process group，所有节点重新做 rendezvous/handshake。

这对 disaggregated training 很不友好，因为你 scale inference 端不应该影响 training 端的运行，但 NCCL 的语义做不到这一点。RDMA 是点对点（point-to-point）的连接模型。每条连接是一个独立的 Queue Pair（QP），节点之间的关系类似于：

```
Node A <--QP--> Node B
Node A <--QP--> Node C
Node B <--QP--> Node C  （新建立，不影响已有连接）
```

加入新的 Infernence 节点时，只需要在新节点和需要通信的已有节点之间建立新的 QP；交换必要的连接信息（GID、QPN、Memory Region key 等）。已有节点之间的连接**完全不受影响**，training 可以继续跑，不需要任何 group 级别的操作。当然，如果你要扩大 training 端的规模，倘若 training 还是使用了 nccl 做通讯，training 内部的 process group 重建还是必须的。只是，不需要考虑在 inference 和 training 之间的重建；或者说 training 和 inference 之间如果用原生的 RDMA 方案，就从未建立过任何 group。

Mooncake 这类系统选择原生 RDMA，用作 disaggregated / elastic 场景简直浑然天成—，它的连接模式自然就适合动态伸缩。

### RDMA 编程模型速览

RDMA 编程使用 **Verbs API**（libibverbs）。一些关键的 API 如下：

1. **打开设备** — 确定使用哪块网卡
2. **创建 QP**（Queue Pair）— 发送队列 + 接收队列，类比 socket
3. **注册内存** — 将缓冲区 pin 住，使 NIC 可以进行 DMA 读写
4. **交换元数据** — 告知对端：我的 QP 编号和内存地址
5. **发起 RDMA Write** — NIC 自主将字节写入远端地址
6. **轮询完成** — 检查传输是否结束

内存注册是其中相当复杂的环节：

- RDMA 访问任何缓冲区之前，该缓冲区必须先向 NIC 驱动注册。
- 注册会 pin 住虚拟地址到物理地址的映射，使 NIC 的 DMA 引擎能够完成地址转换。
- 注册开销很大（内核调用、页面锁定）。逐参数注册很慢；对一整块连续内存一次性注册则很快，需要在 SGLang 中通过 `register_memory_region_v2()` 在注册前合并连续权重块。

### InfiniBand vs RoCE vs TCP

前文讨论的 NCCL 与原生 RDMA 是软件/API 层，而 InfiniBand、RoCE、TCP 是物理传输层/网络层，两者是正交的维度。RDMA 是一种能力（远程直接内存访问），InfiniBand 和 RoCE 是实现这种能力的两种不同网络技术，TCP 则不具备这种能力。NCCL 的底层传输可以走 InfiniBand、RoCE 或 TCP，它会自动检测网络硬件并优先选最快的方式。原生 RDMA（Mooncake）则只能跑在支持 RDMA 的网络上（InfiniBand 或 RoCE），纯 TCP 网络无法使用。

| | InfiniBand (IB) | RoCE v2 | TCP |
|---|---|---|---|
| 网络 | 专用 IB 网络 | 标准以太网 | 标准以太网 |
| RDMA 支持 | 原生支持 | 支持（基于 UDP/以太网） | 不支持（无法内核旁路） |
| 延迟 | ~1μs | ~2-5μs | ~50-100μs |
| 典型带宽 | 200-400Gbps（NDR/XDR） | 100-400Gbps | 25-100Gbps |
| 拥塞控制 | 基于信用的无损机制 | 基于 ECN（需配置 PFC/ECN） | TCP 拥塞控制 |
| 部署场景 | HPC/ML 集群 | 云厂商（Azure、GCP） | 通用 |

对于 SGLang 的应用场景，**IB** 是主要目标。Mooncake TransferEngine 默认的 RDMA 走的就是 IB verbs。

### GPU-Direct RDMA (GDR)

标准 RDMA 在主机（CPU）内存缓冲区之间传输数据。GPU-Direct RDMA 将其扩展到 GPU 显存：

- 无 GDR：GPU → PCIe → CPU 内存 → NIC → 网络 → NIC → CPU 内存 → PCIe → GPU
- 有 GDR：GPU → PCIe → NIC → 网络 → NIC → PCIe → GPU

GDR 消除了两次 CPU 内存拷贝。Mooncake 和 NCCL 正是通过 GDR 实现 GPU 到 GPU 的 RDMA 传输。使用条件：

- 支持 GDR 的 NVIDIA GPU（所有现代数据中心 GPU 均支持）
- 支持 GPUDirect 的 Mellanox/NVIDIA 网卡
- 已加载 `nv_peer_mem` 或 `nvidia-peermem` 内核模块

### CUDA IPC 与 GDR 的关系

CUDA IPC（Inter-Process Communication）解决的问题是：**同一台机器上，不同进程想访问同一块 GPU 显存**。不同进程的 CUDA 虚拟地址空间是隔离的，进程 B 拿到进程 A 的 `data_ptr()` 地址是无意义的。CUDA IPC 允许一个进程将 GPU 内存导出为 handle，另一个进程通过 handle 映射到自己的地址空间：

```
进程 A：cudaIpcGetMemHandle(gpu_buffer) → handle
         ↓ （通过共享内存/socket 传递 handle）
进程 B：cudaIpcOpenMemHandle(handle) → 本地可用的 gpu_ptr
```

CUDA IPC **不涉及** disaggregated training 的权重传输路径。在 disaggregated 场景中，训练节点和推理节点是不同的机器，每个推理端 GPU worker 独立接收属于自己分片的权重（由 Engine Replica 按 TP/PP/EP 分好），各写各的 GPU 显存，不存在同节点跨进程共享 GPU 显存的需求。

CUDA IPC 涉及的是另一个场景——**SGLang 实例间的远程权重加载**（seed instance → new instance）。如果两个 SGLang 实例恰好部署在同一台机器的不同进程上，需要通过 RDMA 引擎共享 GPU 内存时，CUDA IPC 才会被用到。

### NCCL 如何使用 RDMA

NCCL 将上述所有细节抽象掉。当你创建 NCCL process group 时：

1. NCCL 发现网络拓扑（NVLink、PCIe、IB）。
2. 为每对 rank 选择最优传输方式（节点内用 NVLink，节点间用 IB）。
3. 对于 IB 传输，NCCL 内部创建 QP、注册内存、发起 RDMA 操作。
4. `NCCL_SOCKET_IFNAME`、`NCCL_IB_HCA`、`NCCL_IB_GID_INDEX` 控制 NCCL 使用哪些接口。

**Miles 和 SGLang 配置中常见的环境变量**：

| 变量 | 用途 |
|------|------|
| `NCCL_SOCKET_IFNAME` | NCCL TCP 引导使用的网络接口（如 `eth0`、`bond0`） |
| `NCCL_IB_HCA` | 使用哪块 IB HCA（Host Channel Adapter，如 `mlx5_0,mlx5_1`） |
| `NCCL_IB_GID_INDEX` | RoCE 的 GID 索引（原生 IB 无关） |
| `NCCL_CUMEM_ENABLE` | 启用 CUDA 内存管理集成（在 Miles `actor_group.py` 中设置） |
| `NCCL_DEBUG` | 设为 `INFO` 或 `TRACE` 以调试传输选择 |

### 第一部分自查清单

- [ ] 能解释为什么在 NCCL 已经使用 RDMA 的情况下，仍然需要原生 RDMA
- [ ] 理解内存注册：为什么存在、为什么开销大、为什么需要批量注册
- [ ] 了解 IB 和 RoCE 的高层区别
- [ ] 理解 GPU-Direct RDMA 数据路径与非 GDR 路径的差异
- [ ] 能识别与 IB 传输相关的 NCCL 环境变量


## 第二部分：SGLang 代码中的 RDMA

### 三种权重更新策略回顾

这些你已经了解，这里从传输层角度重新梳理：

| 策略 | 传输层 | 适用场景 | 关键文件 |
|------|--------|---------|---------|
| `update_weights_from_disk` | 文件系统 I/O | 弹性伸缩、Checkpointing | `model_runner.py:1126` |
| `update_weights_from_tensor` | 共享内存 / CUDA IPC | 训练与推理共置 | `model_runner.py:1438` |
| `update_weights_from_distributed` | NCCL broadcast（底层走 RDMA） | 训练与推理分离部署 | `model_runner.py:1348` |

这三条路径的调用方都是 **RL framework（Miles）**，在训练完成后主动向 SGLang 推理端发起权重更新请求（`Miles 训练完成 → 调用 SGLang HTTP API → SGLang 执行 update_weights_from_xxx()`），区别在于数据如何到达推理端：

| 路径 | Miles 如何调用 | 数据如何到达推理端 |
|------|--------------|------------------|
| `update_weights_from_disk` | HTTP API 传文件路径 | SGLang 自己从磁盘读取 |
| `update_weights_from_tensor` | Ray 远程调用传 tensor | 内存内传输（co-located） |
| `update_weights_from_distributed` | HTTP API 触发 | Miles 和 SGLang 共建 NCCL group，broadcast 传输 |

**RDMA Transfer Engine 新增了第四条路径**：通过 Mooncake 直接进行 CPU/GPU 到 GPU 的写入，完全绕过 NCCL collective。这正是 Issue #17311 所构建的。同样由 Miles 侧发起，但数据传输方式变为 Engine Replica 通过 `MooncakeTransferEngine` 直接 RDMA 写入 SGLang 已注册的 GPU 显存地址，不再需要建立 NCCL process group。

### SGLang 中的 Mooncake TransferEngine

现有 RDMA 基础设施位于：

```
python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py
```

`MooncakeTransferEngine` 封装了 `mooncake.engine.TransferEngine`，提供以下接口：

| 方法 | 功能 |
|------|------|
| `initialize(hostname, device_name)` | 打开 IB 设备，启动 RPC 服务用于对端发现 |
| `register(ptr, length)` | 注册单个 GPU 内存区域用于 RDMA |
| `batch_register(ptrs, lengths)` | 单次内核调用注册多个区域（更快） |
| `transfer_sync(session_id, buffer, peer_addr, length)` | RDMA Write：将 `length` 字节从本地 `buffer` 拷贝到远端 `session_id` 的 `peer_addr` |
| `batch_transfer_sync(session_id, buffers, peer_addrs, lengths)` | 批量 RDMA Write |
| `get_session_id()` | 返回 `hostname:rpc_port`——远端 peer 用于寻址本引擎的标识符 |

**初始化流程**（`mooncake_transfer_engine.py:191-200`）：

```python
ret_value = self.engine.initialize(
    hostname,
    "P2PHANDSHAKE",    # 发现模式：点对点
    "rdma",             # 传输方式：基于 IB verbs 的 RDMA
    device_name,        # 如 "mlx5_0"
)
```

`"P2PHANDSHAKE"` 模式意味着 peer 之间直接发现（无需中央协调器），通过轻量级 RPC 通道交换 QP 元数据。

### 内存注册：为什么需要 v2

在 `remote_instance_weight_loader_utils.py` 中有两种注册策略：

**v1**（`register_memory_region_v1`）：逐参数注册。

```python
for name, weight in model.named_parameters():
    transfer_engine.register_memory(weight.data_ptr(), weight.numel() * weight.element_size())
```

问题：对于一个拥有约 1000 个参数的 70B 模型，这意味着约 1000 次内核调用来完成内存注册。

**v2**（`register_memory_region_v2`）：遍历 `torch.cuda.memory.memory_snapshot()` 找到存放权重的连续物理块，然后注册合并后的块。

```python
# 遍历内存段，合并相邻的权重块
for segment in memory_snapshot:
    # ... 合并持有权重的相邻 active_allocated 块
weight_blocks_for_reg_mr.append((merged_address, merged_size))

# 注册合并后的块（内核调用次数大幅减少）
for weight_block in weight_blocks_for_reg_mr:
    transfer_engine.register_memory(address, size)
```

任何改变内存布局的 PR（如权重 offloading、量化）都可能破坏 v2 的合并逻辑：合并依赖于权重位于连续的 CUDA allocator 块中。

### Weight Info 字典

注册完成后，SGLang 构建一个 `weight_mr_dict`，将参数名映射到其 GPU 内存位置：

```python
weight_mr_dict[name] = (
    weight.data_ptr(),    # GPU 虚拟地址
    weight.numel(),       # 元素数量
    weight.element_size() # 每个元素的字节数
)
```

该字典通过 HTTP API 的 `/get_remote_instance_transfer_engine_info` 端点暴露，使远端 peer（训练端）可以精确查找每个参数的 RDMA 写入目标地址。

### 远程实例权重加载流程

> **注意**：本节讨论的是 **SGLang 实例间的远程权重加载**（seed instance → new instance），与 RL 场景下的训练端→推理端权重更新是不同的用途。这里的场景是：已有一个运行中的 SGLang 推理实例（种子实例），想快速启动一个新的 SGLang 实例——与其让新实例从磁盘重新加载权重（慢），不如直接从种子实例的 GPU 显存通过 RDMA 拷贝（快）。该流程是一次性的（仅在启动时），且不涉及参数名映射（同一模型，相同命名）。SGLang 中已有的 RDMA 基础设施（`MooncakeTransferEngine`、内存注册、`weight_mr_dict`）最初就是为此场景构建的，Issue #17311 的工作是复用这套基础设施，将其扩展到 RL 场景下的训练端→推理端权重更新。

将以上内容串联起来，现有的 RDMA 权重加载流程（从种子实例加载权重到新实例）如下：

**种子实例侧**：(1) 启动并加载模型，(2) 初始化 `MooncakeTransferEngine`，(3) 注册所有权重内存区域，(4) 通过 HTTP 暴露 `weight_mr_dict`。

**新实例侧**：(5) 启动后，查询种子实例的 HTTP API 获取权重信息（session_id + 地址），(6) 初始化自己的 `MooncakeTransferEngine`，(7) 分配匹配的 GPU 缓冲区，(8) 对每个参数，从种子实例的注册地址 RDMA 读取到本地缓冲区——NIC 自主完成传输，(9) 将缓冲区中的权重加载到模型中。

涉及的关键 HTTP 端点：

- `GET /get_remote_instance_transfer_engine_info?rank=N` — 返回 rank N 的 session_id 和 weight_mr_dict
- `POST /init_weights_send_group_for_remote_instance` — 用于基于 NCCL 的远程加载
- `POST /send_weights_to_remote_instance` — 触发权重发送

### Issue #17311 新增的内容：训练到推理的 RDMA 路径

现有 RDMA 路径是**实例到实例**（一个 SGLang server 到另一个）。Issue #17311 将其扩展到**训练端到推理引擎**：

**训练侧**（Miles/Megatron）：(1) 完成一个训练步骤。(2) 创建一个与 SGLang 相同并行配置的"Engine Replica"——通过 **PR #16860** 暴露并行信息来实现。(3) 将 Megatron 参数名映射到 SGLang 分片名——通过 **PR #17326**（统一权重映射）实现。(4) 对每个参数：AllGather TP/EP 分片，转换为 HF 格式，对参数做 bucket，然后 RDMA 写入 SGLang 引擎的已注册内存地址。

**SGLang 推理侧**：(5) 引擎的权重位置可查询，包括跨节点场景——由 **PR #17389** 修复 `nnode > 1` 的情况。(6) 权重直接出现在 GPU 显存中，零拷贝，无需 NCCL group，甚至无需调用推理侧 API 接口。

### 第二部分自查清单

- [ ] 能追踪完整的数据路径：训练步骤 → 参数 gather → RDMA write → SGLang GPU 显存
- [ ] 理解 `register_memory_region_v2` 为什么合并块以及什么会导致它出错
- [ ] 知道 `weight_mr_dict` 包含什么内容以及如何通过 HTTP 暴露
- [ ] 能解释现有的实例到实例 RDMA 路径与新的训练到推理路径的区别
- [ ] 理解 MooncakeTransferEngine 的 `session_id` 在寻址中的作用


## 第三部分：Miles ↔ SGLang 集成与 PR Review

### Miles 当前的权重传输架构

Miles 目前支持两种权重传输策略（均未直接使用原生 RDMA）：

**策略 A：`UpdateWeightFromTensor`**（共置部署，CUDA IPC + Gloo + Ray）

```
1. 各 rank 在 GPU 上聚合得到完整 tensor（通过 NCCL broadcast/all_gather）
2. 对 GPU tensor 做 MultiprocessingSerializer.serialize 得到 CUDA IPC handle tuple（不拷贝数据，只序列化显存句柄）
3. 通过 dist.gather_object（走 Gloo，因为是 Python 对象）把 handle tuple 聚合到 source rank
4. Source rank 通过 Ray 远程调用把 handle tuple 传给 SGLang
5. SGLang 侧反序列化 handle tuple，通过 CUDA IPC 直接映射到同一块 GPU 显存
```

这里 Gloo 和 CUDA IPC 不是同一层级的概念。**CUDA IPC** 是实际的数据共享机制——通过序列化 GPU 显存的 handle tuple，SGLang 侧反序列化后直接映射到训练端的 GPU 显存，不发生数据拷贝。**Gloo** 只是用来传递轻量的 handle tuple（Python 对象），因为 `dist.gather_object` 需要 CPU 端的通信后端。**Ray** 负责跨进程的远程调用，将 handle tuple 从 source rank 传递给 SGLang Engine。

- 适用于训练和推理在同一节点的场景（CUDA IPC 要求同节点共享 GPU 显存）
- 文件：`miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py`

**策略 B：`UpdateWeightFromDistributed`**（分离部署，NCCL broadcast）

```
训练 Ranks → AllGather TP 分片 → 转换为 HF 格式 → NCCL broadcast → SGLang
```

- 创建临时 NCCL process group（`miles-pp_{rank}`）
- 从训练 rank 0 向所有 SGLang 引擎 GPU broadcast 权重
- 对 MoE 专家参数单独进行 AllGather EP 处理
- 文件：`miles/backends/megatron_utils/update_weight/update_weight_from_distributed.py`

**RDMA 的定位**：策略 B 已经隐式受益于 RDMA（NCCL 在可用时使用 IB）。拟议的 RDMA Transfer Engine 路径将**替代** NCCL broadcast，改为直接的 P2P RDMA 写入，消除以下开销：

- 创建/管理临时 NCCL process group 的需要
- collective 同步开销（broadcast 要求所有参与者就绪）
- 训练与推理进程之间的紧耦合生命周期

### Engine Replica 架构（Issue #17311）

Issue #17311 描述的架构在训练侧引入了一个 **Engine Replica**：

要实现 P2P RDMA 直接写入，训练端必须知道推理端每个 rank 持有哪些参数、参数如何分片，这样才能精准地把正确的数据写入正确的地址。为此，训练端在 **CPU 上**构造一个与 SGLang 推理端并行策略完全一致的 Engine Replica（相同的 TP/PP/EP）。之所以放在 CPU 而不是 GPU 上，是因为训练端的 GPU 显存用于训练本身，额外放一个 replica 会占用宝贵的显存资源。

Engine Replica 在 CPU 上完成参数的名称映射、分片和 bucketing 后，直接从训练端的 CPU 内存通过 RDMA 写入推理端对应 rank 的 GPU 显存。因为两侧的并行策略一致，每个 Engine Replica rank 与推理端 rank 一一对应，写入就是点对点的。推理端不需要做任何 gather/shard/映射的工作，也不需要建立任何 NCCL process group。

**Engine Replica 方案的三个层面**：

**层面一：Engine Replica 本身——并行策略镜像**

Engine Replica 在训练端 CPU 上构造一个与推理端并行策略完全一致的镜像。理论上，也可以不构造 replica，而是让每个 Megatron rank 各自计算偏移、分别 RDMA 写入 SGLang rank 的不同区域。但这会显著增加实现复杂度——每个 Megatron rank 都需要理解 SGLang 的并行策略，多对一写入需要 fence/barrier 协调，TP/PP/EP 三维并行交叉切分的排列组合非常多。Engine Replica 的设计选择是先在训练端内部做一次 AllGather（走 NVLink/IB，很快），然后由一个统一的实体集中处理所有映射逻辑，用一次便宜的 AllGather 换来架构上的简洁性。

**层面二：参数名映射（PR #17326）**

Megatron 和 SGLang 对同一组权重的命名和组织方式不同。最典型的例子是 `qkv_proj`：Megatron 中 `q_proj`、`k_proj`、`v_proj` 是三个独立参数，SGLang 中它们被堆叠成一个 `qkv_proj`。Engine Replica 需要一张翻译表，知道每个 Megatron 参数对应 SGLang 的哪个参数、写入哪个偏移位置。这就是 PR #17326 提供的 `load_weight` 接口：`(sglang_param, num_shards, shard_idx)`。

**层面三：Partial Tensor P2P——传输粒度优化**

"Partial tensor" 是相对于 `update_weights_from_distributed`（NCCL broadcast）方案而言的。在 NCCL broadcast 方案中，训练端将**完整参数**广播给所有 SGLang rank，每个 rank 收到完整参数后再自己 shard 取出需要的部分——带宽被浪费了。而在 Engine Replica 方案中，replica 已经按 SGLang 的并行策略做好了分片，因此 RDMA 写入到每个 SGLang rank 的 tensor 是该 rank 精确需要的那一份，不多不少。相比于 broadcast 完整参数，这份分片就是 "partial tensor"。

```
NCCL broadcast 方案：
  完整参数 [8192, 8192] → broadcast 给所有 SGLang rank → 每个 rank 自己 shard → 取 [2048, 8192]
  （每个 rank 收到完整参数，只用 1/4，带宽浪费 3/4）

RDMA P2P 方案：
  Engine Replica 提前切好 → 直接 RDMA 写入 [2048, 8192] 给对应 rank
  （每个 rank 只收到自己需要的 partial tensor，零浪费）
```

三个层面不冲突，分别解决不同的问题：Engine Replica 提供架构简洁性，参数名映射解决命名翻译，partial tensor P2P 实现精准传输。

**Engine Replica 的职责总结**：

1. **镜像 SGLang 的并行配置** — 知道参数如何分片（TP、PP、EP），与推理端 rank 一一对应
2. **映射参数名** — 将 Megatron 名称转换为 SGLang/HuggingFace 名称
3. **对参数做 Bucket** — 将小参数分组为更大的传输单元
4. **驱动 RDMA 写入** — 从 CPU 内存通过 `MooncakeTransferEngine.transfer_sync()` 直接写入 SGLang 已注册的 GPU 显存，每个 rank 只收到精确需要的 partial tensor

### 三个 PR 的内容与审查要点

**PR #16860：暴露并行配置信息**

**目的**：让 Engine Replica 知道 SGLang 每个 rank 的精确并行配置（TP/PP/EP/attn_tp/attn_dp）。

**新增内容**：HTTP 端点 `/get_parallelism_config`，返回每个 rank 的并行配置。

**已有的审查讨论**（由 slin1237 提出）：这应该是新端点还是现有 server info 的一部分？JD-ETH 认为这是 rank 级信息（非 server 级），因为 P2P 传输需要按 rank 查询。这与同样是按 rank 查询的 `/get_remote_instance_transfer_engine_info` 一致。

**审查要点**：

- 并行配置是否完整到足以让 Engine Replica 重建分片形状？
- 在 DP > 1 时是否正确工作？（该 PR 还修复了 `dp_parallel_controller` 传播 `scheduler_infos` 的问题）
- 安全性：通过 HTTP 暴露并行配置是否可接受？（仅限内部网络）

**PR #17326：统一权重映射接口**

**目的**：暴露参数名称映射层，使 Engine Replica 能够将 Megatron 参数转换为 SGLang 模型参数。

**为什么需要这个 PR**：Engine Replica 解决了并行策略对齐的问题（通过 AllGather + 按 SGLang 的 TP/PP/EP 重新切分），但还有另一个维度的不对齐——**参数名和组织方式**。Megatron 和 SGLang 对同一组权重的命名和"切法"不同。最典型的例子是 `qkv_proj`：

- **Megatron 侧**：`q_proj`、`k_proj`、`v_proj` 是三个独立的参数
- **SGLang 侧**：它们被堆叠（stack）成一个 `qkv_proj` 参数

当 Engine Replica 拿到 Megatron 的 `q_proj` 时，它不能直接写入推理端——因为推理端根本没有一个叫 `q_proj` 的参数，只有 `qkv_proj`。这个 `q_proj` 只是 `qkv_proj` 的 1/3，是一个 "partial tensor"。Engine Replica 需要知道 `q_proj` 应该写入 `qkv_proj` 的第 0 个分片位置，`k_proj` 写入第 1 个，`v_proj` 写入第 2 个，三个 partial tensor 都到齐后 `qkv_proj` 才算完整。

总结来说，Engine Replica 的能力 = 并行策略镜像（PR #16860）+ 参数名映射（PR #17326）+ bucketing + RDMA 写入驱动。本 PR 解决的就是参数名映射这一环。

**新增内容**：一个 ParamsMapper 能读取模型中导出的 `stacked_params_mapping` 和 `expert_params_mapping`，以及用于预处理转换和特殊映射的 hook。

**已支持的模型**：Llama3、Qwen2、Qwen3、Qwen3-MoE、GLM4、GLM4-MOE、DeepseekV2

**核心契约**：

```python
sglang_param, num_shards, shard_idx = model.load_weight(megatron_weights)
```

这告诉 Engine Replica："这个 Megatron 参数映射到这个 SGLang 参数，它是 `num_shards` 个分片中的第 `shard_idx` 个分片。"Engine Replica 据此可以算出写入目标地址的偏移量，并追踪一个参数的所有分片是否已全部到达，从而确定完整参数已就绪。

**为什么不直接从 Megatron 各 rank 分别写入**：理论上可以让多个 Megatron rank 各自 RDMA 写入 SGLang rank 的不同偏移位置（比如 Megatron TP=8 的 rank 0 和 rank 1 分别写入 SGLang TP=4 的 rank 0 的前半和后半），省掉 AllGather。但这会显著增加实现复杂度——每个 Megatron rank 都需要理解 SGLang 的并行策略来计算偏移，多对一写入需要 fence/barrier 协调，TP/PP/EP 三维并行交叉切分加上 stacked params 的排列组合非常多。Engine Replica 的设计选择是先在训练端内部做一次 AllGather（走 NVLink/IB，很快），然后由一个统一的实体集中处理所有映射逻辑，用一次便宜的 AllGather 换来架构上的简洁性。

**审查要点**：

- 是否正确处理了所有 stacked params？（如 `qkv_proj` 由 `q_proj`、`k_proj`、`v_proj` 堆叠而成）
- MoE 专家映射：是否处理了 Expert Parallel 分片？
- 是否有 Miles 需要但未在支持列表中的模型？（对照 Miles 的模型支持列表检查）

**PR #17389：跨节点 Scheduler 信息同步**

**目的**：修复多节点部署下 `/get_remote_instance_transfer_engine_info` 的问题。

**问题所在**：在多节点 SGLang 中，每个节点只知道自己本地的 scheduler。远端训练器查询 `rank=N` 时，如果 rank N 在另一个节点上就会失败。

**解决方案**：本地 scheduler 初始化完成后，使用 Gloo `all_gather` 在所有节点间同步 `scheduler_infos`。这在专用端口（`dist_port + 10000`）上完成，发生在 HTTP server 启动之前。

具体流程是：两个节点各自完成本地 scheduler 初始化后，通过 Gloo `all_gather` 在所有节点间同步 `scheduler_infos`（使用专用端口 `dist_port + 10000`）。只有同步完成后，每个节点才启动 HTTP server——因此每个节点都知道所有 rank 的 Transfer Engine 信息，而不仅仅是本地的。

**审查要点**：

- 端口冲突：`dist_port + 10000`——是否可配置？是否可能冲突？
- Gloo 超时：如果某个节点启动较慢会怎样？
- 是否正确处理了 DP Attention 的情况？

### 替代方案：P2P Transfer Manager（PR #14170）

Risc-lt 提供了一个独立的替代实现：

| 方面 | Issue #17311 (JD-ETH) | PR #14170 (Risc-lt) |
|------|----------------------|---------------------|
| 架构 | 训练侧的 Engine Replica | 带 engine pool 的 P2PTransferManager |
| Transfer Engine | 每个 rank 一个 MooncakeTransferEngine | Engine 池（CPU 核数的 25%，最多 8 个） |
| 协调方式 | Engine Replica 处理映射 | ZMQ 握手 + 任务队列 |
| 范围 | 模块化 PR（3 个 PR + demo） | 单个大型 PR |
| 集成方式 | 为 Miles 设计 | 通用（任何训练框架） |

两种方案底层都使用 Mooncake 的 RDMA 引擎。Engine Replica 方案（Issue #17311）与 Miles 架构更契合，因为映射和 bucketing 逻辑在训练侧完成，使 SGLang 的代码保持精简。

### 三个 PR 的关键审查问题

审查时需要关注以下跨 PR 的共性问题：

1. **容错性**：如果传输中途失败会怎样？有回滚机制吗？系统能否不重启全部组件就恢复？

2. **版本一致性**：Miles 使用 `weight_version` 整数跟踪版本。RDMA 写入完成后，SGLang 如何知道新版本已完全提交？是否有 fence/barrier？

3. **内存安全**：RDMA 直接写入 GPU 显存。如果 SGLang 正在推理时权重被写入，会导致 torn read（撕裂读取）。`pause_generation` → `update` → `continue_generation` 流程必须万无一失。

4. **量化**：Miles 在权重传输后处理 int4/fp4/mxfp8 后处理。RDMA 路径是否保留了量化模型所需的正确 dtype 和 shape？

5. **合并冲突**：三个 PR 都存在需要合并的冲突状态。值得标记——它们需要 rebase。

### 深入阅读推荐

如果 3 小时后还有时间，以下是高价值的延伸阅读：

| 资源 | 为什么值得读 | 时间 |
|------|-------------|------|
| [Awesome-ML-SYS-Tutorial: 权重更新机制](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/sys-design/readme-1-EN.md) | 你自己的教程——带着 RDMA 上下文重新审视 | 20min |
| [Mooncake TransferEngine 文档](https://kvcache-ai.github.io/Mooncake/getting_started/build.html) | 理解 SGLang 所封装的底层库 | 30min |
| [JD-ETH 的 RDMA 集成分支](https://github.com/JD-ETH/slime/tree/jd/rdma-integration) | 可运行的原型——看 Engine Replica 的实际运作 | 45min |
| SGLang `docs/advanced_features/sglang_for_rl.md` | 复习完整的 RL 集成面 | 15min |
| PR #14170 diff | 对比替代的 P2P 方案 | 30min |

### 第三部分自查清单

- [ ] 能凭记忆画出 Engine Replica 架构
- [ ] 理解三个 PR 各自对整体 RDMA 流程的贡献
- [ ] 知道 Engine Replica 方案与 P2P Transfer Manager 方案的区别
- [ ] 能识别关键审查风险：容错性、版本一致性、内存安全、量化
- [ ] 有后续深入阅读的清单


## 快速参考：SGLang 代码中的关键文件

| 文件 | 内容 |
|------|------|
| `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py` | `MooncakeTransferEngine` — RDMA 封装层 |
| `python/sglang/srt/model_loader/remote_instance_weight_loader_utils.py` | 内存注册（v1/v2）、Weight Info 字典、远程加载触发 |
| `python/sglang/srt/model_executor/model_runner.py:1290-1500` | `init_weights_update_group`、`update_weights_from_distributed`、`update_weights_from_tensor` |
| `python/sglang/srt/weight_sync/tensor_bucket.py` | `FlattenedTensorBucket` — 参数 bucketing 用于高效传输 |
| `python/sglang/srt/managers/io_struct.py` | 权重更新 API 的所有请求/响应类型 |
| `python/sglang/srt/managers/scheduler_update_weights_mixin.py` | Scheduler 侧权重更新、内存释放/恢复 |
| `python/sglang/srt/entrypoints/http_server.py` | 所有权重更新 API 的 HTTP 端点定义 |
| `docs/advanced_features/sglang_for_rl.md` | 官方 RL 集成指南 |

## 快速参考：Miles 中的关键文件

| 文件 | 内容 |
|------|------|
| `miles/backends/megatron_utils/update_weight/update_weight_from_distributed.py` | 策略 B：NCCL broadcast 权重同步 |
| `miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py` | 策略 A：Gloo + Ray 权重同步 |
| `miles/backends/megatron_utils/update_weight/common.py` | `all_gather_param()`、`all_gather_params_async()`、名称归一化 |
| `miles/backends/sglang_utils/sglang_engine.py` | SGLang 引擎生命周期管理、权重接收端点 |
| `miles/backends/fsdp_utils/update_weight_utils.py` | FSDP 后端权重同步（两种策略） |
| `miles/ray/rollout.py` | `RolloutManager` — SGLang 引擎初始化、生成、容错 |
| `miles/ray/actor_group.py` | `RayTrainGroup` — 分布式训练编排 |


## 术语表

| 术语 | 定义 |
|------|------|
| **HCA** | Host Channel Adapter — InfiniBand 网卡 |
| **IB** | InfiniBand — 高性能互连网络 |
| **MR** | Memory Region — 已注册的可供 RDMA 访问的缓冲区 |
| **QP** | Queue Pair — RDMA 中等价于 socket 的概念（发送队列 + 接收队列） |
| **GDR** | GPU-Direct RDMA — 直接在 GPU 显存上进行 RDMA，旁路 CPU |
| **RoCE** | RDMA over Converged Ethernet — 基于标准以太网的 RDMA |
| **Verbs** | 底层 RDMA 编程 API（libibverbs） |
| **RNIC** | 支持 RDMA 的网卡 |
| **PD** | Protection Domain — 将 QP 和 MR 分组用于访问控制 |
| **CQ** | Completion Queue — 已完成的 RDMA 操作在此上报 |
| **WR** | Work Request — 提交给 QP 的 RDMA 操作 |
| **Engine Replica** | 训练侧的轻量级模型镜像，知道 SGLang 的并行配置并驱动 RDMA 写入 |
| **Transfer Engine** | Mooncake 的 RDMA 抽象层，在 SGLang 中使用 |
| **FlattenedTensorBucket** | SGLang 将多个参数展平到单个连续缓冲区的机制，用于高效传输 |
