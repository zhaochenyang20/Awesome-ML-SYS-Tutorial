# NCCL and NVIDIA TOPO
## What is NCCL
NCCL (NVIDIA Collective Communications Library) is a high-performance communication library developed by NVIDIA for GPU-to-GPU interactions. As deep learning models grow in size (e.g., GPT-3 with 175 billion parameters), single GPU training becomes impractical. This necessitates splitting models or data across multiple GPUs for parallel training, which requires efficient data exchange between GPUs. NCCL addresses this need by:
- Enabling efficient data exchange in multi-GPU training
- Automatically identifying and optimizing communication topology between GPUs
- Providing standardized collective communication interfaces
- Supporting both intra-node (single-machine multi-GPU) and inter-node (multi-machine multi-GPU) communication

## Key Concepts of NCCL
- **Collective Operations**: Supports operations like Broadcast, Reduction, Aggregation, AllReduce, AllGather, etc., for data synchronization across GPUs/nodes.
- **Processes and Groups**: Processes represent computational nodes participating in communication, organized into groups for collective operations.
- **Communicators**: Objects defining communication relationships between processes, specifying participant groups and modes (e.g., peer-to-peer, broadcast).
- **Devices and Streams**: Works with CUDA streams for asynchronous communication and parallel computation across GPUs.
- **Synchronization**: Provides synchronization primitives (e.g., barriers) to ensure consistent states during collective operations.
- **Performance Optimization**: Techniques like merged collective operations, batched data transfers, and computation-communication overlap.
- **Fault Tolerance**: Reliable communication despite node failures or network instability.

## Supported Collective Operations
1. **AllReduce**: Aggregate data across GPUs (e.g., sum) and broadcast results to all GPUs.
2. **Broadcast**: Transmit data from a source GPU to all others.
3. **Reduce**: Aggregate data to a single target GPU.
4. **AllGather**: Gather data from all GPUs and distribute to all GPUs.
5. **ReduceScatter**: Reduce data and scatter results across GPUs.
6. **Send/Recv**: Peer-to-peer communication.
7. **AllToAll**: Distribute data to all GPUs.

## Hardware Requirements
- NVIDIA GPUs with CUDA support
- NVLink-enabled GPUs recommended for best performance
- InfiniBand or RoCE networks for multi-node communication
- Compatible CUDA and GPU driver versions

## NCCL in PyTorch
PyTorch natively supports NCCL:
```python
import torch.distributed as dist

# Initialize process group
dist.init_process_group(backend='nccl')

# Wrap model
model = torch.nn.parallel.DistributedDataParallel(model)
```

## How NCCL Works
**Ring Algorithm**  
Used for intra-node communication:
1. Organize GPUs into a ring.
2. Each GPU communicates only with adjacent neighbors.
3. Global data exchange via multi-step transfers.
4. Data split into chunks for parallel transfer.

**Advantages**:
- Maximizes GPU-to-GPU bandwidth
- Balanced communication load
- Scalable design

**Tree Algorithm**  
(See [postscript in torch-distributed notes](../torch-distributed/readme.md#ring-all-reduce-and-tree-all-reduce).）

## Communication Protocols
NCCL implements three protocols:
1. **Simple**: Basic protocol.
2. **LL (Low Latency)**: Optimized for small payloads.
3. **LL128**: Optimized for NVLink, achieving 93.75% effective bandwidth.

## NCCL vs. Other Libraries
1. **MPI Comparison**:
   - NCCL: GPU-focused optimizations.
   - MPI: General-purpose but less optimized for GPUs.
   - Hybrid use: MPI manages processes, NCCL handles GPU communication.
2. **Gloo Comparison**:
   - Gloo: Supports CPU/GPU.
   - NCCL: Outperforms in GPU scenarios.

# Common NVIDIA Tools
Reference: [WeLearnNLP Guide](https://www.yourmetaverse.cn/deep_learning/199/).

## `nvidia-smi topo -m`
Compare two cluster topologies:

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

**Key Observations**:
1. **Interconnect**:
   - Cluster 1: PCIe + NUMA nodes (SYS)
   - Cluster 2: NVLink (NV18) → Higher bandwidth/lower latency.
2. **NUMA Architecture**:
   - GPU 0-3: NUMA node 0
   - GPU 4-7: NUMA node 1
   - Optimal task placement: Minimize cross-NUMA communication.
3. **System Scale**:
   - Cluster 1: 64 CPU cores; Cluster 2: 192 cores.
4. **Topology Completeness**:  
   - Every GPU directly connects to all others
   - Eliminates intermediate "hops," improving throughput.  
## NVLink Inspection
```bash
nvidia-smi nvlink --status -i 0
nvidia-smi nvlink --capabilities -i 0
```
<details>
<summary>Example Output (H100 GPUs)</summary>

```bash
GPU 0: NVIDIA H100 80GB HBM3
         Link 0: 26.562 GB/s
         ... (18 links total)
         All links support:
         - Peer-to-peer (P2P) communication
         - System memory access
         - P2P/System atomics
         - SLI support
```
</details>

**Key Features**:
- Full bidirectional P2P support
- High-speed atomics
- System memory access

## GPU Monitoring
Tool recommendation: **[nvitop](https://github.com/Syllo/nvtop)**  
Install via `pip install nvitop` for real-time GPU metrics visualization.