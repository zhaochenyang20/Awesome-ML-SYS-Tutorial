# RDMA Weight Transfer for RL: A Learning Log

> **Reader**: Chenyang Zhao — SGLang core team, Miles/RadixArk founding engineer
> **Goal**: Get fluent enough in RDMA concepts and the SGLang RDMA weight-transfer design to participate in code review and architecture discussions around [sgl-project/sglang#17311](https://github.com/sgl-project/sglang/issues/17311)
> **Time budget**: ~3 hours
> **Prerequisite knowledge assumed**: NCCL collectives, PyTorch distributed, tensor/expert parallelism, the SGLang weight-update APIs, Miles weight-sync architecture


## How to Use This Document

This document is organized into three one-hour blocks. Each block ends with a checklist so you can verify understanding before moving on. The blocks build on each other:

| Block | Topic | You Will Learn |
|-------|-------|---------------|
| 1 | RDMA Fundamentals | The networking primitives underneath NCCL and Mooncake |
| 2 | RDMA in the SGLang Codebase | How TransferEngine, memory registration, and the weight-update APIs connect |
| 3 | Miles ↔ SGLang Integration & PR Review | The full trainer-to-rollout pipeline, the three open PRs, and what to look for in review |


## Block 1: RDMA Fundamentals (~60 min)

### 1.1 What Is RDMA and Why Should You Care?

You already know NCCL. NCCL is the *user-facing* collective library; **RDMA is one of the transports underneath it**. When you run `torch.distributed.broadcast(tensor, src=0)` over NCCL on an InfiniBand cluster, the actual bytes move via RDMA verbs under the hood.

**RDMA** (Remote Direct Memory Access) lets one machine read from or write to another machine's memory **without involving either machine's CPU or OS kernel on the data path**. The NIC (Network Interface Card) handles the transfer autonomously after initial setup.

Why this matters for RL weight sync specifically:

| Concern | TCP/Socket | NCCL (over RDMA) | Raw RDMA (Mooncake) |
|---------|-----------|-------------------|---------------------|
| CPU involvement per transfer | High (memcpy to/from kernel buffers) | Low (NCCL handles internally) | Near-zero (NIC does DMA) |
| Kernel bypass | No | Partial (NCCL manages) | Full |
| Collective semantics | Manual | Built-in (broadcast, allreduce) | Point-to-point only |
| Fine-grained control | High | Low (opaque groups) | High |
| Setup complexity | Low | Medium (process groups) | High (MR, QP, handshake) |

**Key insight**: NCCL already uses RDMA when available, so *why bother with raw RDMA?* Two reasons:

1. **Point-to-point zero-copy**: NCCL's `broadcast` requires all ranks to participate in a collective. With raw RDMA, a trainer can write directly into a specific inference engine's GPU memory without coordinating a group — no barrier, no collective.
2. **Decoupled lifecycle**: NCCL process groups must be created before communication and destroyed after. In RL, training and rollout engines may start/stop independently (elastic scaling, fault recovery). A transfer-engine approach avoids the tight coupling of process-group lifecycles.

### 1.2 The RDMA Programming Model in 60 Seconds

RDMA programming uses the **Verbs API** (libibverbs). You do not need to write verbs code to review the PRs, but understanding the mental model is essential:

```
Step 1: Open device       →  "Which NIC am I using?"
Step 2: Create QP          →  Queue Pair = send queue + receive queue (like a socket)
Step 3: Register Memory    →  Pin a buffer so the NIC can DMA to/from it
Step 4: Exchange metadata  →  Tell the remote side: "here is my QP number and memory address"
Step 5: Post RDMA Write    →  "Write these bytes to that remote address" — NIC does the rest
Step 6: Poll completion    →  Check that the transfer finished
```

The critical concept for code review is **Memory Registration (MR)**:

- Before RDMA can touch a buffer, that buffer must be **registered** with the NIC driver.
- Registration pins the virtual-to-physical mapping so the NIC's DMA engine can translate addresses.
- Registration is **expensive** (kernel call, page pinning). Doing it per-parameter is slow; doing it once for a contiguous block is fast.
- This is exactly why `register_memory_region_v2()` in SGLang merges contiguous weight blocks before registration — see Section 2.3.

### 1.3 InfiniBand vs RoCE vs TCP

| | InfiniBand (IB) | RoCE v2 | TCP |
|---|---|---|---|
| Network | Dedicated IB fabric | Standard Ethernet | Standard Ethernet |
| RDMA support | Native | Yes (over UDP/Ethernet) | No (kernel bypass impossible) |
| Latency | ~1 μs | ~2-5 μs | ~50-100 μs |
| Typical bandwidth | 200-400 Gbps (NDR/XDR) | 100-400 Gbps | 25-100 Gbps |
| Congestion control | Credit-based, lossless | ECN-based (needs PFC/ECN config) | TCP congestion control |
| Deployment | HPC/ML clusters | Cloud providers (Azure, GCP) | Everywhere |

For SGLang's use case, **IB** is the primary target. The Mooncake TransferEngine initializes with `"rdma"` transport by default (see `mooncake_transfer_engine.py:195`), which uses IB verbs.

### 1.4 GPU-Direct RDMA (GDR)

Standard RDMA transfers between host (CPU) memory buffers. **GPU-Direct RDMA** extends this to GPU memory:

```
Without GDR:   GPU → PCIe → CPU RAM → NIC → Network → NIC → CPU RAM → PCIe → GPU
With GDR:      GPU → PCIe → NIC → Network → NIC → PCIe → GPU
```

GDR eliminates two CPU-memory copies. This is how Mooncake and NCCL achieve GPU-to-GPU RDMA transfers. Requirements:
- NVIDIA GPU with GDR support (all modern datacenter GPUs)
- Mellanox/NVIDIA NIC with GPUDirect support
- `nv_peer_mem` or `nvidia-peermem` kernel module loaded
- CUDA IPC for cross-process GPU memory sharing on the same node

### 1.5 How NCCL Uses RDMA (Closing the Loop)

NCCL abstracts all of this. When you create an NCCL process group:

1. NCCL discovers the network topology (NVLink, PCIe, IB).
2. It picks the best transport for each pair of ranks (NVLink for intra-node, IB for inter-node).
3. For IB transport, NCCL internally creates QPs, registers memory, and posts RDMA operations.
4. `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX` control which interfaces NCCL uses.

**Environment variables you will see in Miles and SGLang configs:**

| Variable | Purpose |
|----------|---------|
| `NCCL_SOCKET_IFNAME` | Network interface for NCCL's TCP bootstrap (e.g., `eth0`, `bond0`) |
| `NCCL_IB_HCA` | Which IB HCA (Host Channel Adapter) to use (e.g., `mlx5_0,mlx5_1`) |
| `NCCL_IB_GID_INDEX` | GID index for RoCE (irrelevant for native IB) |
| `NCCL_CUMEM_ENABLE` | Enable CUDA memory management integration (set in Miles `actor_group.py`) |
| `NCCL_DEBUG` | Set to `INFO` or `TRACE` for debugging transport selection |

### Block 1 Checklist

- [ ] Can explain why raw RDMA is useful when NCCL already uses RDMA internally
- [ ] Understand Memory Registration: why it exists, why it is expensive, why batching matters
- [ ] Know the difference between IB and RoCE at a high level
- [ ] Understand GPU-Direct RDMA data path vs non-GDR path
- [ ] Can identify NCCL environment variables related to IB transport


## Block 2: RDMA in the SGLang Codebase (~60 min)

### 2.1 The Three Weight-Update Strategies (Review)

You already know these, but let's frame them by their transport layer:

| Strategy | Transport | When to Use | Key File |
|----------|-----------|-------------|----------|
| `update_weights_from_disk` | Filesystem I/O | Elastic scaling, checkpointing | `model_runner.py:1126` |
| `update_weights_from_tensor` | Shared memory / CUDA IPC | Co-located training + rollout | `model_runner.py:1438` |
| `update_weights_from_distributed` | NCCL broadcast (RDMA under the hood) | Disaggregated training + rollout | `model_runner.py:1348` |

**The RDMA transfer engine adds a fourth path**: direct GPU-to-GPU writes via Mooncake, bypassing NCCL collectives entirely. This is what Issue #17311 is building.

### 2.2 Mooncake TransferEngine in SGLang

The existing RDMA infrastructure lives in:

```
python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py
```

`MooncakeTransferEngine` wraps the `mooncake.engine.TransferEngine` and provides:

| Method | What It Does |
|--------|-------------|
| `initialize(hostname, device_name)` | Opens the IB device, starts RPC service for peer discovery |
| `register(ptr, length)` | Registers a single GPU memory region for RDMA |
| `batch_register(ptrs, lengths)` | Registers multiple regions in one kernel call (faster) |
| `transfer_sync(session_id, buffer, peer_addr, length)` | RDMA Write: copies `length` bytes from local `buffer` to `peer_addr` on remote `session_id` |
| `batch_transfer_sync(session_id, buffers, peer_addrs, lengths)` | Batched RDMA Writes |
| `get_session_id()` | Returns `hostname:rpc_port` — the identifier remote peers use to reach this engine |

**Initialization flow** (`mooncake_transfer_engine.py:191-200`):
```python
ret_value = self.engine.initialize(
    hostname,
    "P2PHANDSHAKE",    # discovery mode: peer-to-peer
    "rdma",             # transport: RDMA over IB verbs
    device_name,        # e.g., "mlx5_0"
)
```

The `"P2PHANDSHAKE"` mode means peers discover each other directly (no central coordinator), exchanging QP metadata over a lightweight RPC channel.

### 2.3 Memory Registration: Why v2 Exists

In `remote_instance_weight_loader_utils.py`, there are two registration strategies:

**v1** (`register_memory_region_v1`): Registers each parameter individually.
```python
for name, weight in model.named_parameters():
    transfer_engine.register_memory(weight.data_ptr(), weight.numel() * weight.element_size())
```
Problem: For a 70B model with ~1000 parameters, this means ~1000 kernel calls for memory registration.

**v2** (`register_memory_region_v2`): Walks `torch.cuda.memory.memory_snapshot()` to find contiguous physical blocks that hold weights, then registers merged blocks.
```python
# Walks memory segments, merges adjacent weight blocks
for segment in memory_snapshot:
    # ... merge adjacent active_allocated blocks that hold weights
weight_blocks_for_reg_mr.append((merged_address, merged_size))

# Register merged blocks (far fewer kernel calls)
for weight_block in weight_blocks_for_reg_mr:
    transfer_engine.register_memory(address, size)
```

**Why this matters for review**: Any PR that changes memory layout (e.g., weight offloading, quantization) could break the v2 merge logic. The merge depends on weights being in contiguous CUDA allocator blocks.

### 2.4 The Weight Info Dictionary

After registration, SGLang builds a `weight_mr_dict` mapping parameter names to their GPU memory locations:

```python
weight_mr_dict[name] = (
    weight.data_ptr(),    # GPU virtual address
    weight.numel(),       # Number of elements
    weight.element_size() # Bytes per element
)
```

This dictionary is exposed via the HTTP API at `/get_remote_instance_transfer_engine_info` so that remote peers (trainers) can look up *exactly where* to RDMA-write each parameter.

### 2.5 The Remote Instance Weight Loading Flow

Putting it all together, the existing RDMA weight loading flow (for loading weights from a seed instance to a new instance) works like this:

**Seed instance side**: (1) Start and load model, (2) init `MooncakeTransferEngine`, (3) register all weight memory regions, (4) expose `weight_mr_dict` via HTTP.

**New instance side**: (5) Start, query seed's HTTP API for weight info (session_id + addresses), (6) init its own `MooncakeTransferEngine`, (7) allocate matching GPU buffers, (8) for each parameter, RDMA-read from the seed's registered address into the local buffer. The NIC handles the transfer autonomously. (9) Load weights from buffers into the model.

The key HTTP endpoints involved:
- `GET /get_remote_instance_transfer_engine_info?rank=N` — returns session_id and weight_mr_dict for rank N
- `POST /init_weights_send_group_for_remote_instance` — for NCCL-based remote loading
- `POST /send_weights_to_remote_instance` — triggers weight sending

### 2.6 What Issue #17311 Adds: Training → Rollout RDMA Path

The existing RDMA path is **instance-to-instance** (one SGLang server to another). Issue #17311 extends this to **trainer-to-inference-engine**:

**Training side** (Miles/Megatron): (1) Complete a training step. (2) Create an "engine replica" with the same parallelism config as SGLang — this is enabled by **PR #16860** which exposes the parallelism info. (3) Map Megatron parameter names to SGLang shard names — enabled by **PR #17326** (unified weight mapping). (4) For each parameter: AllGather TP shards, convert to HF format, bucket parameters, then RDMA-write to the SGLang engine's registered memory address.

**SGLang rollout side**: (5) The engine's weight locations are queryable, including cross-node — fixed by **PR #17389** for `nnode > 1`. (6) Weights appear directly in GPU memory, zero-copy, with no NCCL group needed.

### Block 2 Checklist

- [ ] Can trace the full data path: training step → parameter gather → RDMA write → SGLang GPU memory
- [ ] Understand why `register_memory_region_v2` merges blocks and what could break it
- [ ] Know what `weight_mr_dict` contains and how it is exposed via HTTP
- [ ] Can explain the difference between the existing instance-to-instance RDMA path and the new trainer-to-inference path
- [ ] Understand the role of MooncakeTransferEngine's `session_id` in addressing


## Block 3: Miles ↔ SGLang Integration & PR Review (~60 min)

### 3.1 Miles' Current Weight Transfer Architecture

Miles currently supports two weight transfer strategies (neither uses raw RDMA directly):

**Strategy A: `UpdateWeightFromTensor`** (Co-located, Gloo + Ray)
```
Training Ranks → Gloo gather_object() → Source Rank → Ray remote call → SGLang
```
- Tensors serialized via `MultiprocessingSerializer`
- Gathered to a single rank over Gloo (CPU-based)
- Source rank sends to SGLang via `update_weights_from_tensor()` (in-memory)
- Used when training and inference share the same node
- File: `miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py`

**Strategy B: `UpdateWeightFromDistributed`** (Disaggregated, NCCL broadcast)
```
Training Ranks → AllGather TP shards → Convert to HF → NCCL broadcast → SGLang
```
- Creates temporary NCCL process groups (`miles-pp_{rank}`)
- Broadcasts weights from training rank 0 to all SGLang engine GPUs
- Handles expert (MoE) parameters separately with AllGather EP
- File: `miles/backends/megatron_utils/update_weight/update_weight_from_distributed.py`

**Where RDMA fits**: Strategy B already benefits from RDMA *implicitly* (NCCL uses IB when available). The proposed RDMA transfer engine path would **replace** the NCCL broadcast with direct P2P RDMA writes, eliminating:
- The need to create/manage temporary NCCL process groups
- The collective synchronization overhead (broadcast requires all participants)
- The tight lifecycle coupling between training and rollout processes

### 3.2 The Engine Replica Architecture (Issue #17311)

The architecture described in Issue #17311 introduces an **engine replica** on the training side:

On the **training node**, the Megatron trainer feeds updated parameters to a CPU-offloaded **engine replica** that mirrors SGLang's parallelism config (same TP/PP/EP), performs weight name mapping, buckets parameters, and drives RDMA writes via `MooncakeTransferEngine`. On the **SGLang rollout node**, the inference engine has its GPU memory regions registered for RDMA. The engine replica writes directly into those registered addresses over the network. No NCCL process group is needed; the NIC handles the data movement.

**Engine replica** responsibilities:
1. **Mirror SGLang's parallelism config** — knows how parameters are sharded (TP, PP, EP)
2. **Map parameter names** — translates Megatron names → SGLang/HuggingFace names
3. **Bucket parameters** — groups small parameters into larger transfers
4. **Drive RDMA writes** — uses `MooncakeTransferEngine.transfer_sync()` to write directly into SGLang's registered GPU memory

### 3.3 The Three PRs: What They Do and What to Look For

#### PR #16860: Expose Parallelism Info

**Purpose**: Let the engine replica know SGLang's exact parallelism config (TP/PP/EP/attn_tp/attn_dp) per rank.

**What it adds**: HTTP endpoint `/get_parallelism_config` that returns per-rank parallelism configuration.

**Review concern raised** (by slin1237): Should this be a new endpoint or part of existing server info? JD-ETH argues it is rank-level info (not server-level), since P2P transfer needs per-rank queries. This parallels `/get_remote_instance_transfer_engine_info` which is also per-rank.

**What to look for in review**:
- Is the parallelism config complete enough for the engine replica to reconstruct shard shapes?
- Does it work correctly with DP > 1? (The PR also fixes `dp_parallel_controller` propagation of `scheduler_infos`)
- Security: is exposing parallelism config over HTTP acceptable? (It's internal-network only)

#### PR #17326: Unified Weight Mapping Interface

**Purpose**: Expose the parameter name mapping layer so the engine replica can translate incoming Megatron parameters to SGLang model parameters.

**What it adds**: A `Mixin` interface that surfaces `stacked_params_mapping` and `expert_params_mapping` from model `__init__`, plus hooks for pre-transforms and special-case mappings.

**Supported models**: Llama3, Qwen2, Qwen3, Qwen3-MoE, GLM4, GLM4-MOE, DeepseekV2

**The core contract**:
```python
sglang_param, num_shards, shard_idx = model.load_weight(megatron_weights)
```
This tells the engine replica: "this Megatron parameter maps to this SGLang parameter, and it's shard `shard_idx` of `num_shards` total shards." The engine replica can then track when all shards of a parameter have arrived and the full parameter is ready.

**What to look for in review**:
- Does the mapping handle all stacked params correctly? (e.g., `qkv_proj` is stacked from `q_proj`, `k_proj`, `v_proj`)
- MoE expert mapping: does it handle expert-parallel sharding?
- Are there models missing from the supported list that Miles needs? (Check Miles' supported model list against this PR)

#### PR #17389: Cross-Node Scheduler Info Sync

**Purpose**: Fix `/get_remote_instance_transfer_engine_info` for multi-node deployments.

**The problem**: In multi-node SGLang, each node only knows about its local schedulers. A remote trainer querying `rank=N` could fail if rank N is on a different node.

**The solution**: After local schedulers initialize, use Gloo `all_gather` to synchronize `scheduler_infos` across all nodes. This happens on a dedicated port (`dist_port + 10000`) before the HTTP server starts.

The flow is: both nodes finish initializing their local schedulers, then they perform a Gloo `all_gather` of `scheduler_infos` across all nodes (on a dedicated port at `dist_port + 10000`). Only after this synchronization completes does the HTTP server start on each node — so every node knows the transfer engine info for all ranks, not just its local ones.

**What to look for in review**:
- Port collision: `dist_port + 10000` — is this configurable? Could it conflict?
- Gloo timeout: what happens if one node is slow to start?
- Does it handle the DP attention case correctly?

### 3.4 Alternative Approach: P2P Transfer Manager (PR #14170)

There is a separate, alternative implementation by Risc-lt:

| Aspect | Issue #17311 (JD-ETH) | PR #14170 (Risc-lt) |
|--------|----------------------|---------------------|
| Architecture | Engine replica on trainer side | P2PTransferManager with engine pool |
| Transfer engines | Single MooncakeTransferEngine per rank | Pool of engines (25% of CPU cores, max 8) |
| Coordination | Engine replica handles mapping | ZMQ handshake + task queue |
| Scope | Modular PRs (3 PRs + demo) | Single monolithic PR |
| Integration | Designed for Miles | Generic (any trainer) |

Both approaches use Mooncake's RDMA engine underneath. The engine-replica approach (Issue #17311) is more aligned with Miles' architecture because the mapping and bucketing logic lives on the trainer side, keeping SGLang's code minimal.

### 3.5 Key Review Questions for All Three PRs

When reviewing, keep these cross-cutting concerns in mind:

1. **Fault tolerance**: What happens if a transfer fails mid-way? Is there rollback? Can the system recover without restarting everything?

2. **Version consistency**: Miles tracks `weight_version` integers. After an RDMA write completes, how does SGLang know the new version is fully committed? Is there a fence/barrier?

3. **Memory safety**: RDMA writes directly to GPU memory. If SGLang is serving inference while weights are being written, you get torn reads. The `pause_generation` → `update` → `continue_generation` flow must be airtight.

4. **Quantization**: Miles handles int4/fp4/mxfp8 post-processing after weight transfer. Does the RDMA path preserve the right dtype and shape for quantized models?

5. **Merge conflicts**: All three PRs have dirty mergeable state (conflicts with main). This is worth flagging — they need rebasing.

### 3.6 Reading List for Deeper Dives

If you have more time after the 3 hours, these are high-value next reads:

| Resource | Why Read It | Time |
|----------|-------------|------|
| [Awesome-ML-SYS-Tutorial: Weight Update Mechanisms](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/sys-design/readme-1-EN.md) | Your own tutorial — review it with RDMA context | 20 min |
| [Mooncake TransferEngine docs](https://kvcache-ai.github.io/Mooncake/getting_started/build.html) | Understand the library SGLang wraps | 30 min |
| [JD-ETH's RDMA integration branch](https://github.com/JD-ETH/slime/tree/jd/rdma-integration) | The working prototype — see the engine replica in action | 45 min |
| SGLang `docs/advanced_features/sglang_for_rl.md` | Refresh on the full RL integration surface | 15 min |
| PR #14170 diff | Compare the alternative P2P approach | 30 min |

### Block 3 Checklist

- [ ] Can draw the engine replica architecture from memory
- [ ] Understand what each of the three PRs contributes to the overall RDMA flow
- [ ] Know the difference between the engine-replica approach and the P2P transfer manager approach
- [ ] Can identify the key review risks: fault tolerance, version consistency, memory safety, quantization
- [ ] Have a reading list for going deeper


## Quick Reference: Key Files in the SGLang Codebase

| File | What It Contains |
|------|-----------------|
| `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py` | `MooncakeTransferEngine` — the RDMA wrapper |
| `python/sglang/srt/model_loader/remote_instance_weight_loader_utils.py` | Memory registration (v1/v2), weight info dict, remote loading triggers |
| `python/sglang/srt/model_executor/model_runner.py:1290-1500` | `init_weights_update_group`, `update_weights_from_distributed`, `update_weights_from_tensor` |
| `python/sglang/srt/weight_sync/tensor_bucket.py` | `FlattenedTensorBucket` — parameter bucketing for efficient transfers |
| `python/sglang/srt/managers/io_struct.py` | All request/response types for weight update APIs |
| `python/sglang/srt/managers/scheduler_update_weights_mixin.py` | Scheduler-side weight update, memory release/resume |
| `python/sglang/srt/entrypoints/http_server.py` | HTTP endpoint definitions for all weight update APIs |
| `docs/advanced_features/sglang_for_rl.md` | Official RL integration guide |

## Quick Reference: Key Files in Miles

| File | What It Contains |
|------|-----------------|
| `miles/backends/megatron_utils/update_weight/update_weight_from_distributed.py` | Strategy B: NCCL broadcast weight sync |
| `miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py` | Strategy A: Gloo + Ray weight sync |
| `miles/backends/megatron_utils/update_weight/common.py` | `all_gather_param()`, `all_gather_params_async()`, name normalization |
| `miles/backends/sglang_utils/sglang_engine.py` | SGLang engine lifecycle management, weight reception endpoints |
| `miles/backends/fsdp_utils/update_weight_utils.py` | FSDP backend weight sync (both strategies) |
| `miles/ray/rollout.py` | `RolloutManager` — SGLang engine init, generation, fault tolerance |
| `miles/ray/actor_group.py` | `RayTrainGroup` — distributed training orchestration |


## Glossary

| Term | Definition |
|------|-----------|
| **HCA** | Host Channel Adapter — the InfiniBand network card |
| **IB** | InfiniBand — a high-performance interconnect fabric |
| **MR** | Memory Region — a registered buffer that RDMA can access |
| **QP** | Queue Pair — RDMA's equivalent of a socket (send queue + receive queue) |
| **GDR** | GPU-Direct RDMA — RDMA directly to/from GPU memory, bypassing CPU |
| **RoCE** | RDMA over Converged Ethernet — RDMA over standard Ethernet |
| **Verbs** | The low-level RDMA programming API (libibverbs) |
| **RNIC** | RDMA-capable NIC |
| **PD** | Protection Domain — groups QPs and MRs for access control |
| **CQ** | Completion Queue — where completed RDMA operations are reported |
| **WR** | Work Request — an RDMA operation posted to a QP |
| **Engine Replica** | A lightweight model mirror on the trainer side that knows SGLang's parallelism config and drives RDMA writes |
| **Transfer Engine** | Mooncake's RDMA abstraction layer used in SGLang |
| **FlattenedTensorBucket** | SGLang's mechanism for flattening multiple parameters into a single contiguous buffer for efficient transfer |
