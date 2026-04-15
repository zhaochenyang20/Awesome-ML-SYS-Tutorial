# Ray Placement Group

本节详细说明 slime 在 Ray 上如何进行 GPU 资源编排，包括：
- 如何创建并重排 Placement Group（PG）以实现稳定的 GPU排序
- 训练 Actor 与 Rollout Engine 如何在 PG 上调度
- 两种部署形态：colocate与 dis-agg

---

## High Level Concepts

- [`Ray Placement Group`](https://github.com/THUDM/slime/blob/main/slime/ray/placement_group.py)：在集群中预留一组 bundle（每个包含 1GPU+1CPU），并将后续 actor 固定绑定到这些 bundle 上，实现可控、稳定的资源放置。
- [`RayTrainGroup`](https://github.com/THUDM/slime/tree/main/slime/ray/actor_group.py)：训练侧“同构 actor 组”的管理器。
  - 按稳定顺序为每个 rank 创建训练 actor，并提供并发的 init/train/eval/save/update/offload 接口。
  - `self._actor_handlers`：保存所有训练 actor 列表，长度等于 world_size；后续所有并发操作都是对这个列表做映射调用. Created by [`_allocate_gpus_for_actor`](https://github.com/THUDM/slime/blob/main/slime/ray/actor_group.py#L67-L81)。
  - 详细内容会放在Part 3.
- [`RolloutManager`](https://github.com/THUDM/slime/tree/main/slime/ray/rollout.py)：推理/数据编排器，负责创建 Rollout Engine, Data Buffer、Lock and Router；其细节放在 Part 2。
- [`InfoActor`]()：临时探测 actor，用于获知“bundle 实际落点的 (node_ip, gpu_id)”，从而对 bundle 进行稳定排序。

---

## 核心入口与总体流程
入口位于[`Ray Placement Group`](https://github.com/THUDM/slime/blob/main/slime/ray/placement_group.py) 的 `create_placement_groups`
<details> <summary> create_placement_groups </summary>

```1:121:https://github.com/THUDM/slime/tree/main/slime/ray/placement_group.py
def create_placement_groups(args):
    """Create placement groups for actor and rollout engines."""

    num_gpus = 0
    if args.debug_train_only:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
        rollout_offset = 0
    elif args.debug_rollout_only:
        num_gpus = args.rollout_num_gpus
        rollout_offset = 0
    elif args.colocate:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
        rollout_offset = 0
    else:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node + args.rollout_num_gpus
        rollout_offset = args.actor_num_nodes * args.actor_num_gpus_per_node

    print(f"Creating placement group with {num_gpus} GPUs...")
    pg, actor_pg_reordered_bundle_indices = _create_placement_group(num_gpus)

    rollout_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[rollout_offset:]

    return {
        "actor": (pg, actor_pg_reordered_bundle_indices),
        "rollout": (pg, rollout_pg_reordered_bundle_indices),
    }
```
</details>


要点：
- 计算本次训练所需的总 GPU 数 `num_gpus`。
- 创建一个包含 `num_gpus` 个 bundle 的 PG，每个 bundle 需要 `{"GPU": 1, "CPU": 1}`。
- 获得“重排后的 bundle 索引列表”，用于确保稳定的跨节点/GPU 顺序。
- 根据 `rollout_offset` 将 PG 的索引划分给训练 Actor 与 Rollout 引擎。

---

## 稳定的 Bundle 重排：按节点与 GPU 顺序排序

创建 PG 后，slime 用一个临时 `InfoActor` 在每个 bundle 上运行一次，探测该 bundle 实际分配到的 `(Node IP, GPU ID)`，随后按“节点 IP 数值化”与 “GPU ID”排序，得到稳定序列。
<details> <summary> `InfoActor` and `sort_key` </summary>

```
@ray.remote(num_gpus=1)
class InfoActor:
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]

def sort_key(x):
    index, node_identifier, gpu_id = x
    # Sort by node IP number and then by GPU ID
    try:
        # try to parse it as an IP address.
        ip_address = node_identifier
        node_ip_parts = list(map(int, ip_address.split(".")))
    except ValueError:
        # Try to resolve the hostname to an IP address.
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # Instead, we convert each character of the original identifier string
            # to its ASCII value. This provides a stable and consistent numerical
            # representation that allows for sorting.
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, gpu_id)
```
</details>

`sort_key` 的策略要点：
- 优先尝试将 `node_identifier` 解析成 IPv4 地址，转成 4 个整型并据此排序；
- 若不是 IP，则尝试 DNS 解析；还不行则退化为按主机名字符的 ASCII 序列排序；
- 同节点内再按 `gpu_id` 升序排序。

这样可以获得跨多机的稳定 bundle 顺序，避免由于 Ray 的内部调度差异导致“训练 rank 对应 GPU”与“rollout 分片对 GPU”的映射不稳定。

---

## Colocate vs Dis-agg

### Colocate（同机混布）

场景：训练 Actor 与 Rollout 引擎共用同一批 GPU 资源（例如 8 张卡上既跑训练也跑推理）。

- 计算方式：`num_gpus = actor_num_nodes * actor_num_gpus_per_node`；`rollout_offset = 0`。
- 意味着 Rollout 的 bundle 索引切片与 Actor 相同来源（即同一个 PG 的全部 bundle）。
- 调度层面对资源共享做了“比例切分”：
  - 训练 Actor 默认 `num_gpus_per_actor = 0.8`（一张卡上可并存 1 个 Actor + 若干轻量进程）。
  - Rollout 引擎默认 `num_gpus_per_engine = 0.2`（见后文），让推理与训练在同卡共存。

适用：小规模单/多机、节省节点数，总吞吐优先，但需注意推理与训练争抢显存与算力。

### Dis-agg（训练/推理分离）

场景：训练 Actor 与 Rollout 引擎使用各自独立的 GPU 池（例如训练占 8 卡，rollout 占 4 卡）。

- 计算方式：`num_gpus = actor_total + rollout_total`，`rollout_offset = actor_total`。
- PG 仍然是一个，但将前 `actor_total` 个稳定 bundle 分给训练 Actor，后 `rollout_total` 个分给 Rollout 引擎。
- 完全避免资源争抢，推理服务可更稳定；代价是需要更多 GPU。
- *在这种情况下可以进行async-train*

---

## Train vs Rollout

- Train（RayTrainGroup）：
  - 绑定顺序：使用“稳定重排后的 bundle 索引”按 rank 依序绑定；rank0 回传 `MASTER_ADDR/PORT`。
  - 资源占比：`num_gpus_per_actor`（默认 0.8）允许与 rollout 同卡共享 (when co-locate)。
  - 并发管理：通过 `self._actor_handlers` 批量并发 init/train/eval/save/update。
  - 参考创建循环见上文“组件与职责总览”的代码片段。

- Rollout（create_rollout_engines）：
  - 绑定策略：同样使用重排索引；每引擎默认 `num_gpus=0.2`，`placement_group_capture_child_tasks=True` 使子任务也受 PG 约束。
  - 跨机一致性：为多节点引擎分配服务/NCCL/分布式初始化/DP-attention 端口，保证同一引擎内节点共享 `dist_init_addr`。
  - 详细实现见 Part 2。

---

## 端口分配与多机一致性

在多节点/多卡下，`create_rollout_engines` 会通过 `RayActor._get_current_node_ip_and_free_port` 在目标节点上寻找一段连续可用端口，并将Node 0的 `dist_init_addr` 扩散到同一引擎的其他节点，以保证跨机的进程组一致性。
<details> <summary> `RayActor._get_current_node_ip_and_free_port` </summary>

```python
def _get_current_node_ip_and_free_port(start_port=10000, consecutive=1):
    address = ray._private.services.get_node_ip_address()
    address = address.strip("[]")
    port = start_port
    while not all(is_port_available(port + i) for i in range(consecutive)):
        port += 1
    return address, port
```
</details>

---

## 选择 colocate 还是 dis-agg？

- 选择 colocate，当：
  - 资源有限或部署简化优先；
  - 能接受推理与训练在同卡上带来的资源争用与性能波动；
  - Rollout 引擎较轻，推理吞吐不敏感。

- 选择 dis-agg，当：
  - 追求稳定的推理延迟/吞吐，或推理负载较重；
  - 资源充足，期望训练与推理互不干扰；
  - 需要隔离不同业务的资源池；
  - 希望进行aysnc训练。

---

## 小结

- 通过“InfoActor 探测 + 稳定排序”，slime 获得跨多机的稳定 bundle 顺序；
- 训练与 Rollout 共享或分离资源，由 colocate 与 dis-agg 两种模式切换；
- 端口与分布式地址由引擎所在节点本地探测，确保跨机一致性与可复现部署。

---

## FAQ：为什么要绑定 InfoActor？为什么需要 sort_key？能不能直接按分配顺序？

- 绑定 InfoActor 的原因：
  - **探测实际落点**：PG 的 bundle 索引不等于物理 (node, gpu)；只有把任务调度到该 bundle 上才能知道真实落点。
  - **拿到 GPU 编号**：`InfoActor` 以 `num_gpus=1` 运行，`ray.get_gpu_ids()` 才会返回明确的本地 GPU ID。
  - **与后续一致**：训练/rollout 也会用相同的 `placement_group_bundle_index` 绑定到这些 bundle，先探明映射便于稳定放置与核对。

- 需要 sort_key 的原因：
  - **Ray 不保证 bundle→拓扑 的固定顺序**：不同运行/集群状态下，PG 内部资源映射可能变动。
  - **训练与通信需要稳定顺序**：希望跨节点按 IP 升序、节点内按 GPU 升序，便于 NCCL 拓扑、日志与排障一致。
  - **sort_key 策略**：IP 解析→数值化排序；失败则 DNS；再失败按主机名 ASCII；节点内按 `gpu_id` 升序。

- 不能直接用“分配顺序”的原因：
  - **不可控且不稳定**：随时间/占用/健康状况变化，rank→GPU 映射漂移，复现实验困难。
  - **与切片/跨机策略冲突**：例如 dis-agg 切分、每引擎多卡首卡定位，会因顺序不稳而引入隐性错配。

