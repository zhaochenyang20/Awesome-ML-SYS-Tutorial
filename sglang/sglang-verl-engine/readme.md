# SGLang Verl Engine 优化解析

## 接口总结

1. `update_weights_from_tensor` in `python/sglang/srt/entrypoints/engine.py`

```python
    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]],
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ):
        """Update weights from distributed source. If there are going to be more updates, set `flush_cache` to be true
        to avoid duplicated operations such as clearing cache."""
        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[
                MultiprocessingSerializer.serialize(named_tensors)
                for _ in range(self.server_args.tp_size)
            ],
            load_format=load_format,
            flush_cache=flush_cache,
        )
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.update_weights_from_tensor(obj, None)
        )
```

这个接口不会因为 serializer 的序列化速度而很慢么？

2. RPC in `python/sglang/srt/entrypoints/engine.py`

```python
    """
    Execute an RPC call on all scheduler processes.
    """

    def collective_rpc(self, method: str, **kwargs):
        obj = RpcReqInput(method=method, parameters=kwargs)
        self.send_to_rpc.send_pyobj(obj)
        recv_req = self.send_to_rpc.recv_pyobj(zmq.BLOCKY)
        assert isinstance(recv_req, RpcReqOutput)
        assert recv_req.success, recv_req.message

    def save_remote_model(self, **kwargs):
        self.collective_rpc("save_remote_model", **kwargs)

    def save_sharded_model(self, **kwargs):
        self.collective_rpc("save_sharded_model", **kwargs)
```

RPC (Remote Procedure Call，远程过程调用) 是一种在分布式系统中实现进程间通信的技术。在这段代码中，RPC 被用于在主进程和调度器进程（scheduler processes）之间进行通信。

1. `collective_rpc` 方法用于向所有调度器进程发送 RPC 请求；
2. 通过 ZMQ（消息队列中间件）实现进程间通信：
   - `send_pyobj` 发送序列化的 Python 对象作为请求
   - `recv_pyobj` 接收响应
3. 请求包含：
   - `method`：要调用的远程方法名
   - `parameters`：方法参数（以 kwargs 形式传递）

从代码上下文可以看到，这个 RPC 机制主要用于：
- 保存远程模型（`save_remote_model`）
- 保存分片模型（`save_sharded_model`）

这种设计让主进程能够控制和协调多个调度器进程的行为，是分布式推理系统中的重要组成部分。

3. `DeviceMesh, DTensor` in `python/sglang/srt/entrypoints/verl_engine.py`

```python
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh, DTensor

from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from sglang.srt.server import Engine
from sglang.srt.utils import MultiprocessingSerializer, broadcast_pyobj


class VerlEngine:
    def __init__(
        self,
        device_mesh_cpu: DeviceMesh,
        nnodes: int = 1,
        **kwargs,
    ):
        self._device_mesh_cpu = device_mesh_cpu
        self._tp_rank = device_mesh_cpu.get_local_rank()
        self._tp_size = device_mesh_cpu.size()
        tp_size_per_node = self._tp_size // nnodes
        node_rank = self._tp_rank // tp_size_per_node
        first_rank_in_node = self._tp_rank % tp_size_per_node == 0

        if first_rank_in_node:
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
            self._engine = Engine(
                **kwargs, tp_size=self._tp_size, node_rank=node_rank, nnodes=nnodes
            )
        else:
            self._engine = None

        dist.barrier(group=self._device_mesh_cpu.get_group())
```

DeviceMesh 和 DTensor 是 PyTorch 中用于分布式计算的重要组件，主要用于模型并行和张量并行计算。

1. DeviceMesh（设备网格）：
- 是一个逻辑设备阵列，用于管理分布式计算中的设备拓扑结构；
- 定义了如何在多个设备（如 GPU）之间组织和协调计算
- 在代码中，`device_mesh_cpu` 用于：

```python
self._tp_rank = device_mesh_cpu.get_local_rank()  # 获取当前进程的局部排名
self._tp_size = device_mesh_cpu.size()  # 获取总设备数量
```

2. DTensor（分布式张量）：
- 是 PyTorch 中的分布式张量类型；
- 允许将一个大张量分片到多个设备上；
- 在代码中的处理：

`device_mesh_cpu` 的命名反映了它的实际用途，作为一个在 CPU 层面的控制结构，用于协调分布式系统中的进程通信和任务分配，而具体的 GPU 计算则是通过其他机制来管理的。
