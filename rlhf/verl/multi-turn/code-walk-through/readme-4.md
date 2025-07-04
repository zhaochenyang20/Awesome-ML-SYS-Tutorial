# RL 系统深思：深入理解基于 Hybrid Engine 的 RL 框架的权重更新机制

因为工作需要，最近终于得空能够再次深入去学习思考主流 RL 框架的系统设计。我们希望能够通过一系列文档分享我们的思考，也希望能够得到大家的反馈，和志同道合的朋友一同打造更好的开源 RLHF 框架。我们将这系列文章称为 RL 系统深思。本文是这系列的第一篇，重点讨论 Hybird Engine 下的权重更新机制。本文将会以 SLIME 和 verl 为主，讨论二者更新权重的整体逻辑，重点分析 SGLang `update_weights_from_tensor` 的机制；对于其他两种 update 方法，我们会在本文中进行对比分析。

从逻辑上，在 co-locate 策略下的权重更新都是类似的，我们以 FSDP training backend 为例，给出一个简化而通用的更新流程，核心就是这样一个[代码片段](https://github.com/volcengine/verl/blob/0508af25b66e839772fba8e79d97896bf0d843d3/verl/workers/sharding_manager/fsdp_sglang.py#L160)：

```python
def _preprocess_tensor_for_update_weights(tensor: torch.Tensor):
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    return tensor

async def update_weights(self, params):
    named_tensors = [(k, v) for k, v in params.items()]
    load_format = None
    for tensor_index, (name, tensor) in enumerate(named_tensors):
        serialized_tensor = MultiprocessingSerializer.serialize(_preprocess_tensor_for_update_weights(tensor))

        if self.device_mesh["infer_tp"].get_local_rank() == 0:
            gathered_serialized_tensors = [None for _ in range(self.device_mesh["infer_tp"].mesh.size()[0])]
        else:
            gathered_serialized_tensors = None
        dist.gather_object(
            obj=serialized_tensor,
            object_gather_list=gathered_serialized_tensors,
            dst=self.device_mesh["infer_tp"].mesh.tolist()[0],
            group=self.device_mesh["infer_tp"].get_group(),
        )

        if self.device_mesh["infer_tp"].get_local_rank() == 0:
            await self.inference_engine.update_weights_from_tensor(
                named_tensors=[
                    (
                        name,
                        LocalSerializedTensor(values=gathered_serialized_tensors),
                    )
                ],
                load_format=load_format,
                flush_cache=tensor_index == len(named_tensors) - 1,
            )
```

参数是逐个进行聚合并且更新的，更新完一个参数后 release 一个，然后继续循环。我们以单个参数为例。假设这个参数的 size 是 `[1024, 1024]`，FSDP 的 TP size 是 4，而 SGLang 的 TP size 是 2。因此在更新参数开始前， 每个 rank 上在 FSDP engine 内有 `[256, 1024]` 大小的 tensor，而 SGLang engine 有 `[512, 1024]` 大小的 tensor。

1. 权重导出：每个 rank 调用 `_preprocess_tensor_for_update_weights()` 将当前参数的完整 tensor 进行聚合，实际上把分散在各个 GPU 上的参数分片都聚合到了每个 rank 上，每个 rank 上都有一份当前参数的完整 tensor。此时，这个 parameter 会有三份，前两份是 FSDP 和 SGLang 本身就有的 `[512, 1024]` 和 `[256, 1024]`，第三份是为了聚合而单独开辟的 `[1024, 1024]` 大小的 tensor。
2. tensor 序列化：调用 `MultiprocessingSerializer.serialize()`，将每个 rank 上聚合到的参数序列化，得到序列化后的 handler，称为 `serialized_tensor`。注意，虽然序列化传入的参数是聚合得到的那个 `[1024, 1024]` 的 tensor，但是实际上返回的只有被序列化的 handler。handler 近似于指向 tensor 实际存储的指针，存放了虚拟地址，stripe，size 等等信息，以及后续在 SGLang engine load 新参数过程中需要的 CUDA IPC handler。
3. 聚合 handler 到 FSDP TP 0：虽然每个 rank 上都聚合了当前参数的完整 tensor，但是只有 tp 0 通过 `gather_object` 收集了所有 rank 的 handler。
4. 跨进程传递：FSDP TP rank 0 将收集到的 handler tuple 列表打包为 `LocalSerializedTensor` 对象用于后续重建。接着，通过跨进程通信传递给 SGLang Engine。这里传递的只有序列化后的 handler，而非实际数据。
5. SGLang Engine 重建 tensor：SGLang 的每个 TP rank 调用 `_unwrap_tensor()` 来处理 `LocalSerializedTensor`。在 `_unwrap_tensor` 中，monkey patch 了 torch 的 `rebuild_cuda_tensor`，然后 `MultiprocessingSerializer.deserialize()` 反序列化得到 handler。
5. SGLang engine change handler：注意到，本轮需要更新的参数在 SGLang engine 中本身就存在，对 SGLang engine 而言，只需要将原本的参数的 handler 在 `load_weights` 时更换为反序列化后得到的 handler，然后这个新的 handler 进行切片，再将被替换掉的 handler 指向的原本的 tensor 释放掉，然后释放掉切片后不需要的 tensor。

由此以来，其实在任意一个 TP 上，只是临时创建了一个 `[1024, 1024]` 的 tensor，然后原本的 handler 被更换后，这个 `[1024, 1024]` 的 tensor 所不用的那一半会被 release 掉，原本 SGLang engine 里面的 handler 指向的旧的 tensor 会被释放掉，并没有显存泄露。

![update_weights](./update_weights.jpg)

这么分析后其实还是蛮复杂的，我们继续逐层深入。本文主要采用 FSDP 训练 backend，对 megatron 略有区别但是大同小异。

## 权重导出

在将训练阶段结束后，首先使用 FSDP `state_dict` 导出权重。一般而言，`state_dict` 是一个 `param name -> tensor` 的 dict，而 FSDP 中的 `state_dict` 的值取决于 `StateDictType` 的模式。FSDP 内置了三种模式：`FULL_STATE_DICT`、`SHARDED_STATE_DICT` 以及 `LOCAL_STATE_DICT`。我们分别来看看这三种模式假设有一个 4-rank FSDP 训练，参数形状为 `[1024, 1024]`，每个 rank 负责 1/4 的参数：

1. `FULL_STATE_DICT`

```python
# 每个 rank 都得到完整的参数
{
    'layer1.weight': tensor([1024, 1024]),  # 完整张量，每个 rank 都相同
    'layer1.bias': tensor([1024]),          # 完整张量，每个 rank 都相同
    # ... 所有参数都是完整的
}
```

2. `LOCAL_STATE_DICT`  

```python
# 每个 rank 只得到自己负责的分片
{
    'layer1.weight': tensor([256, 1024]),  # 当前 rank 的分片 (1/4)
    'layer1.bias': tensor([256]),          # 当前 rank 的分片 (1/4)
    # ... 只有当前 rank 负责的参数分片
}
```

3. `SHARDED_STATE_DICT`

```python
# 每个 rank 得到包含元信息的分片对象
{
    'layer1.weight': ShardedTensor {
        metadata: {
            "world_size": 4,           # 总分片数
            "rank": 1,                 # 当前分片索引 (0-3)
            "shape": [1024, 1024],     # 完整张量的形状
            "dtype": torch.float16,    # 数据类型
        },
        local_shard: tensor([256, 1024]),  # 当前 rank 的分片数据
    },
    'layer1.bias': ShardedTensor {
        metadata: { "world_size": 4, "rank": 1, "shape": [1024], "dtype": torch.float16 },
        local_shard: tensor([256]),    # 当前 rank 的分片数据
    }
}
```

其中 `FULL_STATE_DICT` 是最朴素的实现方式。`LOCAL_STATE_DICT` 只保存当前 rank 所存储的部分，没有切片信息，而 `SHARDED_STATE_DICT` 在 `LOCAL_STATE_DICT` 基础上，额外存有当前 rank 负责的参数分片和切片信息。通过 `full_tensor()` 就可以将 `SHARDED_STATE_DICT` 状态下的 tensor 聚合起来：

```python
def _preprocess_tensor_for_update_weights(tensor: torch.Tensor):
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    return tensor
```

## tensor 序列化

如同前文所说，序列化一个 tensor 实际上得到的返回值是序列化后的 handler，或者更严谨的说法是 handler tuple。我们来看看序列化最后层层向下调用的 `reduce_tensor()` 函数的返回值：

```python
 return (
            rebuild_cuda_tensor, # 重建函数
            (
                type(tensor), # tensor 类型
                tensor.size(), # tensor 大小
                tensor.stride(), # tensor 步长
                tensor_offset,  # tensor 在 storage 中的偏移量
                type(storage), # storage 类型
                tensor.dtype, # tensor 数据类型
                device, # tensor 设备
                handle,  # identifier which CUDA allocation is the storage in.
                storage_size_bytes,  # storage 大小
                storage_offset_bytes,  # storage 在 CUDA allocation 中的偏移量
                tensor.requires_grad, # tensor 是否需要梯度
                ref_counter_handle, # 引用计数器 handle
                ref_counter_offset, # 引用计数器偏移量
                event_handle, # 事件 handle
                event_sync_required, # 事件同步是否需要
            ),
        )
```

可见，对一个 CUDA Tensor 调用 `reduce_tensor`，实际上返回的是一个 Python Tuple，包含了重建 Tensor 所需的一切，而绝不包含实际存储的数据本身。接着，这个 handler tuple 通过进程间通信（比如 zmq）传递给接收方。接收方拿到的自然也不是数据本身，而是一组可以帮助重新找到（重建）这个 tensor 的 handler，我们在后文中会以 handler tuple 来指代。


## 聚合 handler

【TODO】

### SGLang 零拷贝反序列化

我们可以继续阅读`SGLang`侧代码。`SGLang Engine`接收到这个对象后，在 [ModelRunner.update_weights_from_tensor](https://github.com/sgl-project/sglang/blob/392e441ad17c78b68638f2d959fcf592d19b4834/python/sglang/srt/model_executor/model_runner.py#L774) 中，每个 `TP-rank` 调用`_unwrap_tensor`。

```python
def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, Union[torch.Tensor, "LocalSerializedTensor"]]],
        load_format: Optional[str] = None,
    ):
        named_tensors = [
            # 取数据
            (name, _unwrap_tensor(tensor, tp_rank=self.tp_rank))
            for name, tensor in named_tensors
        ]
```

每个`tp rank`调用的 [_unwrap_tensor](https://github.com/sgl-project/sglang/blob/392e441ad17c78b68638f2d959fcf592d19b4834/python/sglang/srt/model_executor/model_runner.py#L1464)，来依据自己的 `tp_rank` 从 `LocalSerializedTensor.values` 里取出本 rank 对应的 `handler tuple`。然后通过 `MultiprocessingSerializer.deserialize` 执行 `rebuild_cuda_tensor`，根据`handler tuple`重建`tensor`。最通过 `.to(torch.cuda.current_device())` 把`tensor`指向到当前 `GPU device`。



```python
def _unwrap_tensor(tensor, tp_rank):
    if isinstance(tensor, LocalSerializedTensor):
        monkey_patch_torch_reductions()
        tensor = tensor.get(tp_rank)
    return tensor.to(torch.cuda.current_device())


@dataclass
class LocalSerializedTensor:
    """torch.Tensor that gets serialized by MultiprocessingSerializer (which only serializes a pointer and not the data).
    The i-th element in the list corresponds to i-th rank's GPU."""

    values: List[bytes]

    def get(self, rank: int):
        # 取出对应的handler tuple并反序列化
        return MultiprocessingSerializer.deserialize(self.values[rank])
```



## Weight Loader 按 TP 规则切片并写回参数

继续阅读 [ModelRunner.update_weights_from_tensor](https://github.com/sgl-project/sglang/blob/392e441ad17c78b68638f2d959fcf592d19b4834/python/sglang/srt/model_executor/model_runner.py#L774) ，在 rebuild `tensor`后，`SGLang`内部会根据不同的策略调用对应`weight_loader`。

```python
if load_format == "direct":
    _model_load_weights_direct(self.model, named_tensors)
elif load_format in self.server_args.custom_weight_loader:
    custom_loader = dynamic_import(load_format)
    custom_loader(self.model, named_tensors)
elif load_format is None:
    self.model.load_weights(named_tensors)
else:
    raise NotImplementedError(f"Unknown load_format={load_format}")
return True, "Success"
```

内部的`weigt_loader`通常会对重建好的`full tensor`进行切分，每个`tp rank`只获取自己需要的`tensor slice`并写入显存，不需要将`full tensor`再broadcast到所有`rank`。

样例伪代码：

```python
def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
   input_dim = getattr(param, "input_dim", None)
   …
   shard_size = param_data.shape[input_dim]          # 本 rank 需要的切片宽
   start_idx = self.tp_rank * shard_size             # 偏移 = rank * 宽
   loaded_weight = loaded_weight.narrow(input_dim,   # narrow = 按 dim 切
                                        start_idx,
                                        shard_size)
   param_data.copy_(loaded_weight)                   # 写入显存
```

我们也可以通过下图更加直观的理解这一过程:
> by 标哥

![SGLang load weights (1)](https://hackmd.io/_uploads/HJBb7UzBll.png)



### 流程总结

通过`Verl`侧将所有 `handker tuple`收集到`rank 0`，传递给`SGLang`完成重建，再到`load_weight`中每个`rank`通过`slice`方式从`full tensor`中获取对应的`shard`。`Verl/SGLang`实现了高效按需拷贝权重同步。

# SLIME/Verl 实现对比

# MOE 相关优化


# FULL_STATE_DICT  All-Gather机制

`FULL_STATE_DICT` 的核心机制是在 `state_dict()` 被调用前，通过 All-Gather 操作将模型参数从各个设备（Rank）上聚合，形成一个完整的、未分片的模型状态。这个过程的调用链条清晰，最终汇集到底层的分布式通信操作。

1. **入口钩子 (`Hook`)**: 整个流程始于 [`_full_pre_state_dict_hook`](https://www.google.com/search?q=[https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/torch/distributed/fsdp/_state_dict_utils.py%23L277](https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/torch/distributed/fsdp/_state_dict_utils.py%23L277))。这个钩子函数在 `state_dict()` 运行前被触发，它的主要职责是调用下一个工具函数来准备参数的 unshard（聚合）操作。

   ```python
   # _full_pre_state_dict_hook
   def _full_pre_state_dict_hook(...):
       # ...
       _common_unshard_pre_state_dict_hook(...) # 关键调用
   ```

2. **进入 Unshard 上下文**: `_full_pre_state_dict_hook` 随即调用 [`_common_unshard_pre_state_dict_hook`](https://www.google.com/search?q=[https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/torch/distributed/fsdp/_state_dict_utils.py%23L148](https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/torch/distributed/fsdp/_state_dict_utils.py%23L148))。此函数通过调用 `_enter_unshard_params_ctx` 进入一个特殊的上下文管理器，该管理器负责执行参数聚合。

   ```python
   # _common_unshard_pre_state_dict_hook
   def _common_unshard_pre_state_dict_hook(...):
       # ...
       _enter_unshard_params_ctx(...) # 进入上下文
   ```

3. **获取 Handle 并执行 Unshard**: 在[上下文管理器](https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/torch/distributed/fsdp/_unshard_param_utils.py#L160)内部，系统首先通过 [`_module_handle`](https://www.google.com/search?q=[https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/torch/distributed/fsdp/_common_utils.py%23L195](https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/torch/distributed/fsdp/_common_utils.py%23L195)) 获取到一个 `FlatParamHandle` 对象，这个对象封装了对 FSDP 管理的扁平化参数（`FlatParam`）的操作。随后，调用 [`_unshard`](https://www.google.com/search?q=[https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/torch/distributed/fsdp/_runtime_utils.py%23L273](https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/torch/distributed/fsdp/_runtime_utils.py%23L273)) 方法。

   ```python
   # 上下文管理器内部
   handle = _module_handle(state, module) # 获取 FlatParamHandle
   # ...
   _unshard(state, handle, ...)
   ```

   `_unshard` 方法的核心就是调用 `handle` 自身的 `unshard()` 方法。

   ```python
   # _unshard
   def _unshard(...):
       # ...
       handle.unshard() # 关键调用
   ```

4. **执行 All-Gather**: `FlatParamHandle` 的 [`unshard`](https://www.google.com/search?q=[https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/torch/distributed/fsdp/_flat_param.py%23L1320](https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/torch/distributed/fsdp/_flat_param.py%23L1320)) 方法会调用内部的 `_all_gather_flat_param`。

   ```python
   # FlatParamHandle.unshard
   def unshard(self):
       # ...
       # 调用 All-Gather
       padded_unsharded_flat_param = self._all_gather_flat_param(...)
       # ...
   ```

5. **调用底层分布式通信**: 最后，在 [`_all_gather_flat_param`](https://www.google.com/search?q=[https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/torch/distributed/fsdp/_flat_param.py%23L1401](https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/torch/distributed/fsdp/_flat_param.py%23L1401)) 方法中，根据参数所在的设备（CPU 或 GPU），系统会调用 PyTorch `distributed` 库中相应的 All-Gather 函数来完成最终的跨设备数据聚合。

   ```python
   # _all_gather_flat_param
   def _all_gather_flat_param(...):
       if sharded_flat_param.is_cpu:
           # CPU 使用 all_gather
           dist.all_gather(...)
       else:
           # GPU 使用优化的 all_gather_into_tensor
           dist.all_gather_into_tensor(...)
   ```

通过这一系列调用，FSDP 确保了在生成 `state_dict` 时，每个 Rank 都拥有完整的模型参数，从而可以正确地保存或导出模型权重。

