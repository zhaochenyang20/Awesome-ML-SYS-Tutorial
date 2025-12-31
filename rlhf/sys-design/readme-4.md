# 专家并行（Expert Parallelism）

专家并行（Expert Parallelism）的概念非常straightforward，按照专家切分分配到不同的GPU上，实现计算负载的分布式处理，提高计算效率。

专家并行最大的不同在于，输入数据需要通过一个动态的路由选择机制分发给相应专家，此处会涉及到一个所有节点上的数据重分配的动作。在所有专家处理完成后，又需要将分散在不同节点上的数据按原来的次序整合起来。使用All-to-All的通信方式。专家并行可能存在负载不均衡的问题。

## EP 流程

光做ep的流程是：

1. Gate 选择 expert，按 expert 分桶
2. all-to-all dispatch（把 token 发给 expert 所在 gpu）
3. expert compute
4. all-to-all return
5. 按 gate 权重合并

## FSDP2 + EP

| forward | backward |
| --- | --- |
| gate<br><br>all-to-all dispatch<br><br>expert compute (FSDP2)<br><br>all gather<br><br>Expert FFN compute<br><br><br><br>release<br><br>all-to-all return<br><br>merge | gate<br><br>all-to-all dispatch<br><br>expert compute (FSDP2)<br><br>all gather<br><br>Expert FFN compute<br><br>reduce-scatter<br><br>release<br><br>all-to-all return<br><br>merge |


## 优化点

最大的问题在于all to all 产生的大量的 gpu 间通信，应该怎么优化？读下来主要的优化点是：

1. ep切分 dim0 （专家），fsdp 切分 dim1 hiddensize （见VeOmni）
2. prefetch，指在计算第n层的时候，预先把第n+1层的参数gather起来。只用fsdp的话可以很直白的做prefetch。但是因为ep计算开始之前有通信，可能会乱掉，需要手动做一些操作来前向和反向的prefetch （见VeOmni）
3. 用deepep，苦nccl久矣 （见Automodel）
   Deepep:
https://www.cnblogs.com/CQzhangyu/p/18741625
https://zhuanlan.zhihu.com/p/28867733102
https://zhuanlan.zhihu.com/p/27777601573


4. eplb ep下的load balance （TODO 还要读一下）https://zhuanlan.zhihu.com/p/29963005584
5. fused moe, 这个在ep之外了，指的是gpu负责n个专家，然后用fused moe kernel来加速对他持有的这几个专家的计算


## 实现对比

veomni, torchtitan, automodel 里都是先 ep，然后对 ep 完的每个块做 fsdp。

- **VeOmni**：fsdp 手动 prefetch 做 overlap，reshard policy
- **TorchTitan**：fsdp 手动 prefetch 做 overlap，deepep，reshard policy
- **Automodel**：deepep

我(zhuorany)主要读了verOmni和Automodel


## VeOmni

```
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


整个逻辑看下来还是比较清晰的，就是一个ep + fsdp的结构，对于专家先apply ep 在第0个维度（expert）上切分，然后再fsdp，非专家部分直接按照常规fsdp即可。
```text
    Applies EP (when enabled) + FSDP2 parallel strategy to the model.

    Flow:
    1. Apply EP: Expert tensors [128,H,I] -> [32,H,I] local tensors per EP rank
    2. Apply FSDP2 to expert modules: Shard expert tensors along dim-1 (hidden dim)
    3. Apply FSDP2 to regular modules: Standard dim-0 sharding
    4. Result: Expert params [32,H/fsdp_size,I], regular params use standard FSDP2
```
下面是parallelize_model_fsdp2重的关键部分和代码，[完整代码](https://github.com/ByteDance-Seed/VeOmni/blob/3bd8e6e48c2d741b2b8b4898f90645145bf4287b/veomni/distributed/torch_parallelize.py#L228)

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

## Automodel

```
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

automodel 比较值得一看的就是他是怎么用deepep的。Automodel 通过 `_DeepepManager` 集成了 DeepEP，利用 Fused Dispatch/Combine 算子替代了 NCCL All-to-All。

`token_dispatcher.py` -> `MoEFlexTokenDispatcher` -> `_DeepepManager` -> `fused_dispatch`

### DeepepManager(token_dispatcher.py)

有状态的通信上下文管理器，它封装了 DeepEP 库与上层模型逻辑之间的交互。

在 dispatch 阶段，DeepEP 底层返回一个 handle 对象（包含了通信布局信息。在 combine 阶段，直接取出 self.handle 传给底层。

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

### DeepEP 封装 (fused_a2a.py)

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


## TorchTitan

```
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

[完整代码](https://github.com/pytorch/torchtitan/blob/7e4ab85998576c68902603058adada28fb0ed226/torchtitan/models/llama4/infra/parallelize.py#L494)

### fsdp+ep 代码 (torchtitan/models/llama4/infra/parallelize.py)

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

### prefetch 代码

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
