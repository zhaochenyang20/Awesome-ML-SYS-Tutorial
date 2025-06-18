# SGLang-veRL Server：从 Engine 到 Server，我们需要更灵活的 RLHF rollout 接口

## 序言

这篇文章是 yitianlian 参与 SGLang RL 的工作成果，全程有 jhinpan 和 fzyzcjy 的合作，最后 zhaochenyang20 完成了 review，感谢每位参与者的贡献。

说来惭愧，大概一个半月前，jin 和我为了支持 [openR1 项目](https://github.com/huggingface/open-r1)，我们拉上秋江、jinwei、晓彤还有 xuting 一块儿给 HuggingFace 的三个子项目都加入了 SGLang 支持。蒸馏上，我们支持了 [distilabel](https://github.com/argilla-io/distilabel)，测评上，我们支持了 [lighteval](https://github.com/huggingface/lighteval/blob/main/docs/source/use-sglang-as-backend.mdx)。最后，在训练引擎上，我们支持了 trl 的 [grpo](https://github.com/huggingface/trl/pull/2981)。全线支持 open-r1 的工作真是非常辛苦，但是也算是圆满告成。唯独遗憾的是，trl 耽误了很长时间。一开始，jin 写了一个启动  [SGLang Sever](https://docs.sglang.ai/backend/send_request.html) 的版本，然后我反驳到，大家都用 [Engine](https://docs.sglang.ai/backend/offline_engine_api.html)，为什么我们要用 Server？我当时并没有尊重他的意见和劳动，直接武断地认为我们要和社区保持一致，所以否决了用 Server 的方法。这个其实毫无道理的决策耽误了他两周的工作时间，最后我们确实成功在 trl 上支持了 SGLang engine，但是现在 HuggingFace accelerate 意识到了他们的一些问题，还是没把我们的 PR merge 进去。

虽然确实耽误了 jin 很长时间，但是功不唐捐。现在他已经是我的强大合作者（还有债主了，苦笑）。从中，我还学到一个问题，**即便是看上去非常复杂高深的系统，很多细节的设计也有可能是社区开发者们在早期拍拍脑袋就决定了的。如果我觉得什么设计很奇怪，就应该认真想想，而不是“大家都这么做，我们也得这样”。** 对于 rollout 这件事情，只是因为大家用 Engine 用的顺手，所以开源的 RLHF 框架几乎都用了 Engine 来做 rollout。长远来看，为了和 environment 的交互和 rollout 的灵活性，我们确实需要 Sever 作为一个更灵活的接口。

## 为什么需要 Server 来完成 rollout

为了配合 agentic LLM 的训练，在现有的 PPO/GRPO 算法的基础上，从 single turn rollout 改动为 environment interactive multi-turn rollout 的需求非常强烈。

这一过程中，policy 与 environment 的交互存在绝对不可忽视的延迟，turn 之间的等待时间很长。一直用 `Engine` 做 rollout 的话（`engine.generate`），可能连 continuous batching 都组不起来。所以，改用 server 来通过 https 做 rollout 的需求就呼之欲出了。实际上，这也是 SGLang 最自然的工作方式。除此之外，environment 的交互往往也是通过 https 请求来完成的。譬如，众多 coding sandbox 都是 environment 自己启动一个 server 暴露一个 port，然后往里面发请求来实现交互的。

总之，为了在 training engine，rollout 和 environment 三个子进程中保持良好的通讯和交互，选择 server 势在必行。

## 一个收获良多的废案

最开始，为实现这一目标，我们将 SGLang 的 `launch_server` 函数改写为 `launch_server_from_verl_engine`，尝试在已有 `VerlEngine` 初始化的基础上，复用其 `TokenizerManager` 和 `SchedulerInfo`。这样做的初衷是避免重复创建通信管道。在 SGLang 中，TokenizerManager 和 SchedulerInfo 已经建立了一套完整的进程间通信（IPC）机制（譬如 ZMQ socket 通道）。如果重新创建这些组件而不是复用，可能会建立冗余的通信管道，消耗更多系统资源（内存、文件描述符等），增加系统负担。

然而，这种方案最终导致了更严重的问题：复用了组件出现了资源冲突。经过与 fzyzcjy 的讨论，我们发现根本原因在于 TokenizerManager 被两个线程同时访问——一个是原始的 `_engine` 线程，另一个是新增的 `server_from_verl_engine` 线程。SGLang 当前的设计并不支持 TokenizerManager 的并发访问，导致通信管道混乱。这种冲突最终表现为"第二次调用 `update_weights_from_tensor` 函数后会卡住"：第一次调用能够成功，但第二次调用会在进程间通信环节永久阻塞，最终触发 NCCL 超时错误。

虽然方案失败了，但是我们还是展示这个废案的开发过程。复盘失败经验，走向成功人生（苦笑）。

### `launch_server_from_verl_engine`

如同上文所述，我们增加 `launch_server_from_verl_engine` 函数作为 VerlEngine 的入口。该函数与 [`launch_server`](https://github.com/sgl-project/sglang/blob/ef9a378a209d970e0b5c48ae3eac6f2660d43faf/python/sglang/srt/entrypoints/http_server.py#L659) 类似，但允许外部传入已有的 `tokenizer_manager` 和 `scheduler_info`，并从 `VerlEngine` 内部启动 HTTP Server。

<details>
<summary>废案启动 verl server 的 endpoint</summary>

```python
def launch_server_from_verl_engine(
    tokenizer_manager: TokenizerManager,
    scheduler_info: Dict,
    server_args: ServerArgs,
    pipe_finish_writer: Optional[multiprocessing.connection.Connection] = None,
    launch_callback: Optional[Callable[[], None]] = None,
):

    set_global_state(
        _GlobalState(
            tokenizer_manager=tokenizer_manager,
            scheduler_info=scheduler_info,
        )
    )

    # Add api key authorization
    if server_args.api_key:
        add_api_key_middleware(app, server_args.api_key)

    # Add prometheus middleware
    if server_args.enable_metrics:
        add_prometheus_middleware(app)
        enable_func_timer()

    # Send a warmup request - we will create the thread launch it
    # in the lifespan after all other warmups have fired.
    warmup_thread = threading.Thread(
        target=_wait_and_warmup,
        args=(
            server_args,
            pipe_finish_writer,
            _global_state.tokenizer_manager.image_token_id,
            launch_callback,
        ),
    )
    app.warmup_thread = warmup_thread

    try:
        # Update logging configs
        set_uvicorn_logging_configs()
        app.server_args = server_args
        # Listen for HTTP requests
        uvicorn.run(
            app,
            host=server_args.host,
            port=server_args.port,
            log_level=server_args.log_level_http or server_args.log_level,
            timeout_keep_alive=5,
            loop="uvloop",
        )
    finally:
        warmup_thread.join()
```

</details>

### 修改 `VerlEngine.__init__`

我在 `tp_rank == 0` 的进程中，启动了一个新的线程来运行 `launch_server_from_verl_engine`，从而不阻塞主线程的初始化逻辑。并通过设置 `server_args.port` 为 `30000 + tp_rank` 避免端口冲突。

<details>
<summary>tp rank 0 上调用 launch_server_from_verl_engine</summary>

```python
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

        if self._tp_rank == 0:
            import copy

            new_server_args = copy.deepcopy(self._engine.server_args)
            new_server_args.port = 30000 + self._tp_rank
            print(f"launch_server_from_verl_engine {new_server_args.port}")

            def server_thread_wrapper(tokenizer_manager, scheduler_info, server_args):
                print(f"Server thread begin")
                launch_server_from_verl_engine(
                    tokenizer_manager=tokenizer_manager,
                    scheduler_info=scheduler_info,
                    server_args=server_args,
                )

            server_thread = threading.Thread(
                target=server_thread_wrapper,
                args=(
                    self._engine.tokenizer_manager,
                    self._engine.scheduler_info,
                    new_server_args,
                ),
                daemon=True,
            )
            server_thread.start()

        dist.barrier(group=self._device_mesh_cpu.get_group())
```

</details>

### 废案的具体问题

如我们之前所述，废案可以成功加载模型并且启动 Server，第一个 `update_weights_from_tensor` 调用也能顺利完成参数更新。然而在第二次调用该方法时，程序会**卡住不动，最终报出 NCCL 超时错误**。经过调试，我们发现：

- `scheduler` 在处理完更新任务后，调用了 `send_to_tokenizer.send_pyobj(output)`；
- `tokenizer_manager` 的 `handler_loop` 虽然还在运行，却无法收到该消息，主进程已经被阻塞；
- 如果注释掉 Server 的启动逻辑（在 `VerlEngine` 初始化时不调用 `launch_server_from_verl_engine`），上述问题完全消失，说明是 server 的某些组件影响了原有的通信逻辑。

据此，我们发现之前的多线程设计方案存在问题：

1. 线程安全问题：
    - TokenizerManager 可能不是线程安全的，当同时被多个线程访问时会导致竞争条件。

2. IPC 通道干扰：
    - Server 线程（FastAPI/Uvicorn）创建的事件循环与主线程的 ZMQ 通信通道产生相互干扰。
    - 这会导致 pipe 的一端被意外关闭，正如上文提到的 `BrokenPipeError: [Errno32] Broken Pipe` 这个报错。

3. GIL 限制下的阻塞：
    - Python 的 GIL（全局解释器锁）在处理 I/O 密集型任务时，线程切换不及时。
    - 当 Server 线程长时间占用 GIL，会导致 TokenizerManager 无法及时响应 ZMQ 消息。

4. 资源分配冲突：
    - 两个线程同时操作网络资源导致端口竞争，这可能与服务器上出现的 broken pipe 错误有关。

## 实际实现

鉴于在设计思路一中出现的问题，我们采用了一种完全不同的架构方案——完全分离 HTTP 服务器与 Engine，避免任何资源共享和并发访问冲突。具体来说，在废案中，我们在 `VerlEngine` 的 `_engine` 层面上进行了复用，而 `_engine` 的类型实际上是 `Engine`。在实际执行的方案中，我们将 `Engine` 和 `HTTP 服务` 完全解耦，使它们成为两个相互独立且同级的类。我们引入了 `HttpServerEngineAdapter` 作为替代方案，该方案的关键点是：

1. **完全分离而非共享资源**：与废案不同，我们放弃了资源复用的想法，不再尝试在同一进程的不同线程中共享 TokenizerManager，而是建立完全独立的服务器进程，通过 HTTP 通信进行交互。

2. **替换原有 Engine 对象**：参考下方 HttpServerEngineAdapter 的部分实现，通过将 VerlEngine 内部的 `_engine` 属性替换为 HttpServerEngineAdapter，我们实现了训练和推理服务的解耦。这种设计让训练进程专注于模型更新，而推理服务则独立运行在 HTTP 服务器中，避免了资源竞争和状态同步的复杂性。HttpServerEngineAdapter 实际上是一个代理模式的实现，它不含有 `_engine` 属性，而是直接通过 HTTP 请求与独立运行的服务器通信。如果保留原有的 Engine 对象，训练进程和推理服务会访问相同的资源，导致我们在废案中遇到的并发冲突问题，如通信通道混乱和进程阻塞。

3. **HTTP 请求代替直接调用**：当外部调用 `VerlEngine.update_weights_from_tensor()` 时，内部会通过 HTTP 请求将操作转发到独立的服务器进程，这完全避免了线程间共享资源的问题。

<details>
<summary>HttpServerEngineAdapter 的实现</summary>

```python
if first_rank_in_node and "launch_server" not in kwargs:
    os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
    self._engine = Engine(
        **kwargs, tp_size=self._tp_size, node_rank=node_rank, nnodes=nnodes
    )
elif "launch_server" in kwargs and kwargs["launch_server"]:
    del kwargs["launch_server"]
    if "server_args" in kwargs:
        # Directly load server_args
        server_args = kwargs["server_args"]
    else:
        # Construct server_args from kwargs
        if "log_level" not in kwargs:
            # Do not print logs by default
            kwargs["log_level"] = "error"
        server_args = ServerArgs(**kwargs)
    if self._tp_rank == 0:
        self._engine = HttpServerEngineAdapter(server_args)
    else:
        self._engine = None
else:
    self._engine = None
```
</details>

### 与废案的核心区别

1. **废案的资源共享导致冲突**：废案中，`TokenizerManager` 被两个线程同时访问：`_engine` 线程和 `server_from_verl_engine` 线程，由于 `TokenizerManager` 不是线程安全的，导致通信混乱。

2. **新方案的完全隔离确保稳定**：新方案中，`HttpServerEngineAdapter` 只是一个代理对象，不包含任何与原 `Engine` 共享的资源，它通过 HTTP 请求与独立运行的服务器进程通信，服务器进程拥有自己独立的 `TokenizerManager` 和 `SchedulerInfo`。

3. **请求转发机制**：在主节点（`tp_rank=0`）上，`VerlEngine` 会收集所有节点的张量数据，然后通过 `HttpServerEngineAdapter` 发送 HTTP 请求到服务器。这种 client-server architecture 的设计解决了废案中出现的资源冲突问题，确保了系统的稳定性和可靠性。Client-Server architecture 是一种将系统功能分为服务提供方（Server）和服务消费方（Client）的 distributed architecture。其优点包括职责明确分离、易于扩展、资源隔离和更强的故障隔离能力；缺点则是引入额外的网络通信开销、可能增加系统响应延迟以及需要处理网络故障的容错机制。

<details>
<summary>VerlEngine 中的 update_weights_from_tensor</summary>

```python
# VerlEngine 中的 update_weights_from_tensor
if self._tp_rank == 0:  # 只有主节点发送 HTTP 请求
    self._engine.update_weights_from_tensor(
        named_tensors=[(name, LocalSerializedTensor(values=gathered_serialized_tensors))],
        # 其他参数...
    )
```

</details>

### 为什么我们不采用 `update_weights_from_distributed` 来更新 Server 参数

在 SGLang 中，更新服务器参数有两种主要方法：`update_weights_from_distributed` 和 `update_weights_from_tensor`，它们的核心区别在于适用场景与通信方式。

`update_weights_from_distributed` 依赖 NCCL（NVIDIA Collective Communications Library）进行高效的 GPU 间通信，适用于全量分布式训练场景。
而 `update_weights_from_tensor` 的设计初衷是为了支持同节点、跨进程的权重更新操作，尤其适用于如 VerlEngine 这类部署 HybridEngine 的推理场景。
`update_weights_from_tensor` 实际上是直接在 CPU 或 GPU 之间通过进程间共享机制传递 Tensor 的，接收端直接将其复制到模型参数中。
**实际的数据传递不依赖 NCCL，而是基于指针传递 + 显式拷贝的方式完成。** 因此，`update_weights_from_tensor` 并不存在“通过 HTTP 或 NCCL 传输 tensor 数据”的过程。
如此以来，update weights 过程中，参数不会通过 HTTP 传输，我们从 veRL Engine 切换到 veRL Server 后，只会增加 meta data 的传输开销，不用担心参数传递的额外开销。
这点 meta data 的传输开销相比起 server 带来的编程灵活性，我们完全可以接受。

最终，我们在本版本中选择使用 `update_weights_from_tensor`，这有如下好处：

1. **兼容 VerlEngine 现有逻辑**  ：直接在同一节点内的不同进程间完成模型权重同步，避免对现有推理逻辑的干扰。

2. **数据拷贝由系统显式处理** ：我们在 `update_weights_from_tensor` 中传入的是 tensor 所对应的指针位置，因此由于 Server 中的 model 是在 GPU 上，所传输的 tensor 也必须要放在 gpu 上（ CPU 同理）。因此，不存在 NCCL 传输的需求，也无需担心 GPU 指针跨进程传递的问题。这里如何跨设备传输需要后续研究。

3. **`update_weights_from_distributed` 与 VeRL 框架存在设计冲突** ：当前 `update_weights_from_distributed` 的实现依赖 NCCL 在不同的 placement group 之间进行通讯，但 VeRL 的资源调度主要是 hybrid 的，也即 training 与 Inference 共用 placement group。如何在同一 placement group 上同时利用 NCCL 进行发送和接收还有待研究。

综上所述，`update_weights_from_tensor` 在兼容性、稳定性和实现灵活性方面更符合我们当前的部署需求。

### 序列化方式

在实现 `update_weights_from_tensor` 时，我们需要通过 HTTP POST 请求传递参数。然而，由于无法直接将自定义的 `UpdateWeightsFromTensorReqInput` 类作为参数传递，因此需要对其进行序列化处理。

在早期的实现中，我们采用了 `pickle` 加上 `base64` 的方式对整个类进行打包。这种方式虽然可行，但存在一些问题：首先，它显得不够自然，与常见的序列化习惯不符；其次，使用 `pickle` 在这种场景下并非必要，且可能带来额外的复杂性和潜在的安全风险。

在最新版本的实现中，我们对 `MultiprocessingSerializer` 进行了功能扩展。当需要通过 HTTP 传输数据时，我们会将字节类型的数据（`byte`）通过 `base64` 编码为字符串（`str`），以便能够顺利通过 HTTP POST 请求进行传输。而在反序列化时，如果检测到输入的数据类型是字符串，则会先使用 `b64decode` 将其解码回字节类型，然后再交由 `MultiprocessingSerializer` 原有的处理逻辑继续处理。

通过这一改进，我们不仅简化了 `update_weights_from_tensor` 的代码逻辑复用，还使整个流程更加规范、清晰且易于理解。新的实现方式避免了不必要的依赖（如 `pickle`），同时更符合常见的开发习惯，提升了代码的可读性和可维护性。

伪代码如下：
<details>
<summary>MultiprocessingSerializer 的扩展</summary>

```python
class MultiprocessingSerializer:
    @staticmethod
    def serialize(obj, output_str: bool = False):
        """序列化对象

        Args:
            obj: 需要序列化的对象
            output_str: 是否返回字符串表示（base64编码）
                        当序列化数据需要通过JSON传输时非常有用
        """
        # 原有序列化逻辑
        output = 原有序列化输出

        # 新增：支持字符串输出
        if output_str:
            return base64.b64encode(output).decode('utf-8')
        return output

    @staticmethod
    def deserialize(data, 其他参数):
        """反序列化对象

        Args:
            data: 序列化的数据（bytes或base64编码的字符串）
        """
        # 新增：自动检测并处理字符串输入
        if isinstance(data, str):
            data = base64.b64decode(data.encode('utf-8'))

        # 原有反序列化逻辑
```
</details>

### 多节点场景支持

为了更好地支持多节点部署，可以添加节点感知功能。伪代码当前如下，但是后续 host 应该要换成 ip：

<details>
<summary>Multi-nodes 的探讨</summary>

```python
class HttpServerEngineAdapter:
    def __init__(self, server_args, node_aware: bool = False):
        """初始化HTTP服务器引擎适配器

        Args:
            server_args: 服务器参数
            node_aware: 是否启用节点感知功能
        """
        self.host = server_args.host
        self.port = server_args.port
        self.node_aware = node_aware

        # 如果启用节点感知，收集所有可用节点信息
        if node_aware:
            self.nodes_info = self._discover_nodes()

    def _discover_nodes(self):
        """发现并记录集群中的所有节点"""
        # 节点发现逻辑
        pass
```
</details>

## 最终效果测试

启动新的虚拟环境，这里我们不用 docker，但是还是使用 uv。

```bash
cd ~
python3 -m venv ~/.python/veRL-server
source ~/.python/veRL-server/bin/activate
python3 -m pip install uv

# 安装 sglang
git clone https://github.com/yitianlian/sglang-fork.git
cd sglang-fork
git checkout feature/http_server_engine

# 重要：安装修改后的包
cd python
pip install .
pip install torch_memory_saver

# 测试 veRL Server
cd ../test/srt
python test_verl_engine_server.py
```
