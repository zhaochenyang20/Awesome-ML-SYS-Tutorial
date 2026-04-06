# SGLang 交互式 Profiling 技术介绍

## 1. 前言

最近在研究SGLang多rank的负载均衡问题。

需求是，通过观察实时日志来识别开始发生负载失衡的时间点，并希望通过 trace 数据来分析各 Rank 的关键算子耗时，以此印证负载不均猜想。

痛点在于，如果从 Server 启动时就开启 profiling，生成的 trace 文件往往大到难以处理。因此希望SGLang Server刚开始正常运行，我们通过交互式的方式在特定时间段进行profile并获得相应trace信息。

在学习SGLang Docs的相应内容后，我了解到SGLang现在已经提供Profile a server with HTTP API endpoints的方式来灵活地控制Profile with PyTorch Profiler. 然而文档对于 Profile with Nsys 只介绍了设置 delay 和 duration 的做法。意味着需要在SGLang Server启动时就开启profile, 同时提前设定好profile的时机，这种方法确实是不够灵活。

因此，本文的内容主要包括三部分：

- 针对Pytorch Profile, 介绍文档中Profile a server with HTTP API endpoints的做法

- 针对Nsight Profile, 介绍不同于文档中的交互式profile做法

- 针对多rank profile的结果, 介绍如何在nsight system或perfetto中方便比较各rank的小技巧

## 2. 利用HTTP API endpoints进行动态Profile

本节内容主要参考了SGLang docs的相关内容.

SGLang 在 server 运行期间提供了两组 HTTP API：/start_profile 与 /end_profile . 这使得我们不需要重启服务，也不需要提前设定profiling 时间窗，你可以在server跑起来后，随时开始抓取trace，再在合适时机结束，从而更精准地对齐某类 workload pattern.

### 2.1 endpoint介绍

1. start_profile 

使sglang server开启profile. 我们可以控制输出文件的位置、起始step与抓取的step数、需要抓取的活动等信息。

```bash
# Wait 5 steps (warmup), then profile for 10 steps 
curl -X POST http://127.0.0.1:30000/start_profile \   
  -H "Content-Type: application/json" \   
  -d '{     
    "output_dir": "/tmp/profiles",
    "start_step": 5,
    "num_steps": 10,
    "activities": ["CPU", "GPU", "MEM"],
    "merge_profiles": true,
    "profile_prefix": "my_experiment_001"
    }' 
```

关键参数介绍(均为可选)：

- output_dir: 确定输出文件夹. 如果没有设定，基于SGLANG_TORCH_PROFILER_DIR环境变量或者默认/tmp.

- start_step: 什么时候开始profile, 用于跳过warm-up阶段.

- num_steps: 需要profile的step数. 如果不设定则会一直profiling.

- activities: 需要profile的活动. 包括CPU, GPU, MEM和RPD. 默认是CPU和GPU.

- merge_profiles: 自动合并所有rank的分析结果，默认为false

- profile_prefix: 给trace文件添加指定前缀，用于区分.

生成的trace文件命名格式:

{$profile_prefix}-{profile_id}-TP-{tp}-DP-{dp}-PP-{pp}-EP-{ep}.trace.json.gz

合并文件格式：

merged-{profile_id}.trace.json.gz

全部参数:

```python
# sglang/python/sglang/srt/managers /io_struct.py

class ProfileReqInput(BaseReq):
    # The output directory
    output_dir: Optional[str] = None
    # Specify the steps to start the profiling
    start_step: Optional[int] = None
    # If set, it profile as many as this number of steps.
    # If it is set, profiling is automatically stopped after this step, and
    # the caller doesn't need to run stop_profile.
    num_steps: Optional[int] = None
    # The activities to record. The choices are ["CPU", "GPU", "MEM", "RPD"]
    activities: Optional[List[str]] = None
    # Whether profile by stages (e.g., prefill and decode) separately
    profile_by_stage: bool = False
    # Whether to record source information (file and line number) for the ops.
    with_stack: Optional[bool] = None
    # Whether to save information about operator’s input shapes.
    record_shapes: Optional[bool] = None
    # Merge profiles from all ranks into a single trace
    merge_profiles: bool = False
    # The prefix of the profile filenames
    profile_prefix: Optional[str] = None
    # Only profile these stages and ignore others
    profile_stages: Optional[List[str]] = None
```

2. /end_profile 相关

手动停止server的profile，如果在start_profile中已经设定了num_steps, server将在完成设定steps的profile会停止.

```bash
curl -X POST http://127.0.0.1:30000/end_profile
```

### 2.2 两种典型工作流

假设我们已经跑起来了server

**a. 固定窗口抓取**

只想抓取10个step的trace信息，那么在另一个终端start_profile

```bash
curl -X POST http://127.0.0.1:30000/start_profile \
  -H "Content-Type: application/json" \
  -d '{ 
    "num_steps": 10 
  }' 
```

**b. 灵活控制profiling的时间段**

在另一个终端手动发送start_profile和end_profile

```bash
curl -X POST http://127.0.0.1:30000/start_profile  

# serving...

curl -X POST http://127.0.0.1:30000/end_profile 
```

## 3. 基于Nsys进行交互式Profile

### 3.1 非交互式profile

SGLang Docs在该部分，只介绍了通过设定delay与duration参数的方式来profile sglang server. 这意味着我们需要在server启动时便设定好开始profile的时间与持续时间(秒).

```bash
# launch the server, set the delay and duration times according to needs
# after the duration time has been used up, server will be killed by nsys

nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node \
    -o sglang.out --delay 60 --duration 70 \
    python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --disable-radix-cache

# client
python3 -m sglang.bench_serving --backend sglang \
    --num-prompts 1000 --dataset-name random \
    --random-input 1024 --random-output 512
```

其中,

- cuda-graph-trace用来控制采集cuda graph粒度，决定是把每个cuda graph作为一个整体还是展开成一组nodes

- delay控制开始profile的时间

- duration控制profile的持续时间

最后，用户需要通过

```bash
nsys sessions list
```

找到profile的session-id，然后手动杀掉profiler

```bash
nsys stop --session=profile-XXXXX
```

### 3.2 基于nsys start与nsys stop交互式profile

在启动sglang server时，加入nsys profile的额外参数start-later, 这将使得nsys profile不立刻进行.

```bash
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node \
    -o sglang.out --start-later \
    python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --disable-radix-cache 
```

在另一个终端，我们可以通过nsys sessions list查看正在运行的nsys profile session ID.

```bash
              ID         TIME                       STATE LAUNCH NAME
        10345602        00:41           DelayedCollection      1 profile-345586
```

当想要profile时，使用nsys start:

```bash
nsys start --session=10345602

# we can see
              ID         TIME                       STATE LAUNCH NAME
        10345602        02:35                  Collection      1 profile-345586 
```

当想要结束profile时，使用nsys stop:

```bash
nsys stop --session=10345602
```

将会在当前目录下生成nsys-rep文件. 此时server并不会停止运行，我们可以继续start、stop，从而在抓取server在多段时间的trace情况.

## 4. 可视化观察多rank trace的小技巧

抓取多rank的trace后，如果想可视化比较各rank的kernel执行时间。如何方便观察呢？

目前一个答案是：无论是用perfetto还是nsight system，都可以把关键行pin住(pin row)。之后可以在放缩整个可视化图中方便比较。当然不知道有没有更好的办法，欢迎大佬批评指正。

## 参考文献

https://github.com/sgl-project/sglang/issues/9638

https://docs.sglang.io/developer_guide/benchmark_and_profiling.html
