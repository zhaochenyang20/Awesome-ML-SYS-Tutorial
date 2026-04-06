# Interactive Profiling in SGLang

## 1. Introduction

I’ve recently been investigating load-balancing issues in distributed Serving.

Our goal is to pinpoint the exact moment when load imbalance starts by monitoring real-time logs, and then validate the uneven-load hypothesis by analyzing the execution time of critical operators across ranks using trace data.

The main challenge is that enabling profiling from the very beginning of the server launch often produces trace files that are too large to manage. Therefore, we’d like to capture traces for selected time windows interactively while the SGLang server is running normally.

After reviewing the relevant sections in the SGLang documentation, I learned that SGLang now supports **Profile a server with HTTP API endpoints**, which allows flexible control of PyTorch Profiler. However, the “Profile with Nsys” documentation only describes configuring a delay and duration. This implies profiling must be enabled at server startup and the profiling window must be predetermined, which is not flexible enough for practical debugging.

This article covers three parts:

* **PyTorch Profiler**: how to profile a server via HTTP API endpoints (as described in SGLang docs).
* **Nsight Systems (nsys)**: an interactive profiling approach that differs from the documented delay/duration method.
* **Multi-rank analysis**: tips for conveniently comparing per-rank profiling results in Nsight Systems or Perfetto.

## 2. Interactive Profiling using HTTP API endpoints

This section mainly references relevant content from the SGLang docs.

SGLang provides two sets of HTTP APIs during server runtime: `/start_profile` and `/end_profile`. This allows us to start capturing traces at any time after the server is up and stop at an appropriate moment without restarting the service or pre-setting a profiling time window, thus more accurately aligning with specific workload patterns.

### 2.1 Endpoint Introduction

**start_profile**

Enables profiling on the SGLang server. We can control information such as the output file location, the starting step, the number of steps to capture, and the activities to be profiled.

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

Introduction to key parameters (all optional):

- `output_dir`: Determines the output folder. If not set, it defaults to the `SGLANG_TORCH_PROFILER_DIR` environment variable or `/tmp`.

- `start_step`: When to start profiling; used to skip the warm-up phase.

- `num_steps`: The number of steps to profile. If not set, profiling will continue indefinitely.

- `activities`: The activities to profile. Includes CPU, GPU, MEM, and RPD. The default is CPU and GPU.

- `merge_profiles`: Automatically merges analysis results from all ranks. Default is `false`.

- `profile_prefix`: Adds a specified prefix to the trace file for distinction.

Generated trace file naming format:

`{$profile_prefix}-{profile_id}-TP-{tp}-DP-{dp}-PP-{pp}-EP-{ep}.trace.json.gz`

Merged file format:

`merged-{profile_id}.trace.json.gz`

Full parameters:

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

**/end_profile**

Manually stops the server profile. If num_steps was already set in start_profile, the server will automatically stop after completing the specified steps.

```bash
curl -X POST http://127.0.0.1:30000/end_profile
```

### 2.2 Two Typical Workflows

Assume we have already launched the server.

**a. Fixed Window Capture**

If you only want to capture trace information for 10 steps, send start_profile in another terminal:

```bash
curl -X POST http://127.0.0.1:30000/start_profile \
  -H "Content-Type: application/json" \
  -d '{ 
    "num_steps": 10 
  }' 
```

**b. Flexible Control of Profiling Period**

Manually send start_profile and end_profile in another terminal:

```bash
curl -X POST http://127.0.0.1:30000/start_profile  

# serving...

curl -X POST http://127.0.0.1:30000/end_profile 
```

## 3. Interactive Profiling based on Nsys

### 3.1 Non-interactive Profiling

In this section, the SGLang Docs only introduce profiling the SGLang server by setting the delay and duration parameters. This means we need to configure the start time and duration (in seconds) of the profile when the server starts.

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

Where:  

- `cuda-graph-trace` controls the granularity of CUDA graph collection, determining whether to treat each CUDA graph as a whole or expand it into a set of nodes.

- `delay` controls when profiling starts.

- `duration` controls how long profiling lasts.

Finally, the user needs to find the profile session-id via:

```bash
nsys sessions list
```

And then manually kill the profiler:

```bash
nsys stop --session=profile-XXXXX
```

### 3.2 Interactive Profiling based on nsys start and nsys stop

When launching the SGLang server, add the extra parameter start-later to nsys profile. This will prevent nsys profile from starting immediately.

```bash
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node \
    -o sglang.out --start-later \
    python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --disable-radix-cache 
```

In another terminal, we can view the running nsys profile session ID via nsys sessions list.

```bash
              ID         TIME                       STATE LAUNCH NAME
        10345602        00:41           DelayedCollection      1 profile-345586
```

When you want to profile, use nsys start:

```bash
nsys start --session=10345602

# we can see
              ID         TIME                       STATE LAUNCH NAME
        10345602        02:35                  Collection      1 profile-345586 
```

When you want to end profiling, use nsys stop:

```bash
nsys stop --session=10345602
```

An `nsys-rep` file will be generated in the current directory. At this point, the server will not stop running, and we can continue to `start` and `stop` to capture traces of the server over multiple time periods.

## 4. Tips for Visualizing Multi-Rank Traces

After capturing traces from multiple ranks, how can we conveniently observe and compare the execution time of kernels across different ranks?

Currently, one answer is: Whether using Perfetto or Nsight Systems, you can pin the key rows (pin row). Afterward, you can conveniently compare them while zooming in and out of the entire visualization graph. Of course, if there are better methods, corrections and suggestions are welcome.

## References

https://github.com/sgl-project/sglang/issues/9638

https://docs.sglang.io/developer_guide/benchmark_and_profiling.html
