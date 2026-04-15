# Analyzing GPU Memory Leaks with Torch Memory Snapshot

Recently, we ran into some GPU memory leak issues in both RL training and SGLang's own inference. Yesterday, I finally figured out the specific cause of the leaks. This article shares our debugging process using **Torch Memory Snapshot** and our solutions to these memory leak problems.

Special thanks to: Hongyu Lu (TikTok), Xinpeng Wei (Amazon), Rohan Bavishi (Amazon), Vint Lee (Amazon), Daisy Lin (Amazon), Deniz Birlikci (Amazon), Shahil Patel (Amazon), XJ Wang (Amazon), Huapeng Zhou (UW), Changyi Yang (CMU), Xinyuan Tong (USC), Yuhao Yang (HKU), Biao He (LinkedIn), Chenyang Zhao (LMSYS)

## Background

Interestingly, we didn't learn about Torch Memory Snapshot for the sole purpose of analyzing memory leaks. We had been gradually using it for about a month to solve an FSDP2 issue. Returning to our previous article, [FSDP in verl](../../rlhf/sys-design/readme-2-en.md#fsdp-in-verl), we mentioned that, intuitively, switching from FSDP1 to FSDP2 should be straightforward, only requiring a change of four lines of configuration:

```bash
actor_rollout_ref.ref.strategy=fsdp2
actor_rollout_ref.actor.strategy=fsdp2
critic.strategy=fsdp2
reward_model.strategy=fsdp2
```

Unfortunately, we were surprised to find that when the FSDP1 script was moved to FSDP2, it would consistently result in an OOM (Out-of-Memory) error. Even more surprisingly, when we gave our OOM script to the verl team and the PyTorch engineers responsible for FSDP2, they found that the 8B model wouldn't OOM, but the 3B model consistently did. After a lot of troubleshooting, the issue was finally resolved by using `set_expandable_segments(True)`. The related PR can be found at [3020](https://github.com/volcengine/verl/pull/3020).

<details>
<summary>Expandable Segments Mechanism</summary>

`set_expandable_segments(True)` enables CUDA's expandable memory segment feature, which allows PyTorch to manage GPU memory more flexibly. Memory allocation on PyTorch's CUDA backend is primarily managed by the **CUDA caching allocator**. The allocator doesn't immediately return freed memory to the operating system; instead, it keeps it in an internal memory pool so that subsequent memory requests can be fulfilled quickly. This mechanism improves performance by reducing interactions with the CUDA API. The memory pool is essentially described by two concepts: **segment** and **block**.

1.  **Segments**: Segments are large, contiguous blocks of memory that PyTorch requests from the CUDA driver. These segments are the minimum unit of memory allocation, and all PyTorch tensors and data are stored within them. The sum of all allocated segments is what's referred to as **Reserved Memory**.
2.  **Blocks**: Each memory segment contains many smaller memory blocks. When PyTorch needs to allocate memory, it looks for a suitable free block within an existing segment. If it can't find one, it tries to request a new segment from the CUDA driver. The sum of all allocated blocks is the **Allocated Memory**.

By default, when PyTorch's caching allocator cannot find a large enough free block within the existing memory segments, it requests a new memory segment from the CUDA driver. The size of this new segment is dynamically determined based on the current memory requirements. However, this dynamic expansion mechanism can lead to memory fragmentation. Blocks left over from previously allocated segments may remain unused for a long time, especially when the PyTorch memory allocator frequently releases and requests large chunks of memory.

In FSDP, which defaults to the Zero3 strategy, all-gather operations are required during both the forward and backward passes. Each GPU node temporarily aggregates parameter shards from other nodes, which creates a large number of temporary tensors and significantly increases the demand for contiguous memory. In a traditional memory management model, if the caching allocator can't find a sufficiently large contiguous memory block to accommodate these large temporary tensors, it will immediately OOM. This happens even if the GPU has available memory, because the memory is fragmented and there isn't enough contiguous space for the new tensors.

`torch.cuda.memory._set_allocator_settings("expandable_segments:True")` switches PyTorch's memory management to a more flexible mode. When this feature is enabled, the caching allocator no longer just requests a completely new segment from the CUDA driver when it needs larger contiguous memory; it tries to expand an existing memory segment instead. This expansion mechanism allows PyTorch to rearrange its memory layout, expanding or merging scattered free memory blocks into larger contiguous blocks to satisfy the allocation needs of large temporary tensors.

</details>

In short, our analysis of Torch Memory Snapshot was actually something we learned from the FSDP2 OOM issue, which paved the way for us to solve the GPU memory leaks during RL training.

## How to Monitor GPU Memory Usage

With that context, let's now dive into how to use Torch Memory Snapshot to analyze memory leak issues.

### `torch.cuda.memory_summary`

Before we introduce memory snapshots, let's look at the simplest way to check GPU memory:

<details>
<summary>The simplest way to check GPU memory</summary>

```python
    @DynamicGradMode()
    def event_loop_overlap(self):
        """A scheduler loop that overlaps the CPU processing and GPU computation."""
        self.result_queue = deque()

        # Initialize memory log file
        if not hasattr(self, "_memory_log_file"):
            import datetime

            start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._memory_log_filename = f"{start_time}_memory_log.txt"
            self._memory_log_file = open(self._memory_log_filename, "w")
            self._memory_log_file.write(
                "timestamp,memory_summary,memory_allocated,memory_reserved\n"
            )
            self._memory_log_file.flush()

        while True:
            current_time = time.time()
            if (
                not hasattr(self, "_last_memory_log_time")
                or current_time - self._last_memory_log_time >= 1.0
            ):
                gc.collect()
                torch.cuda.empty_cache()

                # Get memory information
                memory_summary = torch.cuda.memory_summary(
                    device=self.gpu_id, abbreviated=True
                )
                memory_allocated = torch.cuda.memory_allocated()
                memory_reserved = torch.cuda.memory_reserved()

                # Convert to MB
                memory_allocated_mb = memory_allocated / (1024 * 1024)
                memory_reserved_mb = memory_reserved / (1024 * 1024)

                # Record timestamp
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

                # Write to log file
                self._memory_log_file.write(
                    f"{timestamp},\"{memory_summary.replace(',', ';')}\",{memory_allocated_mb:.2f},{memory_reserved_mb:.2f}\n"
                )
                self._memory_log_file.flush()

                # Update time record
                self._last_memory_log_time = current_time

                # Also print to console (optional)
                print(f"[{timestamp}] Memory allocated: {memory_allocated_mb:.2f} MB")
                print(f"[{timestamp}] Memory reserved: {memory_reserved_mb:.2f} MB")
```

</details>

This is a set of code we used to troubleshoot SGLang memory leaks in this [commit](https://github.com/sgl-project/sglang/pull/9071/files#diff-c3b8cc39d10c245933a25aa9c2fd6397f6b31ed8d85c0ecbb926c1f42afdd178), which collects and prints memory logs. In short, the logic of this code is:

1.  Every 1 second, it prints the GPU memory usage using `torch.cuda.memory_summary`, `torch.cuda.memory_allocated`, and `torch.cuda.memory_reserved`.
2.  Every 1 second, it frees up GPU memory using `gc.collect` and `torch.cuda.empty_cache`.

Let's put `gc.collect` and `torch.cuda.empty_cache` aside for a moment and look at the output of `torch.cuda.memory_summary`, `torch.cuda.memory_allocated`, and `torch.cuda.memory_reserved`:

<details>
<summary>Output of `torch.cuda.memory_summary`</summary>

```bash
|===========================================================================|
|                  PyTorch CUDA memory summary, device ID 0                 |
|---------------------------------------------------------------------------|
|            CUDA OOMs: 0            |        cudaMalloc retries: 0         |
|===========================================================================|
|        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  |
|---------------------------------------------------------------------------|
| Allocated memory      | 184648 KiB | 192833 KiB | 258378 KiB |  73729 KiB |
|       from large pool | 184576 KiB | 192768 KiB | 254208 KiB |  69632 KiB |
|       from small pool |     72 KiB |   1060 KiB |   4170 KiB |   4097 KiB |
|---------------------------------------------------------------------------|
| Active memory         | 184648 KiB | 192833 KiB | 258378 KiB |  73729 KiB |
|       from large pool | 184576 KiB | 192768 KiB | 254208 KiB |  69632 KiB |
|       from small pool |     72 KiB |   1060 KiB |   4170 KiB |   4097 KiB |
|---------------------------------------------------------------------------|
| Requested memory      | 184648 KiB | 192832 KiB | 258376 KiB |  73728 KiB |
|       from large pool | 184576 KiB | 192768 KiB | 254208 KiB |  69632 KiB |
|       from small pool |     72 KiB |   1060 KiB |   4168 KiB |   4096 KiB |
|---------------------------------------------------------------------------|
| GPU reserved memory   | 235520 KiB | 235520 KiB | 235520 KiB |      0 B   |
|       from large pool | 233472 KiB | 233472 KiB | 233472 KiB |      0 B   |
|       from small pool |   2048 KiB |   2048 KiB |   2048 KiB |      0 B   |
|---------------------------------------------------------------------------|
| Non-releasable memory |  30391 KiB |  38607 KiB | 132985 KiB | 102594 KiB |
|       from large pool |  28416 KiB |  36608 KiB | 126848 KiB |  98432 KiB |
|       from small pool |   1975 KiB |   2040 KiB |   6137 KiB |   4162 KiB |
|---------------------------------------------------------------------------|
| Allocations           |      21    |      23    |      42    |      21    |
|       from large pool |      12    |      14    |      26    |      14    |
|       from small pool |       9    |      10    |      16    |       7    |
|---------------------------------------------------------------------------|
| Active allocs         |      21    |      23    |      42    |      21    |
|       from large pool |      12    |      14    |      26    |      14    |
|       from small pool |       9    |      10    |      16    |       7    |
|---------------------------------------------------------------------------|
| GPU reserved segments |      10    |      10    |      10    |       0    |
|       from large pool |       9    |       9    |       9    |       0    |
|       from small pool |       1    |       1    |       1    |       0    |
|---------------------------------------------------------------------------|
| Non-releasable allocs |       6    |       6    |      13    |       7    |
|       from large pool |       4    |       5    |      11    |       7    |
|       from small pool |       2    |       2    |       2    |       0    |
|---------------------------------------------------------------------------|
| Oversize allocations  |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| Oversize GPU segments |       0    |       0    |       0    |       0    |
|===========================================================================|
```

</details>

The output looks like a lot, but if you read it carefully, the content is very simple and directly gives the GPU memory usage for **all processes on the corresponding GPU rank**. I will repeatedly emphasize the concept of "process" in this section, and you will eventually understand that the method for obtaining memory usage is highly influenced by the process.

Also, let's recall that for SGLang, if a main process directly initializes an SGLang Engine, it actually results in **three processes**. The main process that initializes the engine gets an Engine instance and the Tokenizer Manager; the second process initializes the SGLang Scheduler, which is the core process that occupies the vast majority of memory; the last process initializes the SGLang Detokenizer Manager.

Note that while our code above is added directly to the SGLang scheduler layer for memory monitoring, `torch.cuda.memory_summary` actually monitors the memory usage of the **entire rank**. `torch.cuda.memory_summary` does not distinguish between processes, while `torch.cuda.memory._dump_snapshot()` only provides the memory usage of the **current process**.

This might not sound like a big deal, but the situation is very different in an RL scenario. Taking verl as an example, the verl FSDP worker's process initializes the SGLang Engine, so the FSDP worker and the SGLang Scheduler are not in the same process. If we continuously monitor memory on the FSDP worker using `torch.cuda.memory._dump_snapshot()`, we can only monitor the FSDP worker's memory usage and not the SGLang Scheduler's. This is precisely why we made no progress for a long time when we were troubleshooting memory leaks during RL training—we simply weren't monitoring the SGLang Scheduler's memory leaks.

### `torch.cuda.memory._dump_snapshot`

Now that we've covered `torch.cuda.memory_summary`, let's look at the output of `torch.cuda.memory._dump_snapshot`. Although `memory_summary` seems more global, as it directly monitors all processes on a rank, you can see that it doesn't give us detailed memory management information. For example, `memory_summary` tells us that the reserved memory is 144GB, but we have no way of knowing which processes or which tensors are actually occupying how much of it. `torch.cuda.memory._dump_snapshot` is designed for this exact need. We can directly get the creation, usage, and destruction of every tensor within the monitoring scope of the current process. Let's look at how to use it:

<details>
<summary>How to use `torch.cuda.memory._dump_snapshot`</summary>

```python
def enable_memory_visualize(
    trace_alloc_max_entries: int = 200_000,
    stack_depth: int = 32,
    context: str = "all",
    stacks: str = "all",
    devices=None,
    record_context: bool = True,
):
    """
    Enables memory history recording for CUDA allocations. This function
    should be called before any large-scale CUDA allocations. For DDP or
    multi-process setups, it must be called on each rank.

    Args:
        trace_alloc_max_entries (int): Maximum number of allocation entries
            to record.
        stack_depth (int): The depth of the call stack to capture for each
            allocation. (Supported by some PyTorch versions).
        context (str): The type of memory events to record.
            'alloc': records only allocation events.
            'state': records memory state changes.
            'all': records both.
        stacks (str): The type of call stacks to record.
            'python': records Python stacks.
            'cpp': records C++ stacks (available in some versions).
            'all': records both.
        devices (Union[int, list[int], None]): The device for which to enable
            memory history. `None` enables it for the current default device.
        record_context (bool): Whether to record context information for
            allocations. Required by older PyTorch versions.
    """
    # Memory history recording is CUDA-specific functionality
    if not is_cuda_available:
        logger.warning("[memory_visualize] Memory history recording is only available on CUDA devices")
        return

    f = get_torch_device().memory._record_memory_history
    params = set(inspect.signature(f).parameters.keys())

    def _one_call(dev_kw=None):
        kwargs = {}
        if "context" in params:
            kwargs["context"] = context
        if "stacks" in params:
            kwargs["stacks"] = stacks
        if "max_entries" in params:
            kwargs["max_entries"] = trace_alloc_max_entries
        elif "trace_alloc_max_entries" in params:
            kwargs["trace_alloc_max_entries"] = trace_alloc_max_entries
        if "stack_depth" in params:
            kwargs["stack_depth"] = stack_depth
        if dev_kw is not None:
            if "device" in params:
                kwargs["device"] = dev_kw
            elif "devices" in params:
                kwargs["devices"] = dev_kw if isinstance(dev_kw, list) else [dev_kw]
        if "record_context" in params:
            kwargs["record_context"] = record_context

        try:
            f(**kwargs)
            return "native", kwargs
        except TypeError:
            try:
                if "trace_alloc_max_entries" in params and "record_context" in params:
                    f(enabled=True, trace_alloc_max_entries=trace_alloc_max_entries, record_context=True)
                    return "legacy", {
                        "enabled": True,
                        "trace_alloc_max_entries": trace_alloc_max_entries,
                        "record_context": True,
                    }
                else:
                    f(enabled=True)
                    return "legacy-min", {"enabled": True}
            except Exception:
                raise

    if devices is None or isinstance(devices, str | int | torch.device):
        mode, used = _one_call(devices if devices is not None else None)
    else:
        mode, used = "multi-device", {}
        for d in list(devices):
            _mode, _used = _one_call(d)
            used[f"dev{d}"] = _used

    device = get_torch_device()
    if device.is_available():
        device.reset_peak_memory_stats()
        device.synchronize()

    rank = int(os.environ.get("RANK", "0") or 0)
    logger.info(f"[memory_visualize][rank {rank}] recording enabled ({mode}); args={used}")


class MemorySnapshotSampler:
    """
    A utility class that dumps GPU memory snapshots.
    This is useful for monitoring memory usage over a long-running process.

    The dumped files can be visualized with https://docs.pytorch.org/memory_viz

    Args:
        out_dir (str): The directory where the snapshots will be saved.
        tag (str): A tag for the snapshot filenames.
    """

    def __init__(self, out_dir: str = "./mem_snapshots", tag: str = "periodic"):
        self.out_dir = out_dir
        self.tag = tag

    def dump_memory_snapshot(self, out_dir: str = "./mem_snapshots", tag: str = "snapshot", sub_dir: str = None):
        """
        Generates a memory snapshot and saves it as a pickle file in a specified directory.
        The files are organized by timestamp in subdirectories, with all ranks' files
        placed in the same timestamp subdirectory.

        Args:
            out_dir (str): The directory where the snapshot file will be saved.
                The directory is created if it does not exist.
            tag (str): A string tag to prepend to the filename for easier identification.
            sub_dir (str): A subdirectory to place the snapshot file in.
        """
        if sub_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M")
            out_path = Path(out_dir) / timestamp
        else:
            out_path = Path(out_dir) / sub_dir
        out_path.mkdir(parents=True, exist_ok=True)

        # get the GPU rank on the current process
        rank = os.environ.get("RANK", "0")
        pid = os.getpid()
        # todo(chenyang): check wether we need to sync all ranks before dump
        fname = f"{tag}_rank{rank}_pid{pid}.pickle"
        path = out_path / fname

        device = get_torch_device()
        if not device.is_available():
            logger.warning("[memory_visualize] is only available on CUDA devices.")
            return
        try:
            device.synchronize()
            # Memory snapshot is CUDA-specific functionality
            device.memory._dump_snapshot(str(path))
            logger.info(f"[memory_visualize] dumped: {path}")
        except Exception as e:
            logger.info(f"[memory_visualize][warn] dump failed: {e}")
```

</details>

The functions above are excerpts from our PR to verl [3099](https://github.com/volcengine/verl/pull/3099). The code looks complex, but it does something very simple. We can think of `torch.cuda.memory._dump_snapshot` as a **video recorder**. The `enable_memory_visualize` function turns on the recording, and each call to `MemorySnapshotSampler.dump_memory_snapshot` saves the recorded content locally. Naturally, the longer we monitor memory, the more memory steps we save, and the larger the file size will be. For this reason, we set `trace_alloc_max_entries` and `stack_depth` in `enable_memory_visualize` to limit the number and depth of the memory traces to save. Every time `MemorySnapshotSampler.dump_memory_snapshot` is called, it saves the creation, usage, and destruction of all tensors within the current monitoring scope. Also, if we enable `enable_memory_visualize` too late, some tensors created before that time will not have their creation, usage, and destruction information monitored.

Thus, the usage of `torch.cuda.memory._dump_snapshot` is also very clear: it monitors the creation, usage, and destruction of all tensors within the monitoring scope of the current process on the current rank. We end up with a number of pickle files, which we can then upload to PyTorch's official [memory viz](https://pytorch.org/memory_viz) website to see a very intuitive visualization of memory usage.

This way, we get a very detailed view of memory usage. Here are two of my most frequently used visualization results:

1.  **Active Memory Timeline**

<img src="./pics/active-memory-timeline.png" alt="Active Memory Timeline" width="50%">

This chart has a lot of detail. First, we can observe the overall memory peak, which is roughly around 25GB. Additionally, we can clearly see many stages during our entire recording phase. Let me zoom in on a small part to look at a specific spike:

<img src="./pics/forward-1.png" alt="Active Memory Timeline" width="50%">

Observing this spike, we can check the stack below to see when the memory was allocated, the allocation process, and the specific size. Here, we can observe that the spike I've pointed to with an arrow actually comes from the verl FSDP forward pass. The specific stack is confidential and can't be disclosed.

A very interesting observation is that similar or identical memory blocks behave quite consistently when a memory snapshot is taken at different stages. For example, they have the same color, relative position, and size. For instance, we recorded a memory snapshot at the end of each training step in verl in [`examples/grpo_trainer/run_qwen2_5_vl-7b-sglang.sh`](https://www.google.com/search?q=%5Bhttps://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_qwen2_5_vl-7b-sglang.sh%5D\(https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_qwen2_5_vl-7b-sglang.sh\)). We observed the memory stack at the end of steps 2, 3, and 4 and got the following three images:

<img src="./pics/step-2.png" alt="step-2" width="50%">
<img src="./pics/step-3.png" alt="step-3" width="50%">
<img src="./pics/step-4.png" alt="step-4" width="50%">

Let's look at step 2. We can see three large, contiguous memory blocks at 7.2GB, 7.6GB, and 7.8GB, each 512MB in size (checking the stack, these are actually optimizer states). Then, at step 3, the 512MB memory block at 7.2GB is still in the exact same spot, but the one at 7.6GB in step 2 has moved to 8.6GB. By step 4, this 512MB memory block has moved above 9.6GB. Based on our experience, these two memory blocks don't shift, but the very scattered memory blocks in between are the leaked content. Let's look at the stack specifically:

<details>
<summary>Stack for memory fragments</summary>

```bash
/usr/local/lib/python3.10/dist-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py:278:_preprocess
??:0:PyMethod_New
/usr/local/lib/python3.10/dist-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py:173:_preprocess_image_like_inputs
??:0:PyMethod_New
/usr/local/lib/python3.10/dist-packages/transformers/image_processing_utils_fast.py:659:preprocess
??:0:PyMethod_New
/usr/local/lib/python3.10/dist-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py:151:preprocess
??:0:PyMethod_New
/usr/local/lib/python3.10/dist-packages/transformers/image_processing_utils_fast.py:623:call
??:0:PyInit__datetime
/usr/local/lib/python3.10/dist-packages/transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py:150:call
??:0:PyMethod_New
/usr/local/lib/python3.10/dist-packages/sglang/srt/multimodal/processors/base_processor.py:218:process_mm_data
??:0:PyMethod_New
/usr/local/lib/python3.10/dist-packages/sglang/srt/multimodal/processors/base_processor.py:540:_process_and_collect_mm_items
??:0:PyMethod_New
/usr/local/lib/python3.10/dist-packages/sglang/srt/multimodal/processors/base_processor.py:597:process_and_combine_mm_data
/usr/local/lib/python3.10/dist-packages/sglang/srt/multimodal/processors/qwen_vl.py:251:process_mm_data_async
/usr/local/lib/python3.10/dist-packages/sglang/srt/managers/tokenizer_manager.py:535:_tokenize_one_request
/usr/local/lib/python3.10/dist-packages/sglang/srt/managers/tokenizer_manager.py:832:_handle_batch_request
??:0:_PyUnicode_IsWhitespace
??:0:PyIter_Send
/usr/local/lib/python3.10/dist-packages/sglang/srt/managers/tokenizer_manager.py:486:generate_request

```

</details>

Clearly, we found the source of these fragments: the Qwen VL fast tokenizer is leaking.

Based on our discussion, you should now have some experience using `torch.cuda.memory._dump_snapshot`. We also used this information to upgrade the [SGLang version](https://github.com/volcengine/verl/pull/3183), which prevented the memory leak in the image processor.

2.  **Allocator State History**

Let's continue to the second visualization method, Allocator State History, which is slightly different from the Active Memory Timeline. We can see the specific memory status of the current process after each recorded event. As shown below:

<img src="./pics/stack.png" alt="step-4" width="50%">

The multicolored bars represent the actually allocated memory. Hovering over them shows the specific allocation time and line number, for example:

<details>
<summary>My previously mentioned optimizer state</summary>

```bash
b7f1ce3742000_0 518.8MiB (543956992 bytes) allocation (stream 0)
CUDACachingAllocator.cpp:0:c10::cuda::CUDACachingAllocator::Native::DeviceCachingAllocator::malloc(signed char, unsigned long, CUstream_st*)
python_torch_functions_0.cpp:0:torch::autograd::THPVariable_zeros_like(_object*, _object*, _object*)
/usr/local/lib/python3.10/dist-packages/torch/optim/adam.py:180:_init_group
/usr/local/lib/python3.10/dist-packages/torch/_dynamo/eval_frame.py:838:_fn
/usr/local/lib/python3.10/dist-packages/torch/optim/adam.py:236:step
/usr/local/lib/python3.10/dist-packages/torch/optim/optimizer.py:79:_use_grad
/usr/local/lib/python3.10/dist-packages/torch/optim/optimizer.py:485:wrapper
??:0:PyMethod_New
/usr/local/lib/python3.10/dist-packages/torch/optim/lr_scheduler.py:124:wrapper
/usr/local/lib/python3.10/dist-packages/verl/workers/actor/dp_actor.py:301:_optimizer_step
/usr/local/lib/python3.10/dist-packages/verl/workers/actor/dp_actor.py:496:update_policy
/usr/local/lib/python3.10/dist-packages/verl/utils/profiler/performance.py:118:log
??:0:PyMethod_New
/usr/local/lib/python3.10/dist-packages/verl/utils/profiler/performance.py:105:f
??:0:PyMethod_New
/usr/local/lib/python3.10/dist-packages/verl/workers/fsdp_workers.py:733:update_actor
/usr/local/lib/python3.10/dist-packages/verl/utils/profiler/nvtx_profile.py:180:wrapper
/usr/local/lib/python3.10/dist-packages/verl/single_controller/base/decorator.py:514:inner
??:0:PyMethod_New
/usr/local/lib/python3.10/dist-packages/verl/single_controller/ray/base.py:720:func
/usr/local/lib/python3.10/dist-packages/ray/util/tracing/tracing_helper.py:463:_resume_span
/usr/local/lib/python3.10/dist-packages/ray/_private/function_manager.py:689:actor_method_executor
_raylet.cpp:0:__pyx_pw_3ray_7_raylet_12execute_task_3function_executor(_object*, _object*, _object*)
```

</details>

The white blocks are **segments**, which, as we mentioned at the beginning, are reserved but not yet allocated memory. The more and more fragmented the segments are, the more severe the memory fragmentation, and the more likely you are to OOM.

-----

## Where Exactly Was the Memory Leaking?

**First, after we bumped the SGLang version, SGLang no longer has memory leak issues for either VLM or LLM. You can use SGLang-verl with confidence. You can refer to [our guide](https://www.google.com/search?q=https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/release_log/latest_sglang.md) to enable it quickly.**

That said, I'd still like to share the specific cause of the leak:

The leak was actually in the **image processor** during the Rollout process. There was some fragmentation or a leak, and since our training scenario at the company is very complex, this, coupled with FSDP fragmentation, occasionally led to OOM issues. This brings us back to the code snippet I provided at the beginning:

<details>
<summary>Code for per-second memory cleanup on the SGLang Scheduler</summary>

```python
    @DynamicGradMode()
    def event_loop_overlap(self):
        """A scheduler loop that overlaps the CPU processing and GPU computation."""
        self.result_queue = deque()
        while True:
            current_time = time.time()
            if (
                not hasattr(self, "_last_memory_log_time")
                or current_time - self._last_memory_log_time >= 1.0
            ):
                gc.collect()
                torch.cuda.empty_cache()
```

</details>

I manually added the call to `gc.collect` and `torch.cuda.empty_cache` every second. Let's see what happens without it. The specific experimental data is in [PR 9071](https://github.com/sgl-project/sglang/pull/9071).

When I enabled the per-second memory cleanup with a high-intensity instruction load:

```bash
python -m sglang.bench_serving \
    --backend sglang-oai-chat \
    --dataset-name random-image \
    --num-prompts 500 \
    --random-image-num-images 3 \
    --random-image-resolution 720p \
    --random-input-len 512 \
    --random-output-len 512
```

We get the following curve showing memory usage over time:

<img src="./pics/with-gc.png" alt="Active Memory Timeline" width="50%">

We can see that because 500 requests were sent at the same time, the total memory on the rank suddenly increased by 30GB. This is reasonable because SGLang's `mem_static` parameter does not control the memory usage of the VLM's image processor; the memory usage set for VLM is meant to be lower than for LLM. Also, the image processor must process the images into tensors, which naturally occupies a lot of memory. What's really noteworthy is that after all the requests are processed, both the reserved and allocated memory return to around 145GB.

> Why 145GB? Because I used a B200, haha\! It's the first time I've ever gotten my hands on a B-card, but unfortunately, installing SGLang was a bit tricky, and I haven't tried running RL on it yet.

In any case, we can observe that with per-second memory cleanup, there are indeed no leaks. However, when I removed the memory cleanup part, the situation immediately changed:

<img src="./pics/without-gc.png" alt="Active Memory Timeline" width="50%">

Unfortunately, our **allocated** memory was cleaned up, but the **reserved** memory kept growing. This is what I mentioned earlier: the segments are being divided more and more, becoming more fragmented. Even though the blocks aren't growing, the memory in the Rollout phase is highly fragmented. In our company's complex multi-turn business, a single rollout request can contain multiple images. These fragments eventually prevent SGLang from allocating large, contiguous blocks of memory, leading to an OOM error during the rollout phase.

At this point, it sounds like a serious problem, but after talking with colleagues in the inference engine space, they mentioned it's normal for image processors, which aren't controlled by the inference engine, to have this kind of fragmentation or leaking. Our method of periodically cleaning up memory on the Scheduler is considered reasonable and common. Unfortunately, my proposed solution above of cleaning up every second comes with a significant performance penalty. If your RL training is also facing similar VLM memory leak issues, I believe there are a few solutions:

1.  Add a periodic memory cleanup mechanism, such as cleaning up every 10 seconds or every 10 requests.
2.  Directly lower the `mem_static` of the rollout engine, for example, from our typical 0.85 to 0.65.

The reason for solution 1 is already clearly explained by my graphs above. The reason for solution 2 is also worth noting: SGLang itself doesn't manage the image processor's memory, so SGLang's recommended `mem_static` parameter for VLM is lower than for LLM. For LLM, we usually set `mem_static` to 0.85, while for VLM, the recommendation might be 0.8. On the other hand, if you're using an SPMD strategy like verl for rollout, the `requests/worker` can be calculated (`train batch size * grpo group size / num workers`). If our calculation shows that `requests/worker` is not high to begin with, say below 20, then setting a larger `mem_static` and having more KV cache space doesn't significantly impact inference performance. Of course, for a more decoupled design like slime, the `requests/worker` on each rollout worker is not fixed, but it's still roughly the same, and we can estimate the average number of requests each worker handles.

Finally, I had suspected rollout fragmentation for a long time. I even referred to the `aggressive_empty_cache` function written by the verl team and submitted [PR 3136](https://github.com/volcengine/verl/pull/3136) for SGLang.

<details>
<summary>Implementation of `aggressive_empty_cache`</summary>

```python
def aggressive_empty_cache(force_sync: bool = True, max_retries: int = 3) -> None:
    """
    More aggressive GPU memory cleanup function, tries to release PyTorch reserved
    but unallocated memory.

    Args:
        force_sync: Whether to force device synchronization
        max_retries: Maximum number of retries
    """
    device = get_torch_device()
    if not device.is_available():
        return

    for attempt in range(max_retries):
        # Record memory status before cleanup
        before_reserved = device.memory_reserved()
        before_allocated = device.memory_allocated()

        # Run garbage collection
        gc.collect()

        # Clear PyTorch cache
        device.empty_cache()

        # Force synchronization (optional)
        if force_sync:
            device.synchronize()

        # Record memory status after cleanup
        after_reserved = device.memory_reserved()
        after_allocated = device.memory_allocated()

        # Calculate freed memory
        reserved_freed = before_reserved - after_reserved
        allocated_freed = before_allocated - after_allocated

        logger.info(
            f"Memory cleanup attempt {attempt + 1}: Freed {reserved_freed / 1024**3:.2f} GB reserved, "
            f"{allocated_freed / 1024**3:.2f} GB allocated"
        )

        # Stop retrying if little memory was freed
        if reserved_freed < 1024**3:  # less than 1GB
            break
```

</details>

This function is not much different from a direct call to `gc.collect` and `torch.cuda.empty_cache`, but it performs a synchronization during cleanup, making the cleanup more thorough. The function itself is correct, but my timing for calling it was wrong. Note that my call times in PR 3136 were:

```python
  async def wake_up(self):
        aggressive_empty_cache(force_sync=True)

    @GPUMemoryLogger(role="FSDPSGLangShardingManager exit", logger=logger)
    async def sleep(self):
        aggressive_empty_cache(force_sync=True)
```

We were only cleaning up memory when `SGLang's wake_up` and `sleep` functions were called within FSDP. This is problematic. SGLang is not in the same process as FSDP, and when we call `wake_up` and `sleep`, a new garbage allocator has already been swapped in, so it can't clean up the rollout memory fragments. After realizing this, we changed our approach to clean up memory at the end of the rollout. The problem was solved immediately.
