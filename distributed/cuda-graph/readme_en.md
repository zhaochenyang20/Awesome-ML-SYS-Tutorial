# Quick Understanding of CUDA Graph

A few days ago, while researching `torch-memory-savor` with [Biao](https://hebiao064.github.io/), I realized how little I knew about CUDA Graphs. I only vaguely understood that CUDA Graphs are an optimization method based on directed acyclic graphs. Coincidentally, I've recently re-evaluated the principles of GPU memory optimization in the `collate` strategy and when SGLang needs to flush its cache. So, this article will quickly cover the concept of CUDA Graphs.

Some valuable documents to share:

  - [Optimizing Memory Usage in verl](https://www.google.com/search?q=https://hebiao066.github.io/rl-memory-management)
  - [How torch-memory-savor keep CUDA Graph](https://github.com/zhaochenyang20/torch_memory_saver-examples/blob/master/examples/rl_example.py)
  - [When SGLang needs to flush cache](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/code-walk-through/readme.md#sglangrolloutasyncengine)

Today's document attempts to clarify these points:

1.  What is a CUDA Graph?
2.  Applications of CUDA Graphs in inference.
3.  Why CUDA Graphs are rarely used in training.
4.  How `torch-memory-savor` protects CUDA Graphs.
5.  How is the memory size of a CUDA Graph typically determined?
6.  Similarities and differences between CUDA Graph and `torch.compile`.

## What is a CUDA Graph?

A CUDA Graph is an optimization technique within the NVIDIA CUDA programming model. It defines a series of independent GPU operations (such as kernel launches, memory copies, memory sets, etc.) as a directed acyclic graph (DAG). Each node in this graph represents a GPU operation, and the edges between nodes indicate operational dependencies.

Generally, each GPU operation requires the CPU to issue a separate launch request (kernel launch). This frequent CPU-GPU interaction incurs some CPU overhead and latency. When our computational flow involves a large number of small, fast GPU operations, these launch overheads accumulate, becoming a significant performance bottleneck.

In essence, a CUDA Graph packages multiple GPU operations into a single executable graph. Once this graph is defined and instantiated, the CPU only needs to issue one launch command to execute all operations within the entire graph, significantly reducing the interaction overhead between the CPU and GPU.

Furthermore, once a graph is created and instantiated, you can launch it multiple times without redefining and instantiating it. This is especially beneficial for static workflows where the sequence of operations and dependencies remain constant. Newer CUDA versions have also introduced features like conditional nodes, allowing parts of the graph to execute conditionally or repeatedly without returning control to the CPU.

## Why CUDA Graphs are Widely Used in Inference

To my surprise, I've long seen SGLang reserving 1-2GB of memory for CUDA Graphs, but I've never encountered CUDA Graphs in training. Therefore, we will discuss this in two parts: first, why CUDA Graphs are commonly used in inference, and then, why they are rarely used in training.

First, for inference, the process is highly deterministic.

1.  **Fixed Model Structure**: Once a model is loaded, the order and connections of its neural network layers (e.g., Transformer layers, fully connected layers, attention mechanisms) are fixed.
2.  **Stable Computation Graph**: For a given input batch size and maximum sequence length, the computation graph executed by the model (a series of operations) is usually predetermined and does not undergo structural changes at runtime.
3.  **Repetitive Execution**: Most inference requests repeatedly execute the same model forward pass, whether for a short request or each step of a long sequence generation.

These operations are typically executed serially, and the execution time of a single operation can be very short. In traditional streaming execution models, the CPU needs to initiate a separate launch request for each small operation, leading to frequent CPU-GPU interactions and significant launch overhead.

-----

## Why CUDA Graphs are Rarely Used in Training

Conversely, a more important question is, why do distributed training frameworks like FSDP rarely use CUDA Graphs directly? I personally believe this is mainly because the uncertainty in training is significantly higher than in inference.

1.  **Dynamic Optimizer Updates**: Optimizers (AdamW, SGD) typically update model parameters dynamically and may involve logic such as gradient clipping and learning rate scheduling, which can change dynamically based on training steps, gradient values, etc.
2.  **Gradient Accumulation**: If gradient accumulation is used, the computation graph for each iteration might have subtle variations, as optimizer updates and gradient clearing only occur after a certain number of accumulated steps. And unfortunately, most people use gradient accumulation.
3.  **Random Operations**: Training often includes random operations like dropout, which make the graph behavior harder to capture statically.
4.  **Dynamic Memory Management**: During training, managing forward pass activations and backward pass gradients involves significant memory allocation and deallocation. CUDA Graphs require all memory needs to be determined at capture time. If memory requirements are dynamic (e.g., due to varying input sequence lengths), it becomes difficult to effectively use CUDA Graphs.

PS: Point 4 was quite a pleasant surprise to me! 😂

## How `torch-memory-savor` Protects CUDA Graphs

CUDA Graphs themselves do not directly manage the GPU's virtual address space. The primary focus of CUDA Graphs is on computational dependencies, not the actual memory layout. A CUDA Graph captures a series of GPU operations (such as kernel launches, memory copies, memory sets, etc.) and their dependencies, so that these operations can be submitted to the GPU as a single unit for execution. However, the execution of a CUDA Graph heavily relies on the stability of the memory referenced by these operations. This means that when building a CUDA Graph, all **memory pointers (GPU virtual addresses)** referenced in the kernel functions are recorded in the graph. This implies that these memory regions must already be allocated at the time of graph capture. Conversely, when a CUDA Graph is executed, it uses the memory pointers recorded during capture. If the memory regions pointed to by these pointers are freed or reallocated during graph execution (even if the same amount of memory is allocated, it is very likely to get a different virtual address), then the graph execution will fail or produce incorrect results.

This is precisely what [`torch-memory-savor`](https://www.google.com/search?q=%5Bhttps://github.com/fzyzcjy/torch_memory_saver%5D\(https://github.com/fzyzcjy/torch_memory_saver\)) aims to solve. In PyTorch, when the lifetime of a Tensor ends, PyTorch's default behavior is to call `cudaFree` to release GPU memory. Once `cudaFree` is called, the operating system or driver may reallocate that virtual address space for other purposes, and this is where `torch-memory-savor` shines.

1.  Calls the [CUDA Virtual Memory Management API](https://github.com/fzyzcjy/torch_memory_saver/blob/87c124c7906a7620653cebb35e7a48b62f3de4cf/csrc/torch_memory_saver.cpp#L135) instead of a simple memory pool:

```cpp
// CUDA memory allocation function - uses CUDA Driver API for fine-grained memory management
cudaError_t malloc(void **ptr, size_t size, const std::string& tag) {
    // Declare CUDA device variable to store the current device ID
    CUdevice device;
    // Get the device ID of the current CUDA context for subsequent memory allocation on the specified device
    CURESULT_CHECK(cuCtxGetDevice(&device));

    // Declare physical memory allocation handle for managing GPU physical memory
    CUmemGenericAllocationHandle allocHandle;
    // Step 1: Create physical memory of the specified size on the specified GPU device
    // This step allocates actual GPU physical memory space
    CUDAUtils::cu_mem_create(&allocHandle, size, device);

    // Step 2: Reserve a contiguous address range in the GPU's virtual address space
    // Parameters: ptr (returned address), size (size), 0 (alignment), 0 (start address), 0 (flags)
    CURESULT_CHECK(cuMemAddressReserve((CUdeviceptr *) ptr, size, 0, 0, 0));

    // Step 3: Map physical memory to the previously reserved virtual address space
    // Establishes the mapping relationship between physical memory and virtual address
    CURESULT_CHECK(cuMemMap((CUdeviceptr) * ptr, size, 0, allocHandle, 0));

    // Set access permissions for this memory on the specified device
    // Ensure the current GPU device can properly access this memory block
    CUDAUtils::cu_mem_set_access(*ptr, size, device);

    // Use RAII lock to protect shared data structures, ensuring thread safety
    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        // Step 4: Record memory allocation information in the metadata map
        // Includes: memory size, device ID, physical memory handle, user tag
        allocation_metadata_.emplace(*ptr, _AllocationMetadata{size, device, allocHandle, tag});
    }

    // Conditional compilation for debug logging - outputs only in DEBUG mode
#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.cuda_malloc "
                << " ptr=" << ptr << " *ptr=" << *ptr << " size=" << size
                << " allocHandle=" << allocHandle << " tag=" << tag
                << std::endl;
#endif

    // Return success status code, indicating memory allocation is complete
    return cudaSuccess;
}
```

2.  Manages virtual addresses and physical memory separately:

- Allocation stage ([malloc](https://github.com/fzyzcjy/torch_memory_saver/blob/87c124c7906a7620653cebb35e7a48b62f3de4cf/csrc/torch_memory_saver.cpp#L277)):

```cpp
// Uses CUDA Driver API for detailed memory management
cudaError_t malloc(void **ptr, size_t size, const std::string& tag) {
    // Declare CUDA device variable to store the current device ID
    CUdevice device;
    // Get the device ID of the current CUDA context for subsequent memory allocation on the specified device
    CURESULT_CHECK(cuCtxGetDevice(&device));

    // Declare physical memory allocation handle for managing GPU physical memory
    CUmemGenericAllocationHandle allocHandle;

    // Create physical memory of the specified size on the specified GPU device
    // This step allocates actual GPU physical memory space
    CUDAUtils::cu_mem_create(&allocHandle, size, device);

    // Reserve a contiguous address range in the GPU's virtual address space
    CURESULT_CHECK(cuMemAddressReserve((CUdeviceptr *) ptr, size, 0, 0, 0));

    // Map physical memory to the previously reserved virtual address space
    CURESULT_CHECK(cuMemMap((CUdeviceptr) * ptr, size, 0, allocHandle, 0));

    // Set access permissions for this memory on the specified device
    CUDAUtils::cu_mem_set_access(*ptr, size, device);

    // Use RAII lock to protect shared data structures, ensuring thread safety
    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        // Record memory allocation information in the metadata map
        allocation_metadata_.emplace(*ptr, _AllocationMetadata{size, device, allocHandle, tag});
    }
    return cudaSuccess;
}
```

- Pause stage ([pause](https://github.com/fzyzcjy/torch_memory_saver/blob/87c124c7906a7620653cebb35e7a48b62f3de4cf/csrc/torch_memory_saver.cpp#L177)):

```cpp
void pause(const std::string& tag) {
    // As with allocation, use an RAII lock to protect shared data structures, ensuring thread safety
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    // Iterate through all allocated memory metadata
    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        void *ptr = it->first; // Get memory pointer
        _AllocationMetadata metadata = it->second; // Get corresponding metadata information

        // If a tag is specified, only process memory blocks that match the tag
        // For SGLang, we currently only have tags for kv cache and model weights
        if (!tag.empty() && metadata.tag != tag) {
            continue; // Skip non-matching memory blocks
        }

        // Unmap physical memory from virtual address
        CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.size));
        // Release physical memory handle, reclaim GPU physical memory, but retain virtual address space!
        CURESULT_CHECK(cuMemRelease(metadata.allocHandle));
    }
}
```

- Resume stage ([resume](https://github.com/fzyzcjy/torch_memory_saver/blob/87c124c7906a7620653cebb35e7a48b62f3de4cf/csrc/torch_memory_saver.cpp#L200)):

```cpp
void resume(const std::string& tag) {
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        void *ptr = it->first;
        _AllocationMetadata &metadata = it->second;

        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }

        // Create new physical memory handle
        CUmemGenericAllocationHandle newAllocHandle;
        CUDAUtils::cu_mem_create(&newAllocHandle, metadata.size, metadata.device);

        // Map new physical memory to the previously reserved virtual address
        CURESULT_CHECK(cuMemMap((CUdeviceptr) ptr, metadata.size, 0, newAllocHandle, 0));

        // Set memory access permissions
        CUDAUtils::cu_mem_set_access(ptr, metadata.size, metadata.device);

        // Update physical memory handle in metadata to the newly allocated handle
        metadata.allocHandle = newAllocHandle;
    }
}
```

3.  Hijacking the CUDA runtime API:

```cpp
cudaError_t cudaMalloc(void **ptr, size_t size) {
    if (RegionManager::is_interesting_region()) {
        // Inside the region: use custom allocator
        return TorchMemorySaver::instance().malloc(ptr, size, RegionManager::get_current_tag());
    } else {
        // Outside the region: use original CUDA allocator
        return APIForwarder::call_real_cuda_malloc(ptr, size);
    }
}
```

Specifically, `torch-memory-saver` compiles its custom `cudaMalloc` function into a shared library `libtorch_memory_saver.so`. Then, by setting the `LD_PRELOAD` environment variable, it instructs the dynamic linker to load this library first when the program starts. When a PyTorch program calls `cudaMalloc`, the dynamic linker will first find the function in the hijacked library instead of the original CUDA runtime function, thereby intercepting all CUDA memory allocation calls. This allows it to use its custom memory management strategy (virtual address and physical memory separation) within protected memory regions, while still using the original memory allocation logic outside these regions.

4.  Region Management ([RegionManager](https://github.com/fzyzcjy/torch_memory_saver/blob/87c124c7906a7620653cebb35e7a48b62f3de4cf/csrc/torch_memory_saver.cpp#L244)):

```cpp
namespace RegionManager {
    static thread_local bool is_interesting_region_ = false;
    static thread_local std::string current_tag_ = "default";

    void enter() {
#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] tms_region_enter" << std::endl;
#endif
        is_interesting_region_ = true;
    }

    void leave() {
#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] tms_region_leave" << std::endl;
#endif
        is_interesting_region_ = false;
    }

    bool is_interesting_region() {
        return is_interesting_region_;
    }

    void set_current_tag(const std::string& tag) {
        current_tag_ = tag;
    }

    const std::string& get_current_tag() {
        return current_tag_;
    }
}
```

The usage of this management is very straightforward, for example:

```python
with torch_memory_saver.region(tag="kv_cache"):
    self.kv_buffer = torch.full(dummy_tensor_size, value, dtype=torch.float32, device='cuda')
print(f'KV cache created: {_ptr(self.kv_buffer)}')
```

This code snippet is taken from [How torch-memory-savor keep CUDA Graph](https://github.com/zhaochenyang20/torch_memory_saver-examples/blob/master/examples/rl_example.py).

## How GPU Memory for CUDA Graphs is Typically Determined

The amount of GPU memory a **CUDA Graph** requires during its **capture** phase is primarily determined by the operations that form the computational graph and the amount of data involved in those operations. The capture process essentially records a series of GPU operations (like kernel launches, memory copies, and memory sets) and all the memory regions referenced by these operations. Therefore, the memory consumption during capture directly reflects the **maximum instantaneous memory peak** required by the model along that specific execution path.

Here are the key factors that contribute to the GPU memory footprint:

1.  **Model Parameter Size**: All of a model's weights (e.g., linear layers, convolutional layers, attention mechanism weights) and biases must be loaded into GPU memory when the graph is captured.
2.  **Input and Output Data Size**: The batch size and sequence length of the **dummy input** used to capture the graph directly determine the memory needed for activations and intermediate variables during capture. When capturing a graph, the system executes a complete computational graph trace (e.g., a forward pass in inference, or a forward and backward pass in training) to identify all memory operations and their peak requirements. Thus, the memory occupied by the input data itself, along with the output and intermediate results it generates during computation, are all accounted for.
3.  **Intermediate Activation Size**: During graph capture, CUDA tracks the memory allocation of these activations to ensure the graph can correctly reference them during replay. The peak for activations usually occurs at the widest layers in the model or during intermediate computation stages of specific operators (like multi-head attention). For training, intermediate activations required for backpropagation (or activations needed for re-computation) are also included.
4.  **Operator Temporary Workspace**: Many high-performance GPU operator libraries (like cuBLAS, cuDNN, cuFFT, etc.) may require additional **temporary workspace memory** to store intermediate results or perform internal optimizations when executing complex computations (e.g., large matrix multiplications, convolutions). This memory is typically requested dynamically by the driver or library functions and is captured as memory operations within the graph.
5.  **Graph Management Overhead Itself**: While a CUDA Graph as a blueprint is small, its **instantiation** and management at the GPU driver level also require a small amount of GPU memory. This includes storing the graph's topology, node information, and related internal metadata. However, this overhead is typically much smaller than that caused by data and activations.

**In summary, the GPU memory occupied during CUDA Graph capture is essentially the maximum instantaneous memory required by that computational graph during its execution.** The CUDA Graph records all memory operations during the capture phase, including `cudaMalloc`, `cudaMemcpy`, `cudaMemset`, and so on. When the graph is instantiated and executed, all these recorded memory allocations must already be completed, and their addresses must remain stable. This means the maximum memory required by the graph is determined at capture time and needs to be pre-allocated. If a model's input shapes are dynamic during inference (e.g., different sequence lengths), to capture a graph that can handle all possible shapes, memory is usually reserved for the worst-case scenario (i.e., the maximum input length), which can sometimes lead to wasted memory.

Furthermore, to avoid frequent memory allocation and deallocation overhead while maintaining the effectiveness of CUDA Graphs, **CUDA Memory Pools** are commonly used. A memory pool pre-allocates a large chunk of GPU memory from the driver and then performs fine-grained allocations and deallocations within it without truly returning the memory to the operating system. This means that during graph execution, as long as the total memory demand doesn't exceed the size of the memory pool, memory can be efficiently reused without issues of virtual address changes causing the graph to become invalid.

## Similarities and Differences Between CUDA Graph and `torch.compile`

CUDA Graph and `torch.compile` both significantly enhance the execution performance of PyTorch models. However, they operate at different levels of abstraction, and their design goals and application mechanisms differ notably. `torch.compile` is a high-level compilation toolchain introduced in PyTorch 2.0. Its core objective is to automate the performance optimization process by analyzing PyTorch code, converting it into an optimized low-level representation, and utilizing backends such as TorchInductor and AOTAutograd to generate efficient GPU code for acceleration.

| Feature            | CUDA Graph                                                                                                           | `torch.compile`                                                                                                                                                                                                 |
| :----------------- | :------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Abstraction Level** | **Low-level CUDA API**: Requires developers to manually capture and manage the graph.                                | **High-level PyTorch API**: Automates the compilation process, simplifying user operation.                                                                                                             |
| **Control Granularity** | **Fine-grained control**: Directly manages GPU operation sequences, with strict requirements for tensor shape and memory address stability. | **Macro-optimization**: Achieves operator fusion, memory optimization, and other improvements without exposing underlying complexities, and can leverage CUDA Graphs when conditions allow.             |
| **Dynamic Handling** | **Strictly requires static graphs**: Graph structure, tensor shapes, control flow, and memory addresses must be fixed. Any dynamic changes may invalidate the graph or necessitate re-capture. | **Intelligent dynamic handling**: Can identify and manage "graph breaks," allowing uncompilable sections to fall back to Eager mode execution, and can adapt to dynamic tensor shapes.                     |
| **Primary Goal** | **Reduce CPU launch overhead**: Minimizes frequent CPU-GPU interaction latency by submitting a large number of GPU operations at once. | **Comprehensive performance enhancement**: Covers multiple layers, from operator fusion and memory optimization to graph-level execution, aiming to maximize the overall execution efficiency of PyTorch code. |
| **Underlying Implementation** | Directly operates via the CUDA Driver API.                                                                           | Internally integrates multiple components (e.g., TorchDynamo, TorchInductor), and **can generate and utilize CUDA Graphs at the low level based on compilation strategies**.                             |
| **Typical Application** | **Highly repetitive, static computation graph workloads**, such as the inference stage of large language models.        | **General PyTorch models**, encompassing both training and inference scenarios. It strives to provide widespread performance acceleration while maintaining PyTorch's flexibility.                    |

`torch.compile`, within its internal optimization pipeline, can selectively leverage CUDA Graphs. For example, in certain `torch.compile` `mode` settings, such as `mode="reduce-overhead"`, its optimization strategy will favor using CUDA Graphs to reduce CPU overhead. For models with fixed input shapes and deterministic computational behavior, enabling CUDA Graphs via `torch.compile` typically yields significant performance improvements.