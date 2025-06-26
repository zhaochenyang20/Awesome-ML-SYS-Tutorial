# 一文理解 CUDA Graph

前几天和标哥在研究 torch-mem-savor 的时候发现自己对 CUDA Graph 知之甚少，我只能模糊地理解到 CUDA Graph 是一种基于有向无环图的优化方式。恰好最近重新思考了下在 collcate 策略里的显存优化原理以及 SGLang 何时需要 flush cache，所以这篇文章快速理解下 CUDA Graph 的概念。

一些值得分享的文档：

- [Optimizing Memory Usage in verl](https://hebiao064.github.io/rl-memory-management)
- [How torch-memory-savor keep CUDA Graph](https://github.com/zhaochenyang20/torch_memory_saver-examples/blob/master/examples/rl_example.py)
- [When SGLang needs to flush cache](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/code-walk-through/readme.md#sglangrolloutasyncengine)

所以今天的文档尝试理解这些内容：

1. 什么是 CUDA Graph；
2. CUDA Graph 在推理中的应用；
3. 为什么 CUDA Graph 在训练中很少使用；
4. torch-memory-savor  如何保护 CUDA Graph；
5. CUDA Graph 的显存大小一般是如何决定的；
6. CUDA Graph 和 torch compile 的异同；

## 什么是 CUDA Graph

CUDA Graph 是一种 NVIDIA CUDA 编程模型中的优化技术。它将一系列独立的 GPU 操作（比如核函数启动、内存拷贝、内存设置等）定义为一个有向无环图（DAG）。这个图中的每个节点代表一个 GPU 操作，而节点之间的边则表示操作的依赖关系。

一般而言，每一个 GPU 操作都需要由 CPU 发起一个单独的启动请求（kernel launch）。这种频繁的 CPU-GPU 交互会产生一定的 CPU 开销和延迟。当我们的计算流涉及大量微小、快速的 GPU 操作时，这些启动开销就会累积起来，成为显著的性能瓶颈。

总之，CUDA Graph 将多个 GPU 操作打包成一个可执行的图。一旦这个图被定义并实例化，CPU 只需要发出一次启动指令，就可以执行整个图中的所有操作，大幅减少了 CPU 和 GPU 之间的交互开销。

此外，一旦图被创建和实例化，你可以重复多次启动它，而不需要重新定义和实例化。这对于操作序列和依赖关系不变的静态工作流尤其有利。而较新的 CUDA 版本引入了条件节点等功能，允许图的一部分有条件地或重复地执行，而无需将控制权返回给 CPU。

## 为什么推理当中大量用到 CUDA Graph

让我惊讶的是，我长期见到 SGLang 保留着 1～2G 显存给 CUDA Graph，但是我并不知道为什么训练当中从未见过 CUDA Graph。因此我们分两部分内容来讨论，先讨论为什么推理常用 CUDA Graph，再讨论为什么训练很少用。

首先，对于推理而言，推理的流程是高度确定性的。

1. 模型结构固定： 一旦模型加载，其神经网络层（如 Transformer 层、全连接层、注意力机制等）的顺序和连接方式是固定的。
2. 计算图稳定： 对于给定的输入批次大小和最大序列长度，模型执行的计算图（一系列运算操作）通常是预先确定的，不会发生运行时结构性变化。
3. 重复执行： 大多数推理请求都会重复执行相同的模型前向计算，无论是一个短请求还是一个长序列的每一步生成。

这些操作通常是串行执行的，且单个操作的执行时间可能非常短。在传统的流式执行模型中，CPU 需要为每个小操作单独发起启动请求，导致频繁的 CPU-GPU 交互，产生大量的启动开销。


## 为什么 CUDA Graph 在训练中很少使用

反过来，更重要的问题是，为什么 FSDP 这样的分布式训练框架很少直接使用 CUDA Graph？我个人觉得主要是训练的不确定性显著高于推理。

1. 优化器（AdamW、SGD）对模型参数的更新通常是动态的，并可能涉及梯度裁剪、学习率调度等逻辑，这些逻辑可能根据训练步数、梯度值等动态变化。
2. 如果使用梯度累积，每次迭代的计算图可能会有细微的变化，因为累积到一定步数后才会进行优化器更新和梯度清零。而且，美中不足，基本大家都会用梯度累计。
3. 训练过程中通常包含 dropout、随机失活等随机操作，这些操作的图行为更难被静态捕获。
4. 训练过程中需要管理正向传播的激活值以及反向传播的梯度，这涉及到大量的内存分配和释放。CUDA Graph 需要在捕获时确定所有的内存需求。如果内存需求是动态变化的（例如，由于输入序列长度变化），则难以有效地使用 CUDA Graph。

PS：第 4 点简直让我感到惊喜😂。

## torch-memory-savor 如何保护 CUDA Graph

CUDA Graph 本身不直接管理 GPU 的虚拟地址空间。CUDA Graph 关注的重点在计算依赖关系，而不是内存的实际布局。CUDA Graph 捕获一系列 GPU 操作（如核函数启动、内存拷贝、内存设置等）以及它们之间的依赖，以便这些操作能作为一个整体被提交给 GPU 执行。然而，CUDA Graph 的执行高度依赖于这些操作所引用的显存的稳定性。这意味着构建 CUDA Graph 时，所有在核函数中引用的**内存指针（GPU 虚拟地址）**都会被记录在图中。这意味着在图捕获时，这些内存区域必须已经被分配好。反过来，当 CUDA Graph 被执行时，它会使用捕获时记录下来的那些内存指针。如果这些指针指向的内存区域在图执行时被释放或重新分配了（即使是分配了同样大小的内存，也很可能得到不同的虚拟地址），那么图的执行就会失败或产生错误的结果。

这就是 [`torch-memory-savor`](https://github.com/fzyzcjy/torch_memory_saver) 要解决的问题。在 PyTorch 中，当 Tensor 的生命周期结束时，PyTorch 的默认行为是调用 `cudaFree` 来释放显存。一旦 `cudaFree` 被调用，操作系统或驱动程序就可能将这块虚拟地址空间重新分配给其他用途，而这就是 torch memory saver 的伟大之处了。

1. 调用 [CUDA 虚拟内存管理 API](https://github.com/fzyzcjy/torch_memory_saver/blob/87c124c7906a7620653cebb35e7a48b62f3de4cf/csrc/torch_memory_saver.cpp#L135)，而不是简单的内存池：

```cpp
// CUDA内存分配函数 - 使用CUDA Driver API进行精细内存管理
cudaError_t malloc(void **ptr, size_t size, const std::string& tag) {
    // 声明CUDA设备变量，用于存储当前设备ID
    CUdevice device;
    // 获取当前CUDA上下文的设备ID，用于后续在指定设备上分配内存
    CURESULT_CHECK(cuCtxGetDevice(&device));

    // 声明物理内存分配句柄，用于管理GPU物理内存
    CUmemGenericAllocationHandle allocHandle;
    // 步骤1: 在指定GPU设备上创建指定大小的物理内存
    // 这一步分配了实际的GPU物理内存空间
    CUDAUtils::cu_mem_create(&allocHandle, size, device);
    
    // 步骤2: 在GPU的虚拟地址空间中预留一段连续地址
    // 参数说明: ptr(返回地址), size(大小), 0(对齐), 0(起始地址), 0(标志)
    CURESULT_CHECK(cuMemAddressReserve((CUdeviceptr *) ptr, size, 0, 0, 0));
    
    // 步骤3: 将物理内存映射到之前预留的虚拟地址空间
    // 建立物理内存和虚拟地址之间的映射关系
    CURESULT_CHECK(cuMemMap((CUdeviceptr) * ptr, size, 0, allocHandle, 0));
    
    // 设置指定设备对这段内存的访问权限
    // 确保当前GPU设备可以正常访问这块内存
    CUDAUtils::cu_mem_set_access(*ptr, size, device);

    // 使用RAII锁保护共享数据结构，确保线程安全
    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        // 步骤4: 将内存分配信息记录到元数据映射表中
        // 包含: 内存大小、设备ID、物理内存句柄、用户标签
        allocation_metadata_.emplace(*ptr, _AllocationMetadata{size, device, allocHandle, tag});
    }

    // 条件编译的调试日志 - 仅在DEBUG模式下输出
#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.cuda_malloc "
              << " ptr=" << ptr << " *ptr=" << *ptr << " size=" << size
              << " allocHandle=" << allocHandle << " tag=" << tag
              << std::endl;
#endif

    // 返回成功状态码，表示内存分配完成
    return cudaSuccess;
}
```

2. 虚拟地址与物理内存分离管理：

- 分配阶段 ([malloc](https://github.com/fzyzcjy/torch_memory_saver/blob/87c124c7906a7620653cebb35e7a48b62f3de4cf/csrc/torch_memory_saver.cpp#L277))：

```cpp
// 使用 CUDA Driver API 进行细致的内存管理
cudaError_t malloc(void **ptr, size_t size, const std::string& tag) {
    // 声明 CUDA 设备变量，用于存储当前设备 ID
    CUdevice device;
    // 获取当前 CUDA 上下文的设备 ID，用于后续在指定设备上分配内存
    CURESULT_CHECK(cuCtxGetDevice(&device));

    // 声明物理内存分配句柄，用于管理 GPU 物理内存
    CUmemGenericAllocationHandle allocHandle;

    // 在指定 GPU 设备上创建指定大小的物理内存
    // 这一步分配了实际的 GPU 物理内存空间
    CUDAUtils::cu_mem_create(&allocHandle, size, device);
    
    // 在 GPU 的虚拟地址空间中预留一段连续地址
    CURESULT_CHECK(cuMemAddressReserve((CUdeviceptr *) ptr, size, 0, 0, 0));
    
    // 将物理内存映射到之前预留的虚拟地址空间
    CURESULT_CHECK(cuMemMap((CUdeviceptr) * ptr, size, 0, allocHandle, 0));
    
    // 设置指定设备对这段内存的访问权限
    CUDAUtils::cu_mem_set_access(*ptr, size, device);

    // 使用 RAII 锁保护共享数据结构，确保线程安全
    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        // 将内存分配信息记录到元数据映射表中
        allocation_metadata_.emplace(*ptr, _AllocationMetadata{size, device, allocHandle, tag});
    }
    return cudaSuccess;
}
```

- 暂停阶段 ([pause](https://github.com/fzyzcjy/torch_memory_saver/blob/87c124c7906a7620653cebb35e7a48b62f3de4cf/csrc/torch_memory_saver.cpp#L177))：

```cpp
void pause(const std::string& tag) {
    // 和分配时一样，使用 RAII 锁保护共享数据结构，确保线程安全
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    // 遍历所有已分配的内存元数据
    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        void *ptr = it->first; // 获取内存指针
        _AllocationMetadata metadata = it->second; // 获取对应的元数据信息

        // 如果指定了 tag，只处理与 tag 匹配的内存块
        // 对 SGLang 而言，我们目前只有 kv cache 和 model weights 的 tags
        if (!tag.empty() && metadata.tag != tag) {
            continue; // 跳过不匹配的内存块
        }

        // 解除物理内存与虚拟地址的映射关系
        CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.size));
        // 释放物理内存句柄，回收 GPU 物理内存，但是虚拟地址空间保留！
        CURESULT_CHECK(cuMemRelease(metadata.allocHandle));
    }
}
```

- 恢复阶段 ([resume](https://github.com/fzyzcjy/torch_memory_saver/blob/87c124c7906a7620653cebb35e7a48b62f3de4cf/csrc/torch_memory_saver.cpp#L200))：

```cpp
void resume(const std::string& tag) {
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        void *ptr = it->first; 
        _AllocationMetadata &metadata = it->second;

        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }

        // 创建新的物理内存句柄
        CUmemGenericAllocationHandle newAllocHandle;
        CUDAUtils::cu_mem_create(&newAllocHandle, metadata.size, metadata.device);

        // 将新物理内存映射到之前保留的虚拟地址
        CURESULT_CHECK(cuMemMap((CUdeviceptr) ptr, metadata.size, 0, newAllocHandle, 0));

        // 设置内存访问权限
        CUDAUtils::cu_mem_set_access(ptr, metadata.size, metadata.device);

        // 更新元数据中的物理内存句柄为新分配的句柄
        metadata.allocHandle = newAllocHandle;
    }
}
```

3. API 劫持：

劫持 CUDA 运行时 API：

```cpp
cudaError_t cudaMalloc(void **ptr, size_t size) {
    if (RegionManager::is_interesting_region()) {
        // 在 region 内：使用自定义分配器
        return TorchMemorySaver::instance().malloc(ptr, size, RegionManager::get_current_tag());
    } else {
        // 在 region 外：使用原始 CUDA 分配器
        return APIForwarder::call_real_cuda_malloc(ptr, size);
    }
}
```

具体来说，torch-memory-saver 将自定义的 cudaMalloc 函数编译成共享库 `libtorch_memory_saver.so`，然后通过设置 `LD_PRELOAD` 环境变量，让动态链接器在程序启动时优先加载这个库。当 PyTorch 程序调用 cudaMalloc 时，动态链接器会首先找到被劫持的库中的函数而不是原始的 CUDA 运行时函数，从而拦截所有 CUDA 内存分配调用，在受到保护的内存区域内使用自定义的内存管理策略（虚拟地址与物理内存分离），而在区域外仍使用原始的内存分配逻辑。

4. Region 管理（[RegionManager](https://github.com/fzyzcjy/torch_memory_saver/blob/87c124c7906a7620653cebb35e7a48b62f3de4cf/csrc/torch_memory_saver.cpp#L244)）：

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

这种管理的用法非常直接，比如：

```python
with torch_memory_saver.region(tag="kv_cache"):
    self.kv_buffer = torch.full(dummy_tensor_size, value, dtype=torch.float32, device='cuda')
print(f'KV cache created: {_ptr(self.kv_buffer)}')
```

这段代码节选自[How torch-memory-savor keep CUDA Graph](https://github.com/zhaochenyang20/torch_memory_saver-examples/blob/master/examples/rl_example.py)。

## CUDA Graph 的显存大小一般是如何决定的

CUDA Graph 本身并不直接占用显存大小，它只是一个执行蓝图。真正占用显存的是图中所涉及的**数据**和**激活值**。因此，CUDA Graph 执行所需的显存大小，是由以下因素决定的：

1. 模型的权重、偏置等参数在整个推理或训练过程中都需要常驻显存。
2. 每次输入给模型的数据以及模型产生的输出，都会占用显存。
3. 在执行 CUDA Graph 时，所有激活值都需要有足够的空间来存储。
4. 某些算子可能需要额外的临时工作空间来执行计算。

CUDA Graph 在捕获阶段会记录所有显存操作，包括 `cudaMalloc`、`cudaMemcpy`、`cudaMemset` 等。当图被实例化并执行时，所有需要分配的显存都必须被分配。这意味着图所需要的最大显存量是在捕获时就确定下来的。如果你的模型在推理时输入的形状是动态变化的（不同的 sequence length），那么为了捕获一个可以处理所有可能形状的图，我们一般为最坏情况（最大输入长度）预留显存，这可能浪费显存。

正如在 `torch-memory-savor` 的想法，为了避免频繁的显存分配和释放开销，同时保持 CUDA Graph 的有效性，通常会使用 CUDA 内存池 (CUDA Memory Pool)。内存池会预先向驱动程序申请一大块显存，然后在其中进行细粒度的分配和回收，而不会将显存真正归还给操作系统。这意味着在图执行期间，只要总的内存需求不超过内存池的大小，就可以高效地复用显存，而不会出现虚拟地址变化导致图失效的问题。

因此，计算图的拓扑结构和数据量决定了 CUDA Graph 执行所需的显存大小。CUDA Graph 确保这些显存操作能够以最优化的方式执行，并依赖于这些显存地址的稳定性。

## CUDA Graph 和 `torch.compile` 的异同

明白了，非常感谢你的具体指导。我将按照你严谨的风格，去除比喻和过于口语化的表达，专注于技术细节和清晰的对比。

---

## CUDA Graph 和 `torch.compile` 的异同

CUDA Graph 和 `torch.compile` 均显著提升 PyTorch 模型的执行性能。然而，二者工作于不同的抽象层级，其设计目标与应用机制存在显著区别。`torch.compile` 是 PyTorch 2.0 引入的高级编译工具链。其核心目标是自动化性能优化流程，通过对 PyTorch 代码的分析，将其转换为优化的底层表示，并利用诸如 TorchInductor、AOTAutograd 等后端生成高效的 GPU 代码以实现加速。

| 特征           | CUDA Graph                                   | `torch.compile`                                           |
| :------------- | :------------------------------------------- | :-------------------------------------------------------- |
| **抽象级别** | **低级 CUDA API**：要求开发者手动进行图的捕获与管理。 | **高级 PyTorch API**：通过自动化编译过程，简化用户操作。 |
| **控制粒度** | **细粒度控制**：直接管理 GPU 操作序列，对张量形状与内存地址的稳定性有严格要求。 | **宏观优化**：在不暴露底层复杂性的前提下，实现算子融合、内存优化等，并在条件允许时利用 CUDA Graph。 |
| **动态性处理** | **严格要求静态图**：图的结构、张量形状、控制流及内存地址必须固定。任何动态变化均可能导致图失效或需要重新捕获。 | **智能处理动态性**：能够识别并处理“图中断”，允许无法静态编译的部分回退至 Eager 模式执行，并能适应动态张量形状。 |
| **主要目标** | **降低 CPU 启动开销**：通过一次性提交大量 GPU 操作，减少 CPU 与 GPU 之间频繁的交互延迟。 | **全面性能提升**：涵盖从算子融合、内存优化到图级执行等多个层面，旨在最大化 PyTorch 代码的整体执行效率。 |
| **底层实现** | 直接通过 CUDA Driver API 进行操作。          | 内部集成多组件（如 TorchDynamo、TorchInductor），**可根据编译策略在底层生成并利用 CUDA Graph**。 |
| **典型应用** | **高重复性、计算图静态的工作负载**，例如大型语言模型的推理阶段。 | **通用 PyTorch 模型**，包括训练与推理场景。它致力于在保持 PyTorch 灵活性的前提下，提供普遍的性能加速。 |

`torch.compile` 在其内部优化流程中，能够有选择性地利用 CUDA Graph。例如，在 `torch.compile` 的不同 `mode` 设置中，如 `mode="reduce-overhead"`，其优化策略会更倾向于利用 CUDA Graph 以减少 CPU 开销。对于具有固定输入形状和确定性计算行为的模型，`torch.compile` 启用 CUDA Graph 通常能带来显著的性能提升。