# Unified FP8: Moving Beyond Mixed Precision for Stable and Accelerated MoE RL

> **TL;DR:**
> 
> **我们实现了在 RL 中完全使用 FP8 进行采样（Rollout）和训练（Training）。实验表明，MoE 模型规模越大，BF16 训练 FP8 采样的训推差异越明显，而统一使用 FP8 有效消除了量化误差导致的训推不一致性，提升了 RL 训练的速度和稳定性。**

SGLang RL 团队与 slime 社区近期在强化学习的训练稳定性与加速方面，做出了一些有意思的探索与工作：

- 在**训练稳定性**方面，我们通过[对齐 SGLang 与 FSDP 后端](https://github.com/THUDM/slime/tree/main/examples/true_on_policy)，在 Dense 模型上实现了 Rollout 与 Training 过程 **KL 散度严格为零**，达成了完美的训推一致。

- 在**训练加速**方面，我们将 [**Speculative Decoding**](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/slime/spec/readme-en.md) 技术引入 RL 采样流程，在合适配置下显著提升了采样速度。

在此基础上，我们进一步向大家分享兼顾稳定性与性能的新进展——**在 RL 中实现全流程的 FP8 训练与采样**。 Qwen3-4B 与 Qwen3-30B-A3B 模型的 FP8 RL 训练已[在 slime 中**全面支持**](https://github.com/THUDM/slime/tree/main/examples/low_precision)，开箱即用。

本次工作由 **InfiXAI 团队、蚂蚁集团 AQ 团队、SGLang RL 团队及 slime 团队**联合完成。特别感谢 **DataCrunch** 为本工作提供的算力赞助，以及 **NVIDIA** 在 Transformer Engine（TE）方面给予的技术支持。

## FP8 训练的硬件基础

### **Tensor Core 与低精度支持**

低精度计算是软硬件统一设计（Hardware-Software Co-Design）的掌上明珠，我们首先介绍其硬件基础 —— **Tensor Core**，一种专为**大规模矩阵乘法和累加运算**（深度学习的核心计算）设计的 **GPU 硬件加速单元**，它能以比传统 CUDA Core **更高的吞吐量**处理低精度数据格式（如 FP16、BF16、FP8）。Tensor Core 的演进始于基础的 FMA（融合乘加）指令，并借助 DP4A 指令实现了早期的向量化。而 Volta 架构的问世则带来了里程碑式的飞跃 —— 它首度引入了 Tensor Core 作为专为大规模矩阵运算而定制的硬件单元。自此以后，Ampere、Hopper 直到最新的 Blackwell 架构，都在持续深化这一理念：

- 扩大规模：让 Tensor Core 一次能处理的矩阵更大，从而提高计算访存比。

- 降低精度：不断增加对 FP/BF16、FP8 乃至更低精度数据格式的支持。

| Arch | FP64 | [F16](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell) | INT8 | INT4 | FP8 | MXFP |
| --- | --- | --- | --- | --- | --- | --- |
| Volta | ❌ | ✅ FP16 | ❌ | ❌ | ❌ | ❌ |
| Turing | ❌ | ✅ FP16 | ✅ | ✅ | ❌ | ❌ |
| Ampere | ✅ | ✅ FP16/BF16 | ✅ | ✅ | ❌ | ❌ |
| Hopper | ✅ | ✅ FP16/BF16 | ✅ | ❌ | ✅ [(累加精度只支持FP22)](https://arxiv.org/html/2505.09343v1) | ❌ |
| Blackwell | ✅ | ✅ FP16/BF16 | ✅ | ❌ | ✅ | ✅ MXFP(8/6/4)<br>NVFP4 |
| Blackwell Ultra | ✅（减少算力） | ✅ FP16/BF16 | ✅ （减少算力） | ❌ | ✅ | ✅ MXFP(8/6/4)<br>NVFP4 |

> 图表来源：[zartbot](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&chksm=f995e4e6cee26df07bf7101b58cbdfdf80d577c67122304482e3e788edfa74a71135dbf77d36&cur_album_id=3289258526057463810&scene=189#wechat_redirect)，[SemiAnalysis](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell)

> 

在这样的硬件发展趋势下，使用更低精度进行存储与计算越发诱人。具体来说，更低精度的浮点数有着许多潜在优势：

1. **显著降低显存占用**：相较于主流的 BF16 格式，FP8 理论上能将模型权重（Weights）与激活值（Activations）所占用的显存减少一半，直接缓解了日益增长的显存压力。

2. **理论上可翻倍的算力**：主流 GPU（如 H100 SXM）的 FP8 Tensor Core 提供了高达 1979 TFLOPS 的理论性能，是其 BF16 单元（989 TFLOPS）的两倍。这种巨大的算力提升，是推动业界探索 FP8 训练的核心驱动力。

3. **优化内存带宽瓶颈**：由于数据表示更紧凑，从 GPU 显存（HBM）到计算核心所需传输的数据量也随之减小。这意味着更少的数据搬运时间，从而有效降低了内存带宽带来的压力。

### **FP8 格式**

FP8 是一种采用 8 位比特进行数值表达的浮点数格式。与 FP32（32 位）、FP16/BF16（16 位）等传统格式相比，FP8 可将同规模数据的存储和传输需求分别降低至 1/4 或 1/2，极大缓解显存与带宽瓶颈，提升模型训练和推理的性能。目前，业界主要有两种主流的 FP8 格式：

- **E4M3**：4 位指数位 + 3 位尾数位。特点是动态范围较小，但精度相对较高。

- **E5M2**：5 位指数位 + 2 位尾数位。特点是动态范围更大，但精度相对较低。

<p align="center">
  <img src="./pic/1_E4vsE5.png" alt="FP8 E4M3 vs E5M2" width="80%" />
</p>

> 图表来源：[OCP白皮书](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)

> 

这种设计使得 FP8 能够在保持足够数值范围与精度的同时，最大化地利用硬件计算吞吐量。

### FP8 scale 选择
| **比较维度** | **FP32 Scale (全精度缩放因子)** | **E8M0 Scale (指数缩放因子)** |
| --- | --- | --- |
| **格式定义** | **FP32** (IEEE 754 单精度浮点) | **E8M0** (8位指数，0位尾数) |
| **数值特性** | 可以表示任意精度的实数。 | 仅支持 **2 的幂** ，如 1, 2, 0.5 等，无法表示 1.5 这类数值。 |
| **核心思想** | 用高精度管理缩放因子，确保训练过程中的数值稳定性。 | 将缩放因子纳入低精度体系，利用位运算特性提升效率。 |
| **主要优势** | 1. **高精度、训练稳定**：能准确表示动态范围，减少量化误差，防止发散。<br>2. **广泛支持**：NVIDIA Transformer Engine 等主流库的默认方式，生态成熟。 | 1. **硬件极致友好**：缩放计算可转化为简单的**位移操作**，速度极快，功耗低。<br>2. **流水线统一**：全链路（含 Scale）均在 8-bit 下运行，简化硬件设计。 |
| **主要劣势** | 1. **存储开销**：每个量化张量需额外存储一个FP32 数据，占用少量显存。<br>2. **计算开销**：Scale 的计算和转换需在 FP32 精度下进行。 | 1. **精度损失风险**：强制舍入到 2 的幂会引入量化噪声，反向传播时易累积误差导致训练发散。<br>2. **动态范围受限**：难以精细适配复杂分布的张量。 |
| **总结** | 目前业界最常用、保险的方案。 | 牺牲部分精度换取极致的硬件效率。 |

经过综合评估，我们最终决定采用 **FP32** 作为训练时的 Scale 精度。决策依据如下：

1. **精度对齐与训练稳定性**：FP32 Scale 提供了精细的数值缩放，能够捕捉 Tensor 的动态范围，确保 FP8 训练的 Loss 曲线能够最大限度地贴近 **BF16** 基线。

2.  **推理生态的一致性**：目前主流的推理模型都使用的 FP32 作为推理 scale

3. **硬件加速的实际收益**：

    - **Hopper 架构 (H100/H800)**：虽然支持 FP8 Tensor Core 计算，但并无针对 E8M0 Scale 的计算单元。

    - **Blackwell 架构 (B100/B200)**：引入了对 MXFP8 (Micro-scaling) 的支持，才真正针对 E8M0 这种块级缩放提供了硬件加速（参考论文 [**arXiv:2506.08027**](https://arxiv.org/abs/2506.08027)）。

因此，在现有的 H 卡集群环境下，强行使用 E8M0 不仅无法带来显著的加速收益，反而可能引入额外的软件模拟开销和精度风险。

### FP8 量化

常见的量化策略有：**per-tensor**（逐张量）、**per-block**（逐块）和 **per-token**（逐 Token）。无论采用哪种粒度，量化通常都遵循以下简单的两步流程：

<p align="center">
  <img src="./pic/2_size.png" alt="FP8 量化流程" width="80%" />
</p>

> 图表来源：[InfiR2: A Comprehensive FP8 Training Recipe for Reasoning-Enhanced Language Models](https://arxiv.org/html/2509.22536v4)

> 

**第一步：计算缩放因子 $S$**

取给定张量（或其子块）中的最大绝对值 $max|X|$，将其除以 FP8 格式所能表示的最大值 $V_{max}$。

$$
	S=\frac{max|X|}{V_{max}}
$$

**第二步：计算量化值 $Q$**

利用缩放因子 **$S$**，将原始张量 $X$ 中的每个数值 $x$ 除以 **$S$** 并进行四舍五入，得到新的量化值。

$$
Q(x)=round(\frac{x}{S})
$$

由于 FP8 的精度低于原始 FP16/BF16 ，实际训练中需要在训练稳定性和计算效率之间做权衡，因此前向与反向传播通常会采用不同的量化策略和粒度。

- **激活值(Activations)**：通常选择 per-token 量化。激活值中常出现显著的离群值（Outliers），较细的量化粒度能够将离群值的影响限制在局部，从而更好地保留整体精度。

- **权重(Weights)**：通常选择 per-block 量化。训练收敛后的权重分布通常较为平稳（接近高斯分布），极少出现异常值，但对量化误差十分敏感，按块量化（如 block_size × block_size）在保证精度的同时能更好地配合硬件优化，兼顾计算效率与存储节省。

- **梯度(Gradients)**：通常选择 per-token 量化。梯度数值的动态范围变化很大，但对精度的绝对值要求较低。过去大部分方案会使用 **per-tensor E5M2** 精度来保证动态范围，但是 DeepSeek-V3 证明了细粒度 E4M3 量化能够兼顾精度和动态范围。

<p align="center">
  <img src="./pic/3_Megatron.png" alt="Megatron 中使用 FP8 的混合粒度量化策略" width="80%" />
</p>

> 图表来源：[InfiR2: A Comprehensive FP8 Training Recipe for Reasoning-Enhanced Language Models](https://arxiv.org/html/2509.22536v4)

上图展示了 Megatron 中使用 FP8 的混合粒度量化策略，并与标准的 BF16 流水线进行了比较。在 FP8 流水线中，应用了不同的量化方法：权重采用 per-block 量化（蓝色），激活值采用 per-token 量化（紫色）。该图展示了完整的训练过程，包括前向传播 (FProp)、权重梯度计算 (Wgrad) 和输入梯度计算 (Dgrad)，并详细展示了 FProp 的工作流程。

> 

## FP8 训练难点

尽管 FP8 展现出巨大的潜力，但在实际工程应用中，尤其是在结合 Megatron-Core 与 TransformerEngine (TE) 的实践中，我们遇到了三大挑战：**显存与效率未达预期、精度对齐困难以及框架自身的稳定性问题**。

### **显存与计算效率：理论与现实的差距**

在工程实践中，FP8 带来的显存节省和计算加速效果并非如理论般显著，主要原因如下：

- **显存优化有限**：

    - **冗余权重副本**：为了加速反向传播计算，TransformerEngine 在量化权重时，[会额外保存一份转置后的量化权重副本](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/transpose/quantize_transpose_square_blockwise.cu#L481)。这使得权重部分占用的显存并未如预期般减半。

    - **高精度激活值副本**：在 Attention 和激活层的前向传播中，框架通常会保留一份高精度的激活值副本，用于后续精确的梯度计算。这部分显存开销并未因 FP8 的引入而减少。

- **计算效率瓶颈：**

    - **小批量（Small Batch Size）下的性能衰退**：当 batch_size 较小时，FP8 训练可能无法充分利用 GPU 的计算单元，导致其性能甚至劣于 BF16。其根本原因在于，FP8 引入了额外的量化（Quantization）和反量化（Dequantization）操作，这些操作增加了 CPU 的计算负担。考虑到 Agentic RL 场景通常采用较小批量（例如 batch_size=4），上述问题尤为突出——频繁的 CPU 开销甚至会导致 FP8 训练速度慢于传统的 BF16 训练。（详情见下图，GPU kernel并不密集，很多时候 GPU已经完成了上一个工作，但由于 CPU bound 导致下一个 kernel launch 还没发出来）

<p align="center">
  <img src="./pic/4_cpu_bound.png" alt="CPU bound for FP8 training" width="80%" />
</p>

> 图表所示：CPU Bound for FP8 Training

> 

### **精度对齐：不可忽视的累积误差**

FP8 的低精度特性天然带来了与 BF16 之间的数值差异，这种差异会在深度模型中被放大，引发训练稳定性问题。

- **量化引入的固有误差**：对于单次 GEMM 操作，即便累加过程在 FP32 下进行，FP8 输入的量化过程本身就会引入误差。实验表明，与 BF16 GEMM 相比，其典型误差值在 0.0007 左右。

- **误差的逐层累积效应**：在层数较深的 Transformer 模型中，这种微小的误差会在前向和反向传播中逐层累积。

    - **在预训练（Pre-training）和微调（SFT）中**：这类任务的梯度主要由真实标签（Ground Truth）对应的对数概率（Log Probability）主导。配合细粒度的块状量化（Block-wise Quantization），误差通常能被控制在可接受范围内，模型不易崩溃。

    - **在强化学习（RL）中**：RL 的梯度通常由两次前向传播结果的对数概率之差（Difference of Log Probabilities）决定。在这种情况下，FP8 带来的累积误差可能会被放大，导致梯度方向出现偏差，进而影响模型的收敛效率，甚至使其“走错方向”。（详情见后文）

### **框架适配挑战：TransformerEngine 的版本兼容性问题**

除了算法层面的挑战，Megatron-Core 与 Transformer Engine (TE) 框架的结合也有一定的改进空间，尤其是 TE 的版本迭代带来了一定的不稳定性。

- **版本依赖与迁移成本**：TransformerEngine 的持续快速迭代带来了新特性，但也意味着严格的版本依赖。我们在项目实践中发现，即便是相同模型的训练脚本，在不同版本的 TE 上运行时，也可能观察到数值结果的差异，甚至需要对代码进行适应性调整才能避免如 NaN (非数值）错误。

- **特定结构兼容性的成熟度**：框架对**所有主流模型结构**的 FP8 训练支持是一个渐进的过程。对于一些**非标准或新型的模型组件**（如 MLA），我们观察到其 FP8 训练支持的成熟度仍在提升中。即使在后续版本（如从 2.4.0 升级至 2.8.0）中，相关报错和限制仍有待解决。

- **显存优化策略冲突**：在 RL 训练中，启用 Optimizer Cpu Offload 本可显著降低显存占用，但当前的 TE 尚未支持该功能与 `--fp8-param-gather` 同时开启。这一兼容性限制导致全链路 FP8 训练的显存开销反而高于 BF16 训练  FP8 采样的混合模式，该问题亟待社区与开发团队进一步优化解决。

## **FP8 ➕ RL：KL Loss 异常归因**

**InfiXAI 团队**此前已经成功在 [Pre-training 任务和 Fine-tuning 任务](https://arxiv.org/html/2509.22536v4)上将 FP8 训练完整运行。在此基础上，我们将 FP8 训练技术应用于 RL。得益于 slime 框架对 Megatron FP8 训练的良好支持，我们顺利展开一系列 FP8 RL 的实验。

### **初始 KL Loss 异常**

直接将 BF16 切换到 FP8 并启动训练后，我们观察到一个显著现象：与 BF16 训练相比，FP8 训练在第一个 step 的 KL loss 明显更高。如下图所示：FP8 训推（训练和推理均使用 FP8）的初始 KL loss 显著大于 BF16 训练、FP8 推理模式下的初始 KL loss。（图中 T 代表 Training，I 代表 Inference）

<p align="center">
  <img src="./pic/5_KLloss.png" alt="初始 KL loss 对比" width="80%" />
</p>

### **定位误差来源**

为了探究初始 KL loss 偏高的原因，我们从量化过程入手，分析两个潜在的误差来源：

1. **量化计算算子误差**：某种特定的 FP8 GEMM（通用矩阵乘法）带来的计算误差。

2. **量化操作固有误差**：数据在量化（Quantization）和反量化（Dequantization）过程中产生的精度损失。

**量化计算算子误差分析**

最初我们猜测 TransformerEngine 里闭源的 cuBLAS GEMM 实现不如开源广泛使用的 DeepGEMM 准确，因此我们设计实验比较了两种 FP8 GEMM 实现（cuBLAS 和 DeepGEMM）与 BF16 的精度差异，以判断是否是 GEMM 的精度问题导致了 KL loss 异常。我们在不同 shape（参考TE的测试用例） 下对两种 GEMM 的误差进行了比较，结果如下表所示：

| **Kernel (M, K, N)** | **cuBLAS(TE)** | **DeepGEMM** |
| --- | --- | --- |
| 128,128,128 | 0.00068 | 0.00036 |
| 256,128,256 | 0.00068 | 0.00037 |
| 320,128,336 | 0.000684 | 0.00037 |
| 320,64,336 | 0.00067 | 0.00024 |
| 320,256,336 | 0.00068 | 0.00048 |
| 1024,4096,1024 | 0.000681 | 0.00065 |
| 2048,2048,512 | 0.00068 | 0.00063 |
| 1024,1024,1024 | 0.000683 | 0.0006 |

实验结果表明，两种GEMM实现的误差处于同一数量级，不存在显著差异，因此我们可以认为替换 TransformerEngine 里的 FP8 GEMM 并不能降低初始的 KL loss。

**量化操作固有误差分析**

针对第二个潜在误差来源，我们设计了一组对比实验来分离并验证量化操作的**固有**误差。

- **实验基准**：Qwen3-4B 模型，单机 H800。

- **实验变量**：

    - 我们设定了三种模式：

        1. **Baseline**：权重（weight）和输入（input）均为 BF16 精度，使用 BF16 GEMM。

        2. **FP8 Real Quant**：权重和输入为 FP8 精度，使用 FP8 GEMM（如 DeepGEMM/cuBLAS GEMM；为了避免大面积改动 TransformerEngine，我们主要测试 cuBLAS GEMM）。

        3. **FP8 Fake Quant**：权重和输入为 BF16 精度，但模拟量化过程（先量化到 FP8，再反量化回 BF16），最后使用 BF16 GEMM。

    - 基于以上模式，我们进行两组对比：

        - **FP8 Real Quant vs. FP8 Fake Quant**：旨在验证 FP8 GEMM 算子（cuBLAS）本身的实现精度，排除算子实现带来的额外误差。

        - **Baseline vs. FP8 Fake Quant**：旨在排除 GEMM 算子影响，专注于评估量化与反量化操作本身引入的固有误差。

- **实验指标**：统计 RL 训练初期（Step 0 和 Step 1）所有 GEMM 操作的输出差异（Diff）。

- **实验结果：**

    下图按执行顺序可视化了一次完整 Forward + Backward 过程中，各层 GEMM 输出的误差分布：

<p align="center">
  <img src="./pic/6_FP8_quant_error.png" alt="FP8 量化误差分布" width="80%" />
</p>

> 上图展示了模型在一次完整迭代中 GEMM 输出的误差变化
> - **灰色/高数值点（Baseline vs. FP8 Fake Quant）**：代表量化本身引入的误差。可以看出，BF16 Baseline 与模拟量化（Fake Quant）之间存在显著差异
> - **绿色/低数值点（FP8 Real Quant vs. FP8 Fake Quant）**：代表算子实现引入的误差。可以看出，真实 FP8 计算与模拟量化之间的差异极小，几乎为零。

由此可见：

- **误差源于量化原理而非算子实现**：Fake Quant 和 Real Quant 均与 Baseline 存在显著误差（高出两个数量级），这有力地证明了**误差主要来源于 FP8 量化和反量化的有损过程本身**，而非计算过程。

- **FP8 GEMM 算子高度可信**：Real Quant 与 Fake Quant 的输出差异微乎其微，说明我们在 TransformerEngine 中调用的 cuBLAS FP8 GEMM 算子精度极高，与理想的数学模拟（Fake Quant）几乎一致，可以直接投入生产环境使用。

### **量化误差如何导致训练异常**

基于以上实验，我们提出以下推测：

1. 训练中的主要误差在量化步骤就已经产生，且影响显著。

2. FP8 训练初始 KL loss 更高，很可能是由这种量化误差导致的。

3. 在 BF16 训练、FP8 推理的混合模式下，这种量化误差同样会导致训练和推理的不一致性。

为了验证这些推测，我们在 Transformer Engine (TE) 上进行了改造，并设计了如下实验：

- **实验基准**：Qwen3-4B 模型，H800 集群

- **实验变量**：

    - **Case 1**：BF16 训练、FP8 推理（Rollout）

    - **Case 2**：BF16 训练、FP8 推理；在训练的 Forward 阶段，对 BF16 的权重和激活值进行“FP8 量化 -> 反量化回 BF16”操作，然后执行 BF16 GEMM

    - **Case 3**：BF16 训练、FP8 推理；在训练的 Forward 和 Backward 阶段，都对输入矩阵 A 和 B 进行“FP8 量化 -> 反量化回 BF16”操作，然后执行 BF16 GEMM

    - **Case 4 (FP8-TI)**：FP8 训练，FP8 推理

**验证推测 2——KL-loss 分析**

下图展示了四种情况下 KL loss 的变化。可以看到，Case 2、Case 3 和 Case 4 (FP8-TI) 在 step 1 的 KL loss 基本一致，并且都显著高于 Case 1。

<p align="center">
  <img src="./pic/7_KLloss2.png" alt="不同方案 KL loss 对比" width="80%" />
</p>

**验证推测 3 ——TIS-clipfrac 分析**

我们引入**截断重要性采样（Truncated Importance Sampling, TIS）** 中的 clipfrac 指标来验证推测 3，该指标可以反映 off-policy 的程度，即模型在训练和推理（生成 experience）时的一致性。clipfrac 越高，通常意味着训推不一致性越严重。

![不同方案 TIS-clipfrac 对比](./pic/8_TIS.png)

从上图可以看出，Case 2、Case 3 和 Case 4（FP8-TI）的 TIS-clipfrac 值在数量级上基本一致，且均显著低于 Case 1。这一结果证实：

1. 初始 KL loss 偏高的根本原因是量化误差的存在

2. **FP8 训推一体（Case 4）相较于 BF16 训练、FP8 推理的混合模式（Case 1），能够极大地缓解训推不一致现象**

3. 在训练偏差问题中，Forward 过程的量化误差比 Backward 过程的影响更大（比较 Case 2 和 Case 3 的相似性可知）。同样地，在训推一致性问题中，Forward 过程的量化误差是主要影响因素

## **FP8 在 MoE 模型强化学习中的应用与验证**

在 Dense 模型的实验表明，采用 FP8 进行统一的训练与推理（简称“FP8 训推”），能有效抑制两者之间的不一致性。基于这一发现，**蚂蚁集团 AQ Team** 进一步将研究拓展至 MoE 模型的 RL 场景，旨在验证 FP8 训推方案在更复杂模型结构下的有效性。我们进一步发现 FP8 训推的方案能够：

1. **降低 TIS clip fraction**：FP8 训推方案的裁剪比例显著低于 BF16 Train / FP8 Rollout 方案，策略更新过程中 TIS 裁剪频率更低，训练稳定性更高；

2. **收敛训练与推理的对数概率差异**：相较于 BF16 Train/ FP8 Rollout 方案，FP8 训推方案的差值波动范围更小；

### MoE 模型实验设计

为隔离变量并进行精确对比，我们设置了以下两种实验方案：

- **Case 1 (混合精度)**：采用 BF16 训练、FP8 推理

- **Case 2 (统一精度)**：采用 FP8 训练、FP8 推理

**关键评估指标：**

- **TIS 裁剪比例 (TIS-clipfrac)**：衡量 off-policy 训练的稳定性；如前文所述，比例越低，稳定性越高

- **训练-推理对数概率绝对差值（train_rollout_logprob_abs_diff）**：衡量训练与推理两个阶段模型行为的一致性。差值越小且越稳定，一致性越好

### MoE **实验结果与分析**

**Qwen3-30B-A3B 模型验证**

- **实验环境:** 双机 H20服务器

在 30B 规模的 MoE 模型上，实验结果清晰地展示了 FP8 训推方案的优势：

- **TIS裁剪比例更低：** FP8训推方案的 **TIS-clipfrac** 显著低于 BF16 训练 / FP8 推理的混合精度方案（方案一）。这表明FP8训推能有效减少策略更新过程中的裁剪操作，从而提升训练稳定性

- **训推概率差异更小：** FP8 训推方案的 **Train_rollout_logprob_abs_diff** 波动范围明显收窄，证明其训练过程与推理过程的行为更加一致。

<p align="center">
  <img src="./pic/9_1_TIS.png" alt="Qwen3-30B-A3B TIS-clipfrac" width="49%" />
  <img src="./pic/9_2_rollout_logprob_abs_diff.png" alt="Qwen3-30B-A3B train_rollout_logprob_abs_diff" width="49%" />
</p>

**Qwen3-235B-A22B**

- **实验环境:** 16机 H20服务器

为了验证该结论在更大规模模型上的普适性，我们在 235B 模型上进行了复现，并得到了一致的结论：

- **TIS裁剪比例与训推差异持续改善：** 如下图所示，即使在 235B 的大规模 MoE 模型上，FP8 训推方案在降低 **TIS-clipfrac** 和 **Train_rollout_logprob_abs_diff** 方面依然表现出一定的优越性，验证了该方案的可扩展性

<p align="center">
  <img src="./pic/10_1_TIS.png" alt="Qwen3-235B-A22B TIS-clipfrac" width="49%" />
  <img src="./pic/10_2_rollout_logprob_abs_diff.png" alt="Qwen3-235B-A22B train_rollout_logprob_abs_diff" width="49%" />
</p>



**结论**：在 MoE 模型的强化学习任务中，采用**统一的 FP8 进行训练与推理**，相比 BF16 Train / FP8 Rollout 的混合精度方案，能够**提升训练稳定性，并有效抑制训推不一致性**。这一优势在从 30B 到 235B 的不同规模 MoE 模型上均得到了有力验证。

## **MOE 模型规模对训推不一致性的影响分析**

我们进一步探究了在混合精度（BF16 Train / FP8 Rollout）设定下，MoE 模型的规模对训推不一致性的影响。实验发现，**随着 MoE 模型规模的增大，训推不一致性问题会愈发严重**。

如下图所示，从 30B 模型到 1TB 模型，**TIS-clipfrac** 和 **Train_rollout_logprob_abs_diff** 均呈现出明显的增长趋势，即表明对于 BF16 Train / FP8 Rollout 方案，模型规模的扩大可能会加剧训推不一致性问题，也反向印证了统一精度方案（如FP8训推）的重要性。

<p align="center">
  <img src="./pic/11_1_TIS.png" alt="不同模型规模下的 TIS-clipfrac" width="49%" />
  <img src="./pic/11_2_rollout_logprob_abs_diff.png" alt="不同模型规模下的 train_rollout_logprob_abs_diff" width="49%" />
</p>

## 未来工作

感谢大家的阅读，我们认为还有一些值得努力的方向：

1. 围绕训推不一致性问题展开更深入的研究，深入探索其根本原因，以及更优的解决方案。

2. 围绕量化策略展开，更深入的探究量化误差的产生的原因，探索误差更小的量化策略。

3. 围绕低精度训练效率展开，通过更加高效的算法、框架和硬件的协同设计，隐藏 kernel launch、quant 等操作的耗时，真正实现训练推理的加速。

## 致谢

InfiXAI Team：Congkai Xie, Mingfa Feng, Shuo Cai

蚂蚁集团 AQ Team：Yanan Gao, Zhiling Ye, Hansong Xiao

SGLang RL Team：JiLi, Yefei Chen, Xi Chen, Chenyang Zhao

slime Team：Zilin Zhu 

NVIDIA：Juan Yu, NeMo-RL Team
