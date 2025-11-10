# 浅析 SGLang 框架的量化设计与思路

编者按：2025 年年初时，本系列的 ML-SYS-Tutorial 刚刚拥有 1k github stars。那时候这系列笔记中还包括有各方各面的内容，包括量化、并行策略，而今天，由于编者的时间有限，这几个月只有 RL sys 部分偶尔更新，确实背离初衷。索性编者结识了社区的好朋友们，打算在近期逐渐将散落在本系列博客的未发表部分逐一审核、重写然后发表。为了做到精益求精，我们会发布 linkedin、知乎、twitter、小红书以及 github 原版。由于时间有限，其中只有 github 的中文版维持更新，其余部分均为一次性 LLM 机器翻译（不过翻译质量肯定会得到保证），感谢大家理解。这段时间编者的工作和学习也发生了许多重大变化，等到事情告一段落再来分享这一路的心历。

【本文由来自百度的姚攀登老师所写，由编者协助进行了一定修改，感谢姚老师的支持和对 SGLang 的厚爱。下文的“我”为姚攀登老师。】

前段时间我深度参与了 deepseek 模型的 w4afp8 量化项目，被迫对 SGLang 的量化部分有了不错的认知。（顺手提了个 [PR](https://github.com/sgl-project/sglang/pull/9598)，最终合进了 [#8247](https://github.com/sgl-project/sglang/pull/8247#issuecomment-3258884128) 中）。

项目做完后，我一直想抽空做个总结，因为 SGLang 在量化这块的设计确实非常清晰，很值得拿出来聊聊。所以就有了这篇文章。本文会以 w4afp8 为例，带大家看看 SGLang 是如何处理量化的。

SGLang 把所有的量化实现都"藏"在了 `python/sglang/srt/layers/quantization/` 目录里。它用了一套非常巧妙的抽象和钩子函数 (Hook Function)，把模型构建、权重加载、推理执行这三个阶段有机地连接起来。

**我们将量化拆解为三个阶段：create_weights → process_weights_after_loading → apply**

- **create_weights**（建立管道）：预分配张量，模型刚开始构建，此函数负责把量化权重、Scale 因子这些参数分配内存。类似于先铺设水管，还没有通水。
- **process_weights_after_loading**（数据转换）：权重从文件加载进来后，不能直接处理，它负责把权重和 Scale 转换成计算内核（比如 CUTLASS）最优的格式和布局，比如做一些重排优化，而后存储下来。
- **apply**：当模型运行 `forward` 时，`apply` 函数就会被调用，它会指挥底层的计算内核（比如 FP8 GEMM），让激活和权重在管道里真正流动起来，完成计算。

## 整体流程

具体看看 SGLang 的实现逻辑：

```
ModelConfig._parse_quant_hf_config → 判定 quant_method（如 w4afp8）
      ↓
weight_utils.get_quant_config → 构造对应 QuantizationConfig 实例
      ↓
_initialize_model(...) → 把 quant_config 传入模型/各层
      ↓
LinearBase.quant_method.create_weights → 注册量化权重占位
      ↓
DefaultModelLoader.load_weights_and_postprocess → 先 load_weights 加载权重，再逐层调用 quant_method.process_weights_after_loading
      ↓
推理时 LinearBase.forward → quant_method.apply 执行量化 GEMM
```

### 类的继承层次

SGLang 的抽象和继承做的恰到好处。

```
配置类继承关系：
QuantizationConfig (抽象基类，定义在 base_config.py)
  职责：解析量化配置、硬件校验与激活数据类型校验，并按层返回正确的量化方法实例
  ├─ from_config()：从配置字典解析并实例化
  ├─ get_quant_method()：根据层类型返回量化方法
  ├─ get_min_capability()：校验硬件兼容性
  │  硬件校验包括：
  │  • NVIDIA GPU：通过 CUDA capability（计算能力）检查，如 70 (Volta)、75 (Turing)、80 (Ampere)、90 (Hopper) 等
  │  • AMD GPU：通过 ROCm/HIP 平台检测，某些方案会检查特定的 GCN 架构（如 gfx94）
  └─ get_supported_act_dtypes()：返回支持的激活数据类型
      ↓
  W4AFp8Config (具体配置类)
      └─→ get_quant_method() 根据层类型返回：
          ├─→ LinearBase → Fp8LinearMethod
          └─→ FusedMoE → W4AFp8MoEMethod

量化方法类继承关系：
QuantizeMethodBase (抽象基类，定义在 base_config.py)
  职责：权重注册、权重加载完成后的处理以及前向执行
  ├─ create_weights()：注册量化权重占位符（模型构建阶段）
  ├─ process_weights_after_loading()：权重后处理（权重加载完成后）
  └─ apply()：前向传播执行量化计算（推理阶段）
      ├─→ LinearMethodBase
      │     └─→ Fp8LinearMethod (用于普通线性层)
      └─→ FusedMoEMethodBase
            └─→ W4AFp8MoEMethod (用于 MoE 层)

量化方法注册表（位于 __init__.py）：
将原生方案（AWQ、GPTQ、FP8、W4AFp8、ModelOpt 等）映射成字符串标识
```

### create_weights 调用流程

```
DeepseekV2DecoderLayer.__init__()
      ↓
DeepseekV2AttentionMLA.__init__()
      ↓
RowParallelLinear.__init__()
      ↓
LinearBase.__init__() (super().__init__()) # self.quant_method 就是在父类初始化函数里
      ↓
quant_config.get_quant_method() → 返回 Fp8LinearMethod
      ↓
RowParallelLinear.__init__() 继续执行
      ↓
Fp8LinearMethod.create_weights()
      ↓
注册 weight、weight_scale、input_scale 等参数占位符
```

### process_weights_after_loading 调用流程

```
Scheduler.__init__()
      ↓
TpModelWorker.__init__()
      ↓
ModelRunner.__init__()
      ↓
ModelRunner.initialize()
      ↓
get_model()
      ↓
DefaultModelLoader.load_model()
      ↓
DefaultModelLoader.load_weights_and_postprocess()
      ↓
model.load_weights() → 加载权重数据
      ↓
逐层遍历，调用 quant_method.process_weights_after_loading()
      ↓
Fp8LinearMethod.process_weights_after_loading()
      └─→ 或 W4AFp8MoEMethod.process_weights_after_loading()
```

### apply 调用流程

```
DeepseekV2DecoderLayer.forward()
      ↓
DeepseekV2AttentionMLA.forward()
      ↓
RowParallelLinear.forward()
      ↓
self.quant_method.apply()
      ↓
Fp8LinearMethod.apply()
      └─→ 或 W4AFp8MoEMethod.apply()
      ↓
调用底层内核（CUTLASS/Marlin/torch）执行量化 GEMM
```

## W4AFp8 量化方案

下面以 W4AFp8（权重 INT4 + 激活 FP8）量化方案为例，详细分析 SGLang 的具体实现逻辑。

### W4AFp8Config

`W4AFp8Config` 继承自 `QuantizationConfig`，负责描述清楚"配置 → 具体量化方法"。

**配置识别**：当 `hf_quant_config.json` 中 `quant_algo == "MIXED_PRECISION"` 时，`ModelConfig` 会把量化方案映射为 `w4afp8` 并校验硬件兼容性：

```python
# ModelConfig._parse_modelopt_quant_config
if quant_algo == "MIXED_PRECISION":
    return {"quant_method": "w4afp8"}
```

**对象构造**：`weight_utils.get_quant_config` 获取 `W4AFp8Config` 类，然后调用 `from_config` 方法进行实例化。

**关键方法**：

- `W4AFp8Config.from_config()`：从配置字典解析并实例化配置对象
- `W4AFp8Config.get_quant_method(layer, prefix)`：核心方法，根据层类型返回对应的量化方法实例：
  
```python
if isinstance(layer, LinearBase):
    return Fp8LinearMethod(self)  # 普通层用 Fp8LinearMethod
elif isinstance(layer, FusedMoE):
    return W4AFp8MoEMethod(self)  # MoE 层用 W4AFp8MoEMethod
```

### W4AFp8MoEMethod 具体流程

`W4AFp8MoEMethod` 是 W4AFp8 在 MoE 层上的具体实现。按照之前的三个步骤，我们来分析具体的实现：

1. `create_weights`

如前所述，这一步是参数预分配。在 `FusedMoE` 模块初始化时，它会为 MoE 层创建量化所需的参数容器，负责把量化权重、Scale 因子这些参数分配内存。

主要工作包括：

- 创建量化权重张量：`w13_weight`（gate 和 up projection）和 `w2_weight`（down projection），注意类型是 `int8`
- 分配权重缩放因子：`w13_weight_scale_inv` 和 `w2_weight_scale_inv`，每组 128 个元素共享一个 scale
- 准备激活缩放因子：`w13_input_scale` 和 `w2_input_scale`
- 初始化计算所需的元数据：如 stride、expert offsets 等

注意，此时参数为空（使用 `torch.empty` 创建），仅完成内存布局的初始化，尚未填充实际数据。

```python
def create_weights(self, layer, num_experts, hidden_size, ...):
    # 创建量化权重容器（INT8 类型）
    layer.register_parameter("w13_weight", torch.empty(..., dtype=torch.int8))
    layer.register_parameter("w2_weight", torch.empty(..., dtype=torch.int8))
    # 创建权重缩放因子（group-wise，每组 128 元素）
    layer.register_parameter("w13_weight_scale_inv", torch.zeros(...))
    layer.register_parameter("w2_weight_scale_inv", torch.zeros(...))
    # 创建输入缩放因子（静态量化时使用）
    layer.register_parameter("w13_input_scale", torch.ones(..., dtype=torch.bfloat16))
    # 初始化 stride 等计算元数据
    self.a_strides1 = torch.full((num_experts, 3), hidden_size, ...)
```

2. `process_weights_after_loading`

权重数据从 Checkpoint 加载后，需要进行格式转换以适配底层计算内核：

- **权重 scale 的格式优化**：将 float32 格式的 scale 转换为 bfloat16（减少 50% 内存占用），并调用 `interleave_scales` 函数对 scale 进行交错重排。（重排的目的是匹配 CUTLASS 内核的内存访问模式（参考了 TRT-LLM 的实现）。重排后，内核在计算时能够更高效地访问数据，提升缓存命中率。）
- **输入 scale 的聚合**：在静态量化模式下，把每个专家的输入 scale 聚合为单一标量，减少推理计算量。

```python
def process_weights_after_loading(self, layer: Module) -> None:
    # 将权重 scale 转换为 bfloat16 并重新排列以匹配 CUTLASS 布局
    w13_weight_scale = layer.w13_weight_scale_inv.to(torch.bfloat16)
    w13_weight_scale = interleave_scales(w13_weight_scale)
    layer.w13_weight_scale_inv = Parameter(w13_weight_scale, requires_grad=False)
    
    # 将输入 scale 聚合为单一标量（静态量化模式）
    w13_input_scale_max = layer.w13_input_scale.max().to(torch.bfloat16).item()
    layer.w13_input_scale = Parameter(torch.tensor([w13_input_scale_max], dtype=torch.bfloat16), requires_grad=False)
```

3. `apply`

万事俱备，只欠 forward。在前向传播阶段，`apply` 函数被调用以执行量化计算。此函数收集所有预处理完成的数据（激活、重排后的权重和 scale、路由结果等），然后调用 `cutlass_w4a8_moe` 底层内核执行两个 GEMM 操作：

- GEMM1：`w13_weight`（gate 和 up）
- GEMM2：`w2_weight`（down）

`cutlass_w4a8_moe` 是封装了 CUTLASS 库的底层函数，实现了 INT4 权重与 FP8 激活的混合精度矩阵乘法，充分利用了硬件的量化计算能力。

```python
def apply(self, layer, dispatch_output) -> CombineInput:
    from sglang.srt.layers.moe.cutlass_w4a8_moe import cutlass_w4a8_moe
    
    x = dispatch_output.hidden_states
    topk_weights, topk_ids, _ = dispatch_output.topk_output
    
    # 调用 CUTLASS 内核执行混合精度 MoE 计算
    output = cutlass_w4a8_moe(
        x, layer.w13_weight, layer.w2_weight,
        layer.w13_weight_scale_inv, layer.w2_weight_scale_inv,
        topk_weights, topk_ids,
        self.a_strides1, self.b_strides1, self.c_strides1,  # GEMM1 的 stride
        self.a_strides2, self.b_strides2, self.c_strides2,  # GEMM2 的 stride
        self.s_strides13, self.s_strides2,                  # Scale 的 stride
        self.expert_offsets, self.problem_sizes1, self.problem_sizes2,
        layer.w13_input_scale, layer.w2_input_scale,
    )
    # 应用路由缩放因子
    if self.moe_runner_config.routed_scaling_factor is not None:
        output *= self.moe_runner_config.routed_scaling_factor
    return StandardCombineInput(hidden_states=output)
```

### Fp8LinearMethod

接着参考线性层的实现，`W4AFp8Config` 会分配 `Fp8LinearMethod`；逻辑类似，但更简单：

- `create_weights`：注册 `weight`、`weight_scale` 和 `input_scale` 占位符。
- `process_weights_after_loading`：根据硬件（Marlin、CUTLASS 等）要求，对权重和 scale 进行格式转换。
- `apply`：调用合适的内核（Marlin、CUTLASS 等）执行 FP8 GEMM 计算。

## 如何扩展新量化方案？

SGLang 的可扩展性以及恰到好处、不多不少的抽象确实非常舒服。要接入新的量化方案（例如 W2A8），无需修改框架核心代码，只需按照以下步骤实现：

1. **实现配置类**：继承 `QuantizationConfig`，解析自定义参数并实现 `get_quant_method` 方法。
2. **实现量化方法类**：继承 `LinearMethodBase`、`FusedMoEMethodBase`，实现 `create_weights`、`process_weights_after_loading` 和 `apply` 三个方法。
3. **注册方案**：在 `__init__.py` 的 `BASE_QUANTIZATION_METHODS` 中注册，建立字符串标识与配置类的映射关系。

SGLang 的解耦设计让这一切变得非常简单。

## 附录：SGLang 中已经支持的量化方法

| Category              | Representative Configurations                                                                 | Description                                                                              |
| --------------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| FP8 Series            | `fp8`, `w8a8_fp8`, `modelopt_fp8`, `fbgemm_fp8`                                               | Native FP8, W8A8-FP8 hybrid, ModelOpt/FBGEMM extensions                                  |
| INT8 Series           | `w8a8_int8`, `blockwise_int8`                                                                 | Classic 8bit weight/activation, blockwise INT8                                           |
| INT4/Mixed Precision  | `w4afp8`, `qoq`, `moe_wna16`                                                                  | 4bit weight + FP8 activation, QoQ, WNA16 (W4A16/W8A16)                                  |
| FP4 / MXFP4           | `modelopt_fp4`, `petit_nvfp4`, `mxfp4`, `quark`                                               | FP4 / MXFP4 schemes, `quark` is ROCm-specific                                           |
| Pre-quantized Formats | `awq`, `awq_marlin`, `gptq`, `gptq_marlin`, `gguf`, `compressed-tensors`, `auto-round`, `modelopt` | Integration with external toolchains or compressed tensor frameworks, `modelopt` auto-detects FP8/FP4 |
| KV Cache Quantization | `BaseKVCacheMethod` and its subclasses in `kv_cache.py`                                       | Provides scale and zero-point management for attention caches                            |
