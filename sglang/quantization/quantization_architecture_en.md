# An Analysis of SGLang Framework's Quantization Design and Approach

Editor's Note: At the beginning of 2025, this ML-SYS-Tutorial had just reached 1k GitHub stars. At that time, the series included content covering various aspects, including quantization and parallelization strategies. However, due to limited time, only the RL sys section has been occasionally updated in recent months, which indeed deviates from the original intention. Fortunately, the editor has met good friends in the community and plans to gradually review, rewrite, and publish the unpublished parts scattered throughout this blog series. To achieve excellence, we will publish on LinkedIn, Zhihu, Twitter, Xiaohongshu, and the original GitHub version. Due to time constraints, only the Chinese version on GitHub will be maintained and updated, while the other versions will be one-time LLM machine translations (though translation quality will be guaranteed). Thank you for your understanding. The editor's work and learning journey have also undergone many significant changes during this period, and we will share this journey once things settle down.

[This article was written by Yao Pandeng from Baidu, with the editor assisting in certain modifications. Thank you to Teacher Yao for his support and love for SGLang. The "I" in the following text refers to Yao Pandeng.]

Recently, I was deeply involved in the w4afp8 quantization project for the DeepSeek model, which forced me to gain a good understanding of SGLang's quantization components. (I also submitted a [PR](https://github.com/sgl-project/sglang/pull/9598), which was eventually merged into [#8247](https://github.com/sgl-project/sglang/pull/8247#issuecomment-3258884128)).

After completing the project, I've been wanting to take some time to summarize, because SGLang's design in quantization is indeed very clear and worth discussing. Hence this article. This article will use w4afp8 as an example to show how SGLang handles quantization.

SGLang hides all quantization implementations in the `python/sglang/srt/layers/quantization/` directory. It uses a very clever abstraction and hook functions to organically connect three stages: model construction, weight loading, and inference execution.

**We break down quantization into three stages: create_weights → process_weights_after_loading → apply**

- **create_weights** (Establish Pipeline): Pre-allocate tensors. When the model is first being constructed, this function is responsible for allocating memory for quantization weights, scale factors, and other parameters. Similar to laying down pipes before water flows through them.
- **process_weights_after_loading** (Data Conversion): After weights are loaded from files, they cannot be directly processed. This function is responsible for converting weights and scales into the optimal format and layout for computational kernels (such as CUTLASS), such as performing rearrangement optimizations, and then storing them.
- **apply**: When the model runs `forward`, the `apply` function is called. It directs the underlying computational kernel (such as FP8 GEMM) to make activations and weights flow through the pipeline, completing the computation.

## Overall Flow

Let's look at SGLang's implementation logic:

```
ModelConfig._parse_quant_hf_config → Determine quant_method (e.g., w4afp8)
      ↓
weight_utils.get_quant_config → Construct corresponding QuantizationConfig instance
      ↓
_initialize_model(...) → Pass quant_config into model/layers
      ↓
LinearBase.quant_method.create_weights → Register quantization weight placeholders
      ↓
DefaultModelLoader.load_weights_and_postprocess → First load_weights to load weights, then call quant_method.process_weights_after_loading layer by layer
      ↓
During inference, LinearBase.forward → quant_method.apply executes quantized GEMM
```

### Class Inheritance Hierarchy

SGLang's abstraction and inheritance are done just right.

```
Configuration class inheritance:
QuantizationConfig (Abstract base class, defined in base_config.py)
  Responsibilities: Parse quantization configuration, hardware validation and activation data type validation, and return correct quantization method instances by layer
  ├─ from_config(): Parse from configuration dictionary and instantiate
  ├─ get_quant_method(): Return quantization method based on layer type
  ├─ get_min_capability(): Validate hardware compatibility
  │  Hardware validation includes:
  │  • NVIDIA GPU: Check via CUDA capability (compute capability), such as 70 (Volta), 75 (Turing), 80 (Ampere), 90 (Hopper), etc.
  │  • AMD GPU: Check via ROCm/HIP platform detection, some schemes check for specific GCN architectures (e.g., gfx94)
  └─ get_supported_act_dtypes(): Return supported activation data types
      ↓
  W4AFp8Config (Concrete configuration class)
      └─→ get_quant_method() returns based on layer type:
          ├─→ LinearBase → Fp8LinearMethod
          └─→ FusedMoE → W4AFp8MoEMethod

Quantization method class inheritance:
QuantizeMethodBase (Abstract base class, defined in base_config.py)
  Responsibilities: Weight registration, post-loading weight processing, and forward execution
  ├─ create_weights(): Register quantization weight placeholders (model construction stage)
  ├─ process_weights_after_loading(): Post-process weights (after weight loading completes)
  └─ apply(): Execute quantized computation during forward propagation (inference stage)
      ├─→ LinearMethodBase
      │     └─→ Fp8LinearMethod (for regular linear layers)
      └─→ FusedMoEMethodBase
            └─→ W4AFp8MoEMethod (for MoE layers)

Quantization method registry (located in __init__.py):
Maps native schemes (AWQ, GPTQ, FP8, W4AFp8, ModelOpt, etc.) to string identifiers
```

### create_weights Call Flow

```
DeepseekV2DecoderLayer.__init__()
      ↓
DeepseekV2AttentionMLA.__init__()
      ↓
RowParallelLinear.__init__()
      ↓
LinearBase.__init__() (super().__init__()) # self.quant_method is set in the parent class initialization function
      ↓
quant_config.get_quant_method() → Returns Fp8LinearMethod
      ↓
RowParallelLinear.__init__() continues execution
      ↓
Fp8LinearMethod.create_weights()
      ↓
Register weight, weight_scale, input_scale and other parameter placeholders
```

### process_weights_after_loading Call Flow

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
model.load_weights() → Load weight data
      ↓
Iterate through layers, calling quant_method.process_weights_after_loading()
      ↓
Fp8LinearMethod.process_weights_after_loading()
      └─→ or W4AFp8MoEMethod.process_weights_after_loading()
```

### apply Call Flow

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
      └─→ or W4AFp8MoEMethod.apply()
      ↓
Call underlying kernel (CUTLASS/Marlin/torch) to execute quantized GEMM
```

## W4AFp8 Quantization Scheme

Below, we use the W4AFp8 (INT4 weights + FP8 activations) quantization scheme as an example to analyze SGLang's specific implementation logic in detail.

### W4AFp8Config

`W4AFp8Config` inherits from `QuantizationConfig` and is responsible for clearly describing "configuration → specific quantization method".

**Configuration Recognition**: When `quant_algo == "MIXED_PRECISION"` in `hf_quant_config.json`, `ModelConfig` will map the quantization scheme to `w4afp8` and validate hardware compatibility:

```python
# ModelConfig._parse_modelopt_quant_config
if quant_algo == "MIXED_PRECISION":
    return {"quant_method": "w4afp8"}
```

**Object Construction**: `weight_utils.get_quant_config` obtains the `W4AFp8Config` class, then calls the `from_config` method to instantiate it.

**Key Methods**:

- `W4AFp8Config.from_config()`: Parse from configuration dictionary and instantiate configuration object
- `W4AFp8Config.get_quant_method(layer, prefix)`: Core method that returns corresponding quantization method instance based on layer type:
  
```python
if isinstance(layer, LinearBase):
    return Fp8LinearMethod(self)  # Regular layers use Fp8LinearMethod
elif isinstance(layer, FusedMoE):
    return W4AFp8MoEMethod(self)  # MoE layers use W4AFp8MoEMethod
```

### W4AFp8MoEMethod Specific Flow

`W4AFp8MoEMethod` is the specific implementation of W4AFp8 on MoE layers. Following the three steps mentioned earlier, let's analyze the specific implementation:

1. `create_weights`

As mentioned earlier, this step is parameter pre-allocation. When the `FusedMoE` module is initialized, it creates parameter containers required for quantization for the MoE layer, responsible for allocating memory for quantization weights, scale factors, and other parameters.

Main tasks include:

- Create quantized weight tensors: `w13_weight` (gate and up projection) and `w2_weight` (down projection), note the type is `int8`
- Allocate weight scale factors: `w13_weight_scale_inv` and `w2_weight_scale_inv`, with each group of 128 elements sharing one scale
- Prepare activation scale factors: `w13_input_scale` and `w2_input_scale`
- Initialize computation metadata: such as stride, expert offsets, etc.

Note that at this point, parameters are empty (created using `torch.empty`), only completing memory layout initialization, without filling in actual data.

```python
def create_weights(self, layer, num_experts, hidden_size, ...):
    # Create quantized weight containers (INT8 type)
    layer.register_parameter("w13_weight", torch.empty(..., dtype=torch.int8))
    layer.register_parameter("w2_weight", torch.empty(..., dtype=torch.int8))
    # Create weight scale factors (group-wise, 128 elements per group)
    layer.register_parameter("w13_weight_scale_inv", torch.zeros(...))
    layer.register_parameter("w2_weight_scale_inv", torch.zeros(...))
    # Create input scale factors (used in static quantization)
    layer.register_parameter("w13_input_scale", torch.ones(..., dtype=torch.bfloat16))
    # Initialize stride and other computation metadata
    self.a_strides1 = torch.full((num_experts, 3), hidden_size, ...)
```

2. `process_weights_after_loading`

After weight data is loaded from the checkpoint, format conversion is needed to adapt to the underlying computational kernel:

- **Weight scale format optimization**: Convert scale from float32 format to bfloat16 (reducing memory usage by 50%), and call the `interleave_scales` function to interleave and rearrange scales. (The purpose of rearrangement is to match CUTLASS kernel's memory access pattern (referencing TRT-LLM's implementation). After rearrangement, the kernel can access data more efficiently during computation, improving cache hit rate.)
- **Input scale aggregation**: In static quantization mode, aggregate each expert's input scale into a single scalar, reducing inference computation.

```python
def process_weights_after_loading(self, layer: Module) -> None:
    # Convert weight scale to bfloat16 and rearrange to match CUTLASS layout
    w13_weight_scale = layer.w13_weight_scale_inv.to(torch.bfloat16)
    w13_weight_scale = interleave_scales(w13_weight_scale)
    layer.w13_weight_scale_inv = Parameter(w13_weight_scale, requires_grad=False)
    
    # Aggregate input scale into a single scalar (static quantization mode)
    w13_input_scale_max = layer.w13_input_scale.max().to(torch.bfloat16).item()
    layer.w13_input_scale = Parameter(torch.tensor([w13_input_scale_max], dtype=torch.bfloat16), requires_grad=False)
```

3. `apply`

Everything is ready, only forward remains. During the forward propagation stage, the `apply` function is called to execute quantized computation. This function collects all preprocessed data (activations, rearranged weights and scales, routing results, etc.), then calls the `cutlass_w4a8_moe` underlying kernel to execute two GEMM operations:

- GEMM1: `w13_weight` (gate and up)
- GEMM2: `w2_weight` (down)

`cutlass_w4a8_moe` is an underlying function that wraps the CUTLASS library, implementing mixed-precision matrix multiplication of INT4 weights and FP8 activations, fully utilizing the hardware's quantized computation capabilities.

```python
def apply(self, layer, dispatch_output) -> CombineInput:
    from sglang.srt.layers.moe.cutlass_w4a8_moe import cutlass_w4a8_moe
    
    x = dispatch_output.hidden_states
    topk_weights, topk_ids, _ = dispatch_output.topk_output
    
    # Call CUTLASS kernel to execute mixed-precision MoE computation
    output = cutlass_w4a8_moe(
        x, layer.w13_weight, layer.w2_weight,
        layer.w13_weight_scale_inv, layer.w2_weight_scale_inv,
        topk_weights, topk_ids,
        self.a_strides1, self.b_strides1, self.c_strides1,  # GEMM1 strides
        self.a_strides2, self.b_strides2, self.c_strides2,  # GEMM2 strides
        self.s_strides13, self.s_strides2,                  # Scale strides
        self.expert_offsets, self.problem_sizes1, self.problem_sizes2,
        layer.w13_input_scale, layer.w2_input_scale,
    )
    # Apply routing scale factor
    if self.moe_runner_config.routed_scaling_factor is not None:
        output *= self.moe_runner_config.routed_scaling_factor
    return StandardCombineInput(hidden_states=output)
```

### Fp8LinearMethod

Next, referring to the linear layer implementation, `W4AFp8Config` will assign `Fp8LinearMethod`; the logic is similar but simpler:

- `create_weights`: Register `weight`, `weight_scale`, and `input_scale` placeholders.
- `process_weights_after_loading`: Convert weights and scales according to hardware requirements (Marlin, CUTLASS, etc.).
- `apply`: Call appropriate kernel (Marlin, CUTLASS, etc.) to execute FP8 GEMM computation.

## How to Extend New Quantization Schemes?

SGLang's extensibility and its just-right, not-too-much, not-too-little abstraction is indeed very comfortable. To integrate a new quantization scheme (e.g., W2A8), you don't need to modify the framework's core code, just follow these steps:

1. **Implement configuration class**: Inherit from `QuantizationConfig`, parse custom parameters and implement the `get_quant_method` method.
2. **Implement quantization method class**: Inherit from `LinearMethodBase`, `FusedMoEMethodBase`, and implement the three methods: `create_weights`, `process_weights_after_loading`, and `apply`.
3. **Register scheme**: Register in `BASE_QUANTIZATION_METHODS` in `__init__.py`, establishing a mapping relationship between string identifiers and configuration classes.

SGLang's decoupled design makes all of this very simple.

## Appendix: Quantization Methods Already Supported in SGLang

| Category              | Representative Configurations                                                                 | Description                                                                              |
| --------------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| FP8 Series            | `fp8`, `w8a8_fp8`, `modelopt_fp8`, `fbgemm_fp8`                                               | Native FP8, W8A8-FP8 hybrid, ModelOpt/FBGEMM implementations                                  |
| INT8 Series           | `w8a8_int8`, `blockwise_int8`                                                                 | Classic 8bit weight/activation, blockwise INT8                                           |
| INT4/Mixed Precision  | `w4afp8`, `qoq`, `moe_wna16`                                                                  | 4bit weight + FP8 activation, QoQ, WNA16 (W4A16/W8A16)                                  |
| FP4 / MXFP4           | `modelopt_fp4`, `petit_nvfp4`, `mxfp4`, `quark`                                               | FP4 / MXFP4 schemes, `quark` is ROCm-specific                                           |
| Pre-quantized Formats | `awq`, `awq_marlin`, `gptq`, `gptq_marlin`, `gguf`, `compressed-tensors`, `auto-round`, `modelopt` | Integration with external toolchains or compressed tensor frameworks, `modelopt` auto-detects FP8/FP4 |
| KV Cache Quantization | `BaseKVCacheMethod` and its subclasses in `kv_cache.py`                                       | Provides scale and zero-point management for attention caches                            |

