# SGLang 模型加载过程

## Overview

SGLang 的模型加载过程由 `model_loader` 文件夹中的代码实现。

在 `__init__.py` 中定义了 `get_model` 函数，负责根据 `load_config` 获取相应的 `loader`，并调用 `loader.load_model` 实际加载模型：

```cpp
def get_model(
    *,
    model_config: ModelConfig,
    load_config: LoadConfig,
    device_config: DeviceConfig,
) -> nn.Module:
    loader = get_model_loader(load_config)
    return loader.load_model(
        model_config=model_config,
        device_config=device_config,
    )
```

下面以 `DefaultModelLoader` 为例，详细介绍如何从开放权重文件加载模型到 SGLang 中。

## DefaultModelLoader

1. **loader.load_model: 模型初始化 (_initialize_model)**
   - `_initialize_model` 调用 `get_model_architecture`，根据 `model_config` 获取模型架构。`ModelRegistry.resolve_model_cls` 会返回实际的模型类。
   - `_initialize_model` 调用 `_get_quantization_config`，根据 `model_config` 和 `load_config` 获取 `quant_config`：
     - 在实际初始化模型时，`Linear` 层（如 `ColumnParallelLinear` 等）会在 `__init__` 中调用 `self.quant_method.create_weights`。对于没有量化方法的模型，`quant_method` 会被设置为 `UnquantizedLinearMethod`，其 `create_weights` 方法会创建指定形状和数据类型的权重参数，并设置输入输出维度等元信息，最终将其注册到层中供后续使用。
     - 在 `Linear` 层的 `forward` 函数中，会调用 `quant_method.apply` 进行实际计算。
2. **loader.load_model: 获取权重迭代器 (_get_all_weights)**
   - 权重分为主要权重和次要权重：
     - **主要权重**：加载 `Source`，并调用 `_get_weights_iterator` 实际加载权重参数。`_get_weights_iterator` 会根据不同的权重格式（如 `.bin`、`safetensors`、`.pt`）加载权重，返回权重迭代器。
     - **次要权重**：目前没有模型使用此特性。
3. **loader.load_model: 调用 model.load_weights**（以 `qwen2` 为例）
   - `stacked_params_mapping` 用于将 checkpoint 中分开的权重参数（如 `q/k/v` 或 `gate/up`）映射并加载到模型中合并后的参数（如 `qkv_proj` 或 `gate_up_proj`）。
     - `param_name`：模型中合并后的参数名。
     - `shard_name`：checkpoint 中单独的原始参数名。
     - `shard_id`：该原始参数在合并后的位置（如 "q"、"k"、"v" 或 0、1）。
   - **跳过不需要加载的权重**。
   - **检查是否属于堆叠参数**：
     - 如果是，将 checkpoint 中的名称（如 `"model.layers.0.attn.q_proj.weight"`）中的 `shared_name`（如 `q_proj`）替换成 `param_name`（如 `qkv_proj`），以便与 `params_dict` 匹配。
   - **weight_loader**：将加载的权重（`loaded_weight`）拷贝到当前模型的参数（`param`）中，并根据参数类型和分布式配置进行一些特殊处理。
     - 每个 `linear` 算子的 `weight_loader` 可能会有所不同，具体差异可参考 [linear.py 文件](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py)。
4. **loader.load_model: 权重加载完成后**，遍历模型中的每个子模块。当量化方法在加载后需要处理权重（例如重新打包、量化等）时，它们期望参数位于全局目标设备上。这个作用域适用于使用 CPU 卸载的情况，在这种情况下，SGLang 会将参数移动到设备上，并执行量化方法中定义的 `process_weights_after_loading`，处理完后再将参数移回原位置。

------

## SGLang 底层算子（以下内容仅供参考，仍需交叉验证）

- **`VocabParallelEmbedding`**：一种支持词表维度并行（Tensor Parallelism）和 LoRA 动态扩展的嵌入层，专为在多卡推理和微调场景中高效、灵活地管理词表权重而设计。

- **`ColumnParallelLinear`**：线性层定义为 `Y = XA + b`，其中矩阵 A 沿第二维（列并行）进行并行化，即 `A = [A_1, ..., A_p]`。

- **`RowParallelLinear`**：支持将权重矩阵按“行”切分到多卡的线性层，用于在多 GPU 上并行计算 `Y = X @ A + b` 中的线性部分。矩阵 A 按第一维并行，X 按第二维并行。

  ```cpp
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
  ```

- **`QKVParallelLinear`**：用于注意力机制中的 QKV 转换线性层，负责查询（query）、键（key）和值（value）向量的线性变换。权重矩阵沿输出维度拼接，并在头（attention head）维度上进行并行化。当键/值的头数少于查询的头数时（如在多查询或分组查询注意力中），键和值的头会被复制，而查询的头会被切分。

  - weight_loader：

    - `output_dim`: 当前这个 param（模型中的某个权重张量）在第几维代表“输出维度”，也就是说——要在这一维上进行分片（sharding）或加载切片。

    1. PART 1：加载的是合并的 QKV 权重（fused tensor）

       1. 无 `output_dim`，直接 copy

       2. 手动定义 Q/K/V 的分片信息

          | 名字 | 偏移      | 大小                     |
          | ---- | --------- | ------------------------ |
          | q    | 从 0 开始 | num_heads × head_size    |
          | k    | 紧接 q 后 | num_kv_heads × head_size |
          | v    | 紧接 k 后 | num_kv_heads × head_size |

       3. 最终调用自身加载单 shard

    2. PART 2：如果 `loaded_shard_id` 是 `"q"`、`"k"`、或 `"v"` ——加载单独一个 shard。

- `MergedColumnParallelLinear`

  带有列并行性的打包线性层。类似于 `ColumnParallelLinear`，但权重矩阵是在输出维度上拼接的。 当加载权重矩阵时，不同的分区会被分别切分。

  - weight_loader：

    - `output_dim`: 当前这个 param（模型中的某个权重张量）在第几维代表“输出维度”，也就是说——要在这一维上进行分片（sharding）或加载切片。

    1. 获取模型参数的 data（实际 tensor）和它的一些标志位：
       - `output_dim`: 要沿哪个维度进行分片（列并行）
       - `is_metadata`: 是否是元数据（特殊格式）
       - `needs_scalar_to_array`: 是否需要把标量 reshape 成向量（通常用于 fused scale）
    2. 如果  `loaded_shard_id` 是 `None`：整块加载
       1. 如果没有 `output_dim`，直接 copy（说明这个权重根本就不用分片）
       2. 如果有 `output_dim`，但没 shard id，手动切片调用自己加载。遍历每个输出 shard，把这块 weight 切分为多个小 shard，分别调用自己。
    3. 有 shard_id 的情况（即单独一个 shard 要加载）：进行裁剪和加载



------

## BitsAndBytesModelLoader

当使用 BitsAndBytes 量化加载模型时，会调用 `BitsAndBytesModelLoader`，而非 `DefaultModelLoader`。主要的区别体现在 `loader.load_model` 中，下面将从 `load_model` 开始详细讲解。

### loader.load_model (loader=BitsAndBytesModelLoader)

1. **模型初始化 (`_initialize_model`)**。

2. **加载权重 (`_load_weights`)**：

   - 验证模型是否支持 BitsAndBytes 量化加载，并根据配置准备所需参数和状态以加载量化权重。

   1. 验证参数。

   2. **获取量化权重迭代器 (`_get_quantized_weights_iterator`)**：获取供 `model.load_weights` 使用的 `qweight_iterator`，同时获取 `QuantState` 对象 `quant_state_dict`：

      - 通过 `_prepare_weights` 获取实际的模型权重。

      - `_quantized_4bit_generator`：实际加载并处理 Hugging Face 模型的 4-bit 量化权重文件。

        ```cpp
        HuggingFace权重文件 →
          ├─ 收集所有量化元数据 → temp_state_dict
          └─ 遍历权重文件 →
                ├─ 若存在 quant_state → 构建 QuantState 对象 → 存入 quant_state_dict
                └─ yield (param_name, weight_tensor)
        ```

   3. 调用 `model.load_weights`，这一过程与 `DefaultModelLoader` 中类似，只是传入的是 `qweight_iterator`。

   4. 将分片的量化权重按统一参数名和分片索引组织起来，为设置量化状态做准备。

   5. 为每个量化参数设置量化状态、分片偏移信息及 8-bit 推理所需的运行时状态。

------

## BNB 模型权重

```cpp
model.embed_tokens.weight
model.layers.0.input_layernorm.weight
model.layers.0.mlp.down_proj.weight
model.layers.0.mlp.down_proj.weight.absmax
model.layers.0.mlp.down_proj.weight.nested_absmax
model.layers.0.mlp.down_proj.weight.nested_quant_map
model.layers.0.mlp.down_proj.weight.quant_map
model.layers.0.mlp.down_proj.weight.quant_state.bitsandbytes__nf4
...
```

| 权重名后缀                       | 作用                                     |
| -------------------------------- | ---------------------------------------- |
| `.absmax`                        | 整个张量的最大绝对值，用于缩放恢复 float |
| `.nested_absmax`                 | 分 chunk 计算的最大值（更细粒度）        |
| `.nested_quant_map`              | 每个分块的量化信息                       |
| `.quant_map`                     | 用于解码的映射索引表                     |
| `.quant_state.bitsandbytes__nf4` | 储存的是实际的 NF4 编码数据              |

------

