# Latency Accelerate for Weight Updates

## 前言

本文是一篇 debug 笔记，因此比较详细地描述了我的 debug 过程，实际上结论非常简单，我们可以一段话总结完：

1. 为了准确测量 GPU 的 latency，我们们要在测速的语句前后加上无数的 `torch.cuda.synchronize()`，否则经常会出现 CPU 跑的飞起，早早 print 了，而 GPU 还卡在之前。具体而言：

```python
torch.cuda.synchronize()
time_begin = time.time()
# ...
torch.cuda.synchronize()
time_end = time.time()
print(f"latency: {time_end - time_begin:.3f}s")
```

2. 为了准确使用 `dist.barrier()`，最好指定 `device_ids`，否则在 CI 可能莫名其妙会因为 device error 卡死。


## 背景

费劲千番力气，我终于成功实现了 `update_parameter_from_distributed` 这个接口。按照 advisor 的意思，这个函数 OpenRLHF 基于 vllm 的实现不超过 50 行。某种意义上，我的实现并不繁琐，只是由于缺乏经验，反复折腾了两周。终于，到 2024 年感恩节的前一天，我成功自顶向下实现了如下的三个接口：

1. `init_parameter_update_group`
2. `update_parameter_from_distributed`
3. `get_weights_by_parameter_name`

三个函数为了达成一个功能，前一个用于建立进程组。我们假定将 Training Engine 传递的 weights 放在 rank 0 上（尽管 rank 0 上可能并不足以存下整个模型，但是 training engine 总可以从 rank 0 上将 weights 传出去）。接着，我们的 sglang server 将会和 rank 0 建立进程组，并从 rank 0 上广播得到 weights，并且 load 到所有的 tensor parall device 上。然后，我们通过 `get_weights_by_parameter_name` 这个函数检查 sglang inference engine 的更新是否完善。注意到，training engine 不必然需要将 model 存储为 huggingface 格式，事实上工业界大规模使用的引擎肯定是整个训练过程利用自身的模型格式，然后训练完成了才将 checkpoint 转为 huggingface 格式用于发布。然而，OpenRLHF 作为偏学术界的开源产品，会使用 huggingface model 作为通用的 protocol。


<details>
<summary>为什么会有两个 engine？</summary>

这里需要提出一个看上去很显然的问题，为什么 RLHF 流程需要 training 和 inference 两个 engine？对于前者，主流系统有非常多选择，譬如 DeepSpeed，而后者，我们希望支持 SGLang。换句话说，为什么不能用 training engine 做 inference，用 inference engine 做 training？

1. training engine 只有 forward，但是得到 logits 之后，无论是为了 evaluate 还是 roll out，都需要实际让模型做 decoding。decoding 就大有文章了，SGLang 的主要贡献是 continuous batching and KV cache management，因此天然适合为了整个训练流程做 evaluation 或者 roll out。
2. 反过来，inference engine 没有 back propagation，当然不可能做 training。不过，inference engine 可以用于计算 KL divergence 么？答曰，不可，因为 KL divergence 需要 logits 的精度较高，而 inference engine 的 logits 精度目前并不满足（不满足的原因我也还在理解）。

</details>

总之，实现了这三个接口之后，我终于手写了单测，然后成功通过了测试，效率却不尽如人意。

## 测试效果

<details>
<summary> 具体的单测 </summary>

```python

import gc
import os
import time
import unittest

import numpy as np
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM

import sglang as sgl
from sglang.srt.utils import init_custom_process_group
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
)

mp.set_start_method("spawn", force=True)


class TestParameterUpdateGroup(unittest.TestCase):

    @classmethod
    def init_process(
        cls,
        rank,
        world_size,
        param_queue,
        truncate_size,
        state_dict_key_to_shape,
        tp_size,
        model_name,
    ):
        torch.cuda.set_device(rank)
        parameters = [
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.2.self_attn.k_proj.weight",
            "model.layers.3.self_attn.v_proj.weight",
            "model.layers.4.self_attn.o_proj.weight",
            "model.layers.5.mlp.gate_proj.weight",
            "model.layers.6.mlp.up_proj.weight",
            "model.layers.7.mlp.down_proj.weight",
            "model.layers.8.post_attention_layernorm.weight",
            "model.norm.weight",
            "lm_head.weight",
        ]
        print(f"testing model: {model_name}")
        print(f"testing tp size: {tp_size}")
        if rank == 0:
            os.environ["NCCL_CUMEM_ENABLE"] = "0"
            os.environ["NCCL_NVLS_ENABLE"] = "0"

            # 加载 instruct 模型
            torch.cuda.synchronize()
            time_begin = time.time()
            cls.hf_instruct_model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype="bfloat16"
            ).to("cuda:0")
            torch.cuda.synchronize()
            time_end = time.time()
            print(f"rank {rank} load instruct model time: {time_end - time_begin:.3f}s")

            # 加载 base 模型
            torch.cuda.synchronize()
            time_begin = time.time()
            base_model_name = model_name.replace("-Instruct", "")
            cls.hf_base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name, torch_dtype="bfloat16"
            ).to("cuda:0")
            torch.cuda.synchronize()
            time_end = time.time()
            print(f"rank {rank} load base model time: {time_end - time_begin:.3f}s")

            cls.hf_instruct_params = []
            cls.hf_base_params = []

            # 获取参数
            torch.cuda.synchronize()
            time_begin = time.time()
            print(f"get parameter in hf instruct model and base model")
            for parameter_name in parameters:
                cls.hf_instruct_params.append(
                    cls.hf_instruct_model.get_parameter(parameter_name)[:truncate_size]
                    .cpu()
                    .detach()
                    .float()
                    .numpy()
                    .tolist()
                )
                cls.hf_base_params.append(
                    cls.hf_base_model.get_parameter(parameter_name)[:truncate_size]
                    .cpu()
                    .detach()
                    .float()
                    .numpy()
                    .tolist()
                )
            torch.cuda.synchronize()
            time_end = time.time()
            print(f"rank {rank} get parameters time: {time_end - time_begin:.3f}s")

            param_queue.put(("hf_instruct_params", cls.hf_instruct_params))
            param_queue.put(("hf_base_params", cls.hf_base_params))

            # 初始化进程组
            torch.cuda.synchronize()
            time_begin = time.time()
            print(f"rank {rank} init custom process group")
            cls.group = init_custom_process_group(
                backend="nccl",
                init_method="tcp://localhost:65500",
                world_size=world_size,
                rank=rank,
                group_name="test_parameter_update_group",
            )
            torch.cuda.synchronize()
            time_end = time.time()
            print(f"rank {rank} init process group time: {time_end - time_begin:.3f}s")

            # 广播参数
            torch.cuda.synchronize()

            print(f"rank {rank} broadcast parameter")

            for parameter_name in state_dict_key_to_shape.keys():
                torch.cuda.synchronize()
                time_begin = time.time()
                torch.distributed.broadcast(
                    cls.hf_base_model.get_parameter(parameter_name),
                    src=0,
                    group=cls.group,
                )
                torch.cuda.synchronize()
                time_end = time.time()
                print(
                    f"rank {rank} broadcast {parameter_name} time: {time_end - time_begin:.3f}s"
                )

            torch.cuda.synchronize()

            del cls.hf_instruct_model
            del cls.hf_base_model
            gc.collect()
            torch.cuda.empty_cache()

        elif rank == 1:
            # 初始化引擎
            torch.cuda.synchronize()
            time_begin = time.time()
            cls.engine = sgl.Engine(
                model_path=model_name,
                random_seed=42,
                base_gpu_id=rank,
                tp_size=tp_size,
            )
            torch.cuda.synchronize()
            time_end = time.time()
            print(f"rank {rank} init engine time: {time_end - time_begin:.3f}s")

            # 获取 instruct 参数
            torch.cuda.synchronize()
            time_begin = time.time()
            cls.engine_instruct_params = []
            print(f"rank {rank} get parameter in engine instruct model")
            for parameter_name in parameters:
                cls.engine_instruct_params.append(
                    cls.engine.get_weights_by_parameter_name(
                        parameter_name, truncate_size
                    )
                )
            torch.cuda.synchronize()
            time_end = time.time()
            print(
                f"rank {rank} get instruct parameters time: {time_end - time_begin:.3f}s"
            )

            param_queue.put(("engine_instruct_params", cls.engine_instruct_params))

            # 初始化参数更新组
            torch.cuda.synchronize()
            time_begin = time.time()
            print(f"rank {rank} init parameter update group")
            cls.engine.init_parameter_update_group(
                master_address="localhost",
                master_port="65500",
                rank_offset=1,
                world_size=world_size,
                group_name="test_parameter_update_group",
                backend="nccl",
            )
            torch.cuda.synchronize()
            time_end = time.time()
            print(
                f"rank {rank} init parameter update group time: {time_end - time_begin:.3f}s"
            )

            # 更新分布式参数
            torch.cuda.synchronize()
            time_begin = time.time()
            print(f"rank {rank} update parameter from distributed")
            for parameter_name in state_dict_key_to_shape.keys():
                torch.cuda.synchronize()
                time_begin = time.time()
                cls.engine.update_parameter_from_distributed(
                    parameter_name,
                    dtype=torch.bfloat16,
                    shape=state_dict_key_to_shape[parameter_name],
                    empty_cache=True,
                )
                torch.cuda.synchronize()
                time_end = time.time()
                print(
                    f"rank {rank} update {parameter_name} from distributed time: {time_end - time_begin:.3f}s"
                )

            torch.cuda.synchronize()
            # 获取 base 参数
            time_begin = time.time()
            cls.engine_base_params = []
            print(f"rank {rank} get parameter in engine base model")
            for parameter_name in parameters:
                cls.engine_base_params.append(
                    cls.engine.get_weights_by_parameter_name(
                        parameter_name, truncate_size
                    )
                )
            torch.cuda.synchronize()
            time_end = time.time()
            print(f"rank {rank} get base parameters time: {time_end - time_begin:.3f}s")

            param_queue.put(("engine_base_params", cls.engine_base_params))
            print(f"rank {rank} shutdown engine")
            cls.engine.shutdown()

    @classmethod
    def setUpClass(cls):
        assert torch.cuda.device_count() >= 2, "At least 2 GPUs are required"
        cls.test_suits = [1]
        cls.model_names = [
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            DEFAULT_MODEL_NAME_FOR_TEST,
        ]

        if torch.cuda.device_count() >= 4:
            cls.test_suits.append(2)

        # 初始化每个模型的 state_dict_key_to_shape
        cls.model_state_dict_shapes = {}
        for model_name in cls.model_names:
            torch.cuda.synchronize()
            time_begin = time.time()
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype="bfloat16"
            ).to("cuda:0")
            state_dict = model.state_dict()
            state_dict_keys = list(state_dict.keys())
            cls.model_state_dict_shapes[model_name] = {
                key: state_dict[key].shape for key in state_dict_keys
            }
            del model
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time_end = time.time()
            print(
                f"Initialize state dict shapes for {model_name} time: {time_end - time_begin:.3f}s"
            )
            time.sleep(2)

    @classmethod
    def test_init_parameter_update_group(cls):
        truncate_size = 10

        for model_name in cls.model_names:
            print(f"Testing model: {model_name}")
            state_dict_key_to_shape = cls.model_state_dict_shapes[model_name]

            for tp_size in cls.test_suits:
                print(f"test tp_size: {tp_size}")
                param_queue = mp.Queue()
                results = {}

                torch.cuda.synchronize()
                time_begin = time.time()
                context = mp.spawn(
                    cls.init_process,
                    args=(
                        1 + tp_size,
                        param_queue,
                        truncate_size,
                        state_dict_key_to_shape,
                        tp_size,
                        model_name,
                    ),
                    nprocs=2,
                    join=False,
                )

                while len(results) < 4:
                    try:
                        key, value = param_queue.get(timeout=5)
                        results[key] = value
                    except Exception as e:
                        if all(not p.is_alive() for p in context.processes):
                            break

                context.join()
                torch.cuda.synchronize()
                time_end = time.time()
                print(f"Total spawn and join time: {time_end - time_begin:.3f}s")

                if len(results) != 4:
                    raise RuntimeError(f"Expected 4 parameters but got {len(results)}")

                hf_instruct_params = results["hf_instruct_params"]
                hf_base_params = results["hf_base_params"]
                engine_instruct_params = results["engine_instruct_params"]
                engine_base_params = results["engine_base_params"]

                for i in range(len(hf_instruct_params)):
                    assert np.allclose(
                        np.array(hf_instruct_params[i]),
                        np.array(engine_instruct_params[i]),
                    )
                    assert np.allclose(
                        np.array(hf_base_params[i]), np.array(engine_base_params[i])
                    )
                    assert not np.allclose(
                        np.array(hf_instruct_params[i]), np.array(hf_base_params[i])
                    )

                del context
                param_queue.close()
                param_queue.join_thread()
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(2)


if __name__ == "__main__":
    unittest.main()
```

</details>


简单来说，这个测试的逻辑如下，对于 model 为 8B llama 3.1 和 1B llama 3.2 分别测试在 sglang engine 的 tp 为 1 和 2 时的正确性和效率：

1. rank 0 (模拟 training engine)
- 利用 huggingface 加载 instruct model 和 base model
- 读取代表性参数作为验证样本（每一类参数都做了抽查）
- 初始化进程组
- 广播 base model 的全部参数

2. rank 1 (SGLang inference engine)
- 初始化 engine，加载 instruct model
- 读取 instruct model 的代表性参数
- 初始化参数更新组
- 接收并更新全部参数
- 获取更新后的 base model 参数进行验证

整体上，在满血 8 卡 H100 上，居然整体测试结束需要 431.264s，我感到非常费解。注意到，实际上的更新函数如下：

<details>
<summary>实际的更新函数</summary>

```python

    def update_parameter_from_distributed(self, name, dtype, shape, empty_cache=False):
        """
        Update specific parameter in the model weights online through the process group.

        Args:
            name: the name of the parameter to be updated.
            dtype: the data type of the parameter to be updated.
            shape: the shape of the parameter to be updated.
            empty_cache: whether to empty the cache after updating the parameter.
        """
        target_dtype = (
            dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
        )
        current_dtype = self.dtype if isinstance(self.dtype, str) else self.dtype
        assert str(target_dtype) == str(
            current_dtype
        ), f"dtype mismatch: target={dtype} vs current model runner={self.dtype}"
        assert (
            self._model_update_group is not None
        ), "model update group must be initialized"

        try:
            weights = torch.empty(shape, dtype=target_dtype, device=self.device)
            torch.distributed.broadcast(weights, src=0, group=self._model_update_group)
            self.model.load_weights([(name, weights)])
            if empty_cache:
                torch.cuda.empty_cache()

            return True, f"Succeeded to update parameter {name} online."

        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            return False, error_msg

```

</details>

看上去每一步都不该很慢，但是出现了非常神奇的事情。在我的 uint test 上，我把每个参数的更新用时都在这几行打印出来了。

```python

            for parameter_name in state_dict_key_to_shape.keys():
                torch.cuda.synchronize()
                time_begin = time.time()
                cls.engine.update_parameter_from_distributed(
                    parameter_name,
                    dtype=torch.bfloat16,
                    shape=state_dict_key_to_shape[parameter_name],
                    empty_cache=True,
                )
                torch.cuda.synchronize()
                time_end = time.time()
                print(
                    f"rank {rank} update {parameter_name} from distributed time: {time_end - time_begin:.3f}s"
                )

```

同时，我在最底层实际调用的 `update_parameter_from_distributed` 函数中试图打印每一步的用时：

<details>
<summary>测试更新代码每一步的耗时</summary>

```python

    def update_parameter_from_distributed(self, name, dtype, shape, empty_cache=False):
        """
        Update specific parameter in the model weights online through the process group.

        Args:
            name: the name of the parameter to be updated.
            dtype: the data type of the parameter to be updated.
            shape: the shape of the parameter to be updated.
            empty_cache: whether to empty the cache after updating the parameter.
        """
        target_dtype = (
            dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
        )
        current_dtype = self.dtype if isinstance(self.dtype, str) else self.dtype
        assert str(target_dtype) == str(
            current_dtype
        ), f"dtype mismatch: target={dtype} vs current model runner={self.dtype}"
        assert (
            self._model_update_group is not None
        ), "model update group must be initialized"

        try:
            torch.cuda.synchronize()
            time_begin = time.time()
            weights = torch.empty(shape, dtype=target_dtype, device=self.device)
            torch.cuda.synchronize()
            time_end = time.time()
            print(
                f"rank {self.tp_rank} {name} create weights time: {time_end - time_begin:.3f}s"
            )
            torch.cuda.synchronize()
            time_begin = time.time()
            torch.distributed.broadcast(weights, src=0, group=self._model_update_group)
            torch.cuda.synchronize()
            time_end = time.time()
            print(
                f"rank {self.tp_rank} {name} broadcast weights time: {time_end - time_begin:.3f}s"
            )
            torch.cuda.synchronize()
            time_begin = time.time()
            self.model.load_weights([(name, weights)])
            torch.cuda.synchronize()
            time_end = time.time()
            print(
                f"rank {self.tp_rank} {name} load weights time: {time_end - time_begin:.3f}s"
            )
            if empty_cache:
                torch.cuda.empty_cache()

            return True, f"Succeeded to update parameter {name} online."

        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            return False, error_msg

```

</details>

对于整个更新函数，我几乎怀疑到了每一步头上。首先是开头的几个检查用的 assert，接着是创建 weights 的 empty tensor，然后是广播，最后是 load weights。

惊人的是，单独每一步的耗时都是 0.000s，然而单测中的返回时间居然是 0.032s。此外，8B model 和 1B model 的单步更新时间完全一致。太有意思了，这样以来我更新整个 1B 模型的用时达到了 7.047s。考虑到 H100 的满血 NV Link 带宽单位是 TB / s，而 1B 模型的 weights 在 bf16 也就 2GB 左右，这样的时间消耗显然是不合理的。

所以，时间都去哪儿了？

## 时间都去哪儿了？

好问题，八千余日夜已经在我的生命中流逝，而我的人生也不过三万多天。初中时，曾经教过我一段时间的数学竞赛的一位老师老师常说，“人生啊，不过三万多天，我也曾年轻过，哦豁，就老咯...”十年前的我绝不曾感受过时间的流逝，然而十年后，我已经二十二岁，想到人类世界的荒诞和虚无，原来时间是对人的惩罚。一方面，我感念，毕竟我只有一生这么短，讨好他人对我而言无疑是在浪费生命，另一方面，如果我的生命的开始和结束都是虚无的，那么我的人生到底有什么意义？

至少，想办法将这 7.047s 的传输开销降到 1s 以内，是我理解的人生意义的一部分。

我合理怀疑，这些开销有如下可能：

1. `https` 请求太慢了：在 sglang 的设计模式中，有两层 `https` 请求，一层是最顶层的 `RunTime` 通过一个 fastapi 向下调用 tokenizer manager，另一层是 `tokenizer manager` 通过另一个 fast api 的 `https` 请求向 `scheduler -> tp worker -> model runner` 传递请求。

2. python 的函数传递开销太大了：Model Runner 的 `update_parameter_from_distributed` 居然每一步都是 0.000s，那么自顶向下，从 `RunTime` 到 `tokenizer manager` 再到 `scheduler -> tp worker -> model runner` 是否存在很大的传递开销。究竟是哪一层显著增大了开销？

3. 我没有异步更新参数：实际上，`update_parameter_from_distributed` 并不会重复写同一片 weights，似乎异步是一个解。

4. update 的时候，是不是在 blocking 状态下？试试只 launch kernel，这样可以全部 overlap 起来（from advisor）

5. nccl 太慢了：这是我觉得很不现实的，因为我的测试机器是 NVDA 提供的满血 H100。

无所谓，我先进行这样一个测试：

```python

            torch.cuda.synchronize()
            time_begin = time.time()
            print(
                f"start to update model_name {model_name} rank {rank} parameter from distributed"
            )
            for parameter_name in state_dict_key_to_shape.keys():
                torch.cuda.synchronize()
                time_begin = time.time()
                cls.engine.update_parameter_from_distributed(
                    parameter_name,
                    dtype=torch.bfloat16,
                    shape=state_dict_key_to_shape[parameter_name],
                    empty_cache=True,
                )
                torch.cuda.synchronize()
                time_end = time.time()
                print(
                    f"model_name {model_name} rank {rank} update {parameter_name} {state_dict_key_to_shape[parameter_name]} from distributed time: {time_end - time_begin:.3f}s"
                )
            torch.cuda.synchronize()
            time_end = time.time()
            print(
                f"fully update model_name {model_name} rank {rank} parameter from distributed time: {time_end - time_begin:.3f}s"
            )

```

让我看看，传输效率究竟和什么有关系。

```bash
model_name meta-llama/Llama-3.1-8B-Instruct rank 1 update lm_head.weight torch.Size([128256, 4096]) from distributed time: 0.055s
fully update model_name meta-llama/Llama-3.1-8B-Instruct rank 1 parameter from distributed time: 0.055s
```

Well，看上去结果不太妙，这似乎是 python 编译器的作用域问题（我没学好编原，也就只知道这个词了）。


我们换种方式来 print 时间：

```python

            torch.cuda.synchronize()
            time_begin_fully_update = time.time()
            print(
                f"start to update model_name {model_name} rank {rank} parameter from distributed"
            )
            for parameter_name in state_dict_key_to_shape.keys():
                torch.cuda.synchronize()
                time_begin_single_update = time.time()
                cls.engine.update_parameter_from_distributed(
                    parameter_name,
                    dtype=torch.bfloat16,
                    shape=state_dict_key_to_shape[parameter_name],
                    empty_cache=True,
                )
                torch.cuda.synchronize()
                time_end_single_update = time.time()
                print(
                    f"model_name {model_name} rank {rank} update {parameter_name} {state_dict_key_to_shape[parameter_name]} from distributed time: {time_end_single_update - time_begin_single_update:.3f}s"
                )
            torch.cuda.synchronize()
            time_end_fully_update = time.time()
            print(
                f"fully update model_name {model_name} rank {rank} parameter from distributed time: {time_end_fully_update - time_begin_fully_update:.3f}s"
            )

```

这样应该不会相互覆盖了。拿到的结果很有趣：

```bash

model_name meta-llama/Llama-3.2-1B-Instruct rank 1 update model.embed_tokens.weight torch.Size([128256, 2048]) from distributed time: 1.620s

rank 0 broadcast model.embed_tokens.weight time: 1.612s

model_name meta-llama/Llama-3.2-1B-Instruct rank 1 update model.layers.0.self_attn.o_proj.weight torch.Size([2048, 2048]) from distributed time: 0.034s

rank 0 broadcast model.layers.0.self_attn.o_proj.weight time: 0.000s

model_name meta-llama/Llama-3.2-1B-Instruct rank 1 update model.layers.0.mlp.gate_proj.weight torch.Size([8192, 2048]) from distributed time: 0.032s

rank 0 broadcast model.layers.0.mlp.gate_proj.weight time: 0.000s

model_name meta-llama/Llama-3.2-1B-Instruct rank 1 update model.layers.1.self_attn.k_proj.weight torch.Size([512, 2048]) from distributed time: 0.031s

rank 0 broadcast model.layers.1.self_attn.k_proj.weight time: 0.000s
```

这些太玄学了，我一时捋不出问题，像极了高中时物理竞赛的同学做的物理实验报告...

但是我还是观察到了这么一件事情：

```bash
rank 0 init process group time: 44.275s
rank 1 init parameter update group time: 0.005s
```

有点逆天，process group 的创立绝对是同步的，但是两个 process group 的创建时间居然差了 44s。我感到非常费解，遂做如下测试：

<details>
<summary>process group 创建时间</summary>

```python
import time
import unittest
import torch
import torch.multiprocessing as mp
from sglang.srt.utils import init_custom_process_group
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
)

mp.set_start_method("spawn", force=True)

class TestProcessGroupInit(unittest.TestCase):
    @classmethod
    def init_process(cls, rank, world_size):
        torch.cuda.set_device(rank)
        
        if rank == 0:
            # 初始化进程组
            print(f"rank {rank} init custom process group")
            torch.cuda.synchronize()
            time_begin = time.time()
            group = init_custom_process_group(
                backend="nccl",
                init_method="tcp://localhost:65500",
                world_size=world_size,
                rank=rank,
                group_name="test_process_group",
            )
            
            torch.cuda.synchronize()
            time_end = time.time()
            print(f"rank {rank} init process group time: {time_end - time_begin:.3f}s")

        elif rank == 1:
            # 初始化引擎的进程组
            print(f"rank {rank} init parameter update group")
            torch.cuda.synchronize()
            time_begin = time.time()
            from sglang import Engine
            engine = Engine(
                model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,  # 使用小模型测试
                random_seed=42,
                base_gpu_id=rank,
                tp_size=1,
            )
            torch.cuda.synchronize()
            time_end = time.time()
            print(f"rank {rank} init engine time: {time_end - time_begin:.3f}s")
            torch.cuda.synchronize()
            time_begin = time.time()
            engine.init_parameter_update_group(
                master_address="localhost",
                master_port="65500",
                rank_offset=1,
                world_size=world_size,
                group_name="test_process_group",
                backend="nccl",
            )
            torch.cuda.synchronize()
            time_end = time.time()
            print(f"rank {rank} init process group time: {time_end - time_begin:.3f}s")
            
            engine.shutdown()

    def test_process_group_init(self):
        assert torch.cuda.device_count() >= 2, "需要至少2个GPU"
        
        torch.cuda.synchronize()
        time_begin = time.time()
        
        context = mp.spawn(
            self.init_process,
            args=(2,),  # world_size = 2
            nprocs=2,
            join=True
        )
        
        torch.cuda.synchronize()
        time_end = time.time()
        print(f"总耗时: {time_end - time_begin:.3f}s")

if __name__ == "__main__":
    unittest.main()
```

</details>

得到的结果如下：

```bash
rank 1 init engine time: 20.817s
rank 1 init process group time: 0.014s
rank 0 init process group time: 20.934s
```

okay，确实创建通讯组非常快，rank 0 卡住的原因是要和 rank 1 的 engine 同步，而 engine 启动耗时 20s，实际上 process group 的创建时间几乎可以忽略不计。

有了这个思路，我决定把我复杂的测例简化下，不读取参数，只测试更新时间，为了避免太多繁琐的同步影响我的测速：

<details>
<summary>只测试 broad cast and update 的时间</summary>

```python
import gc
import os
import time
import unittest
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM
import sglang as sgl
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
)
from sglang.srt.utils import init_custom_process_group
mp.set_start_method("spawn", force=True)

class TestParameterUpdateLatency(unittest.TestCase):
    @classmethod
    def init_process(cls, rank, world_size, param_queue, state_dict_key_to_shape, tp_size, model_name):
        torch.cuda.set_device(rank)
        print(f"Testing model: {model_name}")
        
        if rank == 0:
            os.environ["NCCL_CUMEM_ENABLE"] = "0"
            os.environ["NCCL_NVLS_ENABLE"] = "0"
            
            # 初始化进程组
            torch.cuda.synchronize()
            time_begin = time.time()
            cls.group = init_custom_process_group(
                backend="nccl",
                init_method="tcp://localhost:65500",
                world_size=world_size,
                rank=rank,
                group_name="test_parameter_update_group",
            )
            torch.cuda.synchronize()
            time_end = time.time()
            print(f"Rank {rank} init process group time: {time_end - time_begin:.3f}s")
            
            # 广播参数
            torch.cuda.synchronize()
            time_begin_broadcast = time.time()
            for name, shape in state_dict_key_to_shape.items():
                torch.cuda.synchronize()
                time_begin = time.time()
                weights = torch.ones(shape, dtype=torch.bfloat16, device=f"cuda:{rank}")
                torch.distributed.broadcast(weights, src=0, group=cls.group)
                torch.cuda.synchronize()
                time_end = time.time()
                print(f"Rank {rank} broadcast {name} {shape} time: {time_end - time_begin:.3f}s")
            torch.cuda.synchronize() 
            time_end_broadcast = time.time()
            print(f"Rank {rank} broadcast all parameters time: {time_end_broadcast - time_begin_broadcast:.3f}s")
            
            param_queue.put(("rank0_done", True))

        elif rank == 1:
            # 初始化引擎
            torch.cuda.synchronize()
            time_begin = time.time()
            cls.engine = sgl.Engine(
                model_path=model_name,
                random_seed=42,
                base_gpu_id=rank,
                tp_size=tp_size,
            )
            torch.cuda.synchronize()
            time_end = time.time()
            print(f"Rank {rank} init engine time: {time_end - time_begin:.3f}s")
            
            # 初始化参数更新组
            torch.cuda.synchronize()
            time_begin = time.time()
            cls.engine.init_parameter_update_group(
                master_address="localhost",
                master_port="65500",
                rank_offset=1,
                world_size=world_size,
                group_name="test_parameter_update_group",
                backend="nccl",
            )
            torch.cuda.synchronize()
            time_end = time.time()
            print(f"Rank {rank} init parameter update group time: {time_end - time_begin:.3f}s")
            
            # 更新参数并测量时间
            torch.cuda.synchronize()
            time_begin_update = time.time()
            for name, shape in state_dict_key_to_shape.items():
                torch.cuda.synchronize()
                time_begin = time.time()
                cls.engine.update_parameter_from_distributed(
                    name,
                    dtype=torch.bfloat16,
                    shape=shape,
                    empty_cache=True
                )
                torch.cuda.synchronize()
                time_end = time.time()
                print(f"Rank {rank} update {name} {shape} time: {time_end - time_begin:.3f}s")
            torch.cuda.synchronize()
            time_end_update = time.time()
            print(f"Rank {rank} update all parameters time: {time_end_update - time_begin_update:.3f}s")
            
            param_queue.put(("rank1_done", True))
            cls.engine.shutdown()

    @classmethod
    def setUpClass(cls):
        assert torch.cuda.device_count() >= 2, "At least 2 GPUs are required"
        cls.test_suits = [1]
        cls.model_names = [
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            DEFAULT_MODEL_NAME_FOR_TEST,
        ]

        if torch.cuda.device_count() >= 4:
            cls.test_suits.append(2)

        # 初始化每个模型的 state_dict_key_to_shape
        cls.model_state_dict_shapes = {}
        for model_name in cls.model_names:
            torch.cuda.synchronize()
            time_begin = time.time()
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype="bfloat16"
            ).to("cuda:0")
            state_dict = model.state_dict()
            cls.model_state_dict_shapes[model_name] = {
                key: state_dict[key].shape for key in state_dict.keys()
            }
            del model
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time_end = time.time()
            print(
                f"Initialize state dict shapes for {model_name} time: {time_end - time_begin:.3f}s"
            )

    def test_parameter_update_latency(self):
        for model_name in self.model_names:
            print(f"Testing model: {model_name}")
            state_dict_key_to_shape = self.model_state_dict_shapes[model_name]

            for tp_size in self.test_suits:
                print(f"test tp_size: {tp_size}")
                world_size = 1 + tp_size
                param_queue = mp.Queue()
                results = {}
                
                torch.cuda.synchronize()
                time_begin = time.time()
                
                context = mp.spawn(
                    self.init_process,
                    args=(world_size, param_queue, state_dict_key_to_shape, tp_size, model_name),
                    nprocs=2,
                    join=False
                )

                while len(results) < 2:
                    try:
                        key, value = param_queue.get(timeout=5)
                        results[key] = value
                    except Exception as e:
                        if all(not p.is_alive() for p in context.processes):
                            break

                context.join()
                torch.cuda.synchronize()
                time_end = time.time()
                print(f"Total time for {model_name}: {time_end - time_begin:.3f}s")
                
                if len(results) != 2:
                    raise RuntimeError(f"Expected 2 results but got {len(results)}")
                
                del context
                param_queue.close()
                param_queue.join_thread()
                gc.collect()
                torch.cuda.empty_cache()

if __name__ == "__main__":
    unittest.main()
```

</details>

这一次，我拿到很多有趣的事情：

1. update parameter 的时间和上次复杂的测例几乎一样；
2. ModelRunner 实际上的更新时间非常快，但是接口返回的速度很慢；

```bash
ModelRunner update model.layers.0.self_attn.q_proj.weight time: 0.001s
Rank 1 update model.layers.0.self_attn.q_proj.weight torch.Size([2048, 2048]) time: 0.033s
Rank 0 broadcast model.layers.0.self_attn.q_proj.weight torch.Size([2048, 2048]) time: 0.001s
```

3. `model.embed_tokens.weight torch.Size([128256, 2048])` 参数异常的慢，而且慢的很同步：

```bash
Rank 0 broadcast model.embed_tokens.weight torch.Size([128256, 2048]) time: 1.812s
Rank 1 update model.embed_tokens.weight torch.Size([128256, 2048]) time: 1.819s
ModelRunner update model.embed_tokens.weight time: 1.786s
```

4. `model.layers.12.mlp.up_proj.weight torch.Size([8192, 2048])` 在 Model Runner 上正常，broadcast 似乎卡顿了，而最后整体的 update 时间和其他 update 时间几乎一致：

```bash
ModelRunner update model.layers.12.mlp.up_proj.weight time: 0.001s
Rank 0 broadcast model.layers.12.mlp.up_proj.weight torch.Size([8192, 2048]) time: 0.162s
Rank 1 update model.layers.12.mlp.up_proj.weight torch.Size([8192, 2048]) time: 0.032s
```

`embed_tokens.weight` 和 `up_proj.weight` 不太好解决，但是我明显感受到了其实在 `ModelRunner` 上，broad cast 和 update 的时间几乎可以忽略不计，但是实际回传的时间却很长。那么，我们每一层都打印一次时间，看看究竟是哪儿慢了下来。具体来说，从 `Engine -> scheduler -> tp worker -> model runner` 的每一层都打印一次时间，看看究竟是哪儿慢了下来。

在这个过程中，我看到几行，瞬间就有了感觉：

```python
async def update_parameter_from_distributed(
    self,
    obj: UpdateParameterFromDistributedReqInput,
    request: Optional[fastapi.Request] = None,
):
    torch.cuda.synchronize()
    time_begin = time.time()
    if self.to_create_loop:
        self.create_handle_loop()
    if not self.model_update_lock.locked():

        async with self.model_update_lock:
            # wait for the previous update requests to finish
            for i in range(3):
                while len(self.rid_to_state) > 0:
                    await asyncio.sleep(0.001)
                # FIXME: We add some sleep here to avoid some race conditions.
                # We can use a read-write lock as a better fix.
                await asyncio.sleep(0.01)

            self.send_to_scheduler.send_pyobj(obj)
            self.parameter_update_result = asyncio.Future()

            if self.server_args.dp_size == 1:
                result = await self.parameter_update_result
                torch.cuda.synchronize()
                time_end = time.time()
                print(
                    f"In tokenizer manager: update parameter from distributed time: {obj.name} {obj.shape} {time_end - time_begin:.3f}s"
                )
                return result.success, result.message
            else:  # self.server_args.dp_size > 1
                self.parameter_update_tmp = []
                result = await self.parameter_update_result
                all_success = all([r.success for r in result])
                all_message = [r.message for r in result]
                all_message = " | ".join(all_message)
                return all_success, all_message

    else:
        logger.error(
            f"Another parameter update is in progress in tokenizer manager"
        )
        return (
            False,
            "Another parameter update is in progress. Please try again later.",
        )
```

这三个 `await asyncio.sleep(0.01)` 不是一眼导致了 `0.03` 的 update latency 吗？我试图去掉，并且 print 出来。果然，这次很快时间就降下来了：

```bash
fully update model_name meta-llama/Llama-3.2-1B-Instruct rank 1 parameter from distributed time: 2.202s
```

虽然速度快了很多，但是还是大于 1s，而且继续观察到了这个 `model.embed_tokens.weight torch.Size([128256, 2048])` 占据了超过 1.6s 的时间，甚至是从 broadcast 那一步就开始超过了 1.6s。是因为第一个参数的 broad cast 需要 init NCCL 很慢，还是就这个参数很慢呢？我们跳过这个参数，直接从 `[1:]` 开始，看看结果如何：


```bash
In server: update parameter from distributed time: model.layers.0.self_attn.q_proj.weight torch.Size([2048, 2048]) 0.000s
In tokenizer manager: update parameter from distributed time: model.layers.0.self_attn.q_proj.weight torch.Size([2048, 2048]) 1.726s
In server time function update parameter from distributed time: model.layers.0.self_attn.q_proj.weight torch.Size([2048, 2048]) 1.726s
model_name meta-llama/Llama-3.2-1B-Instruct rank 1 update model.layers.0.self_attn.q_proj.weight torch.Size([2048, 2048]) from distributed time: 1.727s
```

很有意思，就是第一个被广播的参数很慢，其他参数都很快。这是因为没有同步么？我决定加个 barrier 试试同步一次：

```bash
Rank 1 before barrier
Rank 1 after barrier
In server: update parameter from distributed time: model.embed_tokens.weight torch.Size([128256, 2048]) 0.000s
In tokenizer manager: update parameter from distributed time: model.embed_tokens.weight torch.Size([128256, 2048]) 1.444s
In server time function update parameter from distributed time: model.embed_tokens.weight torch.Size([128256, 2048]) 1.444s
Rank 1 update model.embed_tokens.weight torch.Size([128256, 2048]) time: 1.445s
```

看上去还是很寄，的确是第一次通讯的用时特别长。但是似乎也没那么寄，我转手问了下 gpt，貌似第一次通讯的建立一定是慢的，而我可以在 init process group 的时候就 barrier 一次（一次 barrier 实际上等价于一次小的 all reduce），看看此后的效果如何。

...

大功告成，在我的本地机器上，1B 模型更新时间降到了 0.5s 左右，8B 模型降到了 0.6s 左右。想见大部分的开销其实也不是通讯 😂

PS：在 process group init 之后马上 warm up 一次是非常常见的，然后我发现很有趣的事情，直接用 `dist.barrier()` 不指定 devices_id 的话，会在 CI 上因为 device error 卡死，但是本地不会，所以一个更好的尝试是：`dist.barrier(device_ids=[0], group=pg)`
