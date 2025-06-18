# Latency Accelerate for Weight Updates

## [English version](./readme.md) | [简体中文](./readme-CN.md)

## Preface

This is a debug note, so it describes my debugging process in detail. However, the conclusion is actually very simple and can be summarized in a few sentences:

1. To accurately measure GPU latency, we must add numerous `torch.cuda.synchronize()` statements before and after the timing code. Otherwise, the CPU might race ahead and print results early, while the GPU is still stuck processing the previous operations. Specifically:

```python
torch.cuda.synchronize()
time_begin = time.time()
# ...
torch.cuda.synchronize()
time_end = time.time()
print(f"latency: {time_end - time_begin:.3f}s")
```

2. To correctly use `dist.barrier()`, it is best to specify `device_ids`. Otherwise, in CI, it may mysteriously hang due to a device error.

## Background

After much effort, I finally managed to implement the `update_parameter_from_distributed` interface. According to my advisor, the OpenRLHF implementation based on vLLM does not exceed 50 lines. In a way, my implementation is not particularly complex; I just struggled for two weeks due to a lack of experience. Finally, on the day before Thanksgiving 2024, I successfully implemented the following three interfaces from top to bottom:

1. `init_parameter_update_group`
2. `update_parameter_from_distributed`
3. `get_weights_by_parameter_name`

These three functions serve a single purpose. The first function is used to establish a process group. We assume that the weights passed by the Training Engine are stored on rank 0 (even though rank 0 might not be able to store the entire model, the training engine can always distribute weights from rank 0). Then, our SGLang server will establish a process group with rank 0, broadcast the weights from rank 0, and load them onto all tensor parallel devices. Finally, we use `get_weights_by_parameter_name` to check whether the SGLang inference engine has been updated correctly.

It is important to note that the training engine does not necessarily have to store the model in Hugging Face format. In fact, in large-scale industrial applications, the training engine typically uses its own model format throughout the training process and only converts the checkpoint to the Hugging Face format upon completion for release. However, as an academic-oriented open-source product, OpenRLHF uses Hugging Face models as a common protocol.

<details>
<summary>Why are there two engines?</summary>

Here’s an obvious question: Why does the RLHF process require both a training engine and an inference engine? There are many mainstream options for the former, such as DeepSpeed. As for the latter, we want to support SGLang. In other words, why can’t we use the training engine for inference or the inference engine for training?

1. The training engine only performs forward passes, but once logits are obtained, whether for evaluation or rollout, the model must perform decoding. Decoding is a complex process. SGLang’s main contributions lie in continuous batching and KV cache management, making it naturally suitable for evaluation or rollout in the entire training pipeline.

2. Conversely, the inference engine does not perform backpropagation, so it obviously cannot be used for training. However, can the inference engine be used to compute KL divergence? The answer is no because KL divergence requires high precision in logits, which the inference engine currently does not meet (I am still investigating why this is the case).

After implementing these three interfaces, I finally wrote a unit test by hand, and while the test passed successfully, the efficiency was far from ideal.

## Test Results

<details>
<summary>Unit Test</summary>

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

To summarize the test logic, for 8B Llama 3.1 and 1B Llama 3.2, we evaluate correctness and efficiency when the tensor parallelism (TP) of the SGLang engine is set to 1 and 2:

1. Rank 0 (Simulating the Training Engine)
- Loads the instruct model and base model using Hugging Face.
- Extracts representative parameters as verification samples (each type of parameter is randomly sampled).
- Initializes the process group.
- Broadcasts all parameters of the base model.

2. Rank 1 (SGLang Inference Engine)
- Initializes the engine and loads the instruct model.
- Extracts representative parameters from the instruct model.
- Initializes the parameter update group.
- Receives and updates all parameters.
- Retrieves updated base model parameters for verification.

On an 8x H100 system, the entire test took 431.264s, which left me very confused. The actual update function is as follows:

<details>
<summary>Code for the weight update function</summary>

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

Each step shouldn’t be slow, but something strange happened. In my unit test, I logged the update time for each parameter using the following lines:

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

At the same time, in the lowest-level call to the  `update_parameter_from_distributed` function, I attempted to log the execution time for each step:

<details>
<summary>Testing the execution time of each update step</summary>

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

For the entire update function, I suspected almost every step. First the assertion checks at the beginning, then creating the empty tensor for weights, then broadcasting, and finally loading the weights.

Surprisingly, each individual step took 0.000s, yet the return time in the unit test was 0.032s. Additionally, the single-step update times for 8B and 1B models were identical. This is fascinating - it means updating the entire 1B model took 7.047s. Considering that a full H100 NV Link bandwidth is measured in TB/s, and the weights of a 1B model in bf16 are only about 2GB, this time consumption is clearly unreasonable.

So, where did all the time go?

## Where Did All the Time Go?

Good question. Over eight thousand days and nights have already passed in my life, and my entire lifespan is likely only about thirty thousand days. In middle school, a math competition teacher who taught me briefly used to say, "Life is just over thirty thousand days. I was young once too, and whoops, now I'm old..." Ten years ago, I never felt the passage of time, but now at twenty-two, thinking about the absurdity and emptiness of the human world, I realize that time is humanity's punishment. On one hand, I'm mindful that I only have this short life, and pleasing others is undoubtedly wasting my life. On the other hand, if both the beginning and end of my life are emptiness, what meaning does my life really have?

At the very least, figuring out how to reduce this 7.047s transmission overhead to under 1s is part of what I consider the meaning of life.

I reasonably suspect these overheads might come from:

1. `https` requests being too slow: In sglang's design pattern, there are two layers of `https` requests - one where the top-level `RunTime` calls the `tokenizer manager` through fastapi, and another where the tokenizer manager passes requests to `scheduler -> tp worker -> model runner` through another fastapi https request.

2. Python function call overhead being too large: If each step in Model Runner's `update_parameter_from_distributed` is 0.000s, then going top-down from `RunTime` to `tokenizer manager` to `scheduler -> tp worker -> model runner`, is there significant overhead in passing requests between layers? Which layer significantly increases the overhead?

3. Not updating parameters asynchronously: Since `update_parameter_from_distributed` doesn't repeatedly write to the same weights, asynchronous updates seem like a solution.

4. Being in a blocking state during updates: Perhaps we should try just launching kernels so everything can overlap (as suggested by my advisor).

5. NCCL being too slow: I think this is unlikely since my test machine is a full-spec H100 provided by NVIDIA.

Regardless, I'll first run this test:

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

Let me see what the transmission efficiency is actually related to.

```bash
model_name meta-llama/Llama-3.1-8B-Instruct rank 1 update lm_head.weight torch.Size([128256, 4096]) from distributed time: 0.055s
fully update model_name meta-llama/Llama-3.1-8B-Instruct rank 1 parameter from distributed time: 0.055s
```

Well, the results don't look good. This seems to be a scope issue with the Python compiler (I didn't study compiler principles well, so I only know this term).

Let's try a different way to print the time:

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

This way they shouldn't overwrite each other. The results are interesting:

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

These results are too mysterious - I can't figure out the problem right away. It reminds me of physics experiment reports written by high school physics competition students...

But I still observed one thing:

```bash
rank 0 init process group time: 44.275s
rank 1 init parameter update group time: 0.005s
```

This is incredible - creating a process group is definitely synchronous, but the creation times of the two process groups differ by 44s. I'm very confused, so I did the following test:

<details>
<summary>Process group creation time</summary>

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

The results are as follows:

```bash
rank 1 init engine time: 20.817s
rank 1 init process group time: 0.014s
rank 0 init process group time: 20.934s
```

Okay, creating communication groups is indeed very fast. The reason rank 0 got stuck is that it needs to synchronize with rank 1's engine, and starting the engine takes 20s. In reality, the time to create the process group is almost negligible.

With this idea, I decided to simplify my complex test case by not reading parameters and only testing update time, to avoid having too many complicated synchronizations affecting my speed measurements:

<details>
<summary>Only testing broadcast and update time</summary>

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

This time, I discovered many interesting things:

1. The update parameter time is almost the same as in the previous complex test case.
2. The actual update time in ModelRunner is very fast, but the interface return speed is slow.

```bash
ModelRunner update model.layers.0.self_attn.q_proj.weight time: 0.001s
Rank 1 update model.layers.0.self_attn.q_proj.weight torch.Size([2048, 2048]) time: 0.033s
Rank 0 broadcast model.layers.0.self_attn.q_proj.weight torch.Size([2048, 2048]) time: 0.001s
```

3. The `model.embed_tokens.weight torch.Size([128256, 2048])` parameter is unusually slow, and the slowness is very synchronized:

```bash
Rank 0 broadcast model.embed_tokens.weight torch.Size([128256, 2048]) time: 1.812s
Rank 1 update model.embed_tokens.weight torch.Size([128256, 2048]) time: 1.819s
ModelRunner update model.embed_tokens.weight time: 1.786s
```

4. `model.layers.12.mlp.up_proj.weight torch.Size([8192, 2048])` is normal on the Model Runner, but the broadcast seems to have stalled, while the overall update time is almost the same as other update times:

```bash
ModelRunner update model.layers.12.mlp.up_proj.weight time: 0.001s
Rank 0 broadcast model.layers.12.mlp.up_proj.weight torch.Size([8192, 2048]) time: 0.162s
Rank 1 update model.layers.12.mlp.up_proj.weight torch.Size([8192, 2048]) time: 0.032s
```

The `embed_tokens.weight` and `up_proj.weight` issues aren't easy to solve, but I clearly sensed that on the `ModelRunner`, the broadcast and update times are almost negligible, yet the actual return time is quite long. So, I decided to print the time at each layer to see exactly where the slowdown occurs. Specifically, I printed timing data at each layer from `Engine -> scheduler -> tp worker -> model runner` to identify the bottleneck.

During this process, I saw a few lines that immediately gave me a clue:

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

Aren't these three `await asyncio.sleep(0.01)` statements the obvious cause of the `0.03` update latency? I tried removing them and printing the results. Sure enough, the time quickly decreased:

```bash
fully update model_name meta-llama/Llama-3.2-1B-Instruct rank 1 parameter from distributed time: 2.202s
```

Although the speed improved significantly, it's still over 1s, and I continued to observe that `model.embed_tokens.weight torch.Size([128256, 2048])` took over 1.6s, starting from the broadcast step. Is this because the first parameter broadcast needs to initialize NCCL which is slow, or is just this parameter slow? Let's skip this parameter and start directly from `[1:]` to see the results:

```bash
In server: update parameter from distributed time: model.layers.0.self_attn.q_proj.weight torch.Size([2048, 2048]) 0.000s
In tokenizer manager: update parameter from distributed time: model.layers.0.self_attn.q_proj.weight torch.Size([2048, 2048]) 1.726s
In server time function update parameter from distributed time: model.layers.0.self_attn.q_proj.weight torch.Size([2048, 2048]) 1.726s
model_name meta-llama/Llama-3.2-1B-Instruct rank 1 update model.layers.0.self_attn.q_proj.weight torch.Size([2048, 2048]) from distributed time: 1.727s
```

Very interesting - just the first broadcast parameter is slow, while all others are fast. Is this because there's no synchronization? I decided to add a barrier to try synchronizing once:

```bash
Rank 1 before barrier
Rank 1 after barrier
In server: update parameter from distributed time: model.embed_tokens.weight torch.Size([128256, 2048]) 0.000s
In tokenizer manager: update parameter from distributed time: model.embed_tokens.weight torch.Size([128256, 2048]) 1.444s
In server time function update parameter from distributed time: model.embed_tokens.weight torch.Size([128256, 2048]) 1.444s
Rank 1 update model.embed_tokens.weight torch.Size([128256, 2048]) time: 1.445s
```

It still looks problematic - the first communication indeed takes an especially long time. But perhaps it's not that bad. I quickly asked GPT, and it seems the first communication establishment is inevitably slow, but I could add a barrier right after initializing the process group (a barrier is essentially equivalent to a small all-reduce operation) to see how that affects subsequent performance.

...

Mission accomplished! On my local machine, the update time for the 1B model decreased to around 0.5s, and for the 8B model to around 0.6s. As it turns out, most of the overhead wasn't actually from communication 😂

PS: It's very common to warm up once immediately after process group initialization. Then I discovered something interesting: using `dist.barrier()` without specifying `device_ids` will hang in CI due to device errors, but this doesn't happen locally. So a better approach is: `dist.barrier(device_ids=[0], group=pg)`