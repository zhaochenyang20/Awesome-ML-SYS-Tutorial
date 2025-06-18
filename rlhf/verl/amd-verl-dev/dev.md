# VeRL-SGLang dev (On AMD GPU)

## CodeBase
Upstream verl: 
```bash
git clone https://github.com/volcengine/verl.git
cd verl
git checkout ec59b8788cc16038889b0c35ca4aeb47bcd69093
```

## Launch Docker
```bash
docker run --rm -it \
  --device /dev/dri \
  --device /dev/kfd \
  -p 8265:8265 \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  -v $HOME/.ssh:/root/.ssh \
  -v $HOME:$HOME \
  --shm-size 128G \
  --name verl_vllm_upstream \
  -w $PWD \
  lmsysorg/sglang:v0.4.5-rocm630 \
  /bin/bash
```

## Setup Script
```bash
# verl
pip install "tensordict==0.6.2" --no-deps
pip install accelerate \
    codetiming \
    datasets \
    dill \
    hydra-core \
    liger-kernel \
    numpy \
    pandas \
    peft \
    "pyarrow>=15.0.0" \
    pylatexenc \
    "ray[default]>=2.10" \
    torchdata \
    transformers \
    wandb \
    orjson \
    pybind11
pip install -e . --no-deps

# Add torch_memory_saver
export HIPCC_COMPILE_FLAGS_APPEND="--amdgpu-target=gfx90a;gfx942 -D__HIP_PLATFORM_AMD__"
export CFLAGS="-D__HIP_PLATFORM_AMD__"
export CXXFLAGS="-D__HIP_PLATFORM_AMD__"
pip install git+https://github.com/ExtremeViscent/torch_memory_saver.git --no-deps
```

## Training Script
```bash
export HYDRA_FULL_ERROR=1
export RCCL_DEBUG=DEBUG
export RCCL_TRACE=1
export RCCL_TRACE_DUMP=1
KEY=""
wandb login $KEY
YOUR_PROJECT_NAME=r1-verl-upstream
YOUR_RUN_NAME=r1-training_test_del-upstream
GPUS_PER_NODE=2
export HIP_VISIBLE_DEVICES=0,1
export ROCR_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES

python3 examples/data_preprocess/gsm8k.py --local_dir ../data/gsm8k
python3 examples/data_preprocess/math_dataset.py --local_dir ../data/math
python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2.5-0.5B-Instruct')"

MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"
train_files="../data/gsm8k/train.parquet"
test_files="../data/gsm8k/test.parquet"

python3 -m verl.trainer.main_ppo \
   data.train_files="$train_files" \
   data.val_files="$test_files" \
   data.train_batch_size=1024 \
   data.max_prompt_length=1024 \
   data.max_response_length=256 \
   data.filter_overlong_prompts=True \
   data.truncation='error' \
   actor_rollout_ref.model.path=$MODEL_PATH \
   actor_rollout_ref.model.enable_gradient_checkpointing=True \
   actor_rollout_ref.actor.optim.lr=1e-6 \
   actor_rollout_ref.model.use_remove_padding=True \
   actor_rollout_ref.actor.ppo_mini_batch_size=256 \
   actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
   actor_rollout_ref.model.enable_gradient_checkpointing=True \
   actor_rollout_ref.actor.fsdp_config.param_offload=True \
   actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
   actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
   actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
   actor_rollout_ref.rollout.name=sglang \
   actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
   actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
   actor_rollout_ref.ref.fsdp_config.param_offload=True \
   critic.optim.lr=1e-5 \
   critic.model.use_remove_padding=True \
   critic.model.path=$MODEL_PATH \
   critic.model.enable_gradient_checkpointing=True \
   critic.ppo_micro_batch_size_per_gpu=8 \
   critic.model.fsdp_config.param_offload=True \
   critic.model.fsdp_config.optimizer_offload=True \
   algorithm.kl_ctrl.kl_coef=0.0001 \
   trainer.critic_warmup=0 \
   trainer.logger=['console'] \
   trainer.n_gpus_per_node=$GPUS_PER_NODE \
   trainer.nnodes=1 \
   trainer.save_freq=-1 \
   trainer.test_freq=10 \
   trainer.total_epochs=15
```


## Issue
- If I only run on the single, it can work
- When I run on multi-node, it will occure the following error:
```bash
(WorkerDict pid=53922) Actor use_remove_padding=True
Error executing job with overrides: ['data.train_files=../data/gsm8k/train.parquet', 'data.val_files=../data/gsm8k/test.parquet', 'data.train_batch_size=1024', 'data.max_prompt_length=1024', 'data.max_response_length=256', 'data.filter_overlong_prompts=True', 'data.truncation=error', 'actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct', 'actor_rollout_ref.model.enable_gradient_checkpointing=True', 'actor_rollout_ref.actor.optim.lr=1e-6', 'actor_rollout_ref.model.use_remove_padding=True', 'actor_rollout_ref.actor.ppo_mini_batch_size=256', 'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8', 'actor_rollout_ref.model.enable_gradient_checkpointing=True', 'actor_rollout_ref.actor.fsdp_config.param_offload=True', 'actor_rollout_ref.actor.fsdp_config.optimizer_offload=True', 'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16', 'actor_rollout_ref.rollout.tensor_model_parallel_size=1', 'actor_rollout_ref.rollout.name=sglang', 'actor_rollout_ref.rollout.gpu_memory_utilization=0.6', 'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16', 'actor_rollout_ref.ref.fsdp_config.param_offload=True', 'critic.optim.lr=1e-5', 'critic.model.use_remove_padding=True', 'critic.model.path=Qwen/Qwen2.5-0.5B-Instruct', 'critic.model.enable_gradient_checkpointing=True', 'critic.ppo_micro_batch_size_per_gpu=8', 'critic.model.fsdp_config.param_offload=True', 'critic.model.fsdp_config.optimizer_offload=True', 'algorithm.kl_ctrl.kl_coef=0.0001', 'trainer.critic_warmup=0', 'trainer.logger=[console]', 'trainer.n_gpus_per_node=2', 'trainer.nnodes=1', 'trainer.save_freq=-1', 'trainer.test_freq=10', 'trainer.total_epochs=15']
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/yushensu/projects/verl_upstream/verl/trainer/main_ppo.py", line 198, in <module>
    main()
  File "/usr/local/lib/python3.12/dist-packages/hydra/main.py", line 94, in decorated_main
    _run_hydra(
  File "/usr/local/lib/python3.12/dist-packages/hydra/_internal/utils.py", line 394, in _run_hydra
    _run_app(
  File "/usr/local/lib/python3.12/dist-packages/hydra/_internal/utils.py", line 457, in _run_app
    run_and_report(
  File "/usr/local/lib/python3.12/dist-packages/hydra/_internal/utils.py", line 223, in run_and_report
    raise ex
  File "/usr/local/lib/python3.12/dist-packages/hydra/_internal/utils.py", line 220, in run_and_report
    return func()
           ^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/hydra/_internal/utils.py", line 458, in <lambda>
    lambda: hydra.run(
            ^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/hydra/_internal/hydra.py", line 132, in run
    _ = ret.return_value
        ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/hydra/core/utils.py", line 260, in return_value
    raise self._return_value
  File "/usr/local/lib/python3.12/dist-packages/hydra/core/utils.py", line 186, in run_job
    ret.return_value = task_function(task_cfg)
                       ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yushensu/projects/verl_upstream/verl/trainer/main_ppo.py", line 59, in main
    run_ppo(config)
  File "/home/yushensu/projects/verl_upstream/verl/trainer/main_ppo.py", line 77, in run_ppo
    ray.get(runner.run.remote(config))
  File "/usr/local/lib/python3.12/dist-packages/ray/_private/auto_init_hook.py", line 21, in auto_init_wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/ray/_private/client_mode_hook.py", line 103, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/ray/_private/worker.py", line 2755, in get
    values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/ray/_private/worker.py", line 906, in get_objects
    raise value.as_instanceof_cause()

  File "/home/yushensu/projects/verl_upstream/verl/single_controller/ray/base.py", line 439, in func
    return getattr(self.worker_dict[key], name)(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yushensu/projects/verl_upstream/verl/single_controller/base/decorator.py", line 409, in inner
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/yushensu/projects/verl_upstream/verl/workers/fsdp_workers.py", line 419, in init_model
    self.rollout, self.rollout_sharding_manager = self._build_rollout(
                                                  ^^^^^^^^^^^^^^^^^^^^
  File "/home/yushensu/projects/verl_upstream/verl/workers/fsdp_workers.py", line 348, in _build_rollout
    from verl.workers.rollout.sglang_rollout import SGLangRollout
  File "/home/yushensu/projects/verl_upstream/verl/workers/rollout/sglang_rollout/__init__.py", line 14, in <module>
    from .sglang_rollout import SGLangRollout
  File "/home/yushensu/projects/verl_upstream/verl/workers/rollout/sglang_rollout/sglang_rollout.py", line 38, in <module>
    from sglang.srt.entrypoints.verl_engine import VerlEngine
  File "/sgl-workspace/sglang/python/sglang/srt/entrypoints/verl_engine.py", line 22, in <module>
    from sglang.srt.entrypoints.http_server_engine import HttpServerEngineAdapter
  File "/sgl-workspace/sglang/python/sglang/srt/entrypoints/http_server_engine.py", line 15, in <module>
    from sglang.srt.entrypoints.http_server import launch_server
  File "/sgl-workspace/sglang/python/sglang/srt/entrypoints/http_server.py", line 32, in <module>
    from sglang.srt.model_executor.model_runner import LocalSerializedTensor
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py", line 30, in <module>
    from sglang.srt.configs.model_config import AttentionArch, ModelConfig
  File "/sgl-workspace/sglang/python/sglang/srt/configs/model_config.py", line 26, in <module>
    from sglang.srt.layers.quantization import QUANTIZATION_METHODS
  File "/sgl-workspace/sglang/python/sglang/srt/layers/quantization/__init__.py", line 60, in <module>
    from sglang.srt.layers.quantization.fp8 import Fp8Config
  File "/sgl-workspace/sglang/python/sglang/srt/layers/quantization/fp8.py", line 74, in <module>
    from aiter import ActivationType
  File "/sgl-workspace/aiter/aiter/__init__.py", line 15, in <module>
    from .ops.norm import *
  File "/sgl-workspace/aiter/aiter/ops/norm.py", line 6, in <module>
    from ..jit.core import compile_ops, CK_DIR, AITER_CSRC_DIR, AITER_ROOT_DIR, AITER_CORE_DIR
  File "/sgl-workspace/aiter/aiter/jit/core.py", line 77, in <module>
    shutil.rmtree(f'{bd_dir}/ck/library')
  File "/usr/lib/python3.12/shutil.py", line 759, in rmtree
    _rmtree_safe_fd(stack, onexc)
  File "/usr/lib/python3.12/shutil.py", line 703, in _rmtree_safe_fd
    onexc(func, path, err)
  File "/usr/lib/python3.12/shutil.py", line 662, in _rmtree_safe_fd
    os.rmdir(name, dir_fd=dirfd)
OSError: [Errno 39] Directory not empty: '/sgl-workspace/aiter/aiter/jit/build/ck/library/src/tensor_operation_instance/gpu'
```
