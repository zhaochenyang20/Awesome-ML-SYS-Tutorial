# 快速 set up 最新的 verl-sglang

这个文档记录如何快速安装最新的 verl-sglang。

1. 创建新的 docker（如果熟悉这套安装，可以跳过）：

使用前需要配置好 `WANDB_API_KEY`，参考[这个过程](https://community.wandb.ai/t/where-can-i-find-the-api-token-for-my-project/7914)。

```bash
# 如果你的系统没有配置过 HF_TOKEN 和 WANDB_API_KEY，请先配置好
docker run -it --name h100_verl_{your_name} --gpus all \
    --shm-size 32g \
    -v {your_cache_path}:/root/.cache \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env "WANDB_API_KEY=$WANDB_API_KEY" \
    --ipc=host \
    lmsysorg/sglang:latest \
    /bin/bash
```

进入 docker 后，可以查看被映射的环境变量：

```bash
echo $HF_TOKEN
echo $WANDB_API_KEY
```

以后每次从 docker 里面 exit 出来，再用这个指令可以重启：

```bash
docker start -i h100_verl_{your_name}
```

2. 基于源码安装 verl-sglang

配置 python 环境：

```bash
mkdir -p /tmp
chmod 1777 /tmp
apt update
apt install -y python3.10 python3.10-venv
python3 -m ensurepip --upgrade
python3 -m venv ~/.python/verl-sglang
source ~/.python/verl-sglang/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade uv
```

3. 安装 verl-sglang：

```bash
cd ~
git clone https://github.com/volcengine/verl.git
cd verl

python3 -m uv pip install -e ".[sglang]" --prerelease=allow
python3 -m uv pip install -r ./requirements.txt --no-build-isolation
python3 -m uv pip install torch_memory_saver
# to avoid vllm registration error with transformers 4.54.0, install it manually
python3 -m uv pip install vllm==0.10.0 --no-build-isolation
```

#### 测试 gsm8k：

```bash
cd ~/verl
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 拉取并预处理 gsm8k 数据集
python examples/data_preprocess/gsm8k_multiturn_w_tool.py

# 启动 8 卡训练
bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh
```

#### 测试 geo3k：

```bash
cd ~/verl
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 拉取并预处理 geo3k 数据集
python examples/data_preprocess/geo3k.py --local_dir ~/data/geo3k

# 启动 8 卡训练
bash examples/grpo_trainer/run_qwen2_5_vl-7b-sglang.sh
```

#### 测试dapo

对dapo的测试需要我们进行sft训练，然后进行rlhf训练。
我们有一个基于Qwen-3-4B-Instruct-2507 SFT训练好的模型位于font-info/qwen3-4b-sft https://huggingface.co/font-info/qwen3-4b-sft/tree/main ，可以直接跳过sft步骤，直接使用这个模型进行rlhf。如果要进行sft，可以参考以下步骤：


1. 进行sft训练：

```bash
#download sft data
huggingface-cli download JoeYing/ReTool-SFT --repo-type dataset --local-dir ./ReTool-SFT

#data preprocessing
python3 recipe/retool/retool_sft_preprocess.py

#sft training
bash recipe/retool/run_qwen2-32b_sft.sh
```

2. 启动sandbox fusion 修改dapo的tool config：

```bash
#启动sandbox fusion （dapo tool call requirement）
docker run -it -p 8080:8080 volcengine/sandbox-fusion:server-20250609

#修改dapo的tool config
#修改https://github.com/volcengine/verl/blob/main/recipe/retool/sandbox_fusion_tool_config.yaml：
sandbox_fusion_url: "http://localhost:8080/run_code"

```
3. rlhf

需要注意的是，retool中的reward[函数](https://github.com/volcengine/verl/blob/main/recipe/retool/retool.py)为：
回答正确，则reward为1，否则为min(0, -1+ (num_turns - 2) / 2 * 0.1)

可以看到，随着轮数的增多，回答错误的reward也会增多，直到达到0。如果不修改reward函数，则模型可能会出现假收敛，即模型只学会了不断调用tool增大max_turns数来提高reward，看似收敛了但是模型没有真的回答对问题。
可以通过修改reward函数来解决这个问题，比如回答错误，则reward直接为-1，或者设置回答错误的情况下，reward最大为-0.5.

参照以下脚本进行训练.

```bash

set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

pip install --upgrade "huggingface-hub>=0.34.0"
hf download \
    BytedTsinghua-SIA/DAPO-Math-17k \
    --repo-type dataset \
    --local-dir $HOME/data/BytedTsinghua-SIA/DAPO-Math-17k


hf download \
    Maxwell-Jia/AIME_2024 \
    --repo-type dataset \
    --local-dir $HOME/data/Maxwell-Jia/AIME_2024


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files=$HOME/data/BytedTsinghua-SIA/DAPO-Math-17k \
    data.val_files=$HOME/data/Maxwell-Jia/AIME_2024 \
    data.return_raw_chat=True \
    data.train_batch_size=32 \
    data.max_prompt_length=2048 \
    data.max_response_length=5000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.custom_cls.path=$PROJECT_DIR/recipe/retool/retool.py \
    data.custom_cls.name=CustomRLHFDataset \
    custom_reward_function.path=$PROJECT_DIR/recipe/retool/retool.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=font-info/qwen3-4b-sft \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    algorithm.dynamic_filter.enable=False \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=1024 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.multi_stage_wake_up=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=16 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=16 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$PROJECT_DIR/recipe/retool/sandbox_fusion_tool_config.yaml \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=sglang-dapo-multiturn \
    trainer.experiment_name=qwen2_5-3b_dapo_multiturn \
    trainer.n_gpus_per_node=8 \
    trainer.log_val_generations=20 \
    trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    trainer.total_epochs=15 \
    $@

```





## Debug

如果你在启动 bash 后发现了这个错误：

```bash
raise ValueError(f"Feature type '{_type}' not found. Available feature types: {list(_FEATURE_TYPES.keys())}")
ValueError: Feature type 'List' not found. Available feature types: ['Value', 'ClassLabel', 'Translation', 'TranslationVariableLanguages', 'LargeList', 'Sequence', 'Array2D', 'Array3D', 'Array4D', 'Array5D', 'Audio', 'Image', 'Video', 'Pdf']
```

其实这不是实际的报错，这个报错让我费解了非常非常久，我仔细看了 log 才发现问题，其实可以向上看几行报错。在报错栈最开始的地方，报错的 python 环境是 `/root/.python/verl-sglang/lib/python3.10`，结果到了栈底部成了 `/usr/local/lib/python3.10`。毫无疑问，是 python 环境错位了；

1. 主进程使用虚拟环境：`/root/.python/verl-sglang/lib/python3.10/site-packages/`
2. Ray worker 进程使用系统 Python：`/usr/local/lib/python3.10/dist-packages/`

最后对这个问题的解决方式是修改 `verl/trainer/constants_ppo.py` 文件，直接改为：

```python
import os
import sys

# 获取当前Python解释器路径和虚拟环境路径
python_executable = sys.executable
virtual_env = os.environ.get("VIRTUAL_ENV", "")
python_path = os.environ.get("PYTHONPATH", "")

# 如果当前在虚拟环境中，确保包含虚拟环境的site-packages
if virtual_env:
    site_packages = os.path.join(virtual_env, "lib", "python3.10", "site-packages")
    if site_packages not in python_path:
        python_path = f"{site_packages}:{python_path}" if python_path else site_packages

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        # 添加Python环境配置
        "PYTHONPATH": python_path,
        "VIRTUAL_ENV": virtual_env,
    },
    # 指定Python解释器
    "python": python_executable,
}

def get_ppo_ray_runtime_env():
    """
    A filter function to return the PPO Ray runtime environment.
    To avoid repeat of some environment variables that are already set.
    """
    runtime_env = {"env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy()}
    for key in list(runtime_env["env_vars"].keys()):
        if os.environ.get(key) is not None:
            runtime_env["env_vars"].pop(key, None)
    return runtime_env
```


