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
```

4. 测试 DAPO:

```bash
cd ~/verl
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash examples/sglang_multiturn/run_qwen2_3b_dapo_multiturn.sh
```

5. 测试 gsm8k：

```bash
cd ~/verl
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 拉取并预处理 gsm8k 数据集
python examples/data_preprocess/gsm8k_multiturn_w_tool.py

# 启动 8 卡训练
bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh
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


