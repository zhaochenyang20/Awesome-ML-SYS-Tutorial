# 启用 verl 的 agent loop feature

<!-- 在我们最早发布的 [multi-turn RL](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/release_log/verl-multiturn-rollout-Release_ZH.md) 中，我们的 tool call 状态管理是在 SGLang rollout 内部管理的。尽管取得了大量社区用户的认可，但是由于 rollout 和 tool call 管理糅合在一起，长期来看不完全便于维护。此外，在我们最初的设计中，multi-turn 的每个 step 都会调用一次 [`_preprocess_prompt_to_async_rollout_requests`](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/code-walk-through/readme-2.md#_preprocess_prompt_to_async_rollout_requests)来做预处理，而这一部分预处理其实和 step 是无关的。agent loop feature 将 tool call 的管理从 rollout engine 内抽离出来，rollout engine 只是向上提供 token in token out 的接口即可。具体的代码解析将在本文的后半部分；前半部分将介绍如何启用 agent loop 功能。 -->


## Quick Start

简单来说，目前只需要修改两处配置即可启用 agent loop feature：

1. 在启动训练的 bash 脚本中加入 `actor_rollout_ref.rollout.mode=async` 并确保 `actor_rollout_ref.rollout.multi_turn.enable=true`；
2. 在数据集处理脚本中对数据集新增一列 `agent_name`，在 `map_fn` 中补充即可。

我们接下来提供一套逐步的复现过程：这会依赖于最新的 verl 和最新版本的 sglang。注意到 verl 虽然在 [`setup.py`](https://github.com/volcengine/verl/blob/main/setup.py) 还在依赖 sglang 0.4.6.post5，但这是因为 verl 里面的 transformers 依赖被 qwen2.5 vl 在 flash-attn 新版上的 bug block 住了。本身 verl 已经可以启用更高级版本的 sglang 了，而且还可以用上 [multi-turn wake up](https://hebiao064.github.io/rl-memory-management) 的强大 feature。

### 创建新的 docker

如果你本地能够稳定运行 verl，那么大概率不需要新的 docker。用 docker 只是方便我们确定实验能够被严格复现。

使用前需要配置好 `WANDB_API_KEY`，参考[这个过程](https://community.wandb.ai/t/where-can-i-find-the-api-token-for-my-project/7914)。

```bash
# 如果你的系统没有配置过 HF_TOKEN 和 WANDB_API_KEY，请先配置好
docker run -it --name verl_{your_name} --gpus all \
    --shm-size 32g \
    -v {your_cache_path}:/root/.cache \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env "WANDB_API_KEY=$WANDB_API_KEY" \
    --ipc=host \
    lmsysorg/sglang:latest \
    /bin/bash
```

进入 docker 后，检查被映射的环境变量：

```bash
echo $HF_TOKEN
echo $WANDB_API_KEY
```

以后每次从 docker 里面 exit 出来，再用这个指令可以重启：

```bash
docker start -i verl_{your_name}
```

### 基于源码安装 SGLang

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

先安装 veRL，再安装 SGLang：

```bash
cd ~
git clone https://github.com/volcengine/verl.git
cd verl
python3 -m uv pip install -e ".[sglang,geo]"
python3 -m uv pip install -r ./requirements.txt
```

这个过程中的 flash-attn 安装会遇到这个报错：

```bash
Resolved 130 packages in 1.96s
  × Failed to build `flash-attn==2.8.1`
  ├─▶ The build backend returned an error
  ╰─▶ Call to `setuptools.build_meta:__legacy__.build_wheel` failed (exit status: 1)
```

按照下面的步骤 fix 即可：

```bash
python3 -m uv pip install wheel
python3 -m uv pip install -r ./requirements.txt --no-build-isolation
```

然后安装 SGLang upstream：

```bash
cd ~
git clone https://github.com/sgl-project/sglang.git
cd sglang
python3 -m uv pip install --upgrade pip
python3 -m uv pip install -e "python[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python
```

额外安装 vllm 和 weave 的依赖，用作可视化：

```bash
python3 -m uv pip install vllm==0.9.1
python3 -m uv pip install weave
```

### 修改并运行

我们可以通过对现有脚本进行简单修改，在运行脚本中启用 `multi_turn` 和 `async rollout`，在数据集处理脚本中的 `def make_map_fn(split)` 增加一列 `agent_name`。

打开你 docker 里面的 `~/verl/examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh` 文件，去掉结尾一行的 `$@`，更改如下参数：

``` bash
    # 注意去掉原本 total_epochs 这行结尾的 $@
    # 不要把这些两行注释也写进去，否则会报错
    trainer.total_epochs=15 \
    actor_rollout_ref.rollout.trace.backend=weave \
    actor_rollout_ref.rollout.trace.token2text=True \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.enable=true
```

在 `~/verl/examples/data_preprocess/gsm8k_multiturn_w_tool.py` 中追加 `"agent_name": "tool_agent"`

```python
def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            data = {
                "data_source": data_source,
                # new column for weave trace
                "agent_name": "tool_agent",
                "prompt": [
                    {
                        #...
                    }
                ]
            }
            return data

        return process_fn
```

接下来测试即可：

```bash
cd ~/verl
python3 -m uv pip install .
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 拉取并预处理 gsm8k 数据集
python examples/data_preprocess/gsm8k_multiturn_w_tool.py
```

启动 8 卡训练即可。

```bash
bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh
```

### Debug

- 如果你在启动 bash 后发现了这个错误：

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
```
- 如果你遇到了下面这个错误:
```bash
File "/root/.python/verl-sglang/lib/python3.12/site-packages/triton/runtime/driver.py", line 8, in _create _driverraise RuntimeError(f"flen(actives)) active drivers ( factives,). There should only be one."RuntimeError: 0 active drivers ([]). There should only be one(MorkerDict pid-319609) MARMING 07-25 04:31:15 [en override.py:17) WCCL CUMEM EMABLE is set to 0, skipping override. This may increase menory overhead with cudagraph+allreduce: https://github.con/WVIDIA/nccl/issues/1234 [repeated 5x across cluster)
```
请降级 triton 版本([参考链接](https://github.com/triton-inference-server/server/issues/8007)):
```bash
uv pip install triton==3.1.0
```
## Code-Walk-Through

