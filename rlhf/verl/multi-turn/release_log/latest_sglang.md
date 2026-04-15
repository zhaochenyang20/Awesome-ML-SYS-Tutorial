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
mkdir ~/.python
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

4. 测试 gsm8k：

```bash
cd ~/verl
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 拉取并预处理 gsm8k 数据集
python examples/data_preprocess/gsm8k_multiturn_w_tool.py

# 启动 8 卡训练
bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh
```

5. 测试 geo3k：

```bash
cd ~/verl
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m uv pip install qwen-vl-utils
python3 -m uv pip install mathruler

# 拉取并预处理 geo3k 数据集
python examples/data_preprocess/geo3k.py --local_dir ~/data/geo3k

# 启动 8 卡训练
bash examples/grpo_trainer/run_qwen2_5_vl-7b-sglang.sh
```

6. 测试 dapo：

注意 dapo 和先前的设置有所不同，因为 dapo 需要另一个 docker 来启动 sandboxfusion，所以需要回到宿主机上，单独启动 tool 服务器：

```bash
#启动 sandbox fusion （dapo tool call requirement）
docker run -it -p 8080:8080 volcengine/sandbox-fusion:server-20250609
```

此外，为了让 dapo 的训练脚本在一个单独的 docker 内进行，同时能够访问到宿主机上 sandboxfusion 的 8080 端口，我们需要在启动 docker 的时候额外添加 `--network=host`，也即步骤 1 的启动指令需要改为：

```bash
docker run -it --name h100_verl_{your_name} --gpus all \
    --shm-size 32g \
    -v {your_cache_path}:/root/.cache \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env "WANDB_API_KEY=$WANDB_API_KEY" \
    --network=host \
    --ipc=host \
    lmsysorg/sglang:latest \
    /bin/bash
```

之后照样 follow 步骤 2 3 的安装过程，安装完成后启动训练：

```bash
cd ~/verl
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 启动 8 卡训练
bash examples/sglang_multiturn/run_qwen3_4b_dapo_multiturn.sh
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

最后对这个问题的解决方式是，建议不要用虚拟环境，逆天 ray...
