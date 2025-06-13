# veRL-SGLang：Train Multimodal Model with Multi-Turn RL to Reason and Call Tool

大家好，SGLang 社区, Amazon SF AGI Lab 和 PolyU 成员在先前开源的 [multi-turn RL](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/release_log/verl-multiturn-rollout-Release_ZH.md) 加入了多模态支持，欢迎大家上手体验，共同开发。具体来说，我们实现了如下的功能：

- 先前 SGLang 社区已经实现了工具调用，支持在 Actor rollout 期间，让模型能够调用特定工具，并将返回结果无缝地融合到训练流程中。
- 现在我们进一步为 Multi-Turn RL 新增了多模态输入，使模型能够在 Actor rollout 阶段处理多模态数据。
- 我们正在加入对于工具生成图像的支持，请尽情期待

[PR: volcengine/verl#2014](https://github.com/volcengine/verl/pull/2014)

[训练曲线 wandb](tbd)

Project Member:

- Nan Jiang, Congkai Xie (Author)
- Chenyang Zhao (PM)
- Xiang Long (Reviewer, PM)

感谢贡献！

## 快速复现

### 创建新的 Docker 容器

```bash
docker run \
    -it \
    --shm-size 32g \
    --gpus all \
    -v {Huggingface-Cache-Path}:/root/.cache \
    --ipc=host \
    --network=host \
    --privileged \
    --name sglang_{your-name} \
    lmsysorg/sglang:dev \
    /bin/zsh
```

如果退出容器后需要重新启动：

```bash 
docker start -i sglang_{your-name}
```


### 更新 Python 并使用 uv 配置虚拟环境

```bash 
apt update
apt install -y python3.10 python3.10-venv

# 创建虚拟环境
python3 -m venv ~/.python/veRL-multiturn-rollout

# 激活虚拟环境
source ~/.python/veRL-multiturn-rollout/bin/activate

# 安装 uv
python3 -m pip install uv
```


### 安装 veRL upstream

```bash 
cd ~
git clone https://github.com/volcengine/verl.git
cd verl

# 安装 verl
python3 -m uv pip install .
python3 -m uv pip install -r ./requirements_sglang.txt

# 手动安装 flash-attn
python3 -m uv pip install wheel
python3 -m uv pip install packaging
python3 -m uv pip install flash-attn --no-build-isolation --no-deps
```

### 设置 WANDB_API_KEY

如果不理解如何获取 API Key，请参考[此处](https://community.wandb.ai/t/where-can-i-find-the-api-token-for-my-project/7914)。

```bash 
export WANDB_API_KEY={YOUR_WANDB_API_KEY}

# 定义时间戳函数
function now() {
    date '+%Y-%m-%d-%H-%M'
}
```


### **预处理数据集**

注意，以下数据处理以及训练命令需要在 veRL-multiturn-rollout 执行环境中进行。

```bash 
python3 examples/data_preprocess/geo3k.py
```


### 在 8 X H20 上进行测试

```bash 
# 确保 now() 函数已经定义
# 创建日志目录
mkdir -p logs

# 设置 GPU 并运行，使用合适的日志路径
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

nohup bash examples/sglang_multiturn/geo3k/run_qwen2.5-3b_instruct_multiturn.sh trainer.experiment_name=qwen2.5-3b-it_rm-geo3k-sgl-multiturn-$(now) > logs/geo3k$(now).log 2>&1 &
```
## 注意事项
