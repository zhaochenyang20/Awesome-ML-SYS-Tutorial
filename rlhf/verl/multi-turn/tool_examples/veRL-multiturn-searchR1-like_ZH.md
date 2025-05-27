# Search-R1 & veRL-SGLang：Train LLMs with Multi-Turn RL to Reason and Call a Search Engine

大家好，SGLang 社区联合 Search R1 团队基于先前开源的 [multi-turn RL](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/release_log/verl-multiturn-rollout-Release_ZH.md) 快速复现了 Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning，欢迎大家上手体验，共同开发。具体来说，我们实现了如下的功能：

- 先前 SGLang 社区已经实现了工具调用，支持在 Actor rollout 期间，让模型能够调用特定工具，并将返回结果无缝地融合到训练流程中。
- 我们进一步为 Multi-Turn RL 新增了搜索工具调用功能，使模型能够在 Actor rollout 阶段发起检索请求，直接利用检索结果进行训练。**我们支持使用 local dense retriver 作为检索工具，也支持接入用户本地的检索引擎。**
- 我们为社区提供了一种 Search R1 工作的全新复现方案， 并已经集成到了 verl upstream，会持续维护并且更新我们的框架。此外，verl 在效率优化上的最新功能（诸如 FSDP2 和 Megatron）也能直接使用。这是我们相比其他不在主分支维护的工作的巨大优势。

[PR: volcengine/verl#1682](https://github.com/volcengine/verl/pull/1682)
[训练曲线 wandb](https://wandb.ai/lingchang-ustc/search_async_rl/workspace?nw=nwuserlingchang)

感谢 SGLang 团队以及 searchR1 作者的高效支持！

Project Member: Bowen Jin, Ling Chang, Nan Jiang, Chenyang Zhao, Long Xiang

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

### 搭建本地检索引擎

如果您使用本地的检索服务，可以跳过此步骤。我们选择 search-R1 示例中给出的local dense retriever，详细说明文档见[searchR1](https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/refs/heads/main/docs/retriever.md "searchR1")。其中：

- 需要 GPU 的版本，精度较高，速度快；运行后每张卡占用 5~7G 显存。
- 无 GPU 的版本参考 search-R1 中的[ retriever 文档](https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/retriever.md)，可以用于简单测试，但检索精度差，会导致训练效果差

**注意**：为了启动训练进程和本地检索服务，我们启动了两个 Python 执行环境。其中训练使用 uv 搭建上述 veRL-multiturn-rollout 环境，而 retriver 使用 conda 来安装 `faiss-cpu`。

```bash 
#  下载 Miniconda 安装脚本
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

#  以非交互模式安装到 $HOME/miniconda3
bash ~/miniconda.sh -b -p $HOME/miniconda3

#  激活 conda（只在当前 shell 生效）
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

#  （可选）将 conda 加入到默认 shell 启动脚本，以后自动可用
conda init

#  重新加载 shell 配置
source ~/.bashrc

#  创建并激活 retriever 环境，指定 Python 3.10
conda create -n retriever python=3.10 -y
conda activate retriever

#  安装 PyTorch（含 GPU 支持）及相关库
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

#  安装其他 Python 包
pip install transformers datasets pyserini huggingface_hub

#  安装 GPU 版 faiss
conda install faiss-gpu=1.8.0 -c pytorch -c nvidia -y

#  安装 API 服务框架
pip install uvicorn fastapi
```

### 下载 indexing 和 corpus

本地检索文件体积较大，请准备充分的磁盘；下载文件大约 60~70GB，解压后在 132G 左右。

```bash 
conda activate retriever

save_path=/the/path/to/save
python examples/sglang_multiturn/search_r1_like/local_dense_retriever/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

### 启动本地 flat e5 检索服务器

1. 首次启动会下载模型，加载索引等。
2. 除去下载过程，正常启动时间 1~2 分钟。
3. 启动后每张 GPU 占用显存大约 5~7 GB，节点上余下空间可供 multi-turn RL训练。

```bash 
conda activate retriever

index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python examples/sglang_multiturn/search_r1_like/local_dense_retriever/retrieval_server.py --index_path $index_file --corpus_path $corpus_file --topk 3 --retriever_name $retriever_name --retriever_model $retriever_path --faiss_gpu
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
# 若要定义自己的 prompt template，请在 examples/data_preprocess/prompt.yaml 中进行修改
# 预处理好的数据默认存储在 ~/data/searchR1_processed_direct 下
python3 examples/data_preprocess/preprocess_searchR1_dataset.py --config examples/data_preprocess/prompt.yaml
```


### 在 8 X H20 上进行测试

```bash 
# 确保 now() 函数已经定义
# 创建日志目录
mkdir -p logs

# 设置 GPU 并运行，使用合适的日志路径
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

nohup bash examples/sglang_multiturn/search_r1_like/run_qwen2.5-3b_instruct_search_multiturn.sh trainer.experiment_name=qwen2.5-3b-it_rm-searchR1-like-sgl-multiturn-$(now) > logs/searchR1-like$(now).log 2>&1 &s
```
## 注意事项

1. 总训练时长 27 小时左右；同时，validation 数据集非常大（51k），进行一次 validation 需 6000s 左右。
2. 需要 debug 以快速开发时，可以删去 `ray_trainer.py` 中 `fit` 函数初始的 validation 过程，具体修改如下：

```python

# verl/trainer/ppo/ray_trainer.py

# perform validation before training
# currently, we only support validation using the reward_function.
# if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
#     val_metrics = self._validate()
#     assert val_metrics, f"{val_metrics=}"
#     pprint(f"Initial validation metrics: {val_metrics}")
#     logger.log(data=val_metrics, step=self.global_steps)
#     if self.config.trainer.get("val_only", False):
#         return
```

3. 训练效果相比原论文在合理范围内存在波动，我们已于 search-R1 作者分析了相关原因：
   
- special token（如 `<tool_call>、<tool_response>`）等未完全对齐，有待后续开发
- 我们修改了 search-R1 初始实现中的 EM reward 以及对 response 中 `\<answer>, \</answer>` 数量过多的惩罚
- 少部分超参难以完全对齐，计算资源有限，有待社区贡献
  
4. 请控制训练时的 `micro_batch_size_per_gpu`，过大容易 OOM

## 自定义搜索配置

启用多轮推理需在配置中设置以下字段：

```yaml 
actor_rollout_ref:
  rollout:
    name: "sglang_async"
    multi_turn:
      enable: True
```

需要在 `examples/sglang_multiturn/config/tool_config/search_tool_config.yaml` 中指定 `retrieval_service_url`，并设置是否支持并发：

```yaml 
tools:
  - class_name: "verl.tools.search_tool.SearchTool"
    config: {
      "retrieval_service_url": "http://127.0.0.1:8000/retrieve",
      "num_workers": 120,
      "rate_limit": 150,
      "default_timeout": 30
    }
```

## References

感谢 [search-R1](https://github.com/xxx/search-r1) 项目的帮助和启发. 若您在我们的研究中有所借鉴，感谢同时引用原始项目：

```bibtex
@article{jin2025search,
  title={Search-r1: Training llms to reason and leverage search engines with reinforcement learning},
  author={Jin, Bowen and Zeng, Hansi and Yue, Zhenrui and Yoon, Jinsung and Arik, Sercan and Wang, Dong and Zamani, Hamed and Han, Jiawei},
  journal={arXiv preprint arXiv:2503.09516},
  year={2025}
}
```
