# 新特性：多轮RL训练中搜索工具调用

- 为 veRL-sglang MultiTurnRL 增加了搜索工具调用能力，使模型在 Actor rollout 阶段能够调用搜索接口，并将检索结果直接融入训练流程。
- PR 链接：[volcengine/verl#1682](https://github.com/volcengine/verl/pull/1682)
- [训练曲线wandb](https://wandb.ai/lingchang-ustc/search_async_rl/workspace?nw=nwuserlingchang "训练曲线wandb")

# 如何使用

### 环境配置

**创建新的 Docker 容器**

```docker 
docker run \
    -it \
    --shm-size 32g \
    --gpus all \
    -v /models/shared/.cache:/root/.cache \
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


**更新 Python 并使用虚拟环境**

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


**安装 veRL upstream**

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


**搭建自己的本地检索（如果用自己的服务可跳过**

- 这里选择searchR1示例中给出的local dense retriever，详细说明文档见[searchR1](https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/refs/heads/main/docs/retriever.md "searchR1")
  - 需要GPU（运行后每张卡占用5\~7G显存），精度较高，速度快
  - 无GPU版本参考searchR1中的[详细文档](https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/retriever.md "详细文档")（简单测试代码可以使用，但检索精度差，也会导致训练效果差）
- **注意**：**建议使用conda安装检索服务的环境**，venv中faiss-gpu安装未成功。

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


- 下载indexing和corpus

```bash 
conda activate retriever

save_path=/the/path/to/save
python examples/sglang_multiturn/searchR1_like/local_dense_retriever/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```


- 启动本地 flat e5 检索服务器
  - 首次启动会下载模型，加载索引等。网速正常两三分钟。
  - 启动后每张gpu占用显存大约5\~7GB(可以在同一节点上训练）

```bash 
conda activate retriever

index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python examples/sglang_multiturn/searchR1_like/local_dense_retriever/retrieval_server.py --index_path $index_file --corpus_path $corpus_file --topk 3 --retriever_name $retriever_name --retriever_model $retriever_path --faiss_gpu

```


### 在 8 x H20 上展开测试

**设置**`WANDB_API_KEY`

如果不理解如何获取 API Key，请参考[此处](https://community.wandb.ai/t/where-can-i-find-the-api-token-for-my-project/7914 "此处")。

```bash 
export WANDB_API_KEY={YOUR_WANDB_API_KEY}

# 定义时间戳函数
function now() {
    date '+%Y-%m-%d-%H-%M'
}
```


**预处理数据集**（以下数据处理以及训练命令在venv的veRL-multiturn-rollout环境中进行

```bash 
# 若要定义自己的prompt，在examples/data_preprocess/prompt.yaml中修改
# 默认存储在~/data/searchR1_processed_direct下
python3 examples/data_preprocess/preprocess_searchR1_dataset.py --config examples/data_preprocess/prompt.yaml


```


**进行测试**

```bash 
# 确保 now() 函数已经定义
# 创建日志目录
mkdir -p logs

# 设置 GPU 并运行，使用合适的日志路径
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

nohup bash examples/sglang_multiturn/searchR1_like/run_qwen2.5-3b_instruct_search_multiturn.sh trainer.experiment_name=qwen2.5-3b-it_rm-searchR1-like-sgl-multiturn-$(now) > logs/searchR1-like$(now).log 2>&1 &s
```


### 自定义配置

**基础配置**

启用多轮推理需在配置中设置以下字段：

```yaml 
actor_rollout_ref:
  rollout:
    name: "sglang_async"
    multi_turn:
      enable: True
```


需要在examples/sglang\_multiturn/config/tool\_config/search\_tool\_config.yaml中指定retrieval\_service\_url，并支持设置并发

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


## 注：

仅进行debug时，可以注释掉`ray_trainer.py`中`fit`函数开始的val过程（val数据集太大，一次验证需要6000s左右），如下

verl/trainer/ppo/ray\_trainer.py

```python 
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
