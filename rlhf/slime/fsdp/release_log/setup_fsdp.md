# FSDP Setup Guide

这个文档记录如何在 slime 上测试 FSDP，包括 H 卡和 B 卡，以及 Colocate 和 Disaggregated 两种 placement 方式。以下操作在 H 卡上完成：

## Quick Start

### 拉取并启动 Docker 容器

```shell
# 拉取最新镜像
# 最新的镜像是 B 卡 H 卡通用的
docker pull slimerl/slime:latest

# 启动容器
docker run -d --gpus all --ipc=host --shm-size=16g \
  --name slime_wren_fsdp \
  -it slimerl/slime:latest /bin/bash
```

### 安装 slime

进入 Docker 容器后，请按照以下步骤克隆 slime 仓库并进行安装：

```bash
# 路径可根据实际情况调整
cd /root/
git clone https://github.com/THUDM/slime.git
cd slime
pip install -e .
```

### 模型与数据集下载

可以从 Hugging Face、ModelScope 等平台下载所需的模型和数据集。以下是使用 `huggingface_hub` 下载示例资源的命令：

```bash

pip install -U huggingface_hub

# 下载模型权重 (Qwen3-0.6B)
hf download Qwen/Qwen3-0.6B --local-dir /root/Qwen3-0.6B

# 下载训练数据集 (dapo-math-17k)
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k

# 下载评估数据集 (aime-2024)
hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /root/aime-2024
```

### 加载目标模型的配置文件

首先，加载目标模型的配置文件。`slime/scripts/models` 目录下包含了支持模型的配置文件。需要 `source` 对应模型的脚本，将配置参数加载到当前环境中。此处我们以 Qwen3-0.6B 模型为例子，对于 Qwen3-4B、Qwen3-30B-A3B 是类似的。

```bash
cd /root/slime
source scripts/models/qwen3-0.6B.sh 
```

### 训练脚本与参数概览

完成上述准备工作后，即可运行训练脚本。

```bash
cd /root/slime
bash tests/test_qwen3-0.6B_fsdp_colocated_2xGPU.sh # 2GPU 协同训练测试
```

## 特性介绍

### Colocated Actor and Rollout

在默认的配置下，训练（Actor）和推理（Rollout）的资源是分开指定的，通过 `ray` 给训练部分分配 `actor_num_nodes * actor_num_gpus_per_node` 张 GPU，给推理分配 `actor_num_nodes * rollout_num_gpus` 张 GPU，也即训推分离。

**标准（分离）配置**：

```bash
ray job submit ... \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   ...
```

上述配置中，`Actor` 使用 4 张卡，`Rollout` 也使用 4 张卡，两者并行运行。


> 当进行训推分离时，训练和推理的 GPU 总是相互等待着，为了避免这种资源空闲，我们可以开启异步训练。开启的方式即为将启动脚本中的 `train.py` 改变为 `train_async.py`。这样 slime 就会在进行当前 rollout 的训练时进行下一个 rollout 的数据生成了。

> ⚠️ 在异步训练时，sglang 的性能检测日志与训练日志可能会混到一起，不易区分，可以通过 `--sglang-log-level` 来减少 sglang 的日志。

**训推一体化（Colocated）配置**：

要将训练和推理部署在同一组 GPU 上，请添加 `--colocate` 参数，开启后会忽略 `--rollout-num-gpus` 让训练和推理的卡数相等。

```bash
ray job submit ... \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ...
```

此时，训练和推理将共享全部 8 张 GPU。

### FSDP 激活机制

```bash
# 关键：指定后端为 FSDP
"SLIME_BACKEND": "fsdp"
```

GPU 分片设置：

```bash
export CUDA_VISIBLE_DEVICES=1,2  # 使用 GPU 1,2
--actor-num-gpus-per-node 2      # 2 个 GPU 进行模型分片
```

FSDP 模式选择：

```bash
--fsdp-full-params  # 启用 FULL_STATE_DICT 模式
# 注释掉则使用默认的 SHARDED_STATE_DICT 模式
```

## Blackwell GPU 设置

### 启动 Docker

```shell
# 拉取最新镜像，最新的镜像是 B 卡 H 卡通用的
docker pull slimerl/slime:latest

# 启动容器
# 这里 GPU 相关参数完全相同，主要差异是镜像版本和挂载目录（这些是环境配置，不是硬件差异）
docker run \
      -itd \
      --shm-size 32g \
      --gpus all \
      --ipc=host \
      --network=host \
      --privileged \
      -v {your_cache_path}:/root/.cache \
      --name slime_fsdp_{your_name} \
      slimerl/slime:latest \
      /bin/bash
```

剩余步骤和 H 卡操作步骤完全相同。

> 如果遇到 `nccl` 的 error，在 `ray` 启动的时候可以指定一个端口：

```shell
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --port 9987
```
