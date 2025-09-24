# FSDP Setup Guide

这个文档记录如何在Slime上测试FSDP，包括H卡和B卡，以及Colocate和分离的配置。以下操作在H卡上完成


## 基础环境搭建

### 拉取并启动 Docker 容器

请执行以下命令，拉取最新镜像并启动一个交互式容器：

```shell
# 拉取最新镜像
# 最新的镜像是B卡 H卡通用的
docker pull slimerl/slime:latest

# 启动容器
docker run -d --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --name slime_wren_fsdp \
  -it zhuzilin/slime:latest /bin/bash
```

### 安装 slime

进入 Docker 容器后，请按照以下步骤克隆 slime 仓库并进行安装：

```bash
# 路径可根据实际情况调整
cd /root/
git clone https://github.com/Williamren97/slime.git #FSDP开发中的分支
cd slime
pip install -e .
git checkout optimize/fsdp-memory-overhead 
```

## 模型与数据集下载

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

## 加载目标模型的配置文件



首先，加载目标模型的配置文件。`slime/scripts/models` 目录下包含了支持模型的配置文件。需要 `source` 对应模型的脚本，将配置参数加载到当前环境中。此处我们以 Qwen3-0.6B 模型为例子，对于 Qwen3-4B，Qwen3-30B-A3B，是类似的。

```bash
cd /root/slime
source scripts/models/qwen3-0.6B.sh 
```

## 训练脚本与参数概览

完成上述准备工作后，即可运行训练脚本。

```bash
cd /root/slime
bash slime/tests/test_fsdp_colocated_2GPU.sh # 2GPU协同训练测试
bash slime/tests/test_fsdp.sh                # 基础FSDP测试
```



## 特性介绍

### Colocated Actor and Rollout

在默认的配置下，训练（Actor）和推理（Rollout）的资源是分开指定的，通过 ray 给训练部分分配 `actor_num_nodes * actor_num_gpus_per_node` 张 GPU，给推理分配 `rollout_num_gpus` 张 GPU，也即训推分离。

**标准（分离）配置**：
```bash
ray job submit ... \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   ...
```
上述配置中，Actor 使用 4 张卡，Rollout 也使用 4 张卡，两者并行运行。


> 当进行训推分离时，你会发现训练和推理的 GPU 总是相互等待着，为了避免这种资源空闲，我们可以开启异步训练。开启的方式即为将启动脚本中的 train.py 改变为 train_async.py。这样 slime 就会在进行当前 rollout 的训练时进行下一个 rollout 的数据生成了。

> ⚠️ 在异步训练时，sglang 的性能检测日志与训练日志可能会混到一起，不易区分，可以通过 --sglang-log-level 来减少 sglang 的日志。



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

# FSDP 测试脚本分析

## 🎯 **FSDP 激活机制**

```bash
# 关键：指定后端为FSDP
"SLIME_BACKEND": "fsdp"
```

这个环境变量告诉slime使用FSDP后端而不是默认的Megatron后端。

## 🔧 **FSDP 核心配置**

### **GPU 分片设置**
```bash
export CUDA_VISIBLE_DEVICES=1,2  # 使用GPU 1,2
--actor-num-gpus-per-node 2      # 2个GPU进行模型分片
```

### **FSDP 模式选择**
```bash
--fsdp-full-params  # 启用FULL_STATE_DICT模式
# 注释掉则使用默认的SHARDED_STATE_DICT模式
```

##  **为什么这样能用到FSDP**

### **1. 后端路由**
当设置 `SLIME_BACKEND=fsdp` 时，slime会：
- 加载 `slime/backends/fsdp_utils/` 下的FSDP实现
- 使用 `FSDPTrainRayActor` 而不是 `MegatronTrainRayActor`
- 调用 `create_fsdp_v2_model()` 创建FSDP模型

### **2. 模型分片**
```python
# 在FSDP后端中会执行
model = fully_shard(base_model)  # FSDP v2 API
# 70B模型在2个GPU上分片：每GPU ~35B参数
```

### **3. 权重更新测试**
- 训练时使用DTensor（分片存储）
- 权重更新时调用 `dtensor.full_tensor()` 
- 通过IPC发送给SGLang推理引擎

### **4. 协同部署验证**
```bash
--colocate  # 训练和推理进程共享GPU资源
```
验证FSDP训练进程与SGLang推理进程的GPU内存协调。

## 📝 **测试脚本路径**

主要的FSDP测试文件位于：
- `slime/tests/test_fsdp.sh` - 基础FSDP测试
- `slime/tests/test_fsdp_colocated_2GPU.sh` - 2GPU协同训练测试
- `tests/test_fsdp_import.py` - FSDP导入测试

##  **测试目标**

**本质上**：这个测试验证了完整的"FSDP训练 → 权重提取 → SGLang更新"数据流，正是我们分析的内存瓶颈所在。

通过2GPU的最小配置，可以有效验证：
- FSDP v2的DTensor机制
- 权重同步的IPC通信
- 协同训练的资源管理
- 内存优化的实际效果


### B-series GPU Setup
## 启动docker
```shell
# 拉取最新镜像,最新的镜像是B卡 H卡通用的
docker pull slimerl/slime:latest


# 启动容器
# 这里 GPU相关参数完全相同，主要差异是镜像版本和挂载目录（这些是环境配置，不是硬件差异）
docker run -d
    --gpus all 
    --ipc=host 
    --shm-size=32g \
    --ulimit memlock=-1 
    --ulimit stack=67108864 \
    -v /home/yineng/shared_model:/root/.cache \
    -v /home/yineng/william:/workspace \
    --name slime_william \
    -it slimerl/slime:latest /bin/bash
```

## 剩余步骤和H卡操作步骤完全相同


> 如果遇到 nccl 的 error，在ray启动的时候可以指定一个端口
```shell
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --port 9987
```
