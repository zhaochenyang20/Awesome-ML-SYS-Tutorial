# SLIME 框架概述

## 目录

- [1. 简介](#1-简介)
- [2. 核心架构](#2-核心架构)
- [3. 关键特性](#3-关键特性)
- [4. 训练模式对比](#4-训练模式对比)
- [5. 使用场景](#5-使用场景)
- [6. 代码结构](#6-代码结构)

## 1. 简介

**SLIME** (Scalable LLM Infrastructure for Massive Efficiency) 是专为强化学习大规模训练设计的 LLM 后训练框架。

### 1.1 核心能力

1. **高性能训练**: 通过 Megatron-LM 提供分布式训练能力，支持 Dense 和 MoE 模型
2. **灵活数据生成**: 通过 SGLang 引擎和自定义接口，实现任意复杂的数据生成流程
3. **异步训练**: 支持训练和推理的异步执行，显著提升 GPU 利用率

### 1.2 项目链接

- **项目地址**: [https://github.com/THUDM/slime/tree/main/slime](https://github.com/THUDM/slime)
- **文档**: [slime/docs/](https://github.com/THUDM/slime/tree/main/docs)
- **Docker 镜像**: `zhuzilin/slime:latest`

## 2. 核心架构

SLIME 采用分离式架构，将 RLHF 训练流程分解为三个独立协作的模块：

- **Training (Megatron)**: 负责主训练流程，支持多种并行策略
  - *代码位置*: [`slime/backends/megatron_utils/`](https://github.com/THUDM/slime/tree/main/slime/slime/backends/megatron_utils/)
  
- **Rollout (SGLang)**: 生成新数据（含 reward/verifier），基于 SGLang 优化推理
  - *代码位置*: [`slime/ray/rollout.py`](https://github.com/THUDM/slime/tree/main/slime/ray/rollout.py)
  
- **Data Buffer**: 桥梁模块，管理数据流和自定义生成逻辑
  - *代码位置*: [`slime/ray/buffer.py`](https://github.com/THUDM/slime/tree/main/slime/ray/buffer.py)

## 3. 关键特性

### 3.1 分布式资源管理

基于 Ray 框架进行资源调度：
- **Placement Groups**: 资源隔离和分配
- **多种并行策略**: 数据/张量/流水线/专家并行
- **动态扩缩容**: 训练和推理资源独立调整

*核心实现*: [`slime/ray/placement_group.py`](https://github.com/THUDM/slime/tree/main/slime/ray/placement_group.py)

### 3.2 异步训练优化

SLIME 提供两种训练模式：

- **同步训练** ([`train.py`](https://github.com/THUDM/slime/tree/main/slime/train.py)): 传统的顺序执行模式
- **异步训练** ([`train_async.py`](https://github.com/THUDM/slime/tree/main/slime/train_async.py))，在dis-agg情况下，使用```rollout_manager.async_generate```和 ```actor_model.async_train```来分布进行训练，且rollout永远早于train一个step？ （这样是否可以理解为one-step-off-policy）

### 3.3 灵活的数据生成

支持用户自定义复杂的数据生成逻辑：
- 多轮对话 ([例子](https://github.com/THUDM/slime/tree/main/slime/examples/search-r1/))
- 工具调用
- 奖励模型集成
- 自定义验证器

*扩展接口*: [`slime_plugins/rollout_buffer/`](https://github.com/THUDM/slime/tree/main/slime_plugins/rollout_buffer/)


## 4. 使用场景

### 4.1 支持的模型类型

- **Dense 模型**: GLM-4-9B, Qwen3-4B 等
  - *配置示例*: [`slime/scripts/run-qwen3-4B.sh`](https://github.com/THUDM/slime/tree/main/slime/scripts/run-qwen3-4B.sh)
  
- **MoE 模型**: Qwen3-30B-A3B, DeepSeek-R1 等  
  - *配置示例*: [`slime/scripts/run-deepseek-r1.sh`](https://github.com/THUDM/slime/tree/main/slime/scripts/run-deepseek-r1.sh)

### 4.2 训练任务类型

- **强化学习**: PPO, GRPO, DPO 等算法
- **监督微调**: SFT 训练支持

### 4.3 部署模式

- **单机多卡**: 适合中小规模模型
- **多机多卡**: 支持大规模分布式训练 (如 128×H100)
- **混合部署**: 训练和推理资源分离部署

## 5. 代码结构

```
slime/
├── slime/                          # 核心框架代码
│   ├── ray/                        # Ray 分布式组件
│   │   ├── actor_group.py         # 训练 Actor 管理
│   │   ├── rollout.py             # 推理 Actor 管理
│   │   ├── buffer.py              # 数据缓冲区
│   │   └── placement_group.py     # 资源分配
│   ├── backends/                   # 后端引擎集成
│   │   ├── megatron_utils/        # Megatron 训练后端
│   │   └── sglang_utils/          # SGLang 推理后端
│   └── utils/                      # 工具函数
├── slime_plugins/                  # 插件和扩展
│   ├── rollout_buffer/            # 自定义生成插件
│   └── models/                    # 模型适配
├── scripts/                        # 启动脚本
│   └── models/                    # 各模型配置
├── examples/                       # 使用示例
├── docs/                          # 详细文档
├── train.py                       # 同步训练入口
└── train_async.py                 # 异步训练入口
```
---

*参考架构设计: [SGLang Code Walk-through](https://github.com/maocheng23/Awesome-ML-SYS-Tutorial/blob/main/sglang/code-walk-through/readme-CN.md)*

### 5.1. 各目录用途与串联关系

- `scripts/`：启动脚本与模型配置
  - 用于启动 Ray 集群与提交训练作业；示例脚本会选择 `train.py` 或 `train_async.py`
  - 例如：`slime/scripts/run-qwen3-4B.sh`、`slime/scripts/run-deepseek-r1.sh`

- `train.py` / `train_async.py`：训练入口
  - 创建 `PlacementGroup` 分配 GPU → 创建 `actor_group`（训练）与 `rollout_manager`（推理）→ 进入训练循环
  - 同步模式逐步执行；异步模式通过 `rollout_manager.async_generate()` 与 `ray.get()` 交错以并行化

- `slime/ray/`：分布式编排与资源管理
  - `placement_group.py`：基于 Ray Placement Group 的 GPU 资源分配与打包
  - `actor_group.py`：训练 Actor 组管理，暴露 `async_init/async_train/async_update_weights` 等接口
  - `rollout.py`：Rollout Actor（SGLang 引擎容器）、推理服务路由、权重接收
  - `buffer.py`：数据缓冲、样本批次组织、与 Rollout/Training 的中间桥梁

- `slime/backends/`：后端引擎适配
  - `megatron_utils/`：训练后端（优化器、权重更新、与分布式通信集成）
  - `sglang_utils/`：推理后端（包装 SGLang、批处理生成、引擎生命周期管理）

- `slime_plugins/`：可插拔扩展
  - `rollout_buffer/`：通过 HTTP/OpenAI 接口等外部联动的自定义轨迹生成器体系
  - `models/`：不同模型族的小适配层

- `examples/`：最小可运行示例
  - 例如 `examples/search-r1/` 展示多轮对话 + 工具调用的生成与训练串联方式

- `docs/`：说明文档与用法指南
  - 包含模型使用、SFT、AMD/NPU 等平台适配与调优手册

### 5.2 串联关系（从脚本到训练与生成）

1) 脚本层（`scripts/`）
- 启动 Ray → 提交job → 选择 `train.py` 或 `train_async.py` 并传入参数

2) 入口层（`train*.py`）
- `create_placement_groups(args)` 分配/映射 GPU
- `create_actor_group(args, pgs["actor"])` 构建训练 Actor 组
- `create_rollout_manager(args, pgs["rollout"])` 构建推理与数据生成管理器

3) 执行层（`ray/` + `backends/`）
- 训练：`actor_group.async_train(...)` → Megatron 优化/梯度计算
- 生成：`rollout_manager.async_generate(...)` → SGLang 批量推理
- 同步：`actor_group.async_update_weights()` → 将训练权重推送到推理引擎

4) 数据流（`buffer.py` + 插件）
- `Buffer` 负责抽样/拼批/调用自定义生成（`slime_plugins/rollout_buffer/`）→ 返回训练可用样本

通过以上链路，SLIME 将“脚本 → 入口 → 分布式执行 → 数据/权重流”自然地串起来，实现高效可扩展的 RL 后训练。