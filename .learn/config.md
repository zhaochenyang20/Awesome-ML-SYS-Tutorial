# /learn Agent 配置

## 深度偏好

### 距离梯度

```
自己开发的系统 (SGLang/verl/slime/SGLang-Omni) → 修改扩展（源码级）
        ↓
依赖的基础设施 (FSDP/NCCL/CUDA Graph/Megatron) → 理解复现（原理级）
        ↓
相关的算法理论 (PPO/RL Theory/量化理论) → 建立直觉（直觉级）
        ↓
读到的论文 (Kimi K1.5/SWE-Bench) → 摘要提取（信息级）
```

### 系统分类

#### 修改扩展级（modify-extend）
- SGLang（推理引擎）
- verl（RLHF 框架）
- slime（RL 框架）
- SGLang-Omni（多模态推理）

#### 理解复现级（understand-reproduce）
- FSDP / FSDP2（分布式训练）
- NCCL（通信库）
- CUDA Graph（GPU 执行优化）
- Megatron（大模型训练框架）
- PyTorch Distributed（分布式通信）
- Triton（GPU kernel）

#### 建立直觉级（intuition）
- PPO / GRPO / DAPO（RL 算法）
- RL Theory（强化学习理论）
- 量化理论（FP8/INT4/AWQ）
- Attention 机制（FlashAttention 等）
- Diffusion Model（扩散模型）

#### 摘要提取级（summary）
- 论文阅读笔记
- 技术报告解读
- Benchmark 分析

## 引用风格

### 内部引用
- 使用 repo 内相对路径：`[文章标题](./relative/path/readme.md)`
- 引用 published 状态的文章

### 外部代码引用
- 必须包含 commit hash：`https://github.com/org/repo/blob/<commit_hash>/path/to/file.py#L123-L456`
- 禁止引用 main/master 分支的行号

### 外部文章引用
- 知乎文章：直接链接
- 官方文档：链接到 stable/latest 版本
- 论文：arXiv 链接

## 文件路径规则

### 新文章路径
- 按 topic 分类放置在对应目录下
- 路径结构：`[category]/[topic]/readme.md`
- 学习计划：`[category]/[topic]/learn-plan.md`
- 英文版本：`[category]/[topic]/readme-en.md` 或 `readme-EN.md`

### 目录分类
- `rlhf/`：RLHF 系统相关（slime, verl, OpenRLHF, 系统设计）
- `sglang/`：SGLang 推理引擎相关
- `torch/`：PyTorch 基础设施（NCCL, CUDA Graph, Distributed）
- `transformers/`：模型架构相关
- `engineer/`：开发工具与实践

## 语言设置

- 写作语言：中文
- 代码注释语言：保持源码原始语言
- 翻译：中文完成后，经确认可翻译为英文
