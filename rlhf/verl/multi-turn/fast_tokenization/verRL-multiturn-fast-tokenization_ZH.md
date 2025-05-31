# 多轮推理与快速分词（GSM8K）

本示例演示如何在基于SGLang的多轮推理中使用快速分词。

## 如何使用

### 前置条件
从 fast tokenization 分支安装 veRL

```bash
cd ~
git clone https://github.com/jybsuper/verl.git
cd verl
git checkout tokenization

conda create -n verl python==3.10
conda activate verl
bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
```

### 步骤 1：下载 GSM8K 数据集

```bash
cd examples/data_preprocess
python3 gsm8k_multiturn_w_tool.py
```

这将下载并预处理 GSM8K 数据集到 ~/data/gsm8k/。

### 步骤 2：使用默认快速分词运行多轮推理

如果你有 8 块 GPU
使用标准的 8-GPU 脚本：

```bash
bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh
```

如果你只有 4 块 GPU
使用备用的 4-GPU 脚本：

```bash
bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn_4xgpu.sh 
```

### 步骤 3：使用传统全分词和 Sanity Check 模式运行多轮推理

本分支还实现了另外两种分词模式：
1. 传统全分词模式，每一轮都对整个对话进行分词
2. 用于调试的 sanity check 模式，同时运行快速和全分词，并断言它们结果一致。

```bash
# 运行全分词模式
bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh actor_rollout_ref.rollout.multi_turn.tokenization_mode=full
# 运行 sanity check 模式
bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh actor_rollout_ref.rollout.multi_turn.tokenization_mode=sanity_check
```
