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
# 官方脚本没有使用 uv。
# 为了加快安装速度，可以自行在其中的 pip 命令前加上 uv 前缀。
bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
```

### 步骤 1：下载 GSM8K 数据集

```bash
cd examples/data_preprocess
python3 gsm8k_multiturn_w_tool.py
```

这将下载并预处理 GSM8K 数据集到 `~/data/gsm8k/`。

### 步骤 2：验证 Qwen2.5-3B 的多轮推理

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

### 步骤 3：验证 Qwen3-4B 的多轮推理

该脚本使用 8 张 GPU：

```bash
bash examples/sglang_multiturn/run_qwen3-4b_gsm8k_multiturn.sh
```
