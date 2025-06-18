# Multi-Turn Rollout with Fast Tokenization (GSM8K)

This example demonstrates how to perform **multi-turn rollout** using SGLang with fast tokenization.

## Usage

### Prerequisites
Install veRL from fast tokenization branch:

```bash
cd ~
git clone https://github.com/jybsuper/verl.git
cd verl
git checkout tokenization

conda create -n verl python==3.10
conda activate verl
# The offical dependency installation script does not use uv.
# To speed up the installation, feel free to add uv prefix to the pip command in it.
bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
```

### Step 1: Download GSM8K Dataset

```bash
cd examples/data_preprocess
python3 gsm8k_multiturn_w_tool.py
```

This will download and preprocess the GSM8K dataset into `~/data/gsm8k/`.

### Step 2: Validate Multi-Turn Rollout for Qwen2.5-3B

If you have 8 GPUs
Use the standard 8-GPU script:

```bash
bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh
```

If you have only 4 GPUs
Use the fallback 4-GPU script:

```bash
bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn_4xgpu.sh 
```

### Step 3: Validate Multi-Turn Rollout for Qwen3-4B

This validation uses 8 GPUs
```bash
bash examples/sglang_multiturn/run_qwen3-4b_gsm8k_multiturn.sh
```
