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
bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
```

### Step 1: Download GSM8K Dataset

```bash
cd examples/data_preprocess
python3 gsm8k_multiturn_w_tool.py
```

This will download and preprocess the GSM8K dataset into ~/data/gsm8k/.

### Step 2: Run Multi-Turn Rollout with default Fast Tokenization

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

### Step 3: Run Multi-Turn Rollout with Classic Full Tokenization and Sanity Check Mode

This branch also implements two other tokenization modes:
1. a classic full tokenization mode that tokenize the whole conversation at every turn
2. a sanity check mode for debugging purpose that runs both fast and full tokenization and assert that they produce the same results.

```bash
# Run full tokenization mode
bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh actor_rollout_ref.rollout.multi_turn.tokenization_mode=full
# Run sanity check mode
bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh actor_rollout_ref.rollout.multi_turn.tokenization_mode=sanity_check
```
