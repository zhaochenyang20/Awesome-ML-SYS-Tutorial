# SGLang as rollout engine of GRPO trainer 
## Reproduce the Experiment
```
git clone git@github.com:huggingface/trl.git
cd trl
python3 -m uv pip install -e ".[sglang]"

export WANDB_API_KEY=<YOUR_WANDB_API_KEY>

# start sglang-server
python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct

# modify cuda config in run.sh and grpo_config.yaml
# or run "export CUDA_VISIBLE_DEVICES"
# run script
sh ./trl/scripts/grpo_test/run.sh

```