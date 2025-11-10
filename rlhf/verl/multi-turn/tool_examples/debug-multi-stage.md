# A Quick Reproduction of Error in Multi-Stage Wake Up [Fixed Already]

Nan has found that utilzing multi-stage wake up will encounter `illegal memory access` error. We are still debugging to find out. Here is how I am working to reproduce the error.

```bash
docker run \
    -it \
    --shm-size 32g \
    --gpus all \
    -v {Huggingface-Cache-Path}:/root/.cache \
    --ipc=host \
    --network=host \
    --privileged \
    --name sglang_{your-name} \
    lmsysorg/sglang:dev \
    /bin/zsh

apt update
apt install -y python3.10 python3.10-venv

python3 -m venv ~/.python/veRL-multiturn-rollout
source ~/.python/veRL-multiturn-rollout/bin/activate

python3 -m pip install uv

cd ~
git clone -b multi-stage-wake-up https://github.com/volcengine/verl.git
cd verl

python3 -m uv pip install -e ".[sglang]"
python3 -m uv pip install wheel
python3 -m uv pip install packaging
python3 -m uv pip install flash-attn==2.8.0.post2 --no-build-isolation --no-deps

# reinstall latest sglang for verl multi-stage wake up
python3 -m uv pip install "sglang[all]==0.4.8.post1"


function now() {
    date '+%Y-%m-%d-%H-%M'
}

python3 ./examples/data_preprocess/gsm8k_multiturn_w_tool.py


mkdir -p logs


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

nohup bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh trainer.experiment_name=qwen2.5-3b_rm-gsm8k-sgl-multiturn-$(now) > logs/gsm8k-$(now).log 2>&1 &
```
