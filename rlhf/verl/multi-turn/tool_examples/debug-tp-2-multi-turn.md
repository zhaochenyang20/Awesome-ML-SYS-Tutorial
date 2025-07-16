# A Quick Reproduce to Evaluate the Speed of TP 2

```bash
docker run \
    -it \
    --shm-size 32g \
    --gpus all \
    -v {YOUR_HUGGINGFACE_CACHE_DIR}:/root/.cache \
    --ipc=host \
    --network=host \
    --privileged \
    --name sglang-FSDP-TP-2 \
    lmsysorg/sglang:dev \
    /bin/zsh
```

```bash
apt update
apt install -y python3.10 python3.10-venv
python3 -m ensurepip --upgrade

python3 -m venv ~/.python/veRL-multiturn-rollout

source ~/.python/veRL-multiturn-rollout/bin/activate

python3 -m pip install uv
```

```bash
cd ~
git clone -b multiturn_profile_log https://github.com/PrinsYin/verl.git
cd verl

python3 -m uv pip install -e ".[sglang]"

python3 -m uv pip install wheel
python3 -m uv pip install packaging
# best to use for torch 2.6.0 and sglang 0.4.6.post1 flash-attn 2.7.4.post1
python3 -m uv pip install flash-attn==2.7.4.post1 --no-build-isolation --no-deps

python3 -m uv pip install .
python3 -m uv pip install -r ./requirements.txt
```

```bash
export WANDB_API_KEY={YOUR_WANDB_API_KEY}

function now() {
    date '+%Y-%m-%d-%H-%M'
}
```

```bash
python3 ./examples/data_preprocess/gsm8k_multiturn_w_tool.py
```

```bash
bash profile-sglang-multi-turn.sh 1

# The final 1 is the number of SGLang TP size.
```
