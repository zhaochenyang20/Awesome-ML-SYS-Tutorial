# Test log

1. 构建 container container

```bash
docker run \
 -it \
 --shm-size 32g \
 --gpus all \
 -v {CACHE PATH}:/root/.cache \
 --ipc=host \
 --network=host \
 --privileged \
 --name verl_sppo \
 lmsysorg/sglang:dev \
 /bin/zsh
```

2. 安装 verl

```bash
mkdir -p /tmp
chmod 1777 /tmp
apt update
apt install -y python3.10 python3.10-venv
python3 -m ensurepip --upgrade
sudo apt install tmux
python3 -m venv ~/.python/sglang
source ~/.python/sglang/bin/activate
python3 -m pip install uv
python3 -m uv pip install wheel
python3 -m uv pip install packaging
python3 -m uv pip install flash-attn --no-build-isolation --no-deps
cd ~
git clone https://github.com/yhyang201/verl.git && cd verl
git checkout sppo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
python3 -m uv pip install -e ".[sglang]"
```

3. wandb login

```bash
wandb login
```

4. Download dataset and model

```bash
python3 examples/data_preprocess/math_dataset.py --local_dir ~/data/math
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir $HOME/models/Qwen2.5-7B-Instruct
```

5. run bash (tested on h20x4)

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd recipe/sppo
bash run_qwen2.5-7b_rm.sh
```
