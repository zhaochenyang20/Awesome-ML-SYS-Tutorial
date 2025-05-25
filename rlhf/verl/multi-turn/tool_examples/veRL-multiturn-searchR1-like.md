# New Feature: Search Tool Integration in Multi-Turn RL Training

* Added search tool invocation capability to **veRL-sglang MultiTurnRL**, allowing the model to call the search interface during the Actor rollout phase and integrate retrieval results directly into the training loop.
- Pull Request: [volcengine/verl#1682](https://github.com/volcengine/verl/pull/1682)
* [Training Curves on Weights & Biases](https://wandb.ai/lingchang-ustc/search_async_rl/workspace?nw=nwuserlingchang "Training Curves on W&B")

# Usage

### Environment Setup

**Create a new Docker container**

```docker
docker run \
    -it \
    --shm-size 32g \
    --gpus all \
    -v /models/shared/.cache:/root/.cache \
    --ipc=host \
    --network=host \
    --privileged \
    --name sglang_{your-name} \
    lmsysorg/sglang:dev \
    /bin/zsh
```

If you need to restart the container after exiting:

```bash
docker start -i sglang_{your-name}
```

**Update Python and use a virtual environment**

```bash
apt update
apt install -y python3.10 python3.10-venv

# Create a virtual environment
python3 -m venv ~/.python/veRL-multiturn-rollout

# Activate the virtual environment
source ~/.python/veRL-multiturn-rollout/bin/activate

# Install uv
python3 -m pip install uv
```

**Install veRL from upstream**

```bash
cd ~
git clone https://github.com/volcengine/verl.git
cd verl

# Install verl
python3 -m uv pip install .
python3 -m uv pip install -r ./requirements_sglang.txt

# Manually install flash-attn
python3 -m uv pip install wheel
python3 -m uv pip install packaging
python3 -m uv pip install flash-attn --no-build-isolation --no-deps
```

**Set up your own local retrieval (skip if using your own service)**

* This example uses the local dense retriever from the searchR1 demo. See the [searchR1 documentation](https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/refs/heads/main/docs/retriever.md "searchR1") for details.

  * Requires a GPU (approximately 5–7 GB VRAM per card), high accuracy and fast speed.
  * For a CPU-only version, see the [detailed guide](https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/retriever.md "detailed guide") (basic test code works, but retrieval accuracy is lower, which may hurt training performance).
* **Note:** **We recommend using conda to install the retrieval service environment**, as `faiss-gpu` may fail to install in a venv.

```bash
# Download Miniconda installation script
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

# Install to $HOME/miniconda3 in batch mode
bash ~/miniconda.sh -b -p $HOME/miniconda3

# Activate conda (current shell only)
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

# (Optional) Add conda to default shell startup
conda init

# Reload shell configuration
source ~/.bashrc

# Create and activate the 'retriever' environment with Python 3.10
conda create -n retriever python=3.10 -y
conda activate retriever

# Install PyTorch (with GPU support) and related libraries
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install other Python packages
pip install transformers datasets pyserini huggingface_hub

# Install GPU version of faiss
conda install faiss-gpu=1.8.0 -c pytorch -c nvidia -y

# Install API service framework
pip install uvicorn fastapi
```

* Download indexing and corpus

```bash
conda activate retriever

save_path=/the/path/to/save
python examples/sglang_multiturn/searchR1_like/local_dense_retriever/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

* Launch the local flat e5 retrieval server

  * The first start will download the model and load the index, which takes \~2–3 minutes on a normal connection.
  * After startup, each GPU will use approximately 5–7 GB VRAM (allowing training on the same node).

```bash
conda activate retriever

index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python examples/sglang_multiturn/searchR1_like/local_dense_retriever/retrieval_server.py \
  --index_path $index_file \
  --corpus_path $corpus_file \
  --topk 3 \
  --retriever_name $retriever_name \
  --retriever_model $retriever_path \
  --faiss_gpu
```

### Running Experiments on 8 × H20 GPUs

**Set** `WANDB_API_KEY`
If you are unsure how to get an API key, refer to [this guide](https://community.wandb.ai/t/where-can-i-find-the-api-token-for-my-project/7914 "this guide").

```bash
export WANDB_API_KEY={YOUR_WANDB_API_KEY}

# Define a timestamp function
function now() {
    date '+%Y-%m-%d-%H-%M'
}
```

**Preprocess the dataset** (run data processing and training commands in the `veRL-multiturn-rollout` venv)

```bash
# To define custom prompts, modify examples/data_preprocess/prompt.yaml
# By default, output is stored in ~/data/searchR1_processed_direct
python3 examples/data_preprocess/preprocess_searchR1_dataset.py \
  --config examples/data_preprocess/prompt.yaml
```

**Run the tests**

```bash
# Ensure now() is defined
# Create log directory
mkdir -p logs

# Set GPUs and run, using an appropriate log path
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

nohup bash examples/sglang_multiturn/searchR1_like/run_qwen2.5-3b_instruct_search_multiturn.sh \
  trainer.experiment_name=qwen2.5-3b-it_rm-searchR1-like-sgl-multiturn-$(now) \
  > logs/searchR1-like$(now).log 2>&1 &
```

### Custom Configuration

**Basic Configuration**
Enable multi-turn inference by adding the following fields to your config:

```yaml
actor_rollout_ref:
  rollout:
    name: "sglang_async"
    multi_turn:
      enable: True
```

Specify `retrieval_service_url` and concurrency settings in `examples/sglang_multiturn/config/tool_config/search_tool_config.yaml`:

```yaml
tools:
  - class_name: "verl.tools.search_tool.SearchTool"
    config: {
      "retrieval_service_url": "http://127.0.0.1:8000/retrieve",
      "num_workers": 120,
      "rate_limit": 150,
      "default_timeout": 30
    }
```

## Note

For debugging only, you can comment out the validation process at the beginning of the `fit` function in `ray_trainer.py` (the validation dataset is large and takes \~6000 s per run):

`verl/trainer/ppo/ray_trainer.py`

```python
# perform validation before training
# currently, we only support validation using the reward_function.
# if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
#     val_metrics = self._validate()
#     assert val_metrics, f"{val_metrics=}"
#     pprint(f"Initial validation metrics: {val_metrics}")
#     logger.log(data=val_metrics, step=self.global_steps)
#     if self.config.trainer.get("val_only", False):
#         return
```
