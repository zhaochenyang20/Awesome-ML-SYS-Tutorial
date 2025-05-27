# [Feat]: Search Tool Invocation in Multi-Turn RL Training

### What does this PR do?
- As veRL users, we want the model to invoke designated tools during the Actor rollout phase and seamlessly integrate their outputs into the training pipeline.

- We have added **search-tool** invocation capability to **veRL-sglang MultiTurnRL**, enabling the model to issue retrieval requests during Actor rollout and directly leverage the returned results for training.

- providing the community with a reimplementation similar to searchR1.

- PR Link: [volcengine/verl#1682](https://github.com/volcengine/verl/pull/1682)  

- Training curves on Wandb: [search_async_rl](https://wandb.ai/lingchang-ustc/search_async_rl/workspace?nw=nwuserlingchang)  
- Thanks to the SGlang team and the author of searchR1 for their efficient support!  
@BoWen @Chenyang Zhao @Xiang Long @yuhao @nan @Jin Pan @Yuzhen Zhou @Shenggui Li

# How to Use

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

If you need to restart after exiting the container:

```bash
docker start -i sglang_{your-name}
```

**Update Python and use a virtual environment**

```bash
apt update
apt install -y python3.10 python3.10-venv

# Create the virtual environment
python3 -m venv ~/.python/veRL-multiturn-rollout

# Activate the virtual environment
source ~/.python/veRL-multiturn-rollout/bin/activate

# Install uv
python3 -m pip install uv
```

**Install veRL upstream**

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

* Here we choose the local dense retriever provided in the searchR1 example; see [searchR1](https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/refs/heads/main/docs/retriever.md) for detailed documentation.

  * Requires GPU (approximately 5–7 GB GPU memory per card during operation), high accuracy, fast.
  * For a GPU-free version, refer to the [detailed documentation](https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/retriever.md) in searchR1 (simple test code can be used, but lower retrieval accuracy may degrade training performance).
* **Note**: **It is recommended to use conda to install the environment for the retrieval service**, as faiss-gpu installation often fails in venv (due to issues with faiss).
* In this configuration, the above venv environment is used for training; the retriever uses the conda environment.

```bash
# Download the Miniconda installer script
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

# Install to $HOME/miniconda3 in batch mode
bash ~/miniconda.sh -b -p $HOME/miniconda3

# Activate conda (only in the current shell)
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

# (Optional) Add conda to your default shell startup script for automatic availability
conda init

# Reload shell configuration
source ~/.bashrc

# Create and activate the retriever environment with Python 3.10
conda create -n retriever python=3.10 -y
conda activate retriever

# Install PyTorch (with GPU support) and related libraries
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install other Python packages
pip install transformers datasets pyserini huggingface_hub

# Install the GPU version of faiss
conda install faiss-gpu=1.8.0 -c pytorch -c nvidia -y

# Install the API service framework
pip install uvicorn fastapi
```

* Download indexing and corpus

  * The download is about 60–70 GB (approximately 132 GB when uncompressed)

```bash
conda activate retriever

save_path=/the/path/to/save
python examples/sglang_multiturn/searchR1_like/local_dense_retriever/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

* **Start the local flat e5 retrieval server**

  * The first startup will download the model and load the index. Excluding the download, normal startup time is 1–2 minutes.
  * After startup, each GPU uses about 5–7 GB of memory (RL training can be performed on the same node).

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

### Testing on 8 × H20

**Set** `WANDB_API_KEY`

If you do not know how to get an API key, refer to [here](https://community.wandb.ai/t/where-can-i-find-the-api-token-for-my-project/7914).

```bash
export WANDB_API_KEY={YOUR_WANDB_API_KEY}

# Define a timestamp function
function now() {
    date '+%Y-%m-%d-%H-%M'
}
```

**Preprocess the dataset** (the following data processing and training commands are executed in the veRL-multiturn-rollout venv environment)

```bash
# To define your own prompt, modify examples/data_preprocess/prompt.yaml
# Default storage directory is ~/data/searchR1_processed_direct
python3 examples/data_preprocess/preprocess_searchR1_dataset.py --config examples/data_preprocess/prompt.yaml
```

**Run tests**

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

Enable multi-turn inference by setting the following fields in your config:

```yaml
actor_rollout_ref:
  rollout:
    name: "sglang_async"
    multi_turn:
      enable: True
```

In `examples/sglang_multiturn/config/tool_config/search_tool_config.yaml`, specify `retrieval_service_url` and concurrency settings:

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

## Notes:

1. Total training time is about 27 hours, and each validation takes about 6000 s (validation dataset is too large: 51 k). 
For debugging only, you can comment out the val process at the beginning of the `fit` function in `ray_trainer.py`, as follows:
- verl/trainer/ppo/ray\_trainer.py


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

2. The training performance is slightly lower compared to the original paper. After aligning with the searchR1 author, potential reasons are:

   * Special tokens (e.g., `<tool_call>`, `<tool_response>`) are not fully aligned.
   * Modified the original searchR1 EM reward, adding penalties for excessive occurrences of `<answer>` and `</answer>` in the model responses.
   * Minor misaligned hyperparameters, etc.

3. Increasing `micro_batch_size_per_gpu` too much can easily cause OOM.

## References
We would like to thank the [search-r1](https://github.com/xxx/search-r1) project for its valuable support and inspiration. If you find our work helpful, we kindly ask that you also cite the original project:

```bibtex
@article{jin2025search,
  title={Search-r1: Training llms to reason and leverage search engines with reinforcement learning},
  author={Jin, Bowen and Zeng, Hansi and Yue, Zhenrui and Yoon, Jinsung and Arik, Sercan and Wang, Dong and Zamani, Hamed and Han, Jiawei},
  journal={arXiv preprint arXiv:2503.09516},
  year={2025}
}
```