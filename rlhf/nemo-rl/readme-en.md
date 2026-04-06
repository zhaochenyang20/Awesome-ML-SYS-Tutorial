# SGLang × NeMo RL: A Unified High-Performance RL Inference Backend (EN)

> 中文版本: [readme-cn.md](readme-cn.md)

> **TL;DR**
>
> We added a **SGLang generation backend** to **NeMo RL**. The end-to-end pipeline is now fully integrated and working (startup, generation, hot weight updates, shutdown).

---

## Technical Background

To improve compatibility between **NeMo-RL** and the fast-growing **SGLang** reinforcement learning community, we integrated SGLang as an officially supported, high-performance rollout inference backend for NeMo-RL.

With its large and continuously expanding RL user base and proven industrial adoption, SGLang brings unique value to the NeMo ecosystem. By bringing SGLang into NeMo-RL and co-designing the RL systems across both stacks, we can significantly strengthen the NeMo ecosystem in the following ways:

1. Attract a large community of SGLang developers and researchers, expanding NeMo-RL's user base and technical influence.
2. Leverage SGLang's high-throughput data collection for more efficient training, and—through collaboration with key SGLang partners—help keep NeMo-RL at the frontier of the field.

---

## Architecture Design

We treat SGLang as a standalone inference service. The flow is straightforward:

- **Ray** handles resource allocation and server sharding; each server is pinned to a fixed set of GPUs.
- Generation requests are split by the number of servers and dispatched accordingly.
- Weight updates go through an HTTP endpoint: we send the IPC handler to the corresponding server to perform hot updates.

Architecture overview:

<p align="center">
  <img src="nemo_architecture.png" alt="SGLang NeMo RL Architecture Overview" />
</p>

The reward curve produced with the SGLang backend matches the original backend:

<p align="center">
  <img src="reward_curve.png" alt="Reward Curve Comparison between SGLang Backend and Original Backend" />
</p>

---

## Key Implementation

### SGLangGeneration

- Compute `gpus_per_server` and `num_servers`, then construct a DP×TP **NamedSharding**.
- Split the batch along the DP dimension and run generate via the worker group.
- Supports retrieving **server URLs** and **GPU UUIDs**, used for routing hot weight updates.

### SGLangGenerationWorker

- Only the **model owner** launches the server; the remaining workers act as **resource placeholders**.
- Startup method: **separate process** + health check endpoint `/health_generate`.
- Generation uses **asynchronous concurrent requests** via `AsyncLoopThread` + `aiohttp`.
- Supports cache flush and shutdown.

### Hot Weight Update (refit)

- On the training side, tensors are collected via an IPC handler and mapped to the correct server by **GPU UUID**. (Related system design notes: [Awesome-ML-SYS-Tutorial: RL System Design](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/sys-design/readme-1.md))
- Uses SGLang's `/update_weights_from_tensor` API to update weights.
- To reduce dependencies, related serialization utilities are copied from SGLang into `sglang_copied_utils`.

---

## How to Use

### Install Dependencies

```bash
git clone git@github.com:NVIDIA-NeMo/RL.git nemo-rl --recursive
cd nemo-rl
uv venv
uv sync --extra sglang
```

### Example Config

You can use (or reference) `examples/configs/recipes/llm/grpo-qwen2.5-math-1.5b-instruct-1n8g-fsdp2tp1-sglang.yaml`.

Run a GRPO recipe with SGLang:

```bash
uv run examples/run_grpo_math.py \
  --config examples/configs/recipes/llm/grpo-qwen2.5-math-1.5b-instruct-1n8g-fsdp2tp1-sglang.yaml
```

### Migration Path

Switching from other rollout backends only requires updating `backend` and `sglang_cfg`. The model path defaults to `policy.model_name`.

**Minimum config snippet**

```yaml
policy:
  generation:
    backend: "sglang"
    sglang_cfg:
      model_path: ${policy.model_name}
      gpus_per_server: 1
      context_length: 512
      mem_fraction_static: 0.5
      skip_server_warmup: true
```

> **Note:** `gpus_per_server` controls how many GPUs a single SGLang server owns; the total number of servers is derived from the available GPUs.

---

## Future Roadmap

- **Worker Sleep / Awake:** Add a sleep/wakeup mechanism for the worker group to improve resource utilization across different training phases.
- **Disaggregated Mode:** In addition to the current colocated setup, support disaggregated deployments—introducing cross-node weight synchronization and more flexible resource scheduling.
- **Distributed / Megatron Weight Update:** Extend the current FSDP-tensor-based weight update approach to support Megatron and other distributed-state synchronization paths.
- **Generation Features:** Enhance the generation interface to stably support logprob returns, complex sampling parameters, context truncation, and multi-sample batching.
