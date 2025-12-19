---
title: "Power Up Diffusion LLMs: Day‑0 Support for LLaDA 2.0"
author: "Ant Group DeepXPU Team, SGLang Team"
date: "December 19, 2025"
previewImg: /images/blog/dllm/preview.png
---

# Power Up Diffusion LLMs: Day‑0 Support for LLaDA 2.0

## TL;DR

We are excited to introduce the design and implementation of the Diffusion Large Language Model (dLLM) framework within SGLang. By leveraging the existing Chunked-Prefill mechanism, our system achieves:

- Seamless integration: Built into the SGLang ecosystem without core architectural changes.
- Inherited performance: The framework benefits from the existing inference optimization.
- Maximum flexibility: Full flexibility for users to define and customize diffusion decoding algorithms.

## Background

### Motivation
Earlier this year, [LLaDA](https://arxiv.org/pdf/2502.09992) made its debut as the first Diffusion Large Language Model, immediately capturing significant attention from both the academic and industrial communities. This achievement, a collaboration between Renmin University of China and Ant Group, demonstrated that the unique execution paradigm of dLLMs exhibits superior data comprehension capabilities. Moreover, dLLMs enable faster inference speeds compared to Auto-Regressive models, especially in low-latency scenarios such as small batch sizes.

At the same time, as the parameter scale of dLLMs continues to grow, we have also observed scaling-law effects similar to those seen in AR LLMs. In pursuit of better dLLMs, we trained the 100B [LLaDA2.0-flash](https://github.com/inclusionAI/LLaDA2.0/blob/main/tech_report.pdf) model.

However, in the process of training the [LLaDA2.0-flash](https://github.com/inclusionAI/LLaDA2.0/blob/main/tech_report.pdf), we encountered a series of serious AI infrastructure engineering challenges. The most important challenges are the efficency and stability of model evaluation and RL post training.

### Challenges

The current inference engines available for dLLMs are insufficient to support the evaluation and RL post-training requirements of larger-scale dLLMs. For instance, tools like [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) are excellent research tools, better suited for algorithm researchers to tune and validate various Diffusion decoding algorithms. However, they fall short in providing production-ready serving capabilities, such as batching, scheduling, RL ecosystem integration, and parallelism.

In contrast, SGLang is one of the most popular LLM inference engines today and has multiple advantages:

1. Production-Ready: It has been deployed in inference services across thousands of companies, offering mature and reliable engineering capabilities.
2. Technological Lead: SGLang itself incorporates a vast array of excellent and advanced inference optimization techniques, with a continuous flow of new optimizations emerging from the community.
3. Complete Ecosystem: It integrates extremely well with the RL post-training ecosystem, particularly in areas like distributed weight GPU P2P updates.

However, the core issue is that SGLang currently only supports the Auto-Regressive calculation paradigm, and has not yet adapted to the diffusion calculation method for LLMs.

Therefore, the challenge we face is: How can we introduce support for the dLLMs within the existing SGLang framework without compromising its current architecture? The goal is two-fold: allow dLLMs to benefit from all the optimization advantages SGLang offers, while avoiding major, compromising modifications to the SGLang framework just to accommodate diffusion computation.

## Design

### Key Insights

Based on our observations of the current developments in dLLM, we have identified several key insights:

1. Due to the enormous computational cost of Bidirectional Attention Diffusion and its inefficient utilization of the KV Cache, mainstream dLLMs are increasingly moving toward the Block Diffusion architecture.
2. The computation pattern of Block Diffusion bears a high degree of similarity to SGLang's existing Chunked-Prefill process.
3. Unlike auto-regressive language models, diffusion language models utilize various decoding strategies, which require a dedicated interface for flexible decoding algorithm customization.

### Architecture

Our approach is to leverage SGLang’s existing Chunked-Prefill pipeline to implement computational support for Block Diffusion LLM. This method allows us to seamlessly integrate dLLM into the SGLang ecosystem without changing the core SGLang framework, enabling dLLM to directly benefit from all the inference optimization techniques SGLang has accumulated.

<p align="center">
  <img src="./pics/main-flow.png" alt="main execution flow", width="50%">
  <br>
</p>


As illustrated in the diagram, our modifications to the SGLang framework are very restrained, barely touching its core. SGLang's original `generate request` execution flow remains unchanged. Our implementation primarily focuses on leveraging and modifying its existing Chunked Prefill mechanism, with the specific work concentrated on two critical components: the `prefill adder` and `chunked reqs`.

In SGLang, the initial purpose of Chunked Prefill was to maximize GPU utilization. Consequently, the size of a single chunk is typically set quite large—ranging from 2K to 16K tokens in sequence length, depending on the GPU model. When the sequence is long enough, it naturally processes only one request, which is how the current `prefill adder` and `chunked req` are implemented.

However, the decoding process for dLLM differs: it segments the sequence length at the block level. Taking LLaDA2.0 as an example, its block Size is 32 tokens. If we were to follow SGLang's previous logic of processing only one large request at a time, GPU performance would clearly be wasted. Therefore, batching is a crucial problem that must be solved. To achieve efficient batching, we modified both `chunked reqs` and the `prefill adder` to enable them to process multiple Diffusion Blocks within a single computation cycle.

Furthermore, at the actual decoding execution level, we inserted an abstraction layer for the diffusion algorithm between the TP Worker and the Model Runner.

Specifically:
- If the Worker identifies that it is handling a Diffusion model, the execution flow enters this dedicated branch.
- The TP Worker then calls the Diffusion algorithm's `run` function.
- Internally, this algorithm utilizes a forward iteration loop to continuously drive the Model Runner to perform inference computations until the entire Block (e.g., all 32 tokens) is decoded.

### Attention Mask

<p align="center">
  <img src="./pics/casual-mask.png" alt="casual mask", width="50%">
  <br>
</p>

The most significant difference between Block Diffusion and Chunk Prefill during a single model forward pass lies in the handling of the attention mask.

- Block Diffusion utilizes a block-wise causal mask.
- Chunk Prefill for AR model uses the traditional token-wise causal mask.

We can view Block Diffusion as a functional extension to the existing Chunk Prefill mechanism within SGLang. Regarding the specific attention calculation, a single forward pass involves two computational parts, whose final outputs are concatenated:

1. Context Query: This uses the current `Q_curr` (the query vectors of the current block) to perform bidirectional attention against the existing KV Cache. This computation is completely identical for both Block Diffusion and Chunk Prefill. The objective here is to ensure the current block attends to all historical information.
2. Intra-Block Query: This uses the current `Q_curr` against its own KV (i.e., the keys and values within the current block) to perform the forward calculation.
    - Block Diffusion employs bidirectional attention in this step.
    - Chunk Prefill must use a causal Mask in this step.

Simply put, if we visualize the attention mask as a geometric shape for the `Q_curr` portion:
  - The calculation for Chunk Prefill (causal mask) corresponds to a trapezoidal (or triangular) mask.
  - The calculation for Block Diffusion (bidirectional attention) corresponds to a rectangular mask.

## Streaming output animation

Here is an animation comparing the streaming output of LLaDA2.0-flash-CAP (100B / BF16) and gpt-oss-120B (117B / MXFP4). LLaDA2.0-flash-CAP is served using SGLang dLLM with TP8 on 8 × H20, while gpt-oss-120B is served using SGLang's standard AR process on the same hardware.

Both models are asked to implement the quicksort algorithm in 10 programming languages — a task particularly well-suited for diffusion LLMs. As shown, LLaDA2.0-flash-CAP achieves significantly higher throughput at 935 tokens/s, compared to gpt-oss-120B (263 tokens/s) in this scenario.

<p align="center">
  <img src="./pics/llada2-vs-gpt-oss.gif" alt="LLaDA2.0-flash-CAP vs gpt-oss-120B animation", width="50%">
  <br>
</p>

SGLang dLLM supports streaming output just like SGLang auto-regressive models: but it outputs one block (e.g., 32 tokens) at a time instead of one token.

<p align="center">
  <img src="./pics/dllm-animation.gif" alt="Logo preview", width="50%">
  <br>
</p>

## How to Use

### Example Launch Command

```shell
python3 -m sglang.launch_server \
  --model-path inclusionAI/LLaDA2.0-mini \ # example HF/local path
  --dllm-algorithm LowConfidence \
  --dllm-algorithm-config ./config.yaml \ # Optional. Uses the algorithm's default if not set.
  --host 0.0.0.0 \
  --port 30000
```
> NOTE: Use `--dllm-algorithm-config` for advanced configuration of the selected `--dllm-algorithm`. This feature decouples configuration from code, enabling flexible customization and argument passing for user-defined algorithms via a unified entry point.

### Example Client Code Snippet

Just like other supported models, dLLMs can be used via the REST API or offline engine API.

Curl example for making a generation request to the running server:

```bash
curl -X POST "http://127.0.0.1:30000/generate" \
     -H "Content-Type: application/json" \
     -d '{
        "text": [
            "<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>Write the number from 1 to 128<|role_end|><role>ASSISTANT</role>",
            "<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>Write a brief introduction of the great wall<|role_end|><role>ASSISTANT</role>"
        ],
        "stream": true,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 1024
        }
    }'
```

The following contains a code snippet illustrating how to use the offline engine generate content based on given inputs:

```python
import sglang as sgl

def main():
    llm = sgl.Engine(model_path="inclusionAI/LLaDA2.0-mini",
                     dllm_algorithm="LowConfidence",
                     max_running_requests=1,
                     trust_remote_code=True)

    prompts = [
        "<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>Write a brief introduction of the great wall<|role_end|><role>ASSISTANT</role>"
    ]

    sampling_params = {
        "temperature": 0,
        "max_new_tokens": 1024,
    }

    outputs = llm.generate(prompts, sampling_params)
    print(outputs)

if __name__ == '__main__':
    main()
```

## Performance

<p align="center">
  <img src="./pics/llada2_flash_main_bench.png" alt="LLaDA2.0-flash main results", width="50%">
  <br>
</p>

We assessed the task efficacy of LLaDA2.0-flash by benchmarking it against advanced Auto-Regressive (AR) models of a comparable scale on a wide range of standard evaluation tasks.

The overall results indicate that the LLaDA2.0 architecture is not only highly competitive, but also shows a promising trend of closing the capability gap with AR models.

<p align="center">
  <img src="./pics/llada2_despine_comparison.png" alt="LLaDA2.0-flash performance", width="50%">
  <br>
</p>

The chart presents two complementary measurements for LLaDA2.0‑flash:

- Average score and tokens‑per‑forward (TPF) obtained with and without Confidence‑Aware Parallel (CAP) training across 12 benchmark tasks.
- Inference speed (tokens per second) of LLaDA2.0‑flash, benchmarked against AR models of comparable size on HumanEval, MBPP, GSM8K, and CRUXEval suites.

All numbers are collected under a consistent serving environment (SGLang with TP8 on H20), ensuring a fair comparison between the diffusion LLM and the Auto-Regressive baselines.

With a 0.95 threshold decoder, LLaDA2.0-flash-CAP achieved 500 TPS, significantly outperforming standard LLaDA2.0-flash (383 TPS) and delivering up to a 1.9× speedup over AR baselines (258 TPS and 237 TPS) with small batch sizes.

## Roadmap

### Implemented key features

The current implementation fully supports the following critical serving features:

- Block Diffusion LLM framework main logic
- Full KV cache support for sequence management
- Model integration for LLaDA-2.0-mini/flash
- Support for custom decoding algorithm
- Full streaming I/O capability
- Batching support (reviewing)
- Tensor parallelism support
- Cuda graph optimization

### Mid & Long-term Roadmaps

[Roadmap for 2025-Q4 and 2026-Q1](https://github.com/sgl-project/sglang/issues/14199)<br>
[RFC: Block Diffusion Large Language Model (dLLM) Framework In SGLang](https://github.com/sgl-project/sglang/issues/12766)<br>
- Support more system optimizations that autoregressive language models already have
- Integrate additional common diffusion decoding strategies/algorithms (e.g, [Fast-dLLM v2](https://arxiv.org/pdf/2509.26328))
- Add compatibility for non-block dLLMs (e.g., LLaDA & RND1)

## Reference

[LLaDA1 technique report](https://arxiv.org/pdf/2502.09992)<br>
[LLaDA2 technique report](https://github.com/inclusionAI/LLaDA2.0/blob/main/tech_report.pdf)<br>
[Fast-dLLM v2 technique report](https://arxiv.org/pdf/2509.26328)

## Acknowledgements

- Ant Group DeepXPU Team: [Zehuan Li](https://github.com/Clawseven), [Tiwei Bie](https://github.com/btw616), Zhonghui Jiang, Jinghua Yao, Yusong Gao, [Mingliang Gong](https://github.com/brightcoder01), Jianfeng Tan
- Ant Group inclusionAI Team: Kun Chen, [Zenan Huang](https://lccurious.github.io/), Lin Liu, Fuyuan Chen, Lun Du, Da Zheng 
- SGLang dLLM Team: [Jinwei Yao](https://kivi-yao.github.io/), [Mick Qian](https://github.com/mickqian), [Liangsheng Yin](https://www.lsyin.me/), [BBuf](https://github.com/BBuf), Banghua Zhu, [Chenyang Zhao](https://zhaochenyang20.github.io/Chayenne/)
- NVIDIA Fast-dLLM Team: [Chengyue Wu](https://hills-code.github.io/), [Hao Zhang](https://research.nvidia.com/person/hao-zhang), [Enze Xie](https://xieenze.github.io/), [Song Han](https://hanlab.mit.edu/songhan)