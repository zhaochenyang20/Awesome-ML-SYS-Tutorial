<!-- # Power Up Diffusion LLM in SGLang: Day0 Support for the 100B LLaDA2

## Preview

<p align="center">
  <img src="./pics/sgl-dllm.png" alt="Logo preview", width="50%">
  <br>
  <em>Logo preview</em>
</p>

## TL;DR

We have designed and implemented Diffusion LLM (dLLM) support in SGLang. By leveraging SGLang's existing Chunked-Prefill mechanism, we successfully integrated Block-wise dLLMs into current SGLang ecosystem with minimal invasive modifications, enabling dLLMs to directly benefit from all the inference optimization techniques accumulated by SGLang.

## Background

### Motivation

Earlier this year, **LLaDA**[0] made its debut as the **first dLLM**, immediately capturing significant attention from both the academic and industrial communities. This achievement, a collaboration between **Renmin University of China** and **Ant Group**, demonstrated that the unique execution paradigm of dLLMs exhibits **superior data comprehension capabilities** and enables **faster inference speeds** compared to Auto-Regressive models.

【todo：我觉得这里不够严谨，因为 AR 模型的速度和 dllm 的速度比较需要加上一些 setting？比如写出来是很小的 batch size，开不开 speculative decoding】

At the same time, as the parameter scale of dLLMs continues to grow, we have also observed scaling-law effects similar to those seen in AR models. In pursuit of better dLLMs, we trained the 100B-scale **LLaDA2.0-flash**[1] model.

【todo：我一般同时会放 markdown 的 link，比如 `[xxxx](https://github.com/xxxx)` 和 [1] 这种 link，全文修改下，感谢，安琪的博客也这么改过】

However, in the process of training the LLaDA2.0-flash model, we encountered serious AI infrastructure engineering challenges. Key among these were the critical hurdles of model evaluation performance and RL post-training support.

### Challenges

To be specific, current inference engines available for dLLMs are insufficient to support the evaluation and RL post-training requirements of larger-scale dLLMs. For instance, serving frameworks like Fast-dLLM are excellent research tools, better suited for algorithm researchers to tune and validate various Diffusion decoding algorithms. However, they fall short in providing production-ready serving capabilities, struggling to support crucial engineering needs such as batching, scheduling, RL ecosystem integration, and parallelism. In contrast, SGLang as the most popular LLM inference engines today, boast multiple advantages:

- **Production-Ready:** It has been deployed in inference services across thousands of companies, offering mature and reliable deployment capabilities.
- **Technological Lead:** SGLang itself incorporates a vast array of excellent and advanced inference optimization techniques, with a continuous flow of new optimizations emerging from the community. In our case, chunked prefill and CUDA graph optimization are particularly useful for dLLMs.
- **Complete Ecosystem:** SGLang integrates extremely well with the RL post-training ecosystem, particularly in areas like distributed weight GPU P2P updates.

However, the core issue is that SGLang only supports the Auto-Regressive calculation paradigm, and has not yet adapted to the diffusion calculation method for LLMs. Therefore, the challenge we face is: How can we power up dLLMs within SGLang's existing ecosystem without compromising its current architecture? The goal is two-fold: allow dLLMs to benefit from all the optimization advantages SGLang offers, while avoiding major, compromising/invasive modifications to the SGLang framework just to accommodate diffusion computation.

## Design

Based on our observations of the current developments in dLLM, there are several key insights that guide our design:

- Due to the enormous computational cost of **Bidirectional Attention Diffusion** and its inefficient utilization of the KV Cache, mainstream dLLMs are increasingly moving toward the **Block-wise Diffusion** architecture.
- The computation pattern of **Block-wise Diffusion** bears a high degree of similarity to SGLang's existing **Chunked-Prefill** process.
- Unlike auto-regressive language models, diffusion language models utilize various decoding strategies, which require a **dedicated interface for flexible decoding algorithm customization**.

### Architecture

Our approach is to leverage SGLang’s existing Chunked-Prefill pipeline to implement computational support for Block-wise dLLMs. This method allows us to seamlessly integrate dLLM into SGLang ecosystem, enabling dLLMs to directly benefit from all the inference optimization in SGLang.

<p align="center">
  <img src="./pics/main-flow.png" alt="Logo preview", width="50%">
  <br>
  <em>Block-wise diffusion main execution flow</em>
</p>


As illustrated in the diagram, our modifications are very restrained, barely touching its core. SGLang's original `generate request` execution flow remains unchanged. Our implementation primarily focuses on leveraging and modifying its existing Chunked Prefill mechanism, with the specific work concentrated on two critical components: the `prefill adder` and `chunked reqs`.

In SGLang, the initial purpose of Chunked Prefill was to maximize GPU utilization. Consequently, the size of a single chunk is typically set quite large—ranging from 2K to 16K tokens in sequence length, depending on the GPU type. When the sequence is long enough, it naturally processes only one request, which is how the current `prefill adder` and `chunked req` are implemented.

However, the decoding process for Diffusion models differs: it segments the sequence length at the Block level. Taking LLaDA-2 as an example, its Block Size is 32 tokens. If we were to follow SGLang's previous logic of processing only one request at a time, GPU performance would clearly be wasted. Therefore, batching is a crucial problem that must be solved. To achieve efficient batching, we modified both `chunked reqs` and the `prefill adder` to enable them to process multiple Diffusion Blocks within a single computation cycle.

Furthermore, at the actual decoding execution level, we inserted an abstraction layer for the Diffusion algorithm between the TP Worker and the Model Runner (model executor). Specifically:

- If the Worker identifies that it is handling a Diffusion model, the execution flow enters this dedicated branch.
- The TP Worker then calls the Diffusion algorithm's `run` function.
- Internally, this algorithm utilizes a forward iteration loop to continuously drive the Model Runner to perform inference computations until the entire Block (e.g., all 32 tokens) is decoded.

### Casual Mask

<p align="center">
  <img src="./pics/casual-mask.png" alt="Logo preview", width="50%">
  <br>
  <em>Casual mask</em>
</p>

Yet, the most significant difference between Block Diffusion and Chunk Prefill during a single Model Forward Pass lies in the handling of the Attention Mask. Block Diffusion utilizes a Block-wise Causal Mask, while Chunk Prefill for AR model uses the traditional Token-wise Causal Mask.

We take Block Diffusion as a functional extension to the existing Chunk Prefill mechanism within SGLang. Regarding the specific Attention calculation, a single forward pass involves two computational parts, whose final outputs are concatenated:

1. Cache Query: This uses the current `Q_curr` (the query vectors of the current block) to perform Bidirectional Attention against the existing KV Cache. This computation is completely identical for both Block Diffusion and Chunk Prefill. The objective here is to ensure the current block attends to all historical information.
2. Intra-Block Query: This uses the current `Q_curr` against its own KV (i.e., the keys and values within the current block) to perform the forward calculation. Block Diffusion employs Bidirectional Attention in this step, while Chunked Prefill for AR model must use a Causal Mask in this step.

Simply put, if we visualize the attention mask as a geometric shape for the `Q_curr` portion: The calculation for Chunked Prefill (Causal Mask) corresponds to a trapezoidal (or triangular) mask. The calculation for Block Diffusion (Bidirectional Attention) corresponds to a rectangular mask.

### Streaming output animation

SGLang dLLM supports streaming output just like SGLang AR models: but it outputs one Block (e.g., 32 tokens) at a time instead of one token.

<p align="center">
  <img src="./pics/dllm-animation.gif" alt="Logo preview", width="50%">
  <br>
  <em>SGLang dLLM animation</em>
</p>

【TOOD： 有没有一个更直观的gif图片展示inference speed和样式什么的？比如原来那个gif可以分成两个，一个是spin up sglang server，然后是query sglang server以后生成回复很快？甚至可以把gptoss和100b这个inference speed对比一下？比如mercury之前就有个对比 】

【TODO：能不能做那种 AR 和 DLLM 左右对比的】

This figure is not accelerated, and the output speed is representative.

## Usage

### Launch Command Example

```shell
python3 -m sglang.launch_server \
  --model-path inclusionAI/LLaDA2.0-mini \ # example HF/local path
  --dllm-algorithm LowConfidence \
  --dllm-algorithm-config ./config.yaml \ # Optional. Uses the algorithm's default if not set.
  --host 0.0.0.0 \
  --port 30000
```

> ⚠️ **NOTE:** Use `--dllm-algorithm-config` for advanced configuration of the selected `--dllm-algorithm`. This feature decouples configuration from code, enabling flexible customization and argument passing for user-defined algorithms via a unified entry point.

### Client Code Snippet Example

Just like other supported models, dLLMs can be used via the REST API or offline engine API.

Curl example for making a generation request to the launched server:

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
  <img src="./pics/llada2_flash_main_bench-1.png" alt="LLaDA2.0-flash main results", width="50%">
  <br>
  <em>LLaDA2.0-flash main results</em>
</p>

We evaluated the performance of LLaDA2.0-flash models and same-level Auto-Regressive (AR) baselines across 47 benchmarks.

The overall results indicate that the LLaDA2.0 architecture is not only
highly competitive, but also shows a promising trend of closing the performance gap with AR models.

<p align="center">
  <img src="./pics/llada2_despine_comparison-1.png" alt="LLaDA2.0-flash performance", width="50%">
  <br>
  <em> LLaDA2.0-flash performance in SGLang. Average score and tokens-per-forward (TPF) for LLaDA2.0-flash with and without Confidence-Aware Parallel(CAP) training across 12 benchmarks. Inference speed (tokens per second) of LLaDA2.0-flash compared with similarly sized AR models on 4 code and math benchmarks</em>
</p>

We compared the average inference throughput (TPS) of LLaDA2.0-flash models against  AR baselines (Ling-flash-2.0 and Qwen3-30B-A3B-Instruct-2507) on HumanEval, MBPP, GSM8K, and CRUXEval. All models were served using SGLang for a fair comparison.

With a 0.95 threshold decoder, LLaDA2.0-flash-CAP achieved 535 TPS, significantly outperforming standard LLaDA2.0-flash (383 TPS) and delivering up to a 2.1× speedup over AR baselines (256 TPS and 237 TPS). This demonstrates that **dLLMs can surpass AR models in inference speed within the SGLang framework.**

【TODO：这里还是要补充下 setting】

## Roadmap

### Implemented key features

The current implementation fully supports the following critical serving features:

- [x] **Block-wise dLLM Framework** main logic
- [x] Full **KV Cache** support for sequence management
- [x] Model integration for **LLaDA-2.0-mini/flash**
- [x] Support for **Custom Decoding Strategies** (e.g., implementing **low-confidence policy**)
- [x] Full **Streaming I/O** capability
- [x] **Batching** support (leveraging SGLang's core scheduling)
- [x] **Tensor Parallelism (TP)** support
- [x] **CUDA Graph** optimization

### Mid & Long-term Roadmaps

【TODO：这里放上 roadmap 的 link 和 RFC link】

- [ ] Support more system optimizations that autoregressive language models already have
- [ ] Integrate additional common diffusion decoding strategies/algorithms (e.g, Fast-dLLM v2[2])
- [ ] Add compatibility for non-block-wise dLLMs (e.g., LLaDA & RND1)


## Reference

[0] LLaDA 1: [technique report](https://arxiv.org/pdf/2502.09992)

[1] LLaDA 2: [technique report](https://github.com/inclusionAI/LLaDA2.0/blob/main/tech_report.pdf)

[2] Fast-dLLM v2: [technique report](https://arxiv.org/pdf/2509.26328)

## Acknowledgements

- Ant Group DeepXPU Team: [Zehuan Li](https://github.com/Clawseven), [Tiwei Bie](https://github.com/btw616), Zhonghui Jiang, Yusong Gao, [Mingliang Gong](https://github.com/brightcoder01), Jianfeng Tan
- Ant Group inclusionAI Team: Kun Chen, Zenan Huang, Lin Liu, Fuyuan Chen, Lun Du, Da Zheng 
- SGLang dLLM Team: [Jinwei Yao](https://kivi-yao.github.io/), [Mick Qian](https://github.com/mickqian), [Liangsheng Yin](https://www.lsyin.me/), [BBuf](https://github.com/BBuf), [Chenyang Zhao](https://zhaochenyang20.github.io/Chayenne/)
- NVIDIA Fast-dLLM Team: [Chengyue Wu](https://hills-code.github.io/), [Hao Zhang](https://research.nvidia.com/person/hao-zhang), [Enze Xie](https://xieenze.github.io/), [Song Han](https://hanlab.mit.edu/songhan) -->
