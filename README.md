# Awesome-ML-SYS-Tutorial

## [English Version](./README.md) | [Chinese Version](./README-cn.md)

My learning notes/codes for ML SYS.

I've been writing this blog series intermittently for over a year now, and it's almost become an RL Infra Learning Note 😂

I often see discussions about whether ML SYS or AI Infra is worth getting into, and how to start. Everyone's choice is different. For me, I simply want to pursue the truth in algorithms:

> A large number of RL conclusions derived from papers are based on RL infrastructure in the open-source community that may be extremely flawed. I've been involved in RL infra development for over a year, and I've seen numerous community experts diligently working, but the fact is that RL infra, whether open-source or within major companies, still has many problems. It is absolutely worth questioning whether the high-level conclusions drawn from this flawed infrastructure are correct. When I was reviewing for ICLR this year, I often asked the papers assigned to me, "If the framework you are using has implementation issues itself, can your conclusions still hold?" Although I never deducted points for this reason, no one could provide an answer that resolved my fundamental doubt.
>
> Therefore, some excellent researchers I know are keen to participate in infra development, spending most of their time on foundational work to rigorously ensure that the algorithm they plan to develop next has a correct basis. I greatly admire them and agree with such rigor—they are my role models. The same is true for our SGLang RL community. With so much human power and time, we all hope to provide the most correct and concise RL foundation possible, whether it's for companies training models or researchers developing new algorithms, with the goal of genuinely serving everyone in the community. Thank you for your recognition, and I look forward to hearing from interested friends who wish to contact me and join us!

After a year of going around in circles, this is the resolve that keeps me going in Infra: to make a contribution to the community by building a correct foundation, thereby helping to ensure that correct conclusions are drawn.

Coming back to the topic, this series of podcasts started in August 2024, when I began learning ML SYS notes following the opportunity to use [SGLang](https://github.com/sgl-project/sglang) during my research. It's largely written by me, with content focusing on RL infra, online/offline inference systems, and some fundamentals of AI Infra. Over the past year, starting from two or three articles and thirty to fifty Github Stars, to now exceeding 4.5K Stars, I have become a minor technical celebrity. I am deeply honored and grateful for the support.

I would like to thank my advisors, Professor Quanquan Gu, Dr. Ying Sheng, and Dr. Linmin Zheng, for the immense help and guidance they gave me in my study of AI Infra, career choices, and life path. Although I am no longer pursuing a Ph.D. at UCLA due to personal reasons, this journey after my undergraduate graduation has been an incredibly valuable experience. I have now joined RadixArk full-time, continuing my research in RL Infra. We will continue to share AI Infra-related technology and thoughts through my blog, via unofficial channels. I also hope everyone will reach out to us, join the SGLang open-source community, and together build open-source AI Infra that changes the world and is worth being proud of for a lifetime!

## RLHF System Development Notes

### slime Framework

- [Achieving Speed and Accuracy: A Comprehensive Solution to Train-Inference Mismatch in RL](./rlhf/slime/mismatch/blog-en.md): Introduces two solutions provided by the slime framework for the train-inference mismatch problem: achieving perfect True On-Policy training through kernel-level alignment, and mitigating the mismatch using algorithms like TIS/MIS. Also available in [Chinese version](./rlhf/slime/mismatch/blog-cn.md).
- [Support FSDP2 as A Training Backend for slime](./rlhf/slime/fsdp/readme_en.md): Added FSDP as a training backend to slime, and aligned it with Megatron. FSDP is more flexible in supporting models with architectural innovations like Qwen3-Next/gpt-oss and helps us further support VLM RL. Also available in [Chinese version](./rlhf/slime/fsdp/readme.md) and on [Zhihu](https://zhuanlan.zhihu.com/p/1979141713449742500).
- [Unified FP8: Moving Beyond Mixed Precision for Stable and Accelerated MoE RL](./rlhf/slime/fp8/readme_en.md): Fully utilizing FP8 for both sampling (Rollout) and training (Training) in RL. Also available in [Chinese version](./rlhf/slime/fp8/readme.md) and on [Zhihu](https://zhuanlan.zhihu.com/p/1974681194017865986).
- [Power Up Speculative Decoding In Reinforcement Learning](./rlhf/slime/spec/readme-en.md): Introduces speculative decoding into the RL sampling process, significantly boosting sampling speed when the batch size is appropriate; moreover, the draft model is updated during training. Compared to freezing the draft model, the accepted length remains consistently high, yielding long-term stable positive returns. Also available in [Chinese version](./rlhf/slime/spec/readme.md).
- [An In-Depth Look at the Elegant Design and Source Code of the slime RL Framework](./rlhf/slime/code-walk-through/readme_en.md): slime source code appreciation. Also available on [Zhihu](https://zhuanlan.zhihu.com/p/1946402397409740613) and in [Chinese version](./rlhf/slime/code-walk-through/readme.md).
- [Pending Review] [slime FSDP Setup Guide](./rlhf/slime/fsdp/release_log/setup_fsdp.md): Records how to test FSDP on slime, including H-cards and B-cards, and both Colocate and Disaggregated placement methods.
- [Pending Review] [Chunked Parallel Computation of GAE in PPO (slime Implementation)](./rlhf/slime/batch-GAE/ppo-gae-chunk.md): Rewrites the standard backward recurrence of GAE into chunk-based parallel prefix scanning, significantly mitigating the GAE computation bottleneck in long sequence scenarios, achieving about $100\times–300\times$ acceleration in slime. Also available on [Zhihu](https://zhuanlan.zhihu.com/p/1975237289425798560).

### AReal Framework

- [AReal Code Walk Through](./rlhf/areal/code-walk-through_EN.md) AReal source code appreciation. Also available on [Zhihu](https://zhuanlan.zhihu.com/p/1983417813080236770) and in [Chinese version](./rlhf/areal/code-walk-through_CN.md).


### verl Framework

- [Analyzing VLM RL Training Memory Leaks via Torch Memory Snapshot](./torch/mem-snapshot/readme-en.md): Analysis of SGLang memory leak issues and solutions. Also available on [Zhihu](https://zhuanlan.zhihu.com/p/1943202817247519535) and in [Chinese version](./torch/mem-snapshot/readme.md).
- [Latency optimization for weight updates](./sglang/latency-accelerate-for-weight-updates/readme.md): A debug process for efficiency. Also available on [Zhihu: A record of optimizing SGLang weight update latency](https://zhuanlan.zhihu.com/p/9908228168).
- [In-Depth Understanding of verl Source Code (Initialization)](./rlhf/verl/multi-turn/code-walk-through/readme_EN.md): Also available on [Zhihu](https://zhuanlan.zhihu.com/p/1920751852749849692) and in [Chinese version](./rlhf/verl/multi-turn/code-walk-through/readme.md).
- [In-Depth Understanding of verl Source Code (Rollout)](./rlhf/verl/multi-turn/code-walk-through/readme-2-EN.md): Also available on [Zhihu](https://zhuanlan.zhihu.com/p/1923349757566388159) and in [Chinese version](./rlhf/verl/multi-turn/code-walk-through/readme-2.md).
- [Pending Review] [In-Depth Understanding of verl Source Code (Make Experience)](./rlhf/verl/multi-turn/code-walk-through/readme-3.md): Analysis of the logic for the make experience part in verl.
- [AgentLoop Source Code Analysis](./rlhf/verl/multi-turn/code-walk-through/readme-6.md): Analysis of the multi-turn RL implementation based on AgentLoop in verl.
- [verl Parameter Quick Reference](./rlhf/verl/multi-turn/code-walk-through/readme-5-EN.md): Quick reference for verl parameters. Also available on [Zhihu](https://zhuanlan.zhihu.com/p/1925041836998783250) and in [Chinese version](./rlhf/verl/multi-turn/code-walk-through/readme-5.md).
- [Analyzing the Complexity of Agentic Multi-Turn Training from a Tokenizer Perspective](./rlhf/verl/multi-turn/fast_tokenization/multiturn_tokenization_and_masking.md): Also available on [Zhihu](https://zhuanlan.zhihu.com/p/1917126584806139373) and in [Chinese version](./rlhf/verl/multi-turn/fast_tokenization/multiturn_tokenization_and_masking_ZH.md).
- [Pending Review] [DAPO Dynamic Filtering Implementation and Batch Size Analysis](./rlhf/verl/multi-turn/code-walk-through/dapo.md): Exploring how to achieve higher parallelism by padding prompts to a smaller batch size.
- [Systematic Analysis of Time Consumption in verl Multi-Turn Training](./rlhf/verl/multi-turn/tool_examples/profile_en.md): verl multi-turn interaction and tool call profile analysis. Also available in [Chinese version](./rlhf/verl/multi-turn/tool_examples/profile.md) and on [Zhihu](https://zhuanlan.zhihu.com/p/1929748460212552414).
- [SGLang, verl, OpenBMB, and Tsinghua University Team Jointly Open Source: First Support for Multi-Turn Interaction and Tool Calling in Mainstream RLHF Frameworks](./rlhf/verl/multi-turn/release_log/verl-multiturn-rollout-Release_ZH.md): First support for multi-turn interaction and tool calling in mainstream RLHF frameworks. Also available on [Zhihu](https://zhuanlan.zhihu.com/p/1906007821889283171).
- [Search-R1 & veRL-SGLang: Train LLMs with Multi-Turn RL to Reason and Call a Search Engine](./rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like_ZH.md): Integrating the Search-R1 framework into the verl-sglang ecosystem. Also available on [Zhihu](https://zhuanlan.zhihu.com/p/1912156329751081620).
- [SGLang-veRL Server: From Engine to Server, We Need More Flexible RLHF Rollout Interfaces](./rlhf/verl/server-based/veRL-server-based-rollout.md): To implement more complex RLHF systems, we are gradually replacing the rollout engine in veRL with a rollout server. Also available on [Zhihu: SGLang-veRL Server](https://zhuanlan.zhihu.com/p/1890631652486665464).
- [HybridFlow veRL Original Paper Analysis](./rlhf/verl/readme.md): Principles and implementation of SGLang's hybrid engine. Also available on [Zhihu: HybridFlow veRL Original Paper Analysis](https://zhuanlan.zhihu.com/p/24682036412).

### OpenRLHF Framework

- [Illustrated Series on LLM RLHF: PPO Principles and Source Code Interpretation for Everyone](https://zhuanlan.zhihu.com/p/677607581) and [Illustrated Distributed Training Process based on Ray in OpenRLHF](https://zhuanlan.zhihu.com/p/12871616401): Excellent RLHF introductory resources by Ms. Mengyuan. After reading, you will have a good understanding of RLHF's computational flow and the OpenRLHF PPO framework. I have also added my own understanding in [RLHF Computational Flow](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/tree/main/rlhf/OpenRLHF#rlhf-%E7%9A%84%E8%AE%A1%E7%AE%97%E6%B5%81).
- [Brief Analysis of the Computational Flow of Post-Training Systems Represented by OpenRLHF](./rlhf/OpenRLHF/readme.md): Further complement to Ms. Mengyuan's article. The Github native rendering is terrible; you might as well look at [Zhihu](https://zhuanlan.zhihu.com/p/16370000391).

### System Design and Optimization

- [Deep Thoughts on RL Systems: In-Depth Understanding of Weight Update Mechanism](./rlhf/sys-design/readme-1-EN.md): Summary of half a year's work, in-depth understanding of the weight update mechanism. Also available on [Zhihu](https://zhuanlan.zhihu.com/p/1925210722704531547) and in [Chinese version](./rlhf/sys-design/readme-1.md).
- [Deep Thoughts on RL Systems: FSDP Training Backend](./rlhf/sys-design/readme-2-en.md): Discusses the principles and implementation of FSDP, and analyzes verl's use of FSDP. Also available on [Zhihu](https://zhuanlan.zhihu.com/p/1929115059113693341) and in [Chinese version](./rlhf/sys-design/readme-2.md).
- [Pending Review] [Deep Thoughts on RL Systems: Megatron](./rlhf/sys-design/readme-3.md): Brief analysis of Megatron's basic features, focusing on its use in the RL framework.
- [Extending the OpenRLHF Inference Engine](./rlhf/OpenRLHF/develop-log.md): Development notes on integrating SGLang into OpenRLHF. The entire process was very painful, and there's still an nccl hang error that a DeepSpeed core contributor is currently fixing.
- [Pending Review] [SGLang as rollout engine of GRPO trainer](./rlhf/GRPO/SGLang_GRPO.md): Introduction on how to use SGLang as the inference backend for the GRPO Trainer in TRL. GRPO is a PPO variant that optimizes PPO's memory usage while improving mathematical reasoning capabilities.

### Algorithms and Theory

- [Pending Review] [Learning to Reason under Off-Policy Guidance](./rlhf/partial-rollout/Learning_to_Reason_under_Off-Policy_Guidance.md): The LUFFY framework uses off-policy assistance for on-policy learning, dynamically balancing imitation and exploration by combining off-policy inference trajectories with on-policy rollouts.
- [Kimi K1.5: Successful Practice of Long Context RL](./rlhf/partial-rollout/readme.md): Industrial implementation of Long Context RLHF. I have always liked the technical reports from the Kimi team. Also available on [Zhihu: Kimi K1.5: Successful Practice of Long Context RL](https://zhuanlan.zhihu.com/p/1894282607325344277).
- [Rule-based Reward](https://zhuanlan.zhihu.com/p/13211508979): Only on Zhihu, a brief write-up. Honestly, I didn't particularly like the original paper, but determined reward is indeed charming.
- [SWE-Bench: How to Construct an Excellent Benchmark in the LLM Era](https://zhuanlan.zhihu.com/p/16292266518): Reading notes on the SWE-Bench paper. How to construct a good benchmark to provide fine-grained reward for post-training is an eternal and beautiful topic.
- [Brief Analysis of Mainstream Alignment Algorithms and the NeMo-Aligner Framework](https://zhuanlan.zhihu.com/p/5220718268)


## SGLang Learning Notes

### SGLang Diffusion Learning Notes

- [SGLang Diffusion Code Walk Through](./sglang/code-walk-through/sgl_diffusion_en.md): Basic principles of the diffusion model, and the entire process of a request being handled by SGLang-Diffusion. Also available on [Zhihu](https://zhuanlan.zhihu.com/p/1982441236066480797) and in [Chinese version](./sglang/code-walk-through/sgl_diffusion.md).

### Core Architecture and Optimization

- [SGLang Code Walk Through](./sglang/code-walk-through/readme.md): The entire process of a request being handled by the SGLang Engine. Some parts are unfinished, but most are okay and have served as a starting point for many SGLang beginners. [Chinese version is here](./sglang/code-walk-through/readme-CN.md).
- [Walk Through SGLang / VLLM Worker](./sglang/sglang-worker/readme.md): Incomplete analysis of SGLang code. Also available on [Walk Through SGLang / VLLM Worker](https://zhuanlan.zhihu.com/p/6363614076). We also thoughtfully provide an [English version](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/sglang-worker/readme.md). For a more detailed analysis, refer to [SGLang Code Walk Through](./sglang/code-walk-through/readme.md); this one is just supplementary.
- [Walk Through SGLang Scheduler](./sglang/sglang-scheduler/readme-CN.md)
- [Pending Review] [SGLang Scheduler Evolution](./sglang/scheduler-evolution/SGLang%20Scheduler%20技术变迁.md): Detailed introduction to the technical evolution of the SGLang Scheduler from serial to CPU/GPU overlap, and related components, comparing the previous overlap Scheduler with the current one introducing multiple CUDA streams and FutureMap. Can be viewed on [Zhihu article](https://zhuanlan.zhihu.com/p/1969077475129688722).
- [Pending Review] [KV Cache Code Walkthrough](./sglang/kvcache-code-walk-through/readme.md): Overview of KV cache management implementation, starting from the Scheduler component, detailing the update process of KV cache and memory pool during prefill and decode stages.
- [Pending Review] [SGLang Multimodal Request Lifecycle: A Deep Architectural Analysis with Qwen2.5-VL as an Example](./sglang/code-walk-through/multimodal_request_lifecycle.md): Provides a detailed analysis of the multimodal request processing flow within the SGLang framework, using Qwen2.5-VL as a reference model.
- [Pending Review] [How A Model is Loaded in Hugging Face and SGLang](./sglang/how-model-is-loaded/readme.md): Documents the process of loading models in Hugging Face and SGLang to help understand the weight loading mechanism.
- [Pending Review] [Speculative Decoding](./sglang/speculative-decoding/speculative-decoding.md): Introduces the speculative decoding optimization technique, which uses a smaller draft model to predict the next $K$ tokens, achieving up to $K$-fold acceleration.
- [Pending Review] [Zero-Overhead Batch Scheduler](./sglang/zero-overhead-scheduler/zero-overhead-batch-scheduler.md): Introduces the zero-overhead batch scheduler, which solves the GPU Bubble problem caused by serial execution of CPU scheduling and GPU computation in traditional inference systems.
- [Pending Review] [Data Parallelism Attention](./sglang/dp-attention/readme.md): Detailed introduction to the principles and implementation of DP Attention, specifically for models like DeepSeek that use MLA and only have one KV head, to avoid KV cache duplication caused by tensor parallelism.
- [Brief Analysis of SGLang Framework's Quantization Design and Ideas](./sglang/quantization/quantization_architecture_en.md): Also available on [Zhihu: Brief Analysis of SGLang Framework's Quantization Design and Ideas](https://zhuanlan.zhihu.com/p/1971183020338832111) and in [Chinese version](./sglang/quantization/quantization_architecture.md).
- [Constraint Decoding: Concepts, Methods, and Optimization](./sglang/constraint-decoding/readme.md): Also available on [Zhihu: Understanding Constraint Decoding: Concepts, Methods, and Optimization in one article](https://zhuanlan.zhihu.com/p/18336995950).
- [Pending Review] [Online Update Weights](./sglang/online-update-weights/readme.md): Introduction to the implementation of the `online_update_weights` interface in SGLang. Unlike `update_weights` which reads weights from the disk, this interface broadcasts new weights directly from the training engine via NCCL.
- [Pending Review] [SGLang Verl Engine Optimization Analysis](./sglang/sglang-verl-engine/readme.md): Analysis of optimizations in the SGLang verl engine, including the implementation of interfaces like `update_weights_from_tensor`.
- [Latency Accelerate For Weight Updates](./sglang/latency-accelerate-for-weight-updates/readme-CN.md)
- **[🔥 Related Debugging] [Analyzing VLM RL Training Memory Leaks via Torch Memory Snapshot](./torch/mem-snapshot/readme-en.md)**: Analysis of SGLang memory leak issues and solutions. Also available on [Zhihu](https://zhuanlan.zhihu.com/p/1943202817247519535) and in [Chinese version](./torch/mem-snapshot/readme.md).

### Usage and Practice

- [Pending Review] [Qwen3-Coder Usage](./sglang/qwen/coder.md): Introduction to using Qwen3-coder in SGLang, including the use of tool-parser.
- [Pending Review] [NVIDIA Dynamo](./sglang/nvidia-dynamo/dynamo.md): Introduction to NVIDIA Dynamo, a high-throughput, low-latency inference framework designed for generative AI and inference model serving in multi-node distributed environments.
- [Viewing HuggingFace Model Structure](https://zhuanlan.zhihu.com/p/9912733791)
- [SGLang Backend Original Paper Analysis](https://zhuanlan.zhihu.com/p/716543182)
- [Brief Analysis of the Status Quo of Reward / Embed Model Server Engine](https://zhuanlan.zhihu.com/p/4148050391)
- [Newbie Perspective: Experience and Gains from Migrating vllm to SGLang](https://zhuanlan.zhihu.com/p/714833359)
- [Newbie Perspective: Using SGL to Serve Embedding Model](https://zhuanlan.zhihu.com/p/715805386)
- [Newbie Perspective: Using vllm to serve a new Embedding Model](https://zhuanlan.zhihu.com/p/715857723)

## Scheduling and Routing

- [Mooncake: Carrying the P/D Separation to the End](https://zhuanlan.zhihu.com/p/1711346141)
- [Should Prefill and Decode be Separated onto Different Cards?](https://zhuanlan.zhihu.com/p/1280567902)
- [Understanding Prefill and Decode Computation Characteristics Based on Chunked Prefill](https://zhuanlan.zhihu.com/p/718715866)
- [ModelServer: A Frontend Distribution System Based on SGLang](https://zhuanlan.zhihu.com/p/718015016)

## ML System Fundamentals

### Transformers & Model Architecture

- [Pending Review] [Cross-Attention Mechanism in Transformer](./transformers/attention/cross_attention_en.md): Introduction to the cross-attention mechanism in Transformers, allowing the decoder to access and use relevant information from the encoder. Also available in [Chinese version](./transformers/attention/cross_attention.md).
- [Understanding Special Tokens and Chat Templates in One Article](./transformers/special_tokens/special_tokens.md): Also recorded on Zhihu [Understanding Special Tokens and Chat Templates in One Article](https://zhuanlan.zhihu.com/p/17052593700).

### CUDA & GPU

- [Brief Analysis of CUDA Graph Based on torch-memory-savor](./torch/cuda-graph/readme_en.md): Also available on [Zhihu: Brief Analysis of CUDA Graph Based on torch-memory-savor](https://zhuanlan.zhihu.com/p/1921726788574360686) and in [Chinese version](./torch/cuda-graph/readme.md).

### Distributed Training & Communication

- [Pending Review] [Implementing Tensor Parallelism From Scratch](./torch/tensor-parallelism/readme.md): Implementation and practice of Tensor Parallelism.
- [NCCL and NVIDIA TOPO](./torch/nccl/readme.md): Introduction to NCCL and NVIDIA GPU detection. Also available on [NCCL and NVIDIA TOPO](https://zhuanlan.zhihu.com/p/6160835906).
- [NCCL and SGLang](./torch/nccl/readme_en.md): Application of NCCL in SGLang. This is very similar to the Chinese content but includes some additional notes on parallel strategies. I probably won't complete this note and will write a separate one to record parallel strategies.
- [PyTorch Distributed](./torch/torch-distributed/readme.md): Communication practice with `torch.distributed`, details on GIL and `all_reduce`. This part is also available on [Zhihu: PyTorch Communication Practice](https://zhuanlan.zhihu.com/p/5853094319).
- [[Original][In-Depth][PyTorch] DDP Series Part 1: Introductory Tutorial](https://zhuanlan.zhihu.com/p/178402798): Although I didn't fully grasp the DDP content, I used this to learn about GIL and ring all reduce. This step is recorded in the [Postscript of torch-distributed](./torch/torch-distributed/readme.md#gil).
- [Detailed Explanation of nvidia-smi Command and Some Advanced Tips](https://www.yourmetaverse.cn/deep_learning/199/): Mainly about network topology; my local results are recorded in the [NCCL section](./torch/nccl/readme.md#nvlink-查询).

### Quantization

- [Give me BF16 or Give Me Death: Comprehensive Evaluation of Current Quantization Methods](https://zhuanlan.zhihu.com/p/5485556270)
- [AWQ: Model Quantization Should Focus on Activation Values](https://zhuanlan.zhihu.com/p/942485319)

## Developer Guide

- [How to use docker](./engineer/how-to-use-docker/readme_en.md): How to use Docker to manage development environments. Please note that to collectively foster a good research environment and prevent others from being annoyed by the baseline "it runs on my machine," learning Docker is essential for everyone. We also have a [Chinese version](./engineer/how-to-use-docker/readme.md) and [Zhihu](https://zhuanlan.zhihu.com/p/1916764175230801287).
- [Setting up a Clean Development Environment](./engineer/uv/readme.md): Setting up a clean development environment. Also available on [Zhihu: Setting up a Clean Development Environment](https://zhuanlan.zhihu.com/p/23440683394).
- [Compiling and Deploying Jupyter Notebooks as Documentation on CI](https://zhuanlan.zhihu.com/p/2382351079)