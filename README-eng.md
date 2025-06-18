# Awesome-ML-SYS-Tutorial 
## [English version](./README-eng.md) | [简体中文](./README.md)

My learning notes/codes for ML SYS.  English version is under development and only available for some texts.

## RLHF System Development Notes

- **Intro to HybridFlow/veRL**  
  [English TODO] | [[中文版](./rlhf/verl/readme.md)]：SGLang's hybrid RLHF engine design and implementation.

- **Extending OpenRLHF's Inference Engine**  
  [English TODO] | [[中文版](./rlhf/OpenRLHF/develop-log.md)]：Notes on integrating SGLang with OpenRLHF, an exhausting process with frequent NCCL hang bugs.

- **SWE-Bench: How to Construct a Great Benchmark for the LLM Era**  
  [[中文版](https://zhuanlan.zhihu.com/p/16292266518)] 

- **Intro to Workflow in OpenRLHF-like Post-Training Systems**  
  [[中文版](./rlhf/OpenRLHF/readme.md)] 

- **The Illustrated PPO: Theory and Source Code Explanation**  
  [[中文版](https://zhuanlan.zhihu.com/p/677607581)]  
  Also see [RLHF 的计算流](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/tree/main/rlhf/OpenRLHF#rlhf-%E7%9A%84%E8%AE%A1%E7%AE%97%E6%B5%81).

- **Latency Optimization for Weight Updates**  
  [English TODO] | [[中文版](./sglang/latency-accelerte-for-weight-updates/readme.md)]：An experience of debugging loading efficiency.

- **Intro to Alignment Algorithms and NeMo-Aligner Framework**  
  [[中文版](https://zhuanlan.zhihu.com/p/5220718268)] 

---

## SGLang Learning Notes

- **Concepts and Optimization of Constraint Decoding**  
  [English TODO] | [[中文版](./sglang/constraint-decoding/readme.md)] 

- **SGLang Code Walkthrough**  
  [[English version](./sglang/code-walk-through/readme.md)]：The lifecycle of a request in the SGLang Engine, a good start for SGLang beginners.

- **Walk Through SGLang / VLLM Worker**  
  [[English version](./sglang/sglang-worker/readme.md)]：Demystifying the SGLang worker (model executor).

- **Walk Through SGLang Scheduler**
  [[English version](./sglang/sglang-scheduler/readme.md)] | [[中文版](./sglang/sglang-scheduler/readme-CN.md)] : Demystifying the SGLang Scheduler.

- **Latency Accelerate For Weight Updates**
  [[English version](./sglang/latency-accelerte-for-weight-updates/readme.md)] | [[中文版](./sglang/latency-accelerte-for-weight-updates/readme-CN.md)] : A detailed debugging investigation of weight update latency in a distributed system.

- **Reward / Embed Model Server Engine**  
  [English TODO] | [[中文版](https://zhuanlan.zhihu.com/p/4148050391)] 

- **SGLang Backend Analysis**  
  [English TODO] | [[中文版](https://zhuanlan.zhihu.com/p/716543182)] 

- **Using vLLM to Serve New Embedding Models**  
  [English TODO] | [[中文版](https://zhuanlan.zhihu.com/p/715857723)] 

- **Using SGL to Serve Embedding Models**  
  [English TODO] | [[中文版](https://zhuanlan.zhihu.com/p/715805386)] 

- **From vLLM to SGLang: A User's Perspective**  
  [English TODO] | [[中文版](https://zhuanlan.zhihu.com/p/714833359)] 

## Scheduling and Routing

- **Mooncake: Maximizing PD Disaggregation**  
  [[中文版](https://zhuanlan.zhihu.com/p/1711346141)]：Taking prefill and decode separation to the extreme.

- **Should Prefill and Decode Be Separated onto Different Cards?**  
  [[中文版](https://zhuanlan.zhihu.com/p/1280567902)]：A discussion on separating prefill and decode tasks.

- **Understanding Prefill and Decode Computational Characteristics Based on Chunked Prefill**  
  [[中文版](https://zhuanlan.zhihu.com/p/718715866)]：Analyzing computational characteristics using chunked prefill.

- **ModelServer: A Frontend Distribution System Based on SGLang**  
  [[中文版](https://zhuanlan.zhihu.com/p/718015016)]：A frontend distribution system built on SGLang.

---

## ML System Fundamentals

- **NCCL and NVIDIA TOPO**  
  [English](./distributed/nccl/readme_en.md) | [[中文版](./distributed/nccl/readme.md)]：An introduction to NCCL and NVIDIA topology.

- **PyTorch Distributed**  
   [English TODO] | [[中文版](./distributed/torch-distributed/readme.md)]：Practical communication in `torch.distributed`.

- **Give Me BF16 or Give Me Death: A Comprehensive Evaluation of Current Quantization Methods**  
  [[中文版](https://zhuanlan.zhihu.com/p/5485556270)]：A detailed evaluation of current quantization methods.

- **AWQ: Model Quantization Should Focus on Activation Values**  
  [[中文版](https://zhuanlan.zhihu.com/p/942485319)]：Why activation values should be the focus of model quantization.

- **Deep Dive into PyTorch DDP Series Part 1: Beginner's Tutorial**  
  [[中文版](https://zhuanlan.zhihu.com/p/178402798)]：A beginner's guide to PyTorch Distributed Data Parallel (DDP).

- **Detailed Explanation of nvidia-smi Command and Some Advanced Techniques**  
   [[中文版](https://www.yourmetaverse.cn/deep_learning/199/)]：Advanced techniques for using `nvidia-smi`.

---

## Other

- **Setting Up a Clean Development Environment**  
  [English TODO] | [[中文版](./engineer/uv/readme.md)]：How to set up a clean and efficient development environment.

- **Understanding Special Tokens and Chat Templates**  
  [English TODO] | [[中文版](./transformers/special_tokens/special_tokens.md)]：A guide to understanding special tokens and chat templates.

- **Compiling Jupyter Notebooks on CI and Deploying as Documentation**  
  [[中文版](https://zhuanlan.zhihu.com/p/2382351079)]：A guide on compiling Jupyter notebooks in CI and deploying them as documentation.