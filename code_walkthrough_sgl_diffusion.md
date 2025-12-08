# SGLang Diffusion Code Walk Through

本文旨在为开发者提供一份 SGLang `multimodal_gen` 后端的代码导读，梳理请求的处理路径，展示一个文生图/视频请求的处理流程。

### 扩散模型

SGLang-Diffusion 支持 diffusion 的高效推理, diffusion models 是最近几年快速发展的，也是最流行的图像/视频生成框架。作为代码导读教程，这里无需介绍复杂的数学公式和原理，

总的来说，diffusion models 定义了一个前向过程：数据 -> 高斯噪声。从这个前向过程可以推导出逆向过程（噪声 -> 数据，从噪声中重建样本），让模型学习这个过程，并利用训练好的模型执行这个逆向过程，也就是推理框架需要做的事情

从理解逆向过程的方式，可以把 diffusion models 简单分为三类：

1. Variational Perspective (变分视角)：将逆向过程建模为马尔可夫链，通过优化变分下界 (ELBO) 训练模型预测每一步的高斯噪声。
2. Score-based Perspective (基于得分的视角)：将扩散建模为随机微分方程 (SDE)，模型学习数据分布的得分函数（梯度方向），指引噪声向高密度数据区域移动。
3. Flow-based Perspective (基于流的视角)：建立噪声与数据间的确定性传输路径 (Flow)，模型学习连接两者的向量场 (Velocity Field)，通过求解 ODE 将噪声平滑映射为数据。

这三种建模方式，以不同的角度理解逆向过程，但在推理框架内的呈现方式都类似：

- Scheduler 决定时间步 (Timesteps) 和噪声强度 (Sigma)。它们位于 runtime/models/schedulers/ 目录下
- Model (DiT/UNet) 负责预测每一步的噪声。它们位于 runtime/models/dits/ 目录下
- Sampler/Solver 更新样本状态。它们也 位于 runtime/models/schedulers/ 目录下的 Scheduler 类内部，通过 Scheduler 的 step 方法实现


### 一个请求在 SGLang-Diffusion 的前世今生

在 `multimodal_gen` 中，请求的处理流程大致如下：

1. **Server Initialize**: 用户启动 Server，每一个 rank 上启动 Scheduler 和 GPUWorker。GPUWorker 也会新建 `ComposedPipeline` 对象
1.  **Send Request**: 请求以某种方式被发送至 Scheduler。在 offline generate 模式下，是使用客户端 `DiffusionGenerator` 发送的
2.  **Scheduler Process**: `Scheduler` 运行一个 Infinite Event Loop，通过 ZeroMQ 接收请求，并根据请求类型，调用不同的 handler。对于生成任务 `Req`，会把它发送给所有 GPUWorker
3.  **GPUWorker Process**: `GPUWorker` 调用加载好的 `ComposedPipeline` 的 forward 方法，其内部通过提前构造好的 `PipelineExecutor` 执行各个 `PipelineStage`（如编码、去噪、解码）。对于当前版本，默认使用 SyncExecutor，它会线性依次调用所有 Stages
5.  **Stages Execution**: 各个 Stage（Text Encoding, Denoising, VAE Decoding）在 GPU 上执行
6.  **Response**: Main rank 上的 Scheduler 得到生成的 pixel values，返回给客户端的 `DiffusionGenerator`
7.  **PostProcess**: `DiffusionGenerator` 解码数据，得到最终的 图像/视频



### NOTES

1. 本代码导读基于 SGLang-diffusion 版本 (35a9a073706e89a2f5740f578bbb080146cd48bf)
2. 本代码导读灵感来源于 https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/code-walk-through/readme.md, 它是我接触，了解，并熟悉 SGLang 的指南


### References

1. [The Principles of Diffusion Models](https://arxiv.org/abs/2510.21890)
2. [What are Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
3. [Score-based generative modeling with multiple noise perturbations](https://yang-song.net/blog/2021/score/)
