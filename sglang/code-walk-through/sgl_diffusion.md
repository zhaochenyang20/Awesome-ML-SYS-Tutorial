# SGLang Diffusion Code Walk Through

本文旨在为开发者提供一份 SGLang Diffusion (`multimodal_gen`) 后端的代码导读。

## 扩散模型

SGLang-Diffusion 支持 diffusion 的高效推理。diffusion models 是最近几年发展最快的，也是最流行的图像和视频的生成框架。

总的来说，diffusion models 定义了一个前向过程：数据 -> 高斯噪声，从这个前向过程可以推导出逆向过程：噪声 -> 数据，也即从噪声中重建样本。利用训练好的模型执行这个逆向过程就是 SGLang-Diffusion 需要做的事情。

作为代码导读教程，这里不会介绍复杂的数学公式和原理。这里依据模型建模逆向过程的方式，大致把 diffusion models 分为三类：

1. Variational Perspective (变分视角, DDPM)：建模为马尔可夫链，训练模型学习前后两步 $x_{t}$ 的条件概率，预测当前时刻叠加在数据上的噪声
2. Score-based Perspective (基于得分的视角, SGM)：建模为随机微分方程 (SDE)，训练模型学习数据的得分函数（梯度方向），指引噪声向高密度数据（真实数据）区域移动
3. Flow-based Perspective (基于流的视角, Rectified Flow)：建模为噪声点与数据点之间的确定性传输路径 (Flow)，训练模型学习连接两者的速度场 (Velocity Field)，通过求解 ODE(常微分方程) 将噪声平滑转化为数据

这三种建模方式，以不同的角度理解逆向过程，但在推理框架内的呈现方式都类似。它们主要区别于 denoise 阶段，都主要由以下组件负责：


- Model (DiT/UNet) 负责预测每一步的噪声。它们位于 `runtime/models/dits/` 目录下
- Scheduler 决定时间步 (Timesteps) 和噪声强度 (Sigma)。它们位于 `runtime/models/schedulers/` 目录下
- Sampler/Solver 更新样本状态，不同 diffusion models 的类别会对应不同的 Solver。它们也位于 `runtime/models/schedulers/` 目录下的 Scheduler 类内部，通过 Scheduler 的 step 方法实现

## 一个请求在 SGLang-Diffusion 的生命周期

SGLang Diffusion 的设计尽量和 SGLang 保持一致，方便开发者理解和熟悉各种概念。在 SGLang Diffusion 中，请求的生命周期大致如下：

1. **Server Initialize**（仅作用于 offline generate 模式, online 模式下会提前初始化）: 每个 rank 上启动 `Scheduler` 和 `GPUWorker`。`GPUWorker` 初始化时会构建 `ComposedPipeline` 对象，其中会涉及 Pipeline Components 的加载
2. **Send Request**: 请求通过客户端（如 `DiffusionGenerator` 或 HTTP API）发送至 Scheduler。
3. **Scheduler Process**: Rank 0 的 `Scheduler` 通过 ZeroMQ 接收请求，并将其广播 (Broadcast) 给所有 Rank。所有 Rank 上的 Scheduler 收到请求后，调用本地 `GPUWorker` 执行任务。
4. **GPUWorker Process**: `GPUWorker` 调用 `ComposedPipeline` 的 `forward` 方法。Pipeline 内部通过 `PipelineExecutor`（默认使用 `SyncExecutor` 的实现）按顺序调度各个 `PipelineStage`。
5. **Stages Execution**: 各个 Stage（如 Text Encoding, Denoising, VAE Decoding）在 GPU 上执行。
6. **Response**: 所有 Rank 完成计算后，Rank 0 的 `Scheduler` 收集生成的 tensor 数据（Pixel Values），并返回给客户端。
7. **PostProcess**: 客户端（如 `DiffusionGenerator`）接收 tensor 数据，进行后处理（如格式转换、保存文件），最终得到图像/视频。

<div align="center">
<img alt="sgl-diffusion" src="./diffusion-ark.png" width="50%" />
</div>



> [!NOTE]
> 1. 本代码导读基于 SGLang-diffusion 版本 (`35a9a073706e89a2f5740f578bbb080146cd48bf`)
> 2. 本代码导读灵感来源于 [SGLang Code Walk Through](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/code-walk-through/readme.md)


## References

1. [The Principles of Diffusion Models](https://arxiv.org/abs/2510.21890)
2. [What are Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models)
3. [Score-based generative modeling with multiple noise perturbations](https://yang-song.net/blog/2021/score)
