# SGLang Diffusion Code Walk Through

This document aims to provide developers with a code walk-through for the SGLang Diffusion (`multimodal_gen`) backend.

## Diffusion Models

SGLang-Diffusion supports efficient inference for diffusion models. Diffusion models are one of the fastest-developing and most popular generative frameworks for images and videos in recent years.

Broadly, diffusion models define a **forward process**: Data -> Gaussian Noise. From this forward process, the **reverse process** can be derived: Noise -> Data, which is the reconstruction of a sample from noise. Executing this reverse process using a trained model is what SGLang-Diffusion needs to do. 

As a code walk-through tutorial, we will not delve into complex mathematical formulas and principles here. Based on how models conceptualize the reverse process, diffusion models are generally divided into three categories:

1.  **Variational Perspective (DDPM)**: Modeled as a Markov Chain. The model is trained to learn the conditional probability of two adjacent steps $x_{t}$, predicting the Gaussian noise.
2.  **Score-based Perspective (SGM)**: Modeled as a Stochastic Differential Equation (SDE). The model is trained to learn the data's score function (gradient direction), guiding the noise towards high-density (real data) regions.
3.  **Flow-based Perspective (Rectified Flow)**: Modeled as a deterministic transport path (Flow) between a noise point and a data point. The model is trained to learn the **Velocity Field** connecting the two, smoothly transforming noise into data by solving an Ordinary Differential Equation (ODE).

These three modeling approaches interpret the reverse process from different angles, but their presentation within the inference framework is similar. They primarily differ in the denoise stage and are mainly handled by the following components:

* **Model (DiT/UNet)**: Responsible for predicting the noise at each step. These are located in the `runtime/models/dits/` directory.
* **Scheduler**: Determines the **Timesteps** and **Sigma** (noise strength). These are located in the `runtime/models/schedulers/` directory.
* **Sampler/Solver**: Updates the sample state. Different categories of diffusion models correspond to different Solvers. They are also implemented within the Scheduler classes located in the `runtime/models/schedulers/` directory, via the Scheduler's `step` method.

## Life Cycle of a Request in SGLang-Diffusion


The design of SGLang Diffusion aims to remain consistent with SGLang to facilitate developer understanding and familiarity with various concepts. The life cycle of a request in SGLang Diffusion is approximately as follows:

1.  **Server Initialize** (Only for offline generate mode; initialized earlier in online mode): The `Scheduler` and `GPUWorker` are launched on each rank. During initialization, `GPUWorker` builds the `ComposedPipeline` object, which involves loading Pipeline Components.
2.  **Send Request**: A request is sent to the Scheduler via a client (e.g., `DiffusionGenerator` or HTTP API).
3.  **Scheduler Process**: The `Scheduler` on Rank 0 receives the request via ZeroMQ and **Broadcasts** it to all ranks. Schedulers on all ranks receive the request and call their local `GPUWorker` to execute the task.
4.  **GPUWorker Process**: The `GPUWorker` calls the `forward` method of `ComposedPipeline`. Internally, the Pipeline sequentially schedules each `PipelineStage` via a `PipelineExecutor` (defaulting to the `SyncExecutor` implementation).
5.  **Stages Execution**: Each Stage (e.g., Text Encoding, Denoising, VAE Decoding) is executed on the GPU.
6.  **Response**: After all ranks complete the computation, the `Scheduler` on Rank 0 collects the generated tensor data (**Pixel Values**) and returns it to the client.
7.  **PostProcess**: The client (e.g., `DiffusionGenerator`) receives the tensor data and performs post-processing (e.g., format conversion, saving files) to finally obtain the image/video.

<div align="center">
<img alt="sgl-diffusion" src="./diffusion-ark.png" width="50%" />
</div>

> [!NOTE]
> 1. This code walk-through is based on the SGLang-diffusion version (`35a9a073706e89a2f5740f578bbb080146cd48bf`)
> 2. This code walk-through is inspired by the [SGLang Code Walk Through](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/code-walk-through/readme.md)

## References

1. [The Principles of Diffusion Models](https://arxiv.org/abs/2510.21890)
2. [What are Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models)
3. [Score-based generative modeling with multiple noise perturbations](https://yang-song.net/blog/2021/score)
