## 怎么给 verl bump sglang 的新版本
> 这篇文章主要讲诉我把sglang 给verl bump 版本从0.4.9 到0.4.10post2 的一些过程和经验, 每次版本更新遇到和解决的问题并不完全相同, 供参考

以前的PR: [bump sglang 到 0.4.9](https://github.com/volcengine/verl/pull/2720)

### 操作流程
1. 先查看sglang 新版本对应的 torch 版本等 [link](https://github.com/sgl-project/sglang/blob/v0.4.10.post2/python/pyproject.toml#L58)
2. 在相应的文件夹下新建 docker file
例如 `docker/verl0.5-cu126-torch2.7-fa2.7.4/Dockerfile.app.sglang0.4.10.post2.mcore0.12` 注意transformer版本,torch版本和sglang 版本
```bash
# Start from the verl base image
# Dockerfile.base
FROM verlai/verl:base-verl0.5-cu126-cudnn9.8-torch2.7.1-fa2.7.4

# Define environments
ENV MAX_JOBS=8
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV DEBIAN_FRONTEND=noninteractive
ENV NODE_OPTIONS=""
ENV PIP_ROOT_USER_ACTION=ignore
ENV HF_HUB_ENABLE_HF_TRANSFER="1"

# Install sglang-0.4.10
# Install FlashInfer Python package
RUN pip install --upgrade pip setuptools packaging
RUN pip install --resume-retries 999 --no-cache-dir --no-build-isolation flashinfer-python==0.2.9rc1
RUN pip install --resume-retries 999  --no-cache-dir --no-build-isolation "sglang[all]==0.4.10.post2"

# Fix packages
RUN pip install --no-cache-dir "tensordict==0.6.2" "transformers[hf_xet]==4.54.1" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=19.0.1" pandas \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler blobfile xgrammar \
    pytest py-spy pyext pre-commit ruff

RUN pip uninstall -y pynvml nvidia-ml-py && \
    pip install --resume-retries 999 --no-cache-dir --upgrade "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"

RUN pip install --resume-retries 999 --no-cache-dir nvidia-cudnn-cu12==9.8.0.87

# Install TransformerEngine
RUN export NVTE_FRAMEWORK=pytorch && pip3 install --resume-retries 999 --no-deps --no-cache-dir --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@v2.2.1

# Install Megatron-LM
RUN pip3 install --no-deps --no-cache-dir --no-build-isolation git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.2

# Install mbridge
RUN pip3 install --no-cache-dir mbridge
```
3. 用这个 docker file build image
注意不能在一个docker 容器内build, Docker 相关的操作在ubuntu(裸机) 内完成:
3.1 先登录自己的docker
```bash
docker login -u [your_username]
```
3.2 再build 自己的image
```bash
docker build -t popsodazhp/verl:app-verl0.5-sglang0.4.10.post2-mcore0.12.2-te2.2 -f docker/verl0.5-cu126-torch2.7-fa2.7.4/Dockerfile.app.sglang0.4.10.post2.mcore0.12 .
```

4. 接下来尝试 push 上去
```bash
docker push popsodazhp/verl:app-verl0.5-sglang0.4.10.post2-mcore0.12.2-te2.2
```
5. 修改CI配置文件
如这个[commit](https://github.com/volcengine/verl/pull/3183/commits/8ebbea9a0206ce4e258b1cde2eb025b99139b13b)
修改CI中的image为上面 build 的这个, 观察github 上的CI pass 情况, 如果基本都pass了, 说明这个新的版本build 的image基本没什么问题
6. 和verl 的人讨论协商, 让他们build字节官方的image, 讨论细节~

