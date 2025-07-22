# Qwen3-Coder Usage
[Qwen3-coder](https://github.com/QwenLM/Qwen3) is the largest language model developed by Qwen team, Alibaba Cloud.

SGLang has supported Qwen3-coder already. The `tool-parser` shall be used with SGLang >= 0.4.9.post3.

```bash
pip install --upgrade pip
pip install uv
uv pip install "sglang[all]>=0.4.9.post3"
```

Ongoing optimizations are tracked in the Roadmap. 

# Launch Qwen3-Coder with SGLang

To serve Qwen3 model on H200 GPUs:

For BF16 model

```
python3 -m sglang.launch_server --model-path Qwen/Qwen3-Coder-480B-A35B-Instruct --tp 8 --tool-call-parser qwen3 --enable-ep-moe
```

For FP8 model

```
python3 -m sglang.launch_server --model-path Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 --tp 8 --tool-call-parser qwen3 --enable-ep-moe
```


## Configuration Tips
* **FP8 models** : With `--tp` Loading failure is expected; switch to expert-parallel mode using ```--enable-ep-moe```.
* **Tool call**: Add ```--tool-call-parser qwen3``` for tool call parser. 

## Roadmap
* [x] Initial Qwen3-Coder Support
* [x] Initial Qwen3-Coder User Guide
* [ ] Streaming Tool Call Support
