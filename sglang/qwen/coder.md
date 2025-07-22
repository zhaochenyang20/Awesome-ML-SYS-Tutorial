# Qwen3-Coder Usage
[Qwen3-coder](https://github.com/QwenLM/Qwen3) is the large language model developed by Qwen team, Alibaba Cloud.

SGLang has supported Qwen3-coder (480B) since [v0.4.6](https://github.com/sgl-project/sglang/releases/tag/v0.4.6).

Ongoing optimizations are tracked in Roadmap. 

# Launch Qwen3-Coder 4 with SGLang
To serve Qwen3 model on 4/8xH100/200 GPUs:

For BF16 model
```
python3 -m sglang.launch_server --model-path Qwen/Qwen3-Coder-480B-A35B --tp8 --tool-call-parser qwen3_moe
```

For FP8 model
```
python3 -m sglang.launch_server --model-path Qwen/Qwen3-Coder-480B-A35B-FP8 --tp 4 --tool-call-parser qwen3_moe
```
or
```
python3 -m sglang.launch_server --model-path Qwen/Qwen3-Coder-480B-A35B-FP8 --tp 8 --enable-ep-moe --tool-call-parser qwen3_moe
```

## Configuration Tips
* **FP8 models** : With --tp 8 Loading failure is expected; switch to expert-parallel mode using ```--enable-ep-moe```.
* **Tool call**: Add ```--tool-call-parser qwen3_moe``` for tool call parser. 

## Roadmap
* [x] Initial Qwen3-Coder Support
* [x] Initial Qwen3-Coder User Guide
* [ ] Streaming Tool Call Support
