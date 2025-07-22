# Qwen3-Coder Usage
[Qwen3-coder](https://github.com/QwenLM/Qwen3) is the largest language model developed by Qwen team, Alibaba Cloud.

SGLang has supported Qwen3-coder already. The `tool-parser` shall be used in latest main branch of SGLang. 

Ongoing optimizations are tracked in the Roadmap. 

# Launch Qwen3-Coder with SGLang

To serve Qwen3 model on H200 GPUs:

For BF16 model

```
python3 -m sglang.launch_server --model-path Qwen/Qwen3-Coder-480B-A35B --tp 8 --tool-call-parser qwen3 --enable-ep-moe
```

For FP8 model

```
python3 -m sglang.launch_server --model-path Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 --tp 8 --tool-call-parser qwen3 --enable-ep-moe
```

or

```
python3 -m sglang.launch_server --model-path Qwen/Qwen3-Coder-480B-A35B-FP8 --tp 8 --enable-ep-moe --tool-call-parser qwen3 --enable-ep-moe
```

## Configuration Tips
* **FP8 models** : With `--tp` Loading failure is expected; switch to expert-parallel mode using ```--enable-ep-moe```.
* **Tool call**: Add ```--tool-call-parser qwen3``` for tool call parser. 

## Roadmap
* [x] Initial Qwen3-Coder Support
* [x] Initial Qwen3-Coder User Guide
* [ ] Streaming Tool Call Support
