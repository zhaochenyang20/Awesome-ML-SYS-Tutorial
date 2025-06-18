# 一文理解 special tokens 和 chat template

【[English](special_tokens_en.md) | [中文](special_tokens.md)】
和 LLM 打交道的朋友无论如何都绕不过 special tokens 和 chat template 这两个概念。说来神奇，既然是每天打交道的概念，总该有很好的参考资料吧，但是为了写这篇文档，我和朋友四处寻找，能找到的只有 HuggingFace 零星的文档。然而，如果处理不好 special tokens 和 chat template，无论是 training 还是 inference，模型的 performance 都会受到极大影响。令人惊异的是，即便影响巨大，居然也没有很好的博客来讲这事，所以我们在此简述下这两个概念。

PS：这篇文章是我和 [Shuai Shi](https://shuaills.github.io/) 在 debug openrlhf 的时候所写，在 SGLang 也重新检查了 token in token out 的 inference 模式。

## special tokens

在预训练阶段，模型在大量连续文本上进行训练，并不主动涉及到交互场景。而在 post training 阶段，大多数模型需要学习特定的交互方法，在此引入 special tokens 来划分交互过程的阶段。举一些 llama 3 special tokens 的例子：

- **标记对话或文本的开头和结尾**：
    - `<|begin_of_text|>`：对话的开始。
    - `<|end_of_text|>`：对话的结束，和 `<|eot_id|>` 是类似的。
- **区分角色**：
    - `<|start_header_id|>user<|end_header_id|>`：下面的内容来自 `user`；
    - `<|start_header_id|>assistant<|end_header_id|>`：下面的内容是 `assistant` 的回答；
    - `<|start_header_id|>system<|end_header_id|>`：表明 system prompt；
- function calling：
    - 部分模型会用 `<tool_call>`…`</tool_call>` 的形式，或以 JSON schema 发起函数调用；

这些 Special Tokens 需要在模型微调阶段就被明确定义且反复出现，从而让模型学会其含义。反过来，如果在推理时没有正确使用这些 special tokens，那么模型的效果将会大打折扣。幸运的是，`add_special_tokens` 这一参数默认为 `True`。不过 decode 的时候，`skip_special_tokens` 则默认为 `False`，所以能够看到如下的输出：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
text = "what's the wether today in LA?"
encoded_with_special = tokenizer.encode(text, add_special_tokens=True)
decoded_with_special = tokenizer.decode(encoded_with_special, skip_special_tokens=False)
print("Decoded with special tokens:", decoded_with_special)

# Decoded with special tokens: <|begin_of_text|>what's the wether today in LA?
```

## chat template

有了 special tokens 的概念，再想想实际的多轮对话中，每轮模型和用户的交互会构成一个 list，此外还有其他内容，比如函数调用、检索到的文档，而 chat template 作为 tokenzier 的一部分，便用于拼接这些内容成为单个 prompt。自然，chat template 也需要在训练和推理时保持一致。

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

chat = [
    {"role": "user", "content": "What's the weather today in LA?"},
    {"role": "assistant", "content": "The weather in LA is sunny with a high of 75°F."},
    {"role": "user", "content": "Will it rain tomorrow?"},
    {"role": "assistant", "content": "No, it's expected to be clear with a low of 58°F."},
]

encoded_chat = tokenizer.apply_chat_template(chat, tokenize=False)
print("Encoded chat:", encoded_chat)

--------

Encoded chat: <|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

What's the weather today in LA?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The weather in LA is sunny with a high of 75°F.<|eot_id|><|start_header_id|>user<|end_header_id|>

Will it rain tomorrow?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

No, it's expected to be clear with a low of 58°F.<|eot_id|>
```