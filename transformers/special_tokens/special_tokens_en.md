# Understand Special Tokens and Chat Template in One Article

【[English](special_tokens_en.md) | [Chinese](special_tokens.md)】
Friends who deal with Large Language Models (LLMs) can't avoid the two concepts of special tokens and chat template no matter what. It's quite amazing that since these are concepts we deal with every day, there should be good reference materials, right? However, in order to write this document, my friend and I searched everywhere, and all we could find were some sporadic documents from HuggingFace. Nevertheless, if special tokens and chat template are not handled properly, the performance of the model will be greatly affected, whether during training or inference. Surprisingly, even though the impact is significant, there are no good blogs explaining this matter. So, we will briefly introduce these two concepts here.

PS: This article was written by me and [Shuai Shi](https://shuaills.github.io/) when we were debugging openrlhf, and we also rechecked the token in token out inference mode in SGLang.

## Special Tokens

During the pre-training stage, the model is trained on a large amount of continuous text and doesn't actively involve interactive scenarios. In the post-training stage, most models need to learn specific interaction methods, and special tokens are introduced here to divide the stages of the interaction process. Here are some examples of special tokens in Llama 3:

- **Mark the beginning and end of a conversation or text**:
    - `<|begin_of_text|>`: The start of a conversation.
    - `<|end_of_text|>`: The end of a conversation, which is similar to `<|eot_id|>`.
- **Distinguish roles**:
    - `<|start_header_id|>user<|end_header_id|>`: The following content is from the `user`;
    - `<|start_header_id|>assistant<|end_header_id|>`: The following content is the answer from the `assistant`;
    - `<|start_header_id|>system<|end_header_id|>`: Indicates the system prompt;
- Function calling:
    - Some models use the form of `<tool_call>`…`</tool_call>`, or initiate a function call with a JSON schema;

These special tokens need to be clearly defined and repeatedly appear during the model fine-tuning stage, so that the model can learn their meanings. Conversely, if these special tokens are not used correctly during inference, the performance of the model will be greatly reduced. Fortunately, the `add_special_tokens` parameter is set to `True` by default. However, when decoding, the `skip_special_tokens` parameter is set to `False` by default, so you can see the following output:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
text = "what's the wether today in LA?"
encoded_with_special = tokenizer.encode(text, add_special_tokens=True)
decoded_with_special = tokenizer.decode(encoded_with_special, skip_special_tokens=False)
print("Decoded with special tokens:", decoded_with_special)

# Decoded with special tokens: <|begin_of_text|>what's the wether today in LA?
```

## Chat Template

With the concept of special tokens in mind, think about an actual multi-turn conversation. Each turn of interaction between the model and the user forms a list, and there are other contents as well, such as function calls and retrieved documents. As a part of the tokenizer, the chat template is used to concatenate these contents into a single prompt. Naturally, the chat template also needs to be consistent during training and inference.

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