# Scheduler Overview

## [English version](./readme.md) | [简体中文](./readme-CN.md)

> About the Author: Hi, I'm Guanhua Wang, a second-year Master's student in Computer Science at UESTC. I'm currently seeking an internship in inference acceleration. Feel free to get in touch at [281484683@qq.com](mailto:281484683@qq.com)!

The Scheduler structure is located in the file python/sglang/srt/managers/scheduler.py of the SGLang project, and it serves as the core of SGLang scheduling. It receives numerous inference requests from the tokenizer, selects a batch of requests for execution, and is responsible for the following data flow overview:

```python
recv_from_tokenizer -> batch -> schedule
    -> forward -> get_forward_res ->send_to_detokenizer
```

The Scheduler receives numerous inference requests from the tokenizer, but our GPU memory is limited, and we cannot run all requests simultaneously. Therefore, with GPU memory as a constraint, we aim to maximize memory utilization and select as many runnable requests as possible from the received inference requests.

Since this structure has many functions and fields, and some code effects are subtle, to avoid getting overly caught in code details, this article will approach from the overall functionality, first introducing what functions a scheduler should implement, and then mapping these concepts to functions in the SGLang code. This article will not explain functions one by one. If you want to see the complete code flow, you can start from the event_loop_normal function. I believe after reading this article, reading the event loop will be more straightforward. Most optimization points discussed in this article are essential for any inference system, and I hope that after reading this article, you will also be able to apply these concepts to other inference systems.

So, how do we maximize memory utilization? We can start from the following aspects:

- Continuous Batching: Dynamically build scheduling batches to improve throughput. In traditional LLM inference batch processing systems, a decode batch must wait for all requests to complete (i.e., infer a stop token) before returning and scheduling the next batch. This approach is obviously not efficient enough. For example:

  > For two requests - asking an LLM about Tang Dynasty history and asking about today's weather in Chengdu, clearly the Chengdu weather can return after just a few tokens, while the Tang Dynasty history will output a large number of tokens. We don't need to wait for the LLM to finish inferring Tang Dynasty history before returning the Chengdu weather along with Tang Dynasty history in the same batch.

  Continuous Batching was proposed by Orca, with the idea of dynamically constructing batches: for requests in a batch, if a request is completed, remove it from the batch and return it; if new requests arrive, decide whether to add them based on GPU resource status. In SGLang, dynamic addition of requests is implemented in the `get_next_batch_to_run` function, while dynamic removal is completed in the `process_batch_result` function.
- Page Attention: For a request, traditional batch processing systems directly allocate KV cache space according to the model's `max_length`. For example, when using the llama3-70B-8k model, a simple weather query request would pre-allocate 200GB of memory, which is clearly wasteful! Page Attention was proposed by VLLM. We can borrow the page table concept from operating systems. To solve the large amount of internal fragmentation waste in pre-allocated memory, we use a mapping to dynamically allocate the KV cache needed for the current request, rather than pre-allocating.
  SGLang uses `ReqToTokenPool` and `TokenToKVPool` to implement this dynamic mapping mechanism.
- Radix Attention: Different requests may share the same prefix. For tokens with the same content and position encoding, there is no need to repeatedly calculate the KV cache of their prefix. For example, AI agents for multiple users may share the same prompt prefix. We only need to calculate the KV cache of the same prefix once, and then other users can reuse this prefix, without each user's request calculating independently.
- Radix cache aware scheduling: Since requests may reuse the same prefix, selecting more requests with the shared prefixes increases GPU resource utilization. Therefore, we can sort requests according to the longest prefix matching principle, prioritizing requests with the longest matches to join the batch.
- Inference "congestion avoidance": Similar to TCP congestion avoidance mechanism's dynamic adjustment strategy, which gradually increases the congestion window to probe the network's limit, the Scheduler implements a similar approach through the `new_token_ratio` and `new_token_ratio_decay` fields to probe the GPU's limit. When a batch executes successfully, the Scheduler gradually increases the scale of the next batch to achieve higher throughput; when a batch fails, the Scheduler reduces the scale of the next batch to avoid resource overload. This dynamic adjustment mechanism not only approximates the maximum utilization of GPU memory but also ensures relatively stable inference performance.

Through these basic optimization strategies, SGLang significantly improves the efficiency and performance of the inference system while maximizing GPU memory utilization. Next, let's look at how each strategy is implemented.

## Continuous batching

SGLang dynamically adds requests to batches through the `get_next_batch_to_run` function, and processes the dynamic exit of requests through the `process_batch_result` function. Its core process is as follows:

![image-20250302122706407](static/get_next_batch_to_run.png)

SGLang is currently **prefill-prioritized**. If new prefill requests arrive, the system will temporarily interrupt the running decode requests after the current decode requests complete their forward pass. It will then run the new prefill requests, transform them into the decode phase, merge them with the previous decode batch, and continue running.

At this point, we might ask two questions: Why interrupt the current decode requests to prioritize prefill? Won't processing a batch of new prefill requests consume a large amount of GPU resources and lead to poor user experience?

If we prioritize decode, and the time to complete inference for a request is very long, it would cause new inference requests to wait a long time before execution, significantly increasing the user's TTFT (Time To First Token) and leading to user dissatisfaction. In the current computation model, the cost of performing a full prefill on a batch is approximately the same as performing one decode step on a batch - they can be understood as having the same cost per step. Therefore, interrupting decode operations for prefill doesn't cause already-decoding requests to be blocked for a long time on the prefill batch. On the contrary, by converting prefill to decode in one step, it actually benefits the system's continuous operation, thereby improving overall efficiency. (Of course, there are now PD separation and chunked prefill approaches to solve this problem)

### Scheduling Prefill Requests

The function for scheduling prefill requests is `get_new_batch_prefill`, which determines whether to add a request to the Prefill Batch. Its implicit inputs include the current system resource status and the prefill request waiting queue `waiting_list`, and its output is the scheduled ScheduleBatch. Its core performs two levels of scheduling:

- Waiting queue priority calculation: According to the selected SchedulePolicy, it dynamically calculates which requests in the current waiting_queue have higher weights, thus determining which ones will be executed earlier. The scheduling at this stage is resource-independent, meaning that regardless of whether there are currently enough resources to execute these requests, it will first calculate their weights.
- PrefillAdder: After obtaining the requests sorted by weight in descending order, it dynamically decides which requests from the waiting_queue to add to the Batch based on the current GPU resource amount.

At the end of the `get_new_batch_prefill` function, the `prepare_for_extend` function is called to specifically allocate cache and set up prefix reuse. For details about prefix reuse, please refer to the Page Attention section below.

### Scheduling Decode Batch

The function that schedules decode requests is `update_running_batch`, which checks whether the current GPU has sufficient resources for decoding. If not, it evicts requests. Its main process is as follows:

![image-20250302122718301](static/update_running_batch.png)

This function has a two-level memory eviction mechanism to ensure it can allocate the KV cache space needed for this batch from the GPU:

![image-20250302122739244](static/double_check.png)

- The `check_decode_mem` function checks the current system memory. If memory is insufficient, it calls `tree_cache.evict` to first evict **unused caches that are not currently referenced by any req. Note that the evicted KV cache at this stage does not affect the running decode requests.**
- `retract_decode` gradually evicts requests in a while loop until the current remaining space is sufficient for inference. Notably, when this function is entered, it indicates that memory resources are severely insufficient, and an OOM (Out of Memory) has occurred. Therefore, the eviction strategy of this function is very aggressive. To prevent repeatedly entering OOM, it needs to evict enough cache space to support this batch's inference for `retract_decode_steps` rounds. `retract_decode_steps` is set to 20 by default, meaning that if an OOM occurs once, we need to evict space that can guarantee this decode batch runs for 20 steps. Its eviction approach is as follows:
  - Priority is given to evicting requests with few output_ids: The assumption is that the more tokens generated, the easier it is to finish. Early completion leads to early resource release.
  - If output_ids are the same, release reqs with more origin_input_ids: The assumption is that the more `origin_input_ids`, the more tokens need to be generated, so these requests are prioritized for eviction.

After memory detection is completed, we allocate the cache space needed for this batch in the `prepare_for_decode` function (i.e., allocate cache space and set the batch's out_cache_loc field), and implement the autoregressive nature of the transformer through `self.input_ids` =  `self.output_ids`.

### Dynamic Request Exit

After completing this round of inference, we perform checks in the `process_batch_result` function, traversing the current batch to check if there are any completed reqs. If there are, we free the req's kvcache. It's important to note that from a macro process perspective, this "free" is more about releasing the cache reference rather than directly evicting it from the cache. When inference memory is insufficient, we will then perform eviction in the evict function.

The `process_batch_result` function has a small optimization: we can enable TokenToKVPool's free_group before releasing the cache. This way, if we need to release the cache of a bunch of reqs in the batch, we can first buffer them in the free_group, then call `free_group_end` to release them all at once, which is more efficient.

At this point, readers might have a question: **For streaming conversations with long contexts, if we directly free a req's kvcache and the user continues the conversation, wouldn't there be a high probability of recalculation? Would the system's efficiency be high?** My understanding is:

In online processing, if the current request volume is low, although the session's cache is freed in `process_batch_result`, it won't be evicted by `check_decode_memory`. This way, when the next conversation request from that session arrives, it can match the prefix in the cache well and continue computation.

However, if the request volume is very high and GPU memory utilization is high, we can't guarantee that users will definitely continue the conversation consecutively. In this case, the cost of maintaining the cache becomes too high. Considering that the computation time for a prompt prefill and decoding a token are similar, we can just save the user's text prompt while evicting their KV cache. When the user continues to send messages and there are enough resources to support inference, we only need to calculate one prefill to proceed with inference.

## Radix Cache

### Longest Prefix Matching

When selecting newly arrived prefill requests, we can schedule them based on the length of their prefix matches. The more requests with the same prefix we select, the higher the GPU resource utilization. The concept of longest prefix matching is simple - just check if requests have prefixes in the data structure. You can see the specific logic in `radix_cache.py`; I won't elaborate much here.

### In-batch Optimization

If a request has a long prefix match **globally**, then executing it directly is definitely good.

If a request doesn't have a very long **global** prefix, but within the **current batch**, there are many requests that match a significant number of prefixes between them, then we can **choose only one to execute** from the requests in the current batch that have the same prefix, and not execute the others. This approach can improve cache hit rates.

Why would executing only one request improve cache hit rates? This conclusion seems counter-intuitive!

Because if we only process one of them, **subsequent requests can utilize the results of this prefix, reducing redundant calculations.** If we process multiple requests simultaneously, each might need to compute the same prefix, wasting resources. Therefore, **by temporarily downgrading other requests and prioritizing one, subsequent requests can utilize the already calculated prefix, improving hit rates.** For example, in the diagram below, the purple boxes represent the globally matched prefix tokens for each request, while the green represents the prefix-matched tokens within this batch between requests. Requests 2, 3, and 4 don't have very long globally matched prefixes, but they share similar prefixes among themselves. If we run them all, we would need to prefill 12 tokens. But if we only select req2 to execute first, we only need to calculate 6 tokens, and then req3 and 4 will reuse the tokens calculated by req2.

![image-20250302155458252](static/inbatch.png)

This algorithm is implemented in the `_compute_prefix_matches` function, with two core variables: `CHECK_THRESHOLD` and `DEPRIORITIZE_THRESHOLD` shown in the diagram above.

- `IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD`: If a request's global prefix match is greater than this threshold, it will be executed directly without in-batch optimization. If less than this threshold, in-batch matching will be enabled to check if there are duplicate prefixes within the batch. In the diagram above, req1 is executed directly, while 2, 3, and 4 undergo in-batch optimization.
- `IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD`: If the in-batch threshold is less than this value, it indicates that the prefix match is too short, and there's no need for optimization. Examples include common short words in English like "I am" or "the". In the diagram above, 2 and 3 undergo in-batch optimization, but 4 is too short and doesn't undergo in-batch optimization.

## Page Attention

I won't explain the concept of Page Attention as there are many excellent resources about it already available. Instead, I'll focus on how SGLang implements the Page Attention concept through two data structures: `ReqToTokenPool` and `TokenToKVPool`.

TokenToKVPool can be understood as a **page table, responsible for managing the mapping from a token's logical address to specific GPU resources**. When we use `token2kv[idx]` to access an element, we can imagine idx as a virtual address, and through `token2kv[idx]`, we obtain the actual value of its KVcache in the GPU. Meanwhile, ReqToTokenPool stores the logical addresses of each token in a request. `req2token[i, j]` can be understood as: the logical address (i.e., the index in `token2kv`) of the j-th token in the current text of the i-th request in the system.

Note that these two data structures do not store the original values of token_ids in the request; they only handle address mapping. The original values of token_ids are stored in the _origin_input_ids_ field of the request.

![image-20250302161047952](static/page_attn.png)

In the diagram above, req1's ReqToToken field stores the logical addresses `3, 1, 2` for these tokens' KV cache. If we want to access what the KV cache of the first token is, we just need to index `token2kv[3]`. The logic for accessing the KV cache of the jth token of the ith req is: `token2kv[req2token[i, j]]`.

Because we've implemented paged attention, we can naturally reuse previously calculated KV caches - we just need to store the logical address in req2token. For example, in the diagram above, the first two tokens of req1 and req2 can reuse the KV cache numbers 3 and 1.

## Inference "Congestion Avoidance"

This scheduling adjustment is currently only seen in SGLang. Let me first explain why it's implemented this way.

Before adding inference requests, PrefillAdder first calculates the memory resources already occupied in the system, `rem_total_token_offset`. For tokens that haven't been generated yet by a request, we also need to calculate their future cost and reserve space to ensure that the request can complete inference smoothly. However, during execution, the request may not infer all the way to `max_token_length`, and while the current request is running, other requests may dynamically enter and exit. If we rigidly fix the cost of ungenerated request tokens, it would greatly limit the scheduler's flexibility.

When calculating the cost of future tokens, PrefillAdder multiplies it by `new_token_ratio` to reduce its weight. Typically, this ratio is less than 1, meaning that future tokens don't need as much reserved space. The code is as follows:

```python
if running_batch is not None:
    self_.rem_total_token_offset += sum(
        [
            min(
                (r.sampling_params.max_new_tokens - len(r.output_ids)),
                CLIP_MAX_NEW_TOKENS_ESTIMATION,
            )
            * self_.new_token_ratio  
            for r in running_batch.reqs
        ]
    )
```

So how is `new_token_ratio` adjusted? SGLang's approach is to base it on whether there's a decode OOM (Out of Memory) in `update_running_batch` (meaning whether the newly generated tokens will exceed the kv pool size even after evicting free nodes). When a batch executes successfully, the Scheduler decreases `new_token_ratio`, gradually increasing the scale of the next batch. When a batch experiences OOM, the Scheduler increases `new_token_ratio`, reducing the scale of the next batch to avoid resource overload.

![image-20250302161630229](static/new_token_ratio.png)

As reflected in the graph, the batch size (bs) in the first two rounds of decode executes successfully and gradually increases; rounds 3-4 also execute successfully, but reach our self-set upper limit - increasing further might cause OOM, so the current bs is maintained; in round 5, a decode OOM occurs, so we make the scheduling more conservative and reduce the bs; in round 6, execution is successful, so bs gradually increases again.

This graph strongly resembles TCP's linear congestion avoidance. In TCP, the congestion window (cwnd) is increased to approach the network's limit; here, the batch size is increased to approach the GPU's limit. In TCP, packet loss reduces cwnd; here, OOM reduces the batch size.