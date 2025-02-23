# KV Cache Code Walkthrough
<!-- TODO(xiaotong):  CN -->
This document offers a comprehensive overview of the KV cache management system within the SGLang implementation, delving into the lifecycle of requests and the pivotal components that facilitate this lifecycle, encompassing the Scheduler, Radix Cache, Attention Backend, and all the supporting resources.

To facilitate this explanation, we will make a few assumptions in our examples:
- We use Flash Infer as backend
- We use Longest Prefix as priority

## Global State

This section provides a brief overview for some of the important global state that are maintained across requests.

### KV Cache & Memory Pools

KV Cache is the most important global state in the server because it can take a significant fraction of GPU Memory. There are two-level memory pools to manage KV cache. 

#### `req_to_token_pool`
- **Purpose** `req_to_token_pool` maps a request to its tokens' KV cache indices.
- **Layout:** Max Allowed Requests Number * Max Allowed Tokens Number
- **Access:** 
    - Dim0: `req_pool_indices`
    - Dim1: token positions in req, starting from 0, 1, 2...
    - Value: `out_cache_loc` for token
  
#### `token_to_kv_pool`
- **Purpose** `token_to_kv_pool` maps a token KV cache indices to its KV cache data, `token_to_kv_pool` has model-specific implementation like MHA, MLA, DoubleSparse.
- **Layout:** Number of Layers * Max Allowed Tokens Number * Number of Head * Head Dimension
- **Access:** 
    - Dim0: `layer_id`
    - Dim1: `out_cache_loc`
    - Dim2: Head
    - Dim3: Head Dimension
    - Value: `cache_k` for k_buffer and `cache_v` for v_buffer

#### `tree_cache`
- **Purpose** A tree structure to enhance the reuse of prefix KV Cache across requests.
- **Access:**
  - Key: Token ID
  - Value: Token's KV Indices

### Active Requests Management

The `Scheduler` component is responsible for managing active requests. The following core global state are utilized to maintain these active requests in `Scheduler`.

#### `waiting_queue`
- **Purpose** The Waiting Queue is a data structure designed to hold active requests. It dynamically reorders these requests based on priority or available memory to optimize batch processing tasks.
- **Important Notes** 
  - **Enqueue** 
    - Newly arrived requests are enqueued into the waiting queue.
    - Requests that are returned from `retract_decode`.
  - **Dequeue** The highest priority requests are dequeued from the queue to form a batch.

#### `new_batch`
- **Purpose**: A batch of requests that are ready for prefill/extend stage.
- **Important Notes** 
  - **Chunked Prefill**: If a request requires more tokens than the available memory (`remaining_tokens`), it may be **chunked** into smaller parts.
  - Requests in `new_batch` would go through prefill/extend.
  - After prefill/extend, `new_batch` will transit to **Global Batch** for the upcoming iteration.  

#### `running_batch`  
- **Purpose**: A batch of requests that are ready for decode stage.
- **Important Notes** 
  - **Retract**: If available memory is insufficient during decode, the scheduler may retract certain requests (via `retract_decode`) from the `running_batch`, returning them to the waiting queue for later processing.

#### `cur_batch`
- **Purpose**: The batch of requests that are currently being processed in the main loop of Scheduler (`run_batch` function).  
- **Important Notes**
  - `cur_batch` is assigned in `event_loop_normal`.
  - The logic of forming `cur_batch`is: If there's requests ready for prefill (`new_batch`) in this cycle, use `new_batch` as `cur_batch`. Otherwise, `cur_batch` would process those that are ready for decode, thus use `running_batch` as `cur_batch`.  

## Scheduler Overview

This section provides a high-level overview of the `Scheduler`'s request management process.

### **Scheduler**
![alt text](scheduler_overview.png)
The figure illustrates how the **Scheduler** directs requests from the `waiting_queue` into a `new_batch` (for prefill/extend phase) and then into the `running_batch` (for decode phase).

1. **New Request arrived**: The Scheduler continuously calls `recv_requests` to collect newly arrived requests, validate them and place them into the `waiting_queue`. In our example, `Req 7` are received and enqueued.

2. **Merge Batches**: Before form the new batch for this round, Scheduler would merge the `cur_batch` from last round into `running_batch`. (In the diagram, `cur_batch` from last round are shown as `cur_batch(i-1)` and `running_batch` are shown as `running_batch(i-1)`. In our example, `Req 0` and `Req 1` will be merged together into the new `running_batch`. **Merge Batch** will also remove the last round `being_chunked_request`. `being_chunked_request` is the chunked prefilled request generated during the `get_new_batch_prefill` process. (In the diagram, there are finished `being_chunked_request` e.g., `Req 5a` which means the first part of `Req 5`), we will remove this as we do not want them to do decode phase.) 

3. **Forming the New Batch**: Scheduler would check if a `new_batch` could be formed (in `get_new_batch_prefill`), all the requests that can fit available memory would be packed in the batch. If the size of the last request to put into the batch is larger than the remaining available memory, the request will be chunked as `being_chunked_request`. In out example diagram, the Scheduler pulls requests from the `waiting_queue` and creates a `new_batch`(e.g., `Req 6`, `Req 5b`, `Req 5b` is the `being_chunked_request`), and use the `new_batch` as `cur_batch`. Not demonstrated in the diagram but if there is no `new_batch`, the `running_batch` will be filtered(e.g., `Req 1`, `Req 0` will be kept while `Old Finished Req` will be removed), and then be used as `cur_batch`. Also, if the GPU memory is not enough, some decoding requests may be retracted according to certain retract policy. During the `retract_decode` phase, in the diagram, `Req 0` is retracted.

4. **Running the Batch**: Once the **Global Batch** is determined (prefill vs. decode), `run_batch` is called to run a forward pass.

5. **Result Processing**: After `run_batch`, the Scheduler calls `process_batch_result` to to determine which requests have finished and which continue. In our example, `Req 6` is finished and turns grey, `Req 5b` remains unfinished.

6. **Iteration**: The loop repeats until all requests are eventually completed. If insufficient memory is encountered, requests may be chunked (in prefill) or retracted (in decode), then reinserted into the waiting queue for later processing.

<!-- TODO(mingyuan):  CN -->
## One Request Lifecycle
![alt text](kv-cache-request-lifecycle.png)

Following one request lifecycle, this section provides a step-by-step walkthrough of the key functions that updates the KV Cache & Memory Pools, we will use request ABC as an example.

#### Prefill Batch
##### Step 1. Function `get_new_batch_prefill` 
  - Update prefix from radix tree cache for request
    - When `ABC` come in, say the current radix cache have one node `AFG` thats currently being referenced.
    - `match_prefix` would try to find the prefix for `ABC` in the current radix cache, and it matched up to the first token `A` in the node.
    - Radix Cache Split the node into `A` and `FG`, node `A` is the current last node for this request.
  - Invoke `prepare_for_extend`
    - `req_to_token_pool`
      - Allocate to `req_pool_indices`
      - Add prefix to `req_to_token_pool`
    - `token_to_kv_pool`
      - Allocate (the sum of each reqs (number of input id tokens - number of prefix tokens)) `out_cache_loc`
      - In our example of request ABC, the batch size is 1
        - number of input id tokens = 3 -> A,B,C 
        - number of prefix tokens = 1 -> A
        - We will allocate 2 slots to `out_cache_loc` for token B, C
        
##### Step 2. Function `run_batch` 
Run `forward_extend` on the current batch, this will eventually invoke the Attention Backend, who is responsible for 
- Set the kv cache of extend tokens.
  - Set KV cache for extends token to `token_to_kv_pool` (Function `save_kv_cache`)
  - In our example: from above steps, we get 2 slots for token B, C in `out_cache_loc`, their corresponding K, V would be set to this 2 slots here.
- Run forward attention calculation, the input would be
  - Q = extend tokens, in our example token B, C
  - KV = All cached tokens from `req_to_token_pool` by `out_cache_loc` including A(prefix tokens), B, C(extend tokens) (Function `create_flashinfer_kv_indices_triton`).

##### Step 3. Function `process_batch_result_prefill`
  `cache_finished_req` and `cache_unfinished_req` are responsible for managing the KV cache in Radix Cache, ReqToTokenPool, and TokenToKVPool.
  - If the request is finished, invoke `cache_finished_req` (refer to [PLACEHOLDER] for details of `cache_finished_req` )
  - else invoke `cache_unfinished_req` (refer to [PLACEHOLDER] for details of `cache_unfinished_req` )

  In our example, `cache_unfinished_req` is invoked after extend/prefill phase, `BC` was added as a child node for `A`, both nodes `A` and `BC` increase the lock reference, node `BC` become the `last_node` for the request.

#### Decode Batch
##### Step 4. Function `update_running_batch` 
  - Invoke `prepare_for_decode`
    - `req_to_token_pool` No change
    - `token_to_kv_pool`
      - Allocate (batch size * 1) slot to `out_cache_loc` because we only generate one token for each batch in decode mode
      - For example: in above diagram, the round that generate D from C
        - We will allocate 1 slots to `out_cache_loc` for token D

##### Step 5. Function `run_batch`
Run `forward_decode` on the current batch, this will eventually invoke the Attention Backend, who is responsible for 
- Save the kv cache of decode token.
  - Save KV cache for decode token to `token_to_kv_pool` (Function `save_kv_cache`)
  - For example: In above step, we get 1 slots for token D in `out_cache_loc`, it's corresponding K, V would be saved to this 1 slot here.
- Run forward, the input would be:
  - Q = decode token, in our example token D
  - KV = All cached tokens from `req_to_token_pool` by `out_cache_loc` including A, B, C(from previous round), D (Function `create_flashinfer_kv_indices_triton`)

##### Step 6. Function `process_batch_result_decode`
  If the request is finished, invoke `cache_finished_req` (refer to [PLACEHOLDER] for details of `cache_finished_req` ). No operation for cache is needed for unfinished request in decode phase.
  
  In our example, `DE` is appended to node `BC`, and the lock reference for node `A` and `BCDE` got decreased. 

<!-- TODO(yangmin):  CN -->
#### RadixCache `cache_finished_req` & `cache_finished_req`
This section would go deeper on `cache_finished_req` & `cache_finished_req`'s flow.

##### **Quick Overview of `cache_finished_req()` vs. `cache_unfinished_req()`**
| Sequence | `cache_unfinished_req()` | `cache_finished_req()` |
|------|--------------------------|--------------------------|
| **1. Get `kv_indices`** from `req_to_token_pool.req_to_token` | - | - |
| **2. Update Radix Cache** (`insert()`) | - | - |
| **3. Free KV Cache** (`self.token_to_kv_pool.free()`) | - | release duplicate kv cache storage |
| **4. Handle `req_to_token_pool`** | **Writes and updates** `req_to_token_pool | **Releases** `req_to_token_pool` as the request is completed. |
| **5. Handle `req.last_node`** | **Increases** the reference count of `req.last_node` | **Decreases** the reference count of `req.last_node`, as `req` is finished. |

As we can observe, the core functionality is essentially the same for `cache_unfinished_req()` and `cache_finished_req()`, we are going to walk through the code for how req_to_token_pool, token_to_kv_pool and tree_cache are updated during `cache_unfinished_req()`, and explain the key difference between `cache_unfinished_req()` and `cache_finished_req()`.

##### `cache_unfinished_req`
1. Get KV Index: Get KV index from `req_to_token_pool.req_to_token`

2. Update Radix Cache
```python
new_prefix_len = self.insert(token_ids, kv_indices.clone())
```
This method inserts token_ids and their corresponding kv_indices into the Radix Cache. If successful, it returns a new prefix length (new_prefix_len).

3. `Free KV Cache` 

4. Update `prefix_indices` and `last_node`
  
    Calls `match_prefix()` to update the request’s `prefix_indices` and `last_node`. This is important as `prefix_indices` and `last_node` will be used in next iteration.
  - `prefix_indices` is used to calculate the needed size of extend/prefill tokens.
  - `last_node` is used to keep track of a requests last node in the radix tree so that it can be used to trace back towards root node when updating lock.

5. Lock Management for Memory Safety
To prevent unintended deletion of active cache nodes. Keeping a lock on a node shields it from being freed while still needed. After above state transition, the old `last_node` is unlocked using dec_lock_ref(), allowing it to be freed when no longer in use. The new `last_node` is locked, protecting it from deletion.

##### `cache_finished_req`

###### **Similar to `cache_unfinished_req()`, `cache_finished_req()` also has the following steps:**
1. When a request `req` is completed, its `token_ids` are stored in the **Radix Cache**. Update Radix Cache 
2. **Release** redundant **KV Cache space** in `token_to_kv_pool` (removing duplicates).
3. **Release `req_to_token_pool`** and **update `tree_cache`**.  
