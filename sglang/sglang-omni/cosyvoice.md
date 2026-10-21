# 以 CosyVoice 为代表的 AR + DiT + Vocoder 生成模型优化思路

过去几个月，我们观察到相当量的 TTS 模型采用了 AR + DiT + Vocoder 架构，这些模型通常还会额外带一个 speaker encoder 用来做音色克隆，所以严格说是四个 module 串起来的一条链路。为了和主流说法保持一致，本文仍然沿用 AR + DiT + Vocoder 这个简称。

在这之中，DiT 的加入使得这些模型的音质有了显著提升，也带来一些新的优化设计挑战。我们这里总结分享 SGLang Omni 对此类模型的优化思路，主要是 [issue 1652](https://github.com/sgl-project/sglang-omni/issues/1652) 所 track 的各阶段优化。总结思路的同时，我们希望把这套方法论扩展到更多 AR + DiT + Vocoder 架构的模型上，譬如 [dots.tts](https://github.com/studio-dots-ai/dots.tts) 和 [Minimax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3)。

为了准确描述我们的优化思路，本文所有代码片段对照 `sgl-project/sglang-omni` main @ `0ee0304a5`（2026-10-16）。

## 模型架构

这一类的 TTS 模型的主要方案是通过 AR 部分生成 audio token，随后使用扩散部分（以 DiT 结构为主）将 audio token 渲染为 mel 谱，而后则是经典的 vocoder 把 mel 转为波形的过程。

- **AR 部分**：骨干就是 Qwen2-0.5B，词表是 6561 个 speech token 外加 200 个控制 token（`sglang_model.py`），帧率是 25 Hz，也就是 1 秒音频对应 25 个 token。
- **Speaker encoder**：参考音频编码出一个 192 维的 speaker embedding，用来控制音色。严格来说它是这条链路上第四个 module，也是预处理阶段的 heavy work（见后文多线程 CPU 处理）。
- **Flow / DiT**：输入 80 通道的 mel，token 到 mel 的比例是 2，所以 mel 是 50 Hz。DiT 的 hidden size 是 1024，静态 chunk size 是 50 mel 帧，正好等于一个 25 token 的 hop（这个对齐关系见后文 Scheduler 一节）。
- **ODE 求解器**：10 步的 Euler，但由于 cfg，实际前向次数要翻倍。
- **HiFT**：基于 ISTFT 的神经声码器。mel 帧到采样点的倍率是 $8 \times 5 \times 3 \times 4 = 480$，配上 24 kHz 采样率，正好 50 Hz 的 mel。

所以一条 5 秒的音频，整条链路是：

```
125 个 token → 250 帧 × 80 通道 mel → 12 万个采样点
```

从算法设计上来说，这种 multi-stage 的方案通过 diffusion 提升了不少音质，而 CosyVoice 这种模型在引入 speaker encoder 之后，音色相似度也很高。但是多阶段生成在 infra 层面给我们带来了一系列挑战：

- **AR 只有一个 decode step，DiT 有 10 步 × CFG 2 次。** 前者的瓶颈和 LLM 一样是 memory-bound 的 decode，后者每一步都是一次完整的 DiT 前向，走的是完全不同的优化路径。
- **DiT 是 chunk-causal 的，而且带 lookahead。** 每一帧只能看到自己所在的 chunk 和之前的 chunk，但每个 chunk 还要额外需要 3 个 token 的 lookahead 才能开始算。这直接影响了流式调度的切分策略。
- **三个阶段的"长度"含义都不一样。** AR 按 token 数算，DiT 按 mel 帧数算，HiFT 按采样点数算，三者之间差着固定的倍数，但 batch 起来要 pad 的维度各不相同。

## Runtime Profiling

为了优化 CosyVoice，我们整理出一套 runtime profiling 的 method，并给 agent 配套了探测 skill 和优化 skill，具体见 [PR 1850](https://github.com/sgl-project/sglang-omni/pull/1850)，感兴趣的读者欢迎参考。

【我发现 OpenRSI 似乎抓性能提升点的点都不用 profiling agent，我非常惊讶，之后可以用 OpenRSI 来更新下 Profile Agent。此外，OpenRSI 为了 cosy voice merged 的 PR，写到这个文档里了么？】

任何优化都逃不过这三个问题：

1. 哪里速度慢，拖慢了整体的效率？
2. 这个优化到底是不是在拿效果换速度？
3. 这个优化它是在什么 workload 下产生了效益？

而 runtime profiling 则是我们对于第一个问题的一次尝试。

通过给这里多个阶段的 module 进行阶段性的测速，哪个部分拖慢了生成的速度便显而易见了，而在往后的优化过程中，这个方案我们可以反复使用，因此固定出一套方法论从长远维护的角度来看，是很有收益的。同样，这套高度重复的工作也可以交给 agent 进行埋点，从而减少人工投入。

TTS 的效果指标和 LLM 不一样，我们主要看 WER：把生成的音频重新过一遍 ASR 转成文字，再和参考文本比对。也是最能反映音质是否被优化破坏的指标（我们的 baseline 数据就是用 SeedTTS 这套评测跑的）。此外还会看说话人相似度，也就是生成音色与参考音频的接近程度。

而在优化过程中常常会出现由 AI 挖掘的伪需求，实际上这个优化引入之后并没有多大的变化，甚至落在噪声区间内，因此我们要构建一个严格的 AB Test 去衡量最后的效益。

【可能我们有了 OpenRSI 后得写个新的，感觉 OpenRSI 真的牛逼，没有那么多 fake 优化，我们自己的 agent 就 fake 很多】

## 优化分解

推理加速优化大致可以分为两个层次：runtime level 和 kernel level，按照“是否改变单个算子的计算方式”进行分类。目前 SGLang Omni 对大部分模型的优化都还停留在 runtime level，而且这一层本身也还没有优化到位。

这里说的"算子"是指一次实际的计算，比如一次 matmul、一次 attention、一次 conv。改变它的**执行形态**（换一个实现、融合掉、编译掉）算 kernel level，剩下的算 engineering level。

此外，我们的实际经验表明，在工程层面（上游 runtime overhead）没有优化完全的情况下，任何 kernel level 的收益都是缺乏置信度的。

### Runtime Optimized

常见的工程层的优化，一般包括打高 batch，优化流水线里面 CPU 和 GPU 的负载，并行处理音频等，而 CosyVoice 引入了一项更重要的工程措施：优化调度层。对于多阶段的推理，如何编排好每个请求，让它们既能被加速，又不会出现调度错乱，变成了本次优化过程中的一大挑战。

由于 AR 部分的推理加速方案几乎可以直接沿用主流 LLM 推理引擎的方式，因此本文不会用过多的篇幅讲解相关内容，但是这并不代表 AR 部分不重要，只是由于方案成熟，我们采用的优化手段可能并不会产生更多的超额效益。

**Batch inference**

几乎所有的模型，首先想到的卓有成效的优化手段都是 batch inference。而 batch inference 的核心思想就是按照矩阵乘的性质，把一组 target 全部 padding 到一个 M*N 大小的矩阵进行推理，而后对里面每一条都进行裁切，从而把吞吐提上去。这里要注意它提升的是**吞吐**而不是单请求的延迟：单次前向里多算几条几乎不增加耗时，但单条请求该等的计算一步没少，甚至可能因为要凑批而等待更多。

示例代码如下：

```python
# sglang_omni/models/fun_cosyvoice3/stages.py，`CosyVoice3Vocoder.decode_batch`（示意：省略 stream/autocast 上下文与 results 回填）
flow_groups = adaptive_flow_requests_grouping(
    prepared,
    flow_merge_max_gap_frames=self.flow_merge_max_gap_frames,          # 组内最大长度差
    flow_merge_pad_budget_percent=self.flow_merge_pad_budget_percent,   # 全局 padding 预算
)
for flow_group in flow_groups:
    # 一次调用求解整组：padding / masking / CFM 解码都在内部处理
    mel_list = self.flow.inference([r.flow_input for r in flow_group])
    # HiFT 独立分组：按 mel 长度排序后贪心切分，组内 padding 浪费不超过 hift_max_padding_waste
    ordered = sorted(zip(flow_group, mel_list, strict=True),
                     key=lambda pair: int(pair[1].shape[-1]))
    group, total, longest = [], 0, 0
    hift_groups: list[list[tuple[PreparedFlowRequest, torch.Tensor]]] = []
    for pair in ordered:
        length = int(pair[1].shape[-1])
        candidate_longest = max(longest, length)
        candidate_total = total + length
        if (group and candidate_longest * (len(group) + 1)
                > self.hift_max_padding_waste * candidate_total):
            hift_groups.append(group)
            group, total, longest = [], 0, 0
            candidate_longest, candidate_total = length, length
        group.append(pair)
        total, longest = candidate_total, candidate_longest
    if group:
        hift_groups.append(group)
    for group in hift_groups:
        # 右 padding 后一次前向，再按真实长度切回
        wavs = self.mel2wav_batch([mel for _, mel in group])
```

流式场景则不一样。每个 hop 都很短，如果每条请求单独前向一次，launch 开销反而成了主导。这里的做法是把多条请求的行沿着**序列维**拼成 `(1, total, C)`：用一个 `row_ids`（每个 token 属于哪一行）加一个 `positions`（它在行内的位置）来描述打包关系，attention 用 mask 限制在各自行内，所以不同请求之间不会互相 attend。

```python
# sglang_omni/models/fun_cosyvoice3/packed_dit.py，`pack_rows` / `gather_rows` / `scatter_rows`（示意：省略 pack_rows 里 30 帧卷积上下文索引的构造）
# 流式场景：多条请求的 mel 行沿序列维打包成 (1, total, C)，attention 用 mask 限制在各自行内
starts_host = F.pad(torch.tensor(lengths, dtype=torch.int64).cumsum(0), (1, 0))
row_ids = torch.repeat_interleave(                   # 每个 token 属于哪一行
    torch.arange(len(lengths), device=device),
    torch.tensor(lengths, dtype=torch.int64, device=device),
    output_size=total)
positions = torch.arange(total, device=device) - starts[row_ids]

def gather_rows(padded, rows):                       # (rows, width, C) -> (1, total, C)
    flat = padded.reshape(padded.shape[0] * width, padded.shape[2])
    return flat[rows.row_ids * width + rows.positions].unsqueeze(0)

def scatter_rows(packed, rows, width):               # (1, total, C) -> (rows, width, C)
    flat = packed.new_zeros(len(rows.lengths) * width, channels)
    flat[rows.row_ids * width + rows.positions] = packed[0]
    return flat.view(len(rows.lengths), width, channels)
```

再看 CFG。每一步 Euler 都需要算两次前向，一次带条件、一次不带，所以 batch 维度是 2N。打包的时候把不带条件的那一份当成"另一行"打进同一个 batch，一次 DiT 调用就能把两个分支一起算完：

```python
# sglang_omni/models/fun_cosyvoice3/stages.py，`generate_flow_packed`（示意：省略了整批 CUDA Graph 命中时提前返回的分支）
# 求解本身在 packed_dit.py 的 `solve_flow_euler_packed`，twin_rows 是它接到的入参之一
twin_rows = pack_rows(rows.lengths * 2, device)   # CFG：条件行 + 一份无条件孪生行，共用一次 DiT 调用
generated = solve_flow_euler_packed(
    estimator, noise, conditioning.time_span, mu, conditioning.speaker_embedding,
    mel_conditioning, twin_rows,
    estimator.row_attention(twin_rows, streaming=streaming,
                            dtype=conditioning.speaker_embedding.dtype),
    estimator.rope_angles(twin_rows.width),
    cfg_rate=flow.decoder.inference_cfg_rate,
)
```

**CUDA Graph capture**

DiT 的 Euler solve 是典型的 launch-bound：10 步 Euler，每步还要跑一次 CFG，拆出来是一大串几微秒级的小 kernel，CPU launch 的速度跟不上，GPU 在 kernel 之间空转。CUDA 的图捕获把整串 kernel 录下来，之后一次 `replay` 就把整串 kernel 发出去，消除逐个 launch 的间隙。它 replay 的还是同一批 kernel，很适合当作一个优先处理的 low-hanging fruit。

graph 是静态的，输入输出的地址和形状在录制时就固定了，代价就是不够灵活，需要提前把形状定好：

- **按白名单捕获**：默认 55 个 `(B, T)` 组合（`config.py` 的 `FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES`），每个形状录一张图。
- **帧数按 16 上取整分桶**。分桶越粗，padding 浪费越多但录的图越少；越细则相反。多出来的部分右 pad，靠 `mel_mask` 屏蔽。
- **所有形状共用一个 memory pool**，否则 55 张图各占一份显存会很浪费。
- **没命中就回退 eager**。所以正确性永远不依赖 graph 是否命中。

还有一点要说明：**这套 graph 只服务非流式的 buffered 路径**（`stages.py` 里 `streaming or not finalize` 时直接走 eager），流式路径的加速来自沿序列维打包和 PackedDiT 的 compile，不是这张图。另外它也是"消除 host sync"那一项的前提：capture 过程中任何一次 host sync 都会让捕获失败。

```python
# sglang_omni/models/fun_cosyvoice3/stages.py，`FlowCudaGraphRunner.capture`（示意：省略 gc.collect 与断言）
# 图捕获走平台抽象层，`torch.cuda.CUDAGraph` / `graph_pool_handle` 已下沉到 platforms/device_graph.py，
# 这样 NPU / XPU 等后端也能复用同一段代码
graph_backend = current_platform.get_device_graph_backend(self.device)
current_stream = self.device_module.current_stream(self.device)
stream = self.device_module.Stream(self.device)     # side stream 录制，default stream 上的 kernel 不会被录进去
stream.wait_stream(current_stream)
with (current_platform.graph_capture_attention(),
      self.device_module.device(self.device),
      self.device_module.stream(stream)):
    self.pool = graph_backend.graph_pool_handle()   # 所有形状共用一个 memory pool
    for batch_size, mel_frame in capture_shapes:    # 按 (B, T) 白名单逐个捕获
        static_inputs = self.capture_inputs(batch_size, mel_frame)
        with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype,
                            enabled=self.autocast_dtype is not None):
            solve_flow_euler(self.flow.decoder, *static_inputs)   # warmup
        with graph_backend.capture(pool=self.pool, stream=stream,
                                   thread_local_errors=True) as graph, \
             torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype,
                            enabled=self.autocast_dtype is not None):
            # 整个 solver（11 个时间点 = 10 步 Euler，每步带 CFG）录成一张图
            static_output = solve_flow_euler(self.flow.decoder, *static_inputs)
        graphs[(batch_size, mel_frame)] = CapturedFlowCudaGraph(
            graph=graph, static_inputs=static_inputs, static_output=static_output)
current_stream.wait_stream(stream)
```

```python
# sglang_omni/models/fun_cosyvoice3/stages.py，`FlowCudaGraphRunner.run`（示意：省略 ndim/shape/dtype 校验与右 padding）
bucket_mel_frame = (
    (actual_mel_frame + FLOW_CUDA_GRAPH_FRAME_BUCKET - 1)
    // FLOW_CUDA_GRAPH_FRAME_BUCKET * FLOW_CUDA_GRAPH_FRAME_BUCKET
)                                                   # 帧数按 16 上取整分桶
captured = self.graphs.get((batch_size, bucket_mel_frame))
if captured is None:
    return None                                     # 未命中，调用方回退 eager solve_flow_euler
for static, value in zip(captured.static_inputs, inputs, strict=True):
    static.copy_(value)                             # 真实 batch 拷进静态输入（右 pad 到 bucket）
captured.graph.replay()
return captured.static_output[..., :actual_mel_frame].clone()
```

**Scheduler level optimization**

CV3 这一类多阶段模型相对于 Dual-AR 结构模型最特别的部分，就是对 scheduler 的精心优化。

AR 部分原生便是流式的 output，因此如何做好 DiT 部分的 chunk inference，调度好三个 module 之间的推理，以及如何在不同 GPU 之间把各个部分打满（目前还没完成），是这一节要解决的问题。

在先前优化的基础上，各个 stage 目前都已经支持 batch inference 了，但是生产环境中我们常常会面临一个延迟的 tradeoff：有的音频长，有的音频短，长度差距太大，我们没法等一整个 batch 都推理完了再统一丢给下一个 module。所以一个 batch 里的请求必然会在不同阶段之间先拆开，再按下一阶段的特点重新组 batch，反复多次，以此打满 GPU 和各个 module 的利用率。

由于 stream 和非 stream 存在较大的区别，我们这里先讲非 stream 的：

offline 推理比较简单，可以总结为“收集-推理-发送”这三个行为，不过三个部分“收集”的方式并不一样。AR 部分直接交给 sglang 的 continuous batching，请求按 decode step 加入和退出 batch，哪条先生成完就先把整条结果发给下一个 stage，不会被同 batch 里的长请求挡住；到了 DiT 这边，vocoder 入口会按时间/数量检测请求，30 ms / 16 条作为基础临界条件，然后按 mel 帧数计 cost，保证总量不超过 8000 帧，随后再按长度做一次自适应分组（组内长度差不超过 384 帧，全局 padding 预算 25%），尽量让长度相近的请求共用一次 Flow solve；HiFT 则在每个 Flow 组内部按 padding 浪费程度再分一次组。

理论上如果生成速度极快，长音频和短音频之间的完成时间差被压缩到 10 ms 这种级别，我们就不需要这种处理了，直接等整个 batch 全部完成再一起交给下一阶段即可，也就是 static batching。static batching 本质上也是我们目前实现的一个子集：当一批请求恰好同时完成、长度也接近时，上面的流程就会退化成它。不过目前我们还远远没有达到这个极限效率，因此一定的工程实现可以极大地降低延迟。

先看 vocoder 入口如何攒 batch。未能进入当前 batch 的消息会按到达顺序插回队头，不会被丢掉：

```python
# sglang_omni/scheduling/streaming_simple_scheduler.py，`collect_new_request_batch`（示意：两段 defer 分支合并成一行，省略 abort/错误与带 timeout 的二次取消息）
deadline = time.monotonic() + self.max_batch_wait_s      # 时间窗口
while len(batch) < self.max_batch_size:                  # 条数上限
    msg = self.get_batch_message()
    if msg.type != "new_request" or self.is_streaming_payload(msg.data):
        deferred.append(msg)                             # 流式请求不参与攒批
        break
    if batch and batch_cost + msg_cost > self.max_batch_cost:
        deferred.append(msg)                             # cost 上限，攒不进去
        break
    batch.append(msg)
self.pending_messages.extendleft(reversed(deferred))     # 按到达顺序插回队头
```

攒到一批之后，再按长度做一次自适应分组。它是先按 mel 帧数排序，然后跑一个 DP。这里有两层目标：外层先尽量减少分组数（从一个组开始递增，找到第一个满足 padding 预算的方案就返回）；组数固定之后，DP 才在组内最小化 $\sum \text{组大小} \times \text{组内最长}$。两层合起来的总目标是把平均 batch size 顶上去。约束是组内的长度差不超过 `flow_merge_max_gap_frames`，并且全局 padding 预算不超过 `flow_merge_pad_budget_percent`：

```python
# sglang_omni/models/fun_cosyvoice3/stages.py，`adaptive_flow_requests_grouping`
ordered_requests = tuple(sorted(requests, key=lambda r: (r.total_mel_frames, r.index)))
...
for group_end in range(suffix_start + 1, group_end_limit + 1):
    longest_frames = ordered_requests[group_end - 1].total_mel_frames
    group_gap_frames = longest_frames - shortest_frames
    if group_gap_frames > flow_merge_max_gap_frames:     # 组内长度差上限
        break
    candidate = ((group_end - suffix_start) * longest_frames + suffix_plan[0], ...)
...
if (plan[0] / non_patching_workload - 1) * 100 <= flow_merge_pad_budget_percent + 1e-9:
    ...                                                  # 全局 padding 预算
```

然后是 streaming。把这种 AR+NAR 结构的 streaming 跑通已经不容易，在此之上的优化更依赖工程实现。

AR 每生成一段 token，我们就会把这一部分发送给 DiT，首次需要 hop + lookahead = 25 + 3 = 28 个 token，之后每 25 个 token 同步一次，并且第一个 chunk 会一并带上 prompt 的 token/mel/speaker embedding，这样 vocoder 不需要等 AR 整条生成完的 payload 就可以接着进行生成。DiT 和 HiFT 在同一个 vocoder stage 里，DiT 做完 chunk-wise 的推理渲染出 mel 谱之后，会在同一个 step 里直接交给 HiFT 增量地出波形，最终得到音频片段返回给用户侧。

对齐关系：DiT 的静态 chunk size 是 50 mel 帧，正好等于 25 个 token 乘上 token2mel 的倍率 2，所以 hop 取 25 个 token 不是随手定的，而是为了让一个 hop 正好落在一个 chunk 边界上。

vocoder 这边给每个请求维护一个状态机：token 够一个因果窗口（offset + hop + 3）就是 `causal_window`，可以出一个 hop；AR 结束之后剩下的部分走 `leftover`；AR 结束但一个 token 都没有则走 `fallback`；其余情况都是 `wait`。DiT 的 chunk inference 靠 chunk-causal mask 保证每一帧只看得到自己所在的 chunk 和之前的 chunk，新来的 token 不会改变已经输出的部分；同时 prompt 会被 pad 到 hop 的整数倍，让第一个 hop 正好对齐 chunk 边界，不必再额外等待若干个 AR token。hop 的长度也是逐步增长的（25 → 50 → 100），首包只需要 28 个 token，开播之后有了余量就换大 hop 减少调用次数。

```python
# sglang_omni/models/fun_cosyvoice3/streaming_vocoder.py，`CosyVoice3StreamState.next_decode`
causal_token_end = self.token_offset + self.hop_len + PRE_LOOKAHEAD_LEN   # 25 + 3
if self.prompt_token is not None and len(self.tokens) >= causal_token_end:
    return "causal_window"
elif self.done and self.tokens:
    return "leftover"
elif self.done:
    return "fallback"
else:
    return "wait"
```

一个 step 里，选定的请求合成一次成批的 Flow 调用；HiFT 同样一个 step 只跑一次——把每条请求的新 mel 帧接到它自己的历史后面，整组共用一次 HiFT 调用：

```python
# sglang_omni/models/fun_cosyvoice3/streaming_vocoder.py，`run_step`（示意：省略 fallback 分支与 hop 推进）
if plan == "leftover":
    mels = self.vocoder.leftover_batch(items)       # 多条请求的 final 合成一次 Flow 调用
    for (_, state), (delta, offset) in zip(
            participants,
            self.hift_step(participants, mels, is_final=True),
            strict=True):
        state.leftover, state.speech_offset = delta, offset
else:
    # 有 prefix 缓存的行走缓存路径，这一 hop 放不下的行在同一 step 内回退整段历史
    mels = self.hop_batch_with_prefix(participants, items)
    for (request_id, state), (delta, offset) in zip(
            participants,
            self.hift_step(participants, mels, is_final=False),
            strict=True):
        state.speech_offset = offset
        if delta.numel() > 0:
            decoded[request_id] = delta
```

```python
# sglang_omni/models/fun_cosyvoice3/streaming_vocoder.py，`hift_step`（整个 step 一次 HiFT 调用）
token_mel_ratio = self.vocoder.flow.token_mel_ratio
for (_, state), mel in zip(participants, mels, strict=True):
    new_frames = mel[:, :, state.token_offset * token_mel_ratio:].detach()
    state.hift_mel = (new_frames if state.hift_mel is None
                      else torch.cat([state.hift_mel.to(new_frames.device), new_frames], dim=2))
    rows.append(HiftStepRow(history=state.hift_mel,
                            emitted_samples=state.speech_offset, is_final=is_final))
return self.vocoder.hift_step(rows)          # 一次调用覆盖整个 step 的所有行
```

这里的收益来自 HiFT 单次调用本身是 launch-bound（H100 上 50 帧 8 ms、600 帧 12 ms），16 行的 step 逐请求跑要花掉约 200 ms，和一个 Flow hop 同量级。代价是要处理窗口化：F0 和 sine source 仍然覆盖每条请求的整段历史（source 相位是累积和、噪声按绝对位置索引），真正被切短的是 decode——只跑"已产出前缀之后的新帧 + 29 帧左感受野上下文"，29 帧是 decode 的因果卷积感受野之和。final 还要右对齐：source 的 STFT 在结尾做 reflect pad，ISTFT 的 overlap-add 归一化与边缘裁剪都依赖信号在哪儿结束，右对齐之后每条 final 的结尾才和自己单独跑一次一致。

每个 step 选择哪些请求一起跑，规则如下：已经开播的请求优先，按播放余量（已产出音频时长 - 已播放时长）从小到大排，余量越小越容易断流；还没开播的请求排在后面，按到达顺序处理。并且只和同类 step 合批（hop 和 leftover 走的是不同的 Flow 路径），每个 step 最多 16 条，多条请求沿序列维打包之后一次 Flow 调用算完。

```python
# sglang_omni/models/fun_cosyvoice3/streaming_vocoder.py，`select_step_participants`（示意：省略两处 assert 与空列表返回）
for request_id, state in self.stream_state_items():
    if state.next_decode() == "wait" or self.is_aborted(request_id):
        continue
    elif state.first_emit_at is None:
        unstarted.append((state.ready_since, request_id, state))
    else:
        # 播放余量 = 已产出音频时长 - 已播放时长，越小越紧急
        playback_slack = state.speech_offset / self.sample_rate - (now - state.first_emit_at)
        started.append((playback_slack, state.ready_since, request_id, state))
ranked = [(rid, s) for _, _, rid, s in sorted(started)]
ranked += [(rid, s) for _, rid, s in sorted(unstarted)]   # 已开播的优先，未开播的按到达顺序
head_decode = ranked[0][1].next_decode()
if head_decode == "fallback":
    return ranked[:1]
peers = [(rid, s) for rid, s in ranked if s.next_decode() == head_decode]  # 只和同类 step 合批
return peers[: self.max_batch_size]
```

整个 step 由 serving loop 驱动：先把 inbox 里的消息读完，再跑一个 step，保证排序的时候看到的都是最新状态。驱动逻辑在框架层，只有几行：

```python
# sglang_omni/scheduling/streaming_vocoder.py，`StreamingVocoderBase.pump_one_step`（示意：省略失败回调）
participants = self.select_step_participants()      # 选出本 step 一起跑的请求
if not participants:
    return None
plan = self.build_step_plan(participants)           # causal_window / leftover / fallback
decoded = self.run_step(participants, plan)         # Flow 与 HiFT 各一次成批调用
for request_id, _ in participants:
    waveform = decoded.get(request_id)
    if waveform is not None and not self.is_aborted(request_id):
        self.mark_stream_emitted(request_id)
        self.outbox.put(self.stream_chunk_message(request_id, waveform))
```

```python
# sglang_omni/scheduling/streaming_simple_scheduler.py，`start`（serving loop，示意：只摘了 has_ready_work 分支）
if self.has_ready_work():
    try:
        msg = self.get_batch_message()               # 有活就先把 inbox 读完
    except queue_mod.Empty:
        self.run_ready_step()                        # 读空了再跑一个 step
        continue
```

最后是资源编排。vocoder 和 AR 在同一个进程、同一张卡上，所以 vocoder 被放到了独立的 CUDA stream 上，kernel 和 host copy 不再排在 AR 的 decode 后面：

如果进一步做分离推理，还有更多可以展开的方向。

```python
# sglang_omni/models/fun_cosyvoice3/stages.py，`CosyVoice3Vocoder.__init__`
# vocoder 与 AR 同进程，放到独立 stream 上，kernel 和 host copy 不再排在 AR 后面
device = next(self.flow.parameters()).device
if device.type == "cuda":
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    self.stream_context = torch.cuda.stream(stream)
else:
    self.stream_context = contextlib.nullcontext()
```

同时同进程的 stage 按列表顺序构建，vocoder 放在 engine 之前，Flow/HiFT/CUDA Graph 先驻留，再由 SGLang 按剩余显存分配 KV pool，避免 OOM 或者 KV pool 被挤小。

顺着再往下想一步，会发现更根本的问题是：KV pool 本身就不该按整张卡来算。`mem_fraction_static` 默认 0.85，在 H100 上打满之后，我们发现 AR 能占用 5.46M tokens / 62.5 GB 的 pool 大小；而这台机器上最多 32 条并发 × 4096 context = 131k tokens，只有 1.5 GB。也就是说那 62.5 GB 里除 1.5 GB 之外**永远不可能被用到**，这部分用不到的显存，应当分给另外两个 module。

改法是让池子跟着实际能跑的请求数走：

```python
# sglang_omni/models/fun_cosyvoice3/engine_builder.py，`FunCosyVoice3EngineBuilder.adjust_overrides`
# Note (Jiaxin Deng): the fraction alone sized a 62 GB pool on an H100 that the
# running requests can never fill; the vocoder shares the GPU and was left about
# 10 GB of dynamic-shape headroom.
if overrides.get("max_total_tokens") is None:
    overrides["max_total_tokens"] = overrides["max_running_requests"] * self.context_length
```

`mem_fraction_static` 仍然当上界用，所以调大 `max_running_requests` 池子会跟着长；用户显式给了 `max_total_tokens` 就以用户为准。

H100 上池子从 62.5 GB 降到 1.5 GB，分配完的剩余显存从 11.2 GB 变成 72.2 GB；buffered 并发 32 的吞吐 +83%。而省下来的这 60 GB 最大的用处，是把**同卡多副本**这条路打开了，下面会讲相关内容。

至于多副本，框架本身已经支持把 vocoder 拆到单独的进程和 GPU 上，或者整条 pipeline 做数据并行。同卡多副本这条路上现在有了实测，而且收益比预期更明显。单副本根本跑不满卡：

| 副本数 | 单副本 req/s | 聚合 req/s | 相对单副本 | SM active |
| --- | --- | --- | --- | --- |
| 1 | 6.72 | 6.72 | — | 31% |
| 2 | 5.38 / 4.82 | 10.19 | +52% | 54% |
| 3 | 4.61 / 4.09 / 3.63 | 12.32 | +83% | 69% |
| 4 | 3.55 / 3.93 / 3.87 / 3.54 | 14.89 | +122% | 79% |

（H200 141 GB，一个 MPS daemon，N 个副本，每个副本各自按并发 16 流式打满，DCGM 100 ms 采样，256 条样本。）

值得注意的是 Tensor Active 只从 7.8% 走到 18.5%，也就是说加副本之后用上了空转的算力，这也解释了为什么聚合吞吐能涨这么多而单副本只是缓慢下降。

**多线程 CPU 处理**

参考音频的编码（ONNX speaker encoder + speech tokenizer）是预处理阶段的主要负载任务，它原本在 lock 里串行执行，配了 `max_concurrency=8` 也并行不起来。改法是把编码移出临界区，只把最后的组装步骤加锁——**锁的范围应该只覆盖真正有共享状态的那部分**，这是这一类优化里最通用的经验。

```python
# sglang_omni/models/fun_cosyvoice3/request_builders.py，`preprocess_cosyvoice3_payload`（示意：省略前段取 context 与锁后的 state 回填）
state = build_cosyvoice3_state(payload)             # 锁外执行
reference_artifact = None
if state.ref_audio is not None:
    # 参考音频编码（speaker encoder + speech tokenizer）移出临界区，多请求并行
    reference_artifact = context.reference_service.get_or_encode(
        state.ref_audio, desc="Fun-CosyVoice3 reference conditioning")
with _PREPROCESSING_FINALIZE_LOCK:                  # 只有最后的组装步骤加锁
    prepared = prepare_cosyvoice3_request(
        model=context.model, tokenizer=context.tokenizer, state=state,
        reference_artifact=reference_artifact, use_mlx=context.use_mlx)
```

```python
# sglang_omni/models/fun_cosyvoice3/config.py，preprocessing stage（示意：省略 process="pipeline"）
StageConfig(
    name="preprocessing",
    factory_path=f"{_PKG}.stages.create_preprocessing_executor",
    factory=FactoryArgs(max_concurrency=8),         # 预处理并发度
    next="tts_engine",
),
```

```python
# sglang_omni/models/fun_cosyvoice3/engine_builder.py
# 两个 ONNX session 各开一个线程池，线程数按宿主机核数封顶
self.onnx_intra_op_threads = max(1, min(int(onnx_intra_op_threads), os.cpu_count() or 1))

# sglang_omni/models/fun_cosyvoice3/utils.py，`SpeechTokenizerV3.__init__`
option.intra_op_num_threads = max(1, int(intra_op_threads))
providers = (
    [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),  # 不对每个新形状穷举 cuDNN 算法
     "CPUExecutionProvider"]
    if device.startswith("cuda") else ["CPUExecutionProvider"]
)
```

再看下一层。并行之后还有一笔开销：ONNX Runtime 的 intra-op 工作线程在空闲时是 spinning 的。speaker encoder 是跑在 CPU 上的 ONNX session，worker 空闲时仍在空转并抢占 CPU，而 serving 也需要一部分 CPU core。

```python
# sglang_omni/models/fun_cosyvoice3/utils.py，`SpeakerEncoder`（示意）
option.add_session_config_entry("session.intra_op.allow_spinning", "0")   # 空闲时禁止 worker 自旋
```

改动只有一行，收益却不小，而且之前 CPU 压力越大，加入这个改动之后收益也就越大：H200 上流式并发 16 吞吐 +32.51%、非流式并发 16 +44.93%；把进程绑到 8 个核上时 +82.88%，16 核的时候 +33.57%。这个梯度正好说明它省的是 CPU 竞争产生的开销。

**BF16 推理**

示例代码如下：

```python
# sglang_omni/models/fun_cosyvoice3/stages.py
AUTOCAST_DTYPES: dict[str, torch.dtype | None] = {
    "float32": None,                                # None 即关闭 autocast
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# sglang_omni/models/fun_cosyvoice3/config.py：DiT 侧 bf16
FunCosyVoice3VocoderFactoryArgs(dtype="bfloat16", ...)
# sglang_omni/models/fun_cosyvoice3/stages.py：HiFT 侧默认 fp32
hift_dtype: str = "float32"
self.hift_autocast_dtype = AUTOCAST_DTYPES[hift_dtype]
```

```python
# sglang_omni/models/fun_cosyvoice3/stages.py，`CosyVoice3Vocoder.token2wav_chunk`（示意：省略三个 int32 长度张量与各处 .to(device)）
with torch.autocast(device_type=current_platform.device_type,
                    dtype=self.autocast_dtype,
                    enabled=self.autocast_dtype is not None):
    tts_mel, _ = native_flow.inference(token=..., prompt_token=..., prompt_feat=...,
                                       embedding=..., streaming=streaming, finalize=finalize)

# `mel2wav_batch`：HiFT 使用独立的精度域
hift_autocast = torch.autocast(device_type=current_platform.device_type,
                               dtype=self.hift_autocast_dtype,
                               enabled=self.hift_autocast_dtype is not None)
with hift_autocast:
    wav, _ = self.hift.inference(speech_feat=padded, finalize=True)
```

**消除 host sync 与常驻化**

这里面收益最大的一条，反而是最不起眼的一条：AR 每步都要把采样出来的 token id 读回 host。原先是直接 `token_ids.tolist()`，这是一次同步的 D2H 拷贝，而 AR 跑在 legacy default stream 上。

问题在于：一个线程阻塞在这种同步等待上时，同进程其它线程几乎无法 launch kernel，即使它们跑在自己的 stream 上。微基准里，等待期间另一个线程每秒只能 launch 约 225 个 kernel，换成 pinned buffer + event 等待之后是约 27,900 个，差两个数量级。

对 CV3 尤其致命：vocoder 和 AR 同进程，而 vocoder 的 kernel 是一个一个从 Python launch 出去的。于是 decode 一跑 vocoder 基本停摆，流式下首包就得等整句的 decode 走完。

```python
# sglang_omni/models/fun_cosyvoice3/model_runner.py，`collect_tokens`（示意）
# note (Yucheng Hu): a pageable D2H on the default stream stalls other threads'
# kernel launches while it waits, an event wait does not.
self.stage_token_ids(result, token_ids)              # 一次异步拷进 pinned ping-pong buffer
token_ids_cpu = self.redraw_repeated_tokens(
    result, forward_batch, requests,
    self.resolve_host_token_ids(result).tolist(),     # 等 event，而不是等拷贝本身
)
```

改完之后，原先每步三次同步读回（`collect_tokens`、output processor、sglang 的 batch result 转换）合并成异步 copy + 单次等待。Qwen3-Omni 的 talker 和 MOSS-TTS 早就引入了这个方法，CV3 这次与它们对齐。RTX PRO 4500 上并发 4 流式首包 -200 ms、吞吐 +12.6%，并发 16 +18.1%；H100 上并发 16 吞吐 +9.2%、首包 -21.9%。

示例代码如下：

```python
# sglang_omni/models/fun_cosyvoice3/stages.py，`patch_chunk_mask`
empty_rows = ~masks.any(dim=-1, keepdim=True)        # any 而非 sum：求和会先把 bool mask cast 成 int64
masks.masked_fill_(empty_rows, True)                 # 全在 device 上完成，不再 host sync
return masks
```

```python
# sglang_omni/models/fun_cosyvoice3/stages.py，`patch_causal_conv_cache`
def forward(self, x, cache=torch.zeros(0, 0, 0)):
    if cache.size(2) == 0:
        cache = x.new_zeros(x.shape[0], x.shape[1], self.causal_padding)  # 直接在 device 上开
    return original_forward(self, x, cache)
CausalConv1d.forward = forward                       # 每次 HiFT 调用省掉一次 host sync
```

```python
# sglang_omni/models/fun_cosyvoice3/stages.py，`keep_hift_constants_on_device`
hift.stft_window = hift.stft_window.to(device)       # 普通属性不是 buffer，hift.to(device) 带不动它们
sine_gen = hift.m_source.l_sin_gen
sine_gen.rand_ini = sine_gen.rand_ini.to(device)
sine_gen.sine_waves = sine_gen.sine_waves.to(device)
hift.f0_predictor.to(torch.float64)                  # 每次调用都 cast 一次
```

**流式 Flow 的 prefix KV cache**

前面讲 Flash Attention 时提过：AR 那一侧的 prefix caching 直接复用 SGLang，DiT 这一侧需要单独实现。这一项就是该实现的落地，也是流式路径上最大的一笔收益。

原因在于：chunk-causal mask 只规定了一帧能看到自己所在的 chunk 和之前的 chunk，并没有减少实际计算量。所以**hop k 仍然要把前缀所有帧的 10 步 Euler × 2 份 CFG 重跑一遍**，才能算出新的那几帧。一条 300 token 的请求，累计 Flow 工作量大约是 buffered 路径的 2.7 倍，而且这个代价随着流式长度线性增长——首包延迟和 c=16~32 的吞吐主要都消耗在这里。

但前缀是因果的：它在每个 Euler 求解器的 step 中、每层的 K/V 只取决于更早那些 hop 已经解出来的帧，是个纯函数。那就把它存下来，每个 hop 只解新帧，前缀通过 FA3 的 paged KV 接口去 attend。

```python
# sglang_omni/models/fun_cosyvoice3/prefix_cache.py，`PrefixKVPool`（示意）
# keys[euler_step][layer] 各是一张 (pages, 1, head_num, head_dim)，按 BLOCK_FRAMES 分块
block_count = max(int(capacity_frames) // BLOCK_FRAMES, 0)
shape = ((block_count + 1) * BLOCK_FRAMES, FA3_PAGE_SIZE, head_num, head_dim)
self.keys = [[torch.empty(shape, device=device, dtype=dtype) for _ in range(layer_num)]
             for _ in range(euler_steps)]
self.values = [[torch.empty(shape, device=device, dtype=dtype) for _ in range(layer_num)]
               for _ in range(euler_steps)]
self.free_blocks = list(range(block_count))
```

```python
# sglang_omni/models/fun_cosyvoice3/prefix_cache.py，`PrefixRowAttention`（示意）
# query 是每行这一 hop 的新帧，key 是缓存的前缀加上这些新帧，全部通过 pool 的 page 寻址
for half in range(2):                           # CFG 两份：条件行与其孪生行
    for index in range(row_count):
        twin_row = half * row_count + index
        query_lengths.append(new_frames[twin_row])
        query_prefixes.append(prefix_frames[twin_row])
```

池子由 `flow_prefix_cache_gb` 定大小（默认 24 GB，同时不超过整卡的 1/4，0 即关闭），并且**要在 AR engine 去量 KV pool 之前建好**，否则显存先被 AR 占走——这一点和上文里 KV pool 封顶那件事是配套的。池子当中装不下的行，会在同一个 step 里回退到整段历史的路径，之后有余量了还能再进来。

- 单个 16 行 step 的四个 hop：整段历史 897 ms → 缓存 475 ms（-47%）；第一个 hop 因为还要写入池子，比原来略慢。
- H200 流式：并发 16 吞吐 +26%、首包 -26%、延迟 p95 -28%；并发 32 吞吐 +52%。

### Kernel level method

**torch.compile**

自从 torch 2.0 引入这个功能之后，它已经成为最直接的优化手段，加上 `@torch.compile` 即可获得加速。

compile 的本质是将原本的 model 类编译为一个 torch._dynamo 类，把一个动态图编译为一个静态图，进行一系列的算子融合，从而实现加速。而且热更新模型的权重并不会破坏 compile 后的图，对于 rollout 来说也属于一种很有效的优化方式。

示例代码如下：

CV3 这边有两代实现。第一代是编译上游的 DiT：

```python
# sglang_omni/models/fun_cosyvoice3/stages.py，`compile_dit_backbone`（示意：省略 libdevice 打补丁与 rotary 改写）
inductor_config.fx_graph_cache = True               # 跨进程复用编译产物（源码里 torch._inductor.config 的别名）
dynamo.config.cache_size_limit = 1024
dynamo.config.accumulated_cache_size_limit = 1024
# 注意这里没有用 dynamic=True：它会让 LayerNorm 的 eps 也变成符号，
# Inductor 留不住这个符号，每次都要重新编译
estimator.forward = torch.compile(original_forward, options=dict(DIT_INDUCTOR_OPTIONS))
warmup_dtype = autocast_dtype or next(flow.parameters()).dtype
with torch.inference_mode():
    for streaming in (False, True):                 # causal / full 都要 warm 到
        with torch.autocast(device_type=current_platform.device_type,
                            dtype=autocast_dtype, enabled=autocast_dtype is not None):
            estimator(noisy_mel, mel_mask, token_condition, flow_time,
                      speaker_embedding, prompt_mel, streaming=streaming)
```

warmup 这一步不能省，而且必须在和 serving 相同的 `inference_mode` + autocast 精度下跑，否则 guard 会失效，第一个真实请求会触发重编译，并承担约 100 s 的延迟。这条路径默认开启，H200 上中位吞吐并发 1 时 +24.6%、并发 16 时 +59.0%，WER 不变。

第二代是编译我们自研的打包 DiT（`PackedDiT`），也就是流式路径跑的那个。它只有一种 `forward` 方式，要求 `fullgraph=True`、不允许出现 graph break；causal（流式 hop）与 full（非流式 final）的差别表现在 attention 实现上：

```python
# sglang_omni/models/fun_cosyvoice3/packed_dit.py，`PackedDiT.compile`
if not self.is_ragged or dtype not in FA3_DTYPES:   # 没有 FA3 或不是半精度：不编译，走 eager
    return False
# note(ratish): 不用 dynamic=True——它会把 head 数和 head_dim 变成符号，
# 于是 reshape 进 FA3 布局时每个 block 都要再拷一遍 query 和 key
self.forward = torch.compile(
    self.forward, backend="inductor", fullgraph=True,
    options=dict(PACKED_INDUCTOR_OPTIONS))
self.is_compiled = True
return True
```

这里同样没有用 `dynamic=True`：序列长度这类变化是编译前在 `row_attention()` 里用 `dynamo.maybe_mark_dynamic` 逐张量提示的，而不是让整张图符号化。

```python
# sglang_omni/models/fun_cosyvoice3/streaming_vocoder.py，`warmup_packed_dit_compile`（示意：省略计时与 prefix warmup）
packed_estimator = self.vocoder.flow.packed_estimator
if not isinstance(packed_estimator, PackedDiT):
    raise RuntimeError(...)                         # 拿不到 PackedDiT 直接失败，没有静默回退
if not packed_estimator.compile(self.vocoder.autocast_dtype):
    return                                          # 编译成本在启动期支付
hop_tokens = self.token_hop_len + PRE_LOOKAHEAD_LEN
first = [self.make_warmup_flow_input(hop_tokens)]
# 第二批的行数与长度都不同，让 serving 会变的形状在启动期就变成符号，而不是等第一个请求
second = [self.make_warmup_flow_input(hop_tokens),
          self.make_warmup_flow_input(hop_tokens + self.token_hop_len)]
with self.vocoder.stream_context:
    for items in (first, second):
        self.vocoder.hop_batch(items)
        self.vocoder.leftover_batch(items)
```

**TRT Engine**

TRT 是 NVIDIA GPU 专用的一种加速手段：先把模型转成 ONNX，再用 TensorRT 导出成一个 engine。

TRT 这条路我们已经实现了（opt-in，默认关闭），但目前**并没有比 torch.compile 更快，而且差距明显**。单并发下它确实很强（RTF 约 0.09~0.10），但到了高并发反而不如 compile 了。

不过这一套的工程细节还是有价值的，尤其是它和 torch.compile 互斥、按 profile 边界回退这两点，所以代码留在下面供参考。

示例代码如下：

```python
# sglang_omni/models/fun_cosyvoice3/flow_estimator_trt.py，`convert_onnx_to_trt`（示意：省略 parser 失败抛错、build 结果检查与 .tmp + os.replace 原子落盘）
builder = trt.Builder(engine_logger)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)) if strongly_typed \
    else builder.create_network(0)
parser = trt.OnnxParser(network, engine_logger)
parser.parse(open(onnx_path, "rb").read())
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 33)
if not strongly_typed:
    try_enable_fp16_tactics(config, trt)            # fp32 ONNX 下尝试打开 FP16 tactic
profile = builder.create_optimization_profile()
mins, opts, maxs = dynamic_shapes(min_time), dynamic_shapes(opt_time), dynamic_shapes(max_time)
for name in mins:                                   # CFG batch 固定为 2，只对 x/mask/mu/cond 的时间维做动态 profile
    profile.set_shape(name, mins[name], opts[name], maxs[name])
config.add_optimization_profile(profile)
engine_bytes = builder.build_serialized_network(network, config)  # 序列化为 plan 缓存到磁盘
```

运行时有两个必须处理的约束。第一，engine 的 profile 是 CFG=2（对齐官方 ONNX），而 packed 的 CFG batch 是 2N，所以要按"请求对"把它切成 engine 能够接受的形状，再把结果拼回 `[cond..., uncond...]` 的布局。第二，engine 跑在自己的 stream 上，前后各做一次 `wait_stream`，数据依赖才不会断：

```python
# sglang_omni/models/fun_cosyvoice3/flow_estimator_trt.py，`execute_flow_estimator` / `enqueue_once`（示意：省略 profile 边界检查与 try/finally 释放 estimator）
# packed CFG 布局为 [cond_0..B, uncond_0..B]，按请求对切成 engine 能吃的 CFG=2 小块
request_batch = cfg_batch // 2
max_requests = max(1, max_batch // 2)
for start in range(0, request_batch, max_requests):
    end = min(start + max_requests, request_batch)
    chunks = take_cfg_pairs((x, mask, mu, t, spks, cond), start, end, request_batch)
    y = run_estimator(estimator, *chunks)
    n = end - start
    out[start:end] = y[:n]
    out[request_batch + start: request_batch + end] = y[n:]

# ...，`enqueue_once`：engine 在自己的 stream 上异步执行
[context, stream], trt_engine = estimator.acquire_estimator()
stream.wait_stream(caller_stream)
with torch.cuda.stream(stream):
    for name, shape in shapes.items():
        context.set_input_shape(name, shape)
    for index, tensor in enumerate((*inputs, out)):
        context.set_tensor_address(trt_engine.get_tensor_name(index), tensor.data_ptr())
    context.execute_async_v3(stream.cuda_stream)
caller_stream.wait_stream(stream)
```

```python
# sglang_omni/models/fun_cosyvoice3/flow_estimator_trt.py，`FlowEstimatorTRTModule.forward`（示意：省略 fallback 为空时的报错）
frames = int(x.shape[2])
if frames < self.min_time or frames > self.max_time:
    return self.fallback(x, mask, mu, t, spks, cond, streaming=streaming)  # profile 外回退 PyTorch DiT
return execute_flow_estimator(self.trt, x, mask, mu, t, spks, cond)

# sglang_omni/models/fun_cosyvoice3/stages.py，`attach_flow_estimator_trt`（示意）
fallback = flow.decoder.estimator
wrapper = build_flow_estimator_trt(onnx_path, device, fallback=fallback, wrap_module=True)
del flow.decoder.estimator                          # 先删掉 nn.Module child，再挂 TRT wrapper
flow.decoder.estimator = wrapper
flow.packed_estimator = None                        # TRT 与 PackedDiT 打包路径互斥，一并摘掉
```

**Flash Attention**

attention 这边实际在优化三件事：

1. **前缀缓存（prefix caching）**：多个请求如果共享相同的前缀（比如系统提示、或者同一段参考音频），那段 prefix 的 KV 只需要算一次，后面的请求直接复用。
2. **RadixAttention**：SGLang 的核心技术，用一棵基数树（radix tree）管理所有请求的 KV，在 radix tree 上做最长前缀匹配，让上面那个复用自动发生，不用调用方自己管理。
3. **FlashAttention**：算子层面的改进。它减少 HBM 读写，并且不把 $N \times N$ 的注意力矩阵拉出来，而是分块在 SRAM 里算完。这个和前缀缓存是正交的。

CV3 的 AR 部分直接跑在 SGLang 上，所以第 1、2 项它是自动继承的，不需要我们做什么。真正需要我们动手的是 DiT 那一侧，也就是 `packed_dit.py` 里的三个 attention 实现。

沿序列维打包之后，attention 最朴素的做法是把每行重新 scatter 回 `(rows, width, C)` 的 padded 布局再跑 SDPA，也就是 `RowAttention` 那条路：scatter → SDPA → gather。问题不在 `F.scaled_dot_product_attention` 本身（它会自动分发到 flash / mem-efficient / math 后端），而在于两点：一是传了任意 `attn_mask` 之后往往用不上 flash 后端，二是 padded 布局让**每一行都被算到最长那条的长度**。短请求会跟着最长请求一起计算，耗时会被拉到最长请求。

改法是直接用 FA3 的 varlen / paged KV 接口读 ragged 序列，不再 pad：

```python
# sglang_omni/models/fun_cosyvoice3/packed_dit.py，`RaggedRowAttention`（FA3 varlen / paged KV，示意）
segment_rows, segment_ends, offsets = chunk_segments(rows.lengths, chunk_size)  # 每个 (row, chunk) 一段 query
self.cache_seqlens = torch.tensor(segment_ends, dtype=torch.int32, device=device)
self.cu_seqlens_q = torch.tensor(offsets, dtype=torch.int32, device=device)
self.max_seqlen_q = max(end - start for start, end in pairwise(offsets))
starts = rows.starts_host[list(segment_rows)].to(device)
page = torch.arange(max(segment_ends), dtype=torch.int32, device=device)
self.page_table = torch.where(                      # page id 指向 packed 序列里该行的真实位置
    page.unsqueeze(0) < self.cache_seqlens.unsqueeze(1),
    starts.unsqueeze(1) + page.unsqueeze(0),
    0,
)

def __call__(self, query, key, value):
    page_shape = (-1, FA3_PAGE_SIZE, self.heads, self.head_dim)
    if torch.compiler.is_compiling():
        fa3 = packed_fa3     # custom op：给 Dynamo 一个形状语义明确的黑盒节点
    else:
        fa3 = ragged_fa3     # eager 直接调，省掉每个 block 一次 custom op 派发
    out = fa3(
        query[0].reshape(-1, self.heads, self.head_dim),
        key[0].reshape(page_shape),
        value[0].reshape(page_shape),
        self.cache_seqlens,
        self.page_table,
        self.cu_seqlens_q,
        self.max_seqlen_q,
    )
    return out.reshape(1, -1, self.heads * self.head_dim)
```

这里有一个实现上的简化：`FA3_PAGE_SIZE = 1`，一个 page 只装一个 token，于是 `page_table` 退化成一张"逻辑位置 → packed 序列里物理位置"的映射表。KV 直接在 packed 序列上原地读，既不用 pad 也不用拷贝。每段 query 由 `cu_seqlens_q` 描述，它能看到的 KV 长度由 `cache_seqlens` 描述——流式下按 chunk 切段，用 KV 长度来表达 chunk causal，所以调用时传的是 `causal=False`。

另外它被注册成了一个 custom op，并声明输出形状与 `q` 相同（等价于 `register_fake`）。这样 Dynamo 不会探测 FA3 内部，而是把它当成一个有明确形状语义的黑盒节点，上一节 `fullgraph=True` 能成立正是靠这一点：

```python
# sglang_omni/models/fun_cosyvoice3/packed_dit.py，`packed_fa3` / `PackedDiT.row_attention`（示意）
# 包成 custom op，供 torch.compile 作为图边界；out_shape="q" 即输出形状与 q 相同，等价于 register_fake
packed_fa3 = register_custom_op(
    ragged_fa3, op_name="fun_cosyvoice3_packed_fa3", mutates_args=[], out_shape="q")

if not self.is_ragged or dtype not in FA3_DTYPES:   # 没有 FA3 或不是半精度：padded SDPA
    return RowAttention(rows, chunk_size=self.chunk_size if streaming else None, heads=heads)
elif streaming:                                    # 流式：按 chunk 切段的 ragged varlen
    return RaggedRowAttention(rows, chunk_size=self.chunk_size, heads=heads, head_dim=head_dim)
return WholeRowAttention(rows, heads=heads, head_dim=head_dim, max_frames=rows.width)  # 非流式：整行双向
```

所以这一项是三路分发：没有 FA3 或不是半精度时退回上面那套 padded SDPA；流式走按 chunk 切段的 ragged 方案；非流式走整行双向的 varlen 方案。因此不会让某些硬件上的正确性出问题。

**手写算子改写**

前面几项都是换一个现成的实现，这一项是直接改算子本身。

目标很明确：**减少 elementwise kernel 的数量，以及减少显存读写**。在 DiT 这种结构上收益不小——我们在另一个同架构模型 auk 上测过，大 batch 时有 57~71% 的 kernel 时间落在 elementwise 上。这些 kernel 单个看都是几微秒、算力几乎为零，但每发一次都有 launch 开销，而且每个都要把数据从显存读进来再写回去，瓶颈在带宽。

具体到 CV3 这里是两处。第一处是 **QKV 投影融合**：q / k / v 原本是三次独立的 GEMM，现在在 `__init__` 里把每个 block 的 `to_q` / `to_k` / `to_v` 的 weight 和 bias 拼成一份 `qkv_weight` / `qkv_bias`，一次 `F.linear` 生成三个结果：

```python
# sglang_omni/models/fun_cosyvoice3/packed_dit.py，`PackedDiT.attend`
# 一次 GEMM 出 q/k/v——权重已按 block 拼好
query, key, value = F.linear(x, qkv_weight, qkv_bias).chunk(3, dim=-1)
if torch.compiler.is_compiling():
    query = rotated(query, *rope)                    # 编译期：where 版，一个 kernel
    key = rotated(key, *rope)
else:
    rotate_in_place(query, *rope)                    # eager：原地写回，不新开 tensor
    rotate_in_place(key, *rope)
out = attention(query, key, value).to(query.dtype)
return attn.to_out[1](attn.to_out[0](out))
```

融合之后，原先"autocast 下 `to_q` / `to_k` / `to_v` 各自把上游 fp32 输出 cast 一遍，同一个 tensor 被转换三次"的问题也一并消失了——现在只有一次输入、一次 cast。

第二处是 RoPE 改成原地计算。原来要构造新 tensor，现在只取 q/k 里 rotary 的那一段切片，算完用 `copy_` 直接写回原 tensor，不额外分配显存；cos / sin 也只在每步生成一份，q 和 k 共用。

```python
# sglang_omni/models/fun_cosyvoice3/packed_dit.py，`rotate_in_place` / `rotated` / `PackedDiT.rope_angles`
def rotate_in_place(x, cos, sin):
    rotary = x[..., : cos.shape[-1]]                 # 只取 rotary 切片，其余不拷
    half = torch.stack((-rotary[..., 1::2], rotary[..., ::2]), dim=-1).flatten(-2)
    rotary.copy_(rotary * cos + half * sin)          # 直接写回 q/k，不新开 tensor

def rotated(x, cos, sin):                            # 编译期的等价实现：一次 where
    rotary_dims, width = cos.shape[-1], x.shape[-1]
    cos = F.pad(cos, (0, width - rotary_dims))
    sin = F.pad(sin, (0, width - rotary_dims))
    half = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)
    turned = (x * cos + half * sin).to(x.dtype)
    is_rotary = torch.arange(width, device=x.device) < rotary_dims
    return torch.where(is_rotary, turned, x)

def rope_angles(self, frame_count):                  # 每步只生成一份角度，q 和 k 共用
    angles, scale = self.dit.rotary_embed.forward_from_seq_len(frame_count)
    assert not isinstance(scale, torch.Tensor), "the DiT's RoPE has no xpos scale"
    return angles
```

`rope_angles` 只按 `frame_count` 返回角度，不再像早期版本那样在方法里就 `freqs[:, rows.positions]` 切好——位置切片移到了 `solve_flow_euler_packed` 里对 twin rows 做。

**按 Euler steps 提前计算 AdaLN 调制**

前面几项改的是计算方式，这一项讨论的是这些计算是否每次都需要做。

DiT 的每个 block 以及最后一层都有一份 AdaLN 调制：把时间步 embedding 过一层 linear，得到若干个缩放/偏移系数（每个 block 6 个，最后一层 2 个）。而 CV3 用的是**固定的 10 步 cosine Euler schedule**——这 10 个时间步是常量，于是 `time_embed` 的输出是常量，每个 block 那层 linear 的输出也就是常量。可是原来的写法里，这些在每一个 DiT step、每一条请求上都重算了一遍。

既然是常量，就在加载之后按步算好存起来，每条请求直接取：

```python
# sglang_omni/models/fun_cosyvoice3/packed_dit.py，`PackedDiT.build_modulations`（示意）
# 每个 block / 最后一层在每个 Euler 步的调制只读该步的 time，而所有 solve 走的又是同一套步
for t, _ in euler_step_times(time_span):
    flow_time[:] = t
    time_embedding = dit.time_embed(flow_time)
    block_modulations.append(
        tuple(torch.nn.Parameter(block.attn_norm.linear(block.attn_norm.silu(time_embedding)),
                                 requires_grad=False)
              for block in dit.transformer_blocks))
    final_modulations.append(
        torch.nn.Parameter(dit.norm_out.linear(dit.norm_out.silu(time_embedding)),
                           requires_grad=False))
self.block_modulations = tuple(block_modulations)
self.final_modulations = tuple(final_modulations)
```

还有一个实现细节：每个 block 存成一份独立的 `Parameter`，而不是把所有步堆成一张大表。堆表之后表的宽度会变成 Inductor 眼里的符号，按步拆开则每份形状固定，编译出来简单得多；`qkv_weights` 是同样的考虑。

这份缓存在 PackedDiT、prefix KV 路径和整段历史路径上都能用，CUDA Graph 的 capture / replay 也兼容，图里本来就是固定的那几步。fp32 的 mel 输出与原实现 bit-identical；H200 上并发 8 吞吐 +3.99%、TTFP p95 -8.45%。代价是启动多花几秒，计算表本身只有几 MB。

更一般地说，这一项的思路是：先分清哪些计算是"每步都一样"的、哪些是 request specific 的，前者能提前算就提前算。
