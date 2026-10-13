*本文AI含量为0，请放心食用。

# 以 CosyVoice 为代表的 AR + DiT + Vocoder 生成模型优化思路

过去几个月，我们观察到相当量的 TTS 模型采用了 AR + DiT + Vocoder 架构，这些模型通常还会额外带一个 speaker encoder 用来做音色克隆，所以严格说是四个模型串起来的一条链路。为了和社区的说法保持一致，本文仍然沿用 AR + DiT + Vocoder 这个简称。DiT 的加入使得这些模型的音质相比不采用 DiT 方案的模型有了显著提升，但是这带来一些新的优化设计挑战。我们这里总结分享 SGLang Omni 对此类模型的优化思路，主要是 [issue 1652](https://github.com/sgl-project/sglang-omni/issues/1652) 所 track 的各阶段优化，同时我们目前仍旧在进行对于 CosyVoice 通过 Runtime Optimization Agent 进行的进一步优化。总结思路的同时，我们也会把这套方法论扩展到更多 AR + DiT + Vocoder 架构的模型上，譬如 [dots.tts](https://github.com/studio-dots-ai/dots.tts) 和 [Minimax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3)。

## 算法层分析

这一类的 TTS 模型的主要方案是通过 AR 部分生成 audio token，随后使用扩散部分（以DiT结构为主）将 audio token 渲染为 mel 谱，而后则是经典的 vocoder 把 mel 转为波形的过程。

- **AR 部分**：骨干就是 Qwen2-0.5B，词表是 6561 个 speech token 外加 200 个控制 token（`sglang_model.py`），帧率是 25 Hz，也就是1秒音频对应 25 个 token。
- **Speaker encoder**：参考音频编码出一个 192 维的 speaker embedding，用来控制音色。严格来说它是这条链路上第四个模型，也是预处理阶段的heavy work（见后文多线程 CPU 处理）。
- **Flow / DiT**：输入 80 通道的 mel，token 到 mel 的比例是 2，所以 mel 是 50 Hz。DiT 的 hidden size 是 1024，静态 chunk size 是 50 mel 帧，正好等于一个 25 token 的 hop（这个对齐关系见后文 Scheduler 一节）。
- **ODE求解器**：10步的Euler，但是有cfg的缘故所以实际上要翻个倍的前向次数。
- **HiFT**：基于 ISTFT 的神经声码器。mel 帧到采样点的倍率是 \(8 \times 5 \times 3 \times 4 = 480\)，配上 24 kHz 采样率，正好 50 Hz 的 mel。

所以一条 5 秒的音频，整条链路是：

```
125 个 token → 250 帧 × 80 通道 mel → 12 万个采样点
```

从算法设计上来说，这种 multi-stage 的方案通过 diffusion 提升了不少音质，而Cosyvoice这种模型在引入了speaker encoder的情况下也能把音色相似度做的很好。但是多阶段生成在infra层面给我们带来了一系列挑战：

- **AR 只有一个 decode step，DiT 有 10 步 × CFG 2 次。** 前者的瓶颈和 LLM 一样是 memory-bound 的 decode，后者每一步都是一次完整的 DiT 前向，走的是完全不同的优化路径。
- **DiT 是 chunk-causal 的，而且带 lookahead。** 每一帧只能看到自己所在的 chunk 和之前的 chunk，但每个 chunk 还要额外需要 3 个 token 的 lookahead 才能开始算。这直接影响了流式调度的切分策略。
- **三个阶段的"长度"含义都不一样。** AR 按 token 数算，DiT 按 mel 帧数算，HiFT 按采样点数算，三者之间差着固定的倍数，但 batch 起来要 pad 的维度各不相同。

## Runtime Profiling

兵来将挡，水来土掩，为了优化好 CosyVoice 这个模型，我们讨论出了一套 runtime profiling 的方法论，同样给 agent 配套了相关的探测 skill 和优化 skill 等工具，具体见 [PR 1850](https://github.com/sgl-project/sglang-omni/pull/1850)，感兴趣的读者可以直接看那边。

AI Infra 哲学三大问题：1. 哪里速度慢，拖慢了整体的效率？ 2. 这个优化到底是不是在拿效果换速度？ 3. 这个优化它是否切实产生了效益？

而 runtime profiling 回答的是第一个问题：通过给这条链路上多个阶段分别测速，哪一部分拖慢了生成速度便显而易见了。第二个问题交给质量指标，第三个问题则要靠严格的 A/B 才得以判断，两者都放在这一节里讲。而这套方法在往后的优化过程中可以反复使用，因此把它固定成一套方法论、并交给 agent 去埋点执行，从长远维护的角度来看收益很大。

通过给这里多个阶段的模型进行阶段性的测速，哪个部分拖慢了生成的速度便显而易见了，而在往后的优化过程中，这个方案我们可以反复使用，因此固定出一套方法论从长远维护的角度来看，是很有收益的。同样，这套高度重复的工作也可以交给agent进行埋点，从而极大的节省人类的精力。

### 看什么指标

TTS 的效果指标和 LLM 不一样，我们主要看 WER：把生成的音频重新过一遍 ASR 转成文字，再和参考文本比对。也是最能反映"音质有没有被优化搞坏"的指标（我们的 baseline 数据就是用 SeedTTS 这套评测跑的）。此外还会看说话人相似度（音色像不像参考音频）。

工程侧则看 RTF，尤其是目标并发下的 p95 / p99 能不能压到 1 以下（后文调度器一节会详细讲一下）。

而在优化过程中常常会出现由 AI 挖掘出来的伪需求：这个优化引入之后其实没有多大变化，甚至落在噪声区间内。所以我们要构建一套严格的 AB Test 去衡量最后的效益，也就是下面的第四层。

### 开始之前

先固定两样东西。

- **Workload**：短句和长句、离线和流式、有无参考音频、并发数，这些形状下的瓶颈可能完全不同，要分别测量。
- **环境**：记录代码 commit、checkpoint 和数据集的 revision（当然目前固定为seed）、GPU 型号、PyTorch 和 SGLang 版本。共享机器上还要给 server 和压测客户端绑定互不重叠的 CPU 核，确认测量期间 GPU 没有被别的任务占用，而且要经过真实的测试得到应用场景的状态。

## 优化分解

一切的推理加速优化都可以分为两个层次：engineering level 和 kernel level。

判断依据的话，按照“是否改变单个算子的计算方式”进行分类。

To be honest, 目前SGLang Omni对于大部分模型的优化都还停留在engineering level，而且我们也不敢保证在这一层已经优化得很好了。

这里说的"算子"是指一次实际的计算，比如一次 matmul、一次 attention、一次 conv。改变它的**执行形态**（换一个实现、融合掉、编译掉）算 kernel level，剩下的算 engineering level。

在此发表一个暴论，在工程层面没有优化完全的情况下，任何kernel level的收益都是缺乏置信度的。

### Engineering level method

常见的工程层的优化，一般包括打高batch，优化流水线里面CPU和GPU的负载，并行处理音频等，而CosyVoice引入了一项更重要的工程措施：优化调度层。对于多阶段的推理，如何编排好每个请求，让它们既可以享受到加速，同时不会出现乱窜的情况，变成了本次优化过程中的一大挑战。

由于AR部分的推理加速方案几乎可以完全照抄主流LLM推理引擎的方式，因此本文不会用过多的篇幅讲解相关内容，但是这并不代表AR部分不重要，只是由于方案成熟，我们采用的优化手段可能并不会产生更多的超额效益。

#### 1. Batch inference

几乎所有的模型，首先想到的卓有成效的优化手段都是batch inference。而batch inference的核心思想就是按照矩阵乘的性质，把一组target全部padding到一个M*N大小的矩阵进行推理，而后对里面每一条都进行裁切，从而把吞吐提上去。这里要注意它提升的是**吞吐**而不是单请求的延迟：单次前向里多算几条几乎不增加耗时，但单条请求该等的计算一步没少，甚至可能因为要凑批而多等一会儿。

示例代码如下：

```python
# sglang_omni/models/fun_cosyvoice3/stages.py，`CosyVoice3Vocoder.decode_batch`
flow_groups = adaptive_flow_requests_grouping(
    prepared,
    flow_merge_max_gap_frames=384,        # 组内最大长度差
    flow_merge_pad_budget_percent=25.0,   # 全局 padding 预算
)
for flow_group in flow_groups:
    # 一次调用求解整组：padding / masking / CFM 解码都在内部处理
    mel_list = self.flow.inference([r.flow_input for r in flow_group])
    # HiFT 独立分组：右 padding 后一次前向，再按真实长度切回
    for group in hift_grouping(mel_list, self.hift_max_padding_waste):
        wavs = self.mel2wav_batch([mel for _, mel in group])
```

流式场景则不一样。每个 hop 都很短，如果每条请求单独前向一次，launch 开销反而成了主导。这里的做法是把多条请求的行沿着**序列维**拼成 `(1, total, C)`：用一个 `row_ids`（每个 token 属于哪一行）加一个 `positions`（它在行内的位置）来描述打包关系，attention 用 mask 限制在各自行内，所以不同请求之间不会互相 attend。

```python
# sglang_omni/models/fun_cosyvoice3/packed_dit.py，`pack_rows` / `gather_rows` / `scatter_rows`
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

顺带说一下 CFG。每一步 Euler 都需要算两次前向，一次带条件、一次不带，所以 batch 维度是 2N。打包的时候把不带条件的那一份当成"另一行"打进同一个 batch，一次 DiT 调用就能把两个分支一起算完：

```python
# sglang_omni/models/fun_cosyvoice3/packed_dit.py，`solve_flow_euler_packed`
twin_rows = pack_rows(rows.lengths * 2, noise.device)  # CFG：条件行 + 一份无条件孪生行，共用一次 DiT 调用
attention = estimator.row_attention(twin_rows, streaming=streaming, dtype=spks.dtype)
```

#### 2. CUDA Graph capture

DiT 的 Euler solve 是典型的 launch-bound：10 步 Euler，每步还要跑一次 CFG，拆出来是一大串几微秒级的小kernel，CPU launch的速度跟不上，GPU 在 kernel 之间空转。CUDA 的图捕获把整串 kernel 录下来，之后一次 `replay` 就把整串给send出去，消除逐个 launch 的间隙。它 replay 的还是同一批 kernel，很适合当作一个优先处理的 low-hanging fruit。

graph 是静态的，输入输出的地址和形状在录制时就固定了，代价就是不够灵活，需要提前把形状定好：

- **按白名单捕获**：默认 57 个 `(B, T)` 组合（`config.py` 的 `FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES`），每个形状录一张图。
- **帧数按 16 上取整分桶**。分桶越粗，padding 浪费越多但录的图越少；越细则相反。多出来的部分右 pad，靠 `mel_mask` 屏蔽。
- **所有形状共用一个 memory pool**，否则 57 张图各占一份显存会很浪费。
- **没命中就回退 eager**。所以正确性永远不依赖 graph 是否命中。

还有一点要说明：**这套 graph 只服务非流式的 buffered 路径**（`stages.py` 里 `streaming or not finalize` 时直接走 eager），流式路径的加速来自沿序列维打包和 PackedDiT 的 compile，不是这张图。另外它也是"消除 host sync"那一项的前提：capture 过程中任何一次 host sync 都会让捕获失败。

```python
# sglang_omni/models/fun_cosyvoice3/stages.py，`FlowCudaGraphRunner.capture`
stream = torch.cuda.Stream(device=self.device)      # side stream 录制，default stream 上的 kernel 不会被录进去
stream.wait_stream(current_stream)
with torch.cuda.device(self.device), torch.cuda.stream(stream):
    self.pool = torch.cuda.graph_pool_handle()      # 所有形状共用一个 memory pool
    for batch_size, mel_frame in capture_shapes:    # 按 (B, T) 白名单逐个捕获
        static_inputs = self.capture_inputs(batch_size, mel_frame)
        with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype,
                            enabled=self.autocast_dtype is not None):
            solve_flow_euler(self.flow.decoder, *static_inputs)   # warmup
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cuda_graph=graph, pool=self.pool, stream=stream,
                              capture_error_mode="thread_local"), \
             torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype,
                            enabled=self.autocast_dtype is not None):
            # 整个 solver（11 个时间点 = 10 步 Euler，每步带 CFG）录成一张图
            static_output = solve_flow_euler(self.flow.decoder, *static_inputs)
        graphs[(batch_size, mel_frame)] = CapturedFlowCudaGraph(
            graph=graph, static_inputs=static_inputs, static_output=static_output)
current_stream.wait_stream(stream)
```

```python
# sglang_omni/models/fun_cosyvoice3/stages.py，`FlowCudaGraphRunner.run`
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

#### 3. Scheduler level optimization

CV3这一类多阶段模型相对于Dual-AR结构模型最特别的部分，就是对scheduler的精心优化。

AR部分原生便是流式的output，因此如何做好DiT部分的chunk inference，调度好三个模型之间的推理，以及如何在不同GPU之间把各个部分打满（目前还没完成），就是这一小节的核心探讨内容。

在先前优化的基础上，各个stage目前都已经支持batch inference了，但是生产环境中我们常常会面临一个延迟的tradeoff：有的音频长，有的音频短，长度差距太大，我们没法等一整个batch都推理完了再统一丢给下一个模型。所以一个batch里的请求必然会在不同阶段之间先拆开，再按下一阶段的特点重新组batch，反复多次，以此打满GPU和各个模型的利用率。

由于stream和非stream存在较大的区别，我们这里先讲非stream的：

offline推理比较简单，可以总结为“收集-推理-发送”这三个行为，不过三个部分“收集”的方式并不一样。AR部分直接交给sglang的continuous batching，请求按decode step加入和退出batch，哪条先生成完就先把整条结果发给下一个stage，不会被同batch里的长请求影响（赞美sglang）；到了DiT这边，vocoder入口会按时间/数量检测请求，30ms/16条作为基础临界条件，然后按mel帧数计cost，保证总量不超过8000帧，随后再按长度做一次自适应分组（组内长度差不超过384帧，全局padding预算25%），尽量让长度相近的请求共用一次Flow solve；HiFT则在每个Flow组内部按padding浪费程度再分一次组。

理论上如果生成速度极快，长音频和短音频之间的完成时间差被压缩到10ms这种级别，我们就不需要这种处理了，直接等整个batch全部完成再一起交给下一阶段即可（也就是static batching，暂称懒狗方案）。懒狗方案本质上也是我们目前实现的一个子集：当一批请求恰好同时完成、长度也接近时，上面的流程就会退化成它。不过目前我们还远远没有达到这个极限效率，因此一定的工程实现可以极大地降低延迟。

先看vocoder入口的攒批。攒不进去的消息会按到达顺序插回队头，不会被丢掉：

```python
# sglang_omni/scheduling/streaming_simple_scheduler.py，`collect_new_request_batch`
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

攒到一批之后，再按长度做一次自适应分组。它是先按mel帧数排序，然后跑一个DP：目标是最小化 \(\sum \text{组大小} \times \text{组内最长}\)，约束是组内的长度差不超过 `flow_merge_max_gap_frames`，并且全局padding预算不超过 `flow_merge_pad_budget_percent`：

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

然后是streaming的，能把这种AR+NAR结构的streaming跑明白已然让人拼尽全力，在此基础上的优化更是考究工程能力。

AR每生成一段token，我们就会把这一部分发送给dit，首次需要hop + lookahead = 25 + 3 = 28个token，之后每25个token同步一次，并且第一个chunk会顺带把prompt的token/mel/speaker embedding一起带过去，这样vocoder不需要等AR整条生成完的payload就可以接着进行生成。DiT和HiFT在同一个vocoder stage里，DiT做完chunk-wise的推理渲染出mel谱之后，会在同一个step里直接交给HiFT增量地出波形，最终得到音频片段返回给用户侧。

对齐关系：DiT 的静态 chunk size 是 50 mel 帧，正好等于 25 个 token 乘上 token2mel 的倍率 2，所以 hop 取 25 个 token 不是随手定的，而是为了让一个 hop 正好落在一个 chunk 边界上。

vocoder这边给每个请求维护一个状态机：token够一个因果窗口（offset + hop + 3）就是`causal_window`，可以出一个hop；AR结束之后剩下的部分走`leftover`；AR结束但一个token都没有则走`fallback`；其余情况都是`wait`。DiT的chunk inference靠chunk-causal mask保证每一帧只看得到自己所在的chunk和之前的chunk，新来的token不会改变已经输出的部分；同时prompt会被pad到hop的整数倍，让第一个hop正好对齐chunk边界，不用多等几个AR token。hop的长度也是逐步增长的（25 → 50 → 100），首包只需要28个token，开播之后有了余量就换大hop减少调用次数。

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

一个step里，选定的请求合成一次成批的Flow调用，HiFT目前仍然是逐请求跑的：

```python
# sglang_omni/models/fun_cosyvoice3/streaming_vocoder.py，`run_step`
if plan == "leftover":
    mels = self.vocoder.leftover_batch(items)       # 多条请求的 final 合成一次 Flow 调用
    ...
else:
    mels = self.vocoder.hop_batch(items)            # 多条请求的 hop 合成一次 Flow 调用
    for (request_id, state), mel in zip(participants, mels, strict=True):
        delta, state.hift_mel, state.speech_offset = self.vocoder.hift_delta(
            mel[:, :, state.token_offset * TOKEN_MEL_RATIO:],
            hift_mel=state.hift_mel, speech_offset=state.speech_offset, finalize=False)
```

每一个step选哪些请求一起跑也有讲究：已经开播的请求优先，按播放余量（已产出音频时长 - 已播放时长）从小到大排，余量越小越容易断流；还没开播的请求排在后面，按到达顺序处理。并且只和同类step合批（hop和leftover走的是不同的Flow路径），每个step最多16条，多条请求沿序列维打包之后一次Flow调用算完。

```python
# sglang_omni/models/fun_cosyvoice3/streaming_vocoder.py，`select_step_participants`
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

整个step由serving loop驱动：先把inbox里的消息读完，再跑一个step，保证排序的时候看到的都是最新状态。驱动逻辑本身在框架层，很简单的几行代码：

```python
# sglang_omni/scheduling/streaming_vocoder.py，`StreamingVocoderBase.pump_one_step`
participants = self.select_step_participants()      # 选出本 step 一起跑的请求
if not participants:
    return None
plan = self.build_step_plan(participants)           # causal_window / leftover / fallback
decoded = self.run_step(participants, plan)         # 一次成批的 Flow 调用，HiFT 仍逐请求
for request_id, _ in participants:
    waveform = decoded.get(request_id)
    if waveform is not None and not self.is_aborted(request_id):
        self.mark_stream_emitted(request_id)
        self.outbox.put(self.stream_chunk_message(request_id, waveform))
```

```python
# sglang_omni/scheduling/streaming_simple_scheduler.py，`start`（serving loop）
if self.has_ready_work():
    try:
        msg = self.get_batch_message()               # 有活就先把 inbox 读完
    except queue_mod.Empty:
        self.run_ready_step()                        # 读空了再跑一个 step
        continue
```

最后是资源编排。vocoder和AR在同一个进程、同一张卡上，所以vocoder被放到了独立的CUDA stream上，kernel和host copy不再排在AR的decode后面：

当然后面如果考虑分离推理的话，还有更多的研究方向。

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

同时同进程的stage按列表顺序构建，vocoder放在engine之前，Flow/HiFT/CUDA Graph先驻留，再由SGLang按剩余显存分配KV pool，避免OOM或者KV pool被挤小。

至于多卡，框架本身已经支持把vocoder拆到单独的进程和GPU上，或者整条pipeline做数据并行，不过CV3还没有在这种部署下做过实测，这部分还在进行中。

#### 4. 多线程CPU处理

参考音频的编码（ONNX speaker encoder + speech tokenizer）是预处理阶段的主要负载任务，它原本在lock里串行执行，配了 `max_concurrency=8` 也并行不起来。改法是把编码移出临界区，只把最后的组装步骤加锁——**锁的范围应该只覆盖真正有共享状态的那部分**，这是这一类优化里最通用的经验。

```python
# sglang_omni/models/fun_cosyvoice3/request_builders.py，`preprocess_cosyvoice3_payload`
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
# sglang_omni/models/fun_cosyvoice3/config.py，preprocessing stage
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

#### *5. BF16推理

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
# sglang_omni/models/fun_cosyvoice3/stages.py，`CosyVoice3Vocoder.token2wav_chunk`
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

#### 6. 消除 host sync 与常驻化

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

### Kernel level method

#### 1. torch.compile

自从torch 2.0引入这个功能之后，它已然成为最简单且高效的优化方案，只需要加上 `@torch.compile` 的装饰器便可以一键享受加速。

compile的本质是将原本的model类编译为一个torch._dynamo类，把一个动态图编译为一个静态图，进行一系列的算子融合，从而实现加速。而且热更新模型的权重并不会破坏compile后的图，对于rollout来说也属于一种很有效的优化方式。

示例代码如下：

CV3 这边有两代实现。第一代是编译上游的 DiT：

```python
# sglang_omni/models/fun_cosyvoice3/stages.py，`compile_dit_backbone`
torch._inductor.config.fx_graph_cache = True        # 跨进程复用编译产物
torch._dynamo.config.cache_size_limit = 1024
dit_mod.add_optional_chunk_mask = torch.compiler.disable(
    dit_mod.add_optional_chunk_mask)                # chunk mask 留 eager
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

warmup 这一步不能省，而且必须在和 serving 相同的 `inference_mode` + autocast 精度下跑，否则 guard会直接爆炸，并且第一个真实请求会触发重编译，然后某个倒霉蛋就要爽吃100s的延迟了。这条路径默认开启，H200 上中位吞吐并发 1 时 +24.6%、并发 16 时 +59.0%，WER 不变。

第二代是编译我们自研的打包 DiT（`PackedDiT`），也就是流式路径跑的那个。causal（流式 hop）和 full（非流式 final）两条路径各编译一份，并且要求 `fullgraph=True`，不允许出现 graph break：

```python
# sglang_omni/models/fun_cosyvoice3/packed_dit.py，`PackedDiT.compile` / `forward_for_mode`
self.compiled_causal_forward = torch.compile(
    self.forward_causal, backend="inductor", dynamic=True, fullgraph=True,
    options=dict(PACKED_INDUCTOR_OPTIONS))
self.compiled_full_forward = torch.compile(
    self.forward_full, backend="inductor", dynamic=True, fullgraph=True,
    options=dict(PACKED_INDUCTOR_OPTIONS))

if not isinstance(attention, RaggedRowAttention):   # 非 FA3 路径回退 eager
    return self.forward
compiled = self.compiled_causal_forward if streaming else self.compiled_full_forward
return self.forward if compiled is None else compiled
```

```python
# sglang_omni/models/fun_cosyvoice3/streaming_vocoder.py，`warmup_packed_dit_compile`
if not packed_estimator.compile(self.vocoder.autocast_dtype):
    return                                          # 编译成本在启动期支付
with self.vocoder.stream_context:
    self.vocoder.hop_batch([item])                  # 失败即 disable_compile()
    self.vocoder.leftover_batch([item])
```

#### 2. TRT Engine

TRT 是 NVIDIA GPU 专用的一种加速手段：先把模型转成 ONNX，再用 TensorRT 导出成一个 engine。

坦白说，TRT 这条路我们已经实现了（opt-in，默认关闭），但目前**并没有比 torch.compile 更快，而且差得还不止一点**。单并发下它确实很强（RTF 约 0.09~0.10），但到了高并发反而不如compile了。

不过这一套的工程细节还是有价值的，尤其是它和 torch.compile 互斥、按 profile 边界回退这两点，所以代码留在下面供参考。

示例代码如下：

```python
# sglang_omni/models/fun_cosyvoice3/flow_estimator_trt.py，`convert_onnx_to_trt`
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

运行时有两个绕不开的点。第一，engine 的 profile 是 CFG=2（对齐官方 ONNX），而 packed 的 CFG batch 是 2N，所以要按"请求对"把它切成 engine 能够接受的形状，再把结果拼回 `[cond..., uncond...]` 的布局。第二，engine 跑在自己的 stream 上，前后各做一次 `wait_stream`，数据依赖才不会断：

```python
# sglang_omni/models/fun_cosyvoice3/flow_estimator_trt.py，`execute_flow_estimator`
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
# sglang_omni/models/fun_cosyvoice3/flow_estimator_trt.py，`FlowEstimatorTRTModule.forward`
frames = int(x.shape[2])
if frames < self.min_time or frames > self.max_time:
    return self.fallback(x, mask, mu, t, spks, cond, streaming=streaming)  # profile 外回退 PyTorch DiT
return execute_flow_estimator(self.trt, x, mask, mu, t, spks, cond)

# sglang_omni/models/fun_cosyvoice3/stages.py，`attach_flow_estimator_trt`
fallback = flow.decoder.estimator
wrapper = build_flow_estimator_trt(onnx_path, device, fallback=fallback, wrap_module=True)
del flow.decoder.estimator                          # 先删掉 nn.Module child，再挂 TRT wrapper
flow.decoder.estimator = wrapper
```

#### 3. Flash Attention

先花一点篇幅讲清楚 attention 这边到底在优化什么：

1. **前缀缓存（prefix caching）**：多个请求如果共享相同的前缀（比如系统提示、或者同一段参考音频），那段 prefix 的 KV 只需要算一次，后面的请求直接复用。
2. **RadixAttention**：SGLang的核心技术，用一棵基数树（radix tree）管理所有请求的 KV，在 radix tree 上做最长前缀匹配，让上面那个复用自动发生，不用调用方自己管理。
3. **FlashAttention**：算子层面的改进。它不把 \(N \times N\) 的注意力矩阵真正写出来，而是分块在 SRAM 里算完，IO 复杂度从 \(O(N^2)\) 降到 \(O(N)\)。这个和前缀缓存是正交的。

CV3 的 AR 部分直接跑在 SGLang 上，所以第 1、2 项它是自动继承的，不需要我们做什么。真正需要我们动手的是 DiT 那一侧，也就是 `packed_dit.py` 里的两个 attention 实现。

沿序列维打包之后，attention 最朴素的做法是把每行重新 scatter 回 `(rows, width, C)` 的 padded 布局再跑 SDPA，也就是 `RowAttention` 那条路：scatter → SDPA → gather。问题不在 `F.scaled_dot_product_attention` 本身（它会自动分发到 flash / mem-efficient / math 后端），而在于两点：一是传了任意 `attn_mask` 之后往往用不上 flash 后端，二是 padded 布局让**每一行都被算到最长那条的长度**。短请求为长请求陪跑，实际上的时间会被拉长到最长请求的生成耗时。

改法是直接用 FA3 的 varlen / paged KV 接口读 ragged 序列，不再 pad：

```python
# sglang_omni/models/fun_cosyvoice3/packed_dit.py，`RaggedRowAttention`（FA3 varlen / paged KV）
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
    out = flash_attn_with_kvcache(
        q=query[0].reshape(-1, self.heads, self.head_dim),
        k_cache=key[0].reshape(page_shape),
        v_cache=value[0].reshape(page_shape),
        cache_seqlens=self.cache_seqlens,
        page_table=self.page_table,
        cu_seqlens_q=self.cu_seqlens_q,
        max_seqlen_q=self.max_seqlen_q,
        causal=False,
    )
    return out.reshape(1, -1, self.heads * self.head_dim)
```

这里有个取巧的地方值得说一下：`FA3_PAGE_SIZE = 1`，一个 page 只装一个 token，于是 `page_table` 退化成一张"逻辑位置 → packed 序列里物理位置"的映射表。KV 直接在 packed 序列上原地读，既不用 pad 也不用拷贝。每段 query 由 `cu_seqlens_q` 描述，它能看到的 KV 长度由 `cache_seqlens` 描述——流式下按 chunk 切段，用 KV 长度来表达 chunk causal，所以调用时传的是 `causal=False`。

另外它被包成了一个 `custom_op`，并注册了 `register_fake`。这样 Dynamo 不会探测 FA3 内部，而是把它当成一个有明确形状语义的黑盒节点，上一节 `fullgraph=True` 能成立正是靠这一点：

```python
# sglang_omni/models/fun_cosyvoice3/packed_dit.py，`packed_fa3` / `PackedDiT.row_attention`
@torch.library.custom_op("sglang_omni_fun_cosyvoice3::packed_fa3",
                         mutates_args=(), device_types="cuda")
def packed_fa3(q, k_cache, v_cache, cache_seqlens, page_table, cu_seqlens_q, max_seqlen_q):
    return flash_attn_with_kvcache(q=q, k_cache=k_cache, v_cache=v_cache,
                                   cache_seqlens=cache_seqlens, page_table=page_table,
                                   cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q,
                                   causal=False)   # 包成 custom op，供 torch.compile 作为图边界

@packed_fa3.register_fake
def fake_packed_fa3(q, *args):
    return torch.empty_like(q)

self.is_ragged = device.type == "cuda" and _is_fa3_supported()
if self.is_ragged and dtype in FA3_DTYPES:          # FA3 + 半精度走 ragged，否则回退 padded SDPA
    return RaggedRowAttention(rows, chunk_size=chunk_size, heads=attention.heads,
                              head_dim=attention.inner_dim // attention.heads)
return RowAttention(rows, chunk_size=chunk_size, heads=attention.heads)
```

只有 CUDA 上支持 FA3、且 dtype 是半精度时才走 ragged，其他情况自动退回上面那套 padded SDPA，所以这一项是纯增益，不会让某些硬件上的正确性出问题。

#### 4. 手写算子改写

前面几项都是换一个现成的实现，这一项是真正自己动手改算子。

目标很单纯：**减少 elementwise kernel 的数量，以及减少显存读写**。在 DiT 这种结构上收益不小——我们在另一个同架构模型 auk 上测过，大 batch 时有 57~71% 的 kernel 时间在 elementwise 上。这些 kernel 单个看都是几微秒、算力几乎为零，但每计算一次都有 launch 开销，而且每个都要把数据从显存读进来再写回去，瓶颈其实在带宽。

具体到 CV3 这里是两处。第一处是 attention 的输入只 cast 一次：

```python
# sglang_omni/models/fun_cosyvoice3/packed_dit.py，`PackedDiT.attend`
x = x.to(attn.to_q.weight.dtype)                     # autocast 下 to_q/to_k/to_v 会各 cast 一次，这里只 cast 一次
query = attn.to_q(x)
key = attn.to_k(x)
value = attn.to_v(x)
rotate_in_place(query, *rope)
rotate_in_place(key, *rope)
out = attention(query, key, value).to(query.dtype)
return attn.to_out[1](attn.to_out[0](out))
```

autocast 之下 `to_q` / `to_k` / `to_v` 会各自把上游 fp32 的输出 cast 一遍，同一个 tensor 被转换了三次。提前按权重的 dtype 统一 cast 一次就够了。

第二处是 RoPE 改成原地计算。原来要构造新 tensor，现在只取 q/k 里 rotary 的那一段切片，算完用 `copy_` 直接写回原 tensor，不额外分配显存；cos / sin 也只在每步生成一份，q 和 k 共用：

```python
# sglang_omni/models/fun_cosyvoice3/packed_dit.py，`rotate_in_place` / `PackedDiT.rope`
def rotate_in_place(x, cos, sin):
    rotary = x[..., : cos.shape[-1]]                 # 只取 rotary 切片，其余不拷
    half = torch.stack((-rotary[..., 1::2], rotary[..., ::2]), dim=-1).flatten(-2)
    rotary.copy_(rotary * cos + half * sin)          # 直接写回 q/k，不新开 tensor

freqs, scale = self.dit.rotary_embed.forward_from_seq_len(rows.width)
assert not isinstance(scale, torch.Tensor)           # 该 DiT 的 RoPE 没有 xpos scale
freqs = freqs[:, rows.positions]
return freqs.cos(), freqs.sin()                      # cos/sin 每步只生成一份
```
