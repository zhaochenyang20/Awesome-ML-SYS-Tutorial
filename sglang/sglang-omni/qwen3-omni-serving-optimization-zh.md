# 优化的不是一次 Forward，而是整条语音流水线
本文记录 SGLang-Omni team 对 Qwen3-Omni serving 的一轮系统优化。
对普通LLM来说，生成第一个文字 token 往往意味着回答已经开始；但对 Qwen3-Omni，情况要复杂得多。它不仅要理解文字、图片和音频，还要一边生成文本，一边把语义转换成语音编码，再还原为可以播放的波形。因此，文字已经出现，并不代表第一段声音已经准备好；GPU 看起来很忙，也不代表音频能够连续到达。我们真正要优化的，不仅仅是某一次模型forward compute，还有一个请求如何穿过整条语音流水线。
在最开始的优化中，一条 short-prompt trace 给出了最重要的优化证据：device compute 只有约 3.2–3.8 ms，而prefill-to-first-emit 却达到 76.3 ms，从请求进入到首段文本（TTFT）共 88.1 ms，host 还发出了约 2400 次 CUDA API 调用。这个trace结果表示问题显然不只在模型算得多快，而在于去消减大量的重复计算。

如果把 Qwen3-Omni 只看成一个更大的 Transformer，最自然的优化对象就是一次 model forward：换 kernel、加 batch、捕获 CUDA Graph。这个视角能解释局部计算，却解释不了用户为什么还在等第一段声音，也解释不了 GPU 利用率不低时为何播放仍会断续。

一条语音请求不是在单个 engine 内完成的。输入先经过预处理与可选的图像、音频编码，Thinker 生成文本并持续送出隐藏状态，Talker 把语义流展开成 codec frame，Code2Wav 再把逐步到达的离散表示变成可播放波形。

这些模块有不同的计算形态、ready time 与资源需求。请求会在 admission、队列、线程、进程、设备与传输协议之间多次换手。一次小同步、一次无意义等待或一次多余 relay，都会被几百个 frame、几十路并发或多条 stage edge 放大。

因此，在本次的优化工作中的我们的优化单位从来不是一次 forward，而是**状态在异构流式流水线中的持续流动**。我们反复做了四件事：
- 把动态工作变成可重放执行
- 让 ready 状态沿最短安全路径流动
- 移除每帧重复成本
- 让执行 shape 服从音频 streaming protocol

当这些局部固定的成本被移走，瓶颈又会迁移。此时继续压缩原来的热点不会自然增加端到端的 capacity；系统必须重新决定哪些 stage 共置、哪些 stage 独占设备，以及应该复制哪一段，而不是复制整条流水线，我们针对这些设计了新的replica优化，具体的内容会在后文详细说明



> **TODO(FIGURE): Hero 图。** 
- 画出输入侧 fan-out、Thinker hidden stream、Talker frame loop、Code2Wav window 与客户端音频时间线；右侧只放冻结综合实验产生的 headline cards。验收条件是图中能区分 TTFT、TTFA、steady gap 与 capacity，且不混用不同拓扑的数据。

## 1. 性能单位是端到端的流

### 1.1 七个 stage 组成一条请求路径

当前 Qwen3 omni的 speech pipeline 由七个配置 stage 组成：`preprocessing`、`image_encoder`、`audio_encoder`、`thinker`、`decode`、`talker_ar` 与 `code2wav`。server 或 coordinator 父进程不计入 serving stage。

**TODO(figure) 用html生成更好的pipeline表示图
```text
preprocessing ─┬─> image_encoder ─┬─> Thinker ──> decode/text
               ├─> audio_encoder ─┤       │
               ├──────────────────┼──────>│
               └──────────────────┴──────>Talker ──> Code2Wav ──> audio
                                           ▲
                              streamed hidden chunks
```

Preprocessing 与按需执行的 image/audio encoder 会把 payload 传递给 Thinker 和 Talker；Thinker 会把 hidden chunks 流式送往 Talker。Predictor 是 Talker 内部子模块，不是独立 stage。

历史语音路径中，预处理、图像编码器和音频编码器的输出会先汇入 mm_aggregate。在这个阶段不执行模型计算，只负责等待同一请求所需的输入全部到齐，再将结果转发给 Thinker 和 Talker。新路径将这项等待交还给真正使用数据的模块：Thinker 和 Talker 按请求直接接收预处理及各编码器的输出，在各自需要的输入齐备后开始执行。删除这个纯汇合与转发阶段后，语音流水线由八个阶段缩减为了七个。
这次修改只重构了语音输出路径；纯文本路径仍沿用原来的 mm_aggregate，不应写成全局删除。相关实现请见 PR #1548(https://github.com/sgl-project/sglang-omni/pull/1548)。

### 1.2 首包、稳态与容量

```text
request ingress
    ├── preprocess / encoder / Thinker ── first text       ← TTFT
    ├── hidden stream / Talker / first WAV                ← TTFA
    ├── WAV chunks ── gap ── WAV chunks                   ← continuity
    └── all terminal outputs complete                     ← E2E
```

TTFT（time to first token）描述第一段文本，TTFA （time to first audio）描述第一块可播放音频，E2E latency 描述所有终端输出完成时间。RTF (real time factor) 观察语音生产效率，inter-chunk gap 观察播放连续性，request/s 与各 stage queue 共同描述 capacity。

优化是一个关于trade off的选择，任意指标的提升可能会造成其他指标的下降：比如说，更早 partial start 可能让 Code2Wav 收到更碎的窗口，更大 batch 可能推迟TTFA，replica 增加饱和 capacity 也可能增加低并发路由成本。详细定义、输出合同与质量 gate 见附录 A。

### 1.3 Correctness 与 profile 决定优化方向

本文用 `cN` 表示并发 N，用 A/A band 表示同代码、同条件重复运行的波动范围；A/A floor 是据此设定、候选必须越过的最小 promotion 阈值。

历史 extend/mixed batch 曾把 request 级过滤条件用于 token 级 KV 字段；stale slot 回收后，`out_cache_loc=None` 仍可能到达 `store_cache`。初始 c32 audit 约每 64 个请求失败一次，baseline 因而没有资格谈吞吐。

修复后 520/520 个请求完成，其中 480 个运行在 c≥32。我们再把证据分成 mechanism microbenchmark、profile attribution、paired E2E 与 frozen composite；只有语义 gate 和对应层级都成立，结论才会 promotion，相关修复见 [PR #1027](https://github.com/sgl-project/sglang-omni/pull/1027)。

开头的 short-prompt trace 已经把主矛盾从 GEMM 指向 host dispatch。它也解释了为什么后文先消除动态控制、调度空隙和逐帧固定税，再讨论更大的 batch 或更多设备。

沿全链反复出现四类等待：请求尚未获得 admission；ready 工作未被 scheduler 选择；每个 token/frame 重复支付 host、copy 或消息固定税；某个 stage 已饱和而其他设备仍有余量。它们分别指向后文四个系统块。

下面四行来自不同的 controlled campaigns，是诊断地图，不是可累加的一次 A/B：

| 系统块 | Observed constraint | Mechanism | Next exposed constraint / default question |
|---|---|---|---|
| Thinker | 动态多模态控制与 host dispatch | batch-wide merge、stable hidden、prefill graph | scheduler regime、bucket 启动与显存预算 |
| Pipeline | ready state 仍等待 timer、wakeup、relay 或错误 transport | conditional dispatch、outbox drain、consumer-aware data plane | placement、replica 与低并发代价 |
| Talker | 每个 codec frame 重建 state、copy 与 feedback | sampling reuse、native GQA、pinned/dense path | ownership 正确后 overlap 是否真有可隐藏工作 |
| Code2Wav | 串行窗口 launch、batch shape 与 output materialization | exact/batched graph、chunk alignment、depth-2 output | wait、graph memory、首窗与 continuity 的 Pareto |

> **TODO(FIGURE): Measurement 图。** 用一条 request timeline 标出 TTFT、TTFA、E2E、gap、stage residency、queue 与 device self-time，再画 `profile → hypothesis → mechanism → correctness → A/A → paired A/B → promote/revert`。图注固定 benchmark contract 与证据层级。

## 2. Thinker：把动态 Prefill 变成可重放执行

在我们优化Thinker 的过程中，thinker prefill 是最接近传统模型优化的一段，在我们最开始 profile 结果得到的结果并不是 FLOPs 太多，而是动态控制太多。多模态位置要在运行时拼接，hidden state 要跨模块交付，scheduler 还要在 prefill 与 decode 之间决定谁先前进，这些都会造成时间的损失。

要把这种路径变成 CUDA Graph replay，不能从“录一张图”开始。首先要让请求级动态工作收敛成 batch-wide tensor operation；随后要把下游依赖的 hidden state 放进地址稳定、生命周期清楚的 buffer；最后 graph runtime 才能安全处理 bucket、padding 与 fallback。

在这之前，我们先系统的清掉了几块挡住主线的障碍。
1. 更高的安全 admission cap 让 c32–c64 请求真正进入 scheduler，相关能力见 [PR #1135](https://github.com/sgl-project/sglang-omni/pull/1135)。

2. Mixed prefill/decode 让 ready decode 不再被 pure-prefill step 长时间困住，相关能力见 [PR #789](https://github.com/sgl-project/sglang-omni/pull/789)。

这两项工作的处理对象各不相同。admission cap 决定多少请求有资格参与调度，mixed execution 决定一次 scheduler step 如何使用设备。

### 2.1 把逐请求多模态拼接改成 batch-wide merge

多模态 prompt 会在 token 序列中预留 image、video 或 audio placeholder，encoder 输出之后写回这些位置。旧实现按“请求 × 模态”循环，在 GPU mask 上调用 `mask.any()`、`mask.sum().item()` 与 `torch.where()`，把 host sync 放进了 batch 扩展方向。

在普通 multimodal request 时大约会触发 6 次强制同步；interleaved image+video 加上 deepstack 时最多约 18 次。现象是 batch 增大后 merge latency 近似随请求数增长，profile 根因则是 Python 控制流依赖 device scalar，而不是 scatter 本身昂贵。

最符合直觉改法是把动态性前移到 request construction。builder 在 CPU 侧记录 placeholder position，prefill 时为全 batch 一次构造索引，再用一次 `index_copy_` 完成 scatter。这样不是“加速同一段循环”，而是把循环变成可描述、可捕获的 tensor program。

在我们的 microbenchmark 中，单模态 merge 从约 2.1–2.3 ms 降到 0.8–1.0 ms，interleaved image+video 从 7.7–7.8 ms 降到 1.9–2.3 ms。

8192 multimodal-token 上限附近会增加约 32 MiB transient source tensor，且独立 E2E TTFT 未稳定越过噪声带，相关实现见 [PR #1161](https://github.com/sgl-project/sglang-omni/pull/1161)。

多模态 rotary position 也经历了同类转换：向量化 block construction 取代逐 token 处理，Talker 只有在能证明 prompt 不含 multimodal start 时才走 linear MRoPE fast path。

这项工作提供 graph-friendly 前置条件与 differential parity，不承担综合 TTFT headline，相关实现见 [PR #1160](https://github.com/sgl-project/sglang-omni/pull/1160)。

这个故事的端到端边界很重要。microbenchmark 证明 host sync 被删除，不能证明用户延迟必然下降；但如果不先删除这些运行时 scalar 与分支，后续 graph capture 即使勉强成功，也只能覆盖一个语义残缺或形状极窄的路径。

### 2.2 用稳定地址承载需要跨 replay 的 hidden state

Text-output prefill 只需要最终 logits，speech prefill 还要把特定 layer 的 hidden state 交给 Talker。旧路径通过 monkeypatch 或临时返回 tensor 捕获 layer 0/24 hidden，eager 下可以工作，graph replay 下却没有地址与生命周期保证。

如果 replay 覆盖了上一轮 tensor，或者 consumer 读到错误 row，系统可能仍返回正常长度的音频。这里的 correctness 不是 shape 对齐，而是 request、row、logical token span 与 hidden content 一一对应；内容串线比 crash 更难被普通 smoke test 发现。

原则性改法是 registered static buffer 加 layer pre-hook。每次 forward 只向固定地址 `copy_`，consumer 按本次有效 row 数读取。默认 8192-token capacity 的 steady-state 额外显存约 67 MiB，稳定地址由明确容量换取，相关实现见 [PR #1380](https://github.com/sgl-project/sglang-omni/pull/1380)。

Buffer 只是所有权的一半。deferred graph initialization 失败后还必须清除残留参数，eager multimodal cursor 也不能继承 capture 期间的状态。

Bootstrap 恢复和 cursor 生命周期随后被继续加固，相关修复见 [PR #1532](https://github.com/sgl-project/sglang-omni/pull/1532) 与 [PR #1537](https://github.com/sgl-project/sglang-omni/pull/1537)。

这一步没有把一次 forward 直接变快。它建立的是 replay contract：生产者只能写自己的有效 row，消费者只能在约定事件之后读取，失败初始化必须回到干净 eager 状态。没有这个 contract，speech graph 的性能数字没有语义可信度。

### 2.3 Graph replay 改变的不只是 launch 数量

共享 Breakable Prefill CUDA Graph (BCG) runtime 负责 bucket、padding、admission、replay 与按 shape 的 eager path；model-local adapter 负责判断 Qwen batch 能否被完整表示。

未知 auxiliary state、cursor 不完整或超出 capture envelope 时必须拒绝 graph，相关运行时见 [PR #1364](https://github.com/sgl-project/sglang-omni/pull/1364)。

Text-output BCG 先接入 Qwen3-Omni。643-token、`max_tokens=1` 的三轮测试中，c1 TTFT p50 从 55.2 ms 降到 26.3 ms，c1–c16 input throughput 提高约 37–120%。

这条 text 路径不依赖上一节的 stable hidden capture，且能力是 explicit opt-in，相关实现见 [PR #1381](https://github.com/sgl-project/sglang-omni/pull/1381)。

Speech 路径只有在 static hidden contract 就绪后才能接入同一机制。它必须让 Talker 得到与 eager 相同的 projected hidden content，而不是只让 Thinker replay 成功。

Speech BCG 同样是 explicit opt-in，依赖的是稳定 hidden 语义而非 text adapter 的开发顺序，相关实现见 [PR #1519](https://github.com/sgl-project/sglang-omni/pull/1519)。

这次 speech campaign 覆盖 8448 个零失败请求。低并发下，c1 TTFT/TTFA 分别下降 18.0%/13.5%；到 c32，TTFA 却增加 8.4%、E2E 增加 3.9%、request/s 下降 3.1%。同一个 replay 机制跨过并发区间后改变了方向。

日志确认请求仍命中 graph，因此高并发回退不能简单归因于 fallback。更可能的机制是 prefill 更快 drain 后，scheduler 进入更频繁、更小 batch 的 regime；launch 时间减少了，queue position 与 batch composition 却同时变化。

Graph 还把启动和显存变成一等资源。一组 42-bucket capture 增加约 18–31 秒启动时间，并占用约 1.26 GB/TP rank 常驻显存。更多 bucket 能提高 shape coverage，却会侵蚀 KV 与运行 batch 的容量，最终可能把收益从另一条曲线拿回来。

因此，Thinker BCG 的 production 形态必须同时报告 capture time、每 rank graph memory、shape hit、fallback、TTFT、TTFA 与 QPS。它不是“打开后所有请求更快”的布尔优化，而是一种会改变 scheduler regime 的显式 execution profile。

这些独立 campaign 不能按顺序累加，却把下一个系统问题指向 stage boundary：结果已经 ready，为什么 consumer 还没有拿到？

> **TODO(FIGURE): Thinker 图。** 用三层 before/after 表达 request-wise merge → batch-wide merge、ephemeral hidden → static sidecar、约 2400 次 host launch → BCG replay；右侧同时画低并发收益、高并发负向与 42-bucket 启动/显存边界。

## 3. Pipeline：让 ready 状态沿最短安全路径流动

模型算完不等于 consumer 已经拿到结果。每个 stage 有自己的 inbox、outbox、scheduler 与生命周期；大 tensor 和小控制消息又可能选择不同 transport。ready state 可能继续等待 timer、线程唤醒、handle open、join 或多余 relay。

Pipeline 优化的核心不是让所有边都“零拷贝”，而是回答三个问题：状态何时真的 ready，谁是最终 consumer，最小的安全 handoff 是什么。只有 ownership、ordering 与失败回收都明确，路径才有资格缩短。

### 3.1 把 encoder 的三种固定税分别移走

Audio encoder 的 head latency 曾同时包含三种互不相同的成本。第一条消息到达后固定等待约 50 ms，即使没有第二个请求；32-layer forward 以 eager 发出约 460 次 kernel launch，GPU busy 约 8%；约 258 KiB 输出还要为每个请求打开 CUDA IPC handle。

Profile 中 direct IPC 的 `cudaIpcOpenMemHandle` mean 约 13.4 ms、p95 约 69 ms，而同一小 payload 经 pooled relay 的 measured cost 约 1.3 ms。现象都叫“encoder 慢”，根因却分别是 batch policy、host launch 与 transport setup。

第一步让 micro-batch wait 可配置并把默认值设为 0；同一 batch 中重复的 audio cache key 只编码一次。

在对应主干上，c1/c8 mean TTFT 分别下降 30.4%/27.8%，但 c64 QPS 最差下降 3.7%。无 backlog 时等待是纯税，有 backlog 时 batching 仍可能有价值，相关实现见 [PR #1564](https://github.com/sgl-project/sglang-omni/pull/1564)。

最终组合没有重新引入默认等待。Conditional wait 只影响部署方显式配置的正 wait：存在 backlog 才计时，空闲请求立即 dispatch。与此同时，32-layer stack 按 128–4096 token 分桶捕获 graph，`cu_seqlens` split 移到 layer 外，小 audio payload 走 pooled relay，大 image/video tensor 保留 direct path。

这组 headline 使用的 baseline `b3bbff6e` 早于上一项 wait=0 改动，仍包含旧 50 ms head tax。相对这个旧 baseline，c1/c8/c32 mean TTFT 分别下降 54.7%、51.3%、46.1%，c1/c8 mean TTFA 下降 31.2%/20.2%；c8/c32 QPS 基本持平。

12 个 cell 零失败，音频时长在 ±1.6% 内，WER 略优，speaker similarity 位于 eager control band，相关实现见 [PR #1628](https://github.com/sgl-project/sglang-omni/pull/1628)。

两组百分比不能相加。后一个 headline 不是相对上一项 wait=0 改动或 current main 的纯增量，而是把旧 head tax、graph 与 transport 一起纳入 A/B。可推广的结论不是 encoder 获得某个固定加速比，而是 timer、launch 和 handle setup 必须按各自机制与负载条件处理。

### 3.2 一次唤醒处理所有 ready 工作，并删除纯 relay hop

旧 pipeline 每取一条 stage output 就调用一次 `run_in_executor()`，即使 outbox 里已经积累更多 ready message。profiling 显示 72.37% 的 Thinker message 与 44.40% 的 Talker message 可以直接从 backlog 消费，系统却反复支付线程唤醒和 event-loop handoff。

原则性改法是保留第一次 blocking read，随后非阻塞 drain 最多 63 条；连同首条，每轮上限 64，然后主动 yield。它不为未来消息增加等待，也不改变 FIFO、completion 与 abort 语义，只让一次已经付费的唤醒处理当下已 ready 的工作。

c16 SeedTTS 测量中，output throughput 提高 4.4%，TTFT p95 下降 8.9%，TTFA p95 下降 17.1%；160/160 请求的 text、token count、chunk count 与 WAV hash 匹配。这里 batch 的是控制工作，不是 model forward，相关实现见 [PR #1384](https://github.com/sgl-project/sglang-omni/pull/1384)。

更彻底的路径缩短，是删除没有独立计算与不可替代状态的 relay。历史 `mm_aggregate` 等待 preprocessing 与 encoder payload，完成 join 后再转发给 Thinker/Talker；consumer 现在直接等待自己所需的输入，join ownership 回到真正使用数据的一侧。

H200 cold-boot paired tests 中，c1/c8/c16/c32 mean TTFT 分别下降 8.9%、12.0%、20.0%、16.7%，mean TTFA 下降约 5.0–8.3%，QPS 基本持平。

p95 有正有负，所以结论限定为 speech mean head latency 改善，text-only 仍保留该 stage，相关实现见 [PR #1548](https://github.com/sgl-project/sglang-omni/pull/1548)。

这两项工作看似一个是 event-loop 微优化，一个是 topology 变化，背后却是同一原则：ready 数据不应等待与其语义无关的调度边界。可以当场 drain 的消息不再重新睡眠，可以由 consumer join 的状态不再经过专职转发者。

人为拉开请求 arrival 可以打散 prefill wave，却也会让本来 ready 的请求等待。早期 natural-EOS campaign 中，一组配置改善 c32 TTFT，却让 c8 TTFA 变差。后续 colocated sweep 多数落在 A/A band；c64 `gap25` 有多项指标同向负向，但幅度全部仍在 A/A floor 内，不能解释为显著 regression。

因此 admission staggering 只能是默认关闭、由 arrival burst 与 SLO 决定的 policy。没有 online load signal 时，一个全局间隔无法同时服务低延迟和饱和吞吐，相关工作见 [PR #1565](https://github.com/sgl-project/sglang-omni/pull/1565)。

### 3.3 Data plane 必须理解 consumer、大小与生命周期

“tensor 很大”不等于每个中间 stage 都应该 materialize 它。早期 video path 会让不消费 embedding 的 stage 重复承担传输与解析；约 55 MiB payload 因此沿 pipeline 扩散，症状表现为延迟随边数而不是最终 consumer 的计算增长。

Producer 随后只发布一次 tensor，中间 stage 转发轻量引用，最终由 Thinker resolve。对应 video workload 的路径 latency 从约 160–179 秒降到 36–37 秒，accuracy 不变，相关工作见 [PR #808](https://github.com/sgl-project/sglang-omni/pull/808)。

集中式 intra-node CUDA-IPC data plane 继续承接这条 ownership 设计，但不改变“只由 consumer resolve”的原则，相关工作见 [PR #869](https://github.com/sgl-project/sglang-omni/pull/869)。

Consumer-aware 还意味着不发送根本不会被读的字段。Talker projection 不再携带 deepstack visual embedding；这一改动没有发明新 transport，只是让 payload schema 与读者集合一致，相关工作见 [PR #953](https://github.com/sgl-project/sglang-omni/pull/953)。

Zero-copy 也不是无条件答案。几十 MiB tensor 可以摊薄 CUDA IPC setup，小 payload 却可能让 handle open 比 pooled copy 贵一个数量级。正确选择需要同时看 size、fan-out、reuse、receiver device 与 handle 生命周期，而不是给所有 tensor 套一个 backend。

Payload 尚未传输时，构造本身也可能阻塞。Talker receive path 曾为每个 streamed text chunk 重复解析 checkpoint shard resolution，并重新打开同一个 safetensors shard；isolated call 从约 11.4 ms 降到 0.02 ms，三组 c8 paired run 的 TTFA p50 均下降超过 64%。

早期实验 arm 还带后来被认定生产路径不可达的 row cache，因此精确 E2E 归因必须保留 revision 边界。最终可依赖的原则是缓存 source/handle lifecycle，不能把旧 arm 的所有收益自动归给 merged diff，相关工作见 [PR #1187](https://github.com/sgl-project/sglang-omni/pull/1187)。

这条 data-plane 线没有结束。Thinker→Talker stream 仍可继续缩小，但在新主干 paired result 完成前只能作为 open mechanism，相关工作见 [PR #1574](https://github.com/sgl-project/sglang-omni/pull/1574)。

Pending-text queue 的 open work 已消除 31120 次 queue-level `torch.cat` 和约 457 MiB 旧 row 重拷贝，CatArray launch/GPU time 下降约 72%。它证明二次复杂度被删除，却没有证明用户曲线同步变化。

4200 个 E2E 请求的 QPS 与 latency 只变化约 1–2%，所以当前结论仍是 mechanism clear、E2E neutral，不能把内部 72% 写成用户性能收益，相关工作见 [PR #1611](https://github.com/sgl-project/sglang-omni/pull/1611)。

#### Topology consequence：路径变短后重新分配容量

Consumer-aware handoff 解决的是一条 edge 怎么走；当单 stage 接近饱和，下一步就要决定进程和设备怎么摆。CPU thread、GPU colocation 与 replica routing 都是 dataflow topology 的一部分，不是部署完成后才附加的运维细节。

多进程 colocated worker 若各自按整机核数创建 OpenMP pool，会把单进程默认叠加成 host oversubscription。一台 224-CPU H200 host 上，两份单卡 worker 曾产生约 2940 个线程，GPU mean utilization 只有约 71–72%。

Qwen3-Omni colocated stage 随后把默认 `OMP_NUM_THREADS` 限为 8，并关闭单 prompt 不需要的 tokenizer parallelism，仍允许显式覆盖。

线程数降到 1284 后，c32/c64 QPS 分别提高 49.3%/98.4%；这证明 host launch capacity 也必须按整条 pipeline 预算，相关实现见 [PR #1060](https://github.com/sgl-project/sglang-omni/pull/1060)。

GPU placement 的同样原则是把容易争用的重 stage 隔离，把轻 stage 放到有稳态余量的一侧。历史 c8 profile 中 Talker median GPU utilization 约 86%，Code2Wav 约 9–13%，而 Thinker 在 speech 稳态有空闲；让 Talker 与 Code2Wav 共卡，正好把轻量窗口计算放进最敏感的自回归循环旁。

默认布局因此把 Code2Wav 移到 Thinker GPU，让 Talker 独占另一张卡。主证据不是旧两卡默认的直接 A/B，而是 Thinker TP2、Code2Wav 固定 rank 1、只移动 Talker 的 isolation experiment；两组 c8 pair 中 TTFA p90 下降约 46%，stall total 下降约 49–55%。

较弱 control 再把 Code2Wav 从 Thinker rank 1 移到独占 GPU，wall 与 E2E 只变约 1%，共同支持 contention 归因。

低并发存在代价：c1 E2E 增加约 14–17%，因为 hidden handoff 变成跨设备；默认选择优先 TTFA 与高并发 capacity，并非无条件胜利，相关实现见 [PR #1235](https://github.com/sgl-project/sglang-omni/pull/1235)。

当一个 Talker GPU 已饱和，再提高 admission 只会改变排队位置。Process replicas 允许一个逻辑 stage 展开为多个实例，请求在 admission 时绑定 replica，并在 payload、stream、completion、abort 与 admin path 保持 sticky binding；model code 仍只认识逻辑 stage 名。

Replica 收益必须排除“只是多了一张卡”。同一实验比较两卡 baseline、三卡各 stage 隔离 control，以及三卡 `2×(Talker+Code2Wav)`。c64 时 replica2 相对 equal-hardware 三卡 control 的 QPS 提高 41.7%，而 isolated control 与两卡 baseline 几乎相同。

容量并非免费。c1/c8 的 replica2 QPS 比三卡 control 分别低 3.9%/3.8%；c64 虽然 mean audio TTFA 与 E2E 改善，mean text TTFT 却增加 32.5%，inter-chunk p95 增加 30.4%。这是一种高并发 scale profile，不是所有请求默认更快的开关。

这些数据来自 PR head，而不是 merge commit。2-GPU full-replica 的 correctness、abort 与 teardown 已在对应 head 上覆盖；仍缺的是 replica2 表在 merge/current-main 上的语义等价与性能复测。

现阶段证据证明“复制瓶颈 stage”改变 capacity curve，不证明任意部署开启 replica 都会获益，相关实现见 [PR #1175](https://github.com/sgl-project/sglang-omni/pull/1175)。

在另一组 controlled campaigns 中，handoff 与 topology 被单独测量；当这些边界税收缩，Talker 每个 codec frame 都重复支付的成本便成为下一项可归因对象。

> **TODO(FIGURE): Pipeline 图。** 以 ready-state 时间线串起 encoder wait/launch/transport、outbox drain、删除 relay 与 consumer-aware data plane；下方补 CPU budget、placement 和三卡 equal-hardware replica control，明确标出低并发代价。

## 4. Talker 与 Predictor：移除每个 codec frame 都要支付的成本

Thinker 的 prefill 开销集中在请求前部，Talker 的固定税则沿音频时间轴重复。每一帧都依赖上一帧采样结果，内部 Predictor 还要逐组补齐 codebook；单步多一个小 copy、一个 state rebuild 或一次 host sync，整段语音就会重复几百次。

这使 Talker 的优化方法与大矩阵 kernel tuning 不同。我们先问哪些对象在相邻 step 之间实际上没有变化，再问哪些 tensor 本来可以由 backend 广播，最后把确实需要跨 host/device 的反馈收敛到可复用、可批量处理的路径。

这条路线也被 profile 约束：约 98.1% 的 Talker kernel 已位于一张约 2000-node graph replay 中。继续扩大 broad graph coverage 不会自动命中热点；graph 外 state、payload、resolve 与 backpressure 才是问题。

### 4.1 复用没有变化的 sampling state

旧 scheduler 即使 batch composition 与 sampling 参数没有改变，也会每帧重建 sampling state、mask 与 metadata，并发起多次小 H2D 和同步。profile 里看到的不是一段巨大 self-time，而是一串规律重复、随 frame 数线性增长的事件。

原则性改法是让 sampling state 跨 step 持久存在，只在 batch composition、参数或 ownership 真正变化时更新。它把“每帧重新证明状态相同”改成“变化发生时显式失效”，也让 state lifecycle 更容易被测试。

Profiler-on 归因中，每帧 pageable H2D 事件数从 15.13 降到 4.26，forward thread 的 stream synchronization 从 16.09 降到 5.23，`memcpyAsync` 从 44.80 降到 26.28。三组数字都是**每帧事件计数，不是毫秒**，相关实现见 [PR #1043](https://github.com/sgl-project/sglang-omni/pull/1043)。

对应 c8 E2E 变化约 2.8%，落在后来测得的 A/A band 内。因此这项工作的可靠结论是 per-frame host/data movement 被删除，而不是 speech latency 获得稳定的独立百分比。机制证据强于 headline，是这里应保留的证据层级。

同类实验还移除了 cached single-token embedding gather 与 not-ready rollback scalar write 上的 host-blocking sync。局部 round-trip 消失，864/864 请求成功，九个 cell 的 text hash 48/48 一致；可见 c1 TTFA 仅变化约 −1.5%。

结果仍在 A/A band 内，说明删除同步 API 不等于删除真实依赖；ordering 或节流可能在更晚位置重新出现。这项工作命中机制但没有通过用户性能 gate，相关工作见 [PR #1409](https://github.com/sgl-project/sglang-omni/pull/1409)。

### 4.2 让 attention backend 原生表达 GQA

Predictor 的 query head 多于 KV head。旧 attention path 先用 `repeat_kv` 显式扩展 K/V，再交给 SDPA；16 个 code group、5 层 Predictor、K/V 两份 tensor，使每个 Talker token 最多触发约 160 个窄 copy kernel。

这些 copy 的单次代价很小，却处在每帧必经的串行循环。更关键的是，它们并没有新增语义：K/V head 只需要按 GQA 规则广播，物化 expansion 是 backend 表达不足产生的中间工作。

改法是直接启用 SDPA 的原生 GQA，让 attention backend 处理 head broadcasting。真实 Predictor 路径覆盖 KV cache、attention 与 output parity，codec token 和音频保持一致，相关实现见 [PR #1164](https://github.com/sgl-project/sglang-omni/pull/1164)。

Talker Predictor 已经位于 CUDA Graph replay 区域，所以 Python launch 开销大多被摊薄，剩余收益主要是 replay 内部的 memory work。没有冻结主干的独立 replay-time 与 E2E 证据时，它应被理解为低风险的 kernel cleanup，而不是整条 speech path 的 headline。

### 4.3 把反馈路径收敛到 pinned、batched 与 dense fast path

Forward 已经 replay 后，scheduler 仍要读取 sampled token、克隆下游输出，并把 feedback 写入下一步的位置。旧通用路径使用 pageable token D2H、逐对象 clone 和稀疏 scatter，即使当前 batch 的 row 连续，也支付最保守的处理成本。

原则性改法有三层：用 pinned token staging 承接必要的 host handoff，把多个输出 clone 合并处理，并在 row 连续时走 dense feedback fast path。稀疏、reorder 或 retract 情况继续保留通用语义，而不是为了 fast path 假定 batch 永不变化。

对应 profile 中，pageable token D2H 从约 1.99 次/帧降到 0.01 次/帧，clone 从约 18 次/step 降到 6 次/step。它直接说明 scheduler 反复支付的 host work 收缩了，不能单独替代用户侧 A/B。

H100、c8 做了 5 组交错 pair，完整结果是 4/5 方向支持 candidate；pair 2 反向，pair 5 两个 arm 都退化。PR 汇总采用 clean pair 1/3/4：request/s 提高约 12.4%，xRT 提高约 9%，TTFA 从 0.788 秒降到 0.677 秒。

发布时必须同时保留 PR 汇总采用 clean pairs 1/3/4 的记录和完整五组方向，不能只引用三组均值。c1 基本不变，也符合 per-frame host work 随 batch 放大的机制预期，相关实现见 [PR #1167](https://github.com/sgl-project/sglang-omni/pull/1167)。

这三步形成一条递进关系：复用不变的控制状态，删除不需要物化的 tensor，再为确实变化的反馈建立显式 fast path。它们都在缩短同一条 frame loop，却分别操作 cache invalidation、attention representation 与 data ownership。

### 4.4 Ownership 正确只是 overlap 的前提

沿这条路线继续，一个自然想法是让 sampled token 与 feedback embedding 完全驻留在 device slot，再把下一步 launch 与上一步 resolve 重叠。它在结构上很诱人，但最新验证显示：拥有正确的 slot，并不意味着存在足够工作可以隐藏其管理成本。

一版实现覆盖 reorder、retract、finish 与 subtype replay，证明 device-resident feedback slot 可以维护语义。

在 7872 个 natural-EOS 请求零失败的前提下，三次 c64 paired boot 的 TTFA mean/p95 median delta 约为 +4.1%/+6.7%，因此状态保持为 HOLD，相关工作见 [PR #1204](https://github.com/sgl-project/sglang-omni/pull/1204)。

建立在该 ownership 上的 overlap 版本按 request ID 重映射 unresolved token，避免 batch composition 变化时串 row。

正确性测试通过，但 H200 ABBA 中 c16/c32 QPS 分别下降 3.15%/1.78%，TTFA p95 增加 18.8%/31.0%，两个 paired boot 同方向，相关工作见 [PR #1320](https://github.com/sgl-project/sglang-omni/pull/1320)。

负结果指向的是 dispatch policy，而不是否定 device residency。若 direct path 的 startup cost 大于可隐藏工作，或 downstream backpressure 让 unresolved slot 占用更久，无条件 overlap 就会把更复杂的状态机带进 tail latency。

下一次重做应先观测 batch slot 中可重叠的工作量、resolve stall、下游 backpressure 与 fallback 频率，再决定何时启用 direct/slot path。Production default 来自这些 gate，而不是从“异步一定更快”的直觉推出。

Talker 证据同样不能与上游百分比串乘；它把问题继续推到下游：逐帧 token 到达后，什么 window shape 才既可高效执行，又符合播放协议？

> **TODO(FIGURE): Talker 图。** 展开一个 codec frame 的 sampling state、Talker forward、Predictor、feedback write 与 output handoff；用三种颜色区分被删除的事件、仍需保留的 ownership，以及 device-slot/overlap 的负结果。

## 5. Code2Wav：让执行 Shape 服从 Streaming Protocol

Talker 逐帧产生 codec token，Code2Wav 却按窗口解码波形。它既需要左侧历史，又希望尽快产生首块音频；多个请求若能同时 ready，可以共享一次 batched forward，但等待它们对齐本身又会增加 TTFA。

所以 Code2Wav 的 shape 不是纯模型参数。`batch × frame window × left context` 由 streaming protocol、arrival cadence、显存预算与首窗策略共同决定。Graph、batch 和 chunk 若各自优化，最终很容易在另一维度互相抵消。

### 5.1 从 exact serial graph 走向有界 batching

最初的流式解码以 batch 1 逐窗口运行，真实时间维会出现 `T={10,20,30,35}`。只捕获最大 shape 会让小窗口 padding，增加无效计算，还可能模糊不同上下文状态；完全 eager 则为每个串行窗口重复支付 launch overhead。

第一步因此捕获真实的 `B1/T{10,20,30,35}` exact shape。Replay 前若发生 graph key miss、batch ineligible 或 capture-time incompatibility，请求可以直接选择 eager，不进入 replay。

一旦 replay 已开始并失败，当前 request 必须 fail closed、raise，不能静默 eager 重跑；runner 随后会被标记为 disabled。后续 request 可以因为 runner disabled 走 eager，但失败的当前 request 不会被再次执行。

三组 clean H100 c8 pair 中，request/s 提高 7.77%，latency 下降 6.57%，TTFA 基本持平；后续 H200 c8–c32 sweep 的 xRT/request throughput 提高约 9–12%。首窗仍要形成并完成，所以收益主要落在后续窗口，相关实现见 [PR #1101](https://github.com/sgl-project/sglang-omni/pull/1101)。

Exact graph 解决了单 stream launch，却没有让多个 ready stream 合并。下一步 scheduler 在有限 deadline 内收集请求，到 batch floor 或 deadline 就执行；关键不是追求最大 batch，而是给等待一个显式上界。

组件级 batch 8 测量可以达到约 3.8–4.1 倍吞吐，端到端却没有稳定、可推广的收益。真实 frame arrival 不整齐，等待可能推迟首音；当 Code2Wav 尚非瓶颈时，组件收益也会被上游排队或已有 overlap 吸收，因此 bounded batching 能力合入后保持默认关闭，相关实现见 [PR #1126](https://github.com/sgl-project/sglang-omni/pull/1126)。

这一正一中性的结果暴露了新的矛盾：serial exact graph 命中稳定，eager batch 有组件效率，两者却使用不同 shape 空间。要获得端到端收益，调度单位必须和音频窗口协议共同设计。

### 5.2 用 chunk boundary 定义 batched graph 的合法 shape

Chunk-aligned scheduler 只消费完整窗口，让 batch 中每个 row 都落在 codec 可以解释的边界；随后捕获 batch size 2/4/8 的 graph。shape、capacity 或显存预算不满足时，系统拆成更小子批或串行路径，而不是把任意输入塞进最大图。

First-ready 窗口还可以绕过普通 inbox 排队。它不是把所有首窗都特殊化，而是承认首窗优化 TTFA、稳态窗口优化 throughput，两类工作应共享 correctness contract，却可以有不同的 scheduling priority。

在 2% graph memory budget、zero-wait 的生产形态下，c≥8 的 xRT/request throughput 提高约 11–15%，TTFA 保持相当或更好。同一 2% budget 下，更重的 wait policy 在 c16 以上提高约 41–51%，但伤害 c1/c8，并增加 queueing 风险。

5% full-graph control 捕获 B8 后达到约 53–55%，它证明更多 shape 有潜在上限，却不属于 2% production budget 结论。最终 `B1/B2/B4` pool 约占 634 MB；B8 超预算时优雅回退，显存是明确 capability boundary，相关实现见 [PR #1237](https://github.com/sgl-project/sglang-omni/pull/1237)。

这里不能把 41–51% 和 53–55% 写成同一个 headline。前者来自 2% budget 下的 heavy-wait profile，后者来自 5% full-graph control；zero-wait production 证据是 11–15%。三组数字回答的是不同的 latency/memory policy。

这一步的系统含义比“支持 B8”更重要：执行 shape 不再由模型任意 padding，而由可以播放的 chunk、允许等待的时长和可占用的 graph memory 共同约束。Fallback 也因此是协议的一部分，而不是命中率不够时的临时补丁。
