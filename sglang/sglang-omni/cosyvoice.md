# 以 CosyVoice 为代表的 AR + DiT + Vocoder 生成模型优化思路

过去几个月，我们观察到相当量的 TTS 模型采用了 AR + DiT + Vocoder 架构。DiT 的加入使得这些模型的音质相比不采用 DiT 方案的模型有了显著提升，但是这带来一些新的优化设计挑战。我们这里总结分享 SGLang Omni 对此类模型的优化思路，主要是 [issue 1652](https://github.com/sgl-project/sglang-omni/issues/1652) 所 track 的各阶段优化，同时我们目前仍旧在进行对于 CosyVoice 通过 Runtime Optimization Agent 进行的进一步优化。总结思路的同时，我们也会把这套方法论扩展到更多 AR + DiT + Vocoder 架构的模型上，譬如 [dots.tts](https://github.com/studio-dots-ai/dots.tts) 和 [Minimax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3)。

## Runtime Profiling

Profiling 的目的不是产出一张 trace，而是回答一个问题：下一步应该优化哪里。AR、DiT、Vocoder 和调度器对应的优化路径不同，没有先定位瓶颈就动手，很容易花大量时间优化局部 kernel，最后端到端看不到收益。

我们把这个过程整理成一套五层的 runtime profiling 方法论，在 [PR 1850](https://github.com/sgl-project/sglang-omni/pull/1850) 中实现为 `model-profiling` skill。`.claude/skills/model-profiling/METHODOLOGY.md` 是方法论的 source of truth，[issue 1798](https://github.com/sgl-project/sglang-omni/issues/1798) 是各模型 profiling 记录的索引。

### 开始之前

先固定两样东西。

- **Workload**：短句和长句、离线和流式、有无参考音频、并发数，这些形状下的瓶颈可能完全不同，要分别测量。
- **环境**：记录代码 commit、checkpoint 和数据集的 revision、GPU 型号、PyTorch 和 SGLang 版本。共享机器上还要给 server 和压测客户端绑定互不重叠的 CPU 核，确认测量期间 GPU 没有被别的任务占用，并且在采信任何“稳态”数字之前先跑过一轮真实规模的负载。

### 五层

这五层不是固定的 1 到 5 顺序。先做第一层，再根据结果决定往哪里深入。

1. **GPU 忙不忙。** 先判断系统是 GPU 算力受限，还是 CPU 和编排受限。可以用服务自带的 `/start_profile` 拿一段 torch trace，或者在压测期间用 `nvidia-smi`、DCGM 采样 GPU 利用率。解析 trace 时注意 CUDA Graph replay 在不同 PyTorch 和 CUDA 版本里的表示不同，不能假设所有 GPU 活动都出现在普通 kernel 事件里，只算 kernel 会明显低估忙碌率。多种测量都显示忙碌率明显偏低，就继续看 CPU 侧。GPU 已经接近饱和，CPU 侧的调查通常收益很低，应转向 kernel 层面的优化。
2. **CPU 时间花在哪里。** 在真实负载下用 `py-spy record --idle --subprocesses` 采样，定位占比最高的叶子帧，重复采样确认结果稳定。这一层要得到的不是一张火焰图，而是一个可以验证的假设，比如“某个线程大部分时间停在 eager Python 分发上”。
3. **更高并发能不能把 GPU 喂饱。** 固定其他条件逐步提高并发，同时记录 GPU 利用率、吞吐、延迟和质量指标，判断增加并发是真的提高了利用率，还是只增加了排队和尾延迟。
4. **A/B 验证。** 一次只改一个变量，两边使用完全相同的 workload、预热和 GPU 环境。对于个位数百分比的收益，先在同一协议下做一次 A/A，量出这台机器的噪声底。共享机器漂移明显时，优先用两台同时在线的 server 交替压测，而不是每臂重启的 A/B。
5. **功能回归。** 性能提升之后仍然要检查 WER、说话人相似度等质量指标。修改默认值、调度逻辑、CUDA Graph 或 attention backend 的优化，还要在长序列或结构不同的数据集上重新验证，因为不同长度和结构可能走完全不同的代码路径。

### CosyVoice 走一遍

Fun-CosyVoice3 的 profiling（[issue 1883](https://github.com/sgl-project/sglang-omni/issues/1883)）展示了这套方法如何从测量走到优化假设。

并发 16 下，torch trace 给出的 GPU 忙碌率是 17.9%，vocoder 占请求时间的 89.5%。py-spy 采样三次结果一致，`scheduler-vocoder` 线程 79% 到 87% 的样本停在 Flow DiT 的 eager Python 分发上，等待几乎为零。由此形成一个具体假设：瓶颈不是 GPU 算不动，而是 host 侧发不出去。

随后 [PR 1969](https://github.com/sgl-project/sglang-omni/pull/1969) 对 DiT 的 `torch.compile` 做了配对 A/B：两台在线 server 在同一张卡上交替压测，九对有效实验，吞吐中位数在并发 1 提升 24.6%，并发 16 提升 59.0%，吞吐在九对里都为正。同一台机器、同一协议下做了 19 对 A/A 校准，中位数差在并发 1 约 4%、并发 16 约 6% 以内，两个收益都明显超过测得的噪声底。质量检查用 Seed-TTS EN 的 96 个 clip，corpus WER 为 0.66% 对 0.85%，差异来自一个 clip 里 whisper 把“sky jumper”写成了“skyjumper”，两臂的词内容相同。

这里重要的不是 `torch.compile` 本身，而是整条路径：测量，定位，形成假设，校准噪声，控制变量的 A/B，最后验证正确性。

### 用 skill 跑

在 sglang-omni 仓库根目录的 Claude Code 里运行：

```
/model-profiling <model>
```

skill 先检查 benchmark 入口、空闲 GPU 和 profiling 依赖，在任何 GPU 工作开始之前停下来请求确认。首次运行先做发现阶段：从第一层开始，按路由规则决定要不要跑第二、三层，形成具体的第四层假设之后再次停下来等人确认，确认后才继续进入 A/B。每次运行的工作产物保存在被 gitignore 的 `.profiling-runs/<model>/` 下，长期记录是 1798 下对应模型的子 issue。Qwen3-Omni 的 [issue 1914](https://github.com/sgl-project/sglang-omni/issues/1914) 是一个只做发现阶段的例子。

### 关于 harness

第四层用来做 A/A 校准和配对 A/B 的完整实验 harness 目前还没有公开。它还包含给 profiling agent 出题打分的 exam 基础设施，评分逻辑需要和被评估的 agent 隔离，runner 也还依赖我们自己的机器和容器环境。需要使用这套 harness 的读者可以联系我们申请权限。

Profiling 最终要给出一个路由决定：瓶颈在 AR 进入 AR 优化，在 DiT 进入 DiT 优化，在 vocoder 进入 Vocoder 优化，跨阶段的排队和同步问题则进入调度器优化。

## AR 优化

主流的 AR 部分架构都是 Qwen，近乎可以当做 Qwen LLM 来优化。启用 SGLang Omni 中的 Omni Scheduler，利用 Radix Attention 和 Continue Batching 等优化手段，可以显著提升 AR 部分的吞吐量。然后，进一步去考虑 torch.compile / CUDA Graph / TRT-LLM 等优化手段，可以进一步提升 AR 部分的吞吐量。

【TODO：我比较才疏学浅，我印象中，TRT-LLM 其实是一个 inference 框架，你这里说的是用 tensorRT 优化吧。把这几个优化写的明白，然后给出参考的代码段 or PR。我比较推荐给出参考的代码段（比如说，为了实现 CUDA Graph，重要的代码修改可能就是那么一百行。），你可以在 GitHub 上面选定某一个固定的 commit，然后分享那个 commit 上面的几个代码 block，这样的话用户看着会比较明白。】

## DiT优化

比较简洁的架构可以考虑TRT (>torch.compile)进行处理，缺点是对于动态的axis和请求没有那么灵活，需要引入一些音频前端处理方式。
DiT的一项算法优化就是从ODE处下手，注意到ODE求解的过程中，每一步对总体的贡献并不一致，可以考虑裁剪掉贡献较低的步数，或者直接训练一个步数蒸馏的lora。
Multi-lora是一个很不错的feature，虽然目前还没支持，但是它针对用户的场景是极大的便利。
## Vocoder优化
Vocoder的结构一般是因果卷积（casual for streaming），在模型上并没有transformer那么好处理，当然本身的速度也不慢，因此推荐直接compile了事。有条件的也可以考虑直接构建TRT Engine获取更高效率的吞吐，当然最主要的还是激活vocoder的batch inference。
## 调度器优化
和第一阶段任务结合到一起，针对自己的真实场景进行压测，测试ar batch -> dit batch / dit chunk -> vocoder batch分别在多少的时候能够取得RTF的平衡，最终优化目标是希望在一定的并发下实现p95/p99 RTF<1的同时不会出现质量劣化。
观测到CosyVoice在引入stream之后产生了速度的下降，排查之后发现因为stream本身切分过多，导致请求量超过了vocoder的并发，陷入了P1中我们说的排队状态，因此不能简单的认为，只是改了一种生成方式就算OK了。
## KDA
工程层面的优化全部apply之后，就可以研究数学层的优化了，当然我数学不太好，所以这个丢给Agent自己慢慢研究就完事了。如果只是rope fusion这种简单算子应该不会太困难（
## 测试
初中我们就学过控制变量法，在做推理优化的过程中，我们最好保证测试集涵盖自己业务场景的大部分情况以及一些corner case，针对每一次改动都精确地记录信息和数值变化，也方便省后续agent的token，防止它反复测试已经验证过不太行的方案。
