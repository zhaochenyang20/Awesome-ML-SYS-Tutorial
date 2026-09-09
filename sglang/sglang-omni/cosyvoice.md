# 以 CosyVoice 为代表的 AR + DiT + Vocoder 生成模型优化思路

过去几个月，我们观察到相当量的 TTS 模型采用了 AR + DiT + Vocoder 架构。DiT 的加入使得这些模型的音质相比不采用 DiT 方案的模型有了显著提升，但是这带来一些新的优化设计挑战。我们这里总结分享 SGLang Omni 对此类模型的优化思路，主要是 [issue 1652](https://github.com/sgl-project/sglang-omni/issues/1652) 所 track 的各阶段优化，同时我们目前仍旧在进行对于 CosyVoice 通过 Runtime Optimization Agent 进行的进一步优化。总结思路的同时，我们也会把这套方法论扩展到更多 AR + DiT + Vocoder 架构的模型上，譬如 [dots.tts](https://github.com/studio-dots-ai/dots.tts) 和 [Minimax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3)。

## Runtime Profiling

对于真实场景的需求进行一次压测，覆盖离线/实时；短句/长句；有参考/无参考等多种需求，在此基础上针对每一个阶段进行运行时长的dump，这项工作由agent完成效率极高，本质上是一项埋点任务。
一般我们从这个阶段可以明确得到各个场景下推理花费时间的信息。
根据这些信息进行下一步的，针对性的推理优化。
按照个人经验而言，早期的大部分效率下降都是由“调度器中请求排队”造成的。

【这一段其实写得不太好，因为你并没有讲清楚怎么去做这个 profile，只是讲了要做 profile 这件事情。可以看一看之前大家做过的 profile 大概是什么样子的，或者和 xuxiang 老师对齐一下，能不能直接用他们 developed 的 agent 来做这个 profile。】

【https://github.com/sgl-project/sglang-omni/pull/1850/changes 逻辑上，我们应该要讲到用这个 profile 的 skills 来做 profile。然后讲讲我们没有开源 harness，出于 xxx 原因，需要用户来向我们申请 harness 权限。】

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
