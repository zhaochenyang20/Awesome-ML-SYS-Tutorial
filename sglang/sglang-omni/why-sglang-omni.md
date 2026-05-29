# SGLang Omni：从 decode 计算特性出发，重新设计多 stage 生成模型的推理框架

从今年 2 月底开始，我个人的工作重心从 SGLang RL 移到 SGLang Omni。坦诚说，在工作的一开始，我并没有多少底气。熟悉我的人都知道，我并不是系统方向出身，甚至在很长一段时间里，对系统类的工作有一种本能的畏惧。我曾经在朋友圈里写过这样一段话：

> 犹记得本科低年级时，我对系统课程有着极深的抵触。入学第一个学期被动态规划折磨到觉得自己在计算机这一行毫无可能，第二年在计算机系统概论课上的理解困难，更让我对系统的庞大学问产生了一种近乎逃避的心理。恰好那时接触了粗浅的人工智能课程，在玩具工程上找到了一点投机取巧式的成就感，便天真地以为自己可以绕开所有系统，去做那些看起来更“智能”的事情。
> 
> 直到大学三年级，我在美国暑研期间，才在导师和指导我的博士生的带领下，通过我人生的第一个开源项目，开始理解从硬件到软件，系统的方方面面。计算机系统是一种思维，是随时随刻都在思考如何让自己的工作被他人认可，简单易用，能扩展到更大规模，带来更大影响。由衷感谢那个暑假指导我的导师 Graham 和 Sherry，还有耐心负责的 PhD Vijay。让我第一次意识到了系统之美。
>
> 而意识到系统的强大，则一直等到了本科毕业之后。坦诚说，知道了系统的美妙之处后，我本科的系统学习也并没有付诸更多心思。我至今提起都会觉得羞愧的是，我的大多数硬核系统课程，譬如计算机组成原理、计算机网络原理、操作系统，这些都是在我绩点自由的大四修完的。若干年后想起来，每天都在和这些基本功打交道的我，其实这些基本功都没有好好学过。我都还记得本科毕业前其实可以选择章明星老师的一门选修课，但是那课对我当时的状态实在是不可能完成的。一年后加上章老师微信，我都惭愧到不好意思给他提起这件事情。

正是这段不光彩的本科生涯，让我在踏入系统研究之后，始终带着一份小心翼翼的自省。我很清楚，自己在基本功上有所欠缺，基本每向前走一步，都需要去补上过去欠下的功课。事实也是如此，从 SGLang RL 到 SGLang Omni，我们处理的问题看上去完全改变了。以前我们会去处理显存的 offload 和 upload，研究怎么高效地在训练引擎和推理引擎中间去传递参数。现在，对 Omni，我们考虑不同 stage 之间的 fusion，思考如何用一张统一的 CUDA Graph 去覆盖整个 decoding 过程。这是一个非常有趣的变化，我几乎开始从事和以前完全不相关的内容，以至于有时候和朋友提起，我去做 SGLang Omni 了，总会有人诧异，怎么开始做了完全不相关的事情。

对此，我的想法是，一方面，我很早前就分享过 SGLang RL 小组的早期成员 Junrong 的观点：“我们 RL 系统的优化往往 block 在我们对推理系统本身的认知”，所以进一步去理解推理系统，对我喜欢的 RL 工作本身，也大有裨益；另一方面，Omni 模型本身也存在 RL 需求，譬如 TTS 和 Qwen3 Omni 模型的 Thinker，这些组件都具有明确的 reward 目标。

其实，更重要的一件事情是，机器学习系统要处理的核心任务和方法都是高度统一的。看上去 RL 中的参数 offload 与 refit，和 Omni 中的 stage fusion 毫无关联，实际上都是在为特定的计算拓扑设计更合理的结构，并在此之上做极致的通信与调度优化。SGLang Omni 所面对的模型虽然和 LLM 差别显著，但解码阶段的核心计算特性高度相通，用到的优化思路也常常殊途同归。在我看来，机器学习系统研究者只有一个目标，研究指定计算过程的计算特性，并且针对其计算特性设计高效鲁棒的系统。

因此，这篇文章既是我们 SGLang Omni 项目组对当前技术框架和阶段性工作的系统总结，也是我们想认真回答一个根本问题：我们究竟要优化一个什么样的计算过程？它的计算特性是什么？我们又为此设计了怎样的系统？

此外，幸运的是，这次我仍旧不是一个人在探索。从 SGLang RL 社区开始，我们有了越来越多的伙伴，尽管大家可能因为工作和生活的变故，并不会永远坚守在开源社区的一线，但是社区强大的凝聚力和打满每一张 GPU 的热情，永远在此。这一次，我还是幸运地遇见了同样优秀的伙伴们，我们一同来重新审视 Omni 模型的推理过程，并为此设计扎根于计算特性的推理框架。SGLang Omni 还在热烈的开发过程中，我们也永远欢迎和我们志同道合，对机器学习系统有着近乎偏执的美学追求的朋友加入我们。

致谢）（按照姓氏拼音排序）：

Ke Bai, Haoguang Cai, Shangming Cai, Qiujiang Chen, Chen Cheng, Jiaxin Deng, Wenyao Gao, Yifei Gao, Jingwen Gu, Yitong Guan, Chenchen Hong, Hao Jin, Xinli Jing, Shenggui Li, Junrong Lin, Xinyuan Liu, Xinyu Lu, Yuan Luo, Silin Men, Ratish P, Chengliang Qian, Jinjiang Qu, Dongming Sheng, Shuai Shi, Chao Wang, Richard Wang, Shuwen Wang, Zijie Xia, Yuhao Yang, Xuesong Ye, Fan Yin, Gaokai Zhang, Xiaoyu Zhang, Yichi Zhang, Huapeng Zhou, Chenyang Zhao

## 我们要优化怎样的计算过程？

早在去年 4 月份，Qwen 团队第一次发布 Qwen 2.5 Omni 的时候，我们就尝试支持过 Qwen 2.5 Omni 在 SGLang Main 中。从模态上来感受，Qwen 2.5 VL 和 Qwen 2.5 Omni 的输入是非常类似的，可以理解为，Qwen 2.5 Omni 的 thinker 就是在 Qwen 2.5 VL 的基础上增加了音频输入。逻辑上，抛开 audio encoder 采用的具体技术路线，可以抽象立即为 Qwen 2.5 Omni 的 thinker 就是更强大的，更多模态的 Qwen 2.5 VL。但是很遗憾，Omni 模型的技术路线不单单如此，这里又涉及到另一个问题，Omni 模型的定义尚不统一。


在部分模型团队的定义中，Omni Model 可以算作支持音频输入的 VLM，比如 [MiMo Omni](https://mimo.xiaomi.com/mimo-v2-omni) 和 [Nemotron Omni](https://arxiv.org/abs/2604.24954)。而在 Qwen Omni 团队的定义中，Omni Model 不单单支持音频输入，更支持音频输出。我经常举的一个例子其实是 GPT-4o 和 Google 的 Gemini App。例如，假设我们在超市里面买菜，打开 Gemini App 的摄像头模式，这些模型一边接受着摄像头里面的视频输入，与此同时，它们还接受我们的语音输入。而输出端，我们能够看到屏幕上出现文字输出，并且也有对应的语音输出。从输入输出模态上，Qwen 所代表的 Omni 系列支持「语音 + 文字 + 视频 + 图片」的输入，并且支持「文字 + 语音」的输出。这样的 Omni 定义其实也非常常见，譬如特斯拉上的 Grok APP 和豆包的手机 APP。涵盖面最广的 Omni 模型仍旧不止于此，来自蚂蚁团队的 [Ming Omni](https://arxiv.org/abs/2506.09344) 模型，做到了全模态的输入和输出，为了能够兼容文字和图片的输出，其输出部分同时支持 auto regressive decoding 和 denoising/diffusion。除此之外，还有素来与 SGLang 团队合作极多的 LLaDa 团队推出的 [LLaDa Uni](https://github.com/inclusionAI/LLaDA2.0-Uni) 模型，和 Ming Omni 一样拥有令人惊叹的全模态能力。

对于如此多种的 Omni 模型定义，我自己并无偏好，各家模型团队都为此付出了巨大的努力，每一种定义在各自的语境下、对算法研究者而言都成立。只是对我们做推理系统的人来说，按输入输出模态来划分不是一个特别合理的视角——模态告诉我们模型对外的封装是什么样，却没有直接表现出它的计算过程。如同我提到的那样，机器学习系统研究关注计算过程本身，Auto Regressive Decoding 是一个如此统一优雅的建模方法，用模态来划分，则丧失了部分美感。

所以，换一个角度：什么样的模型，是 SGLang Omni 设计之初想服务好的目标？我们的依据不是模态，而是模型的 decoding 过程是不是 multi-stage 的。这里划分的始终是计算过程，而非模型的好坏——下面提到的每一个模型，背后都是一支团队扎实出彩的工作，我们在此只是选择沿着计算过程的维度，用 multi-stage decoding 的方式来进行分类。

我们以 Qwen3-Omni 为例，来解释 multi-stage decoding 的大致流程（详细过程可以参考[我先前的博客](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/transformers/omni/readme.md)）。Qwen3-Omni 生成一条同时带文字和语音的输出，不是一个 decode loop 自回归地从头到尾，而是几个异构的 decode stage 的交替：Thinker 自回归地生成文字，Talker 自回归地生成每个 timestep 的第 0 个 codec token，再由 MTP 补全该 timestep 余下的 codec token，几个 decode stage 构成异步流水线。在 SGLang Omni 的设计理念里，Qwen3-Omni 的关键不在于多模态，而在于 end 2 end decoding 被拆成了计算过程各异的多个 stage，这就是 multi-stage decoding。

有趣的是，multi-stage decoding 并不是 Omni 模型的专利，许多 TTS 模型同样采用了 multi-stage 的 decoding 过程。比如 FishAudio 的 S2 Pro，用一种被称为 Dual-AR 的双段流水来生成语音：一个约 4B 的 Slow AR 沿时间轴逐帧生成语义 token，一个约 400M 的 Fast AR 在每帧内补全声学 codec token，两个 stage 串行嵌套。它们没有理解多模态输入的使命，但 decoding 一侧的 multi-stage 结构，和 Qwen3-Omni 异曲同工。

从“decoding 是否 multi-stage"的视角出发，各个团队的模型自然地落在两侧，各自有最适合它们的推理方式。

一侧是 single-stage decoding。MiMo Omni、Nemotron Omni 这样的模型，能理解音频、视频、图片等丰富的输入，输出则为文本，decoding 就是一条标准的 LLM / VLM decode loop；Qwen ASR、MiMo ASR 接收音频、输出文字转写，同样如此。这些都是非常强大的模型，而它们的 decoding 恰好是 SGLang 主线最擅长的形态——SGLang 本身已经能把它们的推理做到极致，并不额外需要排布 multi-stage pipeline 这一层。

而另一侧是 multi-stage decoding，SGLang Omni 想要补全的拼图。Qwen3-Omni、许多 TTS 模型（S2 Pro、Qwen3-TTS、Voxtral、Higgs 等），以及做到全模态输出的 Ming Omni、LLaDa Uni，它们的 decoding 都被拆成了多个异构 stage。这些模型同样凝结着众多团队令人敬佩的工作，它们共有的那套 multi-stage 计算过程，正是我们设计 SGLang Omni 的初衷。

无独有偶，diffusion 模型也能按此分类清楚。像 Wan、Qwen-Image 这样出色的开源图像生成模型，它们的 decoding 是单独一条 denoising loop，没有第二个 decode stage 接力，属于 single-stage；SGLang Diffsion 已经有专门面向 diffusion 的领先推理方案，所以它们不在 SGLang Omni 的设计目标中，哪怕"输出图像"这件事听上去很 Omni。反之，Ming Omni 里也用到了 diffusion，但 Ming 的 diffusion block 只是其 multi-stage 流水线中的一个 stage（AR 骨干先生成，再分流到 audio decoder 或 diffusion image decoder），所以 Ming Omni 是 multi-stage 的，是 SGLang Omni 的目标。在 SGLang Omni 依照计算过程而划分的设计理念中，决定一个模型的归属，不是用没用 diffusion，而是 decoding 是不是 multi-stage。同样是图像生成模型，在纯图像生成模型里它就是全部，在 Ming Omni 里它只是一个 stage，所处的计算过程截然不同。我们也非常为此感到欣慰，SGLang Omni 的设计目标是统一而精准的。

最后还有一条有趣的经验法则：带有音频输出的模型，往往会落在 SGLang Omni 的支持范围里。因为生成语音的主流技术路线，几乎都会把语音的 decoding 拆成 AR 骨干 + codec completion 这样的多个 stage，于是音频输出和 multi-stage 高度相关。但我始终认为，计算过程才是机器学习系统设计的本质出发点。机器学习系统研究者与算法研究者可以有着不同的美学追求。

【TODO】

它的计算特性是什么？
要说清楚 multi-stage decoding 的计算特性，最好先回到我们最熟悉的那个东西：一个标准的 chat 模型，在 decode 的时候到底是什么样子。
一个普通的语言模型生成回复，从头到尾就是一条 token 序列，一个模型、一个 decode loop，每一步读一遍权重、吐一个 token。它的计算特性其实相当规整。算力上，decode 是典型的显存带宽瓶颈，每生成一个 token 都要把整套权重从显存里搬一遍，单步几乎谈不上什么算术强度，于是把许多请求攒在一起做 continuous batching、摊薄那次搬运，就成了提升吞吐最自然的办法。显存上，序列会一直往后长，长度事先不知道、请求和请求之间也参差不齐，所以才有了 paged KV cache，按页分配、按页共享。再有，每一步喂进去的输入，就是上一步采样出来的那个离散 token id；既然是 id，不同请求只要前缀相同，KV 就能复用，这正是 RadixAttention 的依据；又因为下一步的输入完全由这一步的输出决定，规律得近乎机械，引擎才敢一边采样当前 token、一边把下一步预先算出来，把 overlap 也叠上去。
可以说，SGLang 这样的引擎，整个内核就是围绕这一组特性——单一、同构、序列单调变长、输入是离散 id——把各种优化叠到极致的产物。它服务标准 LLM 之所以又快又稳，恰恰因为这组特性足够规整、足够统一。
而 multi-stage 模型的麻烦，也正出在这里：它的合成段，会把上面这组规整一条一条地打破。
我们还是回到 S2 Pro 和 Qwen3-Omni。它们的理解段其实没什么特别，Slow AR 也好、Thinker 也好，decode 时就是一个标准的自回归 LLM，上面那组特性原封不动，SGLang 现成的能力拿来就能用。真正不一样的，是合成段。
先看码本补全那一截，也就是 S2 Pro 的 Fast AR 和 Qwen3-Omni 的 MTP。它在一帧之内沿着码本深度方向把剩下几层 codec token 补齐，序列是定长的，每进入新的一帧就从头重建、用完即弃，请求与请求之间也没有任何可共享的前缀。这跟“KV 不断变长、跨请求共享”的假设正好反着来。paged KV cache 那一整套动态分配、淘汰、共享的机制，放在这里一点用都没有，它要解决的问题压根不存在。换个角度看，也正因为它形状固定、步数固定，我们反而能把 Slow AR 这一步和 Fast AR 那几步一起，用一张 CUDA Graph 整个罩住，这也是我在开头提到的那件事。
再看输入。标准 LLM 下一步喂的是一个 token id，而 S2 Pro 的 Slow AR 喂的是上一帧十个 codec token 聚合出来的一个连续向量，Qwen3-Omni 的 Talker 喂的是 codec embedding 之和、再加上从 Thinker 投影过来的连续表示，都不是 id。没有 id，RadixAttention 按前缀匹配那套就无从谈起，内核里最值钱的优化在这类模型上直接失效。更微妙的是，这个“下一步的输入”还得等二级网络（Fast AR、Code Predictor）把这一步的结果加工、求和、再回灌回来，也就是说，一个 decode step 内部就藏着一层数据依赖，那种“输入完全由上一步输出决定”的规整也没有了。
最后，Qwen3-Omni 干脆是两个独立的 LLM。Thinker 和 Talker 各跑各的 decode loop，靠 hidden state 异步地喂来喂去，一个进程里同时养着好几套生命周期完全不同的 KV。而 SGLang 的 scheduler，骨子里是“一个实例、一个事件循环、管一批同构请求”的设计，“两个异构 LLM 异步互喂”这种拓扑，在它原本的世界里没有对应的概念。
还有一处差别不在上面那组特性里，却很能说明问题：LLM 的 decode 是显存带宽瓶颈，声码器那个 ConvNet 却是算力瓶颈，两者卡的是不同的资源。知道这一点，就能让它们在同一张卡上用 MPS 并行跑、互不相扰。这是只盯着模态看不出来、只有盯着计算过程才拿得到的一点便宜。
把这些摆在一起，结论其实挺朴素：multi-stage 模型的理解段，和标准 LLM 是同一种计算特性；它的合成段，则是另一种、甚至好几种计算特性。它偏离的不是“算不算 LLM”，而是“算起来还守不守那组规整”。
我们又为此设计了怎样的系统？
知道了这些差别，第一个、也是最关键的决定，反倒是一个“不做什么”的决定：不去改 SGLang 的内核。
道理上一节其实已经讲完了。SGLang 内核的价值，恰恰在于它只服务一种特性——单一、同构、KV 单调变长、输入是 id——并把它优化到极致。一旦为了塞进合成段那些偏离，去给 KV 池加一个区分两种生命周期的分支，或者把 RadixAttention 的匹配键从 token id 换成别的东西，受伤的是内核里那条最关键、被大量线上流量反复验证过的热路径；每多迁就一个模型，就多开一个口子，改着改着，复用就成了 fork。而且这跟用哪个引擎无关，任何为“单一同构 AR 序列”优化的内核都会撞上同样的墙，这不是某个实现的缺陷，而是“通用 AR 内核”这层抽象本身的边界。所以合成段那些计算特性，本就不该住在内核里，它们要的是内核之上的另一层。
SGLang Omni 就是这一层。它和 SGLang 的关系我们想得很清楚：是 composition，不是 fork。理解段照旧交给 SGLang，continuous batching、paged KV、RadixAttention、CUDA Graph，一行都不重写；偏离那组特性的部分，全部收在上面这一层里。为了让这条边界不滑成 fork，我们也立了几条规矩：钉住版本而不是追 main，把 SGLang 的几个核心管理器当黑盒、只碰公开接口，需要新能力时优先去 SGLang 上游加 hook，而不是在下游打补丁。也正因为分层是顺着计算特性切的，这条边界格外干净：哪天我们发现某个改进其实对任何标准 AR 都有好处，它就该回流进 SGLang 自己。属于内核的留在内核，属于编排的留在上层，两边都不脏。
落到具体形态，SGLang Omni 把一个模型描述成一张 stage graph，每个 stage 声明自己接在谁后面、放在哪张卡上，框架据此把整条流水线推导出来。一条请求从 HTTP 进来，由 Coordinator 调度、若干 Stage 接力，每个 Stage 背后挂一个 scheduler 和一个 model runner；stage 之间的控制信令走一条轻量通道、真正的张量走另一条，两者分开、各自可换。而真正承载“按计算特性分类”的，是 model runner 这一层。理解段那种标准 LLM decode，走一类 runner，直接复用 SGLang；像 Fast AR、MTP 那种“变长外层、定长内层、连续反馈”的合成段，走另一类 runner，模型之间的差异收敛成几个回调，接一个结构相近的新模型，大体上就是把这几个回调重写一遍。S2 Pro 这样的纯 TTS 是三段流水，Qwen3-Omni 这样的 Thinker-Talker 是更长的一条，但它们落在同一套框架里，相同的部分复用、不同的部分各自归位。开头说的 stage 之间的 fusion、统一的 CUDA Graph，也都是在这套结构里才谈得上的事。
再往上，一些过去得为每个模型单独写的能力，在 V1 里被提成了框架自带的：流式输出、把本来分散的 stage 收拢到同一张卡上共存（免得小模型把一张大卡空着）、以及整条 pipeline 成副本地横向扩展。新接的模型直接就有，不用再各写一遍。
也正是这套按计算特性组织的结构，让我们能干净地接住 Ming Omni、LLaDa Uni 这种会生成图像的模型。图像那一段的去噪循环，是和自回归完全不同的一种计算特性，但因为框架本来就是按计算特性分 runner 的，多接它要做的，无非是再加一类 runner、把它挂到原来那张 stage graph 上，而不是硬塞进 AR 内核、也不是跟语音混作一类去将就。第一节里说 Ming 的 diffusion 只是它流水线里的一个 stage，落到系统里，就是这么一个 runner 的事。
写到这里，其实又绕回了开头那句话。从 RL 到 Omni，从参数的 offload，到 stage 的 fusion，表面上换了一批名词，骨子里做的还是同一件事：先看清要优化的到底是个什么样的计算过程，它的计算特性在哪里和已有的不一样，再为这点不一样去设计系统。SGLang Omni 今天服务的，是 multi-stage decoding 这一类；往后它会服务的，是所有那些在“还守不守那组规整”上和标准 LLM 分了岔的模型。模态会变，今天的八段流水明天也许压成三段，但只要还有一类计算过程值得被单独认真对待，这件事就值得接着做下去。