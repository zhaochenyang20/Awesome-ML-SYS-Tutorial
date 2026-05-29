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

Ke Bai, Haoguang Cai, Shangming Cai, Qiujiang Chen, Chen Cheng, Jiaxin Deng, Wenyao Gao, Yifei Gao, Jingwen Gu, Chenchen Hong, Hao Jin, Xinli Jing, Shenggui Li, Xinyuan Liu, Xinyu Lu, Yuan Luo, Silin Men, Chengliang Qian, Jinjiang Qu, Dongming Sheng, Richard Wang, Shuwen Wang, Zijie Xia, Yuhao Yang, Xuesong Ye, Gaokai Zhang, Xiaoyu Zhang, Yichi Zhang, Huapeng Zhou, Chenyang Zhao

## 一、我们要优化的，是一个什么样的计算过程？

先把要优化的东西说清楚，因为它和大家熟悉的 LLM 推理，形态上就不一样。

一个标准 chat 模型的推理，是"一个模型、一次 forward、decode 出一串 token"——干净的一条线。而 SGLang Omni 面对的语音 / 音频生成类模型（包括纯 TTS 和带语音输出的 omni 模型），把这条线拆成了一条**异构子计算的流水线**。它大致分四段：把输入音频编码成离散 codec token（Audio Encoding），用一个足够强的模型理解输入并决定要说什么（Understanding），把内容合成成 codec token（Speech Synthesis），再把 codec token 还原成波形（Audio Decoding）。真正驱动架构分化的，是中间两段——理解模块和合成模块之间怎么耦合、合成模块自己用什么策略生成。

我们目前主力支持的两个模型，恰好代表了这两段最不一样的两种做法：

**Fish Audio S2 Pro** 是一个约 5B 参数的纯 TTS 模型，Dual-AR 结构。给定参考音频和目标文本，它直接生成匹配音色的语音：一个约 4B 的 Slow AR 沿时间轴逐帧生成语义 codec token，一个约 400M 的 Fast AR 在每一帧内把剩下的声学 codebook 层补齐，最后一个 ConvNet 声码器还原波形。它的 Understanding 和 Synthesis 是**串行嵌套在同一个 decode loop 里**的——Slow AR 出一个语义 token，Fast AR 紧跟着补 9 步，然后才进下一帧。

**Qwen3-Omni** 是一个端到端多模态模型，Thinker-Talker 结构。一个 30B-A3B MoE 的 Thinker 理解多模态输入、自回归生成文字，一个 3B-A0.3B MoE 的 Talker 把文字侧信号转成语音 codec token，再经 MTP 补码本、Code2Wav 合成波形。它的 Thinker 和 Talker 是**两个独立的 decode loop，靠 hidden state 流式相连、异步并行**。

两个模型放在一起，问题就尖锐了：它们架构差异巨大，但又共享相当一部分计算。**这条流水线里，哪些子计算可以被抽象成框架的通用能力、哪些必须留作模型特定的实现？** 如果抽象做错了，要么过细、维护成本爆炸，要么过粗、每接一个模型都要打补丁。

而要回答"哪些能统一、哪些必须分开"，唯一靠谱的依据，不是看这些子计算输出什么模态，而是看它们**算起来是什么样**——这就是第二个问题。

## 二、它的计算特性是什么？

要讲清楚"我们的模型计算特性上有什么不同"，得先有一个参照系。所以我们从最熟悉的东西讲起。

### 标准 LLM 的 decode 计算特性

一个普通 chat 模型在 decode 时，计算特性可以用五条概括，而现代推理引擎（SGLang 也好，别家也好）的整个内核，本质上就是把这五条优化到极致的产物：

1. **单一、同构**：一次推理就是一条 token 序列，一个模型、一个 decode loop、一个 scheduler 事件循环管着一批同构请求。
2. **显存带宽瓶颈**：decode 每一步都要把全部权重读一遍，只为产出一个 token，单步算术强度很低。所以吞吐不来自单条请求，而来自把很多请求 batch 在一起摊薄权重读取——这就是 continuous batching 存在的根本理由。
3. **KV cache 单调增长**：序列长度事先不可知、请求之间各不相同，KV 随 decode 一步步往上长，于是需要 paged KV cache，按 page 粒度分配、共享、淘汰。
4. **token-id 可索引**：每一步输入是一个离散 token id，不同请求共享同一段前缀时 KV 可复用——这是 RadixAttention / prefix cache 的全部前提。
5. **输出即输入，纯函数反馈**：第 t 步的输入就是第 t-1 步采样出的 token id，干净、确定。正是这个规律性，让 overlap scheduling 敢在采样当前 token 的同时去算下一步。

这五条彼此咬合，构成一个极其自洽的闭环。SGLang 的内核就是这个闭环的结晶。**请记住这五条，因为接下来要讲的模型，会逐条打破它们。**

### 模态不是轴，计算特性才是

在用这五条去切我们的模型之前，得先说清楚一件容易走错的事：分类的轴，是计算特性，不是输出模态。

市面上多数 omni 框架按输出模态分：出文字的、出语音的、出图的，塞进 "any-to-any" 一个大筐，配上 "AR 引擎 + Diffusion 引擎" 两套并列后端。听起来很全，但**模态相同的两个 stage 计算特性可能相反，模态不同的两个 stage 计算特性反而可能一样**。

一个 TTS 里的 flow-matching 声码器，和一个文生图里的 DiT，输出都跟"声音/图像"沾边、内部都用迭代去噪，按模态看是一类；但前者被 AR 帧时钟驱动、步数小且固定、帧局部，后者自己就是时钟、全局迭代、步数与内容无关——计算特性是反的。反过来，一个语音 talker 在 decode 时吐 audio codes，和 chat 模型吐 text token，模态完全不同，但外层那条 AR loop 的计算特性几乎同构。

所以判断该怎么调度，要看的是**这个 stage 里谁在驱动时钟**：是一条 AR loop 以帧率往前走（声码器之类都被 slave 到这个节奏、按窗口 flush），还是一个去噪循环自己就是节奏（整张 latent 全局刷新）。时钟决定计算特性，计算特性决定调度。这把尺子甚至会切出反直觉但正确的边界：一个纯 NAR、全 flow-matching、没有 AR 骨干的 TTS（如 F5-TTS），按模态它是"TTS"，按计算特性它的时钟是去噪循环、和图像 diffusion 同类，**不在**自回归语音这一类里。

### 回到我们的模型：哪里和那五条一致，哪里逐条打破

先说一个让人安心的事实：S2 Pro 和 Qwen3-Omni 的 **Understanding 段**，和那五条**完全一致**。S2 Pro 的 Slow AR（Qwen3-4B）、Qwen3-Omni 的 Thinker（30B-A3B MoE），在 decode 时都是单一、同构、显存带宽瓶颈、KV 单调增长、token-id 可索引的标准 AR。这一段，我们一行都不用重写。

分歧全部发生在 **Synthesis 段**，那五条在这里被逐条打破：

**打破"KV 单调增长"——定长、每帧重建的内层码本 AR。** S2 Pro 的 Fast AR、Qwen3 的 MTP，序列定长、每帧从零重建、用完即弃、跨请求零共享。它的计算特性是一个固定形状、确定跑 N 次的小 kernel，更像"每个外层步之后的一段定长后处理"，而不是一个 serving workload。paged KV 那一整套（动态增长、共享、淘汰）在这里收益为零——它要解决的问题根本不存在。也正因为形状固定，我们才能把它和 Slow AR 一起，用一张统一的 CUDA Graph 覆盖掉整个 decoding 过程。

**打破"token-id 可索引"——连续 embedding 输入。** S2 Pro 的 Slow AR 下一步输入是 MCF 聚合向量，Qwen3 的 Talker 输入是 codec embedding 之和加上投影出来的连续向量——都不是 token id。RadixAttention 按 token-id 序列匹配前缀，而它们的前缀根本没有 id 可匹配。标准内核里最值钱的那个优化，在这类模型上语义直接失效。

**打破"输出即输入的纯函数反馈"——step 内反馈。** 这一类模型，下一步输入依赖的是一个被二级网络（Fast AR / Code Predictor）加工、求和、再回灌的连续向量——**一个 decode step 内部就有数据依赖**。overlap scheduling 所依赖的那个规律性被打破了。

**打破"单一、同构"——双 LLM 异步互喂。** Qwen3-Omni 的 Thinker 和 Talker 是两个独立 LLM、两个独立 decode loop，靠 hidden state relay 异步并行；一个进程里同时活着三种生命周期的 KV（Thinker 的 radix+paged、Talker 的 paged-only、MTP 的 static）。标准内核是"单实例、单事件循环、管一组同构请求"，"两个异构 LLM 异步互喂"这个拓扑在它的世界观里没有对应物。

还有一条不在那五条里、却同样是计算特性、而且很漂亮的差异：**瓶颈资源的互补。** LLM decode 是显存带宽瓶颈（疯狂读写 KV），声码器那个 causal ConvNet 是算力瓶颈（密集卷积）。两者瓶颈在不同资源上，于是可以通过 CUDA MPS 在同一张卡上并行、互不抢占。按模态分类看不到这一点，因为模态不告诉你瓶颈在哪；按计算特性分类，这就是一个免费的调度机会。

一句话收住这个问题：**Understanding 段的计算特性 = 标准 LLM，于是复用；Synthesis 段的计算特性逐条偏离标准 LLM，于是必须另待。** 偏离的不是"它是不是 LLM"，而是它算起来是不是那五条。

## 三、我们为此设计了怎样的系统？

有了上面的对照，系统设计的第一个、也是最关键的决定——**为什么不直接去改 SGLang**——就有了一个干净的判据：

> 一个计算该进 SGLang，当且仅当它能被表达成"单一、同构、KV 单调增长、token-id 可索引"的 AR 序列，也就是上面那个标准 LLM 的计算特性。

Understanding 段满足，于是原样复用；Synthesis 段逐条打破这四个词里的某一个。改 SGLang 去容纳它们，是让通用内核为模型特定的偏离背锅：要么给 KV 池加 if 分支区分两种生命周期，污染那条被千万张 GPU 验证过的热路径；要么把 RadixAttention 的匹配键从 token-id 换成别的，开一个模型特定语义的口子——开一个，下一个模型又要另一种键，前缀树就烂成一堆 if。这也顺手回答了"为什么不直接拿一个现成通用 LLM 引擎来 serve"：任何为那五条计算特性优化的内核都会撞上同样几堵墙。这不是哪个引擎的偶然缺陷，是"通用 AR 内核"这个抽象的固有边界。所以它上面才需要专门有一层。

**于是 SGLang Omni 和 SGLang 的关系，只有一个正确答案：composition，不是 fork，也不是打补丁。** 我们的 `OmniScheduler` 把 SGLang 当成"一个把标准 LLM 计算特性 serve 到极致的引擎"原样复用——continuous batching、paged KV cache、RadixAttention、CUDA Graph、overlap scheduling，一行不重写；凡是计算特性偏离那五条的部分，全部收拢到 Omni 这一层。为了不让 composition 滑成 fork，这条边界有几条死规矩：pin 住版本而不是 track main；把 SGLang 的 `PrefillManager` / `DecodeManager` 当黑盒，只用 public 方法；需要新能力时优先推动 SGLang 上游开 hook，而不是在下游打 patch。这里有一个我觉得最漂亮的推论：**因为我们严格按计算特性分层，SGLang Omni 天然就站在 SGLang 的上层，而且是一个干净的上游贡献者**——当我们发现某个改进其实"任何标准 AR 序列都受益"，它就该被推回 SGLang main。计算特性这把尺子，不只切模型，也切清了哪行代码属于内核、哪行属于上层。下层保持纯粹、可被持续复用，上层承接所有偏离，两边都不脏。

落到代码上，SGLang Omni 是一个声明式的多 stage pipeline 框架。一个模型被表达成一张 stage graph，每个 stage 声明 `next` / `wait_for` / `gpu`，框架据此把拓扑静态推导出来。请求经过 `HTTP → Coordinator → Stage → Scheduler → ModelRunner → forward`：Coordinator 管请求生命周期和多终端合并，Stage 是纯 IO 壳（ZMQ 控制面 + relay 数据面，刻意分离、可各自替换），模型特定逻辑全部收在 ModelRunner + callbacks 里。而 ModelRunner 这一层，正是"按计算特性组织"的落地——每一类计算特性对应一个 runner：`ThinkerModelRunner` 给 Understanding 段（计算特性 = 标准多模态 LLM decode，复用 SGLang）；`FeedbackARModelRunner` 给"变长外层 AR + 定长内层码本 + 连续 embedding 反馈"这一类，Qwen3 的 Talker、Fish 的 Dual-AR 共享它，差异只落在三个 callback 里，接一个新的同类模型大体上就是写一个新的 `callbacks.py`。MOSS、Voxtral 这些 TTS 模型也都落在这一类——它们的 stage 分解各不相同，但 decode 计算特性同属一类，所以共享同一套调度抽象。

V1 的一个突出之处，是把过去要每个模型各自硬写的能力，抬成了框架级的一等公民：streaming（按帧 flush 的音频流、按 token 的文本流）、colocation（thinker + talker 同卡共存，按 SLO 与显存预算自动规划摆放，单卡也能跑整条语音 pipeline）、Router（整条 pipeline 复制做横向扩展，请求永不跨 replica 切分、Coordinator 边界完整）。新模型直接继承，不用重造。

最后值得一说的是：**这套按计算特性组织的设计，反而让我们能干净地支持 Ming-Omni、LLaDA-Uni 这种能生成图片的 Omni 模型。** 这看似和前面"把图像 diffusion 切出语音类"矛盾，其实是同一套哲学的胜利。按计算特性分类，从来不是说"SGLang Omni 只做自回归语音"，而是说框架必须围绕计算特性的"类"来组织、而不是焊死成单一引擎。图像生成确实是**另一个**计算特性的类——去噪循环驱动、固定步、算力瓶颈、没有 KV cache——但正因为框架按计算特性组织，多接这一类，做的事情是为它写一个 `DiffusionModelRunner`、挂到同一张 stage graph 上，而不是把它硬塞进 AR 内核、也不是和语音混成一类用同一个引擎将就。能干净地多接一个图像模型，不是因为我们什么都往里塞，而是因为我们一开始就按计算特性留好了扩展点。

往后看，这把尺子其实已经预示了 SGLang Omni 的未来形状。同样按"谁驱动时钟"去切，video 生成、world model 大概率会落到 diffusion 那一类、或催生一个带时间一致性约束的新类。未来不会是"一个引擎装下所有 any-to-any"，而是**少数几个由计算特性定义的类，每个类一个 runner、各有各的最优调度，上面共享同一套 stage 编排**。现在的现实是，每家发一个新的语音 / omni 模型，往往要自己 fork 一个推理框架、或另起一套服务栈，正是因为上游还没有这样一层标准的、按计算特性组织的编程模型。模态是会过时的表象——今天的 8-stage Qwen3-Omni 明天可能被压成 3-stage，有人在砍 stage、有人在加 stage；但"AR 时钟驱动一串定长码本、连续 embedding 在 step 内反馈、声码器在另一种瓶颈资源上并行"这些计算特性，只要自回归还是语音生成的主流范式，就不会变。我们把框架的关节，对准的是这些不变量，而不是某一张会过期的图。

这，正是我们想为之做对的那个计算过程，和那套系统。
