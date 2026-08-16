# dots3-note Preview：为了挑战 IMO，Infra 该做些什么？——学习计划

## 驱动问题

> **为了支撑 IMO 那样的递归自我批判，以及一条长达 40 到 50 小时的推理，infra 需要准备什么？**

出现位置：开篇第三段直接抛出。这个问题不需要概念铺垫，它是全文的组织轴——前半回答"场景长什么样"，后半回答"引擎该做什么"。

## 结构（两段式硬约束）

| 部分 | 内容 | 目标占比 |
|---|---|---|
| 开篇 | K1.5 引子 → 量级变化 → 抛出问题 → 声明信息缺口 | 6% |
| 一、场景 | IMO 的 Proof-Verify-Refine；TEMPO 如何训出自我批判；ARC-AGI-3 与测试时记忆 | 40% |
| 二、Infra | 两种长程负载的形状；512K 的 KV 账；双几何的引擎代价；40–50 小时的服务问题；小 batch 与 MTP；部署与稳定性；训练侧的状态保存 | 45% |
| 结语 | 归成三条 | 9% |

## 一、场景侧的展开顺序

1. **IMO 的解题过程**：不走形式化的动机与代价 → Proof / Verify / Refine 三环节 → 从 infra 角度指出这是"树状展开、前缀共享极强、prefill 很重"的负载（为第二部分埋线）
2. **TEMPO**：长程 RL 的两个困难（wall-clock 与统计，不是同一个问题）→ scalar value head 为何不够 → macro-step 作为 rollout 与优化的基本单位 → 生成式 critic 与特权信息（放置骑士）→ critic 的训练（TD target、GRPO 化、误差归一化、value warm-up）→ actor 兼任 critic（这里回扣 IMO 的 Verify）→ 附录的梯度等价性与重要性采样修正 → ARC-AGI-3 实验结果
3. **ARC-AGI-3 与测试时记忆**：线性超长的另一种形状；记忆是被任务长度逼出来的；点明"这个机制对 infra 的意义在下一节"

## 二、Infra 侧的展开顺序

1. **两种长程负载的形状对照表**（IMO 树状 vs ARC 线性）——这是第二部分的组织框架
2. **一条 512K 请求要多少 KV**：MLA 公式 → 三档对照（标准 MHA 1472 GiB / 全 full MLA 28.8 GiB / dots3 hybrid 8.19 GiB）→ 并发估算（8 卡 H200 约 43–46 条打满 512K）
3. **双几何在引擎里的代价**：`SWAKVPool` 的前提被打破 → 三处改动 → MTP draft 记账
4. **上下文裁剪与 radix cache 的冲突**（本文的核心 infra 论点，属自有分析）：单调增长时命中率近 100% → 裁剪改动靠近根的前缀 → 前缀树整棵子树作废 → 成本曲线是"廉价增量 decode + 周期性几十万 token 全量重 prefill" → 四个缓解方向，其中 memory 机制是训练侧给 infra 的红利
5. **小 batch 下的 decode 与 MTP**：并发只有几十 → 访存瓶颈 → MTP 收益被放大；DSA 只挂 full 层的道理
6. **部署 recipe 与稳定性**：watchdog、prefill 不进 graph、decode graph max bs 32 与真实并发同量级、H100 算不过来
7. **训练侧的 infra**：环境快照与恢复；重要性权重方差控制（材料里没有，标为未知）

## 资料来源

- [TEMPO 独立博客](https://studio.dots.ai/dots/tempo-blog.html)：macro-step、生成式 critic、TD target、value warm-up、附录梯度推导
- [IMO 2026 博客](https://studio.dots.ai/dots/imo-zh.html)：Proof-Verify-Refine、不走形式化的动机
- [主 blog](https://studio.dots.ai/dots/dots3-zh.html)：测试时记忆、评测脚注（NL2repo 的上下文裁剪策略是第二部分核心论点的证据）
- [VibeLifeBench](https://vibebench.github.io/VibeLifeBench_homepage/) / [VibeSearchBench](https://vibebench.github.io/VibeSearchBench.github.io/) 主页
- SGLang PR #33829（head `4a4746c`）与已合入的 docs commit `6ad3f2d`
- HF `dots-studio/dots3-note-prev` 的 config.json
- 技术报告尚未发布，训练数据配比等空白必须显式标注

## 行文约束

1. 禁止比喻与戏剧化动词。
2. 加粗只用于关键数字、关键论断和列表项引导。
3. 章节标题用平实名词或直述句。
4. 引述对方工作用"作者"。
5. 自有分析（KV 估算、radix cache 冲突）必须标注假设与"这是我的推断"。
6. 材料里没有的直接说不知道。

## 自检

- [x] 两部分配比达标（场景 39.7% / Infra 45.5%）
- [x] 场景部分为 infra 部分埋线（IMO 的树状负载、memory 机制）
- [x] infra 部分每个论点都落到具体数字或具体 commit
- [x] 核心 infra 论点（裁剪 vs radix cache）标注了推断性质
- [x] 无比喻、无戏剧化动词
- [x] 所有外链 200、内链存在
