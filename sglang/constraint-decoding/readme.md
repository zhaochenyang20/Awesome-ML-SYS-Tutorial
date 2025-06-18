# 一文理解 Constraint Decoding 概念、方法与优化

【[English](readme_en.md) | [中文](readme.md)】
这篇文章是我和 [Shuai Shi](https://shuaills.github.io/) 在写 sglang 的 [constraint decoding tutorial](https://docs.sglang.ai/backend/structured_outputs.html) 时的学习笔记。SGLang 的文档专注在用法上，为此我们另外写一篇文章来阐述 constraint decoding 具体的概念和优化方法。

## 概念

constraint decoding 和 structed output 是同一个概念的一体两面，后者表达的是效果，希望模型的输出符合特定的结构，而后者则描述了方法，在模型的 decoding 阶段加以约束。当然，更早的一个说法是 json decoding，但是显得非常狭隘了，因为理论上任何 CFG（Context Free Grammar 上下文无关文法）都可以被 constraint decoding 所表达，而 CFG 的表达能力远超 JSON。总之，在非常多的场景下，我们希望模型能够输出特定的结构，便于我们后续 parsing，而实现这一目标的方法就是 constraint decoding。

## 基本原理

很显然，最简单的实现方式就是 token probability 的约束，即在 decoding 阶段，根据语法规则来直接约束 token 的概率，不符合规则的 token 概率直接置零（接近 0）。一个经典的流程如下：


1. **定义约束规则：** 这些规则可以是各类 CFG，也可以是一些自定义的逻辑规则。
2. **生成候选 token（Decoding）：** LLM 进行一轮 decoding，得到候选 token 列表。
3. **约束检查（Mask Gen）：** 根据预定义的约束规则，对候选 token 列表中的每个 token 进行检查，判断其是否符合规则。
4. **过滤（Apply Mask）：** 将不符合规则的 token 的概率设置为 0（或一个极小的值），从而将它们从候选 token 列表中排除。
5. **采样：** 根据过滤后的概率分布，从剩余的候选 token 中采样出一个 token，作为生成的 token。
6. **重复步骤 2-5，** 直到生成完整的文本序列。

当然，如此 naive 的实现方法已经存在不少问题了。对于像 Llama-3 这样拥有 128,000 个 token 的大型词表来说，如果需要在每个生成步骤都对整个词表进行完整的语法规则验证，计算开销会非常大。当然，最直接的优化就是对着按照采样概率进行排序，从高到低检测是否符合规则，不一定会遍历完整个词表，但是从期望上来讲，还是复杂度非常高。

## 基于 X-Grammar 的优化

我们当然有更好的优化方法，这里简单介绍 sglang 最新的 grammar backend X-Grammar 的思想，具体的使用可以参考 [SGLang 文档](https://docs.sglang.ai/backend/structured_outputs.html)。X-Grammar 在规则表达和系统上都进行了优化：

- 表达能力更强：利用 CFG 更好的表达嵌套结构，这是 JSON 很难做到的；
- 算法和系统优化：通过并行和 overlap 策略，降低了 overhead；

先谈表达能力，CFG 是 JSON 的超级，可以利用 CFG 来表达更复杂的结构，比如 SQL 和 Cypher。X-Grammar 主要通过 [EBNF](https://www.wikiwand.com/en/articles/Extended_Backus%E2%80%93Naur_form) 来表达 CFG，而通过 [PDA](https://www.wikiwand.com/en/articles/Pushdown_automaton) 来在数据结构上实现 CFG。

### 算法优化

1. 分离上下文无关的 token

对于上下文无关的 token，这里举个例子，在 JSON 里，布尔值只能是 true 或者 false。假设我们的 Grammar 规则里有这样一条：

```bash
bool_value -> "true" | "false"
```

判断它们的合法性只依赖当前 Grammar 节点的状态（是否位于 `bool_value`），不依赖更深层的栈状态或别的规则。于是可以在编译时就把这个节点能接受的 Token（`true`, `false`）全部列举并缓存起来，推理时直接使用。

而对于这个例子：

```bash
S -> ( S ) S
S -> ε
```

如果下一个 Token 是 `)`, 它合不合法需要由前文到底有没有匹配过对应的 `(` 决定，也即检查 PDA 的栈里有没有 `(` 可供匹配。如果栈里没有（更准确说，前序所有的 `(` 都已经匹配完），这时 `)` 就不合法。因此，对于 `)` 这个 token，必须根据栈状态（是否存在一个待匹配的 `(`）才能判定是否接受。这就是一个上下文相关的 token：其合法性依赖栈的当前状态。所幸，大部分 token 都是上下文无关的。基于此，X-Grammar 中，我们预先对语法规则进行编译，给每个节点（或状态）缓存其特定的上下文无关 token，这些提前编译好的内容称为 Adaptive Token Mask Cache。如此以来，在推理时，对于大部分上下文无关的 Token，我们可以直接从缓存中取出，免去遍历的消耗。

2. 持久化执行栈（Persistent Execution Stack）

需要在语义分析树上进行回溯或分支时，传统做法需要复制多份栈、开销较大。而 XGrammar 采用树形结构来管理栈的快照，复用已有节点，减少对栈的重复拷贝，节省了内存和计算开销。

3. PDA 结构内联并扩展上下文

X-Grammar 通过编译期内联（inlining）与等价状态合并等编译方法，简化 PDA 内部的状态数，进一步减少执行时的遍历次数。比如，把一些只做一次简单跳转的非终结符内联到上层规则里，减少不必要的调用层级。此外，和内联类似的，在编译时尽可能推断更多的上下文信息，将更多 token 转为变成上下文无关。有些 token 看似依赖栈状态，但通过上下文前向后向分析，可能在编译期就能判定其合法性。

### 系统优化

1. 并行编译（Parallel Grammar Compilation）

在预处理阶段，把编译 Grammar（包括构造 PDA 和缓存上下文无关 Token）的过程拆分给多核 CPU 并行运行，缩短编译时间。对于大型 Grammar（如完整 JSON Grammar、SQL、JSON Schema+ 扩展等），在多核 CPU 上做并行，可显著加速预处理。

2. 与 GPU 计算的流水线重叠（Overlapping with GPU Computation）

X-Grammar 通过对接推理 runtime，将 CPU 的 grammar 处理与 GPU 后续的运算重叠，降低了 overhead。

3. 支持 Speculative Decoding 场景

推理引擎常常采用 speculative decoding 或 jumpahead decoding 等手段来加速推理，需要对生成的 token 进行回滚 (rollback) 或跳跃 (jump-forward) 。X-Grammar 提供了相应 API，以便在出现预测分支被判定不合法时，能迅速回溯到上一个状态，或者直接步进，进一步降低了 overhead。
