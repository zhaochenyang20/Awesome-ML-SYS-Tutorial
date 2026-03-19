# /learn Skill 定义

本文件是 /learn agent 的 prompt 定义文件的文档版本。实际的 skill 定义文件位于 `.claude/commands/` 目录下。

## 子命令

| 命令 | 定义文件 | 功能 |
|------|---------|------|
| `/learn-plan` | `.claude/commands/learn-plan.md` | 生成学习大纲 |
| `/learn-write` | `.claude/commands/learn-write.md` | 完成文章写作 |
| `/learn-review` | `.claude/commands/learn-review.md` | 审查与翻译 |

## 公共上下文

所有子命令共享以下上下文文件：

- **知识图谱**：`.learn/index/knowledge-graph.json` — repo 中所有文章的元信息和引用关系
- **风格指南**：`.learn/index/style-guide.md` — 从 proposal 分析中提炼的机器可读风格指南
- **Agent 配置**：`.learn/config.md` — 深度偏好、引用风格等可调参数
- **文章模板**：`.learn/templates/` — 四种文章类型的模板

## 硬约束（所有子命令共享）

1. **中文优先**：所有写作、计划、审查均以中文进行
2. **排除 [Pending Review] 文章**：绝对不能作为风格参考或知识来源
3. **源码引用必须带 commit hash**：禁止引用 main 分支行号
4. **信息获取不设限制**：最高信息获取权限
5. **概念 → 模型 → 代码顺序硬约束**：所有内容必须遵循此顺序，绝对不能反过来

## 核心改进（基于 CUDA Graph 系列文章实际迭代）

以下改进已反映到三个子命令和 style-guide.md 中：

| 改进项 | /learn-plan | /learn-write | /learn-review |
|---|---|---|---|
| 递进推导 | 每步标注"从 X 推导" | 禁止 checklist 罗列 | 增加推导链检查维度 |
| 概念深度 | 标注哪些概念需要独立展开 | 每个机制展开为子步骤+类比+总结 | 检查概念章节是否足够深 |
| 驱动问题 | 显式标注驱动问题及其位置 | 确保驱动问题在正确位置出现 | 检查是否有清晰的驱动问题 |
| 概念→模型→代码 | 步骤顺序硬约束 | 章节顺序硬约束 | 检查顺序是否正确 |
| 约束映射融入行文 | 不设独立映射步骤 | 不生成独立映射表 | 检查映射是否自然融入 |
| 开篇风格 | — | 遵循回顾+数据+路线图+致谢 | 检查开篇是否模板化 |
| 个人化引用 | — | 允许引用真实人物 | 检查是否有个人化引用 |
| 格式约束 | — | 禁止 ASCII 艺术字 | 检查是否有 ASCII 图 |
| 广泛视野 | 主动搜索同 topic 的相关工作 | — | — |
| 推导式过渡 | — | 每节开篇具体引用前节结论 | 检查过渡句是否具体 |
| 模型全貌先行 | 模型步骤区分"全貌"和"计算特征" | 先交代模型定位/来源/用途 | 检查模型介绍是否缺少全貌 |
| 设计演进路径 | 标注 baseline → 中间方案 → 最终方案 | 展示自然演进 | 检查是否有替代方案讨论 |
| 替代方案对比表 | 标注需要对比的设计决策 | 为重要决策生成对比表 | 检查是否缺少对比 |
| 区分改变/不改变 | — | 统一设计时显式说明 | 检查是否有可能的误解 |
| 工程章节分组 | 同源挑战归为一步 | 同一决策的挑战合并为一个 `##` | 检查章节粒度是否过细 |
| "为什么不用X" | — | 先分析 X 解决什么 → 场景为什么不需要 → 结论 | 检查分析是否完整 |

## 工具权限

| 子命令 | 可用工具 |
|--------|---------|
| `/learn-plan` | Glob, Grep, Read, WebFetch, WebSearch |
| `/learn-write` | 全部工具（需要读源码、读外部代码、写文件） |
| `/learn-review` | Glob, Grep, Read（检查阶段）；Write（翻译阶段，需用户授权） |

## 交互方式

### 示例调用

```
/learn-plan 我想学习 FlashAttention 的实现

/learn-plan 我最近在完成 SGLang Omni 框架支持 Fish Audio S2 模型的过程中，发现同时为 transformers 和 codex loop 两个部分同时开启 CUDA Graph 后，性能有了显著的提升。先前我对 CUDA Graph 的理解程度不深，我现在希望能够基于这个 PR https://github.com/sgl-project/sglang-omni/pull/153 和我已有的知识体系，去进一步学习 CUDA Graph 的原理和实际实现。

/learn-write 参考学习计划 transformers/omni/learn-plan.md 帮我完成 transformers/omni/readme.md

/learn-review 根据 omni 路径下的 readme.md 和 plan.md，检查中文文章的完成程度。
```

### 追问支持

所有子命令都支持追问。例如：
- `/learn-plan` 输出大纲后，可以继续 "把第三步展开"
- `/learn-write` 输出文章后，可以继续 "把源码分析部分重写，深度不够"
- `/learn-review` 输出报告后，可以继续 "帮我翻译为英文"

## 目录结构

```
.learn/
├── readme.md                    # proposal 文档
├── skill.md                     # 本文件：skill 定义文档
├── config.md                    # agent 配置
├── index/
│   ├── knowledge-graph.json     # repo 知识图谱
│   └── style-guide.md           # 风格指南
└── templates/
    ├── code-walkthrough.md      # code walk through 类文章模板
    ├── sys-design.md            # 系统设计分析类文章模板
    ├── paper-reading.md         # 论文阅读笔记模板
    └── tutorial.md              # 教程类文章模板
```
