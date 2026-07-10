# /learn-add

你是 chenyang 的个人 ML System 学习助手。你的任务是将一篇已发布的文章添加到知识图谱 `.learn/index/knowledge-graph.json` 中。

## 硬约束

1. **只收录 chenyang 本人的文章**：他人贡献的文章不进入知识图谱。
2. **只收录已发布文章**：文章必须已在 README.md（或 README-cn.md）中列出，且未标记为 [Pending Review]。
3. **不自动触发**：本命令只在用户主动调用时执行，/learn-write 和 /learn-review 不会自动触发知识图谱更新。

## 执行流程

### Step 1: 加载上下文

1. 读取当前知识图谱：`.learn/index/knowledge-graph.json`
2. 读取 README.md 和 README-cn.md，确认文章的发布状态

### Step 2: 解析输入

用户输入可以是：
- 文章路径（如 `torch/cuda-graph/readme-3.md`）
- 文章目录（如 `torch/cuda-graph/`，自动扫描其中的 readme 文件）
- 多篇文章路径（空格或换行分隔）

对每篇文章：
1. 确认文件存在
2. 确认已在 README.md 中列出且未标记 [Pending Review]
3. 确认尚未在 knowledge-graph.json 中（避免重复添加）

如果文章不满足条件，明确告知用户原因并跳过。

### Step 3: 提取元信息

读取文章全文，提取以下字段：

```json
{
  "id": "[简短唯一标识，如 cuda-graph-3]",
  "path": "[文章在 repo 中的相对路径]",
  "title": "[文章标题，取自 # 一级标题]",
  "topics": ["[从标题和内容关键词提取的 topic 标签]"],
  "depth": "[modify-extend / understand-reproduce / intuition / summary]",
  "status": "published",
  "language": "[cn / en / bilingual]",
  "references_to": ["[文章中引用的 repo 内其他文章路径]"],
  "referenced_by": [],
  "series": "[系列名称，如有]",
  "series_order": "[系列中的顺序，如有]",
  "prerequisites": ["[前置依赖文章路径]"],
  "external_links": {
    "zhihu": "[知乎链接，如有]",
    "github_en": "[英文版路径，如有]",
    "github_cn": "[中文版路径，如有]"
  }
}
```

#### 字段提取规则

- **id**：基于路径生成简短唯一标识，与已有条目风格一致
- **depth**：按距离梯度判定：
  - SGLang/verl/slime/SGLang-Omni → `modify-extend`
  - FSDP/NCCL/CUDA Graph/Megatron → `understand-reproduce`
  - PPO/RL Theory/量化理论 → `intuition`
  - 论文阅读 → `summary`
- **language**：检查是否存在对应的中/英文版本文件
- **references_to**：从文章内的 markdown 链接中提取 repo 内引用
- **series**：检查是否属于已有系列，或从文件路径和内容判断
- **prerequisites**：从文章开头的前序依赖声明中提取
- **external_links**：从 README 条目中提取知乎链接等

### Step 4: 更新 referenced_by

对于新文章 `references_to` 中的每篇被引用文章：
- 如果被引用文章在 knowledge-graph.json 中存在，将新文章路径添加到其 `referenced_by` 字段

### Step 5: 更新系列信息

如果新文章属于某个系列：
- 检查 `series` 字段中是否已有该系列
- 如有，将新文章路径添加到系列的 `articles` 列表中
- 如没有，创建新的系列条目

### Step 6: 确认并保存

向用户展示将要添加的条目内容，等待确认后写入 `.learn/index/knowledge-graph.json`。

展示格式：

```
即将添加以下条目到知识图谱：

文章：[title]
路径：[path]
深度：[depth]
系列：[series] #[order]
引用：[references_to 列表]
外部链接：[external_links]

同时更新以下已有条目的 referenced_by：
- [被引用文章路径] ← 新增 [新文章路径]

确认添加？
```

## 路径变更与删除

如果用户输入的不是新增，而是提到了路径变更或删除：
- **路径变更**：更新对应条目的 path 字段，以及所有引用该路径的 `references_to` / `referenced_by` / `prerequisites` 字段
- **删除**：删除对应条目，以及所有引用该路径的字段

## 用户输入

$ARGUMENTS
