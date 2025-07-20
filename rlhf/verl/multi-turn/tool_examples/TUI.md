# Rollout Viewer TUI

`rollout_viewer.py` 是一个基于 [Textual](https://textual.textualize.io/) 的交互式 JSONL 数据浏览工具。
它可以将Rollout log文件，在终端中以富文本/表格形式展示，并支持分页、搜索、字段过滤等操作。

[Github PR]([[tool\] chore: introduce RolloutViewer TUI tools by Yangruipis · Pull Request #2469 · volcengine/verl](https://github.com/volcengine/verl/pull/2469)) Author:[Yangruipis]([Yangruipis (杨睿)](https://github.com/Yangruipis))

## 依赖环境
```python
pip install typer==0.16.0
pip install ujson==5.10.0
pip install textual==0.52.1
```
## 配置
需要在配置中设置 rollout data 存储目录。
```
trainer.rollout_data_dir=$HOME/data/gsm8k/rollout_data
```

## 运行方式
```bash
python rollout_viewer.py  <JSONL目录路径>
```
示例：
```bash
python rollout_viewer.py  ./data/rollouts
```
程序会异步加载该目录下所有后缀为 `.jsonl` 的文件。

## 数据格式要求
- 程序默认每个 `*.jsonl` 文件代表一个 step，文件名需要可以被 `int` 转换（例如 `0.jsonl`, `1.jsonl` …）。
- 文件内容为标准 JSON Lines，每一行对应一个 sample。
- 加载时会为每个 sample 自动添加字段 `__IDX`，表示其在当前文件中的行号。

数据example可以[参考这里]([verl/docs/sglang_multiturn/sandbox_fusion.rst at 152c599303dd4364aa8d581d405a84922dc8c713 · volcengine/verl](https://github.com/volcengine/verl/blob/152c599303dd4364aa8d581d405a84922dc8c713/docs/sglang_multiturn/sandbox_fusion.rst#e2e-tests))

## Example

![image-20250716225610015](rlhf/verl/multi-turn/imgs/TUI example.png)
