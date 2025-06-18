# OnlineDPO in TRL

## **OnlineDPO 的核心组件**

1. **Policy Model（策略模型）**：被训练的模型
2. **Reference Model（参考模型）**：固定的基准模型，通常是 Policy Model 的冻结副本
3. 评估组件
    
    （二选一）：
    
    - **Reward Model（奖励模型）**：评分模型，为每个生成结果打分
    - **Judge（判断器）**：比较器，比较两个生成结果并选出更好的一个

## **计算核心公式**

OnlineDPO的核心是最大化被选中回复相对于被拒绝回复的概率比。其损失函数为：

**Sigmoid损失**: 

$\mathcal{L}{\text{DPO}}(\theta) = -\mathbb{E}{(x,y_w,y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \cdot \left( \log \frac{p\theta(y_w|x)}{p{\text{ref}}(y_w|x)} - \log \frac{p\theta(y_l|x)}{p{\text{ref}}(y_l|x)} \right) \right) \right]$

**IPO损失**: 

$\mathcal{L}{\text{IPO}}(\theta) = \mathbb{E}{(x,y_w,y_l) \sim \mathcal{D}} \left[ \left( \log \frac{p\theta(y_w|x)/p{\text{ref}}(y_w|x)}{p\theta(y_l|x)/p{\text{ref}}(y_l|x)} - \frac{1}{2\beta} \right)^2 \right]$

其中：

- $p_\theta(y|x)$ 是策略模型对给定提示$x$生成回复$y$的概率
- $p_{\text{ref}}(y|x)$ 是参考模型的相应概率
- $\beta$ 是控制KL约束强度的参数
- $y_w$ 和 $y_l$ 分别是被选中和被拒绝的回复

## **训练步骤**

### 1. 生成阶段

```
# 为每个提示生成两个不同的回复
prompts = inputs["prompt"]  # 形状: [batch_size]
batch_size = len(prompts)

# 使用vLLM或标准生成
if use_vllm:
    prompt_ids, prompt_mask, completion_ids, completion_mask = _generate_vllm(model, prompts)
else:
    prompt_ids, prompt_mask, completion_ids, completion_mask = _generate(model, prompts)
```

这一阶段：

- 从输入批次中提取提示
- 为每个提示生成两个不同的回复（采样两次）
- 检查哪些回复包含了结束标记（EOS token）

### 2. 计算模型概率

```
# 计算策略模型的对数概率
logprobs = _forward(model, prompt_ids, prompt_mask, completion_ids, completion_mask)

# 计算参考模型的对数概率(无梯度)
with torch.no_grad():
    if ref_model is not None:
        ref_logprobs = _forward(ref_model, prompt_ids, prompt_mask, completion_ids, completion_mask)
    else:  # PEFT情况，只需禁用adapter
        with model.disable_adapter():
            ref_logprobs = _forward(model, prompt_ids, prompt_mask, completion_ids, completion_mask)
```

这一阶段：

- 计算策略模型对生成回复的对数概率
- 计算参考模型对同样回复的对数概率
- 如果使用PEFT，可以通过禁用adapter来获得参考模型的概率

### 3. 评估生成结果

```
# 解码生成的回复
completions = processing_class.batch_decode(completion_ids, skip_special_tokens=True)

if judge is not None:
    # 使用判断器进行对比评估
    ranks = judge.judge(prompts, list(zip(completions[:batch_size], completions[batch_size:])))
    mask = torch.tensor([rank == 0 for rank in ranks], device=device)
else:
    # 使用奖励模型进行评分
    scores = reward_model(prompt_completion_ids).scores

    # 处理未包含EOS的回复（可选降低它们的分数）
    if missing_eos_penalty is not None:
        scores[~contain_eos_token] -= missing_eos_penalty

    # 分割分数并比较
    first_half, second_half = scores.split(batch_size)
    mask = first_half >= second_half
```

这一阶段：

- 将生成的token ID解码回文本
- 使用判断器或奖励模型评估生成结果的质量
- 对每对回复确定哪一个更好（被选中的vs被拒绝的）

### 4. 组织数据并计算损失

```
# 获取被选中和被拒绝回复的索引
batch_range = torch.arange(batch_size, device=device)
chosen_indices = batch_range + (~mask * batch_size)
rejected_indices = batch_range + (mask * batch_size)

# 获取被选中和被拒绝回复的对数概率
chosen_logprobs_sum, rejected_logprobs_sum = torch.split(cr_logprobs_sum, batch_size)
chosen_ref_logprobs_sum, rejected_ref_logprobs_sum = torch.split(cr_ref_logprobs_sum, batch_size)

# 计算对数概率比值
pi_logratios = chosen_logprobs_sum - rejected_logprobs_sum
ref_logratios = chosen_ref_logprobs_sum - rejected_ref_logprobs_sum

# 计算DPO损失所需的logits
logits = pi_logratios - ref_logratios

# 根据指定的损失类型计算损失
if loss_type == "sigmoid":
    losses = -F.logsigmoid(beta * logits)
elif loss_type == "ipo":
    losses = (logits - 1 / (2 * beta)) ** 2

loss = losses.mean()
```

这一阶段：

- 识别每对回复中哪个被选中、哪个被拒绝
- 计算策略模型和参考模型之间的对数概率比
- 应用DPO损失函数（sigmoid或IPO）

### 5. 更新模型

```
# 执行反向传播
if n_gpu > 1:
    loss = loss.mean()  # 多GPU上平均损失

accelerator.backward(loss, **kwargs)

# 返回损失
return loss.detach() / args.gradient_accumulation_steps
```

这一阶段：

- 执行反向传播计算梯度
- 由优化器更新模型参数

##
