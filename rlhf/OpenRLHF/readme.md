# OpenRLHF 学习与开发笔记

将 SGLang 接入到 OpenRLHF 中的事情已经做了好几周了，然而非常惭愧，我直到昨天才算是人生第一次用 OpenRLHF 跑了一个完整的 RLHF 流程。仅仅跑起来就遇到了不少坑，这里先记录下，此后不断更新一些 RLHF 的计算特性。**Github 的 markdown 公式渲染实在太烂了，建议 clone 下来本地看。**

<!-- 【PS：此文是我的学习 + debug 笔记，我的学习笔记一贯非常清晰，但是 debug 的时候常常情绪激动，在我的行文中也有所体现...我并不打算把这样的情绪删去，我不想为了冷冰的科研结果，而失去我鲜活的真性情。】 -->

## Quick Start

OpenRLHF 的文档默认用户都比较理解 RLHF 的流程，所以很多地方写的不算入门，对我这种不甚理解 RLHF 的人就比较痛苦。

1. 配环境

我一开始误判了 OpenRLHF 的依赖复杂度，选择了用 docker，其实也没差。这是我自己用的 docker 指令：

```bash
docker run --runtime=nvidia -it --shm-size="40g" --cap-add=SYS_ADMIN -v ~/openrlhf nvcr.io/nvidia/pytorch:24.07-py3 bash
```

主要是我把原本指令里面的 `--rm` 去掉了，不理解为什么文档加了这个参数，让 docker 容器在退出后自动删除。

进入 docker 后，先卸载环境里面的一些库，避免和 OpenRLHF 的依赖冲突。

```bash
pip uninstall xgboost transformer_engine flash_attn -y
```

然后，安装有 vllm 依赖的 OpenRLHF。

```bash
 pip install openrlhf[vllm]
```

PS: 这个发行版可能偶尔会被取消，也可以直接安装最新发行的 openrlhf 和 vllm 0.6.4.post1，前者版本无所谓，后者版本目前得固定在这里。


用 docker 的话，接着可以把 docker commit 保存下来，`docker ps -a` 查找 <container_id>，然后 `docker commit <container_id> openrlhf`，下次直接 `docker run -it openrlhf` 就可以直接进入 docker 了。

然后，配置 `wandb`，老实说我都有快两年没碰过这玩意儿了，越发觉得有点鸡肋。由于 OpenRLHF 可以基于 ray 使用，而 ray 有一套自己 prometheus 的监控，所以配置 `wandb` 的必要性不大。要配置也不麻烦，`wandb init` 一通操作就好了。

2. A Quick Check Out

由于我主要是使用单机多卡做 SGLang 和 vllm 的对拍，所以不使用多机模式。这里简单给两个指令：

```bash
ray start --head --node-ip-address 127.0.0.1 --num-gpus 3 --port 4567 --temp-dir="/opt/dlami/nvme/chenyang/.cache/ray"
```

这是在单机的 3 张卡上启动 ray 的 head 节点，可能会遇到各种启动失败的情况，诸如端口被占用或者卡没分配够，就不断的 `ray stop` 和 `ray start` 直到成功为止。

<details>
<summary> ray start 的输出 </summary>

```bash
ray start --head --node-ip-address 127.0.0.1 --num-gpus 3 --port 4567

Usage stats collection is enabled. To disable this, add `--disable-usage-stats` to the command that starts the cluster, or run the following command: `ray disable-usage-stats` before starting the cluster. See https://docs.ray.io/en/master/cluster/usage-stats.html for more details.

Local node IP: 172.31.54.252

--------------------
Ray runtime started.
--------------------

Next steps
  To add another node to this Ray cluster, run
    ray start --address='172.31.54.252:4567'

  To connect to this Ray cluster:
    import ray
    ray.init(_node_ip_address='172.31.59.18')

  To submit a Ray job using the Ray Jobs CLI:
    RAY_ADDRESS='http://127.0.0.1:8265' ray job submit --working-dir . -- python my_script.py

  See https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html
  for more information on submitting Ray jobs to the Ray cluster.

  To terminate the Ray runtime, run
    ray stop

  To view the status of the cluster, use
    ray status

  To monitor and debug Ray, view the dashboard at
    127.0.0.1:8265

  If connection to the dashboard fails, check your firewall settings and network configuration.
```

</details>

这里给出了 ray 的 start address，也即  `ray start --address='172.31.59.18:4567'`，注意之后要在 OpenRLHF 的指令中使用这个地址。而后也给出了 ray dashboard 的地址，也即 `127.0.0.1:8265`，登上去可以查看到非常精细的监控信息。

接着，submit 一个 test job，这是我在 3 张 H100 上跑通了的脚本，可以参考。

<details>
<summary> Test Job </summary>

```bash
# 根据需求，调整 url ray start address, working_dir 和 save_path

ray job submit --address="172.31.59.18:4567" \
   --runtime-env-json='{"working_dir": "/opt/dlami/nvme/chenyang/rlhf-ckpt"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
   --save_path /opt/dlami/nvme/chenyang/rlhf-ckpt/examples/checkpoint/llama3-8b-rlhf \
   --save_steps 100 \
   --micro_train_batch_size 16 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 32 \
   --rollout_batch_size 128 \
   --max_samples 512 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --packing_samples \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing
```

</details>

任何一套框架都得在易用性和性能之间 trade off，我如上的指令几乎可以最快速地完成 OpenRLHF 的流程测试。注意这么几个参数：

1. `colocate_critic_reward` 和 `colocate_actor_ref`：将 critic/reward 和 actor/ref 放在同一个卡上，显著节省了显存，但是中间有一些 empty cache，会拖慢训练速度。如果不开启，就会各自占据一张卡，显存占用翻倍。
2. `adam_offload`：将 adam 的优化器 offload 到 CPU 上，显著节省了显存，但是会拖慢训练速度。不开启会在 80G H100 上 OOM。
3. `max_samples` 是从 `prompt_data` 里面进行采样的最大样本数，其必须大于 `rollout_batch_size`，否则不够一轮 rollout，会报错。

最后，补充下如何将 openrlhf 进程停下，其实非常暴力：

```bash
pkill -9 -f train_ppo_ray
```
## RLHF 的计算流

在[图解大模型RLHF系列之：人人都能看懂的PPO原理与源码解读](https://zhuanlan.zhihu.com/p/677607581)中已经解释的很清楚了，这里摘录并且总结下，分别会如何计算 critic 和 actor 的 loss。

给定一个 transformer 和任何一个 string，我都可以将整个 string 输入给 reward model 做一次 forward pass，得到每个位置的 token 的 logit。我们取出最后一个 token 的 logit，经过 logit processor 处理，再过一次 softmax 并取 log，得到此处的  log prob。此外，我们也可以对最后一个 token 的 logit 进行其他操作，譬如 pooling 和 projection 等等，拿到 embedding、reward 或者 value。由此可见，对于 string 里的每个 token，我们都可以得到前述所有计算值，但是在 RLHF 中，我们会用到 response 中每个 token 的 log prob 和 value，但是 reward 只会用最后一个 token 的 reward。

### 构造 Reward

这里直接给出 reward 的实际计算，对于第 t 个 response token。当 t 为最后一个 token T 时，才将 reward model 输出的对整个 response 的 reward 加到 $R_t$ 上。换言之，实际上一个 prompt + response 只会让 reward model 推理一次，作为整个 response 的 reward。

$$
R_t = 
\begin{cases} 
-kl\_ctl \cdot \left( \log \frac{P(A_t|S_t)}{P_{ref}(A_t|S_t)} \right), & t \neq T \\
-kl\_ctl \cdot \left( \log \frac{P(A_t|S_t)}{P_{ref}(A_t|S_t)} \right) + R_t, & t = T
\end{cases}
$$


至于其他部分，$kl \_ ctl$ 是个常数，$ \log \frac{P(A_t|S_t)}{P_{ref}(A_t|S_t)} $ 是 reference model 和 actor model 生成 $A_t$ 这个 token 的条件概率比值取对数，也即直接将 actor 的 log prob 和 reference 的 log prob 相减，体现到代码里就是 `kl_ctl * (actor_log_probs - ref_log_probs)`（KL 散度），这样就得到了每个 token 的 reward。注意这里的单复数，`actor_log_probs` 和 `ref_log_probs` 都是所有 response token 的 log prob 构成的 list。

得到 KL 散度后，再在这个 `prompt + response` 的最后一个 token 上加上此处的 reward（称为 reward score），整个 response 每一处的 reward 便构造完成了。当然，实际上的计算还需要考虑 reward score 的 clip 问题，也即不能让 reward 过大。在[知乎](https://zhuanlan.zhihu.com/p/677607581)里面给了非常好的伪代码，这里贴一份代码和解析：

<details>
<summary> 伪代码解析 </summary>

```python
def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score,
                        action_mask):
        """
        reward_function：计算最终的reward分数
        复习一下几个相关参数的默认值：
        self.kl_ctl = 0.1
        self.clip_reward_value = 5
        
        对于batch中的某个prompt来说，它最终的reward分数为：
        (1) 先计算actor和ref_model的logit相似度： -self.kl_ctl * (log_probs - ref_log_probs)
            其实写成self.kl_ctl * (ref_log_probs - log_probs)更好理解些
            这个值越大，说明ref_model对actor生成的结果的认可度越高（即表明rlhf没有训歪），
            没有训歪的情况下我们也应该给模型一些奖励，这个奖励就是self.kl_ctl * (ref_log_probs - log_probs)
            
        （2）由于我们只取最后一个token对应位置的分数作为reward_score，因此我们只需要：
            self.kl_ctl * (ref_log_probs - log_probs)的最后一位 + reward_score
         
         (3) 同时我们对reward_score也做了大小限制，最大不超过self.clip_reward_value（超过统一给成self.clip_reward_value），
             最小不低于-self.clip_reward_value（低于统一给成-self.clip_reward_value）
        
         (4) 最后返回的rewards大小为：（batch_size, 各条数据的长度），对batch中的每条数据来说：
             - response的最后一位：self.kl_ctl * (ref_log_probs - log_probs)的最后一位 + reward_score
             - response的其余位置：self.kl_ctl * (ref_log_probs - log_probs)
        
        """

        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        rewards = kl_divergence_estimate
        # ---------------------------------------------------------------------------------------------------
        # response开始的位置
        # （因为我们对prompt做过padding处理，因此batch中每个prompt长度一致，也就意味着每个response开始的位置一致）
        # （所以这里start是不加s的，只是一个int）
        # ---------------------------------------------------------------------------------------------------
        start = prompts.shape[1] - 1
        # ---------------------------------------------------------------------------------------------------
        # response结束的位置
        # （因为一个batch中，每个response的长度不一样，所以response的结束位置也不一样）
        # （所以这里end是加s的，ends的尺寸是(batch_size,)
        # ---------------------------------------------------------------------------------------------------
        ends = start + action_mask[:, start:].sum(1) + 1
        # ---------------------------------------------------------------------------------------------------
        # 对rewards_score做限制
        # ---------------------------------------------------------------------------------------------------
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value,
                                  self.clip_reward_value)
        batch_size = log_probs.shape[0]
        for j in range(batch_size):
            rewards[j, start:ends[j]][-1] += reward_clip[j] # 

        return rewards
```
注意输入输出的维度，`prompts` 是一个 `[batch size, padded prompt length]` 的 matrix，`ref_log_probs` 和 `log_probs` 是 `[batch size, padded prompt with response length]` 大小的矩阵，然后只有从 `prompt` 结束到 `response` 结束这一块儿的 `reward` 才会实际有作用，`prompt` 的 `reward` 是不计算的。

`prompt` 有统一的 `padding`，所以 `response` 的 `start` 位置是唯一的，而 `ends` 则通过 `action_mask` 中的 1 元素的截止为止计算得到。最后，在这个 `batch` 中，每个 `prompt` 的 `reward` 的结尾那个 `token` 加上 `reward_score` 进过 clip 得到的 `reward`。

</details>

### 构造 Advantage

Advanatage 可以某种程度理解为“意外之喜”，具体的描述参考知乎原文。这里直接给出 Advantage 的构造公式：

$$
Adv_t = \left( R_t + \gamma \cdot V_{t+1} - V_t \right) + \gamma \cdot \lambda \cdot Adv_{t+1}
$$

我们来拆解下，考虑到 $R_t$ 是每个 token 的 reward，前面已经构造了。$V_t$ 和 $V_{t+1}$ 是当前 token 和下一个 token 的 value，而每个 token 的 value 在 value model 的 forward pass 中都可以得到，$Adv_t$ 是当前 token 的 advantage，$\gamma, \lambda$ 都是常数。这种递归的构造方式，可以用尾递归来反推每个位置的 advantage。

这里还是贴下伪代码，注意这个函数一并返回了 `returns`，也即每个 token 的实际收益，这个收益之后会用于更新 critic model：

<details>
<summary> 伪代码 </summary>

```python
 def get_advantages_and_returns(self, values, rewards, start):
        """
        Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        
        没有引入GAE前的t时刻的优势值：
        detal_t = r_t + gamma * V_t+1 - V_t
        其中：
            - r_t表示t时刻的即时收益
            - V_t+1表示未来时刻的预期收益
            - r_t + gamma * V_t+1可理解成t时刻的实际预期收益
            - V_t可理解成t时刻的预估预期收益（是模型，例如critic model自己估算出来的）
        
        引入GAE后的t时刻的优势值：
        A_t = delta_t + gamma * lambda * A_t+1
        粗暴理解为在t时刻时，不仅考虑当下优势，还考虑了未来的优势
        为了知道A_t, 我们得知道A_t+1，所以在本算法中采取了从后往前做动态规划求解的方法，也即：
        假设T是最后一个时刻，则有A_T+1 = 0, 所以有: A_T = delta_T
        知道了A_T, 就可以依次往前倒推，把A_t-1, A_t-2之类都算出来了
        
        引入GAE后t时刻的实际预期收益
        returns_t = A_t + V_t
                  = delta_t + gamma * lambda * A_t+1 + V_t
                  = r_t + gamma * V_t+1 - V_t + gamma * lambda * A_t+1 + V_t
                  = r_t + gamma * (V_t+1 + lambda * A_t+1)
        
        注意，这里不管是advantages还是returns，都只算response的部分
        """
        
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        # 注意这里用了reversed，是采取从后往前倒推计算的方式
        for t in reversed(range(start, length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1) # 优势
        returns = advantages + values[:, start:] # 实际收益
        # values: 预期收益
        return advantages.detach(), returns
```

</details>

### 构造 Actor Loss

这里还是直接给出 Actor Loss 的构造公式：

$$
actor \_  loss = - \min \left[ Adv_t \cdot \frac{P(A_t|S_t)}{P_{old}(A_t|S_t)}, Adv_t \cdot clip \left( \frac{P(A_t|S_t)}{P_{old}(A_t|S_t)}, 0.8, 1.2 \right) \right]
$$

这个构造公式看着复杂，实际上一点也不简单。每个 response token 的 $Adv_t$ 的构造已经在前文给出，而 $P(A_t|S_t), P_{\text{old}}(A_t|S_t)$ 其实都是 actor model 的条件概率。之所以有个 old 是因为我们希望多利用每轮产生的 experiences，因此一组 experiences 会更新多轮。old 表示这一组 experiences 用于更新之前的 actor model，用这个 old actor model 对这几轮更新的大小做了约束。**最后，考虑到某一轮更新里，当前 actor model 和 old actor model 的差距实在太大了，以至于条件概率的比值超出了人为预设的范围，此时 $Adv_t$ 的系数（ratio）会取为约束边界。此时 actor model 的参数不再影响 ratio，换言之 actor model 的参数不再在 actor loss 的计算图中了，这个 loss 也就不会更新 actor 的参数了。** 注意，advantage 的构造是由 old actor model 构造来的，计算结束就固定了，对于更新中的 actor model 没有梯度，所以整个 actor loss 的计算图中只有 ratio 对更新中的 actor model 有梯度。

这里还是贴下代码，注意最后整个 response 每一处的 loss 取均值，就是这个 prompt + response 的 actor loss 了。

<details>
<summary> 伪代码 </summary>

```python

    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        """
        logprobs: 实时计算的，response部分的prob（只有这个是随着actor实时更新而改变的）
        old_logprobs：老策略中，response部分的prob （这个是固定的，不随actor实时更新而改变）
        advantages： 老策略中，response部分每个token对应的优势（这个是固定的，不随actor实时更新而改变）
        mask：老策略中，response部分对应的mask情况这个是固定的，不随actor实时更新而改变）
        
        之所以要引入logprobs计算actor_loss，是因为我们不希望策略每次更新的幅度太大，防止模型训歪
        
        self.cliprange: 默认值是0.2
        """
        ## policy gradient loss
        # -------------------------------------------------------------------------------------
        # 计算新旧策略间的KL散度
        # -------------------------------------------------------------------------------------
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        # -------------------------------------------------------------------------------------
        # 计算原始loss和截断loss
        # -------------------------------------------------------------------------------------
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum() # 最后是取每个非mask的response token的平均loss作为最终loss
        return pg_loss

```

</details>

### 构造 Critic Loss

注意到，在 advantage 的构造中，我们一并得到了 `returns`，将其视为每个 token 的实际收益。


$$
Ret_t = \left( R_t + \gamma \cdot V_{t+1} - V_t \right) + \gamma \cdot \lambda \cdot Adv_{t+1} + V_t = R_t + \gamma \cdot \left( V_{t+1} + \lambda \cdot Adv_{t+1} \right)
$$

而预估收益就是 $V_t$，然后我们构造 MSE loss 来最小化预估收益和实际收益的差距。

$$
critic \_  loss = \left( Ret_t - V_t \right)^2
$$

看上去似乎 $Ret_t - V_t$ 就是 $Adv_t$，**但是实际使用的 `values` 是多轮更新中的 value model 的输出，也即 new value，而 `returns` 是多轮更新开始时就固定了的实际收益（old returns），所以 $Ret_t - V_t$ 并不是 $Adv_t$。**

伪代码如下：

<details>
<summary> 伪代码 </summary>

```python
def critic_loss_fn(self, values, old_values, returns, mask):
        """
        values: 实时critic跑出来的预估预期收益（是变动的，随着ppo epoch迭代而改变）
        old_values：老critic跑出来的预估预期收益（是固定值）
        returns：实际预期收益
        mask：response部分的mask
        
        self.cliprange_value = 0.2
        """
        ## value loss
        # 用旧的value去约束新的value
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        if self.compute_fp32_loss:
            values = values.float()
            values_clipped = values_clipped.float()
        
        # critic模型的loss定义为（预估预期收益-实际预期收益）**2
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.sum(
            torch.max(vf_loss1, vf_loss2) * mask) / mask.sum() # 同样，最后也是把critic loss平均到每个token上
        return vf_loss
```

</details>

### 更新流程

1. 准备一个 batch 的 `prompts`；
2. 将这个 batch 的 `prompts` 输入给 Actor，解码得到 `responses`；
3. 将 `prompt + responses` 输入给 Critic/Reward/Reference，分别计算得得到所有 token 的 values、最后一个 token 的 reward 和所有 token 的 log probs，按照强化学习的术语，称这些数据为经验（experiences）了；
4. 根据 experiences 多轮计算 actor loss 和 critic loss 并更新 Actor 和 Critic 模型。

对于第 4 步，我们当然可以一轮 experiences 就更新一次 actor 和 critic，但是为了尽可能利用这个 batch 的 experiences，我们对 actor 和 critic 做多轮更新。我们将 experiences 中多轮更新开始前的 log probs 和 values 称为 old log probs 和 old values（reward 不会多轮计算）。在每一轮中，actor 和 critic 会生成 new log probs 和 new values，然后在 old 的基础上计算 actor loss 和 critic loss，然后更新参数。

<details>
<summary> 伪代码 </summary>

```python
# --------------------------------------------------------------
# 初始化RLHF中的四个模型
# --------------------------------------------------------------
actor, critic, reward, ref = initialize_models()

# --------------------------------------------------------------
# 训练
# --------------------------------------------------------------
# 对于每一个batch的数据
for i in steps: 
    # 先收集经验值
    exps = generate_experience(prompts, actor, critic, reward, ref)
    # 一个batch的经验值将被用于计算ppo_epochs次loss，更新ppo_epochs次模型
    # 这也意味着，当你计算一次新loss时，你用的是更新后的模型
    for j in ppo_epochs:
        actor_loss = cal_actor_loss(exps, actor)
        critic_loss = cal_critic_loss(exps, critic)
        
        actor.backward(actor_loss)
        actor.step()
        
        critc.backward(critic_loss)
        critic.step()
```

</details>

这时候回看这两个图，整体就清晰太多了：

![make-experience](./make-experience.png)

![learning-stage](./learning-stage.png)

## 基于 Ray 的 OpenRLHF 分布式训练

这里主要是参考了这篇知乎：[图解 OpenRLHF 中基于 Ray 的分布式训练流程](https://zhuanlan.zhihu.com/p/12871616401)，原文讲的很清楚，这里做一些进一步阐述，可以结合原文一起阅读，更加清晰。

### Ray 的一些核心概念

原文中提到了一些 Ray 的概念，不过个人觉得稍微模糊了些，所以进一步补充。

1. Placement Group

OpenRLHF 里有一个变量 `pg`，大多数时候指的都是 Placement Group，而不是 torch 通讯里的 process group。Placement Group 可以理解为一组资源分配方案，允许用户精确控制资源的分配和任务的调度。比如这里：

```python 
import ray

# 创建Placement Group
pg = ray.util.placement_group(
    bundles=[{"CPU": 2, "GPU": 1}, {"CPU": 4, "GPU": 2}],
    strategy="PACK"
)

# 使用Placement Group来指定任务的执行位置
@ray.remote(placement_group=pg)
def train_model():
    # 训练模型的代码
    pass
```

2. Driver

Ray 程序的控制节点，通常是程序的起始点。它通常在一个单独的节点上运行，负责启动 Ray 集群、提交任务并调度执行。Driver 端不会执行计算工作，而是通过远程调用将计算任务分配出去。

3. Worker

Worker 是 Ray 集群中的计算节点，负责执行由 Driver 提交的任务。每个 Worker 节点上运行着多个 Worker 进程，这些进程会处理来自 Driver 或其他 Worker 的任务。

4. Task

Ray Task 是最基本的计算单元，通常表示一个需要执行的函数或者操作，是并行执行的最小单位。每个任务都是一个函数调用，它会被分配到 Ray 集群中的一个 Worker 执行。**任务是无状态的，执行完任务后它不会保存任何状态，每次执行都是独立的。**

5. Actor 与 Actor Handle

与 Task 不同，Actor 是 Ray 中有状态的计算单元，在其生命周期内保存内部状态。创建时，Ray 为其分配独立执行实例并返回其引用 Actor Handle。通过 Actor Handle 调用 Actor 方法时，Driver 会通过 Ray 调度系统将这次请求发送给合适的 Worker 节点。

```python
import ray

# 初始化 Ray 集群
ray.init()

# 定义一个简单的 Actor 类
@ray.remote
class Counter:
    def __init__(self, value=0):
        self.value = value

    def increment(self):
        self.value += 1
        return self.value

# 创建一个 Actor 实例，返回的就是一个 Actor Handle
counter_handle = Counter.remote()

# 通过 Actor Handle 调用 increment 方法
result = ray.get(counter_handle.increment.remote())
print(result)  # 输出 1

# 再次调用 increment 方法
result = ray.get(counter_handle.increment.remote())
print(result)  # 输出 2
```

比较麻烦的是，Ray 系统中的 Actor 和 RLHF 中的 Actor 是两个概念，后文也会特殊区分二者。在 OpenRLHF 中，`PPORayActorGroup` 代表 Ray 系统的 Actor 组，而 `ActorModelRayActor` 代表基于 Ray 的 RLHF 中的 Actor。

### colocate 的资源分配策略

OpenRLHF 实现了 Actor/Reference，Value/Reward 的 colocate 策略，也即 Actor 和 Reference 会共享同一片计算资源，直观上我几乎省下了一半的显存，直接通过 `--colocate_actor_ref` 就可以开启。比较有趣的是，开启 colocate 后，实际上资源并不是对半分的，而是：

```python
actor_model = PPORayActorGroup(
    args.actor_num_nodes,
    args.actor_num_gpus_per_node,
    ActorModelRayActor,
    pg=pg,
    num_gpus_per_actor=0.75 if pg else 1,
)

ref_model = PPORayActorGroup(
    args.ref_num_nodes,
    args.ref_num_gpus_per_node,
    ReferenceModelRayActor,
    pg=pg,
    num_gpus_per_actor=0.25 if pg else 1,
)
```

这里是个 trick，大意是说按照目前的启动逻辑，假设要 actor model 要 data parallelism 占据两张卡，设置 `num_gpus_per_actor=0.5`，则 Ray 先在第一张卡上用 0.5 显存启动了第一个 actor model，接下来要分配第二个占据 0.5 显存的 actor model，Ray 会继续将第二个 actor model 分配到第一张卡上，利用省下的 0.5，而不是第二张卡。所以 colocate 的时候，采取了 `num_gpus_per_actor=0.75, 0.25` 的策略。实际上的显卡并不是对半分的，而且对于只使用一张卡的情况，这种策略不会有影响。

## 扩展 OpenRLHF 的推理引擎

众所周知，我的一大工作是在 OpenRLHF 系统中支持 SGLang backend，有两个具体的需求：

1. 支持 SGLang 的 inference，确保 accuracy 和 speed 都能对拍；
2. 将现在的 vllm engine 抽象为一个 inference Engine Backend 类，然后这个 backend 支持 huggingface，SGLang 和 vllm。

根据我一直以来的开发经验，先在这里捋一捋 OpenRLHF 中的所有 vllm 使用，以此来实现统一的 backend。

-----------------

### `openrlhf/cli/batch_inference.py`

这个文件实现了三个功能，用 vllm 和 transformers 做 generation 以及用 transformers 推理得到 reward。这个做法是非常严谨的，因为 inference engine 在 RLHF 中，目前只能拿去做 generation，生成的 log probs，logits，embedding 和 reward 都是不准的：

> 推理引擎的 kernal fusion 和 training engine 差距不小，batch size 不一样时，推理请求 dispatch 到不同的 kernal 上，然后 numerical 误差逐层累计，到了 log probs 这层就到了不可忽视的程度了。这个问题在 bert 时代就有了，training engine 和 inference engine 的精度差异无法规避，而且全心来搞一两个月可能内都没法修复。
> 
> 所以现在推理引擎在 RLHF 中更多是加速 sampling，reward 和 embedding 还得用训练脚本来算，可能得半年后花好几个月研究研究这个问题。

这三个函数还是非常简单，由于我描述过，要做一个统一的 backend，所以这个 file 大致的修改思路是开一个新的 class GenerationBackend，在 GenerationBackend 里面做一个 branch，实现 SGLang, vllm 和 transformers 的 inference。我先抓紧写一个这个 PR 出来。

写到这里，我才发现一个惊人的事情，OpenRLHF 没有单测。我先测测这个系统的可用性，参考这个 `examples/scripts/train_rejection_sampling_llama.sh`，写一个对拍单侧：

<details>
<summary> 对拍单测 </summary>

```bash
# For vllm
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p ./checkpoint/llama-3-8b-rejection
GENERATE_OUTPUT=./checkpoint/llama-3-8b-rejection/generate.jsonl
RM_OUTPUT=./checkpoint/llama-3-8b-rejection/rm.jsonl
ITER_LOG_PATH=./checkpoint/llama-3-8b-rejection/iter.log
MODEL_OUTPUT_PATH=./checkpoint/llama-3-8b-rejection

TRAINING_ITERS=10
ROLLOUT_BATCH_SIZE=10240

POLICY_MODEL_PATH=OpenRLHF/Llama-3-8b-sft-mixture

iter=0

python batch_inference.py \
   --eval_task generate_vllm \
   --pretrain $POLICY_MODEL_PATH \
   --bf16 \
   --max_new_tokens 2048 \
   --prompt_max_len 2048 \
   --dataset OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --temperature 0.9 \
   --zero_stage 0 \
   --best_of_n 4 \
   --enable_prefix_caching \
   --tp_size 2 \
   --max_num_seqs 64 \
   --iter $iter \
   --rollout_batch_size $ROLLOUT_BATCH_SIZE \
   --output_path $GENERATE_OUTPUT \
   --max_samples 200
```

```bash
# For SGLang

mkdir -p ./checkpoint/llama-3-8b-rejection
GENERATE_OUTPUT=./checkpoint/llama-3-8b-rejection/generate.jsonl
RM_OUTPUT=./checkpoint/llama-3-8b-rejection/rm.jsonl
ITER_LOG_PATH=./checkpoint/llama-3-8b-rejection/iter.log
MODEL_OUTPUT_PATH=./checkpoint/llama-3-8b-rejection

TRAINING_ITERS=10
ROLLOUT_BATCH_SIZE=10240

POLICY_MODEL_PATH=OpenRLHF/Llama-3-8b-sft-mixture

iter=0

python batch_inference.py \
   --eval_task generate_sglang \
   --pretrain $POLICY_MODEL_PATH \
   --bf16 \
   --max_new_tokens 2048 \
   --prompt_max_len 2048 \
   --dataset OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --temperature 0.9 \
   --zero_stage 0 \
   --best_of_n 4 \
   --enable_prefix_caching \
   --tp_size 2 \
   --max_num_seqs 64 \
   --iter $iter \
   --rollout_batch_size $ROLLOUT_BATCH_SIZE \
   --output_path $GENERATE_OUTPUT \
   --max_samples 200
```

</details>


写完我才发现，sglang vllm 还有 openrlhf 有着不可调和冲突，sglang 和 vllm 的 torch 依赖不同，而且目前无法修复，我尝试了诸多 vllm 版本都无法解决这个问题。只能在这里开始 diverge 出两个环境。使用两个环境而不是两个 docker 是因为我还没习惯 docker 的映射，不想重设系统变量了。

装环境其实比较痛苦，sglang 的环境对着文档装了半天还有点问题，只能手动装一堆东西，我都惊奇这文档不是我自己写的么，怎么感觉我一边装都不顺利。不过也没用户提，所以先不管了。至于 openrlhf 和 vllm，按着最开始描述的，先安装 openrlhf，再安装 ray，最后安装 vllm。这么搞了半天，发现新的集群 torch 的 nvidia link 一直失败，几年之后，我又一次遇到了[这个问题](https://github.com/pytorch/pytorch/issues/111469)。

想要获得一个稳定的开发环境，非常麻烦...这里列举下这个 bug 和解决过程。

1. 发现 torch 的链接一直有问题，我按照[这个 issue](https://github.com/pytorch/pytorch/issues/111469) 等一系列方法改了 `LD_LIBRARY_PATH`，试过 docker 和 conda 等等各种环境，还是 torch 2.5.1 的链接有问题：

```bash
ImportError: /usr/local/lib/python3.10/dist-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
```

2. 我发现，ray submit 的 job 有这个问题，但是 ipython 里面没问题。进一步缩小范围，我发现 `python -c` 都没问题，那么问题在哪儿呢？

3. 问了 claude，是 ray 环境的问题：

<details>
<summary> claude 的回答 </summary>

> 不是，凭什么他妈的 python -c 可以 import，但是 ray submit 的就不行？

啊，这是一个很典型的环境变量继承问题。

当你在本地终端执行 `python -c` 时，它继承了你在 `.zshrc` 中设置的所有环境变量（包括 `LD_LIBRARY_PATH`）。但是当通过 ray submit 提交任务时，这些环境变量可能没有被正确传递到 ray worker 进程中。

建议尝试以下解决方案：

1. 在 ray 任务中显式设置环境变量：
```python
import os
os.environ["LD_LIBRARY_PATH"] = "/opt/dlami/nvme/chenyang/.miniconda3/envs/rlhf-vllm/lib/python3.11/site-packages/nvidia/cusparse/lib"
```

2. 或者在 ray.init() 时通过 runtime_env 传递：
```python
ray.init(runtime_env={
    "env_vars": {
        "LD_LIBRARY_PATH": "/opt/dlami/nvme/chenyang/.miniconda3/envs/rlhf-vllm/lib/python3.11/site-packages/nvidia/cusparse/lib"
    }
})
```

3. 如果使用 ray submit，可以在提交脚本中添加：
```bash
export RAY_RUNTIME_ENV='{"env_vars": {"LD_LIBRARY_PATH": "/opt/dlami/nvme/chenyang/.miniconda3/envs/rlhf-vllm/lib/python3.11/site-packages/nvidia/cusparse/lib"}}'
ray submit script.py
```

这样应该能确保环境变量被正确传递到 ray worker 中。


</details>

问题解决，我在 `.zshrc` 里面写成这样：

```bash
export LD_LIBRARY_PATH=/opt/dlami/nvme/chenyang/.miniconda3/envs/rlhf-vllm/lib/python3.11/site-packages/nvidia/cusparse/lib
export RAY_RUNTIME_ENV='{"env_vars": {"LD_LIBRARY_PATH": "/opt/dlami/nvme/chenyang/.miniconda3/envs/rlhf-vllm/lib/python3.11/site-packages/nvidia/cusparse/lib"}}'
```


<!-- 很烦，你看我和 claude 对话都快直接骂人了。今天看上去做了很多事情，实际上总觉得在原地踏步：

1. 发现最新的 vllm, sglang 和 openrlhf 的依赖冲突目前没法解决，不单单是 outlines 的问题，更深层是 torch 的问题。我又尝试用 vllm 0.6.3.post1 和 vllm 0.6.4.post1，看看能不能兼容，结果只有 vllm 0.6.5 的 `update_weights_from_distritbued` 在当时的环境里成功了，其他版本都不行，又用了一个小时。 
2. 没法了，尝试 diverge 环境，却发现第一个集群在写入时崩了，天知道是不是我干的，可我从来不写入到 `/home` 啊。
3. 切换集群，修改若干多配置，终于把另一台 H100 设置好了。接着发现了天使的 torch link error。尝试各种办法 de 了 2h，先用 conda 开新环境，再用 docker 试图绕过 torch link，发现无果，非常绝望。
4. 在群里给大家汇报问题，顺带试了试 `python -c`，看看会不会 error。发现没有，终于问了 claude，发现了 ray 的环境变量问题。要是没有现代 LLM，这 bug 回到两年前真的会让我自闭，又想起了疫情期间我在紫二 308 对着商汤的集群配置 deepspeed 的痛苦，兜兜转转又回到了这种境遇。
5. 其实还遇到一些问题，总体上是我没有耐心，比如观察到 openrlhf 的进程卡在了 DeepSpeedEngine compile 上，我就会停掉重开。事后发现，其实第一次就是要等很久。郭磊一会儿，我的 training 又卡住了，这次卡在 vllm broadcast weights 上。说实话，我有点崩溃，因为我知道这个 broadcast 不会花费那么久，之前调成 0.6.5 可以，现在不行了。我又再重装一次环境，因为之前一模一样的问题都是这样解决的。
6. 还是不对，问了 OpenRLHF 作者，说是 vllm 更新又搞崩了，weights update 的 bug 又出现了。这才发现，大家都焦头烂额的，这就是 mlsys 的常态吧...他建议我用 openrlhf 的稳定版本，别用 main。我换到 0.5.4，还是崩了。

不说了，终于搞到了一个稳定的开发环境，明天去 review 朋友给我的 PR，然后在之前给 OpenRLHF 的 PR 上说明目前的情况。**今天把之前删了的 lolm 下载了回来，妈的，lolm 启动。这 lolm 里面有 llm，真是天意。** -->

PS：这里还跑着跑着遇到了服务器爆炸的情况，结果最后发现是 ray 的 logging 会默认到 `tmp/ray` 下面，然后这个 log 还贼大，把 `tmp` 给撑爆了。而且，ssh 也是要往 `tmp` 里面写东西的，所以直接干垮了一台 H100，感谢 NV 在圣诞节加班给换了台新的。

### `openrlhf/cli/train_ppo_ray.py`

这个可能是我觉得 openrlhf 最重要的 training 脚本，也是我之前主要测试的地方。我先记录下我的环境，避免将来发生环境冲突：

- openrlhf[vllm]: main 上的 openrlhf, vllm 0.6.5, torch 2.5.1, outlines 0.1.11, ray 2.12.0
- openrlhf[sglang]: main 上的 openrlhf, sglang 0.4.1, torch 2.5.1+cu121, vllm 0.6.4.post1, outlines 0.0.46, ray 2.12.0

--------

PPO 几乎是最重要的改动，我在两次动笔写这部分文档之间，已经过了接近一周了，从好处想，我成功接入了 sglang 到 openrlhf，但是坏消息是，二者远远没有到达一个稳定替换的地步，效率相当不稳定。先记录下如何替代的吧，这里照着我 PR 的 file changes 来讨论，顺带我也梳理下为什么，猜测下效率不稳定的原因。

--------

回到 `train_ppo_ray.py` 本身，其实这个改动是很小的，这个文件就是把所有叫做 `vllm_engines` 的变量改为 `inference_engines` 的通用名字，然后加上 `--backend` 参数。

### `openrlhf/trainer/ppo_utils/experience_maker.py`

本质上 RLHF 里面，inference engine 就是用于 make experience 的，所以这个改动蛮大的。

1. `llm.generate`，首先，原先的：

```python
llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
```

我改为了：

```python
llm.generate.remote(
    sampling_params=sampling_params, prompt_token_ids=prompt_token_ids, all_prompts=all_prompts
)
```

这个其实是 vllm 和 sglang 的差异，在 openrlhf 的源代码里面，给 vllm engine 直接传入了 `prompt_token_ids`，这个大概是 `input_ids`，而我先实现了对 sglang 传入 prompts 做 generate。正如我已经说过的，vllm training engine 和 sglang training engine 都会出现效率不稳定而卡死的情况，我不相信这是我引入的问题。但我确实怀疑有些细微的差异带来了不小的影响，譬如 vllm engine 传入 token ids 和 sglang engine 传入 prompts 会不会有差别，加了些奇怪的 tokens？此外，sglang engine 里面还要再 tokenize 一次，带来了不可忽略的 overhead。所以这里我有三个 TODO，都去实现下：

1. 直接传入 token ids 给 sglang，不要再对 prompts tokenize 一次了。
2. 打印出 tokens 的开始和结尾，用于检查 vllm 和 sglang 处理特殊 token 是否有区别。
3. 打印出传入给 experience making 的 tokens 矩阵大小，难道二者的矩阵大小差异（譬如最长的 string 特别长导致 padding 后差异特别大）会有显著影响么？

2. token collection

这个改动就没什么意思了，我手动 collect 了所有的 `input_ids` 和 `output_ids`，避免了下面几个不够优雅的 for 循环。

<details>
<summary> 基于列表推理的改动 </summary>

原本：

```python

max_input_len, max_output_len = 0, 0
for output in outputs:
    max_input_len = max(max_input_len, len(output.prompt_token_ids))
    max_output_len = max(max_output_len, len(output.outputs[0].token_ids))
```

我的改动：

```python

torch.cuda.synchronize()
start = time.time()

input_token_id_list = None
output_token_id_list = None
if backend == "vllm":
    input_token_id_list = [list(output.prompt_token_ids) for output in outputs]
    output_token_id_list = [list(output.outputs[0].token_ids) for output in outputs]
else:
    input_token_id_list = [list(output["input_ids"]) for output in outputs]
    output_token_id_list = [list(output["output_ids"]) for output in outputs]

torch.cuda.synchronize()
end = time.time()
print(f"Tokenize input and output takes: {end - start}s for {backend}")

max_input_len = max(len(input_id) for input_id in input_token_id_list)
max_output_len = max(len(output_id) for output_id in output_token_id_list)

```

</details>

我确信这种改动是完全等价的，毕竟要是这种等价替换都做不好，本科抄的四年代码，早该毕不了业了...

然而，令我费解的是，虽然如此改动在我看来等价，为什么会出现卡顿呢？我没有测试过 main，是否是 main 本身就有问题。于是，我又有了一个 TODO：

- 测试 main 上是否也会卡顿？

### `openrlhf/trainer/ray/ppo_actor.py`

这个文件除开命名之外，我几乎没有什么改动。有个值得注意的地方是，我把 `init_process_group` 的 backend 统一改为了 `nccl`，因为某一次出现了 process group 创立错误，但是 `nccl` 做 backend 是稳定的。难道这是造成不稳的原因：

- 测试是否是 backend 的问题？

### `openrlhf/trainer/ray/vllm_engine.py`

这是最大的改动，道理很简单，就是在这个文件下面加入 branch，然后根据 backend 选择不同的 engine。我还没有来得及修改文件名，按理说要改成 `inference_engine.py`。不过这些问题之后解决都好...

vllm 的地方不用改，只是移动到 if 下面就行，但是 sglang 的地方得改动不少，主要是 `LLMRayActor` 的 `__init__` 传入的是 `*args, **kwargs`，直接对着 vllm 的 server args 在启动，如果我直接传给 `sglang.Engine`，会因为位置参数匹配不上而报错。

所以，我得找寻 sglang 和 vllm 的对应参数，但是这个事情在 [batch_inference.py](#openrlhfclibatch_inferencepy) 里面已经做过了，我怀疑可能也有错。这里记录下我的做法：

<details>
<summary> 从 vllm 到 sglang 的 server args 映射 </summary>

这是 vllm 的 server parameters：

```python
pretrain,
noset_visible_devices=noset_visible_devices,
trust_remote_code=True,
tensor_parallel_size=tensor_parallel_size,
dtype="bfloat16",
seed=seed + i,
enable_prefix_caching=enable_prefix_caching,
enforce_eager=enforce_eager,
max_model_len=max_model_len,
backend=backend,
```        

其中 pretrain 是 model path，而这是我在 sglang 里的映射：

```python
#! TODO chenyang check engine params
sglang_params = {
    "model_path": args[0],  # pretrain path
    "trust_remote_code": kwargs.get("trust_remote_code", True),
    "dtype": kwargs.get("dtype", "auto"),
    "tp_size": kwargs.get("tensor_parallel_size", 1),
    "device": "cuda",
    "disable_radix_cache": not kwargs.get("enable_prefix_caching", False),
    "random_seed": kwargs.get("seed", 42),
    "disable_cuda_graph": not kwargs.get("enforce_eager", False),
    "disable_cuda_graph_padding": not kwargs.get("enable_prefix_caching", False),
    "context_length": kwargs.get("max_model_len", None),
    "log_level": "info",
    "return_token_ids": True,
}
self.llm = sglang.Engine(**sglang_params)
```

</details>

老实说我还挺有信心的，但是也不得不查啊。注意 `return_token_ids` 是专门为 openrlhf 写的新 feature，这里得感谢 [Shuai Shi](https://github.com/shuaills) 的这个 [PR](https://github.com/sgl-project/sglang/pull/2636)，这也是我第一次 mentor 人写的 SGLang PR，很有成就感，但是我自己其实 PR 都没几个 🤣🤣🤣

说回到这些参数，还有 `log_level = "info"` 是怜悯让我加的，看看 inference engine 是不是 fully ultized 了。目前看了看 `token usage = 0.61`，感觉是还可以的，但是怜悯说可以看看 `cache hit rate`，这个之后看看。这里再来三个 TODO:

1. 检查 vllm 到 sglang 的参数映射是否正确？
2. 同样的，检测 sampling params 是否正确？
3. 查看 cache hit rate，性能上应该还有提升空间。

都提到 2 了，当然我也得对 sampling params 做映射，在我的映像中，sglang 应该是完全贴着 openai api 写的 sampling params，但是还是得检查 parameter 映射。

<details>
<summary> 从 vllm 到 sglang 的 sampling params 映射 </summary>

```python
if self.backend == "vllm":
    outputs = self.llm.generate(
        sampling_params=kwargs["sampling_params"], prompt_token_ids=kwargs["prompt_token_ids"]
    )
elif self.backend == "sglang":
    # Note that sglang sampling params are different from vllm
    sampling_params = kwargs["sampling_params"]
    all_prompts = kwargs["all_prompts"]

    # min_tokens, include_stop_str_in_output is not used in sglang

    sampling_params = dict(
        max_new_tokens=sampling_params.max_tokens,
        top_p=sampling_params.top_p,
        top_k=sampling_params.top_k,
        temperature=sampling_params.temperature,
        repetition_penalty=sampling_params.repetition_penalty,
        skip_special_tokens=sampling_params.skip_special_tokens,
    )
    outputs = self.llm.generate(all_prompts, sampling_params)
```

当然，前端传进来的 sampling params 如下：

```python
sampling_params = SamplingParams(
    temperature=kwargs.get("temperature", 1.0),
    top_p=kwargs.get("top_p", 1.0),
    top_k=kwargs.get("top_k", -1),
    max_tokens=kwargs.get("max_new_tokens", 1024),
    min_tokens=kwargs.get("min_new_tokens", 1),
    skip_special_tokens=kwargs.get("skip_special_tokens", False),
    include_stop_str_in_output=True,
)
```
</details>

之后，是 `init_process_group` 和 `update_weight`，我真的太熟悉不过了。因为这俩接口是我写的，我看 openrlhf 貌似目前用的还是他们自己写的 Wrapper，不是vllm 的官方代码？无所谓，这里我很熟悉的切换到 sglang 的代码：

<details>
<summary> 参数更新的相关代码 </summary>

`init_process_group`：

```python
if self.backend == "vllm":
    if self.use_gpu_executor:
        return self.llm.llm_engine.model_executor.driver_worker.init_process_group(
            master_address, master_port, rank_offset, world_size, group_name, backend
        )
    else:
        return self.llm.llm_engine.model_executor._run_workers(
            "init_process_group", master_address, master_port, rank_offset, world_size, group_name, backend
        )
elif self.backend == "sglang":
    return self.llm.init_weights_update_group(
        master_address, master_port, rank_offset, world_size, group_name, backend="nccl"
    )
```

`update_weight`：

```python
if self.backend == "vllm":
    self.stop_remote_worker_execution_loop()

    if self.use_gpu_executor:
        return self.llm.llm_engine.model_executor.driver_worker.update_weight(name, dtype, shape, empty_cache)
    else:
        return self.llm.llm_engine.model_executor._run_workers(
            "update_weight", name, dtype, shape, empty_cache
        )
elif self.backend == "sglang":
    return self.llm.update_weights_from_distributed(name, dtype, shape)
```

</details>

这里其实我也犯过迷糊，因为一开始 sglang 的 training pipeline 会 OOM，我对比了下 openrlhf 给 vllm 写的 Wrapper，看到他们更新完了参数会 `del weights`，但我在 sglang 里面没有，我以为是因此 sglang 内存泄漏了。实际上不是，python 自己就会做这种函数内的内存回收，实际上 OOM 是从 deepspeed engine 来的。我把 training batch size 减小，就不会 OOM 了。这里其实还是前面提到的那个猜想，是否是因为 sglang 给出的 token ids 矩阵太大了，直接导致了 OOM？

### 效率波动的猜想

如我前面所述，在我看来，我的修改都是完全等价的，倘若 sglang engine 和 vllm engine works functionally equivalent，那么不该有任何区别。不过，我坚信两个框架都是无数用户使用后已经非常稳定的产品，差别大概率来自我没有注意到的不等价映射，特别是 serving params 和 sampling params 的映射。这里总结下我所有的猜想和 TODO：

1. 直接传入 token ids 给 sglang，不要再对 prompts tokenize 一次了。
2. 打印出 tokens 的开始和结尾，用于检查 vllm 和 sglang 处理特殊 token 是否有区别。
3. 打印出传入给 experience making 的 tokens 矩阵大小，难道二者的矩阵大小差异（譬如最长的 string 特别长导致 padding 后差异特别大）会有显著影响么？
4. 测试 main 上是否也会卡顿？
5. 测试是否是 backend 的问题？
6. 检查 vllm 到 sglang 的参数映射是否正确？
7. 同样的，检测 sampling params 是否正确？
8. 查看 cache hit rate，性能上应该还有提升空间。
9. 测试是否是环境问题，甚至换一台设备试试。
10. all_prompt_tokens 和 input token ids in engine outputs 的区别。
11. 打印下每个 training step 的 input tensor size 和 时间，检查下为什么有的地方卡一个小时。

## 对拍

### 启动 ray 集群

<details>
<summary> launch ray 在 NV 01 上的 docker</summary>

```bash
al 6

ray stop

ray start --head --node-ip-address 127.0.0.1 --num-gpus 6 --port 1234 --temp-dir="/root/.cache/ray"

pkill -9 -f train_ppo_ray

rm -rf /root/rlhf-ckpt/*
```
</details>

<details>
<summary> launch ray 在 NV 02 上不用 docker</summary>

```bash
al 6

ray stop

ray start --head --node-ip-address 127.0.0.1 --num-gpus 6 --port 1234 --temp-dir="/opt/dlami/nvme/chenyang/.cache/ray"

pkill -9 -f train_ppo_ray

rm -rf /opt/dlami/nvme/chenyang/rlhf-ckpt/*
```

</details>

### 100k 对拍在 NV 01 使用 Docker

<details>
<summary> 100k 对拍 </summary>

```bash
# conda activate rlhf-sglang

rlhf-sglang

TIME=$(now)

echo $TIME

ray job submit --address="172.17.0.3:1234" \
   --runtime-env-json='{
     "working_dir": "/root/rlhf-ckpt",
     "env_vars": {
       "PYTHONPATH": "/root/miniconda3/envs/rlhf-sglang/lib/python3.11/site-packages"
     }
   }' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --backend sglang \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
   --save_path /root/rlhf-ckpt/examples/checkpoint-sglang-$(now)/llama3-8b-rlhf \
   --save_steps 5 \
   --micro_train_batch_size 8 \
   --train_batch_size 64 \
   --micro_rollout_batch_size 32 \
   --rollout_batch_size 1024 \
   --max_samples 100000 \
   --max_epochs 2 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --packing_samples \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --use_wandb $WANDB_API_KEY \
   --wandb_run_name sglang-$TIME \
   --wandb_project openrlhf >> ~/log/sglang-$TIME.log
```

```bash
# conda activate rlhf-vllm

rlhf-vllm

TIME=$(now)

echo $TIME

ray job submit --address="172.17.0.3:1234" \
   --runtime-env-json='{
     "working_dir": "/root/rlhf-ckpt",
     "env_vars": {
       "PYTHONPATH": "/root/miniconda3/envs/rlhf-vllm/lib/python3.11/site-packages"
     }
   }' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --backend vllm \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
   --save_path /root/rlhf-ckpt/examples/checkpoint-vllm-$(now)/llama3-8b-rlhf \
   --save_steps 5 \
   --micro_train_batch_size 8 \
   --train_batch_size 64 \
   --micro_rollout_batch_size 32 \
   --rollout_batch_size 1024 \
   --max_samples 100000 \
   --max_epochs 2 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --packing_samples \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --use_wandb $WANDB_API_KEY \
   --wandb_project openrlhf \
   --wandb_run_name vllm-$TIME >> ~/log/vllm-$TIME.log
```

</details>

### 100k 在 NV 02 非 docker 对拍


```bash
# conda activate rlhf-sglang

rlhf-sglang

TIME=$(now)

echo $TIME

ray job submit --address="172.31.59.18:1234" \
   --runtime-env-json='{
     "working_dir": "/opt/dlami/nvme/chenyang/rlhf-ckpt",
     "env_vars": {
       "PYTHONPATH": "/opt/dlami/nvme/chenyang/.miniconda3/envs/rlhf-sglang/lib/python3.11/site-packages"
     }
   }' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --backend sglang \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
   --save_path /opt/dlami/nvme/chenyang/rlhf-ckpt/examples/checkpoint-sglang-$(now)/llama3-8b-rlhf \
   --save_steps 5 \
   --micro_train_batch_size 8 \
   --train_batch_size 64 \
   --micro_rollout_batch_size 32 \
   --rollout_batch_size 1024 \
   --max_samples 100000 \
   --max_epochs 2 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --packing_samples \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --use_wandb $WANDB_API_KEY \
   --wandb_run_name sglang-$TIME \
   --wandb_project openrlhf >> ~/log/sglang-$TIME.log
```

```bash
# conda activate rlhf-vllm

rlhf-vllm

TIME=$(now)

echo $TIME

ray job submit --address="172.31.59.18:1234" \
   --runtime-env-json='{
     "working_dir": "/opt/dlami/nvme/chenyang/rlhf-ckpt",
     "env_vars": {
       "PYTHONPATH": "/opt/dlami/nvme/chenyang/.miniconda3/envs/rlhf-vllm/lib/python3.11/site-packages"
     }
   }' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --backend vllm \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
   --save_path /opt/dlami/nvme/chenyang/rlhf-ckpt/examples/checkpoint-vllm-$(now)/llama3-8b-rlhf \
   --save_steps 5 \
   --micro_train_batch_size 8 \
   --train_batch_size 64 \
   --micro_rollout_batch_size 32 \
   --rollout_batch_size 1024 \
   --max_samples 100000 \
   --max_epochs 2 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --packing_samples \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --use_wandb $WANDB_API_KEY \
   --wandb_project openrlhf \
   --wandb_run_name vllm-$TIME >> ~/log/vllm-$TIME.log
```

</details>

### 512 作为单测

```bash

al 3

ray stop

ray start --head --node-ip-address 127.0.0.1 --num-gpus 3 --port 1234 --temp-dir="/opt/dlami/nvme/chenyang/.cache/ray"

pkill -9 -f train_ppo_ray

rm -rf /opt/dlami/nvme/chenyang/rlhf-ckpt/*
```

<details>
<summary> 512 对拍单测 </summary>

```bash
rlhf-sglang

TIME=$(now)

echo $TIME

ray job submit --address="172.31.59.18:1234" \
   --runtime-env-json='{
     "working_dir": "/root/rlhf-ckpt",
     "env_vars": {
       "PYTHONPATH": "/opt/dlami/nvme/chenyang/.miniconda3/envs/rlhf-sglang/lib/python3.11/site-packages"
     }
   }' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --backend sglang \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
   --save_path /opt/dlami/nvme/chenyang/rlhf-ckpt/examples/checkpoint-sglang-$(now)/llama3-8b-rlhf \
   --save_steps 10 \
   --micro_train_batch_size 16 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 32 \
   --rollout_batch_size 128 \
   --max_samples 512 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --packing_samples \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --use_wandb $WANDB_API_KEY \
   --wandb_run_name sglang-$TIME \
   --wandb_project openrlhf >> ~/log/sglang-$TIME.log
```

```bash
rlhf-vllm

TIME=$(now)

echo $TIME

ray job submit --address="172.17.0.3:1234" \
   --runtime-env-json='{
     "working_dir": "/root/rlhf-ckpt",
     "env_vars": {
       "PYTHONPATH": "/root/.miniconda3/envs/rlhf-vllm/lib/python3.11/site-packages"
     }
   }' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --backend vllm \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
   --save_path /root/rlhf-ckpt/examples/checkpoint-vllm-$(now)/llama3-8b-rlhf \
   --save_steps 10 \
   --micro_train_batch_size 16 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 32 \
   --rollout_batch_size 128 \
   --max_samples 512 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --packing_samples \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --use_wandb $WANDB_API_KEY \
   --wandb_run_name vllm-$TIME \
   --wandb_project openrlhf >> ~/log/vllm-$TIME.log
```

</details>

### main 上的测试

<details>
<summary> main 上的测试 </summary>

```bash
al 3

ray stop

ray start --head --node-ip-address 127.0.0.1 --num-gpus 3 --port 1234 --temp-dir="/opt/dlami/nvme/chenyang/.cache/ray"

pkill -9 -f train_ppo_ray

rm -rf /opt/dlami/nvme/chenyang/rlhf-ckpt/*
```

```bash
rlhf-vllm

TIME=$(now)

echo $TIME

ray job submit --address="172.31.59.18:1234" \
   --runtime-env-json='{
     "working_dir": "/opt/dlami/nvme/chenyang/rlhf-ckpt",
     "env_vars": {
       "PYTHONPATH": "/opt/dlami/nvme/chenyang/.miniconda3/envs/rlhf-vllm/lib/python3.11/site-packages"
     }
   }' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
   --save_path /opt/dlami/nvme/chenyang/rlhf-ckpt/examples/checkpoint-vllm-main-$(now)/llama3-8b-rlhf \
   --save_steps 5 \
   --micro_train_batch_size 8 \
   --train_batch_size 64 \
   --micro_rollout_batch_size 32 \
   --rollout_batch_size 1024 \
   --max_samples 100000 \
   --max_epochs 2 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --packing_samples \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --use_wandb $WANDB_API_KEY \
   --wandb_run_name vllm-main-$TIME \
   --wandb_project openrlhf >> ~/log/vllm-main-$TIME.log
```

</details>

### 100k 在 8 卡 A6000 上对拍

```bash

al 8

ray stop

ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 --port 3456 --temp-dir="/data/chenyang/.cache/ray"

pkill -9 -f train_ppo_ray

rm -rf /data/chenyang/rlhf-ckpt/*
```

```bash

rlhf-vllm

TIME=$(now)

ray job submit --address="http://131.179.88.84:3456" \
   --runtime-env-json='{"working_dir": "/data/chenyang/rlhf-ckpt"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --backend vllm \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 2 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 2 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
   --save_path /data/chenyang/rlhf-ckpt/examples/checkpoint/llama3-8b-rlhf \
   --micro_train_batch_size 8 \
   --train_batch_size 64 \
   --micro_rollout_batch_size 32 \
   --rollout_batch_size 1024 \
   --max_samples 100000 \
   --max_epochs 2 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --packing_samples \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --use_wandb $WANDB_API_KEY \
   --wandb_run_name vllm-8a6000-$TIME \
   --wandb_project openrlhf >> ~/log/vllm-8a6000-$TIME.log
```
