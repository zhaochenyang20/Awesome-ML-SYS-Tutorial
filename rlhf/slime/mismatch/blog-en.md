# Let Speed Be With Stability: All-In-One Solution to Training-Inference Mismatch with Miles

> TL;DR: We investigate the "Training-Inference Mismatch" in LLM-RL--a phenomenon where numerical inconsistencies between rollout and training engines threaten stability. We introduce two comprehensive solutions implemented in Miles: Truly On Policy training (backend alignment for bitwise precision) and Algorithmic Mitigation (correction via TIS/MIS). While Miles demonstrates impressive stability in practice, we provide these robust tools to ensure correctness and efficiency for the broader RL community.

The SGLang RL Team and the Miles community have recently conducted some interesting explorations around RL training stability and acceleration:

- [Aligning the SGLang and FSDP backends for strictly zero KL divergence](https://github.com/radixark/miles/tree/main/examples/true_on_policy): achieving perfect train–inference consistency on dense models.

- [Power Up Speculative Decoding into Reinforement Learning](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/slime/spec/readme-en.md): significantly speeding up sampling under suitable configurations.

- [Unified FP8: Moving Beyond Mixed Precision for Stable and Accelerated MoE RL](https://lmsys.org/blog/2025-11-25-fp8-rl/): eliminating quantization error and improving both speed and stability of RL training.

- [Support FSDP2 as A Flexible Training Backend for Miles](https://lmsys.org/blog/2025-12-03-miles-fsdp/): adding FSDP2 as a flexible training backend to support architecture-innovative models and align with Megatron.


In this post, we further discuss the first work and share our understanding of the training-inference mismatch problem and our proposed solutions.

"Training-Inference Mismatch" refers to the numerical inconsistencies that arise between the rollout (inference) engine and the training engine. Even when utilizing identical model weights, these engines often produce divergent log-probabilities for the same token sequence. In this post, we analyze the root causes of this divergence and present Miles' dual-approach solution.

For those seeking absolute correctness, we offer a [Truly On Policy mode](https://github.com/radixark/Miles/blob/main/examples/true_on_policy/README.md) that achieves bitwise-exact alignment between SGLang and FSDP. For those prioritizing throughput, we provide Algorithmic Mitigation strategies, such as [Masked Importance Sampling (MIS)](https://richardli.xyz/rl-collapse-3) and [Truncated Importance Sampling (TIS)](https://fengyao.notion.site/off-policy-rl#279721e3f6c48092bbe2fcfe0e9c6b33). Our experiments demonstrate that MIS effectively suppresses mismatch growth during late-stage training while preserving high performance, making it a robust default for RL practitioners.

## What is Training Inference Mismatch?

<div align="center">
  <img src="pics/training-inference-mismatch.png" alt="Training Inference Mismatch" width="50%">
</div>

Training-Inference Mismatch refers to the numerical inconsistency between the rollout (inference) engine and the training engine. Even when both engines utilize identical model weights, they often produce slightly different log-probabilities for the same token sequence. This divergence stems from infrastructure-level variances, such as differing CUDA kernels, batch sizes, expert selection logic, and reduction orders (see Thinking Machine Lab [blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)).

> ⚠️ While it is widely claimed that training-inference mismatch can trigger RL collapse, we have not encountered this issue in practice, even during the post-training of frontier models like GLM 4.6.

To quantify this discrepancy, we use the K3 KL divergence (see [Reference 8](http://joschu.net/blog/kl-approx.html) for details). In dense models, K3 KL typically ranges from $10^{-5}$ to $10^{-3}$, while in Mixture-of-Experts (MoE) models, it increases to between $10^{-3}$ and $10^{-1}$. Although this mismatch is often minor, it technically introduces an off-policy effect: the policy used for sampling is not strictly identical to the one used for loss computation. In complex scenarios, such as multi-turn agent tasks, existing literature suggests that these small discrepancies can accumulate over time, potentially destabilizing or collapsing the training process (e.g., [blog 1](https://richardli.xyz/rl-collapse) and [blog 2](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)).

Miles treats this mismatch as a non-negligible aspect of RL system design. Users can choose to eliminate it entirely for correctness or mitigate it for efficiency.

> ⚠️ Call for Collaboration: Miles has proven remarkably resilient to mismatch across various scales. We attempted to force a baseline collapse but were unsuccessful. If you are aware of open-source RL tasks that reproducibly collapse due to mismatch on a single node, please reach out to us.

## Why Training and Inference Can Be Different

The fundamental reason is the non-associative property of floating point addition. For example, when the batch size is small, kernels may use split-reduction optimizations, which change the reduction order depending on the input size. Since floating-point arithmetic is non-associative, accumulating values in different orders introduces numerical discrepancies. Each tensor-core instruction may also perform reduction internally in a different order (ref: Thinking Machine Lab [blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)).

As a result, even in SGLang, performing inference on the same samples with different batch sizes can yield slightly different numerical outputs. In addition, rollout and training have fundamentally different workloads in RL: rollout generates tokens one-by-one with tiny effective matrices, while training processes full sequences in large batches. These vastly different matrix shapes lead the system to select different GPU kernels, further amplifying the rollout–training mismatch.

## Mitigation of Mismatch

Given the existence and partial cause of  training inference mismatch, we present two solutions:

1. **Truly On Policy**: We align every operator backend between rollout and training so that rollout log probs and training log probs are bitwise identical. This achieves training inference KL = 0, giving you 100% Truly On Policy behavior.
2. **Algorithmic Correction**: Instead of forcing the use of aligned kernels for both inference and training (which reduces efficiency to certain degree), we treat rollout log-probs as the authoritative behavior policy and use importance sampling or rejection sampling to conduct off-policy rollout correction.

We provide these options to the community and try our best to make RL training more stable and debuggable.

## Truly On Policy Training

As we revealed, the key to fully eliminating the mismatch is to align all the operator backends between training and rollout—making every operation in training and inference bitwise-identical. To achieve this goal, we carefully selected the kernels we used for each model component.

Specifically, we use batch-invariant kernels: This is a prerequisite for Truly On Policy, and we adopted the kernels from the Thinking Machines. This implementation provides the batch-invariant kernels for RMSNorm, Matmul, and other common operators, including log_softmax and mean. 

Based on this implementation, we added the following implementations and optimizations:

- FlashAttention-3: We use the Flash Attention 3 backend for both training and inference, since it achieves bitwise equality between prefill and decode operations while staying efficient compared to the Triton version. It also supports Radix Cache.
- DeepGEMM: In our Truly On Policy implementation, we used DeepGEMM's fast matrix multiplication as a deterministic backend, which is more efficient. For different input sizes, DeepGEMM will use a fixed reduction order and tensor core instruction, which is independent of the shape changes.
- Torch.compile(): To improve efficiency when enabling Truly On Policy, we use torch.compile to speed up by avoiding many tiny kernels. Some operations, for example, RoPE is also compiled to speed up.
- Numeric alignment: We also align numeric operation details between the two systems for simplicity, such as op dtype, detailed kernels, etc.

## Algorithmic Mitigation

<div align="center">
  <img src="pics/algorithmic-mitigation.png" alt="Algorithmic Mitigation" width="50%">
</div>

Let's first look at why this mismatch matters from an algorithmic perspective. The original PPO objective is shown below, where \(\pi_\theta\) denotes the current policy being optimized and used to compute the training loss, and \(\pi_{\text{old}}\) denotes the behavior policy that generated the rollout data (i.e., the action probabilities from the model before the current update step).

$$\mathcal{L}_{\text{PPO}}(\theta)
= - \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim \pi_{\textcolor{red}{\text{old}}}} \left[
  \sum_{t=0}^{|y|-1}
  \min \left(
    \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\textcolor{red}{\text{old}}}(y_t \mid x, y_{<t})} A_t,
    \text{clip}\left(
      \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\textcolor{red}{\text{old}}}(y_t \mid x, y_{<t})},
      1 - \epsilon,
      1 + \epsilon
    \right) A_t
  \right)
\right]$$

This is the basic PPO algorithm with the training-inference mismatch issue when the output of SGLang and Megatron does not exactly match. In this formula, the policy used for sampling comes from SGLang, while the one used for computing loss comes from Megatron. This mismatch makes the PPO loss an incorrect form of importance sampling.

$$\mathcal{L}_{\text{PPO}}(\theta)
= - \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim \pi_{\textcolor{red}{\text{SGLang}}}} \left[
  \sum_{t=0}^{|y|-1}
  \min \left(
    \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\textcolor{blue}{\text{Megatron}}}(y_t \mid x, y_{<t})} A_t,
    \text{clip}\left(
      \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\textcolor{blue}{\text{Megatron}}}(y_t \mid x, y_{<t})},
      1 - \epsilon,\,
      1 + \epsilon
    \right) A_t
  \right)
\right]$$

### By-Passing Old Log-Prob in PPO Importance Sampling

<div align="center">
  <img src="pics/bypassing-ppo.png" alt="Bypassing and Unified PPO Importance Sampling" width="50%">
</div>

To achieve algorithmic correctness, one may directly use the rollout engine's log-probs as the old policy in offline PPO's importance sampling, rather than the recomputed log-probs from the training engine. Then it becomes the correct math form:

$$\mathcal{L}_{\text{PPO}}(\theta)
= - \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim \pi_{\textcolor{red}{\text{SGLang}}}} \left[
  \sum_{t=0}^{|y|-1}
  \min \left(
    \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\textcolor{red}{\text{SGLang}}}(y_t \mid x, y_{<t})} A_t,
    \text{clip}\left(
      \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\textcolor{red}{\text{SGLang}}}(y_t \mid x, y_{<t})},
      1 - \epsilon,
      1 + \epsilon
    \right) A_t
  \right)
\right]$$

In this way, the log_prob recomputation on the training engine will be skipped - it will save one forward pass computation on all the generated trajectories.

### Decoupled PPO Importance Sampling

<div align="center">
  <img src="pics/decoupled-ppo.png" alt="Decoupled, 3-policy PPO Importance Sampling" width="50%">
</div>

However, sometimes you may want to decouple the training-rollout mismatch from the general off-policy importance sampling. Decoupled PPO achieves batch-independent PPO by decoupling two roles: Proximal Policy (anchor policy for PPO clipping, control update size) and Behavior Policy (for off-policy correction in importance sampling). Therefore, there are 3 roles engaged in this mode: target policy  $\pi_\theta$ , proximal policy $\pi_{\textcolor{blue}{\text{old}}}$, and behavior policy $\pi_{\textcolor{red}{\text{SGLang}}}$. $\pi_{\textcolor{blue}{\text{old}}}$ is recomputed with Megatron at the beginning of each training step. See reference 6 and 7 for details. The total formula is below:

$$\mathcal{L}_{\text{PPO-decoupled}}(\theta)
= - \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim \pi_{\textcolor{red}{\text{SGLang}}}} \left[
  \sum_{t=0}^{|y|-1}
  \frac{\pi_{\textcolor{blue}{\text{old}}}(y_t \mid x, y_{<t})}{\pi_{\textcolor{red}{\text{SGLang}}}(y_t \mid x, y_{<t})}
  \min \left(
    \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\textcolor{blue}{\text{old}}}(y_t \mid x, y_{<t})} A_t,
    \mathrm{clip}\left(
      \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\textcolor{blue}{\text{old}}}(y_t \mid x, y_{<t})},
      1 - \epsilon,
      1 + \epsilon
    \right) A_t
  \right)
\right].$$

The first importance ratio $\frac{\pi_{\text{old}}(y|x)}{\pi_{\text{SGLang}}(y|x)}$ naturally behaves like a dynamic learning-rate scaling term. When the rollout distribution deviates from the proximal policy, the ratio shrinks the effective update (similar to trust-region control). This directly connects to the later smoothing strategy that prevents large updates induced by rollout-training mismatch.

### Batch Normalization & Bias-Variance Trade-off

While this first importance ratio already acts as a per-token adaptive learning-rate controller, the control is still stochastic at the batch level: batches sampled from “easier” regions of the behavior policy tend to amplify the effective step size, while rare or mismatched samples shrink it dramatically.

Thus, we strongly recommend enabling --tis-batch-normalize (Self-Normalized Importance Sampling) when using Sequence or Geometric levels. This technique addresses two critical issues in off-policy training: Learning Rate Stability and the Bias-Variance Trade-off.

In standard importance sampling, the average weight of each batch can vary dramatically depending on whether the sampled data were “likely” or “unlikely” under the behavior policy, which causes the effective learning rate to oscillate and destabilize training. Self-normalizing the weights so that their mean is always 1 keeps the step size consistent across updates and substantially reduces batch-to-batch variance.

Because this normalization already suppresses variance, we can relax clipping or masking thresholds and therefore reduce the bias they introduce. As the batch size grows large, self-normalization alone can make the estimator both stable and nearly unbiased, without relying on aggressive truncation.

### Masked Importance Sampling

In addition to clipping-based importance sampling, we provide masking and rejection sampling (RS) as a stronger safeguard against training-inference mismatch. When the rollout engine assigns extremely low probability to a sampled token, the importance ratio can grow to an unsafe magnitude (i.e. 1e12). Even if clipped, such cases still inject incorrect gradients into training. RS avoids this issue entirely by discarding those tokens—or the entire sequence, if necessary—when the ratio exceeds a preset trust threshold, preventing harmful updates from taking effect.

This mechanism enforces a more principled trust region: if the sampled behavior deviates too far from the proximal policy, we simply do not learn from that sample. It guarantees that all effective training data remain consistent with the assumed rollout distribution and protects the optimization from collapse in cases where mismatch becomes extreme.

Pure rejection sampling, however, may reduce the amount of usable data and increase variance, especially when mismatch is moderate. Therefore, we combine RS with importance sampling in MIS: IS maintains mathematical correction for most tokens, while RS acts as a safety valve only when discrepancies become severe. In our experiments, this hybrid approach provides stable performance and improves robustness during the late-stage mismatch surge without sacrificing learning efficiency.

> See [here](https://richardli.xyz/rl-collapse-3) for full explanation.

## Experiments

Before diving into experiments, it is worth discussing why training-inference mismatch has only become a widely discussed topic recently. For a long time, the RL community did not have access to the *correct* rollout-engine log probabilities—specifically, the log probs corresponding to the tokens actually sampled after applying various sampling parameters. Historically, many pipelines incorrectly used the raw (pre-adjustment) log probs from the rollout engine. This missing piece made the mismatch issue quietly persist in RL training, and only recently has it been surfaced and studied more systematically.

When identifying a set of importance-sampling (IS) baselines, we encountered a requirement that does not appear in most prior RLHF or agent-training baselines: We must be able to get the log-probabilities from the rollout engine.

This means no post-processing is allowed on the model output, because any modification to the response string breaks the correspondence between the sampled tokens and the tokens whose log-probs we later evaluate.

Unfortunately, many existing agent baselines do rely on lightweight post-processing, often for simple tasks like trimming labels, removing prefixes, or completing partial responses. These operations are common in classic agent examples, but they invalidate log-prob evaluation for IS-correct RL.

For example:
- Search-R1 performs post-processing in response:[Link](https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/llm_agent/generation.py#L54)
- Retool does the same: [Link](https://github.com/radixark/Miles/blob/main/examples/retool/generate_with_retool.py#L147)

At the moment, we have not found a solid theoretical reason why these agent tasks require such post-processing. Fortunately, removing the post-processing entirely and using the model’s raw output still yields rewards that are similar to the original baselines. We therefore adopt this simple workaround for now, though the downstream effects remain uncertain.

⚠️ Some researchers also suggest an alternative: if post-processing is unavoidable, you may re-run a forward pass on the rollout engine for the post-processed sequence to obtain the correct log probs. However, this cost is significant, and we believe that directly removing post-processing is often a practical choice for strong base models.

Additionally, due to limited resource and time, we chose to use GRPO instead of PPO to demonstrate IS behavior.

### Existence of Mismatch

We first confirm that as the training goes on, the K3 KL between Rollout Engine and Training Engine will increase. Our setting is:
- Training dataset: [Link](https://huggingface.co/datasets/aaabiao/dapo_filter)
- Eval dataset: aime 24 + aime 25
- Base Model: Qwen3-4b-base ([Link](https://huggingface.co/Qwen/Qwen3-4B-Base))
- Algorithm: REINFORCE (Williams et al. 1992)

<div align="center">
  <img src="pics/mismatch-existence.png" alt="Existence of Mismatch" width="50%">
</div>

<p align="center">
  <img src="pics/base-eval.png" width="45%" />
  <img src="pics/base-reward.png" width="45%" />
</p>

You can see in the initial step of training, as the model learns and perplexity drops, K3 KL actually drops. But after 600 steps, although the train and eval reward remains stable, the K3 KL metrics start to increase dramatically, indicating the existence of training and rollout mismatch.

### IS Won't Harm Performance

> See our full weight&bias log [here](https://wandb.ai/ch271828n-team/slime-dapo/reports/IS-Has-No-Harm--VmlldzoxNTE3NTM3MQ?accessToken=vbaw93cjkyi8d6iul7gzvccehf2ugff1cicfcmlaxjv88n875i0ip1ixqfr42s9b).

In our experiments, we also verified that enabling distribution correction—including several commonly used configurations—does not degrade performance or destabilize training. To demonstrate this, we enabled different IS-related options at the beginning of training and compared them against a baseline with no IS correction.
Below are the four configurations we evaluated:

1. Baseline
2. Token-level Importance Sampling(IS)
3. Token-level IS + Sequence Masking/Rejection Sampling(RS) [a.k.a [MIS](https://richardli.xyz/rl-collapse-3)]
4. Token-level IS + Sequence Masking/Rejection Sampling(RS) + Batch Normalization(BN) [a.k.a [MIS](https://richardli.xyz/rl-collapse-3)]

Across all settings, we consistently observed stable training curves. All four configurations successfully reproduced the characteristic length increase after ~100 steps, indicating that enabling IS does not negatively impact the learning dynamics. Additionally, across all experiments, the reward begins to improve only after the response length starts to increase; prior to this, the reward stagnates around 0.32. Based on these results, we recommend enabling IS as a default configuration, as it provides mismatch correction without sacrificing performance.

<div align="center">
  <img src="pics/is-performance.png" alt="IS Won't Harm Performance" width="45%">
  <img src="pics/experiment-raw-reward.png" alt="Raw Reward (Moving Average)" width="45%">
</div>
<p align="center">
    <em>Left: Response Length. Right: Raw Reward (smoothed with moving average).</em>
</p>

We also examined the K3 KL divergence for these runs. We observed that across all settings, as the training perplexity (PPL) decreases, the training-inference mismatch (measured by K3 KL) also diminishes, which is consistent with our long base run above.

<div align="center">
  <img src="pics/experiment-mis-k3-kl.png" alt="K3 KL Divergence" width="45%">
  <img src="pics/experiment-ppl.png" alt="Training PPL" width="45%">
</div>
<p align="center">
    <em>Left: K3 KL Divergence. Right: Training Perplexity (PPL).</em>
</p>


### IS Can Suppress KL Increase

To test whether MIS (IS + RS + BN) works, we continue training on step 650, and the result is below. You can see that for the base run, kl continues to increase, but with MIS, the increasing trend is successfully depressed and starts to decrease.

<div align="center">
  <img src="pics/is-kl-suppression.png" alt="IS Can Suppress KL Increase" width="50%">
</div>

⚠️: Due to limited resources and time in academia, our experiments are primarily conducted on small-scale dense models, and the conclusions may differ from those of large-scale MOE training. Please look forward to our future blogs with collaboration across more industrial partners.

## Usage

For more details, we provide complete guides and runnable examples:
- Truly On Policy Training (FSDP): [Link](https://github.com/radixark/Miles/tree/main/examples/true_on_policy)
- Algorithmic Mismatch Correction (Megatron): [Link](https://github.com/radixark/Miles/tree/main/examples/train_infer_mismatch_helper)

If your goal is to fully eliminate the rollout–training mismatch, we recommend the Truly On Policy solution.

If you prefer to retain high performance while mitigating mismatch, algorithmic correction such as [MIS](https://richardli.xyz/rl-collapse-3) is a lightweight and effective choice.

Below is a brief overview of the available options.

### Truly On Policy

To open Truly On Policy mode, add args:

```bash
CUSTOM_ARGS=(
    --true-on-policy-mode
)
```

### Algorithmic Mitigation

> Please refer to [this link](https://github.com/radixark/Miles/blob/main/examples/train_infer_mismatch_helper/README.md) for a long and complete explanation of each attribute.

Miles provides a comprehensive configuration system allowing users to flexibly balance Bias and Variance. To open Importance sampling, you must add the following attribute to your starting script.

```bash
CUSTOM_ARGS=(
   --use-tis
   --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)
```

Then you can adjust the detail configuration in [this link](https://github.com/radixark/Miles/blob/main/examples/train_infer_mismatch_helper/mis.yaml).

<details>
<summary>IS Configuration Details</summary>

In short, you can configure your correction strategy across four key dimensions:

1. Calculation Levels

This determines how important weights are aggregated from tokens to sequences.
- **Token Level**
  - Computes weights independently for each token.
  - Characteristics: Computationally simple but mathematically biased. Suitable for most general scenarios.
- **Sequence Level**
  - The sequence weight is the product of all token weights.
  - Characteristics: Mathematically unbiased but suffers from extreme variance. Recommended only when the mismatch is very small or the batch size is large.
- **Geometric Level**
  - Uses the geometric mean of all token weights as the sequence weight.
  - Characteristics: A trade-off solution. It retains sequence-level information while avoiding the numerical instability of the product method, striking a balance between bias and variance. It also provides some length-invariant property for long-context tasks.

2. Importance Weight Constraints & Trust Regions

To prevent extreme importance weights from destabilizing training and to enforce a hard trust region, we apply specific constraints to the weights.

- **IS Mode (Importance Sampling)**
  - --tis-mode: Strategies include clip or truncate. This constrains importance weights to remain within the $[lower\_bound, upper\_bound]$ range, mitigating high variance.

- **RS Mode (Rejection Sampling)**
  - --use-rs: Instead of clipping weights, RS strictly discards (drops) tokens or sequences that fall outside the specified threshold. While this reduces the effective sample size, it ensures that the gradient update is calculated exclusively using data within the trust region ("gradient purity").

- **Mask Mode (Masking)**
  - --use-mask: This mode applies a mask to tokens or sequences falling outside the threshold during the gradient update. Unlike RS, this preserves the original batch structure (and nominal sample size), while effectively zeroing out the gradient contribution from invalid data.

[MIS](https://richardli.xyz/rl-collapse-3) introduces combinations of IS and RS/Masking at different levels.

3. Veto Mechanism

This acts as a low-level safety net independent of IS/RS settings.
- Mechanism: If a sequence contains any token with a probability lower than the veto threshold (e.g., $p < 10^{-6}$) under the old policy, the entire sequence is discarded.
- Why it's needed: It prevents "catastrophic updates." Even if clipped, a token with near-zero probability in the denominator can introduce numerical instability or destructive gradients.

4. Self-Normalization

`--tis-batch-normalize`: Self-Normalization. Normalizes the importance weights across the entire batch so that their mean equals 1.0. This prevents the magnitude of weights from destabilizing the training step size.

</details>

## More Mismatch-Solving Features

In upstream Miles, you can also find additional mismatch-related tooling, for example:
  - Unbiased KL estimation from Deepseek V3.2: [Link](https://github.com/THUDM/slime/pull/1004)
  - Rollout routing replay: [Link](https://github.com/THUDM/slime/pull/715)
  - Truly On Policy training for VLMs: [Link](https://github.com/radixark/Miles/tree/main/examples/true_on_policy_vlm)

Any mismatch solving tool can be found in Miles (or its upstream Miles)!

## Acknowledgments

Bytedance Inc: Yingru Li, Jiacai Liu, Yuxuan Tong, Qian Liu, Hongyu Lu, Ziheng Jiang

SGLang RL Team: Changyi Yang, Chenxing Xie, Zilin Zhu, Ji Li, Yuzhen Zhou

RadixArk Miles Team: Chenyang Zhao, Yueming Yuan, Jiajun Li, Banghua Zhu, Tom, Yusheng Su

We sincerely thanks Qiwei Di, Xuheng Li, Heyang Zhao and Prof. Quanquan Gu from UCLA, as well as Liyuan Liu and Feng Yao from Thinking Machines Lab for their valuable suggestions and discussions.

## Reference

1. When Speed Kills Stability: Demystifying RL collapse from the training-inference mismatch [blog](https://richardli.xyz/rl-collapse)
  - Part 1: Why Off-Policy Breaks RL — An SGA Analysis Framework [blog](https://richardli.xyz/rl-collapse-1)
  - Part 2: Applying the SGA Framework — Token v.s. Sequence-level Correction [blog](https://richardli.xyz/rl-collapse-2)
  - Part 3: Trust Region Optimization via Sequence Masking [blog](https://richardli.xyz/rl-collapse-3)
  - Mathematical Formulations of Rollout Correction Methods [docs](https://verl.readthedocs.io/en/latest/algo/rollout_corr_math.html)
2. Your Efficient RL Framework Secretly Brings You Off-Policy RL Training [blog](https://fengyao.notion.site/off-policy-rl#279721e3f6c48092bbe2fcfe0e9c6b33)
3. Simple statistical gradient-following algorithms for connectionist reinforcement learning. [link](https://link.springer.com/article/10.1007/BF00992696)
4. Defeating Nondeterminism in LLM Inference [blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
5. Small Leak Can Sink a Great Ship—Boost RL Training on MoE with 𝑰𝒄𝒆𝑷𝒐𝒑! [blog](https://ringtech.notion.site/icepop)
6. Batch size-invariance for policy optimization [link](https://arxiv.org/abs/2110.00641)
7. AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning [link](https://arxiv.org/abs/2505.24298)
8. [K3 KL-Definition](http://joschu.net/blog/kl-approx.html): $ k_3(x) = \frac{p(x)}{q(x)} - 1 - \log \frac{p(x)}{q(x)}$
