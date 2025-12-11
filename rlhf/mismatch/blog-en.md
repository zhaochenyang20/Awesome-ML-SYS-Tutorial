TLDR: We share our two perspectives on solving the training-inference mismatch: eliminate the mismatch by fully aligning SGLang and FSDP backends for bitwise-exact on-policy training, and algorithm mitigation with TIS/MIS methods.

Here, MIS (masked importance sampling) includes two operations: sequence/token masking + importance sampling. In addition, we explored various combinations of different settings and found that enabling MIS/TIS does not harm RL performance — therefore, we recommend enabling them as the default setting. We sometimes observed a notable increase in mismatch during the late training stage, though it did not lead to training collapse, and further demonstrated that applying importance sampling effectively suppresses this mismatch growth.

## What is Training Inference Mismatch?

<img src="pics/training-inference-mismatch.png" alt="Training Inference Mismatch" width="30%">

Training Inference Mismatch, in this post, refers to the numerical inconsistency between the rollout engine and the training engine. Even when both engines use the same model weights, they may produce slightly different log-probabilities for the same token sequence. This happens because rollout and training engines often use different kernels, different batch sizes, different activated experts, and different reduction orders. (ref TML blog)

It is widely said that training inference mismatch could lead to RL collapse. But to be honest, we never encounter this even in the post-training of the frontier model like GLM 4.6.

We use K3 KL to measure the discrepancy between the log probs used in rollout and those used in training (see Appendix for details). In dense model, K3 KL is e-5 to e-3. In MOE models, K3 KL is usually e-3 to e-1. Even though this mismatch is not always magnificent, it still introduces a subtle off-policy: the policy used for sampling is not exactly the same as the one used for computing loss. On difficult tasks, such as multi-turn agents, it is said that this small discrepancy could sometimes accumulate over time and eventually destabilize or even collapse the entire training process (ref blog section 3) .

In all these senses, the Training Inference Mismatch should be treated as a non-negligible issue of an RL system. Users may choose to eliminate entirely for correctness, or mitigate for efficiency. To support both needs, Miles provides two solutions, allowing users to choose the trade-offs that best match their system requirements:

In our experiments, RL training on Miles has been impressively stable in practice.  We spent plenty of time trying to find a collapsed baseline, but we weren't able to find one. If you know any open-source RL tasks that will collapse after certain steps due to mismatch increasing and can be reproduced in one node, feel free to reach out to us.

## Why Training and Inference Can Be Different

The reasons are various. For example, when the batch size is small, kernels may use split-reduction optimizations, which change the reduction order depending on the input size. Since floating-point arithmetic is non-associative, accumulating values in different orders introduces numerical discrepancies. Each tensor-core instruction may also perform reduction internally in a different order (ref: TML blog).

As a result, even in SGLang, performing inference on the same samples with different batch sizes can yield slightly different numerical outputs. In addition, rollout and training have fundamentally different workloads in RL: rollout generates tokens one-by-one with tiny effective matrices, while training processes full sequences in large batches. These vastly different matrix shapes lead the system to select different GPU kernels, further amplifying the rollout–training mismatch.

## Mitigation of Mismatch

Given the existence and partial cause of  training inference mismatch, we present two solutions:

1. We align every operator backend between rollout and training so that rollout log probs and training log probs are bitwise identical. This achieves training inference KL = 0, giving you 100% true on-policy behavior. 
2. Instead of forcing the use of aligned kernels for both inference and training (which reduces efficiency to an unacceptable degree), we treat rollout log-probs as the authoritative behavior policy and use importance sampling or rejection sampling to conduct off-policy rollout correction.

We provide these options to the community and try our best to make RL training more stable and debuggable.

## Truly On-Policy

As we revealed, the key to fully eliminating the mismatch is to align all the operator backends between training and rollout - to every operation in training and inference be bitwise equal. To achieve this goal, we carefully selected the kernels we used for each model component.

Specifically, we use batch-invariant kernels: This is a prerequisite for true on-policy, and we adopted the kernels from the Thinking Machines. This implementation provides the batch-invariant kernels for RMSNorm, Matmul, and other common operators, including log_softmax and mean. 

Based on this implementation, we added the following implementations and optimizations:

- FlashAttention-3: We use the Flash Attention 3 backend for both training and inference, since it achieves bitwise equally between prefill and decode operations while keeping efficient compared to the Triton version. It also supports Radix Cache.
- DeepGEMM: In our true on-policy implementation, we used DeepGEMM's fast matrix multiplication as a deterministic backend, which is more efficient. For different input sizes, DeepGEMM will use a fixed reduction order and tensor core instruction, which is independent of the shape changes.
- Torch.compile(): To improve efficiency when enabling true on-policy, we use torch.compile to speed up by avoiding many tiny kernels. Some operations, for example, RoPE is also compiled to speed up.
- Numeric alignment: We also align numeric operation details between the two systems for simplicity, such as op dtype, detailed kernels, etc.

## Algorithmic Mitigation

<img src="pics/algorithmic-mitigation.png" alt="Algorithmic Mitigation" width="30%">

Let's first look at the reason why this mismatch is important from an algorithm perspective. The original PPO algorithm formula is below, where $$\pi_\theta$$ denotes the current policy being optimized and used to compute the training loss and $$\pi_{\text{old}}
$$ denotes the behavior policy that generated the rollout data, i.e., the action probabilities from the model before the current update step.

$$\mathcal{L}_{\text{PPO}}(\theta)
= - \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim \pi_{\textcolor{red}{\text{old}}}} \left[
  \sum_{t=0}^{|y|-1}
  \min \left(
    \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\textcolor{red}{\text{old}}}(y_t \mid x, y_{<t})} A_t,\,
    \text{clip}\left(
      \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\textcolor{red}{\text{old}}}(y_t \mid x, y_{<t})},\,
      1 - \epsilon,\,
      1 + \epsilon
    \right) A_t
  \right)
\right]$$

This is the basic PPO algorithm with the training-inference mismatch issue when the output of SGLang and Megatron does not exactly match. In this formula, the policy used for sampling comes from SGLang, while the one used for computing loss comes from Megatron. This mismatch makes the PPO loss an incorrect form of importance sampling.

$$\mathcal{L}_{\text{PPO}}(\theta)
= - \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim \pi_{\textcolor{red}{\text{SGLang}}}} \left[
  \sum_{t=0}^{|y|-1}
  \min \left(
    \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\textcolor{blue}{\text{Megatron}}}(y_t \mid x, y_{<t})} A_t,\,
    \text{clip}\left(
      \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\textcolor{blue}{\text{Megatron}}}(y_t \mid x, y_{<t})},\,
      1 - \epsilon,\,
      1 + \epsilon
    \right) A_t
  \right)
\right]$$

## Bypassing and Unified PPO Importance Sampling

<img src="pics/bypassing-ppo.png" alt="Bypassing and Unified PPO Importance Sampling" width="30%">

To achieve algorithmic correctness, one may directly use the rollout engine's log-probs as the old policy in offline PPO's importance sampling, rather than the recomputed log-probs from the training engine. Then it becomes the correct math form:

$$\mathcal{L}_{\text{PPO}}(\theta)
= - \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim \pi_{\textcolor{red}{\text{SGLang}}}} \left[
  \sum_{t=0}^{|y|-1}
  \min \left(
    \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\textcolor{red}{\text{SGLang}}}(y_t \mid x, y_{<t})} A_t,\,
    \text{clip}\left(
      \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\textcolor{red}{\text{SGLang}}}(y_t \mid x, y_{<t})},\,
      1 - \epsilon,\,
      1 + \epsilon
    \right) A_t
  \right)
\right]$$

In this way, the log_prob recomputation on the training engine will be skipped - it will save one forward pass computation on all the generated trajectories.

## Decoupled, 3-policy PPO Importance Sampling

<img src="pics/decoupled-ppo.png" alt="Decoupled, 3-policy PPO Importance Sampling" width="30%">

However, sometimes you may want to decouple the training-rollout mismatch from the general off-policy importance sampling. Decoupled PPO achieves batch-independent PPO by decoupling two roles: Proximal Policy (anchor policy for PPO clipping, control update size) and Behavior Policy (for off-policy correction in importance sampling). Therefore, there are 3 roles engaged in this mode: target policy $$\pi_\theta$$ , proximal policy $$\pi_{\textcolor{blue}{\text{old}}}$$, and behavior policy$$\pi_{\textcolor{red}{\text{SGLang}}}$$. $$\pi_{\textcolor{blue}{\text{old}}}$$ is recomputed with Megatron at the beginning of each training step. The total formula is below:

$$\mathcal{L}_{\text{PPO-decoupled}}(\theta)
= - \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim \pi_{\textcolor{red}{\text{SGLang}}}} \left[
  \sum_{t=0}^{|y|-1}
  \frac{\pi_{\textcolor{blue}{\text{old}}}(y_t \mid x, y_{<t})}{\pi_{\textcolor{red}{\text{SGLang}}}(y_t \mid x, y_{<t})}
  \min \left(
    \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\textcolor{blue}{\text{old}}}(y_t \mid x, y_{<t})} A_t,\,
    \mathrm{clip}\left(
      \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\textcolor{blue}{\text{old}}}(y_t \mid x, y_{<t})},\,
      1 - \epsilon,\,
      1 + \epsilon
    \right) A_t
  \right)
\right].$$

The first importance ratio $$\frac{\pi_{\text{old}}(y|x)}{\pi_{\text{SGLang}}(y|x)}$$ naturally behaves like a dynamic learning-rate scaling term. When the rollout distribution deviates from the proximal policy, the ratio shrinks the effective update (similar to trust-region control). This directly connects to the later smoothing strategy that prevents large updates induced by rollout-training mismatch.

## Batch Normalization & Bias-Variance Trade-off

While this first importance ratio already acts as a per-token adaptive learning-rate controller, the control is still stochastic at the batch level: batches sampled from “easier” regions of the behavior policy tend to amplify the effective step size, while rare or mismatched samples shrink it dramatically.

Thus, we strongly recommend enabling --tis-batch-normalize (Self-Normalized Importance Sampling) when using Sequence or Geometric levels. This technique addresses two critical issues in off-policy training: Learning Rate Stability and the Bias-Variance Trade-off.

In standard importance sampling, the average weight of each batch can vary dramatically depending on whether the sampled data were “likely” or “unlikely” under the behavior policy, which causes the effective learning rate to oscillate and destabilize training. Self-normalizing the weights so that their mean is always 1 keeps the step size consistent across updates and substantially reduces batch-to-batch variance.

Because this normalization already suppresses variance, we can relax clipping or masking thresholds and therefore reduce the bias they introduce. As the batch size grows large, self-normalization alone can make the estimator both stable and nearly unbiased, without relying on aggressive truncation.

## Masked / Rejection Importance Sampling

See here for full detail.

In addition to clipping-based importance sampling, we provide masking and rejection sampling (RS) as a stronger safeguard against training-inference mismatch. When the rollout engine assigns extremely low probability to a sampled token, the importance ratio can grow to an unsafe magnitude. Even if clipped, such cases still inject incorrect gradients into training. RS avoids this issue entirely by discarding those tokens—or the entire sequence, if necessary—when the ratio exceeds a preset trust threshold, preventing harmful updates from taking effect.
This mechanism enforces a more principled trust region: if the sampled behavior deviates too far from the proximal policy, we simply do not learn from that sample. It guarantees that all effective training data remain consistent with the assumed rollout distribution and protects the optimization from collapse in cases where mismatch becomes extreme.
Pure rejection sampling, however, may reduce the amount of usable data and increase variance, especially when mismatch is moderate. Therefore, we combine RS with importance sampling in MIS: IS maintains mathematical correction for most tokens, while RS acts as a safety valve only when discrepancies become severe. In our experiments, this hybrid approach provides stable performance and improves robustness during the late-stage mismatch surge without sacrificing learning efficiency.

## Experiments

When identifying a set of importance-sampling (IS) baselines, we encountered a requirement that does not appear in most prior RLHF or agent-training baselines: We must be able to get the log-probabilities of the model’s raw responses.

This means no post-processing is allowed on the model output, because any modification to the response string breaks the correspondence between the sampled tokens and the tokens whose log-probs we later evaluate.

Unfortunately, many existing agent baselines do rely on lightweight post-processing, often for simple tasks like trimming labels, removing prefixes, or completing partial responses. These operations are common in classic agent examples, but they invalidate log-prob evaluation for IS-correct RL.

For example:
- Search-R1 performs post-processing in response:
  [Link](https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/llm_agent/generation.py#L54)
- Retool does the same:
  [Link](https://github.com/radixark/miles/blob/main/examples/retool/generate_with_retool.py#L147)

At the moment, we have not found a solid theoretical reason why these agent tasks require such post-processing. Fortunately, removing the post-processing entirely and using the model’s raw output still yields rewards that are similar to the original baselines. We therefore adopt this simple workaround for now, though the downstream effects remain uncertain.

Additionally, due to limited engineering bandwidth, we chose to use GRPO instead of PPO to demonstrate IS behavior.

### Existence of Mismatch

We first confirm that as the training goes on, the K3 KL will increase. Our setting is:
- Training dataset: [Link](https://huggingface.co/datasets/aaabiao/dapo_filter)
- Eval dataset: aime 24 + aime 25
- Base Model: Qwen3-4b-base ([Link](https://huggingface.co/Qwen/Qwen3-4B-Base))
- Algorithm: REINFORCE (Williams et al. 1992)

<img src="pics/mismatch-existence.png" alt="Existence of Mismatch" width="50%">

You can see in the initial step of training, as the model learns and perpexity drops, mis k3 kl actually drops. But after 600 steps, although the train and eval reward remains stable, the mis K3 KL metrics start to increase dramatically, indicating the existence of training and rollout mismatch.

### IS Won't Harm Performance

See our weight&bias blog here.

In our experiments, we also verified that enabling distribution correction—including several commonly used configurations—does not degrade performance or destabilize training. To demonstrate this, we enabled different IS-related options at the beginning of training and compared them against a baseline with no IS correction.
Below are the four configurations we evaluated:

1. Baseline
2. Token-level Importance Sampling(IS)
3. Token-level IS + Masking/Rejection Sampling(RS) [a.k.a MIS]
4. Token-level IS + Masking/Rejection Sampling(RS) + Batch Normalization(BN) [a.k.a MIS]

Across all settings, we consistently observed stable training curves. All four configurations successfully reproduced the characteristic length increase after ~100 steps, indicating that enabling IS does not negatively impact the learning dynamics. Based on these results, we recommend enabling IS as a default configuration, as it provides mismatch correction without sacrificing performance.

<img src="pics/is-performance.png" alt="IS Won't Harm Performance" width="50%">

### IS Can Supress KL Increase

To test whether MIS (IS + RS + BN) works, we continue training on step 650, and the result is below. You can see that for the base run, kl continues to increase, but with MIS, the increasing trend is successfully depressed and starts to decrease.

<img src="pics/is-kl-suppression.png" alt="IS Can Supress KL Increase" width="50%">

## Usage

For more details, we provide complete guides and runnable examples:
- True On-Policy Training (FSDP): [Link](https://github.com/radixark/miles/tree/main/examples/true_on_policy)
- Algorithmic Mismatch Correction (Megatron): [Link](https://github.com/radixark/miles/tree/main/examples/train_infer_mismatch_helper)

If your goal is to fully eliminate the rollout–training mismatch, we recommend the true on-policy solution.

If you prefer to retain high performance while mitigating mismatch, algorithmic correction such as MIS is a lightweight and effective choice.

Below is a brief overview of the available options.

### True On Policy

To open true-on policy mode, add args:

```bash
CUSTOM_ARGS=(
    --true-on-policy-mode
)
```

### Algorithmic Mitigation

Please refer to [this link](https://github.com/radixark/miles/blob/main/examples/train_infer_mismatch_helper/README.md) for a long and complete explanation of each attribute.

Miles provides a comprehensive configuration system allowing users to flexibly balance Bias and Variance. To open Importance sampling, you must add the following attribute to your starting script.

```bash
CUSTOM_ARGS=(
   --use-tis
   --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)
```

Then you can adjust the detail configuration in [this link](https://github.com/THUDM/radixark/miles/main/examples/train_infer_mismatch_helper/mis.yaml).

In short, you can configure your correction strategy across four key dimensions:

#### Calculation Levels
This determines how important weights are aggregated from tokens to sequences.
- Token Level
  - Computes weights independently for each token.
  - Characteristics: Computationally simple but mathematically biased. Suitable for most general scenarios.
- Sequence Level
  - The sequence weight is the product of all token weights.
  - Characteristics: Mathematically unbiased but suffers from extreme variance. Recommended only when the mismatch is very small or the batch size is large.
- Geometric Level
  - Uses the geometric mean of all token weights as the sequence weight.
  - Characteristics: A trade-off solution. It retains sequence-level information while avoiding the numerical instability of the product method, striking a balance between bias and variance. It also provide some length-invariant property as will be for long-context tasks.

#### Rejection Sampling & Masking
To prevent extreme importance weights from destabilizing training and enforce hard trust region, we apply constraints to the weights.
- IS Mode (Importance Sampling)
  - --tis-mode: Options include clip or truncate. This forces weights to stay within the [lower_bound, upper_bound] range.
- RS Mode (Rejection Sampling)
  - --use-rs: Instead of capping weights, RS directly masks (drops) tokens or sequences that fall outside the threshold. This ensures gradient purity for valid data but reduces the effective training sample size.

The work of MIS introduced the combinations of IS and RS at different level.

#### Veto Mechanism
This acts as a low-level safety net independent of IS/RS settings.
- Mechanism: If a sequence contains any token with a probability lower than the veto threshold (e.g., $$p < 10^{-6}$$) under the old policy, the entire sequence is discarded.
- Why it's needed: It prevents "catastrophic updates." Even if clipped, a token with near-zero probability in the denominator can introduce numerical instability or destructive gradients.

#### Self-Normalization
--tis-batch-normalize: Self-Normalization. Normalizes the importance weights across the entire batch so that their mean equals 1.0. This prevents the magnitude of weights from destabilizing the training step size.

## What's More On Mismatch Solving

- Miles also supports unbiased KL estimation from Deepseek V3.2: [Link](https://github.com/THUDM/slime/pull/1004)
- Miles also supports rollout routing replay: [Link](https://github.com/THUDM/slime/pull/715)

Any mismatch solving tool can be found in Miles!

## Reference
- Liu, J., Li, Y., Fu, Y., Wang, J., Liu, Q., & Shen, Y. (2025, September). When Speed Kills Stability: Demystifying RL collapse from the training-inference mismatch. Blog [https://richardli.xyz/rl-collapse]
  - Part 1: Why Off-Policy Breaks RL — An SGA Analysis Framework [https://richardli.xyz/rl-collapse-1]
  - Part 2: Applying the SGA Framework — Token v.s. Sequence-level Correction [https://richardli.xyz/rl-collapse-2]
  - Part 3: Trust Region Optimization via Sequence Masking [https://richardli.xyz/rl-collapse-3]
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3–4), 229–256.
- He, Horace and Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference", Thinking Machines Lab: Connectionism, Sep 2025.

## Appendix

### K3 KL-Definition

Ref: [Link](http://joschu.net/blog/kl-approx.html)
K₃ KL is a simple unbiased estimator of the KL divergence based on the likelihood ratio ( $$\frac{p(x)}{q(x)}
$$ ). It is defined as:
$$ k_3(x) = \frac{p(x)}{q(x)} - 1 - \log \frac{p(x)}{q(x)}.$$

## Credit

Chenyang Zhao, Yingru Li, Yueming Yuan, Changyi Yang, Chenxin Xie, Jiajun Li, Banghua Zhu, Yuzhen Zhou, Li Ji, Tom, Jiawei Xu, Hongyu Lu
