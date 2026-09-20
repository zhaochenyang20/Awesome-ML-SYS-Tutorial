# A Batch Size Debate: Performance Validation and Scheduling Trade-offs in SGLang Omni

In [Revisiting CPU Resources as a First-Class Citizen in Speech Model Serving](./cpu-first-class-citizen.md), we discussed a rather awkward problem: throughput from the same commit can change substantially with host contention. Without controlling the test environment, an apparent optimization may simply coincide with a quieter machine. What looks like a CPU contention problem raises a more fundamental question: how do we create a rigorous experimental environment that isolates one variable at a time? If the performance of the same commit can be affected by subtle factors such as available CPU resources and contention, explicit variables such as model serving parameters deserve at least as much scrutiny.

Over the past few days, we have made some small but interesting attempts to optimize Code2Wav batching for MiniCPM-o 4.5. Put simply, we explored its batching parameters. Intuitively, if we no longer wait to form a batch, could raising the batch size limit from 1 to 8 really make performance worse? A community contributor observed exactly that on an A800, while my validation on an H200 produced the opposite result:

| Code2Wav collection policy | Batch limit | Mean throughput (requests/s) | Relative to FIFO batch size = 8 |
|---|---:|---:|---:|
| FIFO, collecting in arrival order | 1 | 4.100 | −27.5% |
| FIFO, the mainline policy at the time | 8 | **5.652** | Baseline |
| Reference-aware, collecting only requests with the same reference audio | 8 | 4.971 | −12.0% |
| Reference-aware, the configuration proposed in the PR | 4 | 4.862 | −14.0% |

These results used a single H200, the first 200 requests from the full SeedTTS EN data source, concurrency 16, and no deliberate batching wait. Each mixed-reference configuration was repeated three times. After every server startup, we ran and excluded another 50 warmup requests, with consistent CPU affinity and contention checks. The full experiment record is available [here](https://github.com/sgl-project/sglang-omni/pull/2271#issuecomment-5751596133).

Initially, I was quite confident that larger batches (`max_batch_size=8`) should help when we do not wait to form them (`max_batch_wait_ms=0`). As we investigated, more trade-offs emerged. In this short article, we want to **share how we turned a seemingly small parameter question into controlled experiments that could be tested one by one, and the lessons we drew from them**.

1. Distinguish concurrency, scheduler batches, and the computational batches inside the model to understand where the debate began.
2. Use three sets of experiments to examine the performance effects of collection policy, batch limits, and result emission timing separately.

Thanks to Shihao Zhu, Sun, Charlie, and everyone who helped with PR reviews, repeated measurements, and discussions of the mechanisms involved. This article follows the experiment records from that time. Some aspects remain incompletely explained, and further feedback is very welcome.

## The Intuitive Benefits of Batching

As is widely understood, batching organizes multiple requests into one processing operation, giving them an opportunity to share fixed overhead and computational resources. This motivation will be familiar to anyone working on LLM serving: processing requests individually requires repeated dispatch, whereas a batched model call can advance several requests at once. However, that shared processing can happen at different levels. Seeing “batch size = 8” in a configuration tells us very little about the computation actually performed by the GPU.

In SGLang Omni's [benchmarking system](https://github.com/sgl-project/sglang-omni/tree/89e60d0bf216bacd0e072d79f969550469614662/benchmarks), we need to distinguish client concurrency from batch sizes at different levels:

First comes client concurrency, which limits the number of requests in flight across the entire system. Those requests may be traveling over the network, undergoing preprocessing, waiting for upstream generation, or already reaching the final waveform generation stage. Concurrency 16 therefore does not mean that every stage always has 16 executable requests in its queue.

Once requests reach a stage, its scheduler collects a batch of ready work. This is the scheduler batch discussed in this article. Specifically, we are examining batching in the Code2Wav stage of MiniCPM-o 4.5: `max_batch_size` caps the number collected at once, while `max_batch_wait_ms` sets the deliberate waiting budget for forming a larger batch. A limit of 8 with zero wait means collecting at most 8 currently available requests. If only 2 can be collected when the previous batch finishes and the next begins, those 2 are processed without waiting for the remaining 6 slots to fill.

Even if the scheduler collects 8 requests, the model may not be able to process them in a single computation. Input conditions, sequence lengths, and the batch forms supported by the backend can require further grouping. In the code2wav implementation, for example, the scheduler's batch of up to 8 requests is grouped again by resolved reference audio identity; only requests with compatible references have an opportunity to batch within a group. One outer call may contain actual GPU batching, or it may simply wrap the execution of several small groups. The former can improve computational efficiency, while the latter can still reduce scheduler round-trips. Additional GPU batching benefits require multiple compatible requests within a group. If all 8 references differ, internal computation may still proceed one request at a time, so reduced scheduling overhead and increased computational parallelism must be examined separately.

Furthermore, `max_batch_wait_ms=0.0` removes only the collection window reserved for additional requests to join the batch. Requests may still wait in a queue for the previous batch, groups may execute serially within a batch, and completed results may wait for the entire batch to return. **Removing deliberate batching waits eliminates one cost; it does not mean that larger batches must be faster.**

## MiniCPM-o's Code2Wav Batching Policy

MiniCPM-o's batching policy makes these distinctions concrete. We tested the speech generation path of MiniCPM-o 4.5, a multi-stage scenario of the kind described in our [framework design article](./why-sglang-omni-en.md). From input understanding and content generation to audio token generation and waveform reconstruction, a request passes through stages with different computational characteristics. These changes focused on the final Code2Wav stage; the other stages were not modified.

Following the data flow, the upstream Talker produces discrete codec tokens for the target speech, and Code2Wav uses those tokens together with reference audio conditioning to generate waveforms. Flow / CFM first generates mel acoustic features, and HiFT converts those features into playable audio. The reference audio supplies conditioning such as the target voice; together with the target speech tokens generated by Talker, it forms the input to the Code2Wav module.

Originally, [PR #2254](https://github.com/sgl-project/sglang-omni/pull/2254) added batching to Code2Wav, which had previously processed requests individually. The scheduler could collect multiple requests, and the model path could handle multiple token sequences. Inputs of different lengths require padding while retaining their valid lengths for subsequent computation and output trimming. Notice also that the [HiFT implementation](https://github.com/sgl-project/sglang-omni/blob/89e60d0bf216bacd0e072d79f969550469614662/sglang_omni/models/minicpm_o/components/code2wav.py#L129-L200) groups again by valid mel length. Even with a shared reference, Flow and HiFT may therefore execute different batches.

Before PR 2254, this Code2Wav path did not support batching multiple requests. Early commits introduced batching but kept the default batch size at 1, recommending that users opt into batch size 4 with a 10 ms wait. I continued this work in [PR #2264](https://github.com/sgl-project/sglang-omni/pull/2264), cleaning up the implementation, adding tests, and enabling batching by default. This follows a principle I often emphasize:

> Once a feature has demonstrated correctness and performance benefits in a clearly defined target scenario, it should be enabled by default so users can benefit without extra configuration. Developers need to take responsibility for their implementations rather than claim improvements and leave validation to users. Whether or not AI helped write the code, we should explain the conditions under which it applies, perform the necessary regression checks, and remove superseded code paths that no longer serve a purpose. Exploratory PRs are also welcome when benefits remain unconfirmed, but they should disclose experimental conditions and uncertainty rather than present an untested idea as an established optimization.

The author of PR 2254 handled this work carefully, providing both the batching implementation and notes on performance and quality concerns. The subsequent measurements were part of our joint effort to understand its boundaries.

Beyond enabling it by default and refactoring the code style, my largest change was actually the following [configuration](https://github.com/sgl-project/sglang-omni/blob/89e60d0bf216bacd0e072d79f969550469614662/sglang_omni/models/minicpm_o/config.py#L112-L130):

```python
def code2wav_stage(*, gpu: int, process: str) -> StageConfig:
    return StageConfig(
        name="code2wav",
        process=process,
        factory_path=f"{PKG}.stages.create_code2wav_executor",
        factory=FactoryArgs(
            max_batch_size=8,
            max_batch_wait_ms=0.0,
            batch_wait_when_idle=False,
        ),
        # Note (Chenyang): As a general comment and my usual understanding
        # of SGLang Omni, SGLang Omni has a poor runtime which leads to a
        # underutilized GPU/SMs. To address this, we recommend users to set
        # batchs for your compute but never wait for grouping the batchs.
        # As SGLang Omni Runtime moves better, we shall probably wait several
        # ms for grouping the batchs, but right now, set it to 0.0.
        gpu=gpu,
        terminal=True,
    )
```

`max_batch_size` allows up to 8 requests to be collected at once, while `max_batch_wait_ms` sets the deliberate waiting budget to zero. `batch_wait_when_idle` constrains waiting behavior when idle; with a zero budget, it introduces no additional wait. Although my choice of `max_batch_size` differed slightly from the author's, `max_batch_wait_ms = 0.0` was the more important difference. The inspiration came from the discussion around [PR 2202](https://github.com/sgl-project/sglang-omni/pull/2202):

My reasoning was that if we spend additional time waiting to form a batch, ideally the GPU should have other work to do during that interval, allowing request collection and computation to overlap. For short, frequent TTS requests, if the GPU has no executable work while the CPU waits for the next request to fill a batch, that deliberate wait can delay the critical path. I therefore started with a zero waiting budget for the MiniCPM-o implementation at the time. Increasing it should depend on whether it delivers enough additional batching benefit, rather than on runtime improvements alone.

As another configuration example, Fun-CosyVoice3's [vocoder collection configuration](https://github.com/sgl-project/sglang-omni/blob/89e60d0bf216bacd0e072d79f969550469614662/sglang_omni/models/fun_cosyvoice3/config.py#L147-L160) uses a 30 ms waiting budget, after which the collected requests participate in Flow batching. It serves a different computational path from MiniCPM-o Code2Wav's zero-wait configuration. The difference alone does not establish which value is better; each waiting budget needs its own end-to-end evaluation.

Returning to Code2Wav batching, the scheduler collects requests in FIFO order with a maximum batch size of 8, but the internal vocoder does not compute directly over that incoming batch. It groups requests again by reference audio. The key loop in [`vocode_code2wav_payloads`](https://github.com/sgl-project/sglang-omni/blob/89e60d0bf216bacd0e072d79f969550469614662/sglang_omni/models/minicpm_o/stages.py#L174-L219) is:

```python
waveforms_by_index = {}
for group_indices in groups.values():
    group_waveforms = model.vocode(
        [codec_tokens[idx] for idx in group_indices],
        references[group_indices[0]],
    )
    for idx, waveform in zip(group_indices, group_waveforms, strict=True):
        waveforms_by_index[idx] = waveform
```

`groups` collects the indices of requests sharing a reference. The outer loop calls `model.vocode` for each group, passing only that group's tokens and shared reference, then stores the waveforms. The function returns the batch's results only after all groups have finished, so earlier groups wait for later ones. In our discussion, we called this wait caused by returning the whole batch together intra-batch head-of-line blocking, or HOL. Since completed waveforms could be delivered earlier, changing the emission timing offers a potential benefit.

For illustration, suppose an outer batch contains 8 requests split by reference into five groups of sizes 3, 2, 1, 1, and 1. Flow then processes five small batches. When the first group finishes, its three requests still wait for the remaining four groups.

Removing HOL has costs of its own. The H200 experiments below show end-to-end regressions from both attempts to remove it; understanding exactly which costs were incurred requires logs and profiling.

## Controlled Experiments to Evaluate HOL Removal

To address this waiting, [PR #2271](https://github.com/sgl-project/sglang-omni/pull/2271) proposed skipping incompatible references during collection, ensuring that each batch contains only one reference group. This policy was called reference-aware batching. The author also reduced the default maximum batch size from 8 to 4. Initial A800 experiments on SeedTTS-50 measured 2.406 and 2.444 requests/s for reference-aware B4 and B8. Against the historical FIFO B8 result of 1.851 requests/s listed in the PR, these represent improvements of about 30.0% and 32.0%. Against B1 in the same table, at 2.226 requests/s, the gains are about 8.1% and 9.8%. Each percentage therefore needs an explicit comparison baseline.

I was surprised by these conclusions, including the author's earlier [follow-up experiment](https://github.com/sgl-project/sglang-omni/pull/2264#issuecomment-5748101622) in PR 2264 showing batch size 8 performing worse than batch size 1, and decided to test two claims:

1. Does batch size 8 really perform worse than batch size 1?
2. Does reference-aware batching really perform better than FIFO batching?

The second trade-off is fairly intuitive: reference-aware batching must filter compatible requests and may also fragment the outer batch, increasing scheduler round-trips. Those costs must be compared with the benefit of reducing HOL. Initially, I did not have a clear explanation for batch size = 8 performing worse than batch size = 1. Given the grouping and collective emission described above, however, zero deliberate waiting does not rule out that possibility; experiments are still needed.

I also noticed that the experimental design in PR 2271 might not isolate the effects rigorously enough, so I looked for clues:

1. PR 2271 tested only 50 samples with client concurrency already at 16. Such a small workload could produce substantial measurement error.
2. As discussed above, short, frequent TTS requests are highly sensitive to available CPU cores. Unfortunately, not every developer may be aware of this, so the new validation needed to control that variable, following the CPU contention discussion in [Revisiting CPU Resources as a First-Class Citizen in Speech Model Serving](./cpu-first-class-citizen.md).
3. The experiments changed two parameters: whether reference-aware batching was enabled and the maximum batch size. We should control these separately to evaluate their individual effects.

We therefore kept the four configurations from the opening table: FIFO batch size = 1 versus FIFO batch size = 8 to examine batching benefits; FIFO batch size = 8 versus reference-aware batch size = 8 to isolate collection policy; and reference-aware batch size = 8 versus batch size = 4 to examine the limit under the new policy.

### Samples, Warmup, and CPU Resources All Define the Experiment

As noted earlier, 50 requests can quickly reveal anomalies, but a short test is more exposed to startup and drain effects. A few long sentences, the reference audio mix, and runtime fluctuations can also have a larger influence on the final result. This experiment therefore used the first 200 requests from SeedTTS EN, containing 131 distinct audio references. Each mixed-reference configuration was repeated three times, with 50 warmup requests after every server startup excluded from the measured results. Warmup reduces interference from first execution, cache initialization, and similar effects, while repeated runs reveal how much a configuration fluctuates on its own. They serve different purposes and cannot replace one another.

Beyond the GPU, we pinned `OMNI_CI_CPUSET=80-95,176-191`, corresponding to 16 physical cores and their SMT siblings on NUMA 1 of that machine. We reused the CI `ContentionSampler`, sampling every two seconds and rejecting and retrying a run if peak CPU use by foreign tasks inside the cpuset, `peak_foreign_cores`, exceeded 2.0. These controls fixed the available core set and monitored contention from external tasks. I emphasize CPU resources because our [earlier investigation of CI fluctuations](./cpu-first-class-citizen.md) showed how sensitive short computations and frequent speech-serving dispatches are to the host: if a thread gets CPU time late, the next GPU work may be dispatched late as well. Fixing the GPU model while leaving CPU resources uncontrolled can make a few percentage points of improvement hard to interpret. No run in this experiment triggered the contamination rule, but sampling every two seconds can still miss shorter interference. Passing the check only means passing the specified detection criteria.

### Removing HOL Can Also Fragment Scheduler Batches

With these controls, we obtained the opening results: FIFO batch size = 8 improved throughput by about 37.9% over batch size = 1, while reference-aware batch size = 8 fell about 12% behind FIFO batch size = 8. To understand the regression and why my conclusion differed from the author's, we looked more closely at the server logs:

| Mixed-reference configuration | Mean scheduler batch size | Mean reference groups per batch | Outer Code2Wav calls recorded in logs |
|---|---:|---:|---:|
| FIFO batch size = 8 | 7.13 | 4.93 | About 30 |
| Reference-aware batch size = 8 | 1.58 | 1.00 | About 137 |
| Reference-aware batch size = 4 | 1.46 | 1.00 | About 148 |

These call counts follow the [published server-log accounting](https://github.com/sgl-project/sglang-omni/pull/2271#issuecomment-5751596133) and help compare mechanisms across configurations. They are not GPU kernel counts, nor should they be assumed to cover exactly the 200 measured requests and used to reconstruct the mean batch size.

With 131 references among 200 requests, opportunities for several requests sharing a reference to be ready in the same collection window are limited. Reference-aware collection guarantees groups=1 but often collects only one or two requests, leaving most of the batch size = 8 capacity unused. FIFO keeps several reference groups within one outer call. It still computes them group by group, but reduces repeated exits from the compute path, collection of the next batch, dispatch, and return.

The outer call count rises from about 30 to about 137, roughly 4.6 times as many. That does not mean total overhead increases to 4.6 times its original value, but it identifies a concrete cost exchanged for the removed intra-batch HOL wait. The logs support the interpretation that, for this H200 workload, FIFO's amortization of outer overhead outweighed the benefit of reducing HOL by splitting scheduler batches. Finer profiling is still needed to attribute the difference to Python, communication, synchronization, or GPU execution.

To further examine the reference mix, we used `--no-ref-audio` as a control sharing the default reference. At batch size = 8, FIFO and reference-aware achieved 7.122 and 7.069 requests/s, only about 0.7% apart. The policies converging after removing reference differences supports this explanation. However, the control also changes the input and reference-conditioning work, and it has limited repetitions, so it cannot exclude every other influence. See the [control results](https://github.com/sgl-project/sglang-omni/pull/2271#issuecomment-5751596133).

## Testing the Batch Limit Separately

Since reference-aware collection rarely forms large batches in this workload, its similar performance at batch size = 4 and batch size = 8 is unsurprising. But that does not establish whether FIFO's limit should drop from 8 to 4, so we kept mainline FIFO and ran a separate batch-size experiment.

This experiment again used an H200, concurrency 16, zero wait, CPU pinning, and contention checks every two seconds, but moved to GPU 0 and `OMNI_CI_CPUSET=48-63,144-159`, with 16 warmup requests per run. Because the GPU, CPU set, and warmup count differed from the previous experiment, the two sets of results must be interpreted separately rather than pooled as repetitions of one experiment.

On three repetitions of official SeedTTS-50, FIFO batch size = 1 achieved 3.689, 3.811, and 3.988 requests/s, while batch size = 8 achieved 4.406, 4.816, and 4.819. The respective means were 3.829 and 4.680, a gain of about 22% for batch size = 8. These short tests also illustrate variability: the highest and lowest results differed by about 8.1% at batch size = 1 and 9.4% at batch size = 8, using the minimum as the denominator. One run is therefore far from enough to judge a change of just a few percent.

We then tested prefixes of different lengths from the full EN data source, obtaining the following trend:

| Requests | Unique references | FIFO batch size = 1 | FIFO batch size = 4 | FIFO batch size = 8 | batch size = 8 relative to batch size = 1 |
|---|---:|---:|---:|---:|---:|
| 50 | 32 | 4.100 | 4.439 | 5.138 | +25% |
| 100 | 64 | 4.056 | 4.823 | 5.391 | +33% |
| 200 | 131 | 4.109 | 5.174 | 5.684 | +38% |
| 400 | 253 | 4.072 | 5.298 | 5.833 | +43% |

Throughput is in requests/s. **Each configuration in this table has only one clean run and is used to observe trends; the first 50 requests from full EN and official SeedTTS-50 are different workloads.** The independent experiment record is available [here](https://github.com/sgl-project/sglang-omni/pull/2271#issuecomment-5751642645).

These results support keeping batch size = 8 under the tested H200 conditions, but they do not establish it as a universal optimum. Longer prefixes change run duration, reference composition, sentence lengths, and the proportion of time spent draining the workload. We cannot attribute the increase from 25% to 43% entirely to having more samples. Nor did these experiments exhaust larger batch sizes or search nonzero waiting times.

As for the different A800 results, the hardware, host environment, and experimental procedure have not been fully aligned. We cannot attribute the discrepancy to GPU architecture simply because the cards differ. A better next step is to repeat the same inputs, code, and procedure across machines, then examine how individual costs change. The H200 results provide a comparison to explain; they do not invalidate the value of the A800 observation.

## Earlier Results Change When Subsequent Requests Arrive

FIFO has lower outer overhead, but returning the whole batch together still introduces waiting. I therefore tried a compromise: preserve FIFO collection and the computations within each group, but return each reference group's results as soon as it finishes, directly reducing the wait for completed results. This early-emit policy does not filter references during collection and appeared capable of retaining much of the existing benefit while allowing completed requests to leave sooner.

We used a switch on the same experimental code to select wait-all or early-emit, fixing the H200, first 200 requests, concurrency 16, zero wait, and CPU set, and excluding 50 warmup requests after every startup. Each mixed-reference arm was repeated three times, and all passed the CPU contamination check. The results were:

| Emission policy, both FIFO batch size = 8 | Mean throughput (requests/s) | Mean latency (s) | Mean reference groups | GPU utilization |
|---|---:|---:|---:|---:|
| Wait-all, return after the whole batch finishes | **5.730** | **2.739** | 4.96 | About 74% |
| Early-emit, return each completed group | 4.729 | 3.291 | 5.08 | About 89% |

Early emission reduced throughput by about 17.5% and increased mean latency by about 20.1%. The collected batch size remained around 7, and the number of reference groups was also broadly unchanged. See the setup and full results [here](https://github.com/sgl-project/sglang-omni/pull/2271#issuecomment-5751688038).

Here is the explanation I could think of:

In this MiniCPM-o test configuration, Thinker, Talker, and Code2Wav share one GPU, and the client uses a closed loop with a concurrency limit of 16. Each completed request releases its concurrency slot and lets the client send a replacement. The emission policy therefore affects when subsequent requests enter the server.

Returning to the example of 8 requests split into groups A through E, wait-all keeps Group A waiting after it finishes until the entire batch completes. Those 8 concurrency slots stay occupied until all results are emitted together, so their replacements are delayed:

```text
time →
Group A ████
Group B      ████
Group C          ████
Group D              ████
Group E                  ████
          └──────── emit all 8 ────────┘
```

Other in-flight requests may still execute upstream work or complete and trigger replacements, however, so this does not imply that Code2Wav has exclusive use of the GPU.

Early-emit returns Group A's 3 results as soon as it finishes, allowing their replacements to be sent sooner. After network transit and preprocessing, the new requests' Thinker / Talker work may compete with the remaining Code2Wav computation and stretch later groups:

```text
time →
Group A ████  → emit A now
Group B      ██████           ← stretched by new prefill
new thinker      ████████
Group C            ██████
Group D                  ██████
Group E                        ██████
```

This explanation is consistent with the observations, but the data has not established whether new requests' prefill is the main source of interference, much less whether it accounts for the entire 17.5% loss. Overhead in the return path itself and the precise overlap among Thinker, Talker, and Code2Wav still require GPU timeline evidence. In the single-run shared-reference control, wait-all and early-emit achieved 7.087 and 6.994 requests/s, about 1.3% apart. That is also consistent with there being little room for earlier emission when there is only one group, but it does not replace a complete causal investigation.

Another observation deserves attention: early-emit increased GPU utilization while completing fewer requests. Spending more time busy does not necessarily mean completing user work more efficiently. When stages share resources, the ultimate measures remain end-to-end throughput, latency, and quality; higher utilization alone does not establish a successful optimization.

After two days of investigation, our chosen configuration remained unchanged: FIFO on the H200, batch size = 8, and maximum batch wait time = 0 😂. On the surface, we had made no progress:

| Policy | Intended benefit | Costs to weigh | Result in this H200 experiment |
|---|---|---|---|
| FIFO + wait-all | Amortize outer collection and dispatch overhead | Earlier groups wait for later groups | Retained as default |
| Reference-aware | Reduce intra-batch waiting from mixed references | Sparse compatible requests shrink batches and increase scheduler round-trips | Throughput regression |
| FIFO + early-emit | Preserve collection while returning completed results sooner | Return overhead and changed refill timing may affect shared-resource contention | Throughput and latency regression |

## How We Want to Validate an Optimization

These experiments support an engineering decision: retain FIFO batch size = 8 with no deliberate waiting under the tested H200 conditions. They also leave work to do, including explaining the A800/H200 discrepancy and identifying the specific resource contention under early-emit. To me, rigorous development requires understanding a request's complete lifecycle across stages and designing controlled experiments around it.

First, we should report how much faster something became alongside how much it already fluctuates. Reference-aware batch size = 4 and batch size = 8 differed by only about 2.2% in the policy experiment, while recorded run-to-run throughput variation was about 3%–7%. We do not have enough evidence to call that a stable difference. The 12%–14% policy regressions and 17.5% early-emit regression warrant more attention, but exceeding the observed variation does not automatically constitute a statistical significance test.

To draw conclusions about smaller gains, we need more independent repetitions, interleaved or randomized A/B execution order, and per-run differences with confidence intervals. Hundreds of requests within one run share queueing, resource contention, and environmental changes; they cannot simply be treated as hundreds of independent experiments. More samples and more experimental repetitions both help, but provide different evidence.

Quality validation must not become less demanding because the performance numbers look good. PR 2271 observed two short-sentence WER outliers at batch size = 8. WER divides the number of word substitutions, deletions, and insertions by the reference word count. Short sentences have small denominators, so a few errors can produce a large percentage change. For example, 3 word errors in a 4-word sentence produce a WER of 75%. Whenever we observe potential performance benefits, we must check whether correctness is affected.

My first reaction in the group discussion was also to interpret these outliers as random error. For a complete experimental conclusion, however, we need to revisit the anomalous samples, compare generated audio, ASR transcripts, and text normalization in paired tests, and repeat runs to see whether the issue reproduces consistently. Batching involves padding, valid lengths, and the mapping between inputs and outputs. All deserve inspection, and performance results cannot replace quality gates.

Even with a correct implementation, changing batch shapes can change kernel selection or floating-point reduction order, producing slightly different numerical results for the same input. [PyTorch's numerical accuracy notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html#batched-computations-or-slice-computations) specifically discuss differences between batched and individual computations. **Batch invariance means that the result for the same input does not change with batch size or composition; the concern here is the absence of that invariance.** Such differences are not limited to neural encoders and may also occur in decoders, Flow, or vocoders. Numerical differences do not necessarily imply lower quality, however, and cannot directly explain these WER outliers; the specific paths and samples still require validation. For related discussion of numerical differences in inference, see our team's article [The Roots of RL Training–Inference Mismatch Are Widespread in Inference Systems](https://www.linkedin.com/pulse/rl-%E8%AE%AD%E6%8E%A8%E4%B8%8D%E4%B8%80%E8%87%B4%E7%9A%84%E6%A0%B9%E6%BA%90%E5%85%B6%E5%AE%9E%E5%9C%A8-inference-%E7%B3%BB%E7%BB%9F%E4%B8%AD%E5%B9%BF%E6%B3%9B%E5%AD%98%E5%9C%A8-xinyu-lu-zbxbc/).

These expectations should become part of everyday development. A PR should tell reviewers what changed, which baseline it uses, its inputs and environment, and how warmup and contaminated runs were handled. Logs should let reviewers trace actual batch sizes, grouping, and emission behavior. Just as our [earlier TTS refactor](./tts-refactor.md) brought repeated lifecycle management into the framework, we want reliable measurement methods to become part of the benchmark and CI infrastructure, so the next contributor can reuse them instead of rediscovering which conditions matter.

We still want SGLang Omni developers to have confidence in validated optimizations, making parameters that suit the target scenario the defaults and reducing the burden on users to tune everything themselves. That confidence comes from reproducible evidence and a willingness to publish unexpected results. Reference-aware and early-emit did not win in these experiments, but they helped us understand scheduling costs and feedback. A community contribution's value extends beyond whether it was ultimately merged.

If this kind of work interests you, you are very welcome to contribute to [SGLang Omni](https://github.com/sgl-project/sglang-omni). You could start by reproducing a comparison across hardware, collecting a more complete GPU timeline, studying how reference distributions affect batches, or opening an issue with timestamps and reproduction steps to test whether client refill timing causes actual batch sizes to repeat periodically. We need people who know kernels, and people willing to carefully examine benchmarks, anomalous samples, and scheduling details.

We hope to build an inference framework whose optimizations can be explained both when they work and when changing conditions make them fail. Following an unexpected result until it can be explained and reproduced, then putting that understanding into code and tests, is substantial systems work in its own right. It is also the kind of engineering practice I hope more of us will undertake together in SGLang Omni.
