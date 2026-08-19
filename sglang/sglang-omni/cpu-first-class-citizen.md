# Revisiting CPU Resources as a First-Class Citizen in Speech Model Serving

Ever since the SGLang project started back in 2023, we have invested heavily in long context inference. In the beginning, the long context we talked about was something like Llama 3's 8K, or the 16K and 32K context lengths of early long-text approaches. Then, in January 2025, the arrival of DeepSeek R1 pushed both context length and the real demand for context to a new height. And this year, the explosion of coding agents drove the demand for context length up even further; in an earlier podcast from our team, we also discussed terrifying models with 1M context windows such as K3 and DeepSeek V4.

PS: You are welcome to listen to the podcasts [《详解 Kimi K3：强到冲击 Anthropic 估值的模型什么样？》](https://mp.weixin.qq.com/s/KydWDORAkByannmR9jt5ZQ) and [《详解DeepSeekV4：Infra巨鲸、百万上下文走进现实、极致效率优化》](https://www.xiaoyuzhoufm.com/episode/69f2e8ef0694c843e7cd91b6?s=eyJ1IjogIjYyNDkxYjY4ZWRjZTY3MTA0YTk0MzljNSJ9)


There are many methods we use for long context inference, which we will not expand on here, but everyone is welcome to read our related blog posts. Today's topic, however, has nothing to do with long context; instead, we want to share some cognitive blind spots that came from the nature of our long-term work. Specifically, the vast majority of our time goes into optimizing GPU efficiency, especially under long context, where we have to handle terrifying long prefill and long decoding. This led us to carry a preconceived optimization objective into inference scenarios like ASR/TTS with short requests and high request rates, madly optimizing the GPU Runtime while overlooking the CPU compute resources that are equally critical during inference. On that basis, this article reviews the CPU bottleneck the SGLang Omni project team discovered while running CI, along with the corresponding experimental observations we made and the optimization results we ended up with, in the hope of prompting everyone to take another look at CPU resources.

## The CPU Bottleneck Found While Running CI

At the very start of the [V1 refactor](https://github.com/sgl-project/sglang-omni/issues/188) of the SGLang Omni project, we established an extremely strict CI policy. Put simply, our CI requires that both performance and correctness only move forward, never backward. To give a concrete example, suppose Qwen3 ASR scores 108 requests per second on the Seed TTS reverse-transcription task at commit A; then every commit after commit A may only push that score higher, no regression is permitted, and correctness must not degrade at all. This sounds entirely trivial, but how do we get our CI to catch every possible regression? Let me give an example: suppose at commit A, Qwen3 ASR scores 108 requests per second, then at commit B it reaches 130 requests per second, and then the next commit C slightly lowers performance, bringing Qwen3 ASR back to the level of 120 requests per second. If we do not raise the CI threshold to 130 right at commit B, and instead keep the threshold of 108 from commit A out of inertia, then the performance regression that occurred between commit B and commit C is very hard for us to detect automatically. To handle this, our approach is actually very simple: right after a commit produces a performance improvement, we immediately run 5 repetitions and take the minimum of those 5 repetitions as the updated CI threshold. We call this process calibration. So, returning to the A B C example I gave earlier, as long as calibration at commit B raises the threshold to 130, we are guaranteed to catch the regression that arises between B and C. This process sounds very simple, but in practice requiring calibration for every PR that improves performance is an extremely painful thing for developers. That said, our consistent view has always been that we would rather make life hard for developers than deny our users the best possible framework. As for how calibration is actually performed, you can refer to the calibration skill in the sglang-omni repo at `.claude/skills/tune-ci-thresholds`.

Given this CI setup of ours, our CI can basically pick up precise regressions on the major models for the tasks we care about with great sensitivity. Conversely, if a regression does appear, it immediately puts our team on alert and we investigate the cause. There is a very interesting example here: two months ago, when we migrated CI from H20 to H100, a larger-than-expected regression occurred (see [Issue 907](https://github.com/sgl-project/sglang-omni/issues/907)). You could say that our almost pathological obsession with CI is what gives us the most direct observability into performance.

Likewise, our discovery of the CPU problem also came from a CI regression. Specifically, after [PR 1183](https://github.com/sgl-project/sglang-omni/pull/1183) updated the SGL version, we ran a full round of calibration on CI through [PR 1260](https://github.com/sgl-project/sglang-omni/pull/1260), and Fun-ASR's speed gate floor went up from 115.3 to 128.2 req/s, while the five repetitions in that calibration round landed in the 128 to 139 req/s band. But over the next few days, CI began failing the throughput bar on a broad scale, even though those PRs appeared to have nothing to do with ASR itself. Even more extreme, for the same commit, one run only produced about 92 req/s. We then re-ran that PR's CI 12 times in a row, and the worst round came in at only 36 req/s. Code dependencies, GPU, and load-testing scripts were all identical, yet throughput differed by nearly 4× across rounds.

Our standard for ourselves has always been this: any anomalous phenomenon either means our understanding is insufficient, or a strange bug has appeared. If we do not resolve it now, it is certain to cause a bigger problem later. After observing such an extreme throughput gap between different CI runs, we resolved to investigate the cause.

1. the calibration mechanism itself is flawed;
2. flashinfer or some other dependency was silently upgraded, causing a performance regression;
3. some PR made Runtime performance extremely unstable, greatly amplifying performance variance;

Given our finding in [Issue 907](https://github.com/sgl-project/sglang-omni/issues/907) about the host-bound nature of the workload, we suspected these problems might be related to CPU resources, so we ran some simple controlled experiments. Keeping every other condition identical, we varied only the CPU load on this machine. The result was very interesting: as the CPU load from handling other processes increased, the throughput of the ASR process kept dropping, and at the heaviest load only about 27 req/s remained, with GPU utilization down to single digits. So we went back to check the machine and found that the five calibration rounds of PR 1260 happened to fall entirely inside CI's idle window, when only 0 to 7 jobs were starting per hour; from the next day onward, that same machine was starting 13 to 27 jobs per hour.

> note: Note that in order to keep the calibration and CI environments consistent, we actually perform calibration on the very machine that will ultimately run CI. In other words, calibration and CI are really two sides of the same coin. Calibration's only job is to faithfully execute CI 5 times and take the worst performance and correctness result as the threshold. As for CI itself, it is in fact also measuring end to end the performance and correctness of the things we care about. CI is likewise just faithfully running our Benchmark evaluation system; in other words, we have built a three-tier pipeline of Benchmark evaluation, CI test, and calibration, and the components the three can share are numerous.

The final conclusion is that the root cause of the severe ASR CI decline lies in fluctuations of host CPU resources: GPU memory has a budgeting mechanism such as [`mem_fraction_static`](../kvcache-code-walk-through/mem-fraction-static-en.md), and GPU compute gets allocated automatically by CI's scheduling, but the CPU has no management mechanism whatsoever. At the same time we found that flashinfer JIT compilation imposes an enormous CPU burden. Under the old flow, a CI cold start would trigger a whole-machine recompilation; we measured this during the validation of [PR 1343](https://github.com/sgl-project/sglang-omni/pull/1343), and one such compilation can knock a co-located ASR service from 122 qps down to 41. To be clear about the mechanism, the CI image at the time did ship a set of prewarmed flashinfer objects, but it did not cover the cutlass MOE kernel family, and `FLASHINFER_WORKSPACE_BASE` pointed inside the job container, so the compiled result was destroyed along with the container, which meant every per-PR venv had to recompile for 30 to 60 minutes. The eventual fix was to bake the compiled artifacts fully into the CI image and make CI actually use them: the new image prewarms everything including the cutlass MOE family, and [PR 1343](https://github.com/sgl-project/sglang-omni/pull/1343) aligned the CI environment with the image's Python 3.12 (CI used to create a separate Python 3.11 venv inside the job, and the version mismatch invalidated the prewarmed objects wholesale), reusing the image's Torch and FlashInfer directly; fused_moe compile time dropped from 481 s to 1.4 s. We had also proposed persisting the JIT workspace in the per-PR CI home as an alternative ([PR 1297](https://github.com/sgl-project/sglang-omni/pull/1297)); once the image-based fix landed, it was not pursued further.

### The CPU Cost of Speech Model Serving

To describe this problem further, we simplify the inference process of speech models. After a request enters SGLang Omni's serving stack, the CPU host side decomposes the request and schedules it across the [multiple stages we designed](./why-sglang-omni-en.md). Most of these stages are GPU compute; the host/CPU repeatedly dispatches compute tasks to the GPU and, once results come back, immediately relays them to the host to prepare the next step. Every request is very short while concurrency is very high, and a single-step kernel is often only a few microseconds, so a large amount of time goes into inter-stage scheduling and request construction.

As we mentioned at the beginning of this article, unlike the long decoding and long prefill workloads that LLMs depend on, speech models themselves depend more strongly on the CPU. [#907](https://github.com/sgl-project/sglang-omni/issues/907) ran a non-rigorous profiling pass on H100, in which the GPU was idle 94.3% of the time (the gap between two adjacent kernels is roughly 17× the duration of the kernel itself); cutting the server's host CPU to about one quarter dropped throughput by about 70%, while dropping the SM clock to 0.455× cost only 10% of throughput. (PS: our runtime fusion is of course better now than it was at the time of Issue 907, so such extreme idling naturally no longer exists)

<div align="center"><img src="images/cpu-blog-gpu-waiting-meme.png" width="640"/></div>

### Why Does the CPU Matter So Much?

The earlier experiments only told us that throughput drops as CPU load increases, but did not answer why. In fact, Linux by default allows any process to run on any CPU core; once there are many tasks on a machine, the scheduler places threads from different processes onto the same set of cores, where they compete for execution units, for cache, and for the power budget, and that is contention. It does not require a fully loaded machine to happen: as soon as two CPU-hungry processes happen to land on the same physical core, they start slowing each other down. The earlier example of throughput falling from 139 to 27 is contention showing up at the macro level.

<div align="center"><img src="images/cpu-blog-three-cuts.svg" width="780"/></div>

To see its microscopic mechanism clearly, we ran two independent sets of experiments to analyze two questions: which dimension of CPU resource does this service actually depend on, and through what pathway does contention cause harm?

### How CPU Resource Contention Happens

CPU resources have at least three dimensions: the number of cores, the CPU time usable per second (managed via cgroup quota), and how fast each cycle is (frequency). A very natural reaction might be that if there is not enough CPU, we should add cores, which obviously only considers the most obvious dimension of CPU resources. For the Higgs TTS pipeline, we limited one resource at a time and observed the effect on throughput. Restricting cores from 32 down to 2 barely moved throughput; restricting CPU time to 25% via cgroup quota left only 16% of throughput (recorded in [Issue 921](https://github.com/sgl-project/sglang-omni/issues/921)). As a GPU-side control, halving the GPU frequency cost only 20% of throughput; as for the CPU frequency dimension, we did not measure it in isolation.

A server process easily has several hundred threads, which looks quite terrifying, but in our scenario those several hundred threads together occupy only about one core, and these threads may be serial, so adding cores can hardly improve processing efficiency linearly. There are two intuitive solutions. The first is to turn a single chain into multiple chains, genuinely using more cores through multiple processes and multiple replicas, which is the path taken by same-GPU DP + MPS and the multi-process router. Second, try to protect the server process's CPU time per second from being preempted by other tasks on the machine, which is the thinking behind the core pinning and allocator that follow. And note that if we adopt the first solution aggressively, then the more replicas and the more processes there are, the more severe the preemption among them on the same core becomes.

<div align="center"><img src="images/cpu-blog-one-core-digging-meme.png" width="640"/></div>

### Contention Makes Each Request Cost More CPU

The first experiment proved that contention hurts the performance of all conflicting processes; we then thought further about why this effect occurs:

One possibility is that there are not enough CPU cores to go around and threads frequently cannot get a core; if that were the case, the number of CPU milliseconds each request needs would stay the same and only the waiting time would grow, and waiting is recorded by PSI (PSI is a Linux kernel pressure metric that measures how much time tasks are forced to wait because they cannot get a CPU). Another possibility is that threads always have a core available, but the SMT sibling on the same physical core is occupied by someone else, and combined with all-core frequency being suppressed, fewer instructions actually make progress per cycle. Less work gets done per millisecond, so the number of core-milliseconds needed to finish the same request goes up. Distinguishing them is simple: look at how PSI and per-request CPU milliseconds each change. Queueing can be alleviated by adding capacity; the latter requires isolating a fixed amount of CPU resource.

We ran an experiment in the same way, comparing Fun-ASR under the same request load: once with exclusive ownership of its own core region, and once sharing the core region with a group of interference processes saturating the CPU. The results show that in the shared-core-region arm, PSI stayed below 0.01 throughout, meaning almost no task was waiting for a core; but the core-milliseconds per request went from 51 to 52 up to 72 to 83, so the same work costs about 1.5× the CPU time, and throughput also went from about 82 down to 48 to 58 req/s. In other words, pinning plus isolation retains about 92% of the quiet baseline throughput, while not pinning retains only 55% to 62%. The relevant data is recorded in [Issue 1296](https://github.com/sgl-project/sglang-omni/issues/1296) and [Issue 1308](https://github.com/sgl-project/sglang-omni/issues/1308).

Contention makes less work possible per millisecond rather than making tasks queue, so adding capacity cannot solve contention. We therefore chose to use cpuset to grant whole physical cores (together with both of their SMT siblings) exclusively to the service; cgroup quota can only cap the total amount and cannot stop two processes from crowding onto the same physical core.

## Fixing CPU Resource Contention on CI

We did the following things to fix CPU resource contention on CI:

1. pin the performance-test processes to the cpuset reserved for each GPU lane ([PR 1321](https://github.com/sgl-project/sglang-omni/pull/1321) introduces `OMNI_CI_CPUSET`, and [PR 1388](https://github.com/sgl-project/sglang-omni/pull/1388) propagates it into the ASR/TTS/Qwen3-Omni CI containers);
2. detect CPU contention, sampling foreign CPU occupancy on the core region every round during calibration, and when occupancy is detected, treat that measurement as contaminated and re-run it ([PR 1415](https://github.com/sgl-project/sglang-omni/pull/1415), with the follow-up [PR 1423](https://github.com/sgl-project/sglang-omni/pull/1423) switching to per-CPU counters to make occupancy accounting more accurate);
3. pin unit tests, environment-preparation scripts, and the runner process tree into their respective core regions, and make calibration reuse exactly the same pinned CPU condition as CI ([PR 1405](https://github.com/sgl-project/sglang-omni/pull/1405), [PR 1417](https://github.com/sgl-project/sglang-omni/pull/1417));
4. adjust the machine usage policy. Previously this H100 machine reserved two cards for development, while the other six cards ran CI in three groups, and development tasks and CI tasks shared the same set of CPU core regions, which is itself a major source of contention; in the end we converted the whole machine to CI-only and moved development work to other machines.

Through these measures, we successfully added a CPU budget to the CI machine. Put simply, before our changes, the CI machine would launch a new CI job as long as it detected sufficient GPU resources. After the changes, we brought CPU resources into the CI machine's consideration as well. When insufficient CPU resources are detected, no new CI job is launched even if GPU resources are sufficient.

Even so, considering that CPU contention can still occur in production environments, we want to build some CPU contention detectors into SGLang Omni ahead of time, at the very least to signal to users that CPU contention is happening, so that users can reasonably adjust the CPU workload to improve overall SGLang Omni throughput efficiency.

## From Manual Core Pinning to a CPU Allocator

The CI fix relies on manually carving out a CPU core region for each CI runner, an approach that works in a fixed environment like CI. Next we wanted to organize CPU isolation into a general mechanism and validate its effect in more contention scenarios.

### Reusing the Optimization

In the previous section, our fix for CI relied on manually assigning and carving out the CPU core region for each CI runner. Naturally, when the environment is fixed and the variety of tasks is limited, we can draw the core regions in advance, but development scenarios can be far more complex. Let us start with this H100 mixed development/CI machine: suppose there are serving tasks running on the cluster, and at the same time there are tens of thousands of Python tasks, 97% of which have no CPU affinity (CPU core binding). Linux by default allows a process to use any core on the machine, so these tasks can perfectly well occupy the cores the serving process is using.

Manually managing the CPU core regions of tens of thousands of processes is unrealistic, for three reasons. First, methods like `taskset` can only constrain the current process, and other processes on the same machine can still run into the current process's core region, so the isolation is one-sided. Second, server topology is not as simple as core numbering; a single physical core usually has two hyperthreads, and as soon as a pair of siblings is handed to two different processes, contention is manufactured. Third, the topology of the serving process itself is highly flexible and varied: the [DP + MPS approach](https://sgl-project.github.io/sglang-omni/basic_usage/mps_dp.html), as well as the later [process-level replicas](https://github.com/sgl-project/sglang-omni/issues/1307), both require fine-grained core division per stage and per replica, and the number and ownership of cores change with configuration, making full manual configuration very difficult. A manual scheme cannot cover the full real serving topology, nor can it control the other tasks on the same machine.

We ran another non-rigorous comparison experiment: two DP replicas of Qwen3-ASR on the same GPU, with an additional group of CPU-saturating interference tasks on the machine, comparing three approaches:

| condition | Qwen3-ASR qps (two rounds) |
|---|---|
| default, today's status quo, no cpu resource management at all | 107 / 115 |
| lock fixed CPU cores for the two DP2 replicas, without restricting other tasks | 110 to 128 |
| lock fixed CPU cores for the two DP2 replicas, and also restrict other tasks | 280 / 278 |

Constraining only our own process yields a gain of just 10% to 15%; constraining the co-located load as well gives 2.5× the throughput. This shows that to obtain the benefit of isolation, there must be a plan that covers the whole machine, enforced uniformly by a serving flag. To close this gap, we built a topology-aware CPU allocator ([PR 1463](https://github.com/sgl-project/sglang-omni/pull/1463), still open at the time of writing). Its positioning should be stated up front: what actually moves throughput is "confining co-located load into a bounded set of cores", not "letting the serial loop hold cores exclusively". Comparing the plan generated by the allocator against hand-written CORE_BLOCKS, five models under DP measure only 0.92 to 1.02×, which is a wash. So what it buys is an automatic, correct placement plan covering the entire process tree plus an interface that makes contention visible, rather than extra throughput on top of masks someone already tuned by hand.

Specifically, host overhead varies a great deal across models. Profiling shows that Fun-ASR pays about 45 ms of host orchestration per request, which breaks down into the pre-LM encoder service thread at 45%, the scheduler loop at 28%, request construction at 17%, and the router at 9%, while the actual fbank audio preprocessing is only 3.2 ms (for Qwen3-ASR this cost is about 35 ms); the higher the share of host processing time, the more sensitive the model is to contention. At saturation for Fun-ASR, ASR's host processing uses about 4.2 cores (pre-LM encoder + scheduler), which is rounded up in the configuration to a final declaration of 5 exclusive physical cores. The allocator is responsible for reading the declaration, pairing physical cores with hyperthreads from sysfs, and isolating those 5 physical cores together with both hyperthreads on each core so that ASR owns them exclusively, while the remaining processes in the serve process tree go into the shared pool.

<div align="center"><img src="images/cpu-blog-allocator-flow.svg" width="780"/></div>

### Validating the Effect Under Contention

More experiments prove that the CPU allocator is highly effective when contention is present. PR #1463 ran on 1× H200 with one 16-physical-core lane per model, alternating the allocator on and off for several rounds each (A-B-A-B) to cancel out drift in machine state over time. Every value in the table represents the ratio of throughput with the allocator on versus off, at a given load level.

| model | no load | moderate load | heavy load | heavy-load retention |
|---|---|---|---|---|
| Fun-ASR | 1.00× | 1.04× | 3.54× | 98% |
| Qwen3-ASR | 0.99× | 1.00× | 2.50× | 98% |
| Higgs TTS | 0.95× (1.01× on an isolated re-run) | 1.22× | 1.63× | 92% to 97% |
| MOSS-TTS-Local | 1.01× | 1.10× | 1.44× | ~100% |
| Fish S2-Pro | 1.00× | 1.15× | 2.18× | 98% |
| dots.tts | 0.99× | 1.11× | 1.88× | 96% |


Take Fun-ASR as an example: the heavy-load column reads 3.54×, meaning that when contention is severe, the throughput obtained with the allocator on is 3.54× the throughput without it. The last column shows the throughput ratio under heavy load with the allocator on relative to the no-load case. For Fun-ASR it is 98%, which means that no matter how fierce the contention, turning on the allocator protects throughput performance almost perfectly, whereas leaving it off degrades throughput significantly.

## Bringing the CPU Isolation Mechanism into Production

Since isolation is so effective under contention, we naturally expected a decent gain on the production side too, so we measured the CPU isolation mechanism in a production configuration: same-GPU DP + MPS, no external CPU load on the machine, with the only variable being whether the allocator is on. Unfortunately, our gain was essentially zero.

- scenario: single-node DP with MPS enabled
- external CPU load: none
- concurrency: 3, 6, 12, 24, 48, 96
- only variable: allocator off (baseline) vs on (static)
- repetitions: 3 per group

| concurrency | baseline req/s | static req/s | paired change |
|---|---|---|---|
| 3 | 36.398 | 36.809 | +1.26% |
| 6 | 69.604 | 69.337 | -0.39% |
| 12 | 118.453 | 116.789 | -1.42% |
| 24 | 177.795 | 180.382 | +1.49% |
| 48 | 200.867 | 211.556 | +5.34% |
| 96 | 100.997 | 97.025 | -3.68% |

The unweighted average over 18 paired points is +0.43%. The change at each level falls between -3.68% and +5.34%, with no stable positive trend. A design that reached 3.54× under heavy load earlier yields essentially zero gain in the production configuration.

### What Is Different About Production

It is actually easy for everyone to see why the gain from CPU isolation drops to essentially zero on the production side. The reason is simple: CPU resources are not a scarce resource in production. A real production deployment usually has a dedicated machine to itself, with no other tasks on it. On the development / CI shared machine we mentioned, contention is the norm. But on the production side, we found contention to be genuinely rare.

In the first set of experiments, there was an additional group of CPU-saturating interference tasks on the machine, and the gain came mainly from keeping external contention outside the serving process tree. In the second set of experiments, there was no external CPU load on the machine (which in fact reverts to the no-load case of the first experiment), and the only remaining variable is whether the static allocator inside the server is off or on, that is, how cores are divided among stages. With no external contention, this variable barely affects throughput. So the conclusion is clear: the benefit of isolation comes from restricting external processes' CPU usage, not from how cores are divided among stages inside serving.

The same hand that made it also unmade it. Both the hypothesis about the CPU and the allocator itself are effective, and on a machine with severe contention the gain is significant. But the task workload on a CI machine is more complex, with dozens of jobs per hour and whole-machine recompilations happening readily, so contention occurs frequently. Production is exactly the opposite: there are only a few services on a machine, each model's host processing uses at most two to five cores, and even if we deploy aggressively in parallel with the MPS DP approach, we would only consume 20-30 cores, while a server CPU has hundreds of cores. CPU is therefore no longer a scarce resource, and protection designed for a high-contention environment naturally shows no measurable gain when moved into an environment with no contention.

Even more unfortunately, the static allocator is also hard to make a stable production solution:

1. after warmup completes, the number of CPU-sensitive processes is limited and so are the cores they occupy, so sustained contention is unlikely to form inside the server. When available cores are plentiful, OS scheduling is already good enough.
2. when available cores are plentiful but the partitioning strategy is poor, performance can actually degrade, and quite a few corner cases were found during implementation.

## Revisiting the Importance of CPU Resources

Having been through this, we have come to fully appreciate the importance of CPU resources:

1. **Use multiple processes to fully exploit multiple cores.** A single host chain cannot saturate even one core, so more processes are needed to use more cores. The combination of DP and MPS has already validated this, and we are also working on finer-grained DP: process-level replicas.
2. **Optimize every link to fully exploit a single core.** Every request pays a fixed host processing cost: encoder service, inter-stage scheduling, request construction, and kernel launch, of which the first three total about 45 ms on Fun-ASR and about 35 ms on Qwen3-ASR. For kernel launch we have already squeezed out the bulk with [CUDA Graph](../../torch/cuda-graph/readme_en.md), so scheduling and request construction are next. Bringing this cost down benefits both throughput and resistance to contention.
3. **Avoid external contention when deploying.** The lesson from CI still holds: what really takes down a service is the external tasks on the same machine. Solving this requires approaching it from container deployment and task management, estimating CPU load properly and avoiding CPU contention among tasks, rather than relying on fine-grained control inside the framework.

This experience also made us realize more deeply that for high-concurrency, low-intensity serving scenarios like speech, the CPU has always been an overlooked resource. It is just that for those of us accustomed to optimizing long context inference, "madly optimizing against the GPU" is a habit, and it blinded us to another resource that is equally decisive for throughput. Any optimization must suit the specific circumstances, start from the essence of the task and the environment, and find and exploit the fundamental limiting factor in order to solve the problems encountered in the system.


Acknowledgements: Jiaxin Deng, Yuhao Chen, Kaige Li, Huapeng Zhou, Ratish P, Ao Sun, Yueying Li, Chenyang Zhao
