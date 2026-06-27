# The Root Cause of RL Training-Serving Skew is Pervasive Across Inference Systems

## The Truth Hidden by Cache Capacity

In voice cloning, each request needs to encode its reference audio into discrete codec tokens before generation starts. As concurrency increases, repeatedly encoding the same references slows down preprocessing. Since the codec encoder itself supports batched encoding, a natural optimization is to bucket concurrent references by length and encode references in the same bucket together as a batch. That was the idea behind [PR #749](https://github.com/sgl-project/sglang-omni/pull/749).

The initial numbers looked promising: at concurrency 16, throughput improved by 23%. At first, we were quite confident that the direction was right. But later we realized that the commit with the large improvement had actually changed two variables. Besides adding the batching optimization, it also increased the reference audio cache limit from 256 to 1024. In other words, the experiment did not properly control for variables.

This detail ended up changing the interpretation of the result. The reference encoding result is cached using the audio content as the key. On a cache hit, the encoded result is reused directly. The default cache holds at most 256 entries. The SeedTTS dataset used in the test contains 666 unique reference files, so a cache of 256 cannot hold the full working set. During the run, entries were repeatedly evicted and missed, forcing many references to be recomputed. After the cache limit was raised to 1024, all 666 references could stay resident. The hit rate became almost saturated, and recomputation was largely eliminated. After we isolated the variables, it became clear that the 23% gain mainly came from the cache effect, not from the batching optimization.

Once we realized the variables were mixed, we reran the experiment with a stricter control: keep only the batching optimization, lock the cache limit to 256 on both sides, and leave everything else unchanged. The result was negative. Throughput dropped by 0.8% at concurrency 1 and by 4.4% at concurrency 16.

The reason was not complicated. Length-based bucketing fragmented the batches. In a single round of concurrent requests, only a limited number of references landed in the same bucket, while the extra costs of loading, resampling, bucketing, waiting, and scheduling all remained. For this workload, the benefit from batching was not enough to cover the overhead. In the end, the batching optimization was not merged, while the cache capacity change was merged separately as an independent improvement.

This was the first lesson from the optimization attempt. We all learn the idea of controlled variables early on, but it is still easy to overlook in real systems. When a change brings a significant improvement, the first thing to check is whether another variable changed at the same time. If multiple variables move together, the source of the gain needs to be separated carefully.

## Batching Changes Discrete Tokens

While trying to further optimize batching performance, we observed another behavior: for the same reference audio, encoding it alone and encoding it as part of a batch produced different discrete tokens.

What made this more interesting is that the mismatch was not caused by randomness or padding.

First, it was not a random jitter. With a fixed batch shape, repeated runs were bit-identical. Encoding the audio alone was also fully deterministic across repeated runs. The difference only appeared between the single-example path and the batched path.

Second, it was not a padding artifact. We specifically tested two audio samples with the same number of frames in the same batch, and the mismatch still appeared. Under same-length batching, the code mismatch rate between single encoding and batched encoding was around 5.8%, while the mismatch rate between repeated single encodings was 0.

We then analyzed the source of the mismatch layer by layer. The first divergence appeared at the output of a BF16 Linear layer inside the codec encoder. The root cause was a GEMM shape effect.

The `nn.Linear` layers in the encoder and quantizer, as well as the `WNConv1d` layers with kernel size 1, are all matrix multiplications under the hood. The M dimension of GEMM is batch size multiplied by sequence length. When the batch changes, M changes. cuBLAS may then choose a different kernel implementation for a different M, and different kernels can use different floating-point accumulation orders. In BF16, floating-point addition is not associative. `(a + b) + c` and `a + (b + c)` can differ by one bit in the lowest precision range. Once the accumulation order changes, the result can drift at a sub-ULP scale. The other operations in the encoder, including normalization, RoPE, activation functions, and patching, are token-wise and independent of batch shape. GEMM was the only source of this drift.

If the computation ended there, a sub-ULP perturbation would likely have no practical effect in a continuous representation. But the encoder is immediately followed by residual vector quantization. Quantization is a hard decision: it maps a continuous vector to the nearest codebook entry and outputs an integer index. There is no intermediate state. For frames that happen to lie close to the boundary between two codebook entries, a sub-ULP perturbation is enough to switch the nearest codeword from A to B. The discrete token flips. Since RVQ computes residuals layer by layer, a flip at layer 0 changes the residual and propagates the difference to later layers.

The quantizer acts as an amplifier here. It turns low-level BF16 numerical noise that would otherwise be invisible into explicit token changes. This is highly relevant to train-inference mismatch in RL. For related discussion, see Thinking Machines' [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/).

## Why Text Tokenizers Do Not Have This Problem

Text tokenizers are symbolic. Whether BPE or unigram, the process is essentially a set of merge rules and vocabulary lookups over strings. It maps discrete inputs to discrete outputs.

There is no floating-point computation, no GEMM, and no cross-sample interaction. When multiple strings are tokenized as a batch, the implementation effectively loops over them independently. There is no kernel switch caused by shape changes.

The determinism of text tokenization comes from its structure: the same string always maps to the same sequence of tokens. Token caching is naturally safe. On the text side, we rarely need to think about this problem.

Speech does not have this property. Audio is a continuous signal. It cannot be segmented symbolically in the same way, so it relies on a learned neural codec to go from continuous input to discrete tokens. That path has exactly the two ingredients needed for this issue: batch-variant numerical noise introduced by GEMM, followed immediately by hard quantization. Together, they create batch-variant token drift.

The real boundary is not text versus speech. It is symbolic tokenizers versus neural tokenizers. Image tokenizers such as VQ-GAN and other learned image tokenization methods follow the same structure: a learned encoder plus a codebook. They inherit the same property. Speech just makes the issue especially visible because the signal is continuous and neural codec tokenization is hard to avoid.

## Implications for RL

In pure text models, non-determinism usually first appears during the forward pass. In Omni and other multimodal models, non-determinism can already enter during input tokenization. The input tokens themselves can drift with batch shape before the data even reaches the LLM backbone. The non-determinism has moved one full stage earlier.

For RL, this introduces a new source of train-inference mismatch that text-only RL does not usually face: rollout and reference paths may receive different input tokens under different concurrency patterns. The root cause is similar to the classic numerical non-determinism seen in LLM inference, but it happens one stage earlier and is much easier to miss.
