# 通过 Torch Memory Snapshot 分析显存泄露问题

前一段时间我们在 RL 训练和 SGLang 本身的推理当中都遇到了一定的显存泄露问题。昨天终于想明白了具体泄露的原因，但是在分享原因之前，先花时间复盘下 Torch Memory Snapshot 的用法。

特别致谢：Hongyu Lu（TikTok），Huapeng Zhou（UW），Changyi Yang（CMU），Xinpeng Wei（Amazon），Rohan Bavishi（Amazon），Xinyuan Tong（USC），Yuhao Yang（HKU），Biao He（LinkedIn），Chenyang Zhao（LMSYS）

## 背景

很有意思的是，我们并不是为了支持分析显存泄露才现场学习的 Torch Memory Snapshot，而是大概一个月前，在解决 FSDP2 问题的时候就在逐步使用了。回到我们先前的文章，[FSDP 训练后端](../../rlhf/sys-design/readme-2.md#fsdp-in-verl)，我们提到过，直觉上从 FSDP1 切换到 FSDP2 并不麻烦，只需要修改四行配置：

```bash
actor_rollout_ref.ref.strategy=fsdp2
actor_rollout_ref.actor.strategy=fsdp2
critic.strategy=fsdp2
reward_model.strategy=fsdp2
```

然而很不幸，我们惊奇的发现，FSDP1 的脚本平移到 FSDP2 上后，稳定会 OOM。更神奇的是，我们把自己 OOM 的脚本交给 verl 团队和 Pytorch 负责 FSDP2 的工程师，他们发觉 8B 模型不会 OOM，但是 3B 模型稳定 OOM。折腾了很久，最后通过 `set_expandable_segments(True)` 解决了问题，相关 PR 见[3020](https://github.com/volcengine/verl/pull/3020)。

<details>
<summary>Expandable Segments 机制</summary>

`set_expandable_segments(True)` 通过开启 CUDA 的可扩展内存段功能，使得 PyTorch 能够更灵活地管理 GPU 内存。PyTorch 在 CUDA 后端上的内存分配主要由 CUDA caching allocator 管理。allocator 不会立即将释放的内存返回给操作系统，而是将其保存在一个内部的内存池中，以便后续的内存请求可以快速得到满足。这种机制通过减少与 CUDA API 的交互来提高性能。内存池实质上由 segment 和 block 两个概念来描述。

1. Segments (内存段)：内存段是 PyTorch 从 CUDA 驱动程序请求的大块连续内存。这些段是内存分配的最小单位，所有的 PyTorch 张量和数据都存储在这些段中。所有分配的 segment 总和就是 Reserved Memory。
2. Blocks (内存块)：每个内存段都包含许多小块内存（blocks）。当 PyTorch 需要分配内存时，它会在一个现有的段中寻找一个合适的空闲块。如果找不到，它会尝试从 CUDA 驱动程序中申请一个新的段。所有分配的 block 总和就是 Allocated Memory。

默认情况下，当 PyTorch 的 caching allocator 无法在现有内存段中找到足够大的空闲块时，它会向 CUDA 驱动程序请求一个新的内存段。这个新段的大小是根据当前的内存需求动态决定的。但是，这种动态扩展机制可能导致内存碎片化，先前分配的 segment 留下的 block 迟迟无法被利用，尤其是在 PyTorch 内存分配器频繁地释放和申请大块内存的情况下。

回到 FSDP 上，FSDP 默认采用 zero3 的策略，在 forward 和 backward 都需要 all gather，每个 GPU 节点会临时聚合其他节点的 parameter shard，这会创建临时的大量张量，导致对连续内存的需求激增。在传统的内存管理模式下，如果 caching allocator 无法找到一个足够大的连续内存块来容纳这些临时的大张量，就会直接 OOM。即使 GPU 仍有可用内存，但由于内存碎片化，没有足够的连续空间来容纳所需的新张量。

`torch.cuda.memory._set_allocator_settings("expandable_segments:True")` 将 PyTorch 的内存管理模式切换为一种更灵活的模式。开启该功能后，当 caching allocator 需要更大的连续内存时，它不再仅仅尝试从 CUDA 驱动程序中请求一个全新的段，而是尝试扩展已有的内存段。这种扩展机制允许 PyTorch 重新调整其内存布局，将分散的空闲内存块扩展或者合并为更大的连续块，从而满足那些对大块内存有需求的临时张量的分配。
</details>