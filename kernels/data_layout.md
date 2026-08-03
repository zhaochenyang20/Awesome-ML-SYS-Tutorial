# 数据的布局与记号

> 对《Modern GPU Programming For MLSys》一书中 [Data Layout and Its Notation](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_data_layout/index.html) 章节的中文翻译整理。

实在是读了一两次没有想明白一些问题，索性自己把思索过程记录下来，作为对原文的补充。本来想水水带过这一章节，但是 BBuf 给我说，就算不自己写 kernel，理解 GPU 架构对于工作本身是很有帮助的。

> kernel 的话还是得算子仙人来搓，这个学习周期很长。没必要懂 cutlass 细节，就是这样子，大概知道 GPU 换代主要是提升了啥，然后有哪些常见的优化事件吧

> 和 agent 交流做优化，我们也是自己去寻找很多可优化的点，然后让 agent 去做，但是没有这个 context 是不可以的

如我所料，know how，或者用一个更时髦的名词来说，situational awareness 是非常重要的。

## TL;DR

1. data layout 负责将 tensor 的逻辑下标映射到物理位置。这个映射决定的不只是程序是否能读到正确的数据，还决定 global memory 访问能否 coalesce、shared memory 访问是否遇到 bank conflict，以及一个 tile 是否具有某个特定硬件单元所要求的格式。

> 额外补充下两个概念。coalesce（合并）：global memory 的访问是按 transaction / cache line 为单位考虑成本的。一个 warp 有 32 个 lane；它们同时 load/store 时，若 32 个地址落在同一条（或少数几条）连续 cache line 上（coalesced），硬件合并成 1 次（或很少几次）transaction；若地址东一块西一块（uncoalesced），则最多可能会去读取 32 个 cache line。bank conflict：shared memory（SMEM）被硬件切成很多 bank（常见是 32 个）。不同 bank 可以并行访问；同一个 bank、不同地址的并发访问必须串行化。无 conflict 时，32 个 lane load 32 个不同 bank 时，1 个 cycle 即可完成；有 conflict 时，多个 lane 撞到同一 bank 的不同地址，则必须分批在多个 cycle 内完成 load。这两个名词本身强调都是“数据排布对 load/store 指令的快慢影响巨大”。

2. Shape-Stride 模型用一个 shape 和一组 stride 来定义这个映射。Tiling 在把原始下标拆分成更多坐标之后，使用的仍是同一个模型。Named axes 把物理位置的表达扩展到 TMEM、warp lane 和寄存器，而 replication 与 offset 则表示数据副本和固定平移。

3. swizzle 在不改变 tile 逻辑 shape 的前提下重排 shared memory 地址。当 element width、alignment 和访问模式相匹配时，XOR swizzle 可以把访问分散到各个 memory bank 上，避免 bank conflict。

无论是 CPU 还是 GPU 计算，数据存储的排布方式都对计算性能影响巨大。Tensor 的逻辑下标并不说明它的字节实际存放位置，但是硬件对这一存放位置高度敏感。譬如，存放位置决定 32 个 lane 的 load 是合并成一个 transaction 还是分裂成多达 32 个，决定这些地址是落在不同的 memory bank 上还是撞在一起被串行化，也决定一个 tile 的字节排布是否能被 Tensor Core 整体读取。

机器学习程序通常只用逻辑 shape 来描述一个 tensor。data layout 则补上了缺失的物理信息：它说明逻辑下标 `(i, j, …)` 处的元素位于何处。是在内存中、在某个寄存器中，还是在另一种硬件存储空间中。

我们从 Shape-Stride 模型开始，然后把同一套记号扩展到 TMEM、register fragment 和多 GPU layout。最后我们讨论 swizzling，通过重排地址，让对同一个 tile 的行方向访问和列方向访问都能得到改善。

## Shape-Stride 模型

为了分析 GPU 专有的 layout，我们先从 Shape-Stride 模型入手。对于一个 tensor，其 `shape` 给出每个维度的大小。对应的 `stride` 则说明当某个维度的逻辑下标增加 1 时，物理上需要移动多少个元素。我们把这一对写作 `S[(shape) : (strides)]`。一个逻辑下标的物理位置就是该下标与 stride 的点积。例如，一个 row-major 的 `4 × 4` 矩阵是：

```
S[(4, 4) : (4, 1)]

addr(i, j) = i·4 + j·1
```

PyTorch 和 NumPy 的 tensor 已经在用这个模型了：一块扁平的 storage buffer，加上描述如何解释这块 storage 的 `shape` 和 `strides` 元数据。

```python
import torch

t = torch.arange(12).reshape(3, 4)
t.shape        # torch.Size([3, 4])
t.stride()     # (4, 1)        ← 正是 S[(3, 4) : (4, 1)]
```

`t` 的底层 storage 仍然是一维的（地址空间是一维的，而不是硅片上的物理结构）：

```
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
```

这里 `t` 用的是 `S[(3, 4) : (4, 1)]`：每一行占据四个连续元素，相邻的列在 storage 中也是相邻的。许多产生 view 的操作只需要改变 `shape` 和 `strides`，并不重排元素。例如对二维 tensor 来说，`permute(1, 0)` 等价于 `t.T`：

```python
tt = t.permute(1, 0)               # 或 t.T
tt.shape                           # torch.Size([4, 3])
tt.stride()                        # (1, 4)        ← stride 互换，数据未移动
tt.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
                                   # True，仍是同一块底层 storage
```

转置后的 view 用的是 `S[(4, 3) : (1, 4)]`，所以 `tt[i, j]` 的地址偏移是 `i·1 + j·4`，正好就是 `t[j, i]` 所在的位置。对 contiguous tensor 调用 `view`，或者在现有 layout 兼容时调用 `reshape`，原理相同。NumPy 遵循同一个模型，区别只是它的 `.strides` 以字节而非元素为单位。

## Tile And Its Layout

GPU kernel 很少一次处理一整个矩阵，通常会把它划分成更小的 tile。例如，我们可以把一个 `8×8` 矩阵划分成 `2×4` 的 tile，tile 之间按 row-major 存放，每个 tile 内部的元素也按 row-major 存放。划分结果如下（格子里是逻辑编号 `i·8+j`）：

```
     c0 c1 c2 c3 │ c4 c5 c6 c7
r0    0  1  2  3 │  4  5  6  7
r1    8  9 10 11 │ 12 13 14 15
    ─────────────┼─────────────
r2   16 17 18 19 │ 20 21 22 23
r3   24 25 26 27 │ 28 29 30 31
    ─────────────┼─────────────
r4   32 33 34 35 │ 36 37 38 39
r5   40 41 42 43 │ 44 45 46 47
    ─────────────┼─────────────
r6   48 49 50 51 │ 52 53 54 55
r7   56 57 58 59 │ 60 61 62 63
```

共 `4 × 2 = 8` 个 tile，每个 tile 含 8 个元素。

注意到，tile 这个定义会在各个层级会反复出现，含义略有区别。总体上，tile 都是指从一个更大的 tensor 中切出的矩形子块，由一个 tile shape 描述（如 `2×4`、`128×64`），沿每个维度对原 tensor 做整齐划分。原 tensor 因此被覆盖为一个 tile 的网格：`8×8` 矩阵按 `2×4` 划分得到 `4 × 2 = 8` 个 tile。

举一些例子：

1. 工作分配单位：一个 CTA（或 warpgroup）负责计算一个 output tile。GEMM 里"每个 CTA 算 C 的一个 `128×128` 块"说的就是这个。
2. 数据搬运单位：一次 TMA / `cp.async.bulk.tensor` 的粒度就是一个 tile；tensor map descriptor 里的 tile shape 字段直接编码它。
3. layout 的分解单位：shape 里多出来的那几个坐标（`tile_row`/`row_in_tile`/…）正是 tiling 在 layout 记号中的痕迹。

tile 是嵌套的，比如说：

CTA tile          128×128     ← 一个 CTA 的输出范围，受 SMEM / accumulator 容量约束
warp tile      64×64       ← warpgroup 内的划分
instruction tile        ← MMA 指令的固有形状，如 m16n8k16 / tcgen05 的 MMA shape

tile 尺寸是不自由的。上界由容量决定（SMEM 大小、寄存器 / TMEM 预算）；下界与对齐由硬件粒度决定（cache line 128 B、swizzle atom 行宽、MMA instruction shape 必须整除）。所谓"调 tile size"实际是在这两组约束的交集里选点。

本章后面 swizzle 一节出现的 atom（如 `8 × 128 B`）是地址置换的最小重复单元，属于 layout 层面的概念，与这里作为计算/搬运单位的 tile 不是一回事。文献中 block、partition、fragment 也常与 tile 混用，读时需按上下文判断指的是哪一层。


【TODO】

### 把 tiling 表达为 layout function

有了 tile 这一层级的抽象后，一个数据想要描述这种排布，需要同时给出 tile 在矩阵中的位置和元素在该 tile 内的位置。我们可以用一个 `8*8` 的矩阵来举个例子。从逻辑矩阵坐标 `(i, j)` 出发，按原始的 `8×8` shape 把它拍平, `addr(i, j) = i·8 + j`。把矩阵划分成 `2×4` 的 tile 之后，行坐标拆分成在四行 tile 中的行序号和在每个 tile 内的行序号；列坐标拆分成在两个 tile 列中的列序号和每个 tile 内的列序号。因此，用来分解 `x` 的 shape 是 `(4, 2, 2, 4)`，也即 `(tile_row_index, inside_tile_row_index, tile_col_index, inside_tile_col_index)`。

【TODO】

layout 按这个 shape 对 `x` 做 unflatten：

```
(c0, c1, c2, c3) = unflatten(x; 4, 2, 2, 4)

c0 = x // 16
c1 = (x // 8) % 2
c2 = (x // 4) % 2
c3 = x % 4
```

代入 `x = i·8 + j` 得到：

```
c0 = i // 2    = tile_row
c1 = i % 2     = row_in_tile
c2 = j // 4    = tile_col
c3 = j % 4     = col_in_tile
```

接下来我们把这四个坐标映射到一个物理地址。每个 tile 含 `2×4=8` 个元素，每个 tile 行含两个 tile，每个 tile 内的每一行含四个连续元素。因此：

$$
\begin{aligned}
f_D(x) &= (c_0\cdot 2 + c_2)\cdot 8 + c_1\cdot 4 + c_3 \\
       &= c_0\cdot 16 + c_1\cdot 4 + c_2\cdot 8 + c_3\cdot 1
\end{aligned}
$$

得到的 layout 是：

```
S[(4, 2, 2, 4) : (16, 4, 8, 1)]
```

> 📊 *在原文的交互图中点击任意一个 cell，可以把它的 tile 坐标和物理地址与 unflatten 过程及 $f_D(x)$ 对照起来看。*

**手算示例**（`x = 11`，即逻辑位置 r1c3）：

```
c0 = 11//16    = 0    tile_row
c1 = (11//8)%2 = 1    row_in_tile
c2 = (11//4)%2 = 0    tile_col
c3 = 11%4      = 3    col_in_tile

f_D = 0·16 + 1·4 + 0·8 + 3·1 = 7
```

逻辑下标 11 → 物理地址 7。注意 `c1 = 1` 贡献的 `1·4` 正是"跳过 tile 内第 0 行的 4 个元素"。逻辑上 `11` 与 `3` 相差 8（隔一整行），物理上只差 4——因为中间的 `4,5,6,7` 属于隔壁 tile，被挪走了。

物理内存的完整排布（每行恰好是一个 tile）：

```
地址 0..7:    0  1  2  3   8  9 10 11     ← tile(0,0)
地址 8..15:   4  5  6  7  12 13 14 15     ← tile(0,1)
地址 16..23: 16 17 18 19  24 25 26 27     ← tile(1,0)
地址 24..31: 20 21 22 23  28 29 30 31     ← tile(1,1)
...
```

### 一般的 layout function

同样的计算可以推广到一般的 Shape-Stride layout：

```
S[(e0, e1, ..., en-1) : (s0, s1, ..., sn-1)]
```

对一个扁平的逻辑下标 $x$，首先按 shape 对它做 unflatten：

$$(c_0, c_1, \ldots, c_{n-1}) = \operatorname{unflatten}(x; e_0, e_1, \ldots, e_{n-1})$$

然后取这些坐标与 stride 的点积：

$$f_D(x) = \sum_{k=0}^{n-1} c_k s_k$$

shape 决定 $x$ 如何被分解成坐标，而 stride 决定这些坐标如何映射到物理位置。上面那个 tile layout，就是选取 shape `(4, 2, 2, 4)` 和 stride `(16, 4, 8, 1)` 的结果。

---

## Named Axes：从线性地址到物理坐标

上面这些 layout 都把每个元素映射到一个线性内存地址。然而，有些 GPU 存储空间需要不止一个坐标才能确定一个物理位置。TMEM 和 register fragment 是两个最直接的例子。

### 二维的 TMEM 地址空间

Blackwell 的 TMEM 本质上是二维的。每个 CTA 有 128 个 lane 行，以及最多 512 个 32-bit 的列。因此，TMEM 中的一个位置需要同时给出 lane 坐标和 column 坐标。

> 📊 *原文配图：TMEM 使用二维地址空间，含 128 个 TLane 行和最多 512 个 TCol 列；图中所示的 accumulator 占据一个 `128×256` 的区域。*

单一的线性内存轴无法区分这两个维度。我们用 `@TLane` 和 `@TCol` 表示 TMEM 的 lane 轴和 column 轴。例如，一个 `128×256` 的 accumulator tile 可以写成：

```
S[(128, 256) : (1@TLane, 1@TCol)]

(row, col) = unflatten(x; 128, 256)
f_D(x) = row@TLane + col@TCol
```

这里 $f_D(x)$ 不再返回单个整数地址，而是同时返回 `TLane=row` 和 `TCol=col`。相比之下，普通的线性内存只有一个地址轴 `@m`。把这个 tag 显式写出来，一个 row-major 的 `8×16` 内存 tile 是：

```
S[(8, 16) : (16@m, 1@m)]

(row, col) = unflatten(x; 8, 16)
f_D(x) = (row·16 + col)@m
```

### Register Fragment

Named axes 也出现在 Tensor Core 所使用的 register fragment 中。考虑一个 m8n8 风格的 fragment。逻辑上它包含一个 `8×8` 的 tile，共 64 个元素。物理上，这些元素分布在一个 warp 的 32 个 lane 上，所以每个 lane 持有两个 fragment slot。

因此，仅凭 lane ID 不足以确定一个元素。它的物理位置由两部分组成：哪个 lane 拥有它，以及它在该 lane 内占据哪个 fragment slot。对这个 layout：

```
laneid = row·4 + col//2
reg    = col%2
```

> 📊 *原文配图展示 `8×8` tile 如何分布到 warp lane 和寄存器上。*

在左侧 `Logical 8×8 Matrix` 中点击第 `r5` 行、第 `c3` 列的 cell 43，图中会显示：逻辑元素 `(5, 3)` 由 lane 21 拥有，并占据该 lane 中的 fragment slot 1。

我们用 `@laneid` 和 `@reg` 表示这两个坐标。`@laneid` 轴是 warp 内的 lane ID；`@reg` 是该 lane 局部的 fragment slot。这里 `@reg` 在 layout 中是一个 lane-local 的坐标。某条具体指令仍然可能把多个低精度元素 pack 进一个 32-bit 的硬件寄存器。

这个 `8×8` tile 可以写成：

```
S[(8, 4, 2) : (4@laneid, 1@laneid, 1@reg)]

(c0, c1, c2) = unflatten(x; 8, 4, 2)
             = (row, col//2, col%2)

f_D(x) = (c0·4 + c1)@laneid + c2@reg
```

> 💡 注意 `c0` 和 `c1` 的 stride 都落在 `@laneid` 轴上（`4@laneid` 和 `1@laneid`）——多个逻辑坐标可以贡献到同一个 named axis。

---

## Replication 与 Offset

### 在 TMEM 中跨 warp 广播 scale factor

Block-scaled MMA 不是某种具体的数据类型，而是一族使用 per-block scale factor 的低精度 MMA 操作。Blackwell 上常见的格式包括 MXFP8 和 NVFP4。就这里讨论的 local block scaling 而言，MXFP8 在沿 K 的每 32 个元素间共享一个 E8M0 的 scale factor，而 NVFP4 在每 16 个 E2M1 FP4 元素间共享一个 E4M3 的 scale factor。

两种情况下底层思想是一样的：把 A 和 B 沿 K 划分成 scale block，为每个 block 指定一个 scale factor。若每个 scale block 沿 K 含 `K_blk` 个元素，则元素 `k` 所属的 block 是：

```
sfk = k // K_blk
```

数学上，block-scaled MMA 等价于在做 matrix multiply-accumulate 之前，先用各自对应的 scale factor 缩放 A 和 B 的元素：

```
A_real[m, k] = A_low[m, k] · SFA[m, k // K_blk]
B_real[k, n] = B_low[k, n] · SFB[n, k // K_blk]
D = C + A_real × B_real
```

`SFA[m, sfk]` 是 A 的第 `m` 行、第 `sfk` 个 K-scale block 的 scale factor；`SFB[n, sfk]` 是 B 的第 `n` 列对应的因子。

下面这个 NVFP4 的 SFA 例子展示了这些 scale factor 如何被放进 TMEM。例子取 `M = 128`、`SF_K = 4`。每个 scale factor 占一个字节，所以逻辑上的 `128×4` SFA 共含：

```
128 rows × 4 bytes/row = 512 bytes
```

搬运这些数据的 `tcgen05.cp.32x128b.warpx4` 指令具有 `.32x128b` 的 base shape：32 个 local lane，每个 lane 128 bit（即 16 字节）。因此它的 base tile 同样含：

```
32 local lanes × 16 bytes/lane = 512 bytes
```

大小正好吻合。但 base tile 只有 32 个 lane 位置，所以 `m` 的 128 个取值不可能各占一个独立的 lane。取而代之的做法是把 `m` 拆分为：

```
local_lane = m % 32
Mgroup     = m // 32
```

`local_lane` 选出 32 个 lane 中的一个，而 `Mgroup` 选出该 lane 上的一个 TCol。对固定的 local lane `l`，四组 SFA 行被并排放置：

```
TCol 0: SFA[l,      0:4]
TCol 1: SFA[l + 32, 0:4]
TCol 2: SFA[l + 64, 0:4]
TCol 3: SFA[l + 96, 0:4]
```

每个 SFA 行含四个一字节的 scale factor，正好填满一个 32-bit 的 TCol cell。坐标 `sfk = 0…3` 用于在该 cell 内选出一个字节。因此完整的 packing 规则是：

```
local_lane = m % 32
Mgroup     = m // 32
TCol       = Mgroup
byte       = sfk

byte_offset = TCol·4 + byte
```

举例来说，`SFA[64, 2]` 的 `local_lane = 0`、`Mgroup = 2`，所以它占据 local lane 0 上 TCol 2 的第 2 个字节。`SFA[0, 2]`、`SFA[32, 2]`、`SFA[64, 2]` 和 `SFA[96, 2]` 都用 local lane 0，但分别占据 TCol 0、1、2、3——它们并不共享同一个 TMEM cell。

到这一步，一个完整的 `128×4` SFA 已经被 pack 进了一个 32-lane 的 base tile。而 block-scaled 的 `tcgen05.mma` 是通过四个 32-lane 的 partition 来读 TMEM 的，并且要求每个 partition 都在相同的 local-lane、TCol 和 byte 位置上提供完整的 scale-factor tile。因此 PTX ISA 要求 SFA 和 SFB 都必须在这四个 partition 上各存一份：

```
partition 0: TLane 0…31
partition 1: TLane 32…63
partition 2: TLane 64…95
partition 3: TLane 96…127
```

`.warpx4` 限定符把 pack 好的 base tile multicast 到这四个 partition。若 `p = 0…3` 是 partition 索引，则物理的 Lane 坐标是：

```
TLane = local_lane + 32·p
```

TCol 和 byte 坐标保持不变。因此 `SFA[64, 2]` 出现在 `(TLane, TCol, byte)` 坐标 `(0,2,2)`、`(32,2,2)`、`(64,2,2)` 和 `(96,2,2)` 处。

> ⚠️ 这里出现了两组互不相关的"四"：四个 `Mgroup` 值把 128 个逻辑 m 行沿 **TCol** 方向 pack 起来；四个 partition 则是那个 pack 好的 tile 沿 **TLane** 方向的物理副本。

SFB 遵循同样的硬件规则，只是用 B 的列索引 `n` 代替 A 的行索引 `m`。例如当 `N = 128`、`SF_K = 4` 时，它的 base packing 使用 `local_lane = n % 32`、`TCol = n // 32`、`byte = sfk`；随后 `.warpx4` 把它复制到全部四个 32-lane partition。

### 用 Replication 表示多个物理位置

上面定义的 $f_D(x)$ 对逻辑元素 $x$ 只返回一个位置，无法表示 `.warpx4` 所创建的额外副本。因此我们在 base layout 之后追加 `R[shape : strides]`。例如，`R[n : s@axis]` 引入一个独立的 replica 坐标 `r = 0…n-1`，产生 `r·s@axis` 的偏移。

对上面的 TMEM 例子，沿 `TLane` 轴的四个副本是：

```
S[(32, …) : (1@TLane, …)] + R[4 : 32@TLane]
```

在 `R[4 : 32@TLane]` 中，`r` 取 `0`、`1`、`2`、`3`，产生 `TLane` 偏移 `0`、`32`、`64`、`96`。replication 项并不增加新的逻辑数据；它记录的是这些副本的物理位置。

> 📊 *原文配图展示 SFA 如何被 pack 进一个 32-lane 的 base tile，以及 `.warpx4` 如何把该 tile 复制到四个 TMEM partition。*

### GPU Mesh 中的 Replication 与 Offset

同样的 replication 结构也可以描述多 GPU layout。一个 **GPU mesh** 把多个 GPU 沿一条或多条逻辑 device 轴排列。一个 `2×2` 的 GPU mesh 含四个 GPU，每个由坐标 `(@gpuid_x, @gpuid_y)` 标识。

先定义一个沿 `@gpuid_y` 做 shard 的 base layout：

```
base = S[(2, 4, 8) : (1@gpuid_y, 8@m, 1@m)]
```

把三个逻辑坐标记作 `(y, row, col)`。在这个 base layout 中，元素 `(1, 2, 3)` 映射到：

```
gpuid_y = 1
m = 2·8 + 3 = 19
```

加上 replication 得到：

```
base + R[2 : 1@gpuid_x]

元素 (1, 2, 3) → devices {(0, 1), (1, 1)}, local offset = 19
```

`R[2 : 1@gpuid_x]` 这一项把该元素同时放在 `gpuid_x = 0` 和 `gpuid_x = 1` 上。而固定的 offset 行为不同：

```
base + O[1@gpuid_x]

元素 (1, 2, 3) → device (1, 1), local offset = 19
```

这个 offset 把 base 位置沿 `@gpuid_x` 平移一个位置；**它不创建副本**。

> 📊 *原文配图把这两种情况与一个完全 shard 的 layout 做对比，可用控件在 fully sharded、shard + replica、shard + offset 之间切换。*

**R 与 O 的核心区别**：R 是"多处都有"，O 是"位置挪了"。

---

## Swizzle Layout

本章最后一种 layout，针对的是 shared memory 中的 bank conflict。

GPU 的 shared memory 被划分成若干 memory bank。每个 bank 可以看作一条独立的通道，为访存请求提供服务。对不同 bank 的访问可以并行进行。但如果多个 lane 在同一时刻访问同一个 bank 中的不同地址，硬件就必须分批服务这些访问，产生 **bank conflict**。

Tensor 程序经常从不止一个方向访问同一个 tile。矩阵代码可能在某处读取一整行连续元素，在另一处抽取一列。简单的 layout 通常只对其中一种模式友好。在 row-major 的 tile 中，一行内相邻的元素地址连续，通常会分散到不同的 bank 上；而一列中相邻的元素之间相隔一个 row stride，如果这个 stride 与 bank 映射的周期相匹配，来自多个 lane 的访问就会集中到同一个 bank。column-major 的 layout 则有着相反的 tradeoff。

**Swizzling** 通过改变物理地址排布、同时保持 tile 的逻辑 shape 不变来缓解这个问题。一种常见技巧是把行索引的一部分 XOR 进列索引，使目标访问模式更均匀地分散到各个 bank 上。

在下面的 `8×8` 例子中，逻辑坐标 `(row, logical_col)` 的映射是：

```
mapped_col    = logical_col XOR row
physical_addr = row·8 + mapped_col
```

`XOR` 是按位异或。当读取逻辑列 `logical_col = 0` 时，`row = 0…7` 产生 `mapped_col = 0 XOR row = 0…7`。因此，同一逻辑列中的元素落在不同的物理列上，进而落在不同的 bank 上。

> 📊 *原文配图：点击一个列索引，可以对比朴素 row-major layout 与 XOR swizzle 的 bank 映射。前者需要八个 cycle，后者只需一个。*

作为对照，原文给出了 NVIDIA PTX ISA 官方文档中的 K-major 128B swizzling layout 图。它与交互图右侧的 "With Swizzle (XOR)" 面板直接对应：两者用的是同一条 XOR 规则来置换每行中的八个位置。

两幅图上的数字看起来不同，只是因为它们采用了不同的标号方案。在每一行中，demo 用 0–7 表示每个元素原本的逻辑列；而官方图则把整个 `8×8` 矩阵从 0 到 63 连续编号。

例如，demo 中的第 1 行包含逻辑列标号 `1, 0, 3, 2, 5, 4, 7, 6`。官方图在同样的排布上加上该行的索引偏移 8，得到 `9, 8, 11, 10, 13, 12, 15, 14`。

### Swizzle atom 的层级结构

我们把图中每个 128-bit 的 cell 称为一个 16 B **sector**。在 `SWIZZLE_128B` 中，一个 atom 的每一行含八个 sector，总宽度为 128 B。在常见的 4 字节 bank 粒度下，一个 sector 横跨四个 bank，所以完整的一行覆盖全部 32 个 bank。swizzle 用行坐标对该行内的八个 sector 做 XOR 置换。

一个 `SWIZZLE_128B` atom 含八行，因此总大小是 `8 × 128 B = 1024 B`。

> ⚠️ 这里的 `128 B` 是 atom **每一行沿连续维度的宽度**，不是 atom 的总大小。atom 是地址置换的最小重复块；更大的 tile 由多个 atom 平铺而成。

其他 swizzle 模式使用同样的层级结构，只是行宽不同：

| 模式 | atom 形状 |
|---|---|
| `SWIZZLE_128B` | `8 × 128 B` |
| `SWIZZLE_64B` | `8 × 64 B` |
| `SWIZZLE_32B` | `8 × 32 B` |
| 16 B interleaved | 无 XOR swizzle |

### 怎么选 swizzle 模式

一条实用的规则是：**使用 tile 所能支撑的最大行宽的 atom**。行宽为 `N` 字节的 atom 要求 tile 的连续维度至少为 `N` 字节，并且最好能被 `N` 整除。

- 行宽至少 128 字节（即 64 个 `float16` 元素）→ `SWIZZLE_128B` 通常是首选
- 连续维度窄于 128 字节 → 使用所能支持的最大替代项：`SWIZZLE_64B` 或 `SWIZZLE_32B`

对上面展示的 fp16 访问模式，`SWIZZLE_128B` 使得连续的行读取和跨八行的列读取都无冲突。但这个保证只在 element width、swizzle 模式和访问模式都与硬件 descriptor 相匹配时才成立。改变 element width、alignment 或访问模式，都可能重新引入冲突。

### Swizzle 不属于 affine layout

在实践中，程序员不会手工计算 swizzle 后的地址。完整的映射可以看作两步：`S[...]` 先把一个逻辑元素映射到 `@m` 上的线性内存地址，然后 swizzle 再重排这个地址。由于 XOR 置换不是 affine 的，swizzle 并不属于 affine layout 本身；它是与该 layout **复合**在一起的一个独立的地址变换。

访问同一个 tile 的每一个操作都必须使用相同的 swizzle 模式。实际的地址变换由复合后的 layout 负责处理。不同的硬件单元施加不同的 swizzle 要求，而这些要求也会随 GPU 世代变化。下一章将考察这些约束。

---

## 附录：几个关键澄清

以下是研读过程中容易卡住的几个点，整理备查。

### A1. 一维 / 二维：函数两端不对称

$$f_D:\ \underbrace{\text{逻辑坐标}}_{\text{几维由 shape 决定}} \longrightarrow \underbrace{\text{物理位置}}_{\text{几维由 axis tag 决定}}$$

- **输入端**几维由 shape 说了算。tile layout 的 shape 是 `(4,2,2,4)`——四维（尽管矩阵本身二维、tile 也是 `2×4` 的二维）。
- **输出端**几维由 stride 上挂了几种 axis tag 说了算。`(16@m, 4@m, 8@m, 1@m)` 只有 `@m` 一种 → 输出一维。

所以 tile layout 是 **4 维 → 1 维**。"tile 是二维的"描述输入端，"物理地址是一维的"描述输出端，两句都对。

**判据**：定位一个元素需要的几个坐标，**能否合并成一个整数**？

| | 需要的坐标 | 能否合并 | 结论 |
|---|---|---|---|
| tiled memory | (哪个 tile, tile 内第几个) | `tile_id·8 + offset` ✓ | 一维 |
| TMEM | (TLane, TCol) | 相加无意义 ✗ | 二维 |
| register fragment | (laneid, reg) | 寄存器不可寻址 ✗ | 二维 |

`@m` 上的量可以互相加，`@TLane` 和 `@TCol` 上的量不能——这就是引入 named axes 的全部理由。

原文交互图右侧面板容易误读成二维：它的行标签（`0, 8, 16, …`）是**地址基址**，列标签（`+0 … +7`）是**行内偏移**，两者相加得到一个 0–63 的整数。之所以画成 8 列宽，是因为 `8` 正好是一个 tile 的元素数，这样每一行恰好是一个 tile。面板底部那条细长横条才是物理存储的本来面目。

### A2. tiling 的二维性去哪了

它变成了 **stride 的具体数值**。tiling 不改变地址空间形态，改变的是"哪些逻辑元素在一维地址上挨着"：

```
row-major:  0  1  2  3  4  5  6  7 | 8  9 ...   ← 一整行连续
2×4 tiled:  0  1  2  3  8  9 10 11 | 4  5 ...   ← 一个 2×4 块连续
```

两边都是 64 个连续整数。区别只在 `8,9,10,11` 被提前了——这是 stride `(16,4,8,1)` 里 `4 < 8` 的效果：`row_in_tile` 的 stride 比 `tile_col` 小。

**一个 `2×4` tile 在内存里的痕迹，就是 8 个连续整数。**

### A3. 为什么 tile 是方块而不是一行：算术强度

设输出 tile 为 `Mt × Nt`，K 全长：

- 读取量：`(Mt + Nt)·K` 个元素
- 乘加数：`Mt · Nt · K` 次

$$\text{每元素乘加数} = \frac{M_t N_t K}{(M_t + N_t)K} = \frac{M_t N_t}{M_t + N_t}$$

K 约掉了——**复用率只与 tile 形状有关**。

| tile 形状 | 复用率 |
|---|---|
| `1 × 4096`（一整行） | ≈ 1 |
| `1 × 1` | 0.5 |
| `16 × 1024` | 15.8 |
| `128 × 128` | **64** |
| `128 × 256` | 85.3 |

直观解释：SMEM 里 A 的每个元素被用 `Nt` 次，B 的每个元素被用 `Mt` 次。SMEM 容量约束的是 `Mt + Nt`，收益却是 `Mt · Nt`。固定和求最大积 → 方形。

另外，C 的 accumulator 全程驻留在寄存器（Hopper）或 TMEM（Blackwell）中，K 维流过去。`128×128` 的 fp32 accumulator = 64 KB，正好塞进一个 CTA 的寄存器预算。

补充：`128×128` 给出 64 FLOP/byte，而 H100 的 fp16 峰值 / HBM 带宽比在 **~295 FLOP/byte** 量级，单靠 tile 尺寸还不够。补上的部分来自 **L2 复用**——同一行的多个 CTA 共享 A 的同一条带，这就是为什么 CTA rasterization order 也要做 swizzle。tile 尺寸和调度顺序是一起调的。

### A4. tile 在每一层存储中的物理形态各不相同

以 row-major、`K = 4096`、fp16 的 A tile `128 × 64` 为例：

```
global memory:  128 段，每段 64 个 fp16 = 128 字节连续
                相邻两段相隔 K·2 = 8192 字节
                → 逻辑上是紧凑矩形，物理上是跨越 1 MB 的 128 个碎片
```

`64` 这个数字不是随便选的：`64 × 2B = 128B`，正好是 cache line 的整数倍，也正好是 `SWIZZLE_128B` 要求的行宽。**tile 的内层尺寸永远被 memory transaction 粒度和 swizzle atom 宽度反过来约束。**

这也正是 **TMA** 存在的理由：Hopper 之前这 128 段跨步拷贝要手写地址计算；TMA 让你预先构造 tensor map descriptor（base 指针、global shape、stride、tile shape、swizzle mode），之后 `cp.async.bulk.tensor` 一条指令给个坐标就搬整块——多维寻址下沉到硬件。

完整链条：

```
global (row-major，tile 是跨步的矩形区域)
   │  TMA / cp.async —— 在这一步做 gather + swizzle
   ▼
SMEM (tile 连续 + swizzled)
   │  ldmatrix / tcgen05 直接读
   ▼
register fragment / TMEM (@laneid+@reg 或 @TLane+@TCol)
```

### A5. "连续"是 layout 的属性，不是数据的属性

同一个 `8×8` 矩阵，问"它连续吗"没有意义：row-major 下行连续，column-major 下列连续，`2×4` tiled 下 tile 连续。前面 `t.T` 的例子最能说明——同一块 storage 一字节未动，只把 stride 从 `(4,1)` 换成 `(1,4)`，"哪个方向连续"就反了。

`2×4` tiled 下逻辑列 c0（`0,8,16,…,56`）的物理地址：

```
物理: 0, 4, 16, 20, 32, 36, 48, 52
间隔:   4, 12,  4, 12,  4, 12,  4
```

**间隔本身不均匀**——这已经不是"固定 stride"能描述的，所以必须用整套 `(shape, strides)` 元数据。

而硬件的所有性能特征都锚在物理排布上：

| 硬件行为 | 看的是 |
|---|---|
| coalescing | 一个 warp 32 个 lane 的地址是否落在同一条 cache line |
| bank conflict | 地址低位 bit 是否撞在同一 bank |
| Tensor Core 能否直接吃 | 字节排布是否符合 descriptor |

而代码里给的是逻辑下标。这中间的鸿沟全靠 layout 填。

### A6. 三个正交的扩展方向

整章其实是在同一个框架上做三件独立的事：

1. **输入怎么拆** → tiling（shape 分解 + stride 重排）
2. **输出落在哪些轴上** → named axes / replication / offset
3. **输出算完之后再怎么改** → swizzle（非 affine，复合上去）