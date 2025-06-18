# Tiny-LLM Day 1

配环境稍微折腾了一会儿，因为我新的 mac 发现要配合 mlx 需要安装 XCoder，搞了半天绕不过去，遂安装。然后只能用 python 3.10 到 3.12 的版本。

整体体验感觉文档写的比较 fly bitch，不知是不是迟先生助教做多了的缘故 🤣

这里实现两个功能，一共就三个函数，简单梳理下：

## Scaled Dot-Product Attention

```python
def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    d_k = key.shape[-1]
    scale = scale or 1.0 / sqrt(d_k)
    
    # 使用乘法而不是除法
    score = (query @ mx.swapaxes(key, -2, -1)) * scale  # 注意这里用 * 而不是 /
    if mask is not None:
        score += mask
    
    attn = softmax(score, axis=-1) @ value
    return attn
```

这个函数自然是比较简单的，快速过一下：

1. 我们一般使用的 attn 公式是 $$ \text{attn} = \text{softmax}(\frac{QK^T}{\sqrt{d_k}}) V $$。其中，维度信息值得格外注意。公式中的维度都是对二维的张量而言的，但是题目输入的维度是 `(N..., seq_len, d_model)`，这意味着 `N...` 表示有任意情况的 batch 维度，实际上做矩阵转置的只有最后两个维度。所以，直接用 `key.T` 必然完蛋。这里转置了最后两个维度，也就是我们的 `(seq_len, d_model)`。
2. 这个 mask 不是 attn mask，而是 attn score 的 mask，需要额外加到 attn score 上。
3. 使用乘法去完成 scaled dot-product，这个地方折磨了我半小时。实际上，我们可以写作 `(QK^T) / sqrt(d_k)`，但是这样数值精度是过不去的。首先，一般除法转换为乘以倒数是很常见的稳定数值方法；此外，大规模的矩阵做除法等价于每个元素都损失了精度，而用乘以倒数的方式只会损失倒数计算的那一次。
4. softmax 需要有维度，但我感觉记住就行。

## SimpleMultiHeadAttention

这道题继续加深我们对维度的理解。题面上说明了多次维度转换，我直接在代码里标注：

```python
class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_dim = hidden_size // num_heads
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.scale = 1 / sqrt(self.hidden_dim) # hidden dim 是 每个 head 的，不是 # 总的 hidden size
        assert wq.shape == (hidden_size, hidden_size)
        assert wk.shape == (hidden_size, hidden_size)
        assert wv.shape == (hidden_size, hidden_size) # wq wk wv 都是 (hidden_size, hidden_size) 方阵
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        N, L, _ = query.shape # N 是 batch size, L 是 sequence length
        # Query 的维度是 N, L, hidden_size (或者说 embedding size，这里和 hidden_size 一样大)

        # 这里直接用写好的 linear 点乘，linear 实际上执行的是 y = xA^T + b
        # 深度学习框架中的 linear 层实现和数学公式的写法有一个约定俗成的差异：
        # 数学公式中我们习惯写 XW，但在实际代码实现中，权重矩阵 W 通常是转置存储的，也就是说，框架中存储的实际上是 W^T
        proejct_q = (
            linear(query, self.wq) # 要把最后一个维度的 hidden_size 拆成 num_heads 个 hidden_dim 得到 (N, L, num_heads, hidden_dim)
            .reshape(N, L, self.num_heads, self.hidden_dim)  # 还要转置，将 num heads 提前
            .transpose(0, 2, 1, 3)  # (N, num_heads, L, hidden_dim)
        )
        projection_k = (
            linear(key, self.wk)
            .reshape(N, L, self.num_heads, self.hidden_dim)
            .transpose(0, 2, 1, 3)
        )
        projection_v = (
            linear(value, self.wv)
            .reshape(N, L, self.num_heads, self.hidden_dim)
            .transpose(0, 2, 1, 3)
        )
        attn = scaled_dot_product_attention_simple(
            proejct_q,
            projection_k,
            projection_v,
            scale=self.scale,
            mask=mask,
        ) # (N, num_heads, L, hidden_dim)
        # 先把 num heads 转置回去，然后 reshape 合并，回到 (N, L, hidden_size)
        attn = attn.transpose(0, 2, 1, 3).reshape(N, L, self.hidden_dim * self.num_heads)
        return linear(attn, self.wo)
```