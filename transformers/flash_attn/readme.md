# Flash Attention 学习随笔

由于工作原因要系统性学习 FA3 的算子库，于是从 FA 第一代开始，系统性学习整个 Flash Attention 的发展。

## Self-Attention 反向传播梯度推导

### 梯度定义与迹的关系

设标量损失函数为 $L$。对于矩阵 $\mathbf{X}$，梯度 $\mathbf{G}_X$ 在深度学习圈子往往被习惯记作 $dX$，这非常容易与微分算子产生混淆。首先证明梯度与迹的关系：


1. 设 $L$ 是一个标量函数，其自变量是一个 $m \times n$ 的矩阵 $\mathbf{X}$。根据多元微积分的定义，当 $\mathbf{X}$ 产生微小变化 $d\mathbf{X}$ 时，$L$ 的微小变化（全微分）$dL$ 是各分量偏导数的线性叠加：

$$dL = \sum_{i,j}^{} \frac{\partial L}{\partial X_{ij}} dX_{ij}$$


对于任意两个维度相同的矩阵 $\mathbf{A}$ 和 $\mathbf{B}$，它们的**弗罗贝尼乌斯内积（Frobenius Inner Product）** 定义为对应元素乘积之和，这个内积有一个恒等式性质：

$$ \langle \mathbf{A}, \mathbf{B} \rangle_F = \sum_{i,j} A_{ij} B_{ij} = \text{Tr}(\mathbf{A}^\top \mathbf{B})$$


证明如下：考虑矩阵乘法 $\mathbf{C} = \mathbf{A}^\top \mathbf{B}$，其对角线元素 $C_{jj}$ 为：


$$C_{jj} = \sum_{i=1}^{m} (\mathbf{A}^\top)_{ji} B_{ij} = \sum_{i=1}^{m} A_{ij} B_{ij}$$


则其迹（对角线之和）为：

$$\text{Tr}(\mathbf{A}^\top \mathbf{B}) = \sum_{j=1}^{n} C_{jj} = \sum_{j=1}^{n} \sum_{i=1}^{m} A_{ij} B_{ij}$$

将此性质应用到梯度，$\mathbf{A} = \nabla_{\mathbf{X}} L$ 且 $\mathbf{B} = d\mathbf{X}$ 代入上式：

$$dL = \sum_{i,j} \frac{\partial L}{\partial X_{ij}} dX_{ij} = \text{Tr}((\nabla_{\mathbf{X}} L)^\top d\mathbf{X})$$

至此，我们有：

$$dL = \text{Tr}(\mathbf{G}_X^\top d\mathbf{X})$$


### 迹的性质

1. 转置不变性：$$ \text{Tr}(\mathbf{A}) = \text{Tr}(\mathbf{A}^\top) $$

2. 循环移位性：$$ \text{Tr}(\mathbf{ABC}) = \text{Tr}(\mathbf{BCA}) = \text{Tr}(\mathbf{CAB}) $$

3. 线性：$$ d(\text{Tr}(\mathbf{A})) = \text{Tr}(d\mathbf{A}) $$

### 反向传播的梯度

在前向传播中：$O = PV$，其中 $P \in \mathbb{R}^{N \times N}, V \in \mathbb{R}^{N \times d}, O \in \mathbb{R}^{N \times d}$。已知输出梯度 $\mathbf{G}_O$（即 $dO$)，求 $dV$（即 $\frac{\partial L}{\partial V}$）:

首先做偏微分 $dO = P(dV)$（视 $P$ 为常数）。代入定义式：$dL = \text{Tr}(\mathbf{G}_O^\top dO) = \text{Tr}(\mathbf{G}_O^\top P dV)$。利用迹的性质，将 $dV$ 孤立在右侧：$dL = \text{Tr}((\mathbf{G}_O^\top P) dV)$。因此，$\mathbf{G}_V^\top = \mathbf{G}_O^\top P \implies \mathbf{G}_V = P^\top \mathbf{G}_O$。结论：$dV = P^\top dO$。


同样，求 $dP$（即 $\frac{\partial L}{\partial P}$）。首先进行偏微分：$dO = (dP)V$（视 $V$ 为常数）。代入定义式：$dL = \text{Tr}(\mathbf{G}_O^\top dP V) = \text{Tr}(V \mathbf{G}_O^\top dP)$。对比得到 $\mathbf{G}_P^\top = V \mathbf{G}_O^\top \implies \mathbf{G}_P = \mathbf{G}_O V^\top$。结论：$dP = dO V^\top$


进一步推导注意力层，前向公式：$S = QK^\top$，其中 $Q, K \in \mathbb{R}^{N \times d}, S \in \mathbb{R}^{N \times N}$。已知条件：梯度 $\mathbf{G}_S$（即经过 Softmax 反向传播后的 $dS$），求 $dQ$（即 $\frac{\partial L}{\partial Q}$）：

偏微分 $dS = (dQ)K^\top$。代入定义式：$dL = \text{Tr}(\mathbf{G}_S^\top dQ K^\top) = \text{Tr}(K^\top \mathbf{G}_S^\top dQ)$。因此 $\mathbf{G}_Q^\top = K^\top \mathbf{G}_S^\top \implies \mathbf{G}_Q = \mathbf{G}_S K$，结论 $dQ = \mathbf{G}_S K$。

求 $dK$（即 $\frac{\partial L}{\partial K}$）。首先进行偏微分：$dS = Q d(K^\top) = Q(dK)^\top$。代入定义式：$dL = \text{Tr}(\mathbf{G}_S^\top Q (dK)^\top) = \text{Tr}(dK Q^\top G_S)$。因此 $\mathbf{G}_K^\top = Q^\top \mathbf{G}_S \implies \mathbf{G}_K = \mathbf{G}_S^\top Q$，结论：$dK = \mathbf{G}_S^\top Q$。


整理所有结论如下：

$$
\begin{aligned}
    dV &= P^\top dO \\
    dP &= dO V^\top \\
    dQ &= \mathbf{G}_S K \\
    dK &= \mathbf{G}_S^\top Q
\end{aligned}
$$

### 记号说明

在之前的记号中，我们将 $dV$ 视作了 $\frac{\partial L}{\partial V}$ 的简记，这在数学上混淆了微分算子和偏微分记号；但在深度学习领域，这种写法相当普遍。为了不写分式在数学推导中，最严谨的写法应该是 $\frac{\partial L}{\partial \mathbf{O}} \in \mathbb{R}^{N \times d}$，但在写复杂的反向传播算法（如 Flash Attention）时，如果满篇都是 $\frac{\partial L}{\partial Q}, \frac{\partial L}{\partial K}, \frac{\partial L}{\partial V}$，公式会变得非常臃肿，难以阅读。于是，深度学习领域习惯用“ d + 变量名”来直接表示“损失函数对该变量的偏导数矩阵”。$dO$ 实际上是 $\frac{\partial L}{\partial O}$ 的缩写。$dV$ 实际上是 $\frac{\partial L}{\partial V}$ 的缩写。这一写法与代码实现一一对应：如果你去写 PyTorch 的底层 C++ 算子或者 CUDA 核函数，你会发现变量名就是这么取的：

```Python
# O 是前向传播的输出
# grad_output 是从上一层传回来的梯度 (即 dO)
# grad_V 是我们要计算的梯度 (即 dV)

def backward(grad_output):
    # 根据公式 dV = P.T @ dO
    grad_V = P.t() @ grad_output 
    return grad_V
```

我们不会定义一个变量叫 `partial_L_over_partial_O`，而是直接叫它 `grad_output` 或者 `dO`。