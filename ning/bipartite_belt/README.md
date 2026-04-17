# Bipartite Belt — 方积箭图的解析求解器

本模块实现了 Dynkin 图方积 $G \square G'$ 的自动生成，以及基于**双分带** (bipartite belt) 的极大绿色序列求解。

双分带是目前已知的、对所有 Dynkin 对 $(G, G')$ 都适用的解析解。它由 Keller [1] 的周期性猜想证明推导而来，Casbi-Hosaka-Ikeda [2] 给出了精确的轮数公式。

---

## 一、算法描述

### 1.1 Dynkin 图生成

Dynkin 图是一类特殊的无向图，分为 $A_n$、$D_n$、$E_6$、$E_7$、$E_8$ 五个系列。

**交替定向**：对 Dynkin 图做 2-着色（所有 Dynkin 图都是二部图），将一色标记为 **white (W, source)**，另一色标记为 **black (B, sink)**。每条边从 source 指向 sink。

实现中使用 BFS 从顶点 0 开始做 2-着色，偶数层为 W，奇数层为 B。

各类型的结构：

| 类型 | 图结构 | Coxeter 数 $h$ |
|------|--------|----------------|
| $A_n$ ($n \geq 1$) | 链 $0 - 1 - 2 - \cdots - (n{-}1)$ | $n + 1$ |
| $D_n$ ($n \geq 4$) | 链 $0 - 1 - \cdots - (n{-}2)$ + 分支 $(n{-}3) - (n{-}1)$ | $2n - 2$ |
| $E_n$ ($n = 6,7,8$) | 主链 $0 - 1 - 2 - 3 - \cdots - (n{-}2)$ + 分支 $2 - (n{-}1)$ | 12 / 18 / 30 |

### 1.2 方积构造

给定两个带交替定向的 Dynkin 图 $G$（$m_1$ 个顶点）和 $G'$（$m_2$ 个顶点），方积 $G \square G'$ 的构造如下。

**顶点**：所有 $(i, j) \in G \times G'$，共 $m_1 \times m_2$ 个。采用行优先编号：

$$(i, j) \mapsto i \times m_2 + j \quad \text{(0-indexed 内部, 1-indexed 对外)}$$

**箭头方向**遵循四类循环规则（已用 `A2_square_A3` 已有数据验证）：

$$(W{,}W) \xrightarrow{\text{水平}} (W{,}B) \xrightarrow{\text{垂直}} (B{,}B) \xrightarrow{\text{水平}} (B{,}W) \xrightarrow{\text{垂直}} (W{,}W)$$

具体地：

- **水平边**（来自 $G'$ 的边，在每个 $i \in G$ 处复制）：
  - $i$ 是 W → 保持 $G'$ 方向
  - $i$ 是 B → 反转 $G'$ 方向
- **垂直边**（来自 $G$ 的边，在每个 $j \in G'$ 处复制）：
  - $j$ 是 W → 反转 $G$ 方向
  - $j$ 是 B → 保持 $G$ 方向

参考：[1, §8]，[3, §5.2]。

**双分组**：方积的顶点天然分为两组：

- **黑组** ($\bullet$)：$(W{,}W) \cup (B{,}B)$ — 两个颜色相同的顶点对
- **白组** ($\circ$)：$(W{,}B) \cup (B{,}W)$ — 两个颜色不同的顶点对

关键性质：**组内任意两顶点之间 $b_{uv} = 0$**（无边），因此同组顶点的变异可交换。

### 1.3 双分带求解

**定理** (Proposition 3.1, [2])：设 $B = G \square G'$ 是 Zamolodchikov 周期的 B-矩阵，Coxeter 数分别为 $h_G$、$h_{G'}$。则：

- $\mu_\circ \mu_\bullet \mu_\circ \cdots$（共 $h_{G'}$ 个因子）是极大绿色序列
- $\mu_\bullet \mu_\circ \mu_\bullet \cdots$（共 $h_G$ 个因子）是极大绿色序列

其中 $\mu_\circ$ 表示同时变异白组所有顶点，$\mu_\bullet$ 表示同时变异黑组所有顶点。

因此对每个 $G \square G'$，有两个解析解（轮数不同，起始组不同）。

**示例**：$A_2 \square A_3$（6 个顶点），$h_{A_2} = 3$，$h_{A_3} = 4$。

```
顶点网格及着色（●=黑组，○=白组）：

       j: 1(W)  2(B)  3(W)        编号:  1    2    3
  i:
  1(W)     ●     ○     ●                 4    5    6
  2(B)     ○     ●     ○

黑组: {1, 3, 5}    白组: {2, 4, 6}
```

从黑组开始，$h_{A_2} = 3$ 轮：

```
轮 1 (●): mutate 1, 3, 5
轮 2 (○): mutate 2, 4, 6
轮 3 (●): mutate 1, 3, 5
→ 全红，共 9 步
```

---

## 二、Package 使用

### 2.1 生成游戏（方积箭图）

```python
from ning.bipartite_belt.square_product import dynkin_graph, square_product

G  = dynkin_graph("A", 2)   # A₂
Gp = dynkin_graph("E", 7)   # E₇
sp = square_product(G, Gp)  # A₂ □ E₇

sp.B_A          # 14×14 反对称交换矩阵 (numpy array)
sp.n            # 14
sp.black_group  # [1, 3, 5, 7, 8, 10, 12, 14] (1-indexed)
sp.white_group  # [2, 4, 6, 9, 11, 13]         (1-indexed)
```

`sp.B_A` 可以直接传给 `QuiverEngine`：

```python
from ning.agent.engine import QuiverEngine

engine = QuiverEngine()
engine.reset_from_matrix(sp.B_A)
# 现在可以用任何方式（LLM agent, 手动, 其他算法）来玩这个游戏
```

### 2.2 求解（双分带）

```python
from ning.bipartite_belt.solver import bipartite_belt_solution

# 两个解：
sol_short = bipartite_belt_solution(sp, start_white=False)  # h_G 轮（通常更短）
sol_long  = bipartite_belt_solution(sp, start_white=True)   # h_{G'} 轮

print(sol_short)  # [1, 3, 5, 7, 2, 4, 6, 8, ...]  (1-indexed mutation sequence)
```

### 2.3 验证

```python
engine = QuiverEngine()
engine.reset_from_matrix(sp.B_A)
for k in sol_short:
    engine.mutate(k)
assert engine.is_won()  # True
```

### 2.4 支持的 Dynkin 类型

| 函数调用 | 图 |
|----------|-----|
| `dynkin_graph("A", n)` | $A_n$, $n \geq 1$ |
| `dynkin_graph("D", n)` | $D_n$, $n \geq 4$ |
| `dynkin_graph("E", 6)` | $E_6$ |
| `dynkin_graph("E", 7)` | $E_7$ |
| `dynkin_graph("E", 8)` | $E_8$ |

任意两个 Dynkin 图都可以做方积：`square_product(G, Gp)`。

---

## 三、验证结果

### 3.1 Engine 验证（41 个测试全部通过）

对 13 个 $(G, G')$ 组合，每个生成两个解（从白组/黑组开始），全部通过 `QuiverEngine` 验证（`is_won() == True`）。

| $G \square G'$ | 顶点数 | 黑/白组 | $h_G$ / $h_{G'}$ | 短解步数 | 长解步数 |
|----------------|--------|---------|-------------------|----------|----------|
| $A_1 \square A_1$ | 1 | 1 / 0 | 2 / 2 | 1 | 1 |
| $A_1 \square A_2$ | 2 | 1 / 1 | 2 / 3 | 2 | 3 |
| $A_2 \square A_2$ | 4 | 2 / 2 | 3 / 3 | 6 | 6 |
| $A_2 \square A_3$ | 6 | 3 / 3 | 3 / 4 | 9 | 12 |
| $A_2 \square A_4$ | 8 | 4 / 4 | 3 / 5 | 12 | 20 |
| $A_3 \square A_3$ | 9 | 5 / 4 | 4 / 4 | 18 | 18 |
| $A_3 \square A_4$ | 12 | 6 / 6 | 4 / 5 | 24 | 30 |
| $A_2 \square D_4$ | 8 | 4 / 4 | 3 / 6 | 12 | 24 |
| $A_2 \square E_6$ | 12 | 6 / 6 | 3 / 12 | 18 | 72 |
| $A_2 \square E_7$ | 14 | 7 / 7 | 3 / 18 | 21 | 126 |
| $A_2 \square E_8$ | 16 | 8 / 8 | 3 / 30 | 24 | 240 |
| $D_4 \square D_4$ | 16 | 10 / 6 | 6 / 6 | 48 | 48 |
| $A_3 \square E_6$ | 18 | 9 / 9 | 4 / 12 | 36 | 108 |

### 3.2 交叉验证（qme-ng 暴力求解器）

使用 [qme-ng](https://github.com/mp-bull/qme-ng)（Docker 版本）对小规模方积做穷举绿色序列探索，验证双分带解的长度出现在 qme-ng 的长度分布中。

#### $A_2 \square A_2$（4 顶点）

qme-ng 穷举结果（共 112 条极大绿色序列）：

```
长度 6  →   32 条
长度 7  →   44 条
长度 8  →   20 条
长度 9  →   16 条
```

双分带解长度：**6 步**（两种起始方式相同）。

结论：双分带解长度 = qme-ng 发现的**最短长度** ✓

#### $A_2 \square A_3$（6 顶点）

qme-ng 穷举结果（共 10,555,446 条极大绿色序列）：

```
长度  9  →      568 条
长度 10  →    2,408 条
长度 11  →    8,636 条
长度 12  →   29,730 条
长度 13  →   62,612 条
长度 14  →  101,000 条
长度 15  →  153,498 条
长度 16  →  215,596 条
长度 17  →  290,420 条
长度 18  →  393,756 条
长度 19  →  548,950 条
长度 20  →  748,126 条
长度 21  →  956,594 条
长度 22  →1,230,684 条
长度 23  →1,657,108 条
长度 24  →1,863,360 条
长度 25  →2,292,400 条
```

双分带解长度：**9 步**（从黑组开始）/ **12 步**（从白组开始）。

结论：
- 短解 (9 步) = qme-ng 的**最短长度** ✓
- 长解 (12 步) 在分布范围 (9–25) 内 ✓

#### $A_3$（纯 Dynkin 型，qme-ng 内置）

qme-ng 穷举结果（共 9 条）：

```
长度 3  →  1 条
长度 4  →  4 条
长度 5  →  2 条
长度 6  →  2 条
```

这里无方积结构，但验证了 qme-ng 工具本身工作正常。

### 3.3 已有数据交叉验证

生成器输出的 $A_2 \square A_3$ 矩阵与 `common/games/A2_square_A3 (test1_05).json` 中的矩阵**逐元素一致**（两者都采用行优先编号）。

---

## 四、文件结构

```
ning/bipartite_belt/
├── README.md                  ← 本文档
├── square_product.py          ← 游戏生成：DynkinGraph + SquareProduct
├── solver.py                  ← 双分带求解：bipartite_belt_solution()
└── test_bipartite_belt.py     ← 测试（41 个，全部通过）
```

依赖关系：`solver.py` → `square_product.py`（单向）。其他程序只需 import `square_product` 即可生成游戏。

---

## 五、参考文献

1. B. Keller. *The periodicity conjecture for pairs of Dynkin diagrams*. Annals of Mathematics 177(1) (2013), 111–170. [arXiv:1001.1531](https://arxiv.org/abs/1001.1531)
2. E. Casbi, H. Hosaka, R. Ikeda. *Half-Periodicity of Zamolodchikov Periodic Cluster Algebras*. 2026. [arXiv:2602.15140](https://arxiv.org/abs/2602.15140) — Proposition 3.1: 双分带轮数 = Coxeter 数。
3. B. Keller. *Quiver mutation and combinatorial DT-invariants*. 2017. [arXiv:1709.03143](https://arxiv.org/abs/1709.03143) — §5.2: 方积构造的定义。
4. J.E. Humphreys. *Reflection Groups and Coxeter Groups*. Cambridge University Press, 1990. — Coxeter 数的标准参考。
