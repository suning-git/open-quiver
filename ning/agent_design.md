# 绿红游戏 Agent 系统设计

## 一、目标

构建一个 agent 系统，让 LLM 自动玩绿红游戏：从 $\text{framed}(A)$（全绿）出发，通过选择内点执行 mutation $\mu_k$，到达全红状态。

游戏规则见 [graph_rule.md](graph_rule.md)，矩阵表达见 [graph_matrix_rule.md](graph_matrix_rule.md)。

---

## 二、整体架构

系统分为三层，各层职责严格分离：

```
┌─────────────────────────────┐
│  Agent (决策循环)            │  选哪个顶点 k 做 μ_k
├─────────────────────────────┤
│  Harness (协议层)            │  prompt 组装、状态呈现、feedback 生成
├─────────────────────────────┤
│  Engine (规则层)             │  图变换、合法性校验、胜负判定
└─────────────────────────────┘
```

Agent 层内部进一步分离出 LLM 适配层，以支持多模型切换：

```
┌─────────────────────────────┐
│  Agent (决策循环)            │  调用 llm.chat(messages) → 拿到动作
├─────────────────────────────┤
│  LLM Interface (适配层)      │  统一接口，屏蔽各家 API 差异
├──┬──────────┬──────────┬────┤
│DS│ GPT-5.4  │  ...     │    │  具体 provider
└──┴──────────┴──────────┴────┘
```

另外有一个独立于三层之外的**图数据管理**模块（Catalog），为整个系统提供图定义。

---

## 三、Engine 层

纯确定性计算，与 LLM 无关。实现在 `mutation.py`（纯函数）和 `engine.py`（有状态包装）。

### 内部表示

- 用 $n \times 2n$ 交换矩阵 $B = (B_0 \mid B_f)$ 表示图（内点数 = 冻结点数 = $n$）
- mutation 通过矩阵公式直接计算（见 [graph_matrix_rule.md](graph_matrix_rule.md)）
- `mutation.py` 中的纯函数可独立于 Engine 使用

### 对外接口

| 方法 / 属性 | 说明 |
|------|------|
| `reset(n, edges)` | 从边列表构造 $\text{framed}(A)$ 作为初始状态 |
| `reset_from_matrix(B_A)` | 从 $n \times n$ 交换矩阵构造初始状态 |
| `mutate(k)` | 对内点 $k$ 执行 mutation，返回新状态 + diff |
| `get_state()` | 返回当前矩阵、边列表、各内点的绿/红状态 |
| `get_state_at(step)` | 返回历史中第 step 步的状态（用于浏览回放） |
| `is_won()` | 判定是否全红 |
| `n` | 内点数量（属性） |
| `total_steps` | 已执行的 mutation 步数（属性） |

### 内部职责

- 每步自动计算各内点的绿/红状态
- 维护历史记录（矩阵 `tobytes()` hash），用于检测循环
- 合法性校验（只允许对内点做 mutation）
- `mutate()` 返回 diff：颜色变化、红色计数变化、循环警告

### 测试

- 独立于 Harness 和 Agent，可单独测试
- 基本测试：`graph_rule.md` 中的示例（$1 \to 2 \to 3$ 对顶点 2 做变换）
- 对合性：$\mu_k \circ \mu_k = \text{id}$
- 回归测试：ZhK/TestSamples 中的 10 顶点长序列
- Catalog 集成测试：所有带 solution 的图验证从全绿到全红

---

## 四、图数据管理（Catalog）

实现在 `catalog.py` + `games/*.json`。

### 设计原则

- 一图一 JSON 文件，文件名即 graph 名
- 只存最小必要信息：`n`（内点数）+ `B_A`（$n \times n$ 反对称矩阵）
- `solution`（可选）：已知的 mutation 序列，不暴露给玩家
- 新增图只需放文件，无需改代码

### JSON 格式

```json
{
  "n": 4,
  "B_A": [
    [ 0,  1,  0, -1],
    [-1,  0,  1,  0],
    [ 0, -1,  0,  1],
    [ 1,  0, -1,  0]
  ],
  "solution": [1, 2, 4, 3, 2, 1]
}
```

### 对外接口

| 方法 | 说明 |
|------|------|
| `list_graphs()` | 返回所有可用图，按 $n$ 排序 |
| `get_graph(name)` | 返回 `{"n", "B_A"}`，不含 solution |
| `get_solution(name)` | 返回 mutation 序列，或 None |

---

## 五、Harness 层

连接 Engine 和 Agent，是设计的重心。实现在 `harness.py`。

### 5.1 状态呈现

采用**混合呈现**策略：

- 主体：内点间的边列表（如 `1→2, 2→3`），冻结边不展示
- 附加：每个内点的绿/红标记（如 `1(G) 2(R) 3(G)`）、红色占比

### 5.2 Feedback 设计

每步 mutation 后返回的信息分层：

| 层级 | 内容 |
|------|------|
| 必给 | 新的边列表 + 各顶点颜色 + 红色占比（如 `3/5`） |
| 必给 | 颜色变化（如 `Vertex 2: green → red`） |
| 必给 | 红色数量变化（如 `Red: 0/3 → 1/3`） |
| 条件 | 循环警告："你回到了第 N 步的状态" |

设计原则：

- **diff 很重要**——让 LLM 追踪因果，而不是每次从头解读整张图
- **红色计数作为主要信号**，但不把它硬编码为奖励。不要在 prompt 中说"每步都要增加红色"，因为大部分简单 case 中红色数量确实单调增加，LLM 可以自己发现这个贪心规律；但复杂 case 中可能需要先退后进，不应限制这种决策
- **红色数量下降时给中性提示**（如 `Red: 3/5 → 2/5`），而非负面评价

### 5.3 循环与终止处理

- 回到已访问状态 → 明确告知 "你回到了第 N 步的状态"
- 步数上限（默认 50，可配置），超过后终止
- 解析失败最多重试 3 次，仍失败则终止
- 允许 undo（$\mu_k \circ \mu_k = \text{id}$，对同一顶点再做一次即可）

### 5.4 Prompt 结构

```
System Prompt:
  - 规则说明（精简版，非 graph_rule.md 全文）
  - 输出格式约束（输出一个顶点编号）

每轮 User Message:
  - 上一步的 diff（首轮无）
  - 当前状态（边列表 + 颜色）
  - "Your move?"

Agent Response:
  - （可选）推理过程
  - 最后一个整数作为动作
```

关键原则：**规则只在 system prompt 说一次，状态每轮更新**。不要每轮重复规则。

### 5.5 扩展：Lookahead 工具（未实现）

对于复杂图，可以给 agent 提供 `simulate(k)` 工具——返回"如果对 $k$ 做 mutation 会怎样"的假设性结果，但不真正执行。这将搜索能力交给 agent 自己。

---

## 六、Agent 层

### 6.1 决策循环

实现在 `agent.py`。核心是 `_run_game_loop`：

```python
while not engine.is_won() and state["step"] < max_steps:
    response = provider.chat(messages)
    k = parse_action(response, n)
    # 解析失败 → 给错误消息重试（最多 max_retries 次）
    if k is None:
        return GameResult(won=False, reason="parse_failure")
    state = engine.mutate(k)
    # 构建下一轮消息：diff + 当前状态
```

两个入口函数：
- `run_game(n, edges, provider)` — 从边列表开始
- `run_game_from_matrix(B_A, provider)` — 从交换矩阵开始（配合 Catalog 使用）

### 6.2 LLM 适配层

统一接口：

```python
class LLMProvider(ABC):
    def chat(self, messages: list[dict]) -> str: ...
```

使用 OpenAI SDK 的 `base_url` 切换方案，各家 provider 通过配置区分。当前支持：

- **DeepSeek**（deepseek-chat）
- **OpenAI GPT-5.4 系列**（gpt-5.4 / gpt-5.4-mini / gpt-5.4-nano）

核心原则：**Agent 循环只看到 `llm.chat(messages) → str`，不知道底下是哪个模型**。换模型只改配置，不改逻辑。

---

## 七、入口与 UI

### 7.1 网页版（play_web.py）

基于 Streamlit，功能：

- 从 Catalog 选图、选 LLM 模型
- **Step**（单步）/ **Auto-play**（连续执行）
- 左侧：图形可视化（pyvis），支持**历史浏览**（前进/后退/slider）
- 右侧：LLM 对话记录
- 图形浏览与 LLM 对话解耦，通过 moves history 松耦合

### 7.2 命令行版（play_cli.py）

终端运行一局完整游戏，打印对话过程。支持 `--list` 列出所有图和模型。

用途：快速测试、批量评估。

---

## 八、实施状态

| 阶段 | 内容 | 状态 |
|------|------|------|
| 1 | Engine：矩阵表示 + mutation + 绿红判定 + 测试 | ✅ 完成 |
| 2 | Harness：状态渲染 + action 解析 + feedback 生成 | ✅ 完成 |
| 3 | Agent 循环 + LLM 适配层 | ✅ 完成 |
| 4 | Catalog + 图数据管理 | ✅ 完成 |
| 5 | Streamlit UI + 历史浏览 | ✅ 完成 |
| 6 | Lookahead 工具 | 🔲 未实现 |
| 7 | 迭代优化：根据 agent 实际失败模式调整 prompt 和 feedback | 🔲 进行中 |
