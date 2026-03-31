# 绿红游戏 Agent 系统设计

## 一、目标

构建一个 agent 系统，让 LLM 自动玩绿红游戏：从 $\text{framed}(A)$（全绿）出发，通过选择内点执行 mutation $\mu_k$，到达全红状态。

游戏规则见 [graph_rule.md](graph_rule.md)。

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
├──┬──────┬──────┬──────┬─────┤
│DS│ GPT  │Claude│ Qwen │ ... │  具体 provider
└──┴──────┴──────┴──────┴─────┘
```

---

## 三、Engine 层

纯确定性计算，与 LLM 无关。

### 内部表示

- 用 $n \times m$ 反对称计数矩阵 $B$ 表示图（见 [graph_matrix_rule.md](graph_matrix_rule.md)）
- mutation 通过矩阵公式直接计算

### 对外接口

| 方法 | 说明 |
|------|------|
| `reset(A)` | 输入图 $A$，构造 $\text{framed}(A)$ 作为初始状态 |
| `mutate(k)` | 对内点 $k$ 执行 mutation，返回新状态 |
| `get_state()` | 返回当前矩阵、边列表、各内点的绿/红状态 |
| `is_won()` | 判定是否全红 |
| `get_history()` | 返回已访问状态的记录 |

### 内部职责

- 每步自动计算各内点的绿/红状态
- 维护历史记录（矩阵 hash），用于检测循环
- 合法性校验（只允许对内点做 mutation）

### 要求

- 独立于 Harness 和 Agent，可单独测试
- 用 `graph_rule.md` 中的示例（$1 \to 2 \to 3$ 对顶点 2 做变换）作为基本测试用例

---

## 四、Harness 层

连接 Engine 和 Agent，是设计的重心。

### 4.1 状态呈现

采用**混合呈现**策略：

- 主体：边列表（如 `1→2, 2→3`）
- 附加：每个内点的绿/红状态、度数信息
- 可选：矩阵 $B$（作为 agent 可请求的"思考工具"）

### 4.2 Feedback 设计

每步 mutation 后返回的信息分层：

| 层级 | 内容 |
|------|------|
| 必给 | 新的边列表 + 各顶点颜色 + 红色占比（如 `3/5`） |
| 必给 | 本步变化 diff（加了/删了/反转了哪些边） |
| 条件 | 循环警告："你回到了第 N 步的状态" |
| 条件 | 红色数量变化方向（`3/5 → 2/5`） |

设计原则：

- **diff 很重要**——让 LLM 追踪因果，而不是每次从头解读整张图
- **红色计数作为主要信号**，但不把它硬编码为奖励。不要在 prompt 中说"每步都要增加红色"，因为大部分简单 case 中红色数量确实单调增加，LLM 可以自己发现这个贪心规律；但复杂 case 中可能需要先退后进，不应限制这种决策
- **红色数量下降时给中性提示**（如 `红色: 3/5 → 2/5`），而非负面评价

### 4.3 循环与终止处理

- 回到已访问状态 → 明确告知 "你回到了第 N 步的状态"
- 步数上限（如 $n^2$ 或 $2n$），超过后终止并给出 summary
- 允许 undo（$\mu_k \circ \mu_k = \text{id}$，对同一顶点再做一次即可）

### 4.4 Prompt 结构

```
System Prompt:
  - 规则说明（精简版，非 graph_rule.md 全文）
  - 输出格式约束（输出一个顶点编号）

每轮 User Message:
  - 当前状态（边列表 + 颜色）
  - 上一步的 diff
  - 步数 / 历史摘要

Agent Response:
  - （可选）推理过程
  - 动作：mutate(k)
```

关键原则：**规则只在 system prompt 说一次，状态每轮更新**。不要每轮重复规则。

### 4.5 扩展：Lookahead 工具

对于复杂图，可以给 agent 提供 `simulate(k)` 工具——返回"如果对 $k$ 做 mutation 会怎样"的假设性结果，但不真正执行。这将搜索能力交给 agent 自己。

---

## 五、Agent 层

### 5.1 决策循环

第一版保持最简：

```python
while not engine.is_won() and steps < max_steps:
    prompt = harness.render_state(engine)
    action = llm.chat(prompt)            # 返回文本
    k = harness.parse_action(action)     # 解析 + 合法性校验
    if k is None:
        # 解析失败，给错误消息重试
        continue
    engine.mutate(k)
    steps += 1
```

### 5.2 LLM 适配层

统一接口：

```python
class LLMProvider:
    def chat(self, messages: list[dict]) -> str:
        """输入标准消息列表，返回文本"""
        ...
```

第一版使用 OpenAI SDK 的 `base_url` 切换方案，各家 provider 通过配置区分：

```python
from openai import OpenAI

providers = {
    "deepseek": {"base_url": "https://api.deepseek.com", "api_key": "..."},
    "qwen":     {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key": "..."},
    "claude":   {"base_url": "https://api.anthropic.com/v1/", "api_key": "..."},
    "gpt":      {"base_url": "https://api.openai.com/v1", "api_key": "..."},
}

client = OpenAI(**providers[model_name])
```

选择理由：够用、无额外依赖。等需要对比多模型表现时再考虑 litellm。

核心原则：**Agent 循环只看到 `llm.chat(messages) → str`，不知道底下是哪个模型**。换模型只改配置，不改逻辑。

---

## 六、实施顺序

1. **Engine**：矩阵表示 + mutation + 绿红判定 + 单元测试
2. **Harness**：状态渲染 + action 解析 + feedback 生成
3. **最简 Agent 循环**：用小图（3-4 个内点）跑通
4. **迭代优化**：根据 agent 实际失败模式调整呈现方式和 feedback
