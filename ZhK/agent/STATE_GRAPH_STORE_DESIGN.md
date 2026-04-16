# 状态图存储设计（State Graph Store）

## 1. 背景

当前 `ZhK/agent` 的游戏执行是线性轨迹：

- 任意时刻只有一个当前状态。
- 仅维护一条 `move_history`。
- 恢复能力主要依赖重试解析与可选 undo。

这会导致一个常见问题：模型一旦走偏，系统缺少“分支记忆”，难以稳定回到更优路径。为了解决这个问题，我们需要把状态存储升级为无向图（搜索视角近似树，无向图是因为 mutation 的对合性）。

## 2. 目标

1. 用无向状态图替代线性历史。
2. 将决策拆成两个模型角色：
   - 状态选择器（State Selector）：决定“下一步从哪个状态继续搜索”。
   - 变换选择器（Mutation Selector）：决定“在该状态下 mutate 哪个顶点”。
3. 把“恢复能力”做成系统能力：
   - 错误分支保留但降权。
   - 可随时跳回旧分支继续探索。
4. 复用现有 `common/quiver/mutation.py` 数学原语。
5. 采用渐进迁移，尽量不破坏现有 CLI/Web 入口。

## 3. 非目标（Phase 1）

- 不做在线训练（policy/critic 联训）。
- 不做分布式并行搜索。
- 不改 quiver 变换数学实现。
- 不绑定某个特定 provider/model。

## 4. 总体架构

### 4.1 组件

1. `StateGraphStore`（新）
   - 管理节点、边、索引、统计量。
   - 提供增量写入、去重合并、查询与更新接口。

2. `DualAgentSearchRunner`（新）
   - 执行搜索主循环：Selection -> Expansion -> Backup。

3. `StateSelectorPolicy`（新）
   - 从候选状态中给出排序与选择。
   - 可混合启发式 + LLM。

4. `MutationSelectorPolicy`（新）
   - 对选中状态输出 mutation 顶点（或候选列表）。
   - 输出协议严格化，降低解析失败。

5. 现有变换原语（复用）
   - `mutate`, `get_colors`, `is_all_red`, `matrix_to_edges`。

### 4.2 与 HTPS 的映射

保留 HTPS 思想：

- Selection / Expansion / Backup 三段式循环。
- 统计量（`N`, `W`, `Q`）驱动探索与利用。
- 节点状态传播（`solved / invalid / unsolved` 思想）。

项目内改造：

- 本任务每个动作只产生一个后继状态（非多子目标超边）。
- 回传采用路径价值更新，不使用 AND 子目标乘积结构。
- 通过矩阵哈希做 DAG 合并。

## 5. 数据模型

### 5.1 节点（Node）

每个节点对应唯一矩阵状态。

建议字段：

- `state_id: str`
- `matrix: np.ndarray`（`n x 2n`）
- `n: int`
- `red_count: int`
- `is_won: bool`
- `colors: dict[int, str]`
- `created_iter: int`
- `visit_count: int`
- `value_sum: float`
- `value_mean: float`
- `status: str`（`unexpanded | expanded | solved | dead_end | invalid`）
- `best_incoming_edge_id: str | None`
- `metadata: dict`

### 5.2 边（Edge）

每条边表示一次从源状态发起的 mutation 尝试。

建议字段：

- `edge_id: str`
- `from_state_id: str`
- `action_vertex: int`
- `to_state_id: str | None`
- `status: str`（`proposed | executed | parse_failed | invalid_action`）
- `delta_red: int | None`
- `created_iter: int`
- `source: str`（`mutation_selector` 或 fallback）
- `selector_score: float | None`
- `visit_count: int`
- `value_sum: float`
- `value_mean: float`
- `llm_payload: dict`

### 5.3 索引

- `nodes_by_id`
- `edges_by_id`
- `out_edges_by_state`
- `in_edges_by_state`
- `state_id_by_matrix_hash`

### 5.4 搜索游标

- `root_state_id`
- `active_state_id`
- `best_solution_state_id`
- `iteration`
- `expanded_node_count`

## 6. 核心算法

### 6.1 Selection（选状态）

输入：候选状态集合。

输出：`selected_state_id`。

策略建议：

1. 先做启发式 Top-K 预筛选：
   - `value_mean` 高。
   - 扩展次数少（保留探索）。
   - 可加入 `red_count` 偏好。
2. 状态选择器 LLM 在 Top-K 中决策。
3. 终分数融合：
   - 先验分数 + UCB 探索项 + 重复低收益惩罚。

兜底：LLM 输出不可解析时，回退到确定性启发式。

### 6.2 Expansion（扩展）

输入：已选状态。

输出：1 条或多条边（Phase 1 建议先单边）。

流程：

1. Mutation Selector 提议 `k`。
2. 校验 `k ∈ [1..n]`。
3. 调 `mutate` 得到后继矩阵。
4. 按哈希去重：
   - 已存在则连边。
   - 不存在则建新节点。
5. 更新状态：
   - `is_won=True` 直接标记 `solved`。
   - 自环/重复可降权，不强行丢弃。

兜底：多次解析失败则记录 `parse_failed` 边并继续搜索。

### 6.3 Backup（回传）

输入：本次选择路径（至少包含源节点、边、目标节点）。

更新：

- 边：`visit_count`, `value_sum`, `value_mean`。
- 节点：`visit_count`, `value_sum`, `value_mean`。
- 可选：对祖先路径做递归回传。

Phase 1 奖励建议：

- 到达解态：`+1.0`
- 进度奖励：`+alpha * delta_red`
- 解析失败/非法动作：小负奖励
- 低价值回环：附加惩罚

### 6.4 状态维护

节点状态迁移：

- `unexpanded -> expanded`：出现第一条有效出边。
- `* -> solved`：`is_won=True`。
- `expanded -> dead_end`：预算耗尽且所有出边长期低收益。

Phase 1 不要过早宣判 dead_end，避免误杀可行分支。

## 7. 双模型协议

### 7.1 状态选择器输出协议

输入建议包含：

- 候选 state 列表。
- 每个 state 的压缩特征：`red_count/value_mean/visits/深度/路径后缀`。

输出（严格 JSON）：

```json
{
  "selected_state_id": "s_123",
  "confidence": 0.0,
  "reason": "..."
}
```

### 7.2 变换选择器输出协议

输入建议包含：

- 选中状态的完整渲染（可复用 `render_state` 风格）。
- 可选：当前最佳路径最近若干步。

输出（严格 JSON）：

```json
{
  "action_vertex": 3,
  "confidence": 0.0,
  "reason": "..."
}
```

Phase 2 可扩展：输出 `candidates: [k1, k2, ...]`。

## 8. 与现有模块集成

### 8.1 稳定保留

- `engine.py` 继续服务线性模式。
- `game_turn_runner.py` 作为基线保留。
- `play_web.py` 增加模式切换：
  - `linear`
  - `graph_search`

### 8.2 新增模块（建议）

- `ZhK/agent/state_graph_store.py`
- `ZhK/agent/dual_agent_runner.py`
- `ZhK/agent/state_selector.py`
- `ZhK/agent/mutation_selector.py`
- `ZhK/agent/search_config.py`

### 8.3 迁移路径

1. 先做 Store + Runner（脱离 Web）。
2. 先接 CLI 入口验证行为。
3. 再接 Web（先只读展示，再加交互）。

## 9. 持久化与回放

### 9.1 保存格式

JSON 包含：

- metadata（图名、provider、配置、时间戳）
- nodes（矩阵与统计）
- edges（完整转移记录）
- root/best-solution 指针

### 9.2 回放能力

- 可完整重建状态图。
- 可提取最佳路径。
- 可查看任意分支，不需重算 mutation。

## 10. 失败处理

1. 解析失败：记录失败边，继续搜索，不整局终止。
2. 非法顶点：记录 `invalid_action`，有限重试。
3. Provider 超时：退避并临时启发式兜底。
4. 预算耗尽：返回最佳已知路径和诊断信息。

## 11. 指标

- `won`
- `iterations_to_win`
- `nodes_created`
- `edges_created`
- `selector_parse_fail_rate`
- `mutator_parse_fail_rate`
- `state_revisit_ratio`
- `best_red_count_reached`
- `solution_path_length`

并与线性基线做对比。

## 12. 测试策略

单测（Store）：

1. 矩阵哈希去重。
2. 多父节点合并。
3. 边与索引一致性。
4. 统计量更新正确性。
5. 状态迁移正确性。

Runner 测试：

1. Selector 非法输出回退。
2. Mutator 非法输出回退。
3. parse 失败不终止整局。
4. 坏分支后可切换分支恢复。
5. DAG 中正确提取解路径。

集成测试：

1. `linear_2`、`linear_3` 等小图在预算内可解。
2. save/load 往返后图与关键指标一致。

## 13. 分阶段落地

Phase 1：Store + 单模型搜索

- 先完成 `StateGraphStore`。
- 先用一个模型串行承担选择与变换。
- 验证“可恢复”能力。

Phase 2：真正双模型

- 拆分 selector/mutator prompt 与 provider。
- 引入更稳定的 UCB 融合与回传策略。

Phase 3：Web 集成

- 分支可视化。
- 最佳路径展示。
- 模式切换与回放。

Phase 4（可选）：训练扩展

- 从成功分支采样监督数据。
- 引入 critic 校准。

## 14. 待决策问题

1. selector 与 mutator 默认是否分模型？
2. Phase 1 扩展单动作还是 top-K？
3. 回环惩罚强度如何设定？
4. Web 默认预算如何兼顾效果与延迟？

## 15. 我建议的第一步

最优先做 **`StateGraphStore` 最小可用实现 + 单元测试**。

原因：

1. 这是后续双模型和搜索循环的共同地基。
2. 先把“节点去重、连边、统计更新、路径提取”做稳，后面替换策略成本很低。
3. 可以在不改 Web 的情况下，用 CLI/测试先验证“错误后可恢复”的核心价值。

建议最小范围：

- 只实现 Node/Edge 插入与哈希去重。
- 只实现基本统计更新（`visit_count/value_sum/value_mean`）。
- 只实现 `extract_best_path()`。
- 配套 5~8 个高价值单测。
