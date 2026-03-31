# Green-Red Mutation Game — LLM Agent

让 LLM 自动玩绿红变换游戏（quiver mutation）。

游戏规则见 [graph_rule.md](../graph_rule.md)，矩阵表达见 [graph_matrix_rule.md](../graph_matrix_rule.md)。

## 安装

```bash
pip install -r requirements.txt
```

在项目根目录创建 `.env` 文件，配置 API key：

```
DEEPSEEK_API_KEY=sk-...
OPENAI_API_KEY=sk-...
```

## 运行

### 网页版（Streamlit）

```bash
streamlit run ning/agent/play_web.py
```

- 左侧选择 LLM 模型和图
- **Step**：单步执行，LLM 选一个顶点做 mutation
- **Auto-play**：连续执行直到游戏结束
- 图形面板支持浏览历史步骤（前进/后退/slider）

### 命令行版

```bash
# 默认：deepseek + linear_2
python -m ning.agent.play_cli

# 指定模型和图
python -m ning.agent.play_cli deepseek test1_07_n4

# 列出所有可用图和模型
python -m ning.agent.play_cli --list
```

## 添加新图

在 `games/` 目录下新建 JSON 文件：

```json
{
  "n": 3,
  "B_A": [
    [ 0,  1,  0],
    [-1,  0,  1],
    [ 0, -1,  0]
  ],
  "solution": [1, 2, 3]
}
```

- `n`：内点数量（冻结点数量相同）
- `B_A`：n x n 反对称交换矩阵
- `solution`（可选）：已知的 mutation 序列

文件名即 graph 名，保存后自动出现在 UI 下拉框中。

## 测试

```bash
python -m pytest ning/agent/tests/ -v
```

## 架构

```
ning/agent/
├── mutation.py      纯函数：矩阵变换、绿红判定
├── engine.py        有状态游戏引擎
├── catalog.py       图数据加载（扫描 games/*.json）
├── games/           图定义（JSON）
│
├── harness.py       engine 状态 <-> LLM 文本
├── llm_provider.py  LLM 接口抽象（OpenAI 兼容）
├── agent.py         无头游戏循环 run_game()
│
├── play_web.py      Streamlit 网页入口
├── play_cli.py      命令行入口
├── graph_viz.py     pyvis 图形渲染
│
└── tests/
    ├── test_engine.py   engine/mutation/catalog 测试
    ├── test_harness.py  harness 测试
    └── test_agent.py    agent 游戏循环测试
```
