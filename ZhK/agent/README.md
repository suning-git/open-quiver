# Green-Red Mutation Game - LLM Agent

让 LLM 自动玩绿红变换游戏（quiver mutation）。

游戏规则见 [graph_rule.md](../graph_rule.md)，矩阵表达见 [common/quiver/MATH.md](../../common/quiver/MATH.md)。

## 安装

```bash
pip install -r requirements.txt
```

在项目根目录创建 `.env` 并配置 API Key：

```env
DEEPSEEK_API_KEY=sk-...
OPENAI_API_KEY=sk-...
```

## 运行

### 网页版（Streamlit）

```bash
streamlit run ZhK/agent/play_web.py
```

- 左侧选择 LLM provider 和图
- 支持模式切换：`Linear (legacy)` 与 `Graph Search (new)`
- `Step`：单步执行
- `Auto-play`：连续执行直到游戏结束
- 支持模型输出 `undo` 撤销最近一步
- 图面板支持历史浏览（前进、后退、slider）

### 命令行版

```bash
# 默认：deepseek-chat + linear_2
python -m ZhK.agent.play_cli

# 指定 provider 和图
python -m ZhK.agent.play_cli deepseek-chat test1_07_n4

# 列出可用图和 provider
python -m ZhK.agent.play_cli --list
```

## 图数据

图定义在共享目录 `common/games/`，不再从本目录读取。新增图请在该目录新增 JSON 文件。

格式说明见 [common/games/README.md](../../common/games/README.md)。

## 测试

```bash
python -m pytest ZhK/agent/tests/ -v
```

## 目录结构

```text
ZhK/agent/
├── engine.py              有状态游戏引擎（包装 common.quiver.mutation）
├── catalog.py             图数据加载（扫描 ../../common/games/*.json）
├── harness.py             engine 状态 <-> LLM 文本
├── initial_prompts.py     prompt 注册表（默认保留 undo 版本）
├── graph_search_initial_prompts.py graph-search 专用 prompt（含模式概要）
├── llm_provider.py        LLM 接口抽象（OpenAI 兼容）
├── provider_registry.py   provider 配置
├── game_turn_runner.py    单步执行器（支持 undo）
├── game_session_runner.py 完整游戏循环
├── state_graph_store.py   图搜索状态存储（无向边）
├── dual_agent_runner.py   图搜索增量执行器（Selection/Expansion/Backup）
├── graph_search_harness.py 图搜索模式下的 UI 文本/渲染辅助
├── play_web.py            Streamlit 入口
├── play_cli.py            CLI 入口
├── graph_viz.py           pyvis 图渲染
└── tests/
```

## 说明

- 核心数学原语在 `common/quiver/`，`ZhK/agent/mutation.py` 已不再作为主实现依赖。
- 本分支保留了 `undo` 能力（prompt、解析、turn runner、engine 全链路支持）。
