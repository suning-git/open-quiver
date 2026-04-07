# Green-Red Mutation Game — LLM Agent

让 LLM 自动玩绿红变换游戏（quiver mutation）。

游戏规则见 [graph_rule.md](../graph_rule.md)，矩阵表达见 [common/quiver/MATH.md](../../common/quiver/MATH.md)。

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

图定义存放在共享目录 `common/games/`（不在本目录下）。新增图就在那里放一个 JSON 文件，UI 下拉框会自动发现。格式说明见 [common/games/README.md](../../common/games/README.md)。

> 注意：`common/` 是跨项目共享的，修改已有文件需要协调；新增文件一般无需协调。

## 测试

```bash
python -m pytest ning/agent/tests/ -v
```

## 架构

```
ning/agent/
├── engine.py             有状态游戏引擎（包装 common.quiver.mutation）
├── catalog.py            图数据加载（扫描 ../../common/games/*.json）
│
├── harness.py            engine 状态 <-> LLM 文本
├── initial_prompts.py    prompt 注册表
├── llm_provider.py       LLM 接口抽象（OpenAI 兼容）
├── provider_registry.py  provider 配置
├── game_turn_runner.py   单步游戏循环
├── game_session_runner.py 完整游戏循环
│
├── play_web.py           Streamlit 网页入口
├── play_cli.py           命令行入口
├── graph_viz.py          pyvis 图形渲染
│
└── tests/
    ├── test_engine.py        engine/catalog 测试
    ├── test_harness.py       harness 测试
    ├── test_game_runner.py   single-turn runner 测试
    ├── test_agent.py         session runner 测试
    └── integration/          真实 LLM 集成测试（默认跳过）
```

依赖的公共库见 [common/quiver/](../../common/quiver/)（纯数学原语）。
