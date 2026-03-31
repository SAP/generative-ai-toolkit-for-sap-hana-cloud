# BAS 通过 JS 调用 hana_ai（Python）——概念架构图

> 目的：用框架/概念层面说明在 SAP Business Application Studio（BAS）中，JS/Node 如何驱动 Python（`hana_ai`）完成对 HANA 与大模型的访问。

## 图例（Legend）

- 方框：组件/运行时/模块
- 实线箭头 `-->`：调用/数据流方向
- 箭头标签：协议或传输方式（`HTTP` / `stdin&stdout(JSON)` / `SQL` 等）
- 分组（subgraph）：部署边界（BAS 容器内 vs 外部服务）

## 架构图（Mermaid）

```mermaid
%%{init: {"themeVariables": {"fontSize": "20px"}, "flowchart": {"nodeSpacing": 50, "rankSpacing": 60}}}%%
flowchart TB
  subgraph BAS["BAS Dev Space（同一开发容器/工作区）"]
    direction TB
    UI["JS/TS 调用方\n（UI/命令/脚本/扩展）"]
    NODE["Node.js 编排层\n（服务封装/SDK/路由）"]
    PY["Python Runner\n（脚本或模块：qa_bot.py 等）"]
    HAAI["hana_ai 包\nAgents / Tools / Memory"]
    HML["hana_ml 客户端\nConnectionContext / DataFrame"]
  end

  subgraph Platform["平台服务（容器外/云服务）"]
    direction TB
    HANA["SAP HANA Cloud"]
    LLM["GenAI Hub / LLM Proxy\n（init_llm / 模型推理）"]
  end

  UI -->|"调用：函数/命令/HTTP"| NODE

  NODE -->|"方式A：child_process\nstdin/stdout 传 JSON"| PY
  NODE -->|"方式B：HTTP/RPC 调用 Python 服务\n（同容器或远端）"| PY

  PY -->|"import & 调用"| HAAI
  HAAI -->|"数据访问/算法执行"| HML
  HML <-->|"SQL/结果集"| HANA

  HAAI <-->|"推理请求/响应"| LLM

  PY -->|"JSON 结果返回"| NODE
  NODE -->|"结果返回（UI/API）"| UI
```

## 最小链路（贴近 qa_bot.py 的常见落地）

```mermaid
%%{init: {"themeVariables": {"fontSize": "20px"}}}%%
sequenceDiagram
  participant UI as JS/TS
  participant NODE as Node.js
  participant PY as Python（qa_bot.py）
  participant HAAI as hana_ai
  participant HANA as HANA Cloud
  participant LLM as LLM Proxy

  UI->>NODE: 调用（命令/HTTP）
  NODE->>PY: 启动进程 + stdin(JSON)
  PY->>HAAI: stateless_call(question, chat_history, tools, llm)
  HAAI->>HANA: SQL/数据访问（via hana_ml）
  HAAI->>LLM: 推理请求（via init_llm）
  PY-->>NODE: stdout(JSON)
  NODE-->>UI: 返回结果
```
