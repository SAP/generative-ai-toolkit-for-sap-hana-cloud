# Assessment 13 安全修复、兼容性与验证报告

> 整理日期：2026-10-27；更新日期：2026-10-28
>
> 代码基线：`main @ e8e65a3`（已包含上游 PR #161）
>
> 评估材料：本地 `13-hana-ml` 扫描包（原始扫描材料不纳入提交）。
>
> 当前状态：`review8.md` 修复及二次核对已通过 `XinKey2` E2E 验证，并提交、推送到 `security/assessment-13-fixes`（PR #162，`58698ad`）；`review9.md` 的兼容性修复与新增 JOIN 边界修复已通过全量 `XinKey2` 验证（362 passed、13 skipped），并提交、推送到同一 PR（`5c146c3`）
>
> PR：[SAP/generative-ai-toolkit-for-sap-hana-cloud#162](https://github.com/SAP/generative-ai-toolkit-for-sap-hana-cloud/pull/162)
>
> 本次 PR 交付纳入本报告及 `security_feedback_2026-10-27.md`；原始 `review*.md`、一次性 probe、notebooks 和生成工件仍不纳入提交。各历史章节的“未提交/仅本地”描述只代表当时状态。
> 2026-10-27 补充验证：同一 `5c146c3` 启用全部 live opt-in 开关重跑后为 **362 passed、11 setup errors、2 skipped**（pytest 退出码 1）。11 项错误共用的 fixture 硬编码 `RaysKey`；2 项缺少 VCAP fixture。详见第十节，不能将此轮描述为全部通过。
> 最近全量基线：2026-10-28 使用 PyPI `hana-ml==2.30.26091800` 全量重跑为 **361 passed、1 failed、11 setup errors、2 skipped**（退出码 1）。新增 stdio 初始化超时在单独复跑时通过，但保留原始失败结果，详见第十一节。
> 本次交付：扫描补丁复核、memory/stdio 兼容性修复及 XinKey2/TLS test fixture 更新，见第十二至十五节。安全/兼容性扩大定向验证 **442 passed、0 failed/errors/skipped**；随后 13 项 live ContextAgent 重跑为 **2 passed、9 failed、0 setup errors、2 skipped**。不得描述为全套测试全通过；定向结果不替代旧全量基线。

> review10 后续修复：WITH HINT 仅允许位于外层 SELECT 末尾，详见第十七节；定向验证 **487 passed、0 failed/errors/skipped**（含 **50 live HANA**）。本轮修复基于 `d8d4714`，不替代此前 ContextAgent 的失败记录。

## 一、审阅结论

本次工作区已经覆盖评估包中 **13/13 个 patch 的安全意图**（包含适配和更严格的后续实现，不是原样重复应用旧 diff），并对 V-155～V-168 的全部 Critical/High 项增加或完善了代码处理。现成 patch 不足以覆盖的 V-162、V-163、V-164、V-165、V-168 采用了手工实现；V-155、V-157、V-159、V-160、V-161、V-166 又在 patch 基础上继续加固。此结论不等于原扫描每条验收勾选项均已字面满足，也不代表所有危险 opt-in 配置都安全。

当前修改在大多数 Python 调用签名上保持兼容，但整体不能承诺完全向后兼容。默认安全边界、协议、输出路径、依赖安装和旧数据加载行为都发生了有意变化，建议作为带迁移说明的安全版本发布。

`review.md` 后续指出的 V-169、V-171、V-174、V-175、V-176 缺口及同类 SQL/TLS/MCP 问题也已处理。审阅后新增的关键变化包括：memory 全链路强制 tenant/session 范围、SQL agent schema/table allowlist、删除 DBAPI repr 身份解析、GitHub Actions 哈希锁、全部 workflow action SHA 固定、预测 SQL 余留 sink 修复、空 TLS 环境变量安全默认，以及非 loopback HTTP MCP 强制认证。`review2.md` 关于目标环境锁和数据库授权边界的原则已采纳；其中“仓库没有锁”和“代码没有对象 allowlist”是基于旧状态的描述，不适用于当前工作区。`review3.md` 确认的 SQL allowlist 绕过、标准账号密码连接身份回归、FAISS 重启、SSE token、HTML 自动渲染和 session 名碰撞也已修复并增加回归测试。`review4.md` 确认的带引号标识符大小写绕过和非回环 SSE 暴露同样成立；现已按 HANA 标识符语义区分大小写，并将 SSE 限制在 loopback。其余 callable 身份、闲置 aiohttp session 和 MCP audit 参数错误也一并修复。`review5.md` 确认 22 项 finding 已在默认暴露面闭环；其新发现的 PSE、REMOTE SOURCE 和 CERTIFICATE SQL 拼接已作为额外加固修复，对象名统一安全引用，证书和配置值统一编码为单个 literal。其后续发现的 df_tools 堆叠 SQL 绕过也已修复：SmartDataFrame、全部 6 个 df tool 模块和 MCP `SelectStatementToTableTool` 在调用 `ConnectionContext.sql()` 前共用同一个只读 scanner。`review7.md` 进一步发现 hana-ml 对引号内分号仍使用原始 `split(";")`；共享 guard 现在只允许一个输入末尾分号并在调用前移除，其余任何位置的分号一律拒绝。`review8.md` 证明该防护尚未覆盖向量、数据拆分、时序和临时表 builder，并证明原 SQL-agent 正则 allowlist 可被括号 table factor、嵌套 CTE、UDF 和 sequence 绕过；当前工作区已统一保护这些 hana-ml sink，并用作用域感知、fail-closed 的对象解析器替换正则授权器。

不能把结果描述为“所有风险在任何配置下都消失”：

- V-156 的 `python_hanaml_exec` 和 V-168 的 Python REPL 都是主机代码执行能力。默认暴露面已经关闭；主机若显式启用危险模式，仍必须只接收完全可信代码和调用者，它不是沙箱。
- V-159 的 SQL 解析器是代码层只读护栏，生产连接仍应使用只读、限 schema 的 HANA 用户。
- V-157/V-158/V-166 所用的 hana-ml API没有发现明确的 bind-parameter 接口，因此当前实现采用严格类型、函数 allowlist、标识符引用和 HANA 单引号转义；不能写成“已经使用数据库参数绑定”。
- V-161 仍保留显式加载可信 FAISS pickle 的迁移开关。默认拒绝反序列化，并且不会覆盖已有或不完整的索引。

## 二、Critical/High 最终修改

| ID | 评级 | 本次代码处理 | 兼容性与剩余边界 |
|---|---|---|---|
| V-155 | Critical | `admin_update_connection_context` 和 env reload 默认不注册；只有 `HANA_ALLOW_RUNTIME_CONN_UPDATE=true` 才注册。HTTP 开启管理工具时强制要求 `auth_token`，通过 FastMCP `StaticTokenVerifier` 真正执行 Bearer 认证；SSE 拒绝管理面及 token，并且只能绑定 loopback；stdio 明确忽略 token 并依赖受信任主机进程边界；关闭 TLS 或证书校验还需 `allow_tls_downgrade=True`。 | loopback HTTP 为本地开发保留无 token 模式；launcher 本身不终结 TLS，远端访问必须使用带 Bearer 认证的 HTTP transport、HTTPS 反向代理并阻止直接明文访问，不能使用 SSE。 |
| V-156 | High | 保留 PR #161 的 fail-closed 边界：危险执行工具不在默认 toolkit 中，单独构造默认也拒绝执行，工具调用参数不能自行开启。文档明确说明它不是沙箱。 | 显式 `allow_dangerous_code=True` 仍等于接受宿主文件、网络和数据库权限；只允许可信主机代码启用。 |
| V-157 | High | `top_n` 强制整数并限制为 1～1000；距离函数使用 allowlist；schema/table/column 安全引用；query 和 `model_version` 按 HANA 规则编码为单个字符串 literal。 | 标识符加双引号后大小写语义更严格；需在真实 HANA 上确认混合大小写对象。当前不是 DB bind parameter。 |
| V-158 | High | retrieval procedure 的 schema/procedure 分组件引用，拒绝空值、双引号和 NUL；同时修复额外发现的 mem0 export `table_name`，以及 PSE、REMOTE SOURCE、CERTIFICATE 创建/删除 SQL 的标识符和值注入。基础设施对象名按 HANA 未加引号/显式引号语义规范化，证书和远端配置编码为单个 literal。 | 普通对象名继续按 HANA 规则折叠为大写；混合大小写对象需传显式双引号名称。非法标识符现在 fail closed。 |
| V-159 | High | SmartDataFrame 最终输出以及执行工具均只接受完整的一条 `SELECT`/`WITH`；共享 scanner 在任何 `ConnectionContext.sql()` 调用前解析 CTE 顶层动词并只允许 `SELECT`，拒绝说明文字、SQL 注释、堆叠语句、DML/DDL、`SELECT INTO` 和 `FOR UPDATE`。鉴于 hana-ml 会对原始文本执行 `split(";")`，输入仅允许一个结尾分号并在调用前删除；字符串 literal 或双引号标识符内的分号也拒绝。覆盖 fetch、可视化、异常检测、三类时序预测工具及 MCP `SelectStatementToTableTool`。 | 含分号的合法字符串值也会被 fail closed，这是阻断 hana-ml 非引号感知切分所需的兼容性变化。`SelectStatementToTableTool` 仍会按设计写入目标表，但其输入查询不能包含写操作或堆叠语句。仍需数据库只读/限权角色作为第二道边界。 |
| V-160 | High | 删除 nltk、mcp、fastmcp、langgraph 的 import-time `pip install`；缺依赖时明确失败。`nltk` 改成 `count_tokens()` 内延迟导入，增加 `nlp`/`graph` extras，toolkit 在无 nltk 时可以导入。 | 缺少可选依赖时不再自动联网安装。`fastmcp` 是核心 requirements，并传递安装 `mcp`。 |
| V-161 | High | FAISS 默认路径迁到 `~/.hana_ai/chat_history_vectorstore` 或 `HANA_AI_DATA_DIR`；应用自行保存的 store 同时写入 SHA-256 完整性清单，重启时校验完全一致才加载；缺失/不匹配清单和不完整 store 默认拒绝且不覆盖。 | 旧 store 没有清单，需要独立验证后用危险反序列化开关信任一次；成功加载后记录当前文件摘要。清单依赖私有 0700 目录隔离，不能防御已经拥有同一 OS 账号写权限的攻击者。 |
| V-162 | High | 所有报告/工件默认目录改为 `tempfile.mkdtemp()` 随机私有目录并尝试设置 0700；注册进程退出清理；跨平台净化 `/`、`\`、`..` 和特殊文件名。 | 默认输出路径从固定目录变为随机目录。调用方应读取工具返回路径，不能写死临时目录。 |
| V-163 | High | AI Core 证书和私钥临时文件的删除移入 `finally`，覆盖 HTTP、状态码和 JSON 解析异常路径。 | API 行为不变；已用请求异常测试确认两个临时文件均被删除。 |
| V-164 | High | HTTP 客户端禁止自动重定向；保留可工作的 loopback `http://localhost` 默认；非回环明文 HTTP 默认拒绝，只有显式 `allow_insecure_http=True` 才允许；HTTPS正常允许。 | 合法 30x 不再自动跟随；远端 MCP 应迁移到 HTTPS。本地默认客户端与默认服务端仍可互通。 |
| V-165 | High | stdio 默认禁止启动进程；必须配置 `HANA_MCP_STDIO_ALLOWED_PROFILES`，匹配解析后的 executable 和完整参数。shell、inline code flag、错误参数前缀全部拒绝；额外参数只有 host profile 明确设置 `allow_additional_args: true` 才允许。 | 原有任意 `python`/`uvx`/`npx` 配置不再直接运行，部署需登记受信任 server profile。 |
| V-166 | High | group-key 字符串值按 HANA 规则把 `'` 编码成 `''`；group/key 标识符安全引用；数值 token 严格校验；日期值也作为单个 SQL literal。 | 常规值保持语义，含单引号的合法分组值现在能正确表达。需真实 hana-ml `filter()`/SQL 回归。 |
| V-167 | Medium / 上下文 High | `HANA_SSL_VALIDATE` 默认值改成 `true`，stdio/HTTP 示例和知识库示例同步使用安全默认值。 | 缺少 CA/trust store 的环境会连接失败；只能显式且可审计地 opt out。 |
| V-168 | High | dataframe agent 默认不再导入、创建或添加 `PythonAstREPLTool`；没有显式安全工具时拒绝构造。增加 host-side `allow_dangerous_code=False`；prompt 只包含列名 schema，不再执行 `df.head(1).collect()` 或放入首行数据。 | 依赖默认 Python REPL 的旧调用方必须迁移到专用工具；危险 opt-in 仅用于完全可信数据和调用者。 |

## 三、其余 V-169～V-176 的当前状态

| ID | 当前处理 | 尚未声称完成的部分 |
|---|---|---|
| V-169 | Mem0 和旧 RAG agent 强制显式 `tenant_id`、`session_id`；添加、搜索、按标签检索、过期清理、导出和清理都合并不可覆盖的范围过滤器。`clear_all()` 只删除当前 tenant/session。 | 这是有意的构造器兼容性变化；调用方必须从已认证应用上下文传入稳定 tenant/session，不能使用用户可伪造值。 |
| V-170 | 聊天中的 `!` 内存管理命令默认禁用，需 `HANA_MEM0_ALLOW_ADMIN_COMMANDS=true`。 | 开启后仍应由应用层认证和授权管理接口保护。 |
| V-171 | SQL agent 要求 `schema` 或 `include_tables`，反射和执行阶段都限制对象；scanner 同时处理字符串与双引号标识符，解析 CTE 后顶层动词，仅允许 `SELECT`。未加引号的标识符按 HANA 规则折叠为大写，带引号标识符保留精确大小写，避免 `SENSITIVE` 错误放行不同对象 `"sensitive"`。仅表名 allowlist 会拒绝其他 schema 的同名表，schema-qualified 对象必须完整列入 allowlist。 | 数据库只读、限 schema 角色是部署配置，代码无法证明或创建生产授权；仍必须作为第二道边界。复杂 SQL 解析继续采取 fail-closed。 |
| V-172 | `session_id` 净化后统一附加原始值的 SHA-256 短摘要，并限制名称长度；resolve 后验证目录仍在 `<storage>/sessions`。这同时避免净化碰撞和 Windows 大小写碰撞。 | 所有既有 session 都会映射到带摘要的新目录名，需要迁移旧本地会话文件。 |
| V-173 | notebook 与 `TimeSeriesDatasetReport` 的直接显示路径默认都不执行原始 HTML；只有 `HANA_ALLOW_RAW_HTML_RENDER=true` 才渲染。 | opt-in 后仍会执行 HTML/JavaScript，只能处理可信工件。 |
| V-174 | 优先使用 `userkey`、结构化连接 `user`/`get_user()`；结构化 `user` 可以是属性或无参方法；标准 `ConnectionContext(user=..., password=...)` 通过常量查询 `SELECT CURRENT_USER FROM DUMMY` 获取认证身份。已删除 repr 解析。 | 无结构化字段且不能执行身份查询的离线/旧 wrapper 会 fail closed，需要适配。 |
| V-175 | CI/开发使用的 `requirements.txt` 保持精确直接依赖版本；GitHub Actions 使用 Python 3.11/Linux 专用、带 hash 的完整传递依赖锁。发布元数据使用兼容范围；build backend 固定为已验证的 setuptools/wheel 版本，`sap-ai-sdk-gen[all]` 限定在与 LangChain 0.3 兼容的 5.x 范围。 | 仍只为实际 GitHub Actions 目标锁定 Python 3.11/Linux；通用 `requirements.txt` 不冒充跨平台 hash lock。因此 V-175 的跨平台供应链范围仍为部分修复。 |
| V-176 | pylint/release 的 checkout、setup-python、PyPI publish action 全部固定 commit SHA；pylint、setuptools、wheel、build、pkginfo、twine 及传递依赖均由 hash lock 安装；release 使用 `build --no-isolation`。 | 依赖升级需重新生成并审阅两个 GitHub Actions 专用锁文件。 |

### 对 review 中几项建议的取舍

- “least-privilege role” 是正确要求，但它属于 HANA 部署授权，不能由 Python 代码可靠证明。本次代码强制 schema/table allowlist，生产环境仍需单独配置只读限权角色。
- `review2.md` 对 artifact/平台相关 hash 的判断正确。本次没有生成虚假的通用锁，而是保留已存在且经过 dry-run 验证的 Python 3.11/Linux workflow 锁；同时把发布给下游的兼容依赖范围拆到 `requirements-runtime.txt`。
- `review2.md` 所称“当前没有 lock”和“SQL agent 对象 allowlist 缺失”与审阅时的工作区不符：`.github/requirements/` 以及 schema/table allowlist 已经存在，所以没有重复实现。
- `review3.md` 的 V-171、V-174、V-161、V-155 及 V-159/V-172/V-173/V-165 复现成立，均已修复。其“给通用 requirements.txt 添加单平台 hash”的建议未采纳，因为这会重新制造 review2 已指出的伪通用锁；当前继续按真实 workflow 目标维护完整 hash lock。
- `review4.md` 的 V-171 quoted identifier case-fold 和非回环 SSE 复现成立并已修复。三个健壮性发现也成立：callable `ConnectionContext.user` 现在会被调用，未使用且可能跟随重定向的 aiohttp session 已删除，`fetch_mcp_audit_rows()` 不再向 `build_appsource_pack()` 传入不存在的参数。
- `review5.md` 对前四轮闭环状态的判断成立。它发现的 retrieval 基础设施对象名插值属于宿主误用时可触发的加固项，已连同同一路径中的证书/配置字符串 literal 一起修复。HTTP launcher 不内建 TLS 的边界也已写入函数说明和示例 CLI。其所称总结仍写“未推送”及额外 untracked 文件不符合当前工作区。
- `review5.md` 后续指出的 df_tools 间接执行路径复现成立：hana-ml 会先执行分号前语句，原 guard 只验证 agent 最终输出。现已将 scanner 提取到 `hana_ai.sql_guard`，所有受影响工具统一先验证再构造 DataFrame；生产数据库最小权限要求仍保留。
- `review7.md` 的 quoted literal/identifier 分号绕过复现成立。当前安装的 hana-ml `2.31.26120800` 源码确认使用原始 `sql.split(";")`；因此不能仅按 SQL 词法忽略引号内分号，guard 必须保证交给 hana-ml 的文本完全不含分号。
- `Mem0HanaAdapter` 是供受信任宿主代码使用的低层接口；agent/manager 的非可信请求路径已强制 tenant/session。拥有 Python 对象和数据库连接的宿主代码本身不属于该应用层授权边界，因此没有把低层 adapter 改成破坏兼容性的强制 scope API。
- HTTP MCP 对非 loopback 绑定强制 token；loopback 开发服务仍允许无 token。把本机回环服务也一律认证会造成额外破坏，而不会保护远端攻击面。runtime connection admin 即使在 loopback 也仍需显式 host opt-in。
- 保留对 fenced/commented SQL 的严格拒绝。自动剥离说明或注释会重新引入“从不可信模型文本中猜 SQL”的歧义。
- `secure_report_dir` 的进程退出清理是临时安全工件的生命周期设计，未改成长久保存；调用方若要保留报告，应在进程退出前复制到受控目录。
- hash lock 明确只覆盖 GitHub Actions 的 Ubuntu/Python 3.11。生成一个声称跨 Windows/Linux、Python 3.10～3.13 通用的锁文件并不真实。

## 四、兼容性与迁移速查

| 变化 | 对旧调用方的影响 | 迁移方式 |
|---|---|---|
| MCP runtime connection admin 默认不注册 | 旧 admin 工具默认不可见；HTTP 开启后必须认证 | 设置 `HANA_ALLOW_RUNTIME_CONN_UPDATE=true`，并配置和发送 Bearer token |
| TLS 证书校验默认开启 | 缺少 trust store 的部署可能连接失败 | 配置 CA；仅在受控环境显式设置 `HANA_SSL_VALIDATE=false` |
| 非回环明文 HTTP 默认拒绝 | 远端 `http://` MCP URL 构造失败；launcher 本身不终结 TLS | 使用 HTTPS 反向代理、Bearer token，并阻止直接访问明文监听端口 |
| SSE 仅允许 loopback | 旧的远端 SSE server 启动会被拒绝 | 远端改用带认证的 HTTP transport；SSE 仅用于本机进程边界 |
| HTTP 不跟随重定向 | 依赖 30x 的 endpoint 不再自动工作 | 直接配置最终 HTTPS endpoint |
| stdio 进程默认禁止 | 任意 `python`、`uvx`、`npx` server 配置被拒绝 | 配置 JSON 格式的 `HANA_MCP_STDIO_ALLOWED_PROFILES`；额外参数需单独 opt-in |
| Python 执行工具默认关闭 | 依赖默认 REPL/exec 的 agent 流程失效 | 使用专用工具；危险 opt-in 只允许完全可信代码 |
| SmartDataFrame/df_tools 仅接受单条只读 SQL | 说明、注释、写操作、多语句以及内部含分号的输入会在执行前拒绝；单个末尾分号会先删除 | 让模型返回不含内部分号的纯 `SELECT`/`WITH`，并让 HANA 账号使用只读限权角色 |
| SQL agent 强制 schema/table allowlist | 未传范围的构造调用失败；带引号标识符采用精确大小写 | 显式列出允许对象；混合大小写对象使用双引号 |
| Mem0/RAG 强制 tenant/session | 依赖 `global_session` 默认值的调用失败 | 从认证应用上下文传入稳定的 `tenant_id` 和 `session_id` |
| FAISS 路径与加载规则改变 | cwd 下旧索引不会自动加载；无清单的 pickle 默认拒绝 | 迁移到 `HANA_AI_DATA_DIR`；确认来源后执行一次可信迁移以生成清单 |
| 临时工件路径随机化 | 写死固定临时目录的调用失效 | 使用工具返回的实际文件路径 |
| import 时不再自动安装依赖 | 缺少可选依赖时直接 `ImportError` | 在部署阶段安装 package extra 或所需依赖 |
| Python 下限改为 3.10 | Python 3.9 及以下无法安装 | 升级 Python |
| CI 与发布依赖分离 | 源码环境使用精确版本；wheel/sdist 按兼容范围解析 | 维护者使用目标 workflow 的 hash lock；下游安装发布包 |
| NumPy 按 Python 版本选择 | Python 3.11 不再安装 NumPy 2.2.3 | Python `<3.12` 使用 1.26.4，`>=3.12` 使用 2.2.3 |
| Retrieval 基础设施名称严格编码 | 非法对象名由数据库语法错误变为提前 `ValueError` | 普通名称无需修改；混合大小写对象使用显式双引号名称 |
| 不再从 DBAPI repr 推导身份 | 标准账号密码连接会额外执行一次常量身份查询；无法查询的离线 wrapper 会失败 | 暴露稳定身份属性或无参方法，或支持 `SELECT CURRENT_USER FROM DUMMY` |
| ContextAgent session 目录增加摘要 | 既有 session 映射到新的目录名 | 将旧 `<storage>/sessions/<name>` 内容迁移到新目录 |

V-159 有一项明确的兼容性取舍：hana-ml 的 `ConnectionContext.sql()` 对原始文本使用不理解 SQL 引号的 `split(";")`。为了阻断 quoted literal/identifier 中的堆叠语句绕过，共享 guard 会拒绝任何内部位置的分号，即使该分号在标准 SQL 中本可作为合法字符串内容。输入末尾的一个分号仍允许，但会在数据库调用前删除。

## 五、13 个现成 patch 的纳入清单

以下 patch 的安全意图全部保留在当前工作区，部分又被继续扩展：

1. `V-155-mcp-admin-auth-tls-guard.patch`
2. `V-157-vectorstore-query-parametrize.patch`
3. `V-158-retrieval-procedure-identifier-quote.patch`
4. `V-159-smart-dataframe-strict-sql-extract.patch`
5. `V-160-remove-runtime-pip.1-nltk.patch`
6. `V-160-remove-runtime-pip.2-mcp.patch`
7. `V-160-remove-runtime-pip.3-langgraph.patch`
8. `V-161-faiss-safe-deserialization.patch`
9. `V-166-groupkey-escape.1-predict.patch`
10. `V-166-groupkey-escape.2-ts-check.patch`
11. `V-175-pin-deps.1-requirements.patch`
12. `V-175-pin-deps.2-requires-python.patch`
13. `V-176-cicd-sha-pin.release.patch`

这些 patch 此前已在干净 `e8e65a3` 上逐个通过 `git apply --check`。当前工作区已经包含并扩展相同代码，不能再次机械执行 `git apply`。

## 六、验证结果

已实际执行并通过：

- `review4` 后 `test_security_fixes.py`：**90 passed**；与 `test_hana_mcp_audit_view.py` 合并执行为 **97 passed**。新增覆盖 quoted identifier 大小写、非回环 SSE、callable 身份字段和 JSONL audit row 构造。
- `review5` 后 `test_security_fixes.py`：**92 passed**。新增覆盖 PSE、REMOTE SOURCE、CERTIFICATE 对象名和证书/配置 literal 编码。
- df_tools 堆叠 SQL 修复后 `test_security_fixes.py`：**95 passed**；使用 `XinKey` 的 Automatic Time Series、Additive Model Forecast、Intermittent Forecast 和 TimeSeriesDatasetReport 真实 HANA 回归：**7 passed**。
- `review7` 分号语义修复后 `test_security_fixes.py`：**98 passed**。使用 `XinKey` 实测普通 `SELECT 1 ...;` 正常返回，`SELECT '; DELETE ... --'` 在任何数据库调用前抛出 `ValueError`。
- `review8` 修复将 `ensure_hana_ml_single_statement()` 应用于向量查询、forecast split、时序 future-table builder 和临时表 builder；dataset split 同时要求 `order_by` 为真实源列。Retrieval 的 PSE/CERTIFICATE 多语句改为由受信任代码拆分后逐条通过 DBAPI 执行，不再把插值后的多语句交给 `ConnectionContext.execute_sql()`。
- SQL agent 的 `FROM|JOIN` 正则授权已替换为保守的对象 parser：解析括号 table factor 和嵌套 CTE 作用域，拒绝三段式对象、table function、sequence、qualified/未知 UDF 以及无法识别的 table expression。HANA SQLAlchemy 反射名称按目标 dialect 规范化，保证大写 HANA allowlist 可正常反射。
- 新增零执行安全回归，覆盖 vector `input`/`model_version`、split 标识符与 `order_by`、存储型 group 值，以及 SQL-agent 括号表、嵌套 CTE、UDF、sequence、table function 和三段式名称；断言恶意生成语句不会到达执行入口。临时表 builder 已加共享 guard，但没有单独的零执行回归用例。
- 使用 `XinKey2` 运行 `_review8_xinkey2_probe.py`：连接身份/当前 schema 为 `PAL_TEST`；H8-1、H8-2 的 canary DML 均未执行；H8-3 的括号表、嵌套 CTE、scalar UDF 和 sequence 均由 guard 拒绝；probe 创建的 UUID `R8_*` 对象全部清理。
- `test_appsource_pack.py`：**10 passed**。此前 `review3` 阶段的安全回归与 Mem0 组合为 **90 passed**，本轮未重复该组合。
- 安全回归加 ContextAgent memory/tool/skill 相关测试：**116 passed**。使用 `XinKey` 的真实 HANA 连接去掉 `userkey` 属性后，`SELECT CURRENT_USER FROM DUMMY` 回退成功返回 `PAL_TEST`。
- 预测表 HANA 集成测试：**2 passed**；MCP HTTP tool schema：**1 passed**；随机私有报告路径 HANA 集成测试：**1 passed**。
- `python -m compileall -q src/hana_ai examples ...`：通过。
- `git diff --check`：通过。
- `python -m pylint ./src`：**10.00/10**。
- `python -m build`：隔离环境按固定的 setuptools 84.2.1 / wheel 0.48.0 构建 wheel 和 sdist 成功；`python -m twine check dist/*` 两项通过。
- wheel 的 `Requires-Dist` 已检查为 `requirements-runtime.txt` 中的兼容范围，而不是 CI 使用的精确 `==`；sdist 已确认包含该文件。
- 单独下载并检查固定依赖：`fastmcp==2.10.6` 不含 `fastmcp.server.auth.providers.jwt.StaticTokenVerifier`；已改为当前请求级认证测试通过的 `fastmcp==2.14.7`。
- 初次 requirements dry-run 发现 patch 的 `langchain-hana==0.1.0` 要求 `numpy<2`，与 `numpy==2.2.3` 冲突；已升级为同属 LangChain 0.3 生态且接受 NumPy 2 的 0.3.4。
- 当前 Python 3.13 的 `pip install --dry-run -r requirements.txt`：解析成功。
- `requirements-runtime.txt` 已分别按 Python 3.10/Linux、3.11/Linux、3.12/Windows、3.13/Windows 解析成功；当前 Python 3.13 环境的 pip dry-run 也成功。
- `pip install --dry-run ".[all]"` 解析成功，选择 `sap-ai-sdk-gen==5.11.0`、LangChain 0.3.x 与 LangGraph 0.6.x；实测 SDK 6.x 已切换到不兼容的 LangChain 0.4/1.x，因此没有把范围错误地放宽到 6.x。
- 两份 hash lock 使用 `uv pip install --dry-run --require-hashes --python-version 3.11 --python-platform linux` 验证成功；Pylint 锁解析 140 个包，release 锁解析 35 个包。

完整项目测试也已实际执行。测试基类已改为从 `HANA_USERKEY` 读取可配置 userkey（默认仍为 `RaysKey`）。由于仓库中两个测试直接导入 `testML_BaseTestClass`，且两个目录存在同名测试模块，运行时给 `PYTHONPATH` 加入 `nutest/testscripts`，并使用 pytest 的 `--import-mode=importlib`：

```powershell
$testRoot = Join-Path (Get-Location) "nutest\testscripts"
$env:PYTHONPATH = $testRoot + [IO.Path]::PathSeparator + $env:PYTHONPATH
$env:HANA_USERKEY = "XinKey2"
python -m pytest --import-mode=importlib -ra
```

`XinKey2` 已通过 `sslValidateCertificate=True` 和 `SELECT CURRENT_USER FROM DUMMY` 验证，连接用户为 `PAL_TEST`。首次 review8 修复后的完整测试于 2026-10-27 执行，耗时 **26 分 57 秒**，结果为 **293 collected：280 passed、0 failed、13 skipped**。此前 12 项失败已全部修正，包括 forecast/intermittent 返回约定、时序报告文字和浮点容差、模板数量旧断言、VCAP fixture、HTTP/stdio runtime reload 的 userkey/admin/auth 初始化，以及 stdio 大消息与 timeout 处理。二次核对仍发现下述解析器缺口，因此该次全绿结果不能替代针对新 payload 的验证。

13 项跳过是显式 opt-in 的 live ContextAgent/LLM 测试：11 项需要 `RUN_CONTEXT_AGENT_E2E=1`，HTTP MCP 场景需要 `RUN_CONTEXT_AGENT_MCP_E2E=1`，stdio MCP 场景需要 `RUN_CONTEXT_AGENT_MCP_STDIO_E2E=1`。它们不是失败，也未在没有相应外部 LLM 配置时强制启用。

仍未执行，因此不作成功声明：

- 带真实 TLS 证书、远端 MCP 和完整工具调用的网络端到端测试。
- 既有生产 FAISS 数据的一次性信任、清单生成及后续重启迁移演练。
- Python 3.10～3.13 的完整 clean-install 与运行矩阵；当前只完成 Python 3.11/Linux lock 解析和 Python 3.13/Windows requirements 解析。

## 七、发布和迁移重点

这次修改不是完全无感的 patch release。部署前至少需要处理：

- 为远端 MCP 配置带认证的 HTTP transport 和 HTTPS 反向代理，并在网络层阻止直接访问明文监听端口；launcher 不内建 TLS。SSE 只允许 loopback，本机 loopback HTTP 可继续使用。
- 管理连接更新同时配置 `HANA_ALLOW_RUNTIME_CONN_UPDATE=true` 和 HTTP Bearer token。
- 为 stdio server 配置完整 `HANA_MCP_STDIO_ALLOWED_PROFILES`。
- 安装需要的 `nlp`/`graph` extra，不能依赖运行时自动安装。
- 配置 HANA CA/trust store；证书校验现在默认开启。
- 迁移旧 FAISS 路径，并对任何允许反序列化的 pickle 做来源和完整性确认；旧 store 首次可信加载后会生成完整性清单，新建 store 可正常重启加载。
- 标准账号密码连接会用 `SELECT CURRENT_USER FROM DUMMY` 获取稳定身份；自定义 wrapper 必须暴露身份字段或支持 DBAPI cursor 查询。
- dataframe agent 改用显式专用工具；不要为不可信数据重新开启 Python REPL。
- 按目标平台重新生成和审阅 GitHub Actions hash lock；不要把 Python 3.11/Linux lock 用于其他平台。

## 八、review8 修改的二次核对（2026-10-27）

再次核对发现并修复了三个实际问题：

- **数字 token 可吞掉危险对象表达式。** 原 tokenizer 把数字后的任意字母、点号和加减号都视为数字的一部分，导致 `1+GET_SECRET()`、`1e-2+SCHEMA.GET_SECRET()` 和 `1+SCHEMA.SEQ.NEXTVAL` 躲过函数/sequence 检查。新增测试先复现 5 个 payload 到达 DBAPI；现改用严格的十进制/科学计数法 token，所有这些输入均在 execute 前拒绝。
- **合法 SQL 被误拒绝。** `WHERE (...)`、`AND (...)`、`NOT (...)` 被误当作函数，`LEFT OUTER JOIN` 缺少 `OUTER` 支持。现已区分保留关键字与函数，并补全 outer join，增加实际查询回归。
- **公开 factory 重复规范化表名。** `create_hana_sql_agent()` / `create_hana_sql_toolkit()` 先规范化反射名，再由 `_ReadOnlySQLDatabase` 第二次处理，丢失显式引号语义。现将原始 allowlist 交给构造器，仅规范化一次；混合大小写、空格和点号表名均增加回归。

此外，forecast split 会在读取源表元数据前拒绝 source/schema/order/output 名称中的分号，并增加 3 个输出表名的零访问回归。`_review8_xinkey2_probe.py` 从仅输出诊断改为显式断言：创建 UDF/sequence 失败、攻击未被拒绝、SQL-agent 攻击到达 execute 或清理失败都会使 probe 失败。

二次验证：

- 修复 parser/factory 后，安全回归与 audit 测试 **142 passed**；随后补充的 3 个 split 输出名用例纳入全量运行。
- 扩展 `XinKey2` probe 成功，身份/schema 为 `PAL_TEST`：4 个数字前缀 HANA 用例及原 H8-3 用例均拒绝，SQLAlchemy execute 事件计数为 **0**；合法 grouped/outer-join 查询返回 `1`，公开 toolkit factory 对带引号混合大小写表返回 `7`。
- probe 清理后查询 `SYS.OBJECTS`，当前 schema 的 `R8_*` 对象数为 **0**。
- 本次修改的生产模块 Pylint **10.00/10**；编译与 `git diff --check` 通过。
- 最终全量 `XinKey2` 重跑完成：**313 collected：300 passed、0 failed、13 skipped**，耗时 **1612.65 秒（26 分 52 秒）**，pytest 退出码为 **0**。13 项仍为前文列出的显式 opt-in ContextAgent/LLM 场景；65 条 warning 为依赖弃用和已有 SQLAlchemy `case_sensitive` 参数提示，未出现测试失败。

## 九、review9 兼容性与 JOIN 边界修复（2026-10-27）

提交状态：代码、兼容性回归测试和公开 API 文档已于 2026-10-27 提交为 `5c146c3` 并推送至 `security/assessment-13-fixes`（PR #162）。本报告、review 文件与一次性 probe 继续仅保留在本地，未纳入提交。

`review9.md` 指出的合法 SQL 误拒绝成立。修复前新增的 16 个合法查询用例全部被 parser 拒绝；现已支持 `CAST`（含内置类型精度/长度参数）、`TO_VARCHAR`、`ANY/SOME`、`WITH HINT`、`PARTITION` 以及相关子查询/CTE 的 `LATERAL`。真实 HANA 验证的抽样语法是 `TABLESAMPLE SYSTEM/BERNOULLI (...)`；该环境不支持报告所写的 `SAMPLE ...`，因此按数据库实际语法实现。

实现仍逐层校验 CAST 操作数、ANY/SOME 与 LATERAL 子查询中的所有对象。类型/分区/抽样参数仅接收数值 literal，hint 接收名称与扁平参数，不能跳过其中隐藏的 SQL、UDF 或 sequence。字符串在扫描时保留不透明 operand 占位，实际执行仍使用原始查询，含引号和分号的 SQL-agent 字符串值不被改写。

再次检查 JOIN 时还发现报告未覆盖的对象授权绕过：`FROM ALLOWED A JOIN ALLOWED B ON A.ID=B.ID, SECRET S` 中 ON 后的逗号及 SECRET 表被当作条件文本吞掉。离线新增回归确认旧实现确实到达执行。现将顶层逗号交回 FROM parser 拒绝，并增加真实 HANA 的零 execute 断言；函数参数内部的逗号不受影响，`LEFT(...)` 等函数也可继续用于 ON 条件。

其他处理：

- 更新公开 SQL-agent 文档，列明支持语法及保留的对象限制。
- 对仍经过 hana-ml raw split 的 vector/group-key/identifier 路径，继续拒绝分号，并明确记录兼容性代价；没有为了兼容性重新开放已确认的注入路径。
- `_create_pse_sql_string()` 在 docstring 中明确标记为 deprecated 的兼容展示 helper，禁止交给 hana-ml 执行；生产继续逐条 DBAPI 执行。
- 报告提到的 `SECURITY-REVIEW-13-ZH-CN.md`、`SECURITY-FIXES-13-hana-ml 2.md` 及部分未跟踪目录不在当前工作区；本轮更新实际存在的本报告和公开 API 文档，未删除用户本地文件。

验证记录：

- 首批修复及 JOIN 修复后，compatibility/security/audit/appsource 组合 **213 passed**。
- 新增 live 测试将合法查询的结果逐行与直接 HANA DBAPI 执行结果比较，并验证表行数；fixture 使用随机 `R9_COMPAT_*` 名称，结束后删除。
- 最终兼容性矩阵 **62 passed**：42 个离线语法/对象边界用例，20 个真实 HANA 用例（19 个合法查询与直接 DBAPI 返回逐行一致，1 个 JOIN 后逗号绕过零执行用例）。当前 schema 的 `R9_*` fixture 对象数已确认回到 **0**。
- 修改的生产模块 Pylint **10.00/10**，编译和 `git diff --check` 通过。
- 最终全量 `XinKey2` 重跑完成：**375 collected：362 passed、0 failed、13 skipped**，耗时 **1640.08 秒（27 分 20 秒）**，pytest 退出码为 **0**。13 项仍为显式 opt-in 的 live ContextAgent/LLM 场景（11 项 ContextAgent、1 项 HTTP MCP、1 项 stdio MCP），未启用，不能视为已通过；65 条 warning 为依赖弃用及已有 SQLAlchemy 参数提示。

## 十、启用全部 live 开关的复跑与安全问题回复（2026-10-27）

按本轮要求，在已推送的 `5c146c3` 上重新运行全量测试，同时设置 `HANA_USERKEY=XinKey2`、`RUN_CONTEXT_AGENT_E2E=1`、`RUN_CONTEXT_AGENT_MCP_E2E=1` 和 `RUN_CONTEXT_AGENT_MCP_STDIO_E2E=1`。本轮未修改生产代码或已提交测试，也未再次提交、推送。

结果：**375 collected：362 passed、0 assertion failures、11 setup errors、2 skipped、65 warnings**，耗时 **1792.25 秒（29 分 52 秒）**，pytest 退出码 **1**。这不是全部通过；原先 13 个 opt-in 跳过不能被当成已完成验证。

- `test_security_fixes.py`：**138 passed**。
- `test_sql_agent_compatibility.py`：**62 passed**，包括 42 个离线用例和 20 个真实 HANA 用例。
- audit view：**7 passed**；appsource pack：**10 passed**；其他可执行测试：**145 passed**。
- 11 个 live ContextAgent 场景在同一个 `setUpClass` 中失败：fixture 硬编码 `RaysKey`，没有使用 `HANA_USERKEY=XinKey2`；HANA 返回 `-10104 Invalid value for KEY "RaysKey"`。这是共享的连接初始化问题，不是 11 个独立产品缺陷；LLM 初始化尚未执行，场景仍未验证。单独复跑其中一个场景也复现相同错误。
- 2 个 live HTTP/stdio MCP 场景即使开关已开启也仍跳过：所需 `nutest/testscripts/env.example` VCAP fixture 不存在。后续还需将这些旧 fixture 与 runtime-admin opt-in/auth、stdio profile 和安全 TLS 配置对齐，不能通过关闭生产安全控制来让测试通过。

补充安全验证：

- `_review8_xinkey2_probe.py` 退出码为 0；HANA 用户/schema 为 `PAL_TEST`。vector/split 注入 canary DML 均未执行，括号表、CTE 作用域、UDF、sequence 及数字前缀绕过被拒；被拒 SQL-agent 语句的执行事件数为 **0**。合法 grouped/outer-join 和带引号混合大小写 factory 查询保持正常。
- probe 与 review9 测试完成后，当前 schema 的 `R8_*`/`R9_*` 对象数为 **0**。
- 本地隔离的 Python 工具边界检查 **8 passed**：默认执行关闭，调用参数不能开启，非 Boolean True 的宿主参数不能开启；显式宿主 opt-in 仍能读取合成 CSV 并调用 mock 的原始 SQL 方法，证明其仍不是沙箱。未读取真实秘密、未执行外传，也没有将该补充检查的 SQL 发送给真实数据库。该检查不是 LLM 端到端提示注入演练。
- `python -m pylint ./src` 为 **10.00/10**；编译与 `git diff --check` 通过。

环境：Windows/Python 3.13.11，pytest 9.1.1，hana-ml 2.31.26120800，hdbcli 2.30.27，SQLAlchemy 2.0.48，FastMCP 2.14.7。`hana_ai` 来自本仓库源码，`hana_ml` 来自相邻 `../hanamlapi` 源码；这不是已发布 wheel 的 clean-install 验证。机器可读结果保存在 git-ignored 的 `.pytest_cache/security-feedback/` 中。

新增英文回复与验证总结：[security_feedback_2026-10-27.md](security_feedback_2026-10-27.md)。主要结论：

- 对不可信模型输出进入 Python 执行工具的漏洞，**Code Injection / CWE-94** 分类合理；prompt injection 是投递机制。原实现有 AST blocklist，不能说完全没有检查，但这些检查无法隔离 `pd`/`cc` 的文件、网络和数据库能力。
- 客户自管部署需要升级到经确认包含修复的发布版本或批准构建并重新部署，保持危险执行关闭并落实最小权限；不是自动获得保护。PR #162 当前仍 open/unmerged，不能仅以未改变的包版本 `1.1.26081400` 声称已发布完整修复。
- outbound OSS 组件为 SAP 的 Generative AI Toolkit for SAP HANA Cloud / `hana-ai`。本仓库不能确认 SAP 内部消费方清单，也不能据此断言“没有其他 SAP 产品受影响”；需组件和产品安全负责人确认实际版本、配置、复用关系及上线状态。

## 十一、PyPI hana-ml 全量验证（2026-10-28）

按要求改用 PyPI 当日最新发布的 **`hana-ml==2.30.26091800`**，不再从相邻 `../hanamlapi` 源码加载。该版本满足 runtime 范围 `>=2.29.26072000,<3`，但不是开发 requirements 中较旧的固定版本 `2.29.26072000`。

- 下载公开 wheel `hana_ml-2.30.26091800-py3-none-any.whl` 并与 PyPI 元数据核对 SHA-256：`73fc9f2df49d6e466ec0f334e4b56d530b34f47d72560e5790efa8c466f9ae3f`。
- 在 `.pytest_cache/pypi-hana-ml-2026-10-28/venv` 建立 `--system-site-packages` 虚拟环境，仅在其中安装此 wheel；复用其他已安装依赖进行受控比较，没有卸载或修改原 Conda/editable 安装。这不是全依赖 clean-install；`hana_ai` 仍来自本仓库源码。
- 验证主进程与 Python 子进程的 `hana_ml.__file__` 均来自该 venv 的 `Lib/site-packages/hana_ml`，`__version__` 与发行元数据均为 `2.30.26091800`。Windows 下裸 `python` 子进程最初仍加载 editable 版本，现通过仅对测试进程设置 `PYTHONPATH` 优先包含 venv site-packages 解决，确认后才启动全量测试。
- 使用 `HANA_USERKEY=XinKey2` 并开启全部三个 live opt-in 开关。生产代码与已提交测试均不修改，不更改或新增 userstore key。

全量结果：**375 collected：361 passed、1 failed、11 setup errors、2 skipped、65 warnings**，耗时 **1735.72 秒（28 分 55 秒）**，pytest 退出码 **1**。不能描述为全部通过。JUnit 对比确认与 2026-10-27 基线为同一组 375 个测试 ID，只有 stdio connection-reload 用例由通过变为失败，其余结果一致。

- 安全回归 **138 passed**，SQL 兼容性/安全矩阵 **62 passed**（42 个离线、20 个 live HANA），两组均无失败、错误或跳过。audit view **7 passed**，appsource pack **10 passed**，其他通过项 **144**。
- 新增失败：`test_mcp_server_reload_connection_context_from_env_stdio.py::TestMCPReloadConnectionContextFromFileSTDIO::test_reload_connection_context_from_file_stdio` 在 `client.initialize()` 阶段超过 **120 秒**，尚未调用 admin reload。错误为 `STDIO RPC timeout for method=initialize`；JUnit 用例耗时 120.085 秒，stderr tail 只有依赖弃用提示，不能据此确定延迟根因。
- 同一 PyPI 环境、相同 timeout 和未修改测试的独立复跑结果为 **1 passed in 45.75 seconds**，退出码 0。该复跑未复现失败，但不覆盖/替代原全量失败，也不能据此认定 PyPI 版本必然不兼容或超时已经修复；仍需调查执行时序、套件交互或环境等因素。
- 11 项 setup error 仍是 fixture 硬编码 `RaysKey`（`-10104 Invalid value for KEY`）；2 项 skip 仍为缺少 VCAP fixture。新 traceback 中的 hana-ml 文件来自 venv 的公开发行包，依赖切换没有解决这 13 个 live LLM 场景的配置前提。
- 补充 Python 执行边界检查 **8 passed in 2.38 seconds**，只使用合成 CSV 和 mock SQL 方法；此补充检查和独立 stdio 复跑均不计入 375 项全量结果。

PyPI 环境下的 `_review8_xinkey2_probe.py` 退出码为 **0**：vector/split canary 写入均被拦截，未授权表/CTE/UDF/sequence 与数字前缀 payload 被拒，拒绝查询的执行事件数为 **0**；合法 grouped/outer-join 与 quoted mixed-case factory 查询保持正常。probe 与兼容性测试结束后，当前 schema 的 `R8_*`/`R9_*` 对象数为 **0**。该环境下全 src Pylint 为 **10.00/10**，编译和 `git diff --check` 通过。

PyPI wheel 的 `hana_ml/dataframe.py:1713` 仍使用原始 `sql.split(";")`，因此保留 hana-ml sink 的分号限制仍有必要。PR #162 于本日再次确认仍 open/unmerged；不能把公开 hana-ml 依赖版本误当成已经发布的 hana-ai 安全修复版本。

详细复现命令、对比结果和边界说明更新在 [security_feedback_2026-10-27.md](security_feedback_2026-10-27.md) 第六节。新 wheel、pip 安装报告、环境以及 `full-suite.xml`、`stdio-reload-diagnostic.xml`、`python-boundary.xml` 位于 git-ignored 的 `.pytest_cache/pypi-hana-ml-2026-10-28/`；保留原 2026-10-27 验证结果供比较。本轮未修改生产代码、未修改已提交测试、未提交或推送。

## 十二、重新核对扫描包与 Critical/High 补丁（2026-10-28）

本节是第十一节之后的新修改。核对本地 `13-hana-ml` 扫描包
中的索引、跟踪表、Critical/High 修复要求、全部 13 个 patch，以及两份本地报告和
当前源码。扫描与补丁基于旧 `main @ 5cc559b`，跟踪表的 open/partial 不是当前
`5c146c3` 分支的新扫描结果；原扫描目录未修改。

13/13 补丁逐文件对应关系见英文报告第七节。保留现有修复而不机械重放旧 diff：
V-159 原 patch 仍会抽取 SELECT 子串且未充分验证 WITH；V-161 原 patch 保留宽泛异常
fallback；V-175 原依赖组合存在兼容性问题。当前对应实现更严格，不能用旧补丁覆盖。

本轮新增修改：

- **V-155/V-167**：连接更新的 `encrypt`/`ssl_validate_certificate` 省略或 null
  现在显式传入 True，涵盖测试连接和实际替换；环境连接的 `HANA_ENCRYPT` 默认 True，
  涵盖 userkey 和账号密码模式。显式降级仍需确认，不能把驱动默认值当成 TLS 保证。
- **V-157**：向量 constructor 的 `has_table()` 及查询的懒加载 `.columns` 曾在
  验证前触达 hana-ml。新增配置检查，覆盖初始化、get/upsert 和 query，在 metadata/DDL
  前拒绝危险值。PyPI hana-ml `has_table()` 对对象名采用单引号字符串拼接，因此本 API
  的 table/schema 名额外拒绝单引号；普通混合大小写名称不变，查询文本中的单引号仍安全转义。
- **V-163**：证书/私钥临时文件创建后立即记录路径，再编码和写入，修复编码异常及部分
  写入失败时 finally 找不到文件路径的遗漏；直接尝试删除并容忍文件已不存在，删除失败仍警告。
- **V-164**：远程明文 HTTP 只允许真正的 Boolean True 显式开启；字符串 `"false"`、
  整数及非空字典不再因 truthy 意外绕过 TLS 要求；禁止自动重定向的行为保留。
- **V-165**：即使 host profile 匹配，也拒绝已知 shell 和 `.bat`/`.cmd`/`.ps1`
  入口，同时检查 executable 解析前后名称。属于防止危险配置的加固，不表示远端调用者能够
  修改 host profile；已批准的普通可执行文件/参数仍可用，并非可执行内容沙箱。

新增 `nutest/testscripts/test_assessment13_followup.py`。先在修复前复现：第一批
24 项为 **17 failed、7 passed**；另 9 项 vector 前置验证均失败。只使用合成凭据、
不执行的占位程序和 mock 数据库，没有读取真实秘密或向外传输。新增 HTTP 测试通过
实际 toolkit launcher 构造的 FastMCP ASGI app 验证无/错误 token 返回 401 且没有
调用连接构造器，合法 token 才能到达工具。README 同步说明迁移影响。

不能将以上说成“所有风险消失”或“全套测试全通过”：Python 默认关闭而不是实现沙箱；
loopback HTTP 是有意保留的本地兼容方案；SQL 未全面改成 bind parameter；hash lock
验证目标仍是 Python 3.11/Linux。危险 opt-in、HANA 最小权限、远程 HTTPS、可信存储/
进程环境仍属部署边界；具体修复发布版本和 SAP 内部消费者仍需负责人确认。
此前完整测试的错误/跳过继续保留，新定向测试不能代替缺少配置的 LLM/VCAP 场景。

最终定向验证结果：**266 passed、0 failed、0 errors、0 skipped、42 warnings**，
耗时 **277.75 秒（4 分 38 秒）**，退出码 **0**。继续使用第十一节隔离环境中的 PyPI
`hana-ml==2.30.26091800` 和 `HANA_USERKEY=XinKey2`，再次核对包版本与加载路径。

- 新增 follow-up 安全用例 **41 passed**；原安全回归 **138 passed**。
- SQL 兼容/安全 **62 passed**（42 个离线、20 个真实 HANA）；audit/appsource **17 passed**。
- 真实 vector create/upsert/query **1 passed**；MCP HTTP/stdio reload、HTTP/stdio/SSE
  schema 和环境参数单测共 **7 passed**。
- 独立 Python 边界检查 **8 passed in 3.94 seconds**，不计入上述 266 项。
- 全 src Pylint **10.00/10**，编译与 `git diff --check` 通过。

中间 247 项较小集合全部通过；随后扩展运行有 **263 passed、1 failed**，失败原因是
已有环境参数单测的预期字典尚未包含新默认 `encrypt=True`。更新该预期后重跑上述完整
定向集合，并增加两个 shell/alias 用例，得到最终 266 项全通过，没有为了通过测试关闭 TLS。
stdio reload 此轮通过不能证明此前全量超时根因已解决。当前结果不是整个仓库全量重跑。

证据目录 `.pytest_cache/assessment13-followup/`：`before.xml`、`vector-before.xml`、
`security.xml`、`final-security.xml`（中间 263/1）、`verified-security.xml`（最终 266/0）
和 `python-boundary.xml`。原全量基线保持不变。英文报告第七节提供精确复现命令与逐补丁
对应表。代码、README、两份本地报告及测试已更新；本轮未提交或推送。

## 十三、修复兼容性回归与区分安全行为变更（2026-10-28）

本轮修复复核中发现的两项回归，保留第十二节已有改动，未修改原始扫描包：

- **Memory filter 回归**：search 和 export 的两个重复 post-filter 只识别 equality、
  `$contains` 和 `$lte`，会丢弃合法的 `$eq`、`$gte`、`$in` 等查询结果。新增共享
  `src/hana_ai/mem0/metadata_filters.py`，由 adapter、manager 和默认 HANA export
  handler 共同使用，支持 equality/comparison、membership、between、LIKE、typed date
  及嵌套 AND/OR。继续逐条校验 tenant/session；同字段多个条件全部生效，非法条件或
  不可比较值不能匹配，不再因 TypeError 中断结果处理。保留原 memory API 的 list
  `$contains` 语义，不声称复刻 HANA full-text search 或所有数据库隐式类型转换。
- **Stdio 绝对路径回归**：`/code/server.py`、`/config/server.py` 等多层 POSIX 路径
  曾误判为 `/c` shell switch。现在区分路径和 switch，支持已批准的 script/data 路径
  （包含空格的路径也有测试）。仍要求 host profile 精确匹配 resolved executable 和
  参数；额外参数必须 host opt-in；shell/batch 入口及 `-c`、`-e`、`--eval`、`/c`、
  `/cCOMMAND` 等危险参数继续拒绝。这不是取消 executable 信任边界或新增代码沙箱。

新增 `nutest/testscripts/test_pr_compatibility_regressions.py`。初始 157 项在修复前
为 **70 failed / 87 passed**，修复后与原 memory/security/follow-up 测试合跑为
**342 passed in 42.77 seconds**；随后补充多条件、无效 NOT IN 和含空格路径用例。
基于 mock 的 memory 测试同时检查 search、manager export、默认 HANA export，故意
让 backend 返回其他 tenant/session 的数据；fallback API 和嵌套 OR 不能绕过 scope。
stdio 恶意参数测试只调用 validator，不执行任何攻击命令。

### Intentional functionality changes 是否由安全修复引起？

**大部分是，但安全目标和具体实现方式需要分开。上述两项回归并非必须接受的安全代价。**

| 行为变化 | 安全依据与边界 |
|---|---|
| 默认不启用通用 Python/legacy dataframe REPL | V-156/V-168，避免不可信 LLM 输入触发 host/DB 权限下的任意代码执行。显式危险 opt-in 不是沙箱；直接 Python tool 的 default-off 在本 PR base 已存在，不能全部归功于本 PR。 |
| 对受影响 SQL 路径施加只读、对象权限、标识符和输入校验 | V-157/158/159/166/171，防注入与越权。专用 write tools 仍可写；PyPI hana-ml 原始分号拆分路径仍需要拒绝分号。 |
| MCP 管理功能 opt-in/auth、TLS、禁止自动重定向、可信 stdio profiles | V-155/164/165/167，防未授权管理、秘密泄露及非预期进程执行。远程 HTTPS/最小权限配置仍由部署方负责。 |
| Memory tenant/session scope、可信 FAISS 加载、隔离 session storage | V-161/169/170/172，防跨租户数据泄漏、路径碰撞和不可信反序列化；身份必须由应用可信地提供。 |
| 私有随机 artifact 目录、HTML opt-in、临时密钥清理 | V-162/163/173，防共享文件泄漏、注入显示和秘密残留。 |
| 不再运行时 pip install、依赖/CI 固定和校验 | V-160/175/176，减少供应链与不可重复部署风险；用户需预先安装所需 extras。 |

具体 session path hash/迁移方式、退出时清理 artifact 的时机、确切版本号和 Python
最低版本、禁止所有 SQL comments 或某些安全 SQL 语法，是实现/兼容性选择，并非
漏洞恢复唯一要求；可以在保持安全边界并补测试的前提下优化。本轮不撤销这些策略。
两份报告是本地 untracked 文件；本轮未提交或推送。

### 本轮最终验证

在同一隔离环境再次确认实际加载公开 PyPI `hana-ml==2.30.26091800`，不是相邻
editable hanamlapi；继续使用 `HANA_USERKEY=XinKey2`。最终扩大定向运行结果：
**442 passed、0 failed、0 errors、0 skipped、42 warnings，216.20 秒，退出码 0**。
JUnit 独立核对总数一致。

- 新兼容性回归 **170**；已有 memory adapter **6**。
- 原安全回归 **138**；Assessment-13 follow-up **41**。
- SQL 兼容/安全 **62**（42 离线 + 20 live HANA）；audit/appsource **17**。
- 真实 vector create/upsert/query **1**；MCP reload/schema/env **7**，其中真实 stdio
  reload **34.172 秒通过**。
- 全 src Pylint **10.00/10**，compileall 与 `git diff --check` 通过。

证据：`.pytest_cache/assessment13-followup/compatibility-before.xml`（157 项修复前）、
`compatibility-fixed.xml`（342 项首次通过）、`compatibility-verified.xml`（最终 442 项）。
精确复现命令见英文报告第八节。此次是定向扩大验证，不是全仓库重跑；不覆盖此前
缺少 RaysKey/VCAP 的 LLM 场景，也不能证明此前全量 stdio timeout 根因已经解决。
原全量结果和前几轮报告均保留，不将任何 skipped/error 改报为 passed。

## 十四、Intentional functionality changes 的必要性分析（2026-10-28）

本节补充第十三节的分析：**并非每项 intentional functionality change 都必须保持
当前形式。安全保护目标需要保留，但具体功能限制、配置形式和迁移方式可以调整。**
“由安全修复引起”不等于“当前实现是唯一必要方案”。以下结论针对不可信 agent 输入、
外部数据和多租户使用场景，不是放宽现有保护的授权或已经实施的代码修改。

| 行为变化 | 必要性判断 | 可以保留或恢复的合法功能 / 调整条件 |
|---|---|---|
| 默认关闭 unrestricted agent-generated Python / legacy REPL | 对不可信输入，必须阻止其直接获得 host/DB 权限；default-off 是当前安全默认值。 | 不要求永久删除所有 Python 功能。若要面向不可信输入恢复执行，需要经过验证的隔离和权限控制；现有 dangerous host opt-in 不是沙箱，不能作为安全恢复方案。 |
| SQL 校验和数据库访问授权 | 防注入、越权和未经授权写入是必要保护。 | 不必把所有功能都改为只读；明确授权的专用 write tools 可以保留。受影响 hana-ml sink 的原始分号拆分风险未消除前，不能解除相应分号限制。 |
| MCP authentication、remote TLS、受控 stdio process launching | 对暴露的受保护接口、远程传输和不可信命令配置，需要相应保护。 | profile/config 的具体格式不是唯一方案；可调整实现，但必须保持可信主机授权、执行目标约束和传输保护。不能因兼容性直接恢复任意进程启动。 |
| Tenant/session isolation、可信 FAISS 加载和反序列化保护 | 在多租户/多会话及不可信存储场景下必要。 | session path 的具体 hash 和迁移方式可调整，但不能引入路径碰撞、跨租户访问或不可信 pickle 加载；应用仍须提供可信身份。 |
| 私有 artifact 存储和 unsafe HTML 防护 | 防止文件暴露和不可信内容执行的目标必要。 | 报告不必在进程退出时一律删除，可采用权限受控、隔离的持久化存储及明确 retention policy；安全渲染方案也不必永久关闭全部 HTML。报告保留策略不等同于临时私钥/证书清理，后者的安全清理保护应保留。 |
| 移除运行时 pip install | 属于合理的供应链安全加固，应保留显式安装边界。 | 依赖和 extras 仍可在构建/部署阶段预先安装；不需要因此删除依赖这些包的合法功能。 |
| 精确 dependency pins、Python minimum version、禁止所有 SQL comments | 这些具体取值或 blanket restrictions 并非上述漏洞本身必然要求。 | 需要分别给出安全版本、源代码语法/依赖兼容性、可重复构建或 SQL parser 能力的依据。不能任意降级到已知不安全/不兼容版本；允许 comments 或更多安全 SQL 语法前，必须验证解析和授权边界不被绕过。 |

**不必要的回归已修复**：合法 memory filters 被丢弃，以及已批准的 stdio absolute
script/config paths 被误拒，都不是必须接受的安全代价；第十三节记录了修复及验证。

建议保留安全边界，优先评估 artifact retention 和过宽 SQL rejection 等可避免的
兼容性限制；如后续调整，应同时补充正常功能与攻击输入回归测试，并明确迁移影响。
本次仅将必要性分析写入本文件，没有更改上述策略、生产代码或测试；没有重新运行测试，
第十三节的 442 项结果仍是上一轮定向验证记录。本次未提交或推送。

## 十五、将 live test fixtures 从 RaysKey 切换至 XinKey2（2026-10-28）

用户确认使用已有的 `XinKey2`（不是 `XinKeys2`）。本轮只调整自动化测试 fixture，
不修改 hdbuserstore、生产代码、demo notebooks 或生产安全开关：

- `testML_BaseTestClass.py` 的 `HANA_USERKEY` 默认值改为 `XinKey2`。
- `test_context_agent_e2e_scenarios.py` 的硬编码 `RaysKey` 改为
  `os.environ.get("HANA_USERKEY", "XinKey2")`，并更新说明。
- `test_context_agent_mcp_reload_connection_context_from_file_stdio.py` 的 server
  `--userkey` 同样改为环境配置加 `XinKey2` 默认值，保留原 VCAP fixture 要求。

首次开启三个 ContextAgent live opt-in 开关运行 13 项时，缺失 key 错误不再出现，
但暴露 scenario fixture 原有 Windows TLS 配置错误：certifi CA PEM 被当成
`sslKeyStore`，导致 `CertOpenSystemStoreA / 0x80070057`。结果为 **11 setup errors、
2 skipped，118.92 秒**，未执行 scenario 断言。随后将该 fixture 改为使用 HANA client
原生 certificate trust，移除错误的 `sslKeyStore`，保持 `encrypt=True`，并将
`sslValidateCertificate` 从 False 改为 True；没有通过关闭证书校验绕过错误。

补充检查：共享 base fixture 不设置 `HANA_USERKEY` 时，实际使用默认 `XinKey2`
成功执行 live HANA `SELECT 1`；使用 mock connection 验证显式环境 key override
仍被保留。这两项是独立诊断，不并入 pytest E2E 计数。PyPI
`hana-ml==2.30.26091800` 的版本与 venv site-packages 加载路径再次确认。

初次运行证据：`.pytest_cache/assessment13-followup/xinkey2-context-e2e.xml`。
原生 TLS 修正后的重跑结果：**13 collected：2 passed、9 failed、0 setup errors、
2 skipped、2 warnings，863.51 秒（14 分 23 秒），退出码 1**。JUnit 总数一致，证据为
`.pytest_cache/assessment13-followup/xinkey2-context-e2e-native-tls.xml`。缺失 key 和
Windows TLS setup 错误均已解决，11 项 scenario 真正开始执行，但不能描述为全部通过。

通过项：data preview/dataset report、standard forecast train/predict。

失败项分类（细分到用例的英文说明见 security feedback 第九节）：

- **4 项输出/措辞断言**：CAP 实际生成了 package.json，但 LLM 返回 `Output directory`，
  测试只识别 `Root directory`；import/split/profile 和 prediction visualization 的回答
  没有包含测试需要的 HTML path；single-series outlier 返回了三个异常日期，但使用单数
  `outlier`，未包含断言要求的 `outliers` 或 `result_select_statement`。
- **3 项交互未完成预期动作**：massive additive 使用另一个模型名并要求确认；massive
  forecast 询问 AutoML 配置；prediction auto-repair 在前置预测阶段询问 exog/key，
  尚未到达 repair 断言。这些不是已证明的安全回归，也不能当成已完成对应操作。
- **1 项与有意安全默认值冲突**：notebook render 用例已生成 HTML 文件，但仍期望自动
  raw HTML display。V-173 默认不启用 `HANA_ALLOW_RAW_HTML_RENDER`，故 `rendered` 为空。
  本轮没有为使测试通过而开启该危险选项或更改断言。
- **1 项真实 SQL runtime 错误**：massive outlier 返回 `invalid column name: STORE_A`。
  源码检查发现公开 PyPI `hana_ml/algorithms/pal/tsa/outlier_detection.py:710` 在 timestamp
  result remapping 中将字符串 group name 直接拼入比较条件，缺少字符串引号，与错误一致。
  该 hana.ai tool 文件相对本 PR base 没有改动；本轮未修改 PyPI 包，仍需独立回归/修复，
  不能把此错误解释为必须接受的安全限制。

两项 MCP file-reload 仍因缺失 `nutest/testscripts/env.example` / VCAP fixture 而跳过。
已向用户询问可用 fixture 的路径，没有虚构或写入凭据。修改 userstore key 不会补齐该前提。

运行后按唯一前缀 `CTXE2E_D29D8991` 读取并核对遗留对象，再清理 **1 个本轮模型和
9 张本轮表（包含其模型表）**；复查该前缀的 table/model 数均为 0。它们是合成测试数据，
可通过重跑重新生成，未删除无关模型或业务表。测试自身也清理了本轮临时 artifact 目录。
compileall 与 `git diff --check` 通过。此次是受影响的 13 项定向 E2E 重跑，不是全仓库
重跑；此前 442 项定向通过和旧全量基线均保留。代码与两份报告更新，未提交或推送。

## 十六、PR 交付前验证（2026-10-28）

按用户要求将代码、测试及两份报告纳入本次 PR 交付；原始扫描/review 文件、一次性
probe、未跟踪 notebooks 和生成工件不纳入提交。报告移除工作站绝对路径，历史各轮
“未提交/仅本地”表述保留为当时状态，首页已说明本次交付范围。

同一 PyPI 环境提交前定向离线复跑：`test_security_fixes.py`、
`test_assessment13_followup.py`、`test_pr_compatibility_regressions.py` 和
`test_hana_mem0_adapter.py` 共 **355 passed、0 failed/errors/skipped、36 warnings，
57.43 秒，退出码 0**。compileall 与 staged `git diff --cached --check` 通过。
JUnit：`.pytest_cache/assessment13-followup/pre-push-offline.xml`。
此次不重跑 live 场景，不能覆盖第十五节的 **2 passed / 9 failed / 2 skipped**。

## 十七、review10：WITH HINT 末尾边界修复（2026-10-28）

本节记录基于 `d8d4714` 的后续修复。对应 V-171 的对象授权解析器
缺口；未修改无关的本地 review、邮件草稿、notebooks 或生成工件。

**问题与验证：** `WITH` 会结束 FROM 解析，hint 解析后继续出现的逗号/JOIN 表名
被当成普通表达式扫描，未进入对象授权。修复前，6 个合成变体（含子查询和 CTE）
到达了 mock executor。使用 PyPI `hana-ml==2.30.26091800`、`XinKey2`、开启加密
和证书校验执行只读 DUMMY 探针时，这 6 种写法均被 HANA 以语法错误 **257** 拒绝；
合法末尾 hint 和 hint 前的 JOIN 正常执行。因此这是已确认的 guard 缺口，不代表
已证明这些语句在所测 HANA 上能够窃取数据。

**最终修改：** `_SQLObjectParser._parse_hints()` 同时检查当前扫描范围是外层语句，
且 hint 的右括号位于语句末尾；拒绝子查询/CTE body/表达式内的 hint、重复 hint，
以及 hint 后的 JOIN、逗号、WHERE、ORDER、LIMIT、UNION 等。拒绝发生在数据库执行前。
原有 hint 参数检查及表、函数、sequence 授权保持不变，单个末尾分号继续支持。

**功能影响：** 正常 SELECT、JOIN、过滤/排序、CTE 最终 SELECT、UNION 查询的末尾
hint 继续可用；多 hint、简单参数、大小写以及包含 hint 字样的字符串均有测试。
这是将畸形 SQL 的拒绝提前到应用层，所测合法查询结果不变；不新增数据库权限。
公开 SQL API 文档同步明确允许的位置；live fixture 显式设置 `encrypt=True` 和
`sslValidateCertificate=True`，没有通过关闭安全控制获得测试通过。

**回归与验证：**

- 新增 20 个非法位置模板，每个都运行离线拒绝检查及真实 HANA 零执行检查；另新增
  10 个合法查询，每个都运行离线兼容性检查及真实 HANA 逐行结果比较，共 **60 个新测试实例**。
- 修复生产代码前，20 个新离线拒绝测试全部失败，原因是没有抛出 ValueError；其余
  102 项未选中。mock 下游执行器故意放行，不能依赖数据库语法报错假装 guard 生效。
- 修复后扩大定向运行：**487 passed、0 failed/errors/skipped、39 warnings，
  103.18 秒，退出码 0**。JUnit 独立核对一致；实际加载上述公开 PyPI 版本。
- 分组：原安全 138、assessment follow-up 41、memory/stdio 兼容 170、audit 7、
  memory adapter 6、Mem0 agent 3、SQL 122（72 离线 + 50 live HANA）。
- 修改的生产模块 Pylint **10.00/10**；编译、`git diff --check` 通过。
  fixture 已删除本轮生成的合成 R9_COMPAT 表；后续只读查询确认当前 schema 的该前缀
  表数量为 0。这些测试数据可以重跑生成，未修改无关业务对象。

证据：`.pytest_cache/review10/hint-before.xml`、`hint-verified.xml`（本地、git-ignored）。
具体改动见 `src/hana_ai/agents/hana_sql_agent.py`、
`nutest/testscripts/test_sql_agent_compatibility.py` 及公开 SQL-agent RST 文档；
复现命令见英文安全报告第十一节。

这不是全仓库或 live LLM 重跑。此前 ContextAgent 的 **2 passed / 9 failed / 2 skipped**、
部署要求和发布边界保持不变，不能将本轮结果描述为全套测试全通过。
