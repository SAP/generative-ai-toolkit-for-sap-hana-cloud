# Security feedback and test rerun — Generative AI Toolkit for SAP HANA Cloud

Originally prepared: 2026-10-27; updated: 2026-10-28. Reviewed source: `5c146c3f08ad3a580e0b316a10725d10a07cb648`, branch `security/assessment-13-fixes`, [PR #162](https://github.com/SAP/generative-ai-toolkit-for-sap-hana-cloud/pull/162). Based on [change_log.md](change_log.md), current source, historical source, and the verification below.

Latest full-suite baseline: **PyPI `hana-ml==2.30.26091800`: 361 passed, 1 failed, 11 setup errors, 2 skipped; pytest exit code 1.** All 138 security regression and 62 SQL compatibility/security tests passed. The additional stdio initialization timeout did not reproduce in an isolated retry; it remains a failure in the full-run totals. Section 6 records that unchanged-code baseline. **Sections 7–9 document subsequent production fixes, compatibility regressions, and test-fixture updates on top of `5c146c3`; they are not a new release or a replacement for the baseline full-suite results.** This PR delivery includes this report and `change_log.md`. Statements about local-only/uncommitted work in dated sections describe their historical state; raw scans, review files, probes, notebooks, and generated artifacts remain outside the commit.

Earlier follow-up verification: **442 selected security/compatibility tests passed,
0 failed/errors/skipped** with the PyPI package and `XinKey2` (section 8). After
the key/TLS fixture correction, the 13 live ContextAgent cases produced **2 passed,
9 failed, 0 setup errors, 2 skipped** (section 9). These are targeted runs, not a
full-suite rerun or an all-green result. Section 7 preserves the earlier 266-case
run and supplemental Python-boundary checks.

Latest Review10 follow-up: **487 selected tests passed, 0 failed/errors/skipped**,
including **50 live HANA checks** using PyPI hana-ml and XinKey2. Section 11 records
the statement-final WITH HINT fix on top of `d8d4714`; these changes do not
supersede the outstanding ContextAgent failures.

## 1. Feedback on the proposed technical description

**Code Injection (CWE-94) is an appropriate primary classification for the unsafe Python-execution issue.** Prompt injection is the delivery mechanism: attacker-controlled content can influence the model to submit code to a tool. Executing that code with host capabilities is the security-sensitive operation. Merely generating code is not itself the confidentiality/integrity/availability impact.

The description is directionally correct for the affected implementation/configuration, but needs the following qualifications:

| Proposed wording | Recommended correction |
|---|---|
| “The tool performs no sanitization” | The historical implementation has an AST denylist and reduced builtins. These checks are insufficient to contain the capabilities exposed by `pd`, `dataframe`, `cc`, and returned objects; they are not a sandbox. |
| “An attacker … causes the LLM to … execute arbitrary code” | An attacker **may induce** a reachable agent to invoke an enabled code-execution tool. Success depends on the deployment exposing that capability and attacker-controlled content reaching the model. |
| “Any actor … no prior system access” | No host/OS login is inherently required, but the attacker must be able to submit input to an exposed application or influence documents/table data that the application retrieves. Application authentication and ingestion permissions can be prerequisites. Do not assign unauthenticated access or a CVSS score without deployment evidence. |
| “Exfiltrate … credential stores” | Host-readable files, environment/configuration secrets, and usable credentials **may** be exposed within the process's OS privileges. Encrypted or otherwise protected credential-store contents are not automatically readable as plaintext. A live `cc` may still allow use of database credentials without revealing them. |
| “Run arbitrary SQL” | The raw connection can submit SQL permitted by its configured database principal. This is not an inherent HANA privilege escalation. The separate SQL string-construction vulnerabilities in the changelog should be distinguished from direct raw-connection use. |
| Description stated as current behavior without qualification | The reviewed revision disables general-purpose Python execution by default and adds other controls. Explicit dangerous opt-in remains unsandboxed; do not describe the default fixed configuration as still exposing the original execution path. |

### Recommended technical description

**Vulnerability type:** Code Injection (CWE-94), reachable through direct or indirect prompt injection in affected deployments. SQL-injection flaws in separate query-building paths are related findings, not the primary classification of unrestricted Python execution.

**Attack scenario:** In an affected deployment that exposes a general-purpose Python execution tool to an LLM agent, an attacker who can influence chat input, retrieved documents, or table values subsequently supplied to the model may induce the agent to submit attacker-directed code. The execution environment exposes full-capability pandas/hana-ml objects and a HANA connection. Its AST denylist and reduced builtins do not establish a security isolation boundary. If the agent invokes the tool with the injected code, execution takes place in the application's process and may access host-readable data, disclose that data through tool responses or permitted outbound channels, and submit SQL using the configured HANA principal. Exploitability and scope depend on tool registration, application access controls, retrieval behavior, host permissions, network access, and database privileges.

**Root cause:** Untrusted model-generated instructions cross into in-process execution with powerful `pd`/`dataframe`/`cc` objects, while a best-effort AST blocklist is treated as if it restricted their capabilities. Syntax restrictions are not capability isolation.

**Impact:** Potential loss of confidentiality through disclosure of host-readable secrets and database data; loss of integrity through host/file or database modifications permitted to the process and HANA principal; and loss of availability through expensive or non-terminating execution, resource exhaustion, or destructive operations within those privileges. Actual severity requires deployment-specific assessment.

**Fix boundary:** The default tool list no longer contains `python_hanaml_exec`; directly constructed instances also reject execution unless trusted host code passes the actual Boolean `allow_dangerous_code=True`. Tool-call arguments cannot enable it. The legacy dataframe agent no longer creates a Python REPL by default. These are capability-removal/default-deny changes, not a claim that arbitrary Python is now safely sandboxed or that prompt injection itself has been eliminated.

Prompt injection can still attempt to misuse capabilities intentionally exposed by other tools. Removing arbitrary Python does not establish per-user data authorization, make every tool read-only, or prevent all disclosure through otherwise permitted tool outputs. Applications must scope and authorize those capabilities separately.

## 2. Suggested answers to the requested questions

### Q1. Does this vulnerability impact AI services offered by any SAP product?

**Recommended answer:**

> The affected component identified by this review is the SAP-maintained Generative AI Toolkit for SAP HANA Cloud (`hana-ai`). Applications or AI services that deploy an affected version and expose the unsafe execution capability to untrusted agent input can be affected. The issue is in the toolkit/application execution boundary; it does not by itself demonstrate a vulnerability in the SAP HANA Cloud database service or the underlying hosted LLM service. Impact to any additional SAP product requires confirmation of that product's actual use, version, configuration, and exposure of the component.

“Yes, it impacts Generative AI Toolkit for SAP HANA Cloud” is acceptable **with the affected-version/configuration qualification**. It is not enough evidence to assert that every SAP AI service, or SAP HANA Cloud itself, is affected.

### Q2. How do customers become secure? Is customer action required?

**Recommended answer:**

> Customer action is required for customer-managed installations. Customers must upgrade to a released toolkit version confirmed to contain the relevant fixes, or consume an approved fixed build, and rebuild/redeploy/restart their application as appropriate. They must keep dangerous Python execution disabled for untrusted input and migrate workflows that previously relied on the default REPL to explicitly scoped tools. They should also apply the documented deployment requirements, including least-privilege HANA credentials and any applicable authenticated/TLS-protected MCP configuration. Installing a package without updating the running deployment, or re-enabling unrestricted Python for untrusted callers, does not establish protection.

Important release qualification: at the time checked, **PR #162 is open and unmerged**. The direct `PythonHanaMLExecTool` default-off change is already in historical commit `566de3a` (included in the local `main` baseline); PR #162 contains the additional hardening summarized in `change_log.md`, including the legacy dataframe agent and SQL fixes. The source package metadata still says `1.1.26081400`; that string alone does not identify a published fixed artifact. This report does **not** establish an affected-version range, fixed released version, or automatic rollout. Release/security owners must provide those before issuing a customer upgrade instruction with a concrete version.

For SAP-managed deployments, whether SAP can apply an update without customer action must be confirmed by the owning service team. It cannot be inferred from this OSS repository.

Additional deployment actions depend on enabled features:

- Use separate, least-privilege HANA principals: read-only approved objects for SQL-reading agents; narrowly scoped write privileges only for explicitly authorized training/import/materialization tools. The toolkit is not globally read-only.
- Configure SQL-agent schema/table allowlists; these are defense in depth, not a replacement for HANA authorization.
- Keep runtime connection-admin tools disabled unless required. When enabled, apply the host opt-in and required HTTP authentication. Use HTTPS termination for remote MCP, and prevent direct access to its plaintext backend listener.
- Configure HANA trust/CA material and approved stdio server profiles where those features are used. Do not broadly disable TLS validation to restore compatibility.
- Follow scoped-memory, session-path, private-artifact, and trusted FAISS migration guidance where applicable. Never enable unsafe pickle loading for untrusted files.
- If prior compromise is suspected, investigate and rotate/revoke affected credentials; upgrading does not recover already disclosed secrets.

### Q3. Is this related to an outbound OSS component? Which SAP internal users/products consume it?

**Recommended answer:**

> Yes. The affected code is in the SAP-maintained outbound open-source project [SAP/generative-ai-toolkit-for-sap-hana-cloud](https://github.com/SAP/generative-ai-toolkit-for-sap-hana-cloud), distributed as the Python package `hana-ai`. The issue is in the toolkit's execution integration, rather than evidence that every upstream dependency is independently vulnerable.
>
> This repository review does not identify a verified inventory of SAP internal customers, consuming services, or embedded product deployments. The component owner and SAP internal reuse/SBOM and product-security owners must confirm where it is used and whether an affected execution path is exposed. No additional affected SAP product has been established by this review; **“No other SAP products are affected” has not been substantiated and should not be stated as a confirmed conclusion.**

To close this question, obtain each consumer's product/service name, owner, deployed package/build version, enabled tools and trust boundary, external input exposure, and remediation/rollout status. Public integration with HANA or GenAI Hub is not proof that those services embed the vulnerable component.

## 3. Current remediation evidence and remaining boundaries

The identifiers below are local assessment references from `change_log.md`, not CVE identifiers. Passing regression tests confirms specific behavior, not the absence of all vulnerabilities.

| Assessment items | Current controls verified in code / covered by tests | Remaining boundary |
|---|---|---|
| V-156, V-168: Python/REPL execution | Python tool defaults off, is absent from default toolkit registration, and rejects caller-supplied opt-in; legacy dataframe agent requires explicit safe tools by default and no longer places the first row in its prompt. | Host opt-in still allows in-process execution with host/database capabilities. No sandbox, resource isolation, or Python-tool execution timeout. |
| V-159, V-171; review7–9: SQL execution/authorization | Shared single-read-only-statement guard before affected hana-ml sinks; SQL-agent object parsing checks nested CTEs/tables/functions/sequences; numeric-token and JOIN-comma bypass regressions; schema/table restrictions and exact quoted-name semantics. | Application checks do not prove HANA grants. The shared dataframe guard is not the SQL agent's object allowlist. Purpose-built write tools remain write-capable. |
| V-157, V-158, V-166: query and identifier construction | Literal/identifier encoding, numeric/function allowlists, semicolon rejection on affected raw-splitting hana-ml paths, and trusted individual DBAPI execution for PSE/certificate statements. | These paths are not universally parameter-bound. Legitimate embedded semicolons remain rejected at unsafe hana-ml APIs; SQL-agent string literals can preserve them. |
| V-155, V-164, V-165, V-167: MCP and transport | Runtime-admin opt-in/authentication gates, non-loopback transport restrictions, no automatic HTTP redirects, TLS defaults, and explicit stdio executable/argument profiles. | Remote HTTPS termination is a deployment task. Local process trust and enabled custom/admin tools still require authorization. |
| V-161–163: persistence/artifacts/secrets | Trusted FAISS loading/integrity checks, private randomized artifact paths, and certificate/private-key temporary-file cleanup on exception paths. | Same-OS-user writers are outside the FAISS integrity trust boundary; production legacy-store migration has not been rehearsed here. |
| V-169–174: identity, memory, rendering | Tenant/session scoping, disabled chat-admin commands by default, collision-resistant session paths, raw-HTML opt-in, stable authenticated database identity. | Applications must provide trustworthy tenant/session identity; raw HTML opt-in and host-provided custom tools remain trusted-only features. |
| V-160, V-175, V-176: dependencies/build | Import-time installation removed; dependency/build and action pinning reviewed in the changelog. | This rerun is not a fresh clean-install/build or cross-platform dependency audit. Hash locks target the documented Python 3.11/Linux workflows; no universal cross-platform assurance. |

Review9 compatibility changes restore `CAST`, `TO_VARCHAR`, `ANY/SOME`, hints, partitions, lateral queries, and actual HANA `TABLESAMPLE SYSTEM/BERNOULLI` syntax without removing the relevant object checks. The PSE joined-string helper is documented as deprecated and must not be executed through hana-ml raw splitting.

### Supplemental security checks — 2026-10-27 baseline

- `_review8_xinkey2_probe.py` completed with exit code 0 against HANA user/schema `PAL_TEST` using `XinKey2` and certificate validation enabled. Injected vector/split canary DML did not execute. Parenthesized-table, CTE-scope, UDF, sequence, and numeric-prefix payloads were rejected; SQLAlchemy execution-event count for rejected SQL-agent queries was **0**. Legal grouped/outer-join and quoted mixed-case factory queries returned the expected values. Probe-created UUID-named objects were dropped without reported cleanup errors.
- After the review8 probe and review9 compatibility tests completed, a read-only `SYS.OBJECTS` check found **0** objects matching `R8_%` or `R9_%` in the current schema.
- Supplemental local Python-boundary tests: **8 passed in 1.53 seconds**. Default execution rejected synthetic file-read/raw-SQL inputs, caller arguments could not enable execution, and non-Boolean host values did not enable it. Explicit host opt-in could read a test-only CSV and call a **mocked** raw SQL method, confirming the documented lack of a sandbox. No real secrets were read, no external exfiltration was attempted, and no SQL from this supplemental check reached a real database. These are direct-tool checks, not an LLM prompt-injection penetration test.
- `python -m pylint ./src`: **10.00/10**.
- `python -m compileall -q src/hana_ai nutest` and `git diff --check`: passed.

## 4. Baseline full-suite results — editable hana-ml, 2026-10-27

Environment: Windows, Python 3.13.11, pytest 9.1.1; hana-ml 2.31.26120800, hdbcli 2.30.27, SQLAlchemy 2.0.48, FastMCP 2.14.7. `hana_ai` loads from this repository's `src` directory; `hana_ml` loads from the adjacent local `../hanamlapi/src/hana_ml` checkout. This is a source-workspace integration run, not a clean installation of a released wheel. HANA userstore key `XinKey2` resolves to current user/schema `PAL_TEST`. No credential values are included in this report.

Unlike the previous 362-pass/13-skip run, **all three live opt-in flags were enabled**:

```powershell
$securityTestRoot = (Resolve-Path -LiteralPath 'nutest/testscripts').Path
$env:PYTHONPATH = $securityTestRoot + [IO.Path]::PathSeparator + $env:PYTHONPATH
$env:HANA_USERKEY = 'XinKey2'
$env:RUN_CONTEXT_AGENT_E2E = '1'
$env:RUN_CONTEXT_AGENT_MCP_E2E = '1'
$env:RUN_CONTEXT_AGENT_MCP_STDIO_E2E = '1'
python -m pytest --import-mode=importlib -ra --tb=short --junitxml=.pytest_cache/security-feedback/full-suite.xml
```

**Final result: 375 collected; 362 passed, 0 assertion failures, 11 setup errors, 2 skipped, 65 warnings.** Duration: **1792.25 seconds (29 minutes 52 seconds)**; pytest exit code: **1**. The errors are failures of the test run even though pytest reports them separately from assertion failures. **This opt-in-enabled run is not all-green.** Warnings concern dependency deprecations and the existing SQLAlchemy `case_sensitive` reflection argument.

JUnit results independently confirm the following non-overlapping groups:

| Test group | Passed | Setup errors | Skipped |
|---|---:|---:|---:|
| `test_security_fixes.py` | 138 | 0 | 0 |
| `test_sql_agent_compatibility.py` | 62 | 0 | 0 |
| `test_hana_mcp_audit_view.py` | 7 | 0 | 0 |
| `test_appsource_pack.py` | 10 | 0 | 0 |
| Remaining executable unit/integration tests | 145 | 0 | 0 |
| Live ContextAgent scenarios | 0 | 11 | 0 |
| Live ContextAgent + HTTP/stdio MCP scenarios | 0 | 0 | 2 |
| **Total** | **362** | **11** | **2** |

The 62 SQL compatibility/security tests include 42 offline cases and 20 live HANA cases (19 valid-query comparisons and one rejected-JOIN zero-execution check). The separate 8 synthetic Python-boundary checks and one-test diagnostic rerun are not added to these totals.

### Previously skipped live scenarios

The 11 tests in `nutest/testscripts/test_context_agent_e2e_scenarios.py` fail during their common class setup, before exercising an LLM scenario. A separate single-test diagnostic reproduced `hdbcli.dbapi.Error (-10104): Invalid value for KEY "RaysKey"` at line 102. That fixture hardcodes `RaysKey` instead of honoring `HANA_USERKEY=XinKey2`. Its LLM initialization comes after the failing connection creation, so LLM connectivity has not been tested by these cases.

| Test method | Baseline status |
|---|---|
| `test_e2e_data_preview_and_dataset_report_scenario` | Setup error: hardcoded HANA key |
| `test_e2e_forecast_train_predict_scenario` | Setup error: hardcoded HANA key |
| `test_e2e_prediction_visualization_and_insight_scenario` | Setup error: hardcoded HANA key |
| `test_e2e_prediction_visualization_renders_in_notebook_scenario` | Setup error: hardcoded HANA key |
| `test_e2e_cap_artifact_generation_scenario` | Setup error: hardcoded HANA key |
| `test_e2e_dataset_import_split_and_profile_scenario` | Setup error: hardcoded HANA key |
| `test_e2e_prediction_auto_repairs_labelled_holdout_scenario` | Setup error: hardcoded HANA key |
| `test_e2e_outlier_detection_scenario` | Setup error: hardcoded HANA key |
| `test_e2e_massive_forecasting_scenario` | Setup error: hardcoded HANA key |
| `test_e2e_massive_additive_forecasting_scenario` | Setup error: hardcoded HANA key |
| `test_e2e_massive_outlier_detection_scenario` | Setup error: hardcoded HANA key |

The two opt-in MCP tests still skip because their VCAP-services fixture is absent:

- `test_context_agent_mcp_reload_connection_context_from_file_http.py::TestContextAgentMCPReloadConnectionContextFromFileHTTP::test_context_agent_calls_mcp_reload_from_file` — expected VCAP-services file not found; the HTTP fixture allows `VCAP_SERVICES_FILE` as an override.
- `test_context_agent_mcp_reload_connection_context_from_file_stdio.py::TestContextAgentMCPReloadConnectionContextFromFileSTDIO::test_context_agent_reload_switches_connection` — fixed VCAP-services fixture missing; this test also hardcodes `RaysKey` for its server.

Further static fixture issues to address before a successful live run:

- The required `nutest/testscripts/env.example` VCAP-services fixture is absent. The sales CSV used by the scenario suite exists at `nutest/testscripts/demo/e2e_scenarios/SALES_REFUNDS.csv`; its existence does not resolve the earlier HANA connection setup failure.
- The HTTP admin scenario does not configure the new runtime-admin opt-in and authenticated client/server flow; the stdio scenario lacks an approved server profile. Adapt the test fixtures to the documented security controls instead of disabling those controls in production. These are static observations, not additional executed test failures.
- The scenario/stdio fixtures explicitly disable HANA certificate validation and should be aligned with the intended secure connection settings when updated.

The rerun leaves the committed code/tests unchanged to report their actual current behavior. It does not silently substitute credentials, create userstore aliases, disable guards, or count setup failures/skips as passed tests.

## 5. Evidence scope and release follow-up

Local machine-readable reports are in `.pytest_cache/security-feedback/` (git-ignored): `full-suite.xml`, `optin-diagnostic.xml`, and `python-boundary.xml`. The supplemental boundary test and synthetic CSV are in the same directory. They are not part of the 375 committed-suite test count.

Remaining work before a complete security/customer statement:

1. Identify an approved fixed release/build and affected-version range; confirm the status of all fixes needed for the advisory.
2. Confirm actual SAP-internal consumers and affected product configurations with component/product owners. Do not infer a negative product-impact conclusion from a repository-only review.
3. Repair/configure the opt-in live test fixtures, then run those 13 scenarios successfully. HANA key availability alone is not sufficient.
4. Separately validate deployment authorization, remote TLS termination, actual LLM-driven adversarial inputs, approved custom tools, and resource/egress restrictions as applicable. Unit tests and SQL canaries are not substitutes for those assessments.
5. Investigate the stdio initialization timeout observed in the PyPI full run (section 6); a successful isolated retry does not establish full-suite reliability.

### Source references

- [Python tool trust boundary and host opt-in](src/hana_ai/tools/hana_ml_tools/python_exec_tools.py). Historical comparison: `196fdeb` (AST-restricted executor) versus `566de3a` (default-off capability boundary).
- [Toolkit default registration and connection refresh](src/hana_ai/tools/toolkit.py).
- [Legacy dataframe-agent safe default](src/hana_ai/agents/hana_dataframe_agent.py).
- [Shared hana-ml SQL guard](src/hana_ai/sql_guard.py) and [SQL-agent object authorization](src/hana_ai/agents/hana_sql_agent.py).
- [Security regression tests](nutest/testscripts/test_security_fixes.py) and [SQL compatibility/security tests](nutest/testscripts/test_sql_agent_compatibility.py).
- [README deployment boundaries](README.md), [public changelog](doc/changlog.rst), and [local remediation/migration history](change_log.md).

This is a technical verification and response draft, not a product-owner attestation, released security advisory, CVSS assessment, or proof that every configuration is secure.

## 6. PyPI hana-ml verification — 2026-10-28

### Package selection and isolation

- Tested package: **`hana-ml==2.30.26091800`**, the latest release reported by [PyPI](https://pypi.org/pypi/hana-ml/2.30.26091800/json) on 2026-10-28. This satisfies `requirements-runtime.txt` (`>=2.29.26072000,<3`). It is newer than the development pin `2.29.26072000`; this run tests the latest public wheel, not that older pin.
- Wheel: `hana_ml-2.30.26091800-py3-none-any.whl`. SHA-256 verified against PyPI metadata: `73fc9f2df49d6e466ec0f334e4b56d530b34f47d72560e5790efa8c466f9ae3f`.
- Created `.pytest_cache/pypi-hana-ml-2026-10-28/venv` with `--system-site-packages` and installed the verified wheel there with `--no-deps --ignore-installed`. The original Conda/editable hana-ml installation was not uninstalled or changed.
- Other installed dependencies are deliberately reused for comparison. This is **not a fully clean dependency installation**, and `hana_ai` is still the current repository source, not a published toolkit wheel.
- Verified `hana_ml.__version__` and distribution metadata both report `2.30.26091800`, and `hana_ml.__file__` is under the test venv's `Lib/site-packages/hana_ml`, not `../hanamlapi`.
- On Windows, a bare `python` subprocess initially selected the Conda executable and editable source despite the adjusted `PATH`. Prepending the test venv's site-packages to the test process's `PYTHONPATH` resolved the import. A second subprocess check explicitly confirmed it loads the same PyPI hana-ml package. The full suite started only after these checks passed. No persistent user/system environment variables were modified.

### Reproduction

Initial setup (the venv path must be new):

```powershell
python -m venv --system-site-packages .pytest_cache/pypi-hana-ml-2026-10-28/venv
python -m pip download --no-deps --index-url https://pypi.org/simple --only-binary=:all: --dest .pytest_cache/pypi-hana-ml-2026-10-28/packages hana-ml==2.30.26091800
# Verify the downloaded wheel's SHA-256 against the value above before installation.
& ./.pytest_cache/pypi-hana-ml-2026-10-28/venv/Scripts/python.exe -m pip install --no-deps --ignore-installed --report .pytest_cache/pypi-hana-ml-2026-10-28/install-report.json .pytest_cache/pypi-hana-ml-2026-10-28/packages/hana_ml-2.30.26091800-py3-none-any.whl
```

Full-suite command:

```powershell
$pypiTestScripts = (Resolve-Path -LiteralPath '.pytest_cache/pypi-hana-ml-2026-10-28/venv/Scripts').Path
$pypiTestSite = (Resolve-Path -LiteralPath '.pytest_cache/pypi-hana-ml-2026-10-28/venv/Lib/site-packages').Path
$pypiTestPython = Join-Path $pypiTestScripts 'python.exe'
$env:PATH = $pypiTestScripts + [IO.Path]::PathSeparator + $env:PATH
$env:PYTHONPATH = $pypiTestSite + [IO.Path]::PathSeparator + (Resolve-Path -LiteralPath 'src').Path + [IO.Path]::PathSeparator + (Resolve-Path -LiteralPath 'nutest/testscripts').Path
$env:HANA_USERKEY = 'XinKey2'
$env:RUN_CONTEXT_AGENT_E2E = '1'
$env:RUN_CONTEXT_AGENT_MCP_E2E = '1'
$env:RUN_CONTEXT_AGENT_MCP_STDIO_E2E = '1'
& $pypiTestPython -u -m pytest --import-mode=importlib -ra --tb=short --junitxml=.pytest_cache/pypi-hana-ml-2026-10-28/full-suite.xml
```

### Results

Full-suite result: **375 collected; 361 passed, 1 failed, 11 setup errors, 2 skipped, 65 warnings**, in **1735.72 seconds (28 minutes 55 seconds)**; pytest exit code **1**. All three live opt-in flags were enabled, as in the source baseline. The test fixtures and production code remain unchanged. **This run is not all-green.**

| Run | hana-ml source/version | Passed | Failed | Setup errors | Skipped | Exit code |
|---|---|---:|---:|---:|---:|---:|
| 2026-10-27 baseline | Editable source, 2.31.26120800 | 362 | 0 | 11 | 2 | 1 |
| 2026-10-28 PyPI | Verified wheel, 2.30.26091800 | 361 | 1 | 11 | 2 | 1 |

The JUnit comparison confirms the same 375 test IDs and only one changed outcome: the non-LLM stdio connection-reload test changed from passed to failed. All other case outcomes match the baseline.

Security-focused results, verified from the PyPI run's JUnit file:

- `test_security_fixes.py`: **138 passed**, no failures/errors/skips.
- `test_sql_agent_compatibility.py`: **62 passed**, including 42 offline and 20 live HANA cases, no failures/errors/skips.
- `test_hana_mcp_audit_view.py`: **7 passed**; `test_appsource_pack.py`: **10 passed**.
- The remaining cases comprise 144 passed, the one stdio failure, 11 live ContextAgent setup errors, and two live MCP skips.

### Additional full-run failure and isolated retry

`nutest/testscripts/tools/test_mcp_server_reload_connection_context_from_env_stdio.py::TestMCPReloadConnectionContextFromFileSTDIO::test_reload_connection_context_from_file_stdio` failed at `await client.initialize()`, **before the admin reload call**. The RPC did not complete within the test's **120-second timeout** (`STDIO RPC timeout for method=initialize`); JUnit records 120.085 seconds for the case. The captured server stderr tail contains dependency deprecation warnings, not a definitive cause of the initialization delay.

An immediate isolated rerun under the **same PyPI environment and unchanged timeout/test code passed: 1 passed in 45.75 seconds**, exit code 0. Its JUnit result is `stdio-reload-diagnostic.xml`. This does not erase the original full-run failure. The failure was not reproduced in isolation; the available evidence does **not** establish whether it arose from timing, suite interaction, environment behavior, or a hana-ml change. Do not claim either a confirmed PyPI incompatibility or a fixed timeout.

The 11 setup errors and two skips have the same causes as the baseline: the ContextAgent fixture hardcodes unavailable `RaysKey`, and the two LLM-driven MCP tests lack their VCAP-services fixture. The new error traces confirm `hana_ml/dataframe.py` was loaded from the test venv's published package. Changing the dependency does not resolve these fixture prerequisites; all 13 live LLM scenarios remain unverified.

### Supplemental checks

Supplemental Python-boundary checks against the PyPI package: **8 passed in 2.38 seconds**; these are separate from the committed full-suite count. The test inputs remain synthetic and the raw SQL method is mocked.

Additional PyPI-environment evidence:

- `_review8_xinkey2_probe.py`: exit code **0**. Both injected vector/split canary writes were blocked; unauthorized table/CTE/UDF/sequence and numeric-prefix queries were rejected; rejected SQL-agent queries produced **0 execution events**. Legal grouped/outer-join and quoted mixed-case factory queries still returned their expected values.
- After the probe and SQL compatibility tests, the current HANA schema contained **0 `R8_*`/`R9_*` objects**.
- `python -m pylint ./src` under the PyPI test environment: **10.00/10**. Compilation and `git diff --check` passed.

The published wheel's `hana_ml/dataframe.py` still contains raw `sql.split(";")` processing (line 1713). The documented semicolon restrictions at these hana-ml execution paths remain relevant; switching from the editable checkout to this PyPI release does not remove that behavior.

PyPI-run artifacts are under `.pytest_cache/pypi-hana-ml-2026-10-28/`: downloaded wheel, installation report, test venv, `full-suite.xml`, `stdio-reload-diagnostic.xml`, and `python-boundary.xml`. The isolated retry and eight supplemental checks are **not added to or substituted into** the 375-case full-suite totals. The baseline artifacts in `.pytest_cache/security-feedback/` were preserved.

PR #162 was rechecked on 2026-10-28 and remains open/unmerged at `5c146c3`. A public hana-ml release is not a released fix of the separate `hana-ai` toolkit; the release qualification and product-scope caveats in section 2 still apply.

## 7. Assessment-13 package reconciliation and follow-up fixes — 2026-10-28

Reviewed the local `13-hana-ml` scan package, its index/tracker,
Critical/High remediation criteria, all **13 supplied patch files**, both local
reports, and the corresponding current implementations. The package targets
`main @ 5cc559b`; its October 20 open/partial labels are historical, not a scan of
the current branch. The source inspected here is `5c146c3` plus this local follow-up.
The external assessment package was left unchanged.

### Disposition of every supplied patch

All 13 proposals have corresponding protections in the current branch. They were
not blindly reapplied: later fixes correct shortcomings in the proposals, and
reapplying the old diffs would undo compatibility/security work. “Covered” below
means the security intent is implemented, **not byte-for-byte patch application**.

| Supplied patch filename | Current disposition |
|---|---|
| `V-155-mcp-admin-auth-tls-guard.patch` | Covered and extended: default-off admin registration, actual HTTP bearer verification, SSE restrictions; this follow-up makes omitted update TLS options and env encryption explicitly secure. |
| `V-157-vectorstore-query-parametrize.patch` | Covered and extended: literal/identifier encoding, function allowlist, result-count bound, semicolon rejection. This follow-up validates constructor and lazy metadata paths before hana-ml calls. Despite the patch filename, this is not universal DB parameter binding. |
| `V-158-retrieval-procedure-identifier-quote.patch` | Covered: quoted procedure/schema components; subsequent infrastructure/PSE/certificate encoding remains intact. |
| `V-159-smart-dataframe-strict-sql-extract.patch` | Superseded by the stricter shared whole-statement guard. The supplied patch still extracts a SELECT substring and only checks the initial WITH keyword; the current scanner also checks the CTE body/top-level operation and hana-ml raw semicolon splitting. |
| `V-160-remove-runtime-pip.1-nltk.patch` | Covered with a lazy optional import; no runtime installation. |
| `V-160-remove-runtime-pip.2-mcp.patch` | Covered for both MCP packages; no runtime installation. |
| `V-160-remove-runtime-pip.3-langgraph.patch` | Covered; missing dependency fails with installation guidance. |
| `V-161-faiss-safe-deserialization.patch` | Covered and extended: private default storage, reject untrusted/incomplete existing stores without overwriting, integrity-checked local reload and explicit trusted migration. The old broad exception fallback must not be restored. |
| `V-166-groupkey-escape.1-predict.patch` | Covered and extended to remaining string/numeric/date branches, identifiers, and unsafe semicolon paths. |
| `V-166-groupkey-escape.2-ts-check.patch` | Covered, with numeric validation and identifier handling retained. |
| `V-175-pin-deps.1-requirements.patch` | Security intent retained with corrected compatible pins and Python 3.11/Linux hash locks; do not downgrade to the proposal's incompatible dependency set. |
| `V-175-pin-deps.2-requires-python.patch` | Superseded by `requires-python >=3.10`, reflecting the source syntax rather than the proposal's >=3.9. |
| `V-176-cicd-sha-pin.release.patch` | Covered: the release action SHAs are present; other workflows and build locks have also been hardened. |

No patch was supplied for V-156; its default-off host capability boundary is
already present. V-162/163/164/165/168 required manual implementation and already
have private artifacts, exception cleanup, transport/profile controls, and no
default dataframe REPL. This follow-up closes the additional gaps below. V-167
(context-adjusted High) is included in the TLS work.

### New confirmed gaps and modifications

1. **V-155/V-167 — explicit TLS defaults.** The authenticated connection-update
   tool previously omitted driver TLS arguments when callers omitted them or sent
   null. It now sets both encryption and certificate verification to true for
   both the optional connection test and actual replacement. The environment
   builder also defaults `HANA_ENCRYPT` to true for userkey and password flows.
   An explicit TLS downgrade still requires acknowledgement; host environment
   opt-outs remain deliberate deployment decisions, not safe defaults.
2. **V-157 — validation before metadata/DDL.** The final vector SELECT was guarded,
   but constructor `has_table()` and lazy `.columns` access preceded validation.
   PyPI hana-ml's `has_table()` interpolates names into single-quoted metadata SQL.
   Configuration is now checked before constructor metadata/DDL, reads, upserts,
   and query column lookup. Single quotes in table/schema names are rejected at
   this API boundary; input/model literals retain apostrophe escaping. Invalid
   query text is rejected before lazy column access. This is toolkit-side
   hardening, not a modification or security attestation of the dependency.
3. **V-163 — early-failure credential cleanup.** File paths are now recorded
   immediately after creation, before encoding/writing. Previously an encoding
   error or partial-write failure left a `delete=False` file untracked by finally.
   Cleanup directly attempts deletion and tolerates missing files, without a
   separate existence check. OS deletion failures still produce warnings; process
   termination and hostile same-user access are not solved by finally.
4. **V-164 — strict insecure-HTTP opt-in.** Only Boolean True enables remote
   plaintext HTTP. Truthy strings such as `"false"`, integers, and mappings no
   longer accidentally disable the TLS requirement. Redirect following stays off.
5. **V-165 — shell/batch rejection independent of profiles.** Known shell names
   and `.bat`/`.cmd`/`.ps1` entrypoints are rejected before profile matching; the
   resolved executable is checked too. Previously a matching host profile could
   admit these despite the documented prohibition. This is defense against unsafe
   configuration, not evidence that a remote caller could edit host profiles.
   Trusted executable/argument profiles remain required; this is not a general
   executable-content sandbox.

The regression file `nutest/testscripts/test_assessment13_followup.py` uses
synthetic credential material, inert executable placeholders, and mocked database
connections for adversarial inputs. Its HTTP auth test exercises the actual
toolkit launcher's FastMCP ASGI app: missing/wrong bearer tokens return 401 without
calling the connection constructor; the correct token reaches the tool.

### Closure qualifications

The Critical/High findings have code protections for the documented safe
configuration, but **not every original scanner checkbox is literally satisfied**:
Python execution is disabled rather than sandboxed; loopback-only HTTP is retained
instead of changing the default URL to HTTPS; SQL encoding is not universal
parameter binding; dependency hash-lock validation is target-specific. These are
explicit remediation choices, not grounds to assert “zero remaining risk.” Do not
enable dangerous Python/pickle/HTTP options for untrusted callers or data. Least-
privilege HANA roles, remote HTTPS termination, trusted host profiles/storage,
release identification, and product-consumer confirmation remain required.

The prior full-suite failures/skips in section 6 remain historical evidence and
are not erased by targeted passing tests. This follow-up does not run the missing
LLM/VCAP scenarios or establish a fully green full-suite run.

### Follow-up verification results

Final selected run: **266 passed, 0 failed, 0 errors, 0 skipped, 42 warnings** in
**277.75 seconds (4 minutes 38 seconds)**, exit code **0**. JUnit independently
confirms the totals. The runtime was the existing isolated PyPI
`hana-ml==2.30.26091800` environment from section 6, with `HANA_USERKEY=XinKey2`;
the package version and site-packages import path were rechecked. Other dependencies
remain shared with the base environment; this is not a clean-install test.

| Selected test group | Passed |
|---|---:|
| New Assessment-13 follow-up regressions | 41 |
| Existing security regressions | 138 |
| SQL compatibility/security (42 offline + 20 live HANA) | 62 |
| Audit view + appsource pack | 17 |
| Live vector create/upsert/query | 1 |
| MCP HTTP/stdio env reload, HTTP/stdio/SSE schemas, env parameter unit test | 7 |
| **Total** | **266** |

Supplemental local Python-boundary checks: **8 passed in 3.94 seconds**, exit 0;
these are not added to the 266-case count. They reconfirm default denial, caller
opt-in rejection, and the deliberately unsandboxed trusted-host mode using only
a synthetic CSV and mocked SQL. All-source Pylint: **10.00/10**. Compilation and
`git diff --check` passed. Warnings concern dependency deprecations and the existing
SQLAlchemy reflection argument, not test failures.

Intermediate evidence was preserved rather than replaced:

- `before.xml`: 17 failing/7 passing newly added cases before the initial code fixes.
- `vector-before.xml`: nine failing pre-validation regressions before the vector fix.
- `security.xml`: first narrower fixed-code run, 247 passed.
- `final-security.xml`: expanded intermediate run, 263 passed/1 failed. The failure
  was an existing unit-test expected dictionary missing the newly required
  `encrypt=True`. That expectation was updated; TLS was not weakened to pass it.
- `verified-security.xml`: the final 266-pass rerun, also including two additional
  shell/alias tests. The live stdio reload passed, but this does not establish a
  root cause or fix for the earlier full-suite initialization timeout.
- `python-boundary.xml`: the separate eight supplemental checks.

All these artifacts are under `.pytest_cache/assessment13-followup/`. The earlier
375-case full-suite artifacts remain untouched. No full-suite, LLM adversarial,
Windows ACL, release-wheel, or production authorization reassessment is claimed.

Reproduce after the PowerShell environment setup in section 6:

```powershell
& $pypiTestPython -u -m pytest `
  nutest/testscripts/test_assessment13_followup.py `
  nutest/testscripts/test_security_fixes.py `
  nutest/testscripts/test_sql_agent_compatibility.py `
  nutest/test_hana_mcp_audit_view.py `
  nutest/test_appsource_pack.py `
  nutest/testscripts/vectorstore/test_hana_vector_engine.py `
  nutest/testscripts/tools/test_mcp_server_reload_connection_context_from_env_http.py `
  nutest/testscripts/tools/test_mcp_server_reload_connection_context_from_env_stdio.py `
  nutest/testscripts/tools/test_mcp_server_tool_schema.py `
  nutest/testscripts/tools/test_mcp_server_tool_schema_stdio.py `
  nutest/testscripts/tools/test_mcp_server_tool_schema_sse.py `
  nutest/testscripts/tools/test_vcap_services_file_parsing.py `
  --import-mode=importlib -q --tb=short `
  --junitxml=.pytest_cache/assessment13-followup/verified-security.xml
```

## 8. Compatibility regressions and security-driven behavior changes (2026-10-28)

Two regressions found during the PR compatibility review have been corrected:

1. **Valid memory filters discarded results.** The duplicated post-filters only
   handled equality, `$contains` and `$lte`. The shared
   `src/hana_ai/mem0/metadata_filters.py` now handles scalar/comparison,
   membership, BETWEEN, LIKE, typed-date and nested AND/OR conditions for adapter
   search, manager export and the default HANA export handler. All sibling field
   conditions and field operators are checked. Malformed/incomparable conditions
   fail closed without aborting the result batch. Tenant/session scope is still
   added by the manager and checked locally even when the backend ignores filters.
   The memory API's list-membership `$contains` semantics are retained; this does
   not emulate HANA full-text search or every implicit database type conversion.
2. **Approved absolute stdio paths were mistaken for shell flags.** Multi-component
   paths such as `/code/server.py` and `/config/server.py` now pass validation when
   approved by the host profile, including paths containing spaces. Exact resolved
   executable/argument matching, explicit opt-in for additional arguments, and
   shell/batch denial remain. Inline `-c`/`-e`/`--eval` and `/c`/`/cCOMMAND` switches
   remain rejected; ambiguous single-component `/c...` arguments are still denied.
   No arbitrary executable-content sandbox is claimed.

The initial 157 new cases reproduced **70 failures / 87 passes** before the fix.
The first corrected-code run with existing memory/security/follow-up suites passed
**342 tests in 42.77 seconds**. Additional cases cover multiple field conditions,
malformed/incomparable NOT IN operands and paths containing spaces. Memory tests
use deliberately unfiltered backend responses containing other tenants/sessions;
nested OR and fallback search APIs cannot override the outer scope. Attack-shaped
stdio arguments are passed only to the validator, never executed.

### Are the intentional functionality changes caused by security remediation?

**Mostly yes, but the security requirement is not the same as its exact
implementation. The two regressions above are bugs, not necessary security
tradeoffs.**

| Intentional behavior change | Security reason and qualification |
|---|---|
| Default-off generic Python / legacy dataframe REPL | V-156/V-168: prevent untrusted agent input from reaching host-capable code execution. Dangerous host opt-in is not a sandbox. The direct Python tool's default-off setting already existed at this PR's base; it is not all newly introduced by this PR. |
| Read-only/object/input restrictions on affected SQL paths | V-157/158/159/166/171: injection and authorization controls. Dedicated write tools remain write-capable. Raw semicolon splitting in the tested PyPI hana-ml still requires restrictions at the affected sinks. |
| MCP admin opt-in/authentication, TLS, no automatic redirects, trusted stdio profiles | V-155/164/165/167: unauthorized management, credential disclosure and process-execution controls. Remote HTTPS and least-privilege deployment remain required. |
| Tenant/session memory scope, trusted FAISS loading, isolated session storage | V-161/169/170/172: cross-tenant isolation, path separation and unsafe-deserialization controls. Applications must supply trustworthy identities. |
| Private randomized artifact paths, HTML opt-in, temporary-key cleanup | V-162/163/173: shared-file exposure, unsafe rendering and secret-residue controls. |
| No runtime pip installation; dependency/CI pinning and verification | V-160/175/176: supply-chain and reproducibility controls. Required extras must be installed in advance. |

**Implementation/compatibility choices:** the particular session hash/migration
scheme, artifact deletion at process exit, exact package pins/Python minimum, and
blanket rejection of SQL comments or some otherwise-safe grammar are not the only
possible remediations. These can be refined with tests while retaining isolation,
safe parsing and trusted dependency boundaries. This change does not revert them.

Both Markdown reports remain local untracked files. No commit or push was made.

### Final regression verification

**442 passed, 0 failed, 0 errors, 0 skipped, 42 warnings in 216.20 seconds;
exit code 0.** JUnit independently confirms the totals. The existing isolated
environment loaded public PyPI `hana-ml==2.30.26091800` from its venv site-packages,
not the adjacent editable hanamlapi checkout. Live tests used `HANA_USERKEY=XinKey2`.
Other dependencies still come from the base environment; this is not a clean
dependency-install test.

| Test group | Passed |
|---|---:|
| New memory/stdio compatibility regressions | 170 |
| Existing memory adapter tests | 6 |
| Existing security regressions | 138 |
| Assessment-13 follow-up tests | 41 |
| SQL compatibility/security (42 offline + 20 live HANA) | 62 |
| Audit view + appsource pack | 17 |
| Live vector create/upsert/query | 1 |
| MCP reload/schema and env parameter tests | 7 |
| **Total** | **442** |

The real stdio reload test passed in **34.172 seconds**. All-source Pylint scored
**10.00/10**; compileall and `git diff --check` passed. The 42 warnings are dependency
deprecations and the previously recorded SQLAlchemy reflection warning.

Evidence under `.pytest_cache/assessment13-followup/`:

- `compatibility-before.xml`: initial 157 cases, 70 failing before the fixes.
- `compatibility-fixed.xml`: first 342-case passing run.
- `compatibility-verified.xml`: final expanded 442-case passing run.

This targeted run does **not** replace the earlier full-suite record, execute the
LLM cases lacking RaysKey/VCAP configuration, or establish the root cause of the
earlier intermittent stdio initialization timeout. No full-suite-green claim is made.

Reproduce after the PowerShell PyPI environment setup in section 6:

```powershell
& $pypiTestPython -u -m pytest `
  nutest/testscripts/test_pr_compatibility_regressions.py `
  nutest/testscripts/test_hana_mem0_adapter.py `
  nutest/testscripts/test_assessment13_followup.py `
  nutest/testscripts/test_security_fixes.py `
  nutest/testscripts/test_sql_agent_compatibility.py `
  nutest/test_hana_mcp_audit_view.py `
  nutest/test_appsource_pack.py `
  nutest/testscripts/vectorstore/test_hana_vector_engine.py `
  nutest/testscripts/tools/test_mcp_server_reload_connection_context_from_env_http.py `
  nutest/testscripts/tools/test_mcp_server_reload_connection_context_from_env_stdio.py `
  nutest/testscripts/tools/test_mcp_server_tool_schema.py `
  nutest/testscripts/tools/test_mcp_server_tool_schema_stdio.py `
  nutest/testscripts/tools/test_mcp_server_tool_schema_sse.py `
  nutest/testscripts/tools/test_vcap_services_file_parsing.py `
  --import-mode=importlib -q --tb=short `
  --junitxml=.pytest_cache/assessment13-followup/compatibility-verified.xml
```

## 9. XinKey2 fixture update and previously blocked live scenarios (2026-10-28)

The user confirmed `XinKey2`, not `XinKeys2`. Three automated-test fixtures now
use `HANA_USERKEY` with a default of `XinKey2`: the shared `testML_BaseTestClass.py`,
`test_context_agent_e2e_scenarios.py`, and the ContextAgent stdio file-reload test.
The two latter fixtures no longer hardcode `RaysKey`. No userstore entries,
production code, demo notebooks, or installed dependency files were changed.

The first 13-case rerun progressed past the missing-key problem but exposed an
existing Windows TLS fixture error: a certifi CA PEM was passed as `sslKeyStore`,
causing `CertOpenSystemStoreA / 0x80070057`. That run had **11 setup errors and
2 skips in 118.92 seconds**, with no scenario assertions reached. The scenario
fixture now uses native HANA-client certificate trust, removes that incorrect
keystore argument, keeps `encrypt=True`, and changes `sslValidateCertificate`
from False to True. Certificate validation was not disabled to make it connect.

The second run completed with **2 passed, 9 failed, 0 errors, 2 skipped,
2 warnings in 863.51 seconds (14 minutes 23 seconds), exit code 1**. All 11
non-MCP scenarios reached execution. The missing-key/TLS setup failures are
resolved, but the live E2E suite is **not green**.

Environment: public PyPI `hana-ml==2.30.26091800` in the existing isolated venv,
`HANA_USERKEY=XinKey2`, and all three ContextAgent live opt-in flags enabled.
Package version and import path were rechecked. Supplemental checks confirmed
the shared fixture can execute a live `SELECT 1` with its default key and honors
an explicit environment override with a mocked connection. Those two diagnostics
are not added to the pytest counts.

### Per-scenario results

The following names are suffixes of `test_e2e_..._scenario` in
`test_context_agent_e2e_scenarios.py`. Failure messages come from JUnit; the
interaction details were also inspected in this run's synthetic session logs
before the fixture removed its temporary directory.

| Scenario | Result | Observation |
|---|---|---|
| `data_preview_and_dataset_report` | Passed | Preview and report assertions completed. |
| `forecast_train_predict` | Passed | Model training and prediction-table assertions completed. |
| `cap_artifact_generation` | Failed | `artifact_root` was None. CAP files, including package.json, existed; the LLM said “Output directory,” while the test parser only accepts “Root directory.” |
| `dataset_import_split_and_profile` | Failed | Import and split checks passed, but the final response omitted an HTML path; `html_path` was None. |
| `massive_additive_forecasting` | Failed | The agent trained under another name and requested confirmation to rename/retrain; loading the test's requested model raised ModelStorageError. |
| `massive_forecasting` | Failed | The LLM requested a default/custom AutoML configuration choice instead of training; the expected model name was absent. |
| `massive_outlier_detection` | Failed | Tool execution returned HANA error 260: `invalid column name: STORE_A`. This is a runtime problem, not merely response wording. |
| `outlier_detection` | Failed | The response listed three outlier dates but used singular “outlier”; the assertion requires “outliers” or `result_select_statement`. |
| `prediction_auto_repairs_labelled_holdout` | Failed | The preliminary prediction requested key/exog confirmation; no prediction-table name was returned. The auto-repair step was not reached. |
| `prediction_visualization_and_insight` | Failed | The response reported plot generation without an HTML path; the path assertion failed before the insight request. |
| `prediction_visualization_renders_in_notebook` | Failed | HTML path/existence checks passed, but the display hook received nothing. The fixture still expects automatic raw-HTML rendering despite V-173's default-off policy. |

### Security and compatibility interpretation

- Four failures concern response extraction/wording; three involve an LLM asking
  for confirmation or choosing a different model name. They do not establish a
  security regression, nor prove completion of actions the LLM did not perform.
  Follow-up tests should inspect structured tool results and actual artifacts,
  and make required user choices explicit, without weakening assertions about
  real operations or converting failures into skips.
- The notebook-rendering assertion conflicts with the intentional security
  default in `ContextAgent._display_html_artifact`. `HANA_ALLOW_RAW_HTML_RENDER`
  was not enabled for this run. Keep the default-denial test; any opt-in rendering
  test must be separate and limited to trusted synthetic HTML.
- Inspection of the published
  `hana_ml/algorithms/pal/tsa/outlier_detection.py:710` found a timestamp-result
  remapping path constructing `...={group_name}` without quoting string group
  names. This is consistent with the `STORE_A` SQL error. The hana.ai massive
  outlier tool has no diff against this PR's base; the PyPI package was not
  patched. A focused reproduction/fix is still needed, not a security-policy
  rollback or a claim that this is a necessary compatibility restriction.

The two ContextAgent MCP file-reload tests still skipped because
`nutest/testscripts/env.example` is absent. The HTTP test accepts an existing
`VCAP_SERVICES_FILE` override; the stdio test currently uses the fixed fixture
path. The user was asked for an existing fixture path, not credential contents.
No fixture or credentials were fabricated.

### Evidence, cleanup, and reproduction

- `.pytest_cache/assessment13-followup/xinkey2-context-e2e.xml`: first run with
  the key corrected, before the TLS fixture correction.
- `.pytest_cache/assessment13-followup/xinkey2-context-e2e-native-tls.xml`:
  final 13-case run; JUnit confirms 2 passes, 9 failures, 0 errors and 2 skips.
- Compileall of the three changed test files and `git diff --check` passed.
- After test teardown, an inventory identified one model and nine tables
  belonging to the unique run prefix `CTXE2E_D29D8991`. Only those verified
  objects were removed (the nine include the model's storage table). A second
  inventory found zero remaining models/tables with that prefix. These are
  synthetic test outputs, regenerable by rerunning; no unrelated business
  objects were removed. The fixture also removed its temporary artifact tree.

After the PyPI PowerShell environment setup in section 6:

```powershell
$env:HANA_USERKEY = 'XinKey2'
$env:RUN_CONTEXT_AGENT_E2E = '1'
$env:RUN_CONTEXT_AGENT_MCP_E2E = '1'
$env:RUN_CONTEXT_AGENT_MCP_STDIO_E2E = '1'
& $pypiTestPython -u -m pytest `
  nutest/testscripts/test_context_agent_e2e_scenarios.py `
  nutest/testscripts/test_context_agent_mcp_reload_connection_context_from_file_http.py `
  nutest/testscripts/test_context_agent_mcp_reload_connection_context_from_file_stdio.py `
  --import-mode=importlib -v --tb=short -ra `
  --junitxml=.pytest_cache/assessment13-followup/xinkey2-context-e2e-native-tls.xml
```

This is an affected-scenario rerun, not a full-repository run. The earlier 442-case
targeted pass and historical full-suite evidence remain unchanged. No commit or
push was made.

## 10. Pre-push verification (2026-10-28)

The user requested delivery to the existing PR. The intended code, tests, and
both Markdown reports are included; raw scans/reviews, one-off probes, untracked
notebooks and generated artifacts are excluded. Workstation-specific absolute
paths were removed from these reports. Dated local-only/uncommitted statements
above describe earlier states, not the scope of this delivery.

The four focused suites (`test_security_fixes.py`,
`test_assessment13_followup.py`, `test_pr_compatibility_regressions.py` and
`test_hana_mem0_adapter.py`) were rerun with the PyPI environment: **355 passed,
0 failed/errors/skipped, 36 warnings in 57.43 seconds; exit code 0**. Compilation
and staged whitespace checks passed. Evidence:
`.pytest_cache/assessment13-followup/pre-push-offline.xml`.

This pre-push run does not rerun or supersede the live result in section 9:
**2 passed, 9 failed, 2 skipped**. The PR must not be described as fully green.

## 11. Review10: statement-final WITH HINT boundary (2026-10-28)

This follow-up builds on `d8d4714`. It addresses the V-171 object-authorization
parser gap reported in
`review10.md`. No unrelated local reviews, email drafts, notebooks, or artifacts
were modified.

### Finding and final behavior

Before the fix, `WITH` ended FROM parsing. After parsing `WITH HINT (...)`, the
scanner treated subsequent comma/JOIN table names as ordinary expressions and did
not authorize them. Six synthetic bypass variants reached a mocked executor,
including nested-query and CTE variants. Read-only checks using `XinKey2` and
PyPI `hana-ml==2.30.26091800` confirmed that HANA rejected these six forms with
syntax error **257**; normal terminal hints and joins preceding hints executed.
This was a confirmed guard gap, not proof of an executable data-disclosure exploit
for those statements on the tested HANA system.

`_SQLObjectParser._parse_hints()` now requires both the outermost scan boundary
and a closing parenthesis at the end of the statement. It rejects hints inside
subqueries/CTE bodies/expressions, repeated hint clauses, and trailing joins,
commas, filters, ordering, limits or set operations **before execution**. A single
trailing statement terminator remains supported. Hint argument validation and all
existing table/function/sequence checks remain in place.

Legitimate final hints remain supported after ordinary SELECTs, joins, predicates,
ordering, the final SELECT of a CTE, and UNION queries. Multiple hints, scalar hint
arguments, mixed keyword case and strings containing hint-like text are covered.
This moves rejection of malformed SQL from HANA into the application guard; it
neither grants additional access nor changes valid query results in the tested
matrix. The existing live SQL fixture now explicitly enables encryption as well
as certificate verification.

Traceability: [parser](src/hana_ai/agents/hana_sql_agent.py),
[regression and live tests](nutest/testscripts/test_sql_agent_compatibility.py),
and [public grammar documentation](doc/agents/hana_ai.agents.hana_sql_agent.create_hana_sql_agent.rst).

### Verification

- Before the code fix: all **20** new offline rejection tests failed because no
  ValueError was raised (102 other cases deselected). The permissive mocked
  downstream executor ensures these tests cannot pass merely because a database
  rejects malformed SQL.
- After the fix: **487 passed, 0 failed/errors/skipped, 39 warnings in 103.18 seconds (exit 0)**. Breakdown: 138 security, 41 assessment follow-up,
  170 memory/stdio compatibility, 7 audit, 6 memory adapter, 3 Mem0 agent and
  **122 SQL cases** (72 offline + 50 live HANA).
- The SQL matrix adds 20 invalid placements, each checked offline and against the
  live executor boundary, and 10 valid queries, each checked offline and against
  direct HANA results: **60 new test instances**. All live valid-query results are
  compared row-for-row, and invalid placements must produce zero SQLAlchemy
  execution events.
- Changed production module Pylint: **10.00/10**; compilation and `git diff --check`
  passed. The fixture dropped its synthetic R9_COMPAT table; a follow-up
  read-only check found zero matching tables in the current schema. No unrelated
  business objects were changed.

JUnit evidence: `.pytest_cache/review10/hint-before.xml` and
`.pytest_cache/review10/hint-verified.xml` (local, git-ignored).

Reproduce after the PyPI PowerShell environment setup in section 6:

```powershell
$env:HANA_USERKEY = 'XinKey2'
& $pypiTestPython -u -m pytest --import-mode=importlib -q --tb=short -p no:cacheprovider `
  nutest/testscripts/test_security_fixes.py `
  nutest/testscripts/test_sql_agent_compatibility.py `
  nutest/testscripts/test_assessment13_followup.py `
  nutest/testscripts/test_pr_compatibility_regressions.py `
  nutest/test_hana_mcp_audit_view.py `
  nutest/testscripts/test_hana_mem0_adapter.py `
  nutest/testscripts/test_mem0_hana_agent.py `
  --junitxml=.pytest_cache/review10/hint-verified.xml
```

This is targeted security/compatibility validation, not a full-repository or live
LLM rerun. The previously recorded ContextAgent result of **2 passed / 9 failed /
2 skipped**, deployment requirements and release limitations remain unchanged.
