# Dependency upgrade acceptance

These opt-in tests complement `test_context_agent_e2e_scenarios.py`. They use
live HANA/PAL and, for memory, a deployed GenAI Hub embedding model. They do
not replace tools, embeddings, audit writers, or IPython with mocks.

| Test | Acceptance | Boundary |
| --- | --- | --- |
| Forecast migration | Same CSV and parameters in base and candidate environments; identical dates, columns and types; point forecasts and RMSE/MAD within `rtol=1e-7, atol=1e-6`; candidate also loads the base's saved model | One additive model configuration. PAL requires at least one interval sample; stochastic confidence bounds are not compared. This is not a forecast-quality benchmark or an LLM wording comparison |
| MCP audit (stateful/stateless) | Two real HTTP clients call `fetch_data`, then successful/failing HANA probes concurrently; disk records and live HANA invocation/session/client attribution agree; independent HANA plan-cache readback | Stateless mode deliberately has one process-level session ID, with request-specific client identity. Plan cache is not a permanent DBA audit trail |
| Vector memory | Real embedding, correct top hit, metadata retained, new HANA connection/manager reads persisted data, deletion preserves another tenant and another session | Public memory manager and vector adapter; no deprecated RAG agents or reranking |
| Notebook/browser | Real IPython kernel trains/predicts and calls `ForecastLinePlot`; emitted display HTML renders in a headless browser; legend interaction and report link work without JavaScript exceptions | A focused notebook for the demo's forecast flow, exported to HTML. Does not test all notebooks or the JupyterLab application UI |

## Prerequisites

- Candidate environment with this repository's `requirements.txt` and the
  GenAI Hub integration (`sap-ai-sdk-gen`), plus `pytest`, `nbclient`,
  `nbformat`, `nbconvert`, `ipykernel`, and `playwright`. Install test-only
  dependencies in an isolated environment; do not upgrade the application
  dependencies while preparing a comparison.
- A separate baseline Python environment installed from the chosen base
  commit's `requirements.txt`. The test extracts **that commit's source**
  using `git archive`, and checks which source each interpreter imports.
- A configured HANA userkey with permission to create/drop its own test
  tables, use PAL/model storage/vector search, and read `M_SQL_PLAN_CACHE`.
  TLS encryption and certificate validation stay enabled.
- GenAI Hub credentials and a deployed embedding model. Default:
  `text-embedding-ada-002`; override with `HANA_E2E_EMBEDDING_MODEL`.
- A Playwright Chromium installation (`python -m playwright install chromium`),
  or installed Edge/Chrome selected with `HANA_E2E_BROWSER_CHANNEL=msedge`
  or `chrome`. The browser is headless and serves only the test output
  directory over loopback.

## Run from the repository root (PowerShell)

```powershell
$env:RUN_DEPENDENCY_UPGRADE_E2E = '1'
$env:HANA_USERKEY = '<configured test userkey>'
$env:HANA_E2E_BASE_REF = '<PR base commit SHA>'
$env:HANA_E2E_BASELINE_PYTHON = '<absolute path to baseline python.exe>'
$env:HANA_E2E_BROWSER_CHANNEL = 'msedge'
$env:PYTHONPATH = Join-Path $PWD 'src'
$env:PYTHONIOENCODING = 'utf-8'
python -u -m pytest nutest/testscripts/e2e_dependency_upgrade -v --tb=short
```

Without `RUN_DEPENDENCY_UPGRADE_E2E=1`, tests skip before accessing live
services. Once opted in, missing credentials, packages, browser, or baseline
configuration are failures, not skips or passing checks. Run with a candidate
interpreter whose installed versions have been recorded.

To preserve artifacts, supply a **new** `--basetemp` directory and
`--junitxml` path. Create the parent directory first. Pytest deletes an
existing `--basetemp` directory: never point it at old evidence or user files.
Use a new directory for every attempt, including retries.

Each case owns a random `DEPE2E_` prefix. Cleanup removes only its model names
and tables and verifies no matching tables remain. The tests close HANA
connections, MCP servers, notebook kernels, and browsers in `finally` blocks.
Artifacts include environment/version records for the migration workers,
forecast comparisons, disk/HANA audit evidence, memory results, the executed
notebook, HTML reports, and browser screenshots. Treat audit evidence as
local diagnostic data; do not publish it without reviewing its contents.

E2E results establish compatibility of the tested flows. Dependabot alert
closure still requires the default branch's updated dependency analysis.

The companion fast regression tests need no live service:

```powershell
python -m pytest nutest/testscripts/test_mcp_connection_serialization.py -q
```

They check that cancellation, tool errors, another server's event loop,
initialization, and connection reload cannot release or replace a connection
in use. Stateless requests with missing identity headers must clear the
preceding caller's identity. `test_mcp_dependency_compatibility.py` also
checks that stateful audit IDs match the actual HTTP session header and
preserve initialization `clientInfo`.
