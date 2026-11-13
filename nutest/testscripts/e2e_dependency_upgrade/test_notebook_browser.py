"""Execute real IPython display hooks, then inspect their HTML in a browser."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[3]


def test_real_kernel_displays_interactive_forecast_and_report_link(hana_case, tmp_path):
    import nbformat
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from playwright.sync_api import sync_playwright
    _, prefix = hana_case
    kernel_dir = tmp_path / "kernels" / "dependency-e2e"
    kernel_dir.mkdir(parents=True)
    kernel_env = dict(PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(ROOT)]),
                      HANA_ALLOW_RAW_HTML_RENDER="false")
    (kernel_dir / "kernel.json").write_text(json.dumps(dict(
        argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        display_name="Dependency acceptance", language="python", env=kernel_env,
    )), encoding="utf-8")
    # A small notebook focused on the same forecast/report flow as the demo.
    # No LLM prose or mocked IPython; display data must come from a real kernel.
    source = '''
import json, os, sys
from pathlib import Path
from IPython import get_ipython
from IPython.display import FileLink, display
import plotly.io as pio
from hana_ml import dataframe
from hana_ai.tools.hana_ml_tools.ts_visualizer_tools import ForecastLinePlot
from nutest.testscripts.e2e_dependency_upgrade.forecast_worker import forecast
assert get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
pio.renderers.default = 'notebook'
cc = dataframe.ConnectionContext(userkey=os.environ['HANA_USERKEY'], encrypt=True, sslValidateCertificate=True)
try:
    result = forecast(cc, PREFIX, CSV)
    tool = ForecastLinePlot(cc)
    report = json.loads(tool.invoke(dict(predict_result=result['prediction_table'], actual_table=result['raw_table'], output_dir='plots')))
    assert 'error' not in report, report
    assert Path(report['html_file']).is_file()
    display(FileLink(report['html_file'], result_html_prefix='Open forecast report: '))
    Path('kernel-result.json').write_text(json.dumps(dict(python=sys.executable, kernel=type(get_ipython()).__name__, report=report, rows=len(result['yhat']))), encoding='utf-8')
finally:
    cc.close()
'''
    parameters = "PREFIX = " + repr(prefix + "_NOTEBOOK") + "\nCSV = " + repr(
        str(ROOT / "nutest/testscripts/demo/e2e_scenarios/SALES_REFUNDS.csv")) + "\n"
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(parameters + source)])
    manager = KernelManager(kernel_name="dependency-e2e", kernel_spec_manager=KernelSpecManager(
        kernel_dirs=[str(kernel_dir.parent)]))
    client = NotebookClient(notebook, km=manager, timeout=600, allow_errors=False,
                            resources={"metadata": {"path": str(tmp_path)}})
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        if client.kc is not None:
            client.kc.stop_channels()
        nbformat.write(notebook, tmp_path / "executed.ipynb")
    outputs = [output for cell in notebook.cells for output in cell.get("outputs", [])]
    assert not any(output.output_type == "error" for output in outputs)
    assert any("plotly-graph-div" in output.get("data", {}).get("text/html", "") for output in outputs)
    result = json.loads((tmp_path / "kernel-result.json").read_text(encoding="utf-8"))
    assert Path(result["python"]).resolve() == Path(sys.executable).resolve()
    assert result["kernel"] == "ZMQInteractiveShell" and result["rows"] == 30
    exported, _ = HTMLExporter(template_name="classic").from_notebook_node(notebook)
    (tmp_path / "notebook.html").write_text(exported, encoding="utf-8")

    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(tmp_path)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            options = {"headless": True}
            if os.environ.get("HANA_E2E_BROWSER_CHANNEL"):
                options["channel"] = os.environ["HANA_E2E_BROWSER_CHANNEL"]
            browser = playwright.chromium.launch(**options)
            try:
                page = browser.new_page(viewport={"width": 1400, "height": 1000})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.context.on("page", lambda opened: opened.on(
                    "pageerror", lambda error: errors.append(str(error))))
                response = page.goto(f"http://127.0.0.1:{server.server_port}/notebook.html")
                assert response.status == 200
                page.wait_for_function("document.querySelector('.plotly-graph-div')?.data?.length >= 2")
                graph = page.locator(".plotly-graph-div").first
                assert graph.is_visible()
                before = graph.evaluate("g => g.data.map(t => t.visible ?? true)")
                graph.locator(".legendtoggle").first.click()
                page.wait_for_function(
                    "before => JSON.stringify(document.querySelector('.plotly-graph-div').data.map(t => t.visible ?? true)) !== JSON.stringify(before)",
                    arg=before)
                page.screenshot(path=str(tmp_path / "notebook.png"), full_page=True)
                # IPython FileLink deliberately opens a new tab (target=_blank).
                with page.expect_popup() as opened:
                    page.get_by_role("link", name=Path(result["report"]["html_file"]).as_posix(), exact=True).click()
                report_page = opened.value
                report_page.wait_for_function("document.querySelector('.plotly-graph-div')?.data?.length >= 2")
                assert report_page.locator(".plotly-graph-div").is_visible()
                report_page.screenshot(path=str(tmp_path / "report.png"), full_page=True)
                assert errors == [], errors
                (tmp_path / "browser-result.json").write_text(json.dumps(dict(
                    browser_version=browser.version, legend_toggle_verified=True,
                    report_link_verified=True, javascript_errors=errors,
                ), indent=2), encoding="utf-8")
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
        assert not thread.is_alive()
