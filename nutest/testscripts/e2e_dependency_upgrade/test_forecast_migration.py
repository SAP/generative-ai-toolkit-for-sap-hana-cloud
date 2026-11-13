"""Compare the PR with its base, including loading a model saved by the base."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "nutest/testscripts/demo/e2e_scenarios/SALES_REFUNDS.csv"
WORKER = Path(__file__).with_name("forecast_worker.py")


def test_forecast_matches_base_and_loads_old_model(hana_case, tmp_path):
    import numpy as np
    _, prefix = hana_case
    baseline = os.environ.get("HANA_E2E_BASELINE_PYTHON")
    base_ref = os.environ.get("HANA_E2E_BASE_REF")
    if not baseline or not base_ref:
        pytest.fail("Set HANA_E2E_BASELINE_PYTHON and HANA_E2E_BASE_REF; see README.md")
    baseline = str(Path(baseline).resolve(strict=True))
    base_sha = subprocess.check_output(["git", "rev-parse", "--verify", base_ref + "^{commit}"],
                                       cwd=ROOT, text=True).strip()
    archive = subprocess.check_output(["git", "archive", "--format=zip", base_sha, "src"], cwd=ROOT)
    base_dir = tmp_path / "base"
    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        for item in source.infolist():
            assert (base_dir / item.filename).resolve().is_relative_to(base_dir.resolve())
        source.extractall(base_dir)
    results = {}
    for role, python, source, reuse in (
        ("BASE", baseline, base_dir / "src", None),
        ("NEW", sys.executable, ROOT / "src", None),
        ("REUSED", sys.executable, ROOT / "src", prefix + "_BASE_MODEL"),
    ):
        output = tmp_path / (role + ".json")
        request = tmp_path / (role + "-request.json")
        request.write_text(json.dumps(dict(prefix=prefix + "_" + role, source_root=str(source),
                                          csv_path=str(CSV), reuse_model=reuse, output=str(output))),
                           encoding="utf-8")
        env = dict(os.environ, PYTHONPATH=str(source), PYTHONIOENCODING="utf-8")
        log_path = tmp_path / (role + ".log")
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run([python, str(WORKER), str(request)], cwd=tmp_path,
                                       env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600, check=False)
        assert completed.returncode == 0, f"{role} failed; see {log_path}"
        results[role] = json.loads(output.read_text(encoding="utf-8"))
    old = results["BASE"]
    assert old["packages"]["langchain-community"] != results["NEW"]["packages"]["langchain-community"]
    comparisons = {}
    for role in ("NEW", "REUSED"):
        new = results[role]
        for field in ("columns", "dtypes", "dates", "csv_sha256", "fit_parameters"):
            assert new[field] == old[field], (role, field)
        # Acceptance tolerance fixed before the run, never relaxed to make it pass.
        np.testing.assert_allclose(new["yhat"], old["yhat"], rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose([new["rmse"], new["mad"]], [old["rmse"], old["mad"]],
                                   rtol=1e-7, atol=1e-6)
        comparisons[role] = float(np.max(np.abs(np.array(new["yhat"]) - old["yhat"])))
    (tmp_path / "comparison.json").write_text(json.dumps(
        dict(base_sha=base_sha, max_absolute_prediction_difference=comparisons,
             rtol=1e-7, atol=1e-6), indent=2), encoding="utf-8")
