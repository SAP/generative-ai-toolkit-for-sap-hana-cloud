"""Same public-tool workload executed by two isolated Python interpreters."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def forecast(cc, prefix, csv_path, *, reuse_model=None):
    import numpy as np
    import pandas as pd
    from hana_ml import dataframe
    from hana_ml.model_storage import ModelStorage
    from hana_ai.tools.hana_ml_tools.additive_model_forecast_tools import (
        AdditiveModelForecastFitAndSave, AdditiveModelForecastLoadModelAndPredict,
    )
    # PAL's point forecast uses deterministic fitting here. PAL requires at
    # least one interval sample; do not compare stochastic confidence bounds.
    np.random.seed(42)
    pdf = pd.read_csv(csv_path).sort_values("BOOKING_DATE")
    assert len(pdf) == 180
    raw, train, future = (prefix + suffix for suffix in ("_RAW", "_TRAIN", "_FUTURE"))
    frame = dataframe.create_dataframe_from_pandas(
        cc, pdf, table_name=raw, force=False, disable_progressbar=True,
    ).to_datetime({"BOOKING_DATE": "YYYY-MM-DD"}).sort_values("BOOKING_DATE")
    frame.smart_save(raw, force=True)
    frame.head(150).save(train, force=False)
    frame.tail(30).select("BOOKING_DATE").save(future, force=False)
    name = reuse_model or prefix + "_MODEL"
    fit_parameters = dict(key="BOOKING_DATE", endog="REFUNDS", exog=None,
                          growth="linear", weekly_seasonality="true",
                          yearly_seasonality="false", daily_seasonality="false")
    if reuse_model is None:
        result = json.loads(AdditiveModelForecastFitAndSave(cc).invoke(
            dict(fit_table=train, name=name, version=1, **fit_parameters)))
        assert result["model_storage_name"] == name and result["model_storage_version"] == 1
    model = ModelStorage(cc).load_model(name=name, version=1)
    assert type(model).__name__ == "AdditiveModelForecast"
    result = json.loads(AdditiveModelForecastLoadModelAndPredict(cc).invoke(
        dict(predict_table=future, key="BOOKING_DATE", exog=None, name=name,
             version=1, uncertainty_samples=1)))
    assert "predicted_results_table" in result, result
    prediction = cc.table(result["predicted_results_table"]).collect().sort_values("BOOKING_DATE")
    actual = cc.table(raw).collect().sort_values("BOOKING_DATE").tail(30)
    joined = prediction.merge(actual, on="BOOKING_DATE", validate="one_to_one")
    assert len(joined) == len(prediction) == 30
    assert np.isfinite(joined["YHAT"]).all()
    errors = joined["YHAT"] - joined["REFUNDS"]
    return {
        "columns": list(prediction.columns),
        "dtypes": {col: str(dtype) for col, dtype in prediction.dtypes.items()},
        "dates": prediction["BOOKING_DATE"].astype(str).tolist(),
        "yhat": prediction["YHAT"].tolist(),
        "rmse": float(np.sqrt((errors ** 2).mean())), "mad": float(errors.abs().mean()),
        "csv_sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
        "fit_parameters": fit_parameters, "model_name": name,
        "raw_table": raw, "prediction_table": result["predicted_results_table"],
    }


def main():
    import hana_ai
    from hana_ml import dataframe
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    imported = Path(hana_ai.__file__).resolve()
    assert imported.is_relative_to(Path(request["source_root"]).resolve()), str(imported)
    cc = dataframe.ConnectionContext(userkey=os.environ["HANA_USERKEY"],
                                     encrypt=True, sslValidateCertificate=True)
    try:
        result = forecast(cc, request["prefix"], request["csv_path"],
                          reuse_model=request.get("reuse_model"))
        result["python"] = sys.version
        result["hana_ai_file"] = str(imported)
        result["packages"] = {d.metadata["Name"]: d.version
                              for d in importlib.metadata.distributions()}
        Path(request["output"]).write_text(json.dumps(result, indent=2), encoding="utf-8")
    finally:
        cc.close()


if __name__ == "__main__":
    main()
