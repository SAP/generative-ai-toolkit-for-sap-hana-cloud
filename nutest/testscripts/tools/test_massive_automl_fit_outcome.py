"""PAL can return group errors without raising: do not save empty models."""

import json
import os
from pathlib import Path
import uuid
from unittest.mock import Mock

import pandas as pd
import pytest

from hana_ai.tools.hana_ml_tools import massive_automatic_timeseries_tools as module


@pytest.mark.parametrize("model_rows,has_errors", [(0, True), (None, True), (0, False), (10, False), (10, True)])
def test_fit_outcome_preserves_success_and_never_saves_an_empty_model(monkeypatch, model_rows, has_errors):
    monkeypatch.delenv("HANA_AI_AUTOML_REQUIRE_CONFIG_DICT", raising=False)
    connection = Mock()
    data = connection.table.return_value
    data.columns = ["G", "DAY", "Y"]
    data.get_table_structure.return_value = {"DAY": "INTEGER"}
    data.select.return_value = data
    model = Mock()
    model.model_ = [None if model_rows is None else Mock(), None]
    if model_rows is not None:
        model.model_[0].count.return_value = model_rows
    errors = [{"G": "STORE_A", "MESSAGE": "timestamp intervals are not equal"}] if has_errors else []
    model.error_.collect.return_value = pd.DataFrame(errors)
    monkeypatch.setattr(module, "MassiveAutomaticTimeSeries", Mock(return_value=model))
    storage_factory = Mock()
    monkeypatch.setattr(module, "ModelStorage", storage_factory)
    tool = object.__new__(module.MassiveAutomaticTimeSeriesFitAndSave)
    object.__setattr__(tool, "connection_context", connection)
    result = json.loads(tool._run(fit_table="T", key="DAY", group_key="G", endog="Y", name="MODEL", version=1))
    model.fit.assert_called_once()
    if not model_rows:
        assert result["error"] == "massive_automl_fit_failed"
        assert result["group_errors"] == errors
        storage_factory.assert_not_called()
    else:
        assert "error" not in result
        assert result["model_storage_name"] == "MODEL"
        assert result["model_storage_version"] == 1
        assert result.get("group_errors", []) == errors
        storage_factory.return_value.save_model.assert_called_once_with(model=model, if_exists="replace")


def test_live_irregular_timestamps_report_pal_error_without_saving_model():
    if not os.environ.get("HANA_USERKEY"):
        pytest.skip("HANA_USERKEY is required for live HANA regression tests")
    from hana_ml import ConnectionContext
    from hana_ml.model_storage import ModelStorage
    from hana_ai.tools.hana_ml_tools.config_dict_validator_tools import GetAutoMLConfigDict

    connection = ConnectionContext(userkey=os.environ["HANA_USERKEY"], encrypt=True, sslValidateCertificate=True)
    cursor = connection.connection.cursor()
    suffix = uuid.uuid4().hex.upper()
    table = "#CTX_AUTOML_GAP_" + suffix
    model_name = "CTX_AUTOML_GAP_" + suffix
    storage = ModelStorage(connection)
    owns_model_name = False
    try:
        assert storage.list_models(name=model_name).empty
        owns_model_name = True
        cursor.execute(f'CREATE LOCAL TEMPORARY TABLE {table} ("STORE_ID" NVARCHAR(100), "BOOKING_DATE" TIMESTAMP, "REFUNDS" DOUBLE)')
        csv_path = Path(__file__).resolve().parents[1] / "demo" / "e2e_scenarios" / "SALES_REFUNDS.csv"
        data = pd.read_csv(csv_path, parse_dates=["BOOKING_DATE"]).sort_values("BOOKING_DATE").head(150)
        assert data["BOOKING_DATE"].diff().dropna().nunique() > 1
        rows = [(group, day.to_pydatetime(), float(value)) for group in ("STORE_A", "STORE_B")
                for day, value in data.itertuples(index=False, name=None)]
        cursor.executemany(f'INSERT INTO {table} VALUES (?, ?, ?)', rows)
        config = json.loads(GetAutoMLConfigDict(connection_context=connection).invoke(
            {"pipeline_type": "timeseries", "config_dict": "default"},
        ))["config_dict"]
        result = json.loads(module.MassiveAutomaticTimeSeriesFitAndSave(connection_context=connection).invoke({
            "fit_table": table, "key": "BOOKING_DATE", "group_key": "STORE_ID", "endog": "REFUNDS",
            "name": model_name, "version": 1, "config_dict": config, "generations": 1,
            "population_size": 4, "offspring_size": 4, "elite_number": 1, "fold_num": 2,
            "random_seed": 42, "max_eval_time_mins": 1.0,
        }))
        assert result["error"] == "massive_automl_fit_failed"
        assert {item["STORE_ID"] for item in result["group_errors"]} == {"STORE_A", "STORE_B"}
        assert all("timestamp intervals are not equal" in item["MESSAGE"] for item in result["group_errors"])
        assert storage.list_models(name=model_name).empty
        assert connection.table(table).count() == len(rows)
    finally:
        # Remove only a model leaked by a failing regression, never a preexisting model.
        try:
            if owns_model_name and not storage.list_models(name=model_name).empty:
                storage.delete_models(name=model_name)
        finally:
            cursor.close()
            connection.close()
