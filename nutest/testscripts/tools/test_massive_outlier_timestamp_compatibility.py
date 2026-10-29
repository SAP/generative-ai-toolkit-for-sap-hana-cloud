"""Live regression for PyPI hana-ml grouped timestamp result remapping."""

from datetime import datetime, timedelta
import json
import os
import uuid

import pytest


@pytest.fixture(params=[("STORE_A", "STORE_B"), ("O'Reilly", "x'; SELECT 1 FROM DUMMY; --")])
def timestamp_groups(request):
    if not os.environ.get("HANA_USERKEY"):
        pytest.skip("HANA_USERKEY is required for live HANA regression tests")
    from hana_ml import ConnectionContext

    connection = ConnectionContext(
        userkey=os.environ["HANA_USERKEY"], encrypt=True, sslValidateCertificate=True,
    )
    table = "#CTX_OUTLIER_" + uuid.uuid4().hex.upper()
    cursor = connection.connection.cursor()
    try:
        cursor.execute(f'CREATE LOCAL TEMPORARY TABLE {table} ("STORE" NVARCHAR(100), "DAY" TIMESTAMP, "VALUE" DOUBLE)')
        groups = request.param
        start = datetime(2024, 1, 1)
        rows = [(group, start + timedelta(days=i), 1000000.0 if i == 12 else 10.0)
                for group in groups for i in range(30)]
        cursor.executemany(f'INSERT INTO {table} VALUES (?, ?, ?)', rows)
        yield connection, table, groups, start + timedelta(days=12)
    finally:
        cursor.close()
        connection.close()  # Local temporary tables belong only to this connection.

def test_timestamp_groups_preserve_names_dates_and_outliers(timestamp_groups):
    from hana_ai.tools.hana_ml_tools.massive_ts_outlier_detection_tools import MassiveTSOutlierDetection

    connection, table, groups, outlier_day = timestamp_groups
    result = json.loads(MassiveTSOutlierDetection(connection_context=connection).invoke({
        "table_name": table, "group_key": "STORE", "key": "DAY", "endog": "VALUE",
        "auto": False, "smooth_method": "no", "outlier_method": "z1",
        "detect_seasonality": False, "threshold": 3.0,
    }))
    assert "error" not in result
    assert {item["group_id"] for item in result["outliers"]} == set(groups)
    for item in result["outliers"]:
        assert len(item["outlier_ids"]) == 1
        assert str(item["outlier_ids"][0]).startswith(outlier_day.date().isoformat())
    frame = connection.sql(result["result_select_statement"]).collect()
    assert len(frame) == 30 * len(groups)
    assert set(frame["STORE"]) == set(groups)
    assert set(frame.groupby("STORE").size()) == {30}
    assert connection.table(table).count() == 30 * len(groups)
