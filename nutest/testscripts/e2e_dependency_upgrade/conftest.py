"""Live services are opt-in; failed prerequisites never count as passes."""
import os
import re
import uuid

import pytest


@pytest.fixture(autouse=True)
def live_opt_in(monkeypatch):
    if os.environ.get("RUN_DEPENDENCY_UPGRADE_E2E") != "1":
        pytest.skip("Set RUN_DEPENDENCY_UPGRADE_E2E=1; see this directory's README.md")
    if not os.environ.get("HANA_USERKEY"):
        pytest.fail("HANA_USERKEY is required for live dependency acceptance")
    import certifi
    monkeypatch.setenv("SSL_CERT_FILE", certifi.where())
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", certifi.where())
    monkeypatch.setenv("FASTMCP_CHECK_FOR_UPDATES", "off")
    monkeypatch.setenv("HANA_ALLOW_RAW_HTML_RENDER", "false")


@pytest.fixture
def hana_case(live_opt_in):
    from hana_ml import dataframe
    from hana_ml.model_storage import ModelStorage
    cc = dataframe.ConnectionContext(
        userkey=os.environ["HANA_USERKEY"], encrypt=True, sslValidateCertificate=True,
    )
    prefix = "DEPE2E_" + uuid.uuid4().hex[:12].upper()
    try:
        yield cc, prefix
    finally:
        # All persistent objects belong to this random prefix. Never clean a
        # shared demo table, an existing user model, or an entire schema.
        try:
            assert re.fullmatch(r"DEPE2E_[0-9A-F]{12}", prefix)
            storage = ModelStorage(cc)
            for suffix in ("BASE_MODEL", "NEW_MODEL", "NOTEBOOK_MODEL"):
                name = prefix + "_" + suffix
                if not storage.list_models(name=name).empty:
                    storage.delete_models(name=name)
            cursor = cc.connection.cursor()
            try:
                pattern = "%" + prefix.replace("_", "!_") + "!_%"
                query = ("SELECT TABLE_NAME FROM SYS.TABLES WHERE SCHEMA_NAME = CURRENT_SCHEMA "
                         "AND TABLE_NAME LIKE ? ESCAPE '!'")
                cursor.execute(query, (pattern,))
                names = [row[0] for row in cursor.fetchall()]
                for name in names:
                    assert prefix + "_" in name
                    cursor.execute("DROP TABLE " + dataframe.quotename(name))
                cursor.execute(query, (pattern,))
                assert cursor.fetchall() == [], "Synthetic tables remain after cleanup"
                print(f"Cleaned {prefix}: {len(names)} tables; remaining=0")
            finally:
                cursor.close()
        finally:
            cc.close()
