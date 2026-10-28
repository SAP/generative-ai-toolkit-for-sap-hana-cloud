"""Assessment-13 follow-up checks: synthetic credentials, no external processes/DB."""

import asyncio
import importlib
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def admin_server(monkeypatch):
    """Capture the real launcher's FastMCP instance without starting a listener."""
    mod = importlib.import_module("hana_ai.tools.toolkit")
    real_mcp = mod.FastMCPHTTP
    captured = []

    def make_mcp(*args, **kwargs):
        instance = real_mcp(*args, **kwargs)
        captured.append(instance)
        return instance

    monkeypatch.setenv("HANA_ALLOW_RUNTIME_CONN_UPDATE", "true")
    monkeypatch.setattr(mod, "FastMCPHTTP", make_mcp)
    monkeypatch.setattr(mod, "Middleware", None)
    monkeypatch.setattr(mod.HANAMLToolkit, "_global_mcp_servers", {})
    monkeypatch.setattr(mod.HANAMLToolkit, "is_port_available", lambda *_: True)
    monkeypatch.setattr(mod.HANAMLToolkit, "get_tools", lambda _: [])
    monkeypatch.setattr(mod, "_refresh_tools_for_new_context", lambda _: {})
    connection_factory = Mock()
    monkeypatch.setattr(mod, "ConnectionContext", connection_factory)
    toolkit = mod.HANAMLToolkit.model_construct(connection_context=None)
    with monkeypatch.context() as launch_patch:
        launch_patch.setattr(mod.threading.Thread, "start", lambda _: None)
        toolkit.launch_mcp_server(
            transport="http", auth_token="synthetic-assessment-token", max_retries=1,
        )
    return captured[0], connection_factory


@pytest.mark.parametrize("options", [{}, {"encrypt": None, "ssl_validate_certificate": None}])
def test_admin_update_omitted_tls_options_are_secure(admin_server, options):
    mcp, connection_factory = admin_server
    tool = asyncio.run(mcp.get_tool("admin_update_connection_context"))
    result = tool.fn(address="hana.example.test", test_connection=True, **options)
    assert result["ok"] is True
    assert connection_factory.call_count == 2
    for call in connection_factory.call_args_list:
        assert call.kwargs["encrypt"] is True
        assert call.kwargs["sslValidateCertificate"] is True


@pytest.mark.parametrize("option", ["encrypt", "ssl_validate_certificate"])
def test_admin_update_tls_downgrade_needs_acknowledgement(admin_server, option):
    mcp, connection_factory = admin_server
    tool = asyncio.run(mcp.get_tool("admin_update_connection_context"))
    result = tool.fn(address="hana.example.test", **{option: False})
    assert result["ok"] is False
    connection_factory.assert_not_called()


@pytest.mark.parametrize("userkey", [None, "synthetic-key"])
@pytest.mark.parametrize("setting", [None, "", "   "])
def test_env_connection_requires_encryption_by_default(monkeypatch, userkey, setting):
    from hana_ai.tools.toolkit import _build_cc_params_from_env

    monkeypatch.delenv("HANA_USERKEY", raising=False)
    if userkey:
        monkeypatch.setenv("HANA_USERKEY", userkey)
    monkeypatch.setenv("HANA_ADDRESS", "hana.example.test")
    monkeypatch.setenv("HANA_USER", "synthetic-user")
    monkeypatch.setenv("HANA_PASSWORD", "synthetic-password")
    monkeypatch.delenv("HANA_ENCRYPT", raising=False)
    monkeypatch.delenv("HANA_SSL_VALIDATE", raising=False)
    if setting is not None:
        monkeypatch.setenv("HANA_ENCRYPT", setting)
    params = _build_cc_params_from_env()
    assert params["encrypt"] is True
    assert params["sslValidateCertificate"] is True


def test_launcher_auth_rejects_unauthenticated_admin_calls(admin_server):
    """Use auth configured by HANAMLToolkit, not a separately built verifier."""
    from starlette.testclient import TestClient

    mcp, connection_factory = admin_server
    app = mcp.http_app(path="/mcp", stateless_http=True)
    payload = {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "admin_update_connection_context",
                   "arguments": {"address": "hana.example.test"}},
    }
    with TestClient(app) as client:
        for token in (None, "incorrect-test-token"):
            headers = {} if token is None else {"Authorization": f"Bearer {token}"}
            assert client.post("/mcp", json=payload, headers=headers).status_code == 401
        response = client.post(
            "/mcp", json=payload,
            headers={"Authorization": "Bearer synthetic-assessment-token",
                     "Accept": "application/json, text/event-stream"},
        )
        assert response.status_code == 200
        assert not response.json()["result"].get("isError", False)
    connection_factory.assert_called_once()


@pytest.mark.parametrize("value", ["false", "true", 1, [], {"enabled": True}])
def test_remote_plaintext_requires_boolean_true(value):
    from hana_ai.client.mcp_client import HTTPMCPClient

    with pytest.raises(ValueError, match="plaintext MCP HTTP"):
        HTTPMCPClient("http://remote.example.test/mcp", allow_insecure_http=value)


@pytest.mark.parametrize("field", ["table_name", "schema", "model_version"])
def test_vector_constructor_validates_before_metadata_or_ddl(field):
    from hana_ai.vectorstore.hana_vector_engine import HANAMLinVectorEngine

    connection = Mock()
    options = {"table_name": "Vectors", "schema": "APP", "model_version": "MODEL"}
    options[field] = "x'; SELECT * FROM SENSITIVE; --"
    with pytest.raises(ValueError):
        HANAMLinVectorEngine(connection, **options)
    assert connection.mock_calls == []


@pytest.mark.parametrize("field", ["table_name", "schema"])
def test_vector_metadata_rejects_single_quote_in_names(field):
    from hana_ai.vectorstore.hana_vector_engine import HANAMLinVectorEngine

    connection = Mock()
    options = {"table_name": "Vectors", "schema": "APP"}
    options[field] = "x' OR '1'='1"
    with pytest.raises(ValueError):
        HANAMLinVectorEngine(connection, **options)
    assert connection.mock_calls == []


@pytest.mark.parametrize("field", ["table_name", "schema", "model_version", "input"])
def test_vector_query_validates_before_lazy_columns_lookup(field):
    from hana_ai.vectorstore.hana_vector_engine import HANAMLinVectorEngine

    engine = object.__new__(HANAMLinVectorEngine)
    engine.connection_context = Mock()
    engine.connection_context.table.return_value.columns = ["MODEL", "DESCRIPTION", "EXAMPLE", "EMBEDDINGS"]
    engine.table_name, engine.schema, engine.model_version = "Vectors", "APP", "MODEL"
    engine.columns = None
    query = "normal query"
    if field == "input":
        query = "x; SELECT * FROM SENSITIVE"
    else:
        setattr(engine, field, "x; SELECT * FROM SENSITIVE")
    with pytest.raises(ValueError):
        engine.query(query)
    assert engine.connection_context.mock_calls == []


@pytest.mark.parametrize("filename", ["sh", "bash", "cmd.exe", "powershell.exe", "pwsh.exe",
                                     "server.cmd", "server.bat", "server.ps1"])
def test_stdio_shell_or_batch_rejected_even_with_matching_profile(monkeypatch, tmp_path, filename):
    from hana_ai.client.mcp_client import _validate_stdio_command

    command = tmp_path / filename
    command.touch()  # Inert placeholder: the test never starts a subprocess.
    monkeypatch.setenv("HANA_MCP_STDIO_ALLOWED_PROFILES", json.dumps([
        {"command": str(command), "args_prefix": ["trusted-script"]},
    ]))
    with pytest.raises(ValueError, match="shell|batch"):
        _validate_stdio_command(str(command), ["trusted-script"])


def test_stdio_checks_resolved_executable_as_well_as_alias(monkeypatch, tmp_path):
    mod = importlib.import_module("hana_ai.client.mcp_client")
    monkeypatch.setattr(mod, "_resolve_stdio_executable",
                        lambda _: str(tmp_path / "powershell.exe"))
    with pytest.raises(ValueError, match="shell"):
        mod._validate_stdio_command("apparently-safe-alias", ["trusted-script"])


@pytest.mark.parametrize("failed_file", [1, 2])
@pytest.mark.parametrize("failure", ["encode", "write"])
def test_certificate_cleanup_covers_creation_write_failures(monkeypatch, tmp_path, failed_file, failure):
    from hana_ai.retrieval.utility import _get_access_token
    import requests

    created = []
    real_temporary_file = tempfile.NamedTemporaryFile
    network = Mock(side_effect=AssertionError("must fail before any network request"))
    monkeypatch.setattr(requests, "post", network)

    def temporary_file(**kwargs):
        handle = real_temporary_file(dir=tmp_path, **kwargs)
        created.append(Path(handle.name))
        if len(created) == failed_file and failure == "write":
            real_write = handle.write

            def partial_write(data):
                real_write(data[:4])
                raise OSError("synthetic disk write failure")

            handle.write = partial_write
        return handle

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", temporary_file)
    credentials = {"certurl": "https://auth.example.test", "clientid": "test-client",
                   "certificate": "synthetic-certificate", "key": "synthetic-private-key"}
    if failure == "encode":
        credentials["certificate" if failed_file == 1 else "key"] = "\ud800"
    with pytest.raises((UnicodeEncodeError, OSError)):
        _get_access_token(credentials)
    assert len(created) == failed_file
    assert all(not path.exists() for path in created)
    assert not list(tmp_path.iterdir())
    network.assert_not_called()


@pytest.mark.parametrize("stage", ["success", "status", "json"])
def test_certificate_cleanup_covers_response_paths(monkeypatch, tmp_path, stage):
    from hana_ai.retrieval.utility import _get_access_token
    import requests

    real_temporary_file = tempfile.NamedTemporaryFile
    monkeypatch.setattr(tempfile, "NamedTemporaryFile",
                        lambda **kwargs: real_temporary_file(dir=tmp_path, **kwargs))

    def post(*_args, **kwargs):
        assert all(os.path.isfile(path) for path in kwargs["cert"])
        response = Mock(status_code=200 if stage != "status" else 503, text="synthetic error")
        response.json.return_value = {"access_token": "synthetic-token"}
        if stage == "json":
            response.json.side_effect = ValueError("synthetic invalid JSON")
        return response

    monkeypatch.setattr(requests, "post", post)
    credentials = {"certurl": "https://auth.example.test", "clientid": "test-client",
                   "certificate": "synthetic-certificate", "key": "synthetic-private-key"}
    if stage == "success":
        assert _get_access_token(credentials) == "synthetic-token"
    else:
        with pytest.raises(Exception, match="503|synthetic invalid JSON"):
            _get_access_token(credentials)
    assert not list(tmp_path.iterdir())
