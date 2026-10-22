"""
Offline regression tests for the Assessment-13 (13-hana-ml) security fixes.

Run:
    pytest nutest/testscripts/test_security_fixes.py -v

No live HANA, credentials, or network are required. These tests exercise the pure,
code-level guards introduced by the fixes; they do NOT cover criteria that need a live
MCP server / HANA backend / notebook frontend (noted inline).
"""
import importlib
import asyncio
import json
import os
import types

import pytest


def _import_hana_ai_utility():
    """Import the real module; V-160 moved nltk to count_tokens() only."""
    return importlib.import_module("hana_ai.utility")


# ============================ V-162: secure temp dir + name sanitize ============================
class TestV162SecureArtifacts:
    def test_secure_report_dir_is_unique_private_and_exists(self):
        util = _import_hana_ai_utility()
        d1 = util.secure_report_dir("hanaml_report")
        d2 = util.secure_report_dir("hanaml_report")
        try:
            assert os.path.isdir(d1) and os.path.isdir(d2)
            # Not a predictable, shared path -> two calls never collide.
            assert d1 != d2
            assert "hanaml_report" in os.path.basename(d1)
        finally:
            os.rmdir(d1)
            os.rmdir(d2)

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("../../etc/passwd", "passwd"),
            ("good_name", "good_name"),
            ("a/b/c", "c"),
            ("x; rm -rf /", "artifact"),
            ("", "artifact"),
            (None, "artifact"),
            ("..", "artifact"),
        ],
    )
    def test_sanitize_artifact_name_strips_traversal(self, raw, expected):
        util = _import_hana_ai_utility()
        out = util.sanitize_artifact_name(raw)
        assert "/" not in out and "\\" not in out
        assert ".." not in out
        assert out == expected


# ============================ V-165: stdio command validation ============================
class TestV165StdioCommandValidation:
    def _validator(self):
        mod = importlib.import_module("hana_ai.client.mcp_client")
        return mod._validate_stdio_command

    @pytest.mark.parametrize("cmd", ["bash", "sh", "zsh", "/bin/sh", "powershell", "cmd.exe"])
    def test_shell_interpreters_rejected(self, cmd):
        with pytest.raises(ValueError):
            self._validator()(cmd, [])

    def test_inline_code_rejected(self):
        with pytest.raises(ValueError):
            self._validator()("python", ["-c", "import os; os.system('x')"])

    def test_non_allowlisted_command_rejected(self, monkeypatch):
        monkeypatch.delenv("HANA_MCP_STDIO_ALLOWED_PROFILES", raising=False)
        with pytest.raises(ValueError):
            self._validator()("python", ["trusted_server.py"])

    def test_matching_profile_allowed(self, monkeypatch):
        monkeypatch.setenv(
            "HANA_MCP_STDIO_ALLOWED_PROFILES",
            json.dumps([{"command": "python", "args_prefix": ["trusted_server.py"]}]),
        )
        resolved = self._validator()("python", ["trusted_server.py"])
        assert os.path.isabs(resolved)

    def test_additional_args_require_host_opt_in(self, monkeypatch):
        monkeypatch.setenv(
            "HANA_MCP_STDIO_ALLOWED_PROFILES",
            json.dumps([{"command": "python", "args_prefix": ["trusted_server.py"]}]),
        )
        with pytest.raises(ValueError, match="do not match"):
            self._validator()("python", ["trusted_server.py", "--plugin", "attacker.py"])

        monkeypatch.setenv(
            "HANA_MCP_STDIO_ALLOWED_PROFILES",
            json.dumps([{
                "command": "python",
                "args_prefix": ["trusted_server.py"],
                "allow_additional_args": True,
            }]),
        )
        assert os.path.isabs(
            self._validator()("python", ["trusted_server.py", "--port", "9000"])
        )

    def test_profile_does_not_allow_arbitrary_python_script(self, monkeypatch):
        monkeypatch.setenv(
            "HANA_MCP_STDIO_ALLOWED_PROFILES",
            json.dumps([{"command": "python", "args_prefix": ["trusted_server.py"]}]),
        )
        with pytest.raises(ValueError, match="do not match"):
            self._validator()("python", ["attacker.py"])

    def test_invalid_profile_json_rejected(self, monkeypatch):
        monkeypatch.setenv("HANA_MCP_STDIO_ALLOWED_PROFILES", "not-json")
        with pytest.raises(ValueError, match="valid JSON"):
            self._validator()("python", ["trusted_server.py"])


# ============================ V-164: HTTP transport boundary ============================
class TestV164HTTPTransport:
    def _validator(self):
        return importlib.import_module("hana_ai.client.mcp_client")._validate_http_base_url

    @pytest.mark.parametrize(
        "url",
        ["http://localhost:8000/mcp", "http://127.0.0.1:8000", "http://[::1]:8000"],
    )
    def test_loopback_http_allowed(self, url):
        assert self._validator()(url) == url

    def test_remote_http_rejected_by_default(self):
        with pytest.raises(ValueError, match="plaintext MCP HTTP"):
            self._validator()("http://mcp.example.test:8000/mcp")

    def test_remote_https_allowed(self):
        url = "https://mcp.example.test/mcp"
        assert self._validator()(url) == url

    def test_remote_http_requires_explicit_opt_in(self):
        url = "http://mcp.internal.test/mcp"
        assert self._validator()(url, allow_insecure_http=True) == url

    def test_http_client_disables_redirects(self, monkeypatch):
        mod = importlib.import_module("hana_ai.client.mcp_client")
        observed = {}

        class _Response:
            headers = {}

        class _Client:
            headers = {}

            async def post(self, *_args, **_kwargs):
                return _Response()

            async def aclose(self):
                pass

        def make_client(**kwargs):
            observed.update(kwargs)
            return _Client()

        monkeypatch.setattr(mod.httpx, "AsyncClient", make_client)

        async def exercise():
            client = mod.HTTPMCPClient()
            await client.initialize()
            await client.close()

        asyncio.run(exercise())
        assert observed["follow_redirects"] is False

    def test_sync_http_client_disables_redirects(self):
        mod = importlib.import_module("hana_ai.tools.hana_ml_tools.utility")
        client = mod.SyncHTTPMCPClient("https://mcp.example.test/mcp")
        try:
            assert client.client.follow_redirects is False
        finally:
            client.client.close()
        with pytest.raises(ValueError, match="plaintext MCP HTTP"):
            mod.SyncHTTPMCPClient("http://mcp.example.test/mcp")


# ============================ V-159: LLM SQL output is read-only ============================
class TestV159ReadOnlySQLExtraction:
    def _extract(self, text):
        mod = importlib.import_module("hana_ai.smart_dataframe")
        return mod._extract_readonly_select(text)

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM T",
            "SELECT ';' AS VALUE FROM DUMMY;",
            "WITH X AS (SELECT 1 AS A FROM DUMMY) SELECT * FROM X",
        ],
    )
    def test_single_readonly_statement_allowed(self, sql):
        assert self._extract(sql).upper().startswith(("SELECT", "WITH"))

    @pytest.mark.parametrize(
        "sql",
        [
            "Here is the SQL: SELECT * FROM T",
            "SELECT * FROM T; DROP TABLE T",
            "SELECT * FROM T -- ignore guard",
            "SELECT * FROM T /* comment */",
            "WITH X AS (DELETE FROM T) SELECT * FROM X",
            "SELECT * FROM T FOR UPDATE",
            "SELECT * INTO X FROM T",
        ],
    )
    def test_explanations_comments_stacking_and_writes_rejected(self, sql):
        with pytest.raises(ValueError):
            self._extract(sql)


# ============================ V-157/V-158: SQL value + identifier encoding ============================
class TestV157V158SQLGuards:
    def test_vector_string_literal_doubles_quotes(self):
        mod = importlib.import_module("hana_ai.vectorstore.hana_vector_engine")
        assert mod._sql_string_literal("x'; DROP TABLE T;--") == "'x''; DROP TABLE T;--'"

    def test_vector_identifier_quotes_one_component(self):
        mod = importlib.import_module("hana_ai.vectorstore.hana_vector_engine")
        assert mod._quote_identifier("MixedCase") == '"MixedCase"'
        with pytest.raises(ValueError):
            mod._quote_identifier('x"."y')

    def test_vector_query_rejects_unbounded_or_unknown_structure(self):
        mod = importlib.import_module("hana_ai.vectorstore.hana_vector_engine")
        engine = object.__new__(mod.HANAMLinVectorEngine)
        engine.schema = "SCHEMA"
        engine.table_name = "VECTORS"
        engine.columns = ["MODEL", "DESCRIPTION", "EXAMPLE", "EMBEDDINGS"]
        engine.model_version = "SAP_NEB.20240715"
        engine.connection_context = types.SimpleNamespace()
        with pytest.raises(ValueError, match="top_n must be <= 1000"):
            engine.query("query", top_n=1001)
        with pytest.raises(ValueError, match="distance must be one of"):
            engine.query("query", distance="COSINE_SIMILARITY); DROP TABLE T;--")

    def test_retrieval_procedure_identifiers_are_quoted(self):
        mod = importlib.import_module("hana_ai.retrieval.utility")
        sql = mod._call_procedure_sql("hello", {}, "SAFE_SCHEMA", "SAFE_PROC")
        assert 'CALL "SAFE_SCHEMA"."SAFE_PROC"' in sql
        with pytest.raises(ValueError):
            mod._call_procedure_sql("hello", {}, 'S"; DROP TABLE T;--', "P")

    def test_mem0_export_table_name_is_identifier_quoted(self):
        mod = importlib.import_module("hana_ai.mem0.memory_manager")
        assert mod._quote_hana_identifier("MEMORIES") == '"MEMORIES"'
        with pytest.raises(ValueError):
            mod._quote_hana_identifier('T"; DROP TABLE X;--')


# ============================ V-166: second-order group-key SQL ============================
@pytest.mark.parametrize(
    "module_name",
    [
        "hana_ai.tools.hana_ml_tools.ts_make_predict_table",
        "hana_ai.tools.hana_ml_tools.ts_check_tools",
    ],
)
def test_v166_group_values_and_identifiers_are_encoded(module_name):
    mod = importlib.import_module(module_name)
    assert mod._sql_str_lit("O'Reilly") == "'O''Reilly'"
    assert mod._sql_ident("GROUP_ID") == '"GROUP_ID"'
    with pytest.raises(ValueError):
        mod._sql_ident('G" OR 1=1--')
    assert mod._sql_number(1.25) == "1.25"
    with pytest.raises(ValueError):
        mod._sql_number("0; DROP TABLE T")


def test_v166_increment_type_is_allowlisted_and_none_keeps_legacy_default():
    mod = importlib.import_module(
        "hana_ai.tools.hana_ml_tools.ts_make_predict_table"
    )
    assert mod._normalize_increment_type(None) == "seconds"
    assert mod._normalize_increment_type("day") == "days"
    with pytest.raises(ValueError, match="increment_type"):
        mod._normalize_increment_type("SECONDS); DROP TABLE T;--")


# ============================ V-156: dangerous Python tool defaults off ============================
def test_v156_python_hanaml_exec_requires_host_opt_in():
    mod = importlib.import_module("hana_ai.tools.hana_ml_tools.python_exec_tools")
    connection_context = object.__new__(mod.ConnectionContext)
    tool = mod.PythonHanaMLExecTool(connection_context=connection_context)
    result = tool._run(code="result = 1", allow_dangerous_code=True)
    assert "disabled" in result
    assert "not a sandbox" in result


# ============================ V-161: FAISS pickle is never implicit ============================
class TestV161FAISSDeserialization:
    def _agent_shell(self, path):
        mod = importlib.import_module("hana_ai.agents.hanaml_rag_agent")
        agent = object.__new__(mod.HANAMLRAGAgent)
        agent.vectorstore_path = str(path)
        agent.embedding_service = object()
        return agent

    def test_incomplete_store_is_not_overwritten(self, tmp_path):
        (tmp_path / "index.faiss").write_bytes(b"existing")
        agent = self._agent_shell(tmp_path)
        with pytest.raises(RuntimeError, match="incomplete FAISS store"):
            agent._initialize_faiss_vectorstore()
        assert (tmp_path / "index.faiss").read_bytes() == b"existing"
        assert not (tmp_path / "index.pkl").exists()

    def test_existing_pickle_requires_explicit_trust(self, tmp_path, monkeypatch):
        (tmp_path / "index.faiss").write_bytes(b"existing")
        (tmp_path / "index.pkl").write_bytes(b"pickle")
        monkeypatch.delenv("HANA_ALLOW_DANGEROUS_FAISS_DESERIALIZATION", raising=False)
        agent = self._agent_shell(tmp_path)
        with pytest.raises(RuntimeError, match="pickle deserialization.*disabled"):
            agent._initialize_faiss_vectorstore()
        assert (tmp_path / "index.pkl").read_bytes() == b"pickle"


# ============================ V-168: dataframe rows never enter the prompt ============================
class TestV168DataframeAgent:
    class _DataFrame:
        columns = ["ID", "VALUE"]

        def head(self, *_args, **_kwargs):
            raise AssertionError("row data must not be collected for the prompt")

    def test_no_default_repl_tool(self, monkeypatch):
        mod = importlib.import_module("hana_ai.agents.hana_dataframe_agent")
        monkeypatch.setattr(mod, "_validate_hana_df", lambda _df: True)
        with pytest.raises(ValueError, match="PythonAstREPLTool is disabled by default"):
            mod.create_hana_dataframe_agent(llm=object(), df=self._DataFrame())

    def test_safe_tool_uses_schema_only(self, monkeypatch):
        mod = importlib.import_module("hana_ai.agents.hana_dataframe_agent")
        captured = {}

        class _Prompt:
            def partial(self, **kwargs):
                captured.update(kwargs)
                return self

        class _Agent:
            @staticmethod
            def create_prompt(*_args, **_kwargs):
                return _Prompt()

            def __init__(self, **_kwargs):
                pass

        class _Chain:
            def __init__(self, **_kwargs):
                pass

        class _Executor:
            @staticmethod
            def from_agent_and_tools(**kwargs):
                return kwargs

        monkeypatch.setattr(mod, "_validate_hana_df", lambda _df: True)
        monkeypatch.setattr(mod, "ZeroShotAgent", _Agent)
        monkeypatch.setattr(mod, "LLMChain", _Chain)
        monkeypatch.setattr(mod, "AgentExecutor", _Executor)
        tool = types.SimpleNamespace(name="safe_tool")
        result = mod.create_hana_dataframe_agent(
            llm=object(), df=self._DataFrame(), tools=[tool]
        )
        assert result["tools"] == [tool]
        assert captured["df"] == "HANA dataframe columns: ['ID', 'VALUE']"


# ============================ V-171: read-only SQL agent ============================
class TestV171ReadOnlySQLDatabase:
    def _db(self):
        sqlalchemy = pytest.importorskip("sqlalchemy")
        pytest.importorskip("langchain_community")
        mod = importlib.import_module("hana_ai.agents.hana_sql_agent")
        engine = sqlalchemy.create_engine("sqlite://")
        return mod._ReadOnlySQLDatabase(engine)

    def test_select_allowed(self):
        db = self._db()
        # SELECT executes against the empty in-memory sqlite without error.
        assert db.run("SELECT 1") is not None

    @pytest.mark.parametrize(
        "stmt",
        [
            "DROP TABLE foo",
            "DELETE FROM foo",
            "UPDATE foo SET a=1",
            "INSERT INTO foo VALUES (1)",
            "SELECT 1; DROP TABLE foo",  # stacked
            "SELECT 1 INTO foo",
        ],
    )
    def test_non_readonly_rejected(self, stmt):
        db = self._db()
        with pytest.raises(ValueError):
            db.run(stmt)

    def test_table_access_requires_and_enforces_allowlist(self):
        sqlalchemy = pytest.importorskip("sqlalchemy")
        mod = importlib.import_module("hana_ai.agents.hana_sql_agent")
        engine = sqlalchemy.create_engine("sqlite://")
        with engine.begin() as connection:
            connection.exec_driver_sql("CREATE TABLE allowed_table (id INTEGER)")
            connection.exec_driver_sql("CREATE TABLE secret_table (id INTEGER)")

        unscoped = mod._ReadOnlySQLDatabase(engine)
        with pytest.raises(ValueError, match="no table allowlist"):
            unscoped.run("SELECT * FROM secret_table")

        scoped = mod._ReadOnlySQLDatabase(
            engine, include_tables=["allowed_table"],
            allowed_tables=["allowed_table"]
        )
        assert scoped.run("SELECT * FROM allowed_table") == ""
        with pytest.raises(ValueError, match="not allowlisted"):
            scoped.run("SELECT * FROM secret_table")
        with pytest.raises(ValueError, match="comma-separated"):
            scoped.run("SELECT * FROM allowed_table, secret_table")

    def test_cte_cannot_hide_a_write(self):
        db = self._db()
        with pytest.raises(ValueError, match="forbidden keyword"):
            db.run("WITH changed AS (DELETE FROM foo) SELECT 1")


# ============================ V-174: stable identity, not repr parsing ============================
class TestV174Identity:
    def _get_user_info(self):
        mod = importlib.import_module("hana_ai.agents.utilities")
        return mod._get_user_info

    def test_prefers_userkey(self):
        cc = types.SimpleNamespace(userkey="MYKEY", connection=object())
        assert self._get_user_info()(cc) == "MYKEY"

    def test_structured_user_attribute(self):
        conn = types.SimpleNamespace(user="BOB")
        cc = types.SimpleNamespace(userkey=None, connection=conn)
        assert self._get_user_info()(cc) == "BOB"

    def test_repr_parsing_is_rejected(self):
        class _Conn:
            def __str__(self):
                return "<dbapi.Connection Connection object : host,port,CAROL,extra>"
        cc = types.SimpleNamespace(userkey=None, connection=_Conn())
        with pytest.raises(ValueError, match="stable HANA identity"):
            self._get_user_info()(cc)


# ============================ V-163: credential temp files always cleaned ============================
def test_v163_certificate_temp_files_removed_on_request_failure(monkeypatch):
    requests = pytest.importorskip("requests")
    mod = importlib.import_module("hana_ai.retrieval.utility")
    observed_paths = []

    def fail_post(_url, *, data, cert, timeout):
        assert data["grant_type"] == "client_credentials"
        assert timeout
        observed_paths.extend(cert)
        assert all(os.path.exists(path) for path in cert)
        raise RuntimeError("network failure")

    monkeypatch.setattr(requests, "post", fail_post)
    with pytest.raises(RuntimeError, match="network failure"):
        mod._get_access_token(
            {
                "certurl": "https://auth.example.test",
                "clientid": "client",
                "certificate": "certificate",
                "key": "private-key",
            }
        )
    assert len(observed_paths) == 2
    assert all(not os.path.exists(path) for path in observed_paths)


# ============================ V-155/V-167: MCP admin auth + TLS default ============================
def test_v155_static_token_verifier_enforces_http_requests():
    """Exercise FastMCP's actual ASGI authorization boundary, without a live port."""
    pytest.importorskip("fastmcp")
    from fastmcp import FastMCP
    from fastmcp.server.auth.providers.jwt import StaticTokenVerifier
    from starlette.testclient import TestClient

    token = "secret-token-xyz"
    verifier = StaticTokenVerifier({token: {"client_id": "hana-ai-admin", "scopes": []}})
    mcp = FastMCP("HANATools", auth=verifier)
    app = mcp.http_app(path="/mcp", stateless_http=True)
    with TestClient(app) as client:
        assert client.get("/mcp").status_code == 401
        assert client.get("/mcp", headers={"Authorization": "Bearer wrong"}).status_code == 401
        # GET is not a valid MCP request, so 406 here proves authentication passed and the
        # request reached the MCP transport layer.
        assert client.get("/mcp", headers={"Authorization": f"Bearer {token}"}).status_code == 406


def test_v155_runtime_http_admin_requires_token(monkeypatch):
    mod = importlib.import_module("hana_ai.tools.toolkit")
    toolkit = object.__new__(mod.HANAMLToolkit)
    monkeypatch.setenv("HANA_ALLOW_RUNTIME_CONN_UPDATE", "true")
    with pytest.raises(ValueError, match="require auth_token"):
        toolkit.launch_mcp_server(transport="http", auth_token=None, max_retries=1)


def test_http_mcp_non_loopback_binding_requires_token(monkeypatch):
    mod = importlib.import_module("hana_ai.tools.toolkit")
    toolkit = object.__new__(mod.HANAMLToolkit)
    monkeypatch.delenv("HANA_ALLOW_RUNTIME_CONN_UPDATE", raising=False)
    with pytest.raises(ValueError, match="outside loopback require auth_token"):
        toolkit.launch_mcp_server(
            transport="http", host="0.0.0.0", auth_token=None, max_retries=1
        )


def test_v155_runtime_admin_rejected_for_sse(monkeypatch):
    mod = importlib.import_module("hana_ai.tools.toolkit")
    toolkit = object.__new__(mod.HANAMLToolkit)
    monkeypatch.setenv("HANA_ALLOW_RUNTIME_CONN_UPDATE", "true")
    with pytest.raises(ValueError, match="not supported on SSE"):
        toolkit.launch_mcp_server(transport="sse", max_retries=1)


def test_v167_tls_certificate_validation_defaults_on(monkeypatch):
    mod = importlib.import_module("hana_ai.tools.toolkit")
    monkeypatch.setenv("HANA_ADDRESS", "hana.example.test")
    monkeypatch.setenv("HANA_USER", "USER")
    monkeypatch.setenv("HANA_PASSWORD", "secret")
    monkeypatch.delenv("HANA_SSL_VALIDATE", raising=False)
    assert mod._build_cc_params_from_env()["sslValidateCertificate"] is True


def test_v167_empty_tls_setting_uses_secure_default(monkeypatch):
    mod = importlib.import_module("hana_ai.tools.toolkit")
    monkeypatch.setenv("HANA_ADDRESS", "hana.example.test")
    monkeypatch.setenv("HANA_USER", "USER")
    monkeypatch.setenv("HANA_PASSWORD", "secret")
    monkeypatch.setenv("HANA_SSL_VALIDATE", "   ")
    assert mod._build_cc_params_from_env()["sslValidateCertificate"] is True
    monkeypatch.setenv("HANA_SSL_VALIDATE", "maybe")
    with pytest.raises(ValueError, match="must be a boolean"):
        mod._build_cc_params_from_env()


# ============================ V-169: tenant/session memory isolation ============================
class TestV169MemoryIsolation:
    class _Store:
        def __init__(self):
            self.docs = []
            self.deleted_filters = []

        def add_documents(self, docs):
            self.docs.extend(docs)
            return [str(index) for index, _ in enumerate(docs)]

        def similarity_search_with_score(self, query, k=5, **_kwargs):
            # Deliberately return every tenant. The adapter must filter again.
            return [(doc, 1.0) for doc in self.docs[:k]]

        def delete(self, filter):
            self.deleted_filters.append(filter)
            before = len(self.docs)
            self.docs = [
                doc for doc in self.docs
                if not all(doc.metadata.get(key) == value for key, value in filter.items())
            ]
            return before - len(self.docs)

    @staticmethod
    def _manager(store, tenant, session):
        adapter_mod = importlib.import_module("hana_ai.mem0.hana_mem0_adapter")
        manager_mod = importlib.import_module("hana_ai.mem0.memory_manager")
        adapter = adapter_mod.Mem0HanaAdapter(vectorstore=store)
        return manager_mod.Mem0MemoryManager(
            connection_context=None, table_name=None, adapter=adapter,
            tenant_id=tenant, session_id=session
        )

    def test_scope_is_required(self):
        manager_mod = importlib.import_module("hana_ai.mem0.memory_manager")
        with pytest.raises(ValueError, match="tenant_id and session_id"):
            manager_mod.Mem0MemoryManager(
                connection_context=None, table_name=None, adapter=object()
            )

    def test_add_retrieve_and_clear_are_scoped(self):
        store = self._Store()
        manager_a = self._manager(store, "tenant-a", "session-1")
        manager_b = self._manager(store, "tenant-b", "session-1")
        manager_a.add_memory("alpha")
        manager_b.add_memory("beta")

        assert manager_a.retrieve_texts("anything") == ["alpha"]
        assert manager_b.retrieve_texts("anything") == ["beta"]
        manager_a.clear_all()
        assert [doc.page_content for doc in store.docs] == ["beta"]
        assert store.deleted_filters[-1] == {
            "tenant_id": "tenant-a", "session_id": "session-1"
        }
