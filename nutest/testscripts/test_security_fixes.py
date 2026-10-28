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

    @pytest.mark.parametrize(
        "inline_arg", ["-cprint(1)", "-econsole.log(1)", "--eval=code", "/cwhoami"]
    )
    def test_inline_code_flag_prefix_rejected(self, inline_arg):
        with pytest.raises(ValueError, match="inline code"):
            self._validator()("python", [inline_arg])

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
            "SELECT 'safe' AS VALUE FROM DUMMY;",
            "WITH X AS (SELECT 1 AS A FROM DUMMY) SELECT * FROM X",
            "SELECT REPLACE('abc', 'a', 'z') FROM DUMMY",
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
            "WITH X AS (SELECT 1) REPLACE T (ID) VALUES (1)",
            "SELECT * FROM T FOR UPDATE",
            "SELECT * INTO X FROM T",
            "SELECT ';' AS VALUE FROM DUMMY",
            "SELECT '; DELETE FROM SENSITIVE_TABLE --' FROM DUMMY",
            'SELECT "; DELETE FROM SENSITIVE_TABLE --" FROM DUMMY',
        ],
    )
    def test_explanations_comments_stacking_and_writes_rejected(self, sql):
        with pytest.raises(ValueError):
            self._extract(sql)

    def test_shared_guard_rejects_before_connection_context_sql(self):
        guard = importlib.import_module("hana_ai.sql_guard")
        observed = []
        connection = types.SimpleNamespace(
            sql=lambda statement: observed.append(statement)
        )
        malicious_statements = [
            "DELETE FROM SENSITIVE_TABLE; SELECT 1 FROM DUMMY",
            "SELECT '; DELETE FROM SENSITIVE_TABLE --' FROM DUMMY",
            'SELECT "; DELETE FROM SENSITIVE_TABLE --" FROM DUMMY',
        ]
        for statement in malicious_statements:
            with pytest.raises(ValueError):
                guard.readonly_dataframe(connection, statement)
        assert observed == []

        frame = object()
        connection.sql = lambda statement: observed.append(statement) or frame
        assert guard.readonly_dataframe(
            connection, "SELECT 'safe' AS VALUE FROM DUMMY;"
        ) is frame
        assert observed == ["SELECT 'safe' AS VALUE FROM DUMMY"]

    def test_fetch_data_tool_rejects_stacked_sql_before_execution(self):
        mod = importlib.import_module("hana_ai.tools.df_tools.fetch_tools")
        observed = []
        connection = types.SimpleNamespace(
            sql=lambda statement: observed.append(statement)
        )
        tool = mod.FetchDataTool.model_construct(
            connection_context=connection,
            return_direct=False,
            is_transform=False,
        )
        with pytest.raises(ValueError, match="semicolon|SELECT or WITH|stacked"):
            tool._run(
                select_statement="DELETE FROM SENSITIVE_TABLE; SELECT 1 FROM DUMMY"
            )
        assert observed == []

    def test_select_statement_to_table_rejects_stacked_sql_before_execution(self):
        mod = importlib.import_module(
            "hana_ai.tools.hana_ml_tools.select_statement_to_table_tools"
        )
        observed = []
        connection = types.SimpleNamespace(
            sql=lambda statement: observed.append(statement)
        )
        result = mod.SelectStatement_to_table(
            "DELETE FROM SENSITIVE_TABLE; SELECT 1 FROM DUMMY",
            connection,
            "TARGET_TABLE",
            None,
            False,
        )
        assert "Operation failed" in result
        assert observed == []


# ============================ V-157/V-158: SQL value + identifier encoding ============================
class TestV157V158SQLGuards:
    def test_vector_string_literal_doubles_quotes(self):
        mod = importlib.import_module("hana_ai.vectorstore.hana_vector_engine")
        assert mod._sql_string_literal("O'Reilly") == "'O''Reilly'"
        with pytest.raises(ValueError, match="semicolons"):
            mod._sql_string_literal("x'; DROP TABLE T;--")

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

    @pytest.mark.parametrize(
        "field,payload",
        [
            ("input", "; DELETE FROM SENSITIVE_TABLE; --"),
            ("model_version", "MODEL; DELETE FROM SENSITIVE_TABLE; --"),
            ("schema", "SCHEMA; DELETE FROM SENSITIVE_TABLE; --"),
            ("table_name", "VECTORS; DELETE FROM SENSITIVE_TABLE; --"),
        ],
    )
    def test_vector_semicolon_payload_executes_zero_sql(self, field, payload):
        mod = importlib.import_module("hana_ai.vectorstore.hana_vector_engine")
        executed = []
        engine = object.__new__(mod.HANAMLinVectorEngine)
        engine.schema = "SCHEMA"
        engine.table_name = "VECTORS"
        engine.columns = ["MODEL", "DESCRIPTION", "EXAMPLE", "EMBEDDINGS"]
        engine.model_version = "SAP_NEB.20240715"
        engine.connection_context = types.SimpleNamespace(sql=executed.append)
        query = "safe query"
        if field == "input":
            query = payload
        else:
            setattr(engine, field, payload)
        with pytest.raises(ValueError, match="semicolon|identifier"):
            engine.query(query)
        assert executed == []

    def test_retrieval_procedure_identifiers_are_quoted(self):
        mod = importlib.import_module("hana_ai.retrieval.utility")
        sql = mod._call_procedure_sql("hello", {}, "SAFE_SCHEMA", "SAFE_PROC")
        assert 'CALL "SAFE_SCHEMA"."SAFE_PROC"' in sql
        with pytest.raises(ValueError):
            mod._call_procedure_sql("hello", {}, 'S"; DROP TABLE T;--', "P")

    @pytest.mark.parametrize(
        "overrides",
        [
            {"order_by": "ID; DELETE FROM SENSITIVE_TABLE; --"},
            {"schema_name": "SAFE; DELETE FROM SENSITIVE_TABLE; --"},
            {"source_table": "SOURCE; DELETE FROM SENSITIVE_TABLE; --"},
            {"train_table": "TRAIN; DELETE FROM SENSITIVE_TABLE; --"},
            {"test_table": "TEST; DELETE FROM SENSITIVE_TABLE; --"},
            {"validation_table": "VALIDATION; DELETE FROM SENSITIVE_TABLE; --"},
        ],
    )
    def test_dataset_split_semicolon_payload_executes_zero_sql(self, overrides):
        mod = importlib.import_module(
            "hana_ai.tools.hana_ml_tools.dataset_prep_tools"
        )
        executed = []
        source_df = types.SimpleNamespace(columns=["ID", "VALUE"])
        connection = types.SimpleNamespace(
            table=lambda *_args, **_kwargs: executed.append("table metadata") or source_df,
            sql=lambda statement: executed.append(statement),
        )
        tool = mod.SplitTableForForecastingTool.model_construct(
            connection_context=connection, return_direct=False
        )
        request = {
            "source_table": "SOURCE",
            "schema_name": "SAFE",
            "split_mode": "time_ordered",
            "train_ratio": 0.7,
            "test_ratio": 0.2,
            "validation_ratio": 0.1,
            "order_by": "ID",
            "force": False,
        }
        request.update(overrides)
        result = tool._run(**request)
        assert result.startswith("Operation failed while splitting")
        assert executed == []

    def test_dataset_split_order_column_must_exist_before_sql_execution(self):
        mod = importlib.import_module(
            "hana_ai.tools.hana_ml_tools.dataset_prep_tools"
        )
        executed = []
        connection = types.SimpleNamespace(
            table=lambda *_args, **_kwargs: types.SimpleNamespace(columns=["ID"]),
            sql=executed.append,
        )
        tool = mod.SplitTableForForecastingTool.model_construct(
            connection_context=connection, return_direct=False
        )
        result = tool._run(
            source_table="SOURCE", schema_name="SAFE",
            split_mode="time_ordered", train_ratio=0.7, test_ratio=0.2,
            validation_ratio=0.1, order_by="MISSING", force=False,
        )
        assert "was not found in the source table" in result
        assert executed == []

    def test_retrieval_infrastructure_sql_encodes_names_and_values(self, monkeypatch):
        mod = importlib.import_module("hana_ai.retrieval.utility")
        assert mod._quote_hana_object_identifier("mixed_name") == '"MIXED_NAME"'
        assert mod._quote_hana_object_identifier('"MixedName"') == '"MixedName"'
        with pytest.raises(ValueError, match="invalid HANA object identifier"):
            mod._quote_hana_object_identifier(None)
        with pytest.raises(ValueError, match="invalid HANA object identifier"):
            mod._quote_hana_object_identifier('PSE"; DROP PSE SAFE;--')

        pse_sql = mod._create_pse_sql_string(
            {"key": "private'key", "certificate": "certificate"}, "safe_pse"
        )
        assert 'CREATE PSE "SAFE_PSE"' in pse_sql
        assert "private''key" in pse_sql

        monkeypatch.setattr(mod, "_get_deployment_id", lambda _credentials: "deploy'id")
        remote_sql = mod._create_ai_core_remote_source_sql_string(
            {
                "serviceurls": {"AI_API_URL": "https://api.example.test/a'b"},
                "certurl": "https://auth.example.test",
                "clientid": "client'id",
            },
            "safe_source",
            "safe_pse",
        )
        assert 'CREATE REMOTE SOURCE "SAFE_SOURCE"' in remote_sql
        assert 'PSE "SAFE_PSE"' in remote_sql
        assert "a''b" in remote_sql
        assert "deploy''id" in remote_sql
        assert "client''id" in remote_sql

    def test_retrieval_infrastructure_create_and_drop_identifiers(self, monkeypatch):
        mod = importlib.import_module("hana_ai.retrieval.utility")
        executed = []
        connection = object()
        monkeypatch.setattr(
            mod, "_execute_sql_string", lambda _connection, sql: executed.append(sql)
        )

        mod._create_certificate_and_add_to_pse(
            connection, "root_cert", "certificate'body", "safe_pse"
        )
        assert executed == [
            'CREATE CERTIFICATE "ROOT_CERT" FROM \'certificate\'\'body\'',
            'ALTER PSE "SAFE_PSE" ADD CERTIFICATE "ROOT_CERT"',
        ]

        mod._delete_ai_core_pse(object(), "safe_pse")
        mod._drop_ai_core_remote_source(object(), "safe_source", cascade=False)
        mod._drop_certificate(object(), "root_cert")
        assert executed[-3:] == [
            'DROP PSE "SAFE_PSE" CASCADE;',
            'DROP REMOTE SOURCE "SAFE_SOURCE";',
            'DROP CERTIFICATE "ROOT_CERT";',
        ]

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


def test_v166_stored_group_semicolon_executes_zero_sql():
    mod = importlib.import_module(
        "hana_ai.tools.hana_ml_tools.ts_make_predict_table"
    )
    executed = []

    class _GroupSelection:
        def distinct(self):
            return self

        @staticmethod
        def collect():
            return {"GROUP_ID": ["A; DELETE FROM SENSITIVE_TABLE; --"]}

    class _Data:
        columns = ["GROUP_ID", "TIME_ID"]
        index = None
        connection_context = types.SimpleNamespace(sql=executed.append)

        @staticmethod
        def get_table_structure():
            return {"GROUP_ID": "NVARCHAR(100)", "TIME_ID": "INTEGER"}

        @staticmethod
        def select(column):
            assert column == "GROUP_ID"
            return _GroupSelection()

        @staticmethod
        def filter(_condition):
            raise AssertionError("the stored payload must be rejected before filtering")

    with pytest.raises(ValueError, match="semicolons"):
        mod.make_future_dataframe_for_massive_forecast(
            data=_Data(), key="TIME_ID", group_key="GROUP_ID"
        )
    assert executed == []


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

    def test_locally_created_store_reloads_after_integrity_check(
        self, tmp_path, monkeypatch
    ):
        mod = importlib.import_module("hana_ai.agents.hanaml_rag_agent")

        class _CreatedStore:
            @staticmethod
            def save_local(path):
                target = os.fspath(path)
                with open(os.path.join(target, "index.faiss"), "wb") as handle:
                    handle.write(b"faiss-index")
                with open(os.path.join(target, "index.pkl"), "wb") as handle:
                    handle.write(b"trusted-pickle")

        agent = self._agent_shell(tmp_path)
        agent.vectorstore = _CreatedStore()
        agent._save_faiss_vectorstore()
        assert (tmp_path / ".hana_ai_faiss.sha256").is_file()

        loaded = object()

        class _FAISS:
            @staticmethod
            def load_local(path, embedding, allow_dangerous_deserialization):
                assert path == str(tmp_path)
                assert embedding is not None
                assert allow_dangerous_deserialization is True
                return loaded

        monkeypatch.setattr(mod, "FAISS", _FAISS)
        monkeypatch.delenv("HANA_ALLOW_DANGEROUS_FAISS_DESERIALIZATION", raising=False)
        restarted = self._agent_shell(tmp_path)
        restarted._initialize_faiss_vectorstore()
        assert restarted.vectorstore is loaded

        (tmp_path / "index.pkl").write_bytes(b"tampered")
        with pytest.raises(RuntimeError, match="manifest is missing or invalid"):
            self._agent_shell(tmp_path)._initialize_faiss_vectorstore()


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
        with pytest.raises(ValueError, match="only SELECT/WITH"):
            db.run("WITH changed AS (SELECT 1) REPLACE foo (id) VALUES (1)")

    def test_quoted_identifier_cannot_mask_cross_schema_table(self):
        sqlalchemy = pytest.importorskip("sqlalchemy")
        mod = importlib.import_module("hana_ai.agents.hana_sql_agent")
        db = mod._ReadOnlySQLDatabase(
            sqlalchemy.create_engine("sqlite://"),
            allowed_schema="PUBLIC",
            allowed_tables=["T"],
        )
        with pytest.raises(ValueError, match="allowed schema"):
            db.run("SELECT \"a'b\" FROM SECRET.T WHERE y = 'z' AND \"c'd\"")

    def test_unqualified_table_allowlist_rejects_other_schema(self):
        sqlalchemy = pytest.importorskip("sqlalchemy")
        mod = importlib.import_module("hana_ai.agents.hana_sql_agent")
        db = mod._ReadOnlySQLDatabase(
            sqlalchemy.create_engine("sqlite://"), allowed_tables=["T"]
        )
        with pytest.raises(ValueError, match="schema-qualified tables"):
            db.run("SELECT * FROM OTHER.T")

    def test_replace_scalar_function_remains_allowed(self):
        db = self._db()
        assert db.run("SELECT REPLACE('abc', 'a', 'z')") is not None

    def test_quoted_identifier_case_does_not_bypass_allowlist(self):
        sqlalchemy = pytest.importorskip("sqlalchemy")
        mod = importlib.import_module("hana_ai.agents.hana_sql_agent")
        db = mod._ReadOnlySQLDatabase(
            sqlalchemy.create_engine("sqlite://"),
            allowed_schema="PUBLIC",
            allowed_tables=["SENSITIVE"],
        )
        db._enforce_object_allowlist('SELECT * FROM PUBLIC."SENSITIVE"')
        with pytest.raises(ValueError, match="not allowlisted"):
            db.run('SELECT * FROM PUBLIC."sensitive"')

    @pytest.mark.parametrize(
        "stmt,error",
        [
            (
                "SELECT SECRET_TABLE.ID FROM (SECRET_TABLE CROSS JOIN ALLOWED_TABLE)",
                "not allowlisted",
            ),
            ("SELECT SECRET_TABLE.ID FROM (SECRET_TABLE)", "not allowlisted"),
            (
                "SELECT SECRET_TABLE.ID FROM "
                "(SECRET_TABLE AS SECRET_TABLE CROSS JOIN ALLOWED_TABLE AS ALLOWED_TABLE)",
                "not allowlisted",
            ),
            (
                "WITH A AS (WITH SECRET_TABLE AS (SELECT 1 AS ID) "
                "SELECT * FROM SECRET_TABLE) SELECT * FROM SECRET_TABLE",
                "not allowlisted",
            ),
            (
                "SELECT SECRET_SCHEMA.GET_SECRET() FROM ALLOWED_TABLE",
                "qualified function",
            ),
            (
                "SELECT SECRET_SCHEMA.SECRET_SEQ.NEXTVAL FROM ALLOWED_TABLE",
                "sequence access",
            ),
            ("SELECT * FROM DATABASE.PUBLIC.ALLOWED_TABLE", "three-part"),
            ("SELECT * FROM SECRET_TABLE_FUNCTION()", "table functions"),
            ("SELECT GET_SECRET() FROM ALLOWED_TABLE", "function 'GET_SECRET'"),
            ("SELECT 1+GET_SECRET() FROM ALLOWED_TABLE", "function 'GET_SECRET'"),
            ("SELECT 1-SECRET_SCHEMA.GET_SECRET() FROM ALLOWED_TABLE", "qualified function"),
            ("SELECT 1e-2+SECRET_SCHEMA.GET_SECRET() FROM ALLOWED_TABLE", "qualified function"),
            ("SELECT 1+SECRET_SCHEMA.SECRET_SEQ.NEXTVAL FROM ALLOWED_TABLE", "sequence access"),
            ("SELECT 1-SECRET_SEQ.CURRVAL FROM ALLOWED_TABLE", "sequence access"),
        ],
    )
    def test_object_parser_blocks_review8_bypasses_before_execute(self, stmt, error):
        sqlalchemy = pytest.importorskip("sqlalchemy")
        mod = importlib.import_module("hana_ai.agents.hana_sql_agent")
        engine = sqlalchemy.create_engine("sqlite://")
        with engine.begin() as connection:
            connection.exec_driver_sql("CREATE TABLE allowed_table (id INTEGER)")
            connection.exec_driver_sql("CREATE TABLE secret_table (id INTEGER)")
            connection.exec_driver_sql("INSERT INTO secret_table VALUES (42)")
        db = mod._ReadOnlySQLDatabase(
            engine, include_tables=["allowed_table"],
            allowed_tables=["allowed_table"],
        )
        executed = []
        sqlalchemy.event.listen(
            engine,
            "before_cursor_execute",
            lambda *_args: executed.append(_args[2]),
        )
        with pytest.raises(ValueError, match=error):
            db.run(stmt)
        assert executed == []

    @pytest.mark.parametrize(
        "statement",
        [
            "SELECT (1+2)",
            "SELECT 1 WHERE (1=1) AND (2=2) OR (3=3)",
            "SELECT 1 WHERE NOT (1=2)",
            "SELECT 1 WHERE 1 IN (SELECT (1))",
            "WITH X AS (SELECT 1 AS ID) SELECT * FROM X WHERE (ID=1)",
            "WITH X AS (SELECT 1 AS ID) SELECT A.ID FROM X AS A "
            "LEFT OUTER JOIN X AS B ON A.ID=B.ID",
        ],
    )
    def test_object_parser_accepts_parenthesized_sql_expressions(self, statement):
        assert self._db().run(statement)

    def test_object_parser_preserves_cte_scope_and_parenthesized_allowed_joins(self):
        sqlalchemy = pytest.importorskip("sqlalchemy")
        mod = importlib.import_module("hana_ai.agents.hana_sql_agent")
        db = mod._ReadOnlySQLDatabase(
            sqlalchemy.create_engine("sqlite://"),
            allowed_tables=["ALLOWED_TABLE"],
        )
        db._enforce_object_allowlist(
            "WITH X AS (SELECT * FROM ALLOWED_TABLE) SELECT * FROM X"
        )
        db._enforce_object_allowlist(
            "SELECT * FROM (ALLOWED_TABLE AS A CROSS JOIN ALLOWED_TABLE AS B)"
        )

    @pytest.mark.parametrize("table_name", ["MixedCase", "with space", "literal.dot"])
    @pytest.mark.parametrize("factory_name", ["create_hana_sql_toolkit", "create_hana_sql_agent"])
    def test_factories_preserve_quoted_reflection_names(self, monkeypatch, table_name, factory_name):
        sqlalchemy = pytest.importorskip("sqlalchemy")
        mod = importlib.import_module("hana_ai.agents.hana_sql_agent")
        engine = sqlalchemy.create_engine("sqlite://")
        identifier = '"' + table_name + '"'
        with engine.begin() as connection:
            connection.exec_driver_sql(f"CREATE TABLE {identifier} (id INTEGER)")
            connection.exec_driver_sql(f"INSERT INTO {identifier} VALUES (7)")
        monkeypatch.setattr(mod, "_sql_toolkit", lambda llm, db, **_kwargs: db)
        monkeypatch.setattr(mod, "create_sql_agent", lambda **kwargs: kwargs["toolkit"])
        db = getattr(mod, factory_name)(
            llm=object(),
            connection_context=types.SimpleNamespace(to_sqlalchemy=lambda: engine),
            include_tables=[identifier],
        )
        assert db.run(f"SELECT * FROM {identifier}") == "[(7,)]"


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

    def test_callable_context_user_attribute(self):
        cc = types.SimpleNamespace(
            userkey=None, user=lambda: "BOB", connection=object()
        )
        assert self._get_user_info()(cc) == "BOB"

    def test_standard_connection_falls_back_to_current_user_query(self):
        class _Cursor:
            closed = False

            def execute(self, statement):
                assert statement == "SELECT CURRENT_USER FROM DUMMY"

            @staticmethod
            def fetchone():
                return ("PAL_TEST",)

            def close(self):
                self.closed = True

        cursor = _Cursor()
        connection = types.SimpleNamespace(cursor=lambda: cursor)
        cc = types.SimpleNamespace(userkey=None, connection=connection)
        assert self._get_user_info()(cc) == "PAL_TEST"
        assert cursor.closed is True

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


def test_v155_auth_token_rejected_for_unsupported_sse(monkeypatch):
    mod = importlib.import_module("hana_ai.tools.toolkit")
    toolkit = object.__new__(mod.HANAMLToolkit)
    monkeypatch.delenv("HANA_ALLOW_RUNTIME_CONN_UPDATE", raising=False)
    with pytest.raises(ValueError, match="auth_token is not supported on SSE"):
        toolkit.launch_mcp_server(transport="sse", auth_token="secret", max_retries=1)


def test_v155_sse_non_loopback_binding_rejected(monkeypatch):
    mod = importlib.import_module("hana_ai.tools.toolkit")
    toolkit = object.__new__(mod.HANAMLToolkit)
    monkeypatch.delenv("HANA_ALLOW_RUNTIME_CONN_UPDATE", raising=False)
    with pytest.raises(ValueError, match="restricted to loopback"):
        toolkit.launch_mcp_server(
            transport="sse", host="0.0.0.0", auth_token=None, max_retries=1
        )


def test_v167_tls_certificate_validation_defaults_on(monkeypatch):
    mod = importlib.import_module("hana_ai.tools.toolkit")
    monkeypatch.delenv("HANA_USERKEY", raising=False)
    monkeypatch.setenv("HANA_ADDRESS", "hana.example.test")
    monkeypatch.setenv("HANA_USER", "USER")
    monkeypatch.setenv("HANA_PASSWORD", "secret")
    monkeypatch.delenv("HANA_SSL_VALIDATE", raising=False)
    assert mod._build_cc_params_from_env()["sslValidateCertificate"] is True


def test_v167_empty_tls_setting_uses_secure_default(monkeypatch):
    mod = importlib.import_module("hana_ai.tools.toolkit")
    monkeypatch.delenv("HANA_USERKEY", raising=False)
    monkeypatch.setenv("HANA_ADDRESS", "hana.example.test")
    monkeypatch.setenv("HANA_USER", "USER")
    monkeypatch.setenv("HANA_PASSWORD", "secret")
    monkeypatch.setenv("HANA_SSL_VALIDATE", "   ")
    assert mod._build_cc_params_from_env()["sslValidateCertificate"] is True
    monkeypatch.setenv("HANA_SSL_VALIDATE", "maybe")
    with pytest.raises(ValueError, match="must be a boolean"):
        mod._build_cc_params_from_env()


def test_v167_userkey_env_uses_secure_tls_default(monkeypatch):
    mod = importlib.import_module("hana_ai.tools.toolkit")
    monkeypatch.setenv("HANA_USERKEY", "XinKey2")
    monkeypatch.delenv("HANA_ENCRYPT", raising=False)
    monkeypatch.delenv("HANA_SSL_VALIDATE", raising=False)
    assert mod._build_cc_params_from_env() == {
        "userkey": "XinKey2",
        "encrypt": True,
        "sslValidateCertificate": True,
    }


def test_v170_memory_admin_commands_default_to_disabled(monkeypatch):
    mod = importlib.import_module("hana_ai.agents.mem0_hana_agent")
    agent = object.__new__(mod.Mem0HANARAGAgent)
    monkeypatch.delenv("HANA_MEM0_ALLOW_ADMIN_COMMANDS", raising=False)
    assert "disabled" in agent.chat("!clear_long_term_memory").lower()


def test_v172_sanitized_session_names_do_not_collide():
    mod = importlib.import_module("hana_ai.iagents.context_agent")
    assert mod._session_storage_name("a/b") != mod._session_storage_name("a_b")
    assert "/" not in mod._session_storage_name("../../outside")
    assert ".." not in mod._session_storage_name("../../outside")


def test_v173_raw_report_rendering_requires_opt_in(monkeypatch, tmp_path):
    mod = importlib.import_module(
        "hana_ai.tools.hana_ml_tools.ts_visualizer_tools"
    )

    class _Frame:
        columns = ["TS", "VALUE"]

        def select(self, *_args):
            return self

    frame = _Frame()
    connection = types.SimpleNamespace(
        has_table=lambda *_args, **_kwargs: True,
        table=lambda *_args, **_kwargs: frame,
    )
    display_calls = []

    class _Report:
        def build(self, **_kwargs):
            return self

        def display(self, **kwargs):
            display_calls.append(kwargs)

    monkeypatch.setattr(mod, "UnifiedReport", lambda _frame: _Report())
    tool = mod.TimeSeriesDatasetReport.model_construct(
        connection_context=connection, return_direct=False, bas=False
    )
    monkeypatch.delenv("HANA_ALLOW_RAW_HTML_RENDER", raising=False)
    tool._run(
        table_name="T", key="TS", endog="VALUE", output_dir=str(tmp_path)
    )
    assert display_calls == [{"save_html": os.path.join(str(tmp_path), "T_ts_report")}]

    monkeypatch.setenv("HANA_ALLOW_RAW_HTML_RENDER", "true")
    tool._run(
        table_name="T", key="TS", endog="VALUE", output_dir=str(tmp_path)
    )
    assert display_calls[-2:] == [
        {"save_html": os.path.join(str(tmp_path), "T_ts_report")}, {}
    ]


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
