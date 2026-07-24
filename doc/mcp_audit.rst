MCP session auditing
====================

This page answers the practical question: *when a tool call happens on an
MCP server that is running on some host you don't control, what does the
HANA side see, and how does an auditor with only HANA access query it?*

Two audit sinks are involved:

* **HANA-side** — populated automatically on every tool call via
  ``setclientinfo``. Queryable from ``M_SQL_PLAN_CACHE`` and (when the DBA
  enables audit policies) ``AUDIT_LOG``. No coordination with the MCP host.
* **MCP-side JSONL** — the full audit event stream, written where the MCP
  server runs. Carries the MCP-only fields that HANA never sees, notably
  ``RESPONSE_SIZE``, ``STATUS``, ``DURATION_MS``.

The two are linked by ``INVOCATION_ID`` / ``HANA_CORRELATION_ID`` so the
picture is stitchable across both sides.


What lands on the HANA side automatically
-----------------------------------------

Every tool call sets five ``setclientinfo`` channels on the toolkit's HANA
connection before executing SQL:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - setclientinfo channel
     - Carried value
     - HANA-side visibility
   * - ``APPLICATION``
     - MCP client's declared name
     - ``M_CONNECTIONS.APPLICATION_NAME``, ``AUDIT_LOG.APPLICATION_NAME``, ``M_SQL_PLAN_CACHE.APPLICATION_NAME``
   * - ``APPLICATIONUSER``
     - MCP client's declared end-user / id
     - ``AUDIT_LOG.APPLICATION_USER_NAME``
   * - ``APPLICATIONVERSION``
     - client version / env tag
     - ``M_CONNECTIONS.APPLICATION_VERSION``
   * - ``APPLICATIONCOMPONENT``
     - ``TOOL_NAME``
     - ``M_CONNECTIONS.APPLICATION_COMPONENT``
   * - ``APPLICATIONSOURCE``
     - Pipe-delimited pack of MCP identity + per-call correlation ids + redacted tool args
     - ``M_SQL_PLAN_CACHE.APPLICATION_SOURCE``, ``AUDIT_LOG`` (when audit policy is on)

The ``APPLICATIONSOURCE`` pack is the key mechanism: HANA gives us five
setclientinfo slots, but we have ~15 fields to surface, so all the MCP-only
identity + invocation ids ride on one channel in a compact key=value form.

Pack layout::

    mcp=hana-ai/<ver>|sess=<mcp_session_id>|agent=<agent_name>|model=<model_name>
      |cli=<client_name>|mcp_ip=<client_ip>|tool=<tool_name>|inv=<invocation_id>
      |corr=<hana_correlation_id>|args=<urlsafe_b64_of_redacted_json>[|argstrunc=1]

Real example captured from a live MCP HTTP session::

    mcp=hana-ai/1.1.26072000|sess=d71dfba1752941aab632db0f6c2adb01
    |agent=probe-agent|model=claude-opus-4-8|cli=appsource-probe-535c0807
    |mcp_ip=127.0.0.1|tool=list_models
    |inv=inv-b98880b44944408286382131b7e2f92c
    |corr=hana-corr-d4ba732436c8422085147bf45e81b8ae

Pack contract:

* **ASCII-only**, **≤ 254 bytes** — HANA silently truncates
  ``APPLICATIONSOURCE`` past 256 wire bytes, so the builder enforces a hard
  254-byte cap and refuses non-ASCII input (``tool_args_json`` is serialised
  with ``ensure_ascii=True`` then base64-encoded).
* **Field order is fixed** so simple ``SUBSTR_REGEXPR`` on the HANA side
  works.
* **args is always last** and is the only field that can be truncated. When
  it is, ``argstrunc=1`` is appended so consumers know to fall back to the
  MCP-side sink for the full payload.
* **Sanitised** — every field value is passed through a
  ``[A-Za-z0-9._:/@-]`` whitelist so a malicious tool name cannot inject
  fake ``|sess=`` segments.
* **Plan-cache safe** — verified against HANA Cloud 4.00 that
  ``setclientinfo`` does not participate in ``STATEMENT_HASH``, so a fresh
  ``inv=`` per call does not fragment ``M_SQL_PLAN_CACHE``.


Querying HANA-side without an audit policy
------------------------------------------

If your DBA cannot or will not enable HANA audit policies, an auditor with
``SELECT`` on ``M_SQL_PLAN_CACHE`` alone can already reconstruct which MCP
session / agent / tool / invocation produced which SQL:

.. code-block:: sql

    SELECT DISTINCT
           STATEMENT_HASH,
           SUBSTR_REGEXPR('sess=([^|]*)' IN APPLICATION_SOURCE GROUP 1)  AS MCP_SESSION_ID,
           SUBSTR_REGEXPR('agent=([^|]*)' IN APPLICATION_SOURCE GROUP 1) AS AGENT_NAME,
           SUBSTR_REGEXPR('model=([^|]*)' IN APPLICATION_SOURCE GROUP 1) AS MODEL_NAME,
           SUBSTR_REGEXPR('tool=([^|]*)'  IN APPLICATION_SOURCE GROUP 1) AS TOOL_NAME,
           SUBSTR_REGEXPR('inv=([^|]*)'   IN APPLICATION_SOURCE GROUP 1) AS INVOCATION_ID,
           APPLICATION_USER_NAME,   -- MCP-declared end user
           APPLICATION_NAME,         -- MCP-declared client name
           USER_NAME,                -- HANA-authenticated DB user
           EXECUTION_COUNT,
           LAST_EXECUTION_TIMESTAMP
    FROM   M_SQL_PLAN_CACHE
    WHERE  APPLICATION_SOURCE LIKE 'mcp=hana-ai/%'
    ORDER  BY LAST_EXECUTION_TIMESTAMP DESC;

Limits of this view:

* **First-execution-wins semantics**. HANA writes
  ``M_SQL_PLAN_CACHE.APPLICATION_SOURCE`` at plan-caching time and does
  not update it on subsequent executions. Different tools produce
  different SQL, so each tool's rows carry that tool's own pack; but
  repeated calls to the *same* tool from *different* MCP sessions all
  appear under the first session's pack. For per-invocation resolution
  across identical SQL calls, use the MCP-side JSONL sink.
* **Plan cache ages out** — this is a near-term audit view, not long-term
  retention. Once a plan is evicted the row is gone.
* **No fields the plan cache doesn't already carry** — ``RESPONSE_SIZE``,
  ``STATUS``, ``DURATION_MS``, and the full un-truncated ``TOOL_ARGS_JSON``
  live only in the MCP-side sink.


Querying HANA-side with an audit policy
---------------------------------------

For long-term retention on HANA, the DBA enables audit policies on the
relevant schemas:

.. code-block:: sql

    -- One-time DBA setup
    ALTER SYSTEM ALTER CONFIGURATION ('global.ini','SYSTEM')
      SET ('auditing configuration','global_auditing_state') = 'true'
      WITH RECONFIGURE;

    ALTER SYSTEM ALTER CONFIGURATION ('global.ini','SYSTEM')
      SET ('auditing configuration','maximum_statement_string_length') = '16000'
      WITH RECONFIGURE;

    CREATE AUDIT POLICY p_mcp_tools
      AUDITING SUCCESSFUL EXECUTE ON SCHEMA "MY_MCP_SCHEMA"
      LEVEL INFO TRAIL TYPE TABLE;
    ALTER AUDIT POLICY p_mcp_tools ENABLE;

Once policies are on, the same pack lands in ``AUDIT_LOG`` alongside every
audited SQL, permanently:

.. code-block:: sql

    SELECT TIMESTAMP,
           USER_NAME,
           APPLICATION_USER_NAME,
           APPLICATION_NAME,
           SUBSTR_REGEXPR('sess=([^|]*)' IN CLIENT_APPLICATION_SOURCE GROUP 1) AS MCP_SESSION_ID,
           SUBSTR_REGEXPR('inv=([^|]*)'  IN CLIENT_APPLICATION_SOURCE GROUP 1) AS INVOCATION_ID,
           SUBSTR_REGEXPR('tool=([^|]*)' IN CLIENT_APPLICATION_SOURCE GROUP 1) AS TOOL_NAME,
           STATEMENT_STRING
    FROM   AUDIT_LOG
    WHERE  CLIENT_APPLICATION_SOURCE LIKE 'mcp=hana-ai/%'
      AND  TIMESTAMP > ADD_SECONDS(CURRENT_TIMESTAMP, -86400)
    ORDER  BY TIMESTAMP DESC;

.. note::

   The audit-log column that mirrors ``setclientinfo('APPLICATIONSOURCE')``
   may be exposed under different names on different HANA builds
   (``CLIENT_APPLICATION_SOURCE`` on recent HANA Cloud). Adjust the
   projection accordingly.


Fields → HANA-side mapping
--------------------------

The audit event columns emitted by
:func:`hana_ai.tools.hana_ml_tools.utility.fetch_mcp_audit_rows` map to
HANA-side visibility as follows:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Audit event field
     - HANA-side location
   * - ``MCP_SESSION_ID``
     - Pack ``sess=`` in ``M_SQL_PLAN_CACHE.APPLICATION_SOURCE`` / ``AUDIT_LOG.CLIENT_APPLICATION_SOURCE``
   * - ``CLIENT_IP``
     - Pack ``mcp_ip=`` (MCP-observed remote IP, distinct from HANA-observed peer)
   * - ``CLIENT_DECLARED_NAME``
     - ``M_CONNECTIONS.APPLICATION_NAME``, pack ``cli=``
   * - ``CLIENT_DECLARED_AGENT_NAME``
     - Pack ``agent=``
   * - ``CLIENT_DECLARED_MODEL_NAME``
     - Pack ``model=``
   * - ``HANA_DB_USER``
     - ``M_CONNECTIONS.USER_NAME`` / ``AUDIT_LOG.USER_NAME`` (HANA-authenticated identity)
   * - ``HANA_DB_SESSION_USER``
     - ``M_CONNECTIONS.SESSION_USER_NAME``
   * - ``HANA_CONNECTION_ID``
     - ``M_CONNECTIONS.CONNECTION_ID``
   * - ``HANA_APPLICATION_USER``
     - ``AUDIT_LOG.APPLICATION_USER_NAME`` (from ``APPLICATIONUSER``)
   * - ``HANA_APPLICATION``
     - ``M_CONNECTIONS.APPLICATION_NAME`` (from ``APPLICATION``)
   * - ``HANA_CLIENT_HOST``
     - ``M_CONNECTIONS.CLIENT_HOST``
   * - ``TOOL_NAME``
     - Pack ``tool=`` + ``M_CONNECTIONS.APPLICATION_COMPONENT``
   * - ``TARGET_TABLES``
     - Not packed (length unpredictable). Derivable per-statement from ``M_SQL_PLAN_CACHE.OBJECT_NAMES``. Full list in MCP-side sink.
   * - ``TOOL_ARGS_JSON``
     - Pack ``args=`` (urlsafe b64 of redacted JSON), possibly truncated with ``argstrunc=1`` flag. Full copy in MCP-side sink.
   * - ``RESPONSE_SIZE``
     - **MCP-side only.** HANA sees rowset sizes per SQL; the tool-level response size after aggregation is an MCP-layer measurement.
   * - ``STATUS``
     - **MCP-side only.** A tool can return ``error`` without any SQL failing, so HANA cannot infer tool-level status.
   * - ``DURATION_MS``
     - **MCP-side only.** Wall-clock over the whole tool call, potentially covering multiple SQL statements.
   * - ``INVOCATION_ID``
     - Pack ``inv=`` — the join key against the MCP-side sink
   * - ``HANA_CORRELATION_ID``
     - Pack ``corr=``
   * - ``OCCURRED_AT``
     - ``M_SQL_PLAN_CACHE.LAST_EXECUTION_TIMESTAMP`` / ``AUDIT_LOG.TIMESTAMP``

The three MCP-layer-only fields (``RESPONSE_SIZE``, ``STATUS``,
``DURATION_MS``) are the honest gap in the HANA-only story. If they matter
for compliance, use the pack's ``INVOCATION_ID`` as a join key against the
MCP-side sink documented below.


The MCP-side sink (JSONL)
-------------------------

Enable it on the toolkit:

.. code-block:: python

    from hana_ai.tools.toolkit import HANAMLToolkit

    toolkit = HANAMLToolkit(
        connection_context=cc,
        audit_enabled=True,
        audit_log_path="logs/mcp-audit.jsonl",   # or MCP_AUDIT_LOG_PATH env var
        audit_service_name="hana-ai-mcp-service",
        audit_environment="prod",
    )
    toolkit.launch_mcp_server(transport="http", host="127.0.0.1", port=8001)

Each tool call appends one JSON line with the full audit event: the Group A
identity fields snapshotted at handshake time, plus the Group B payload
fields (tool name, args, response size, status, duration, invocation ids)
for that specific invocation. Session teardown does not lose events — they
are already on disk.

Reading it back:

.. code-block:: python

    from hana_ai.tools.hana_ml_tools.utility import fetch_mcp_audit_rows

    df = fetch_mcp_audit_rows("logs/mcp-audit.jsonl", session_id="<id>")
    df[[
        "OCCURRED_AT", "TOOL_NAME", "RESPONSE_SIZE", "DURATION_MS", "STATUS",
        "HANA_APPLICATION_USER", "CLIENT_DECLARED_AGENT_NAME",
        "INVOCATION_ID", "APPLICATIONSOURCE_PACK",
    ]]

The DataFrame now includes an ``APPLICATIONSOURCE_PACK`` column showing
exactly the pack string that would have landed on the HANA side for each
event, so the JSONL file is self-describing for the "what did HANA see?"
question without a second connection.


Inspecting a live MCP connection's pack
---------------------------------------

For interactive debugging you can decode the pack from the currently active
HANA connection:

.. code-block:: python

    from hana_ai.tools.hana_ml_tools.utility import fetch_hana_appsource_pack

    fetch_hana_appsource_pack(cc.connection)
    # -> {
    #   'mcp': 'hana-ai/1.1.26072000',
    #   'sess': 'd71dfba1752941aab632db0f6c2adb01',
    #   'agent': 'probe-agent',
    #   'model': 'claude-opus-4-8',
    #   'cli': 'appsource-probe-535c0807',
    #   'mcp_ip': '127.0.0.1',
    #   'tool': 'list_models',
    #   'inv': 'inv-b98880b44944408286382131b7e2f92c',
    #   'corr': 'hana-corr-d4ba732436c8422085147bf45e81b8ae',
    #   'args': {...}  # only when the args segment was not truncated
    # }


Long-term retention recipes
---------------------------

**HANA-only auditor, no MCP-host access:**
Enable an audit policy on the target schemas (see the ``CREATE AUDIT
POLICY`` snippet above). Every audited SQL then carries the pack in
``AUDIT_LOG`` — permanent, queryable with the ``SUBSTR_REGEXPR`` template.

**When ``RESPONSE_SIZE`` / ``STATUS`` / ``DURATION_MS`` are required on the
HANA side:**
Run a small batch-ingest job on the MCP host that ships
``fetch_mcp_audit_rows`` output into a HANA table, e.g. ``MCP_AUDIT_EVENTS``.
Recommended DDL:

.. code-block:: sql

    CREATE COLUMN TABLE MCP_AUDIT_EVENTS (
      INVOCATION_ID              NVARCHAR(64) NOT NULL PRIMARY KEY,
      OCCURRED_AT                TIMESTAMP    NOT NULL,
      MCP_SESSION_ID             NVARCHAR(128),
      CLIENT_IP                  NVARCHAR(64),
      CLIENT_DECLARED_NAME       NVARCHAR(256),
      CLIENT_DECLARED_AGENT_NAME NVARCHAR(256),
      CLIENT_DECLARED_MODEL_NAME NVARCHAR(256),
      HANA_DB_USER               NVARCHAR(256),
      HANA_APPLICATION_USER      NVARCHAR(256),
      TOOL_NAME                  NVARCHAR(256),
      TARGET_TABLES              NCLOB,       -- JSON array
      TOOL_ARGS_JSON             NCLOB,
      RESPONSE_SIZE              BIGINT,
      STATUS                     NVARCHAR(32),
      DURATION_MS                INTEGER,
      HANA_CORRELATION_ID        NVARCHAR(64),
      APPLICATIONSOURCE_PACK     NVARCHAR(256)
    );

    CREATE INDEX IX_MCP_AUDIT_SESSION_TIME
      ON MCP_AUDIT_EVENTS (MCP_SESSION_ID, OCCURRED_AT DESC);

Ingest is one call:

.. code-block:: python

    from hana_ml.dataframe import create_dataframe_from_pandas
    from hana_ai.tools.hana_ml_tools.utility import fetch_mcp_audit_rows

    df = fetch_mcp_audit_rows("logs/mcp-audit.jsonl", session_id=None)
    create_dataframe_from_pandas(cc, df, "MCP_AUDIT_EVENTS", append=True,
                                 primary_key="INVOCATION_ID")

Once loaded, an auditor joins HANA-side plan cache / audit-log records to
this table on ``INVOCATION_ID`` = pack ``inv=`` and has every field.
