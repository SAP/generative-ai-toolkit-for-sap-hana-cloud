MCP session auditing
====================

This page answers the practical question: *when a tool call happens on an
MCP server that is running on some host you don't control, what does the
HANA side see, and how does an auditor with only HANA access query it?*

Every field this page describes is populated automatically on every tool
call via ``setclientinfo``. No coordination with the MCP host is required;
an auditor with ``SELECT`` on ``M_SQL_PLAN_CACHE`` — or, with an audit
policy enabled, ``AUDIT_LOG`` — has the full picture.


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
     - Pipe-delimited pack of MCP identity + per-call correlation ids + optional ``resp=<response_size>``
     - ``M_SQL_PLAN_CACHE.APPLICATION_SOURCE``, ``AUDIT_LOG`` (when audit policy is on)

The ``APPLICATIONSOURCE`` pack is the key mechanism: HANA gives us five
setclientinfo slots, but we have ~10 fields to surface, so all the MCP-only
identity + invocation ids ride on one channel in a compact key=value form.

Pack layout::

    mcp=hana-ai/<ver>|sess=<mcp_session_id>|agent=<agent_name>|model=<model_name>
      |cli=<client_name>|mcp_ip=<client_ip>|tool=<tool_name>|inv=<invocation_id>
      |corr=<hana_correlation_id>[|resp=<response_size>]

Real example captured from a live MCP HTTP session::

    mcp=hana-ai/1.1.26072000|sess=d71dfba1752941aab632db0f6c2adb01
    |agent=probe-agent|model=claude-opus-4-8|cli=appsource-probe-535c0807
    |mcp_ip=127.0.0.1|tool=list_models
    |inv=inv-b98880b44944408286382131b7e2f92c
    |corr=hana-corr-d4ba732436c8422085147bf45e81b8ae
    |resp=42

Pack contract:

* **ASCII-only**, **≤ 254 bytes** — HANA silently truncates
  ``APPLICATIONSOURCE`` past 256 wire bytes, so the builder enforces a hard
  254-byte cap and refuses non-ASCII input.
* **Field order is fixed** so simple ``SUBSTR_REGEXPR`` on the HANA side
  works.
* **resp is optional and last** — it is only emitted on the post-completion
  "beacon" write (see below); the started-path pack has no ``resp=``
  segment. When identity plus ``resp=`` would overflow the 254-byte cap,
  ``resp=`` is dropped first (tail-first truncation on a ``|`` boundary).
* **Sanitised** — every field value is passed through a
  ``[A-Za-z0-9._:/@-]`` whitelist so a malicious tool name cannot inject
  fake ``|sess=`` segments.
* **Plan-cache safe** — verified against HANA Cloud 4.00 that
  ``setclientinfo`` does not participate in ``STATEMENT_HASH``, so a fresh
  ``inv=`` per call does not fragment ``M_SQL_PLAN_CACHE``.

Surfacing ``RESPONSE_SIZE`` on the HANA side
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``RESPONSE_SIZE`` is only known *after* the tool body runs, so it cannot
travel in the started-path pack that HANA freezes onto the tool's own
plan-cache rows (``M_SQL_PLAN_CACHE.APPLICATION_SOURCE`` follows
first-execution-wins semantics). Instead, after each successful tool call
the MCP host rewrites ``APPLICATIONSOURCE`` with a beacon pack carrying
``resp=<N>`` and executes one uniquely-tagged synthetic statement::

    SELECT /* mcp-audit-beacon inv=<invocation_id> */ CURRENT_UTCTIMESTAMP FROM DUMMY

The comment embeds the invocation id so each beacon hashes to a distinct
plan-cache entry (no cross-invocation reuse of a cached beacon plan).
:func:`fetch_hana_mcp_audit_view` reads back the plan cache, extracts
``resp=`` per invocation, left-joins it onto every row of the same
``INVOCATION_ID``, and drops the beacon rows themselves (identified by
``STATEMENT_STRING`` containing ``mcp-audit-beacon``). Rows without a
beacon — failed tool calls, or plan cache evicted before the beacon fired
— show ``RESPONSE_SIZE = None``.


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
           SUBSTR_REGEXPR('resp=([0-9]+)' IN APPLICATION_SOURCE GROUP 1) AS RESPONSE_SIZE,
           APPLICATION_USER_NAME,   -- MCP-declared end user
           APPLICATION_NAME,         -- MCP-declared client name
           USER_NAME,                -- HANA-authenticated DB user
           EXECUTION_COUNT,
           LAST_EXECUTION_TIMESTAMP
    FROM   M_SQL_PLAN_CACHE
    WHERE  APPLICATION_SOURCE LIKE 'mcp=hana-ai/%'
    ORDER  BY LAST_EXECUTION_TIMESTAMP DESC;

Or, via the Python convenience wrapper that also fans out ``resp=`` per
invocation and hides the beacon rows:

.. code-block:: python

    from hana_ai.tools.hana_ml_tools.utility import fetch_hana_mcp_audit_view

    audit_df = fetch_hana_mcp_audit_view(
        cc.connection,
        application_name='context-agent-notebook',
    )
    audit_df[[
        'LAST_EXECUTION_TIMESTAMP', 'TOOL_NAME', 'RESPONSE_SIZE',
        'AGENT_NAME', 'MODEL_NAME',
        'MCP_SESSION_ID', 'INVOCATION_ID',
        'HANA_AUTHENTICATED_USER', 'EXECUTION_COUNT',
    ]].head(20)

Limits of this view:

* **First-execution-wins semantics.** HANA writes
  ``M_SQL_PLAN_CACHE.APPLICATION_SOURCE`` at plan-caching time and does
  not update it on subsequent executions. Different tools produce
  different SQL, so each tool's rows carry that tool's own pack; but
  repeated calls to the *same* tool from *different* MCP sessions all
  appear under the first session's pack.
* **Plan cache ages out** — this is a near-term audit view, not long-term
  retention. Once a plan is evicted the row is gone. For permanent
  retention, enable audit policies (next section).
* **``RESPONSE_SIZE`` is best-effort** — populated for successful tool
  calls whose beacon plan-cache row is still resident. Failed tool calls
  do not emit a beacon and evicted beacons drop the value.


Querying HANA-side with an audit policy
---------------------------------------

For long-term retention on HANA Cloud, the DBA enables audit policies on
the relevant schemas. On HANA Cloud, ``global_auditing_state`` is enabled
by default at the tenant level, so no ``ALTER SYSTEM ... CONFIGURATION``
call is required (and the on-prem ``'SYSTEM'`` layer is not exposed to
tenants anyway). Only ``TRAIL TYPE TABLE`` is supported on HANA Cloud:

.. code-block:: sql

    -- One-time DBA setup on HANA Cloud
    CREATE AUDIT POLICY p_mcp_tools
      AUDITING SUCCESSFUL EXECUTE ON SCHEMA "MY_MCP_SCHEMA"
      LEVEL INFO TRAIL TYPE TABLE;   -- TABLE is the only trail type on HANA Cloud
    ALTER AUDIT POLICY p_mcp_tools ENABLE;

.. note::

   ``maximum_statement_string_length`` is a platform-managed setting on
   HANA Cloud and cannot be adjusted by the tenant. If ``STATEMENT_STRING``
   in ``AUDIT_LOG`` is truncated for your workload, fall back to the
   ``M_SQL_PLAN_CACHE`` path (previous section), whose full SQL text is
   not subject to this cap.

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
           SUBSTR_REGEXPR('resp=([0-9]+)' IN CLIENT_APPLICATION_SOURCE GROUP 1) AS RESPONSE_SIZE,
           STATEMENT_STRING
    FROM   AUDIT_LOG
    WHERE  CLIENT_APPLICATION_SOURCE LIKE 'mcp=hana-ai/%'
      AND  TIMESTAMP > ADD_SECONDS(CURRENT_TIMESTAMP, -86400)
    ORDER  BY TIMESTAMP DESC;

.. note::

   The audit-log column that mirrors ``setclientinfo('APPLICATIONSOURCE')``
   is exposed as ``CLIENT_APPLICATION_SOURCE`` on HANA Cloud.


HANA-side field reference
-------------------------

The columns surfaced by :func:`fetch_hana_mcp_audit_view` and their
positions in the HANA views:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - HANA-side location
   * - ``MCP_SESSION_ID``
     - Pack ``sess=`` in ``M_SQL_PLAN_CACHE.APPLICATION_SOURCE`` / ``AUDIT_LOG.CLIENT_APPLICATION_SOURCE``
   * - ``CLIENT_IP``
     - Pack ``mcp_ip=`` (MCP-observed remote IP, distinct from HANA-observed peer)
   * - ``CLIENT_DECLARED_NAME``
     - ``M_CONNECTIONS.APPLICATION_NAME``, pack ``cli=``
   * - ``AGENT_NAME``
     - Pack ``agent=``
   * - ``MODEL_NAME``
     - Pack ``model=``
   * - ``MCP_VERSION``
     - Pack ``mcp=`` (e.g. ``hana-ai/1.1.26072000``)
   * - ``HANA_AUTHENTICATED_USER``
     - ``M_SQL_PLAN_CACHE.USER_NAME`` / ``AUDIT_LOG.USER_NAME`` (HANA-authenticated identity)
   * - ``HANA_SESSION_USER``
     - ``M_SQL_PLAN_CACHE.SESSION_USER_NAME``
   * - ``APPLICATION_NAME``
     - ``M_SQL_PLAN_CACHE.APPLICATION_NAME`` (from ``setclientinfo('APPLICATION')``)
   * - ``TOOL_NAME``
     - Pack ``tool=`` + ``M_CONNECTIONS.APPLICATION_COMPONENT``
   * - ``INVOCATION_ID``
     - Pack ``inv=`` — the join key across plan-cache rows of a single tool call
   * - ``HANA_CORRELATION_ID``
     - Pack ``corr=``
   * - ``RESPONSE_SIZE``
     - Pack ``resp=`` on the post-completion beacon row; :func:`fetch_hana_mcp_audit_view` projects it onto every row of the same ``INVOCATION_ID``.
   * - ``LAST_EXECUTION_TIMESTAMP``
     - ``M_SQL_PLAN_CACHE.LAST_EXECUTION_TIMESTAMP`` / ``AUDIT_LOG.TIMESTAMP``
   * - ``EXECUTION_COUNT``
     - ``M_SQL_PLAN_CACHE.EXECUTION_COUNT``
   * - ``STATEMENT_HASH``
     - ``M_SQL_PLAN_CACHE.STATEMENT_HASH`` — join key onto the tool's underlying SQL
   * - ``APPLICATION_SOURCE``
     - The raw pack string, kept for troubleshooting when a field decodes unexpectedly


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
    #   'response_size': 42,      # only after the beacon has fired
    # }
