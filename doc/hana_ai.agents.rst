hana_ai.agents
==============

hana.ai agents provides focused entry points for dataframe operations, SQL operations, and the recommended ContextAgent workflow for conversational forecasting and dataset preparation.

.. automodule:: hana_ai.agents
   :no-members:
   :no-inherited-members:

.. _hana_dataframe_agent-label:

hana_dataframe_agent
--------------------
.. autosummary::
   :toctree: agents/
   :template: function.rst

   hana_dataframe_agent.create_hana_dataframe_agent

.. _hana_sql_agent-label:

hana_sql_agent
--------------
.. autosummary::
   :toctree: agents/
   :template: function.rst

   hana_sql_agent.create_hana_sql_agent
   hana_sql_agent.create_hana_sql_toolkit

.. _context_agent-label:

context_agent
-------------

The file-based ContextAgent is the recommended conversational agent for new workflows. It combines Markdown-backed memory, tool calling, runtime skill routing, and command-style memory or skill controls.

.. autosummary::
   :toctree: agents/
   :template: class.rst

   context_agent.ContextAgent

Legacy RAG memory
------------------

``HANAMLRAGAgent`` and ``Mem0HANARAGAgent`` are deprecated. Existing applications
must supply non-empty ``tenant_id`` and ``session_id`` from trusted application
identity, not from model output. Reads, cleanup and deletion are scoped to that
tenant/session; existing unscoped memory needs deliberate migration.

For ``HANAMLRAGAgent``, ``vectorstore_path`` is a FAISS base directory; the agent
adds a tenant/session hash. Its default is ``HANA_AI_DATA_DIR/chat_history_vectorstore``
or ``~/.hana_ai/chat_history_vectorstore``. Existing FAISS files require a valid
local integrity manifest. Keep indexes and manifests in host-protected storage;
hashes do not make untrusted pickle safe. The explicit
``HANA_ALLOW_DANGEROUS_FAISS_DESERIALIZATION=true`` override is only for
independently verified trusted indexes during migration, not routine operation.

Mem0 ``!`` administration commands default off; enable
``HANA_MEM0_ALLOW_ADMIN_COMMANDS=true`` only behind application-enforced admin
authentication. This setting does not control ContextAgent's file-memory commands.
