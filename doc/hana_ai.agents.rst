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

.. _context_agent-label:

context_agent
-------------

The file-based ContextAgent is the recommended conversational agent for new workflows. It combines Markdown-backed memory, tool calling, runtime skill routing, and command-style memory or skill controls.

.. autosummary::
   :toctree: agents/
   :template: class.rst

   context_agent.ContextAgent
