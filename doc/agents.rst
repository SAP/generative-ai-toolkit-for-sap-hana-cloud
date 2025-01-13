generative_ai_toolkit_for_sap_hana_cloud.agents
==============

.. automodule:: generative_ai_toolkit_for_sap_hana_cloud.agents
   :no-members:
   :no-inherited-members:

.. _hana_dataframe_agent-label:

HANA Dataframe Agent
--------------------
.. autosummary::
   :toctree: agents/
   :template: function.rst

   hana_dataframe_agent.create_hana_dataframe_agent

.. _hana_sql_agent-label:

HANA SQL Agent
--------------
.. autosummary::
   :toctree: agents/
   :template: function.rst

   hana_sql_agent.create_hana_sql_agent

.. _scenario_agents-label:

Scenario Agents
---------------
.. autosummary::
   :toctree: agents/
   :template: class.rst

   scenario_agents.HANAChatAgent

.. _scenario_utility-label:

Scenario Utility
----------------
.. autosummary::
   :toctree: agents/
   :template: function.rst

   scenario_utility.find_substring_bracketed_by_tag
   scenario_utility.find_all_substrings_bracketed_by_tag
   scenario_utility.get_fields_by_llm
   scenario_utility.execute_code_with_fields
   scenario_utility.llm_invoke_wait_for_ratelimit
   scenario_utility.agent_invoke_wait_for_ratelimit
