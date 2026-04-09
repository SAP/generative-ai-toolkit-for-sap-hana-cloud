ContextAgent
===============================================

The file-based ContextAgent is the recommended conversational agent for new forecasting and dataset-preparation workflows. It stores memory in Markdown files, retrieves compact notes into context, and activates focused workflow skills based on the current request.

Supported workflow skills
-------------------------

- ``data_ingestion_and_dataset_preparation`` for CSV import and time-ordered train, test, and validation table creation
- ``timeseries_data_profiling`` for dataset reports and statistical checks before model selection
- ``timeseries_forecasting`` for single-series train, predict, score, and plot workflows
- ``prediction_result_analysis`` for predicted-versus-actual comparison and quality analysis
- ``outlier_detection_and_repair_prep`` for anomaly inspection before model training
- ``massive_forecasting`` for grouped forecasting across many related series
- ``model_lifecycle_and_artifacts`` for listing, deleting, and packaging saved models
- ``hana_dataframe_fallback`` for SQL and restricted Python fallback transformations

Command-style controls
----------------------

- Memory commands: ``!clear_notes``, ``!clear_session``, ``!reset_memory``, ``!clear_notes_file``, ``!clear_todo``, ``!clear_decisions``, ``!clear_context``, ``!clear_chat``, ``!clear_summary``
- Skill commands: ``!list_skills``, ``!active_skills``, ``!skills_on``, ``!skills_off``, ``!enable_skill <skill_name>``, ``!disable_skill <skill_name>``

These command-style controls are handled through ``chat()`` and let you reset persisted Markdown state or override the active skill set without changing the configured tool catalog.

.. currentmodule:: hana_ai.agents.context_agent

.. autoclass:: ContextAgent
   :members:
   :inherited-members: BaseTool, BaseModel
   :no-undoc-members:
   :exclude-members: _global_mcp_servers, launch_mcp_server, mcp_servers, stop_all_mcp_servers, stop_mcp_server, from_orm, get_prompts, parse_file, parse_obj, parse_raw, schema, schema_json, to_json_not_implemented, validate

.. raw:: html

    <div class="clearer"></div>