create_hana_sql_toolkit
=============================================

.. currentmodule:: hana_ai.agents.hana_sql_agent

.. autofunction:: create_hana_sql_toolkit

The SQL toolkit uses the same read-only checks and object scope as the SQL agent.
See :doc:`hana_ai.agents.hana_sql_agent.create_hana_sql_agent` for supported SELECT
syntax and deployment requirements. It is separate from ``HANAMLToolkit`` and
does not change the permissions or behavior of model-training/storage tools.
