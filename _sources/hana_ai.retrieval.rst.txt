hana_ai.retrieval
=================

The ``hana_ai.retrieval`` package hosts the HANA AI Core retrieval clients used by the MCP-exposed retrieval tools (``object_discovery`` and ``data_retrieval``). ``RetrievalBase`` encapsulates the remote-source, procedure invocation, and progress-monitor plumbing shared by both clients.

.. automodule:: hana_ai.retrieval
   :no-members:
   :no-inherited-members:

.. _retrieval-classes-label:

Clients
-------
.. autosummary::
   :toctree: retrieval/
   :template: class.rst

   object_discovery.ObjectDiscovery
   data_retrieval.DataRetrieval
   retrieval_base.RetrievalBase
