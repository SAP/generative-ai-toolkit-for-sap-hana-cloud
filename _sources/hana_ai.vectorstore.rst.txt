hana_ai.vectorstore
===================

HANA ML Vector Store is a service that provides a set of APIs to store and retrieve vectors. The service is built on top of the HANA Vector Engine using HANA Machine Learning knowledge base, which is a high-performance vector engine that can store and retrieve vectors in real-time.

.. automodule:: hana_ai.vectorstore
   :no-members:
   :no-inherited-members:

.. _code_templates-label:

code_templates
--------------
.. autosummary::
   :toctree: vectorstore/
   :template: function.rst

   code_templates.get_code_templates

.. _embedding_service-label:

embedding_service
-----------------
.. autosummary::
   :toctree: vectorstore/
   :template: class.rst

   embedding_service.PALModelEmbeddings
   embedding_service.HANAVectorEmbeddings

.. _hana_vector_engine-label:

hana_vector_engine
------------------
.. autosummary::
   :toctree: vectorstore/
   :template: class.rst

   hana_vector_engine.HANAMLinVectorEngine

.. _union_vector_stores-label:

union_vector_stores
-------------------
.. autosummary::
   :toctree: vectorstore/
   :template: class.rst

   union_vector_stores.UnionVectorStores
