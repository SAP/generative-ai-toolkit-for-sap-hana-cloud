create_hana_sql_agent
=================================================================

.. currentmodule:: hana_ai.agents.hana_sql_agent

.. autofunction:: create_hana_sql_agent

Supported SELECT syntax
-----------------------

The SQL agent checks every table reference against the configured ``schema`` and/or
``include_tables``. It supports explicit joins, parenthesized table expressions,
subqueries, scoped CTEs, built-in scalar/aggregate functions, and the following HANA
constructs:

* ``CAST(expression AS type)`` with built-in HANA types, including numeric precision
  and string length parameters, and ``TO_VARCHAR(expression)``.
* ``ANY``, ``SOME`` and ``ALL`` subquery comparisons.
* ``WITH HINT (NO_INLINE, CS_JOIN)`` and hint names with flat literal/name arguments.
* ``table PARTITION (1, 2) AS alias``.
* ``table AS alias TABLESAMPLE SYSTEM (50)`` or ``TABLESAMPLE BERNOULLI (50)``.
* ``CROSS JOIN LATERAL (SELECT ...) AS alias``, including correlated subqueries
  and CTEs inside the lateral query.

CAST operands and all nested queries retain the same object checks as the outer
query. Type/partition/sample parameters accept numeric literals; they cannot hide
function calls or SQL expressions. Hint arguments cannot contain nested SQL.

Use explicit ``JOIN`` syntax. Comma joins, arbitrary scalar/table UDFs, sequence
access and three-part table names remain unsupported and are rejected before query
execution. A read-only HANA account with restricted object permissions remains
necessary; these checks do not replace database authorization.

Semicolons in other hana-ml tools
---------------------------------

The SQL agent executes through SQLAlchemy and preserves semicolons inside string
literals. Tools that construct SQL for ``ConnectionContext.sql()`` or
``execute_sql()`` reject semicolons in values and identifiers because hana-ml splits
the raw text before the database parses SQL quotes. This includes vector query
text/model versions, forecasting group values and split/future-table identifiers.
The rejection is intentional; SQL quoting alone cannot make these inputs safe at
those execution APIs. The read-only SELECT tools accept one trailing statement
terminator and remove it before calling hana-ml.

.. raw:: html

    <div class="clearer"></div>
