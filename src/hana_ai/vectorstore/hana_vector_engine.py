"""
HANA vector store to save and get embeddings for hana-ml.

The following class is available:

    * :class `HANAMLinVectorEngine`
"""

#pylint: disable=no-name-in-module
#pylint: disable=redefined-builtin

import logging
import pandas as pd
from hana_ml import ConnectionContext, dataframe

from hana_ai.sql_guard import ensure_hana_ml_single_statement
from hana_ai.vectorstore.code_templates import get_code_templates

logger = logging.getLogger(__name__) #pylint: disable=invalid-name


def _quote_identifier(name):
    """Quote one HANA identifier component and reject ambiguous embedded quotes."""
    value = str(name)
    if not value or '"' in value or "\x00" in value or ";" in value:
        raise ValueError(f"invalid HANA identifier: {value!r}")
    return '"' + value + '"'


def _sql_string_literal(value):
    """Encode a value as one HANA string literal when this API cannot bind parameters."""
    value = str(value)
    if "\x00" in value or ";" in value:
        raise ValueError("NUL bytes and semicolons are not allowed in HANA string values")
    return "'" + value.replace("'", "''") + "'"


class HANAMLinVectorEngine(object):
    """
    HANA vector engine.

    Parameters
    ----------
    connection_context: ConnectionContext
        Connection context.
    table_name: str
        Table name.
    schema: str, optional
        Schema name. Default to None.
    model_version: str, optional
        Model version. Default to 'SAP_NEB.20240715'.
    """
    connection_context: ConnectionContext = None
    table_name: str = None
    schema: str = None
    vector_length: int = None
    columns: list = None
    def __init__(self, connection_context, table_name, schema=None, model_version='SAP_NEB.20240715'):
        self.connection_context = connection_context
        self.table_name = table_name
        self.schema = schema
        self.model_version = model_version
        self.current_query_distance = None
        self.current_query_rows = None
        self._validate_configuration()
        if schema is None:
            self.schema = self.connection_context.get_current_schema()
            self._validate_configuration()
        if not self.connection_context.has_table(table=self.table_name, schema=self.schema):
            self.connection_context.create_table(table=self.table_name,
                                                 schema=self.schema,
                                                 table_structure={"id": "VARCHAR(5000) PRIMARY KEY",
                                                                  "description": "VARCHAR(5000)",
                                                                  "example": "NCLOB",
                                                                  "embeddings": f"REAL_VECTOR GENERATED ALWAYS AS VECTOR_EMBEDDING(\"description\", 'DOCUMENT', {_sql_string_literal(self.model_version)})"})

    def _validate_configuration(self):
        """Validate before hana-ml metadata/DDL calls, not only the final SELECT."""
        names = [self.table_name] if self.schema is None else [self.table_name, self.schema]
        for name in names:
            if not isinstance(name, str):
                raise ValueError("vector table/schema identifiers must be strings")
            _quote_identifier(name)
            # hana-ml has_table() interpolates these names into SQL string literals.
            # Identifier quoting alone cannot protect that separate metadata sink.
            if "'" in str(name):
                raise ValueError("single quotes are not allowed in vector table/schema identifiers")
        _sql_string_literal(self.model_version)

    def get_knowledge(self):
        """
        Get knowledge dataframe.
        """
        self._validate_configuration()
        return self.connection_context.table(table=self.table_name, schema=self.schema)

    def create_knowledge(self, option='python'):
        """
        Create knowledge base.

        Parameters
        ----------
        option: {'python', 'sql'}, optional
            The option of language.  Default to 'python'.
        """
        self.upsert_knowledge(get_code_templates(option=option))

    def upsert_knowledge(self,
                         knowledge):
        """
        Upsert knowledge.

        Parameters
        ----------
        knowledge: dict
            Knowledge data. {'id': '1', 'description': 'description', 'example': 'example'}
        """
        self._validate_configuration()
        dataframe.create_dataframe_from_pandas(connection_context=self.connection_context,
                                               pandas_df=pd.DataFrame(knowledge, columns=['id', 'description', 'example']),
                                               table_name=self.table_name,
                                               schema=self.schema,
                                               upsert=True,
                                               table_structure={"id": "VARCHAR(5000) PRIMARY KEY",
                                                                "description": "VARCHAR(5000)",
                                                                "example": "NCLOB",
                                                                "embeddings": f"REAL_VECTOR GENERATED ALWAYS AS VECTOR_EMBEDDING(\"description\", 'DOCUMENT', {_sql_string_literal(self.model_version)})"})

    def query(self, input, top_n=1, distance='cosine_similarity'):
        """
        Query.

        Parameters
        ----------
        input: str
            Input text.
        top_n: int, optional
            Top n. Default to 1.
        distance: str, optional
            Distance. Default to 'cosine_similarity'.
        """
        # Never interpolate untrusted values or identifiers without encoding.
        try:
            top_n = int(top_n)
        except (TypeError, ValueError):
            raise ValueError(f"top_n must be an integer, got {top_n!r}")
        if top_n < 1:
            raise ValueError("top_n must be >= 1")
        if top_n > 1000:
            raise ValueError("top_n must be <= 1000")
        _ALLOWED_DISTANCE = {
            "COSINE_SIMILARITY", "L2DISTANCE", "EUCLIDEAN_DISTANCE",
            "DOT_PRODUCT", "MANHATTAN_DISTANCE", "INNER_PRODUCT",
        }
        distance_fn = str(distance).upper()
        if distance_fn not in _ALLOWED_DISTANCE:
            raise ValueError(f"distance must be one of {sorted(_ALLOWED_DISTANCE)}")

        self._validate_configuration()
        query_literal = _sql_string_literal(input)
        schema = self.schema
        if schema is None:
            schema = self.connection_context.get_current_schema()
        quoted_schema = _quote_identifier(schema)
        if self.columns is None:
            self.columns = self.connection_context.table(table=self.table_name, schema=schema).columns

        sql = (
            "SELECT TOP {top_n} {col2}, {dist}({col3}, "
            "TO_REAL_VECTOR(VECTOR_EMBEDDING({inp}, 'QUERY', {mv}))) AS \"DISTANCE\", "
            "{col0} \"MODEL_TYPE\" FROM {schema}.{table} ORDER BY \"DISTANCE\" DESC"
        ).format(
            top_n=top_n, col2=_quote_identifier(self.columns[2]), dist=distance_fn,
            col3=_quote_identifier(self.columns[3]), inp=query_literal,
            mv=_sql_string_literal(self.model_version), col0=_quote_identifier(self.columns[0]),
            schema=quoted_schema, table=_quote_identifier(self.table_name),
        )
        result = self.connection_context.sql(
            ensure_hana_ml_single_statement(sql)
        ).collect()
        self.current_query_rows = result.shape[0]
        if self.current_query_rows < top_n:
            top_n = self.current_query_rows
        self.current_query_distance = result.iloc[top_n-1, 1]
        self.model_type = result.iloc[top_n-1, 2]
        return result.iloc[top_n-1, 0]
