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

from hana_ai.vectorstore.code_templates import get_code_templates

logger = logging.getLogger(__name__) #pylint: disable=invalid-name

class HANAMLinVectorEngine(object):
    """
    HANA vector engine.

    Parameters
    ----------
    connection_context: ConnectionContext
        Connection context.
    table_name: str
        Table name.
    schema: str
        Schema name.
    model_version: str, optional
        Model version.

        Defaults to 'SAP_NEB.20240715'.
    option: str, optional
        Option to get code templates (python or sql).
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
        if schema is None:
            self.schema = self.connection_context.get_current_schema()
        if not self.connection_context.has_table(table=self.table_name, schema=self.schema):
            self.connection_context.create_table(table=self.table_name,
                                                 schema=self.schema,
                                                 table_structure={"id": "VARCHAR(5000) PRIMARY KEY",
                                                                  "description": "VARCHAR(5000)",
                                                                  "example": "NCLOB",
                                                                  "embeddings": f"REAL_VECTOR GENERATED ALWAYS AS VECTOR_EMBEDDING(\"description\", 'DOCUMENT', '{self.model_version}')"})

    def get_knowledge(self):
        """
        Get knowledge dataframe.
        """
        return self.connection_context.table(table=self.table_name, schema=self.schema)

    def create_knowledge(self, option='python'):
        """
        Create knowledge base.
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
        dataframe.create_dataframe_from_pandas(connection_context=self.connection_context,
                                               pandas_df=pd.DataFrame(knowledge, columns=['id', 'description', 'example']),
                                               table_name=self.table_name,
                                               schema=self.schema,
                                               upsert=True,
                                               table_structure={"id": "VARCHAR(5000) PRIMARY KEY",
                                                                "description": "VARCHAR(5000)",
                                                                "example": "NCLOB",
                                                                "embeddings": f"REAL_VECTOR GENERATED ALWAYS AS VECTOR_EMBEDDING(\"description\", 'DOCUMENT', {self.model_version}"})

    def query(self, input, top_n=1, distance='cosine_similarity'):
        """
        Query.
        """
        schema = self.schema
        if self.columns is None:
            self.columns = self.connection_context.table(table=self.table_name, schema=self.schema).columns
        if self.schema is None:
            schema = self.connection_context.get_current_schema()

        sql = """SELECT TOP {} "{}", {}("{}", TO_REAL_VECTOR(VECTOR_EMBEDDING('{}', 'QUERY', '{}'))) AS "DISTANCE", "{}" "MODEL_TYPE" FROM "{}"."{}" ORDER BY "DISTANCE" DESC""".format(top_n, self.columns[2], distance.upper(), self.columns[3], input, self.model_version, self.columns[0], schema, self.table_name)
        result = self.connection_context.sql(sql).collect()
        self.current_query_rows = result.shape[0]
        if self.current_query_rows < top_n:
            top_n = self.current_query_rows
        self.current_query_distance = result.iloc[top_n-1, 1]
        self.model_type = result.iloc[top_n-1, 2]
        return result.iloc[top_n-1, 0]
