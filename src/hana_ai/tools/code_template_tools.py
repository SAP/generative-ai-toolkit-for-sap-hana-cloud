"""
Code Template tools.

This module contains tools for generating code templates.

The following class are available:

    * :class `GetCodeTemplateFromVectorDB`
"""
# pylint: disable=unused-argument

from typing import Optional, Type
from pydantic import BaseModel
from langchain.tools import BaseTool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from hana_ai.vectorstore.hana_vector_engine import HANAMLinVectorEngine

class GetCodeTemplateFromVectorDB(BaseTool):
    """
    Get code template from vector database.

    Examples
    --------
    Assume cc is a connection to a SAP HANA instance:

    >>> from hana_ai.tools.code_template_tools import GetCodeTemplateFromVectorDB
    >>> from hana_ai.vectorstore.hana_vector_engine import HANAMLinVectorEngine
    >>> from hana_ai.agents.hana_dataframe_agent import create_hana_dataframe_agent

    >>> hana_vec = HANAMLinVectorEngine(connection_context=cc, table_name="hana_vec_hana_ml_python_knowledge")
    >>> hana_vec.create_knowledge()
    >>> code_tool = GetCodeTemplateFromVectorDB()
    >>> code_tool.set_vectordb(vectordb=hana_vec)
    >>> agent = create_hana_dataframe_agent(llm=llm, tools=[code_tool], df=hana_df, verbose=True, handle_parsing_errors=True)
    >>> agent.invoke("Create a dataset report for this dataframe.")
    """
    name: str = "CodeTemplatesFromVectorDB"
    description: str = "useful for when you need to create hana-ml code templates."
    args_schema: Type[BaseModel] = None
    vectordb: HANAMLinVectorEngine = None

    def set_vectordb(self, vectordb):
        """
        Set the vector database.

        Parameters
        ----------
        vectordb : HANAMLinVectorEngine
            Vector database.
        """
        self.vectordb = vectordb

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        if self.vectordb is None:
            raise ValueError("No vector database set.")
        model = self.vectordb
        result = None
        result = model.query(query)
        return result

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Does not support async")
