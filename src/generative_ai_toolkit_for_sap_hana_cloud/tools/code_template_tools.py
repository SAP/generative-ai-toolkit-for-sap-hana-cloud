"""
Code Template tools.

This module contains tools for generating code templates.

The following class are available:

    * :class `GetCodeTemplateFromVectorDB`
"""
# pylint: disable=unused-argument

from typing import Optional, Type, Union
from pydantic import BaseModel
from langchain.tools import BaseTool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from generative_ai_toolkit_for_sap_hana_cloud.vectorstore.corrective_retriever import CorrectiveRetriever
from generative_ai_toolkit_for_sap_hana_cloud.vectorstore.hana_vector_engine import HANAMLinVectorEngine

class GetCodeTemplateFromVectorDB(BaseTool):
    """
    Get code template from vector database.
    """
    name = "CodeTemplatesFromVectorDB"
    description = "useful for when you need to create hana-ml code templates."
    args_schema: Type[BaseModel] = None
    vectordb: Union[HANAMLinVectorEngine, CorrectiveRetriever] = None

    def set_vectordb(self, vectordb):
        """
        Set the vector database.
        
        Parameters
        ----------
        vectordb : Union[HANAMLinVectorEngine, CorrectiveRetriever]
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
