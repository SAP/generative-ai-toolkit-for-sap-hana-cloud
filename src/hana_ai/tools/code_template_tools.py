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

try:
    from hana_ai.vectorstore.corrective_retriever import CorrectiveRetriever
except ImportError:
    pass
from hana_ai.vectorstore.hana_vector_engine import HANAMLinVectorEngine

class GetCodeTemplateFromVectorDB(BaseTool):
    """
    Get code template from vector database.
    """
    name: str = "CodeTemplatesFromVectorDB"
    description: str = "useful for when you need to create hana-ml code templates."
    args_schema: Type[BaseModel] = None
    vectordb: Union[HANAMLinVectorEngine] = None

    def set_vectordb(self, vectordb):
        """
        Set the vector database.
        
        Parameters
        ----------
        vectordb : Union[HANAMLinVectorEngine]
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
