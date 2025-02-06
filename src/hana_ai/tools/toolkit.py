"""
Toolkit for interacting with hana-ml.

The following class is available:

    * :class `HANAMLToolkit`
"""
from typing import List, Union
from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools import BaseTool

from hana_ai.tools.code_template_tools import GetCodeTemplateFromVectorDB
from hana_ai.vectorstore.corrective_retriever import CorrectiveRetriever
from hana_ai.vectorstore.hana_vector_engine import HANAMLinVectorEngine

class HANAMLToolkit(BaseToolkit):
    """Toolkit for interacting with HANA SQL."""
    vectordb: Union[HANAMLinVectorEngine, CorrectiveRetriever] = None

    def set_vectordb(self, vectordb):
        """
        Set the vector database.
        
        Parameters
        ----------
        vectordb : Union[HANAMLinVectorEngine, CorrectiveRetriever]
            Vector database."""
        self.vectordb = vectordb

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""

        get_code = GetCodeTemplateFromVectorDB()
        get_code.set_vectordb(self.vectordb)

        return [get_code]
