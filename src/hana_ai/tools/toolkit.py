"""
Toolkit for interacting with hana-ml.

The following class is available:

    * :class `HANAMLToolkit`
"""
from typing import List, Optional
from hana_ml import ConnectionContext
from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools import BaseTool

from hana_ai.tools.code_template_tools import GetCodeTemplateFromVectorDB
from hana_ai.tools.hana_ml_tools.fetch_tools import FetchDataTool
from hana_ai.tools.hana_ml_tools.model_storage_tools import ListModels
from hana_ai.vectorstore.hana_vector_engine import HANAMLinVectorEngine
from hana_ai.tools.hana_ml_tools.additive_model_forecast_tools import AdditiveModelForecastFitAndSave, AdditiveModelForecastLoadModelAndPredict
from hana_ai.tools.hana_ml_tools.cap_artifacts_tools import CAPArtifactsTool
from hana_ai.tools.hana_ml_tools.intermittent_forecast_tools import IntermittentForecast
from hana_ai.tools.hana_ml_tools.ts_visualizer_tools import ForecastLinePlot, TimeSeriesDatasetReport
from hana_ai.tools.hana_ml_tools.automatic_timeseries_tools import AutomaticTimeSeriesFitAndSave, AutomaticTimeseriesLoadModelAndPredict, AutomaticTimeseriesLoadModelAndScore
from hana_ai.tools.hana_ml_tools.ts_check_tools import TimeSeriesCheck
from hana_ai.tools.hana_ml_tools.ts_outlier_detection_tools import TSOutlierDetection
from hana_ai.tools.hana_ml_tools.ts_accuracy_measure_tools import AccuracyMeasure

class HANAMLToolkit(BaseToolkit):
    """
    Toolkit for interacting with HANA SQL.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.
    used_tools : list, optional
        List of tools to use. If None or 'all', all tools are used. Default to None.

    Examples
    --------
    Assume cc is a connection to a SAP HANA instance:

    >>> from hana_ai.tools.toolkit import HANAMLToolkit
    >>> from hana_ai.agents.hanaml_agent_with_memory import HANAMLAgentWithMemory

    >>> tools = HANAMLToolkit(connection_context=cc, used_tools='all').get_tools()
    >>> chatbot = HANAMLAgentWithMemory(llm=llm, toos=tools, session_id='hana_ai_test', n_messages=10)
    """
    vectordb: Optional[HANAMLinVectorEngine] = None
    connection_context: ConnectionContext = None
    used_tools: Optional[list] = None
    default_tools: List[BaseTool] = None

    def __init__(self, connection_context, used_tools=None, return_direct=None):
        super().__init__(connection_context=connection_context)
        self.default_tools = [
            AccuracyMeasure(connection_context=self.connection_context),
            AdditiveModelForecastFitAndSave(connection_context=self.connection_context),
            AdditiveModelForecastLoadModelAndPredict(connection_context=self.connection_context),
            AutomaticTimeSeriesFitAndSave(connection_context=self.connection_context),
            AutomaticTimeseriesLoadModelAndPredict(connection_context=self.connection_context),
            AutomaticTimeseriesLoadModelAndScore(connection_context=self.connection_context),
            CAPArtifactsTool(connection_context=self.connection_context),
            FetchDataTool(connection_context=self.connection_context),
            ForecastLinePlot(connection_context=self.connection_context),
            IntermittentForecast(connection_context=self.connection_context),
            ListModels(connection_context=self.connection_context),
            TimeSeriesDatasetReport(connection_context=self.connection_context),
            TimeSeriesCheck(connection_context=self.connection_context),
            TSOutlierDetection(connection_context=self.connection_context)
        ]
        if isinstance(return_direct, dict):
            for tool in self.default_tools:
                if tool.name in return_direct:
                    tool.return_direct = return_direct[tool.name]
        if isinstance(return_direct, bool):
            for tool in self.default_tools:
                tool.return_direct = return_direct
        if used_tools is None or used_tools == "all":
            self.used_tools = self.default_tools
        else:
            if isinstance(used_tools, str):
                used_tools = [used_tools]
            self.used_tools = []
            for tool in self.default_tools:
                if tool.name in used_tools:
                    self.used_tools.append(tool)

    def add_custom_tool(self, tool: BaseTool):
        """
        Add a custom tool to the toolkit.

        Parameters
        ----------
        tool : BaseTool
            Custom tool to add.

            .. note::

                The tool must be a subclass of BaseTool. Please follow the guide to create the custom tools https://python.langchain.com/docs/how_to/custom_tools/.
        """
        self.used_tools.append(tool)

    def delete_tool(self, tool_name: str):
        """
        Delete a tool from the toolkit.

        Parameters
        ----------
        tool_name : str
            Name of the tool to delete.
        """
        for tool in self.used_tools:
            if tool.name == tool_name:
                self.used_tools.remove(tool)
                break

    def set_bas(self, bas=True):
        """
        Set the BAS mode for all tools in the toolkit.
        """
        for tool in self.used_tools:
            if hasattr(tool, "bas"):
                tool.bas = bas
        return self

    def set_vectordb(self, vectordb):
        """
        Set the vector database.

        Parameters
        ----------
        vectordb : HANAMLinVectorEngine
            Vector database.
        """
        self.vectordb = vectordb

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        if self.vectordb is not None:
            get_code = GetCodeTemplateFromVectorDB()
            get_code.set_vectordb(self.vectordb)
            return self.used_tools + [get_code]
        return self.used_tools
