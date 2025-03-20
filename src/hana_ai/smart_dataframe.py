"""
Smart DataFrame.

The following class is available:

    * :class `SmartDataFrame`
"""
from typing import List
from langchain.agents.agent import AgentExecutor
from langchain.llms.base import BaseLLM
from langchain.tools import BaseTool
from hana_ml.dataframe import DataFrame
from hana_ai.agents.hana_dataframe_agent import create_hana_dataframe_agent

class SmartDataFrame(DataFrame):
    """
    Smart DataFrame.

    Parameters
    ----------
    dataframe : DataFrame
        Dataframe.

    Examples
    --------
    >>> from hana_ai.smart_dataframe import SmartDataFrame

    >>> sdf = SmartDataFrame(dataframe=hana_df)
    >>> sdf.configure(tools=[code_tool], llm=llm)
    >>> sdf.ask(question="Show the samples of the dataset", verbose=True)
    >>> new_sdf = sdf.transform(question="Get last two rows", verbose=True)
    """
    llm: BaseLLM
    _dataframe: DataFrame
    tools: List[BaseTool]
    agent: AgentExecutor
    kwargs: dict
    def __init__(self, dataframe: DataFrame):
        super(SmartDataFrame, self).__init__(dataframe.connection_context, dataframe.select_statement)
        self._dataframe = dataframe
        self._is_configured = False

    def configure(self,
                  llm: BaseLLM,
                  tools: List[BaseTool],
                  **kwargs):
        """
        Configure the Smart DataFrame.

        Parameters
        ----------
        llm : BaseLLM
            LLM.
        toolkit : HANAMLToolkit
            Toolkit.
        """
        self.llm = llm
        self.tools = tools
        self.kwargs = kwargs
        self.agent = create_hana_dataframe_agent(llm=llm,
                                                 tools=tools,
                                                 df=self._dataframe,
                                                 **kwargs)
        self._is_configured = True
        suffix ="""
        This is the result of `print(df.head().collect())`:
        {df}

        Begin!
        Question: {input}
        {agent_scratchpad}
        save the transformed result to trans_df and return the SQL string from calling trans_df.select_statement only.

        """
        self.agent_transform = create_hana_dataframe_agent(llm=llm,
                                                           tools=tools,
                                                           df=self._dataframe,
                                                           suffix=suffix,
                                                           **kwargs)

    def ask(self, question: str, verbose: bool = False):
        """
        Ask a question.

        Parameters
        ----------
        question : str
            Question.
        verbose : bool, optional
            Verbose. Default to False.
        """
        if self._is_configured is False:
            raise Exception("The SmartDataFrame is not configured. Please call the configure method first.")
        self.agent.verbose = verbose
        return self.agent.invoke(question)

    @classmethod
    def _construct(cls, dataframe: DataFrame, llm: BaseLLM, tools: List[BaseTool], **kwargs):
        sdf = cls(dataframe)
        sdf.configure(llm, tools, **kwargs)
        return sdf

    def transform(self, question: str, verbose: bool = False, output_key='output'):
        """
        Transform the dataframe.

        Parameters
        ----------
        question : str
            Question.
        verbose : bool, optional
            Verbose. Default to False.
        """
        if self._is_configured is False:
            raise Exception("The SmartDataFrame is not configured. Please call the configure method first.")
        self.agent_transform.verbose = verbose
        select_statement = self.agent_transform.invoke(question)
        if isinstance(select_statement, dict):
            if output_key in select_statement:
                select_statement = select_statement[output_key]
        sdf = self._construct(self._dataframe.connection_context.sql(select_statement), self.llm, self.tools, **self.kwargs)
        return sdf
