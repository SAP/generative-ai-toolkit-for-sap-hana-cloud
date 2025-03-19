"""Agent for working with hana-ml objects.

The following function is available:

    * :func `create_hana_dataframe_agent`
"""
from typing import Any, Dict, List, Optional

from langchain.agents.agent import AgentExecutor
from langchain.tools import BaseTool
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
try:
    from langchain.tools.python.tool import PythonAstREPLTool
except:
    from langchain_experimental.tools.python.tool import PythonAstREPLTool
from hana_ai.agents.hana_dataframe_prompt import PREFIX, SUFFIX

def _validate_hana_df(df: Any) -> bool:
    try:
        from hana_ml.dataframe import DataFrame as HANADataFrame

        return isinstance(df, HANADataFrame)
    except ImportError:
        return False

def create_hana_dataframe_agent(
    llm: BaseLLM,
    df: Any,
    tools: List[BaseTool] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = PREFIX,
    suffix: str = SUFFIX,
    input_variables: Optional[List[str]] = None,
    verbose: bool = False,
    return_intermediate_steps: bool = False,
    max_iterations: Optional[int] = 15,
    max_execution_time: Optional[float] = None,
    early_stopping_method: str = "force",
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Dict[str, Any],
) -> AgentExecutor:
    """
    Construct a hana-ml agent from an LLM and dataframe.

    Parameters
    ----------
    llm : BaseLLM
        The LLM to use.
    df : DataFrame
        The HANA dataframe to use. It could be None.
    tools : BaseTool, optional
        The tools to use. Default to None.
    callback_manager : BaseCallbackManager, optional
        The callback manager to use. Default to None.
    prefix : str, optional
        The prefix to use.
    suffix : str, optional
        The suffix to use.
    input_variables : List[str], optional
        The input variables to use. Default to None.
    verbose : bool, optional
        Whether to be verbose. Default to False.
    return_intermediate_steps : bool, optional
        Whether to return intermediate steps. Default to False.
    max_iterations : int, optional
        The maximum number of iterations to use. Default to 15.
    max_execution_time : float, optional
        The maximum execution time to use. Default to None.
    early_stopping_method : str, optional
        The early stopping method to use. Default to "force".
    agent_executor_kwargs : Dict[str, Any], optional
        The agent executor kwargs to use. Default to None.

    Examples
    --------

    Assume cc is a connection to a SAP HANA instance:

    >>> from hana_ai.tools.code_template_tools import GetCodeTemplateFromVectorDB
    >>> from hana_ai.vectorstore.hana_vector_engine import HANAMLinVectorEngine
    >>> from hana_ai.agents.hana_dataframe_agent import create_hana_dataframe_agent

    >>> hana_df = cc.table("MY_DATA")
    >>> hana_vec = HANAMLinVectorEngine(connection_context=cc, table_name="hana_vec_hana_ml_python_knowledge")
    >>> hana_vec.create_knowledge()
    >>> code_tool = GetCodeTemplateFromVectorDB()
    >>> code_tool.set_vectordb(vectordb=hana_vec)
    >>> agent = create_hana_dataframe_agent(llm=llm, tools=[code_tool], df=hana_df, verbose=True, handle_parsing_errors=True)
    >>> agent.invoke("Create a dataset report for this dataframe.")
    """

    if not _validate_hana_df(df):
        raise ImportError("hana-ml is not installed. run `pip install hana-ml`.")

    #suppress all the warnings
    import warnings
    warnings.filterwarnings("ignore")

    if input_variables is None:
        input_variables = ["df", "input", "agent_scratchpad"]
    if tools is None:
        tools = [PythonAstREPLTool(locals={"df": df})]
        prefix = "You are working with a HANA dataframe in Python that is similar to Spark dataframe. The name of the dataframe is `df`. `connection_context` is `df`'s attribute. To handle connection or to use dataframe functions, you should use python_repl_ast tool. You should use the tools below to answer the question posed of you. :"
    else:
        tools = tools + [PythonAstREPLTool(locals={"df": df})]
    prompt = ZeroShotAgent.create_prompt(
        tools, prefix=prefix, suffix=suffix, input_variables=input_variables
    )
    partial_prompt = prompt.partial(df=str(df.head(1).collect()))
    llm_chain = LLMChain(
        llm=llm,
        prompt=partial_prompt,
        callback_manager=callback_manager,
    )
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(
        llm_chain=llm_chain,
        allowed_tools=tool_names,
        callback_manager=callback_manager,
        **kwargs,
    )
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        verbose=verbose,
        return_intermediate_steps=return_intermediate_steps,
        max_iterations=max_iterations,
        max_execution_time=max_execution_time,
        early_stopping_method=early_stopping_method,
        **(agent_executor_kwargs or {}),
    )
