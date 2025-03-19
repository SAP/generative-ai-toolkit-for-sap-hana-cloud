"""
SQL Agent for working with hana-ml objects.

The following function is available:

    * :func `create_hana_sql_agent`
"""
# pylint: disable=no-name-in-module
# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)

from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain_core.prompts import BasePromptTemplate
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.tools import BaseTool

from hana_ml.dataframe import ConnectionContext

class _sql_toolkit(object):
    def __init__(self, llm, db, tools=None):
        self.tools = tools
        self.llm = llm
        self.db = db

    @property
    def dialect(self) -> str:
        """Return string representation of SQL dialect to use."""
        return self.db.dialect

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        if self.tools is None:
            return SQLDatabaseToolkit(llm=self.llm, db=self.db).get_tools()
        return self.tools + SQLDatabaseToolkit(llm=self.llm, db=self.db).get_tools()

    def get_context(self) -> dict:
        """Return db context that you may want in agent prompt."""
        return self.db.get_context()

def create_hana_sql_agent(
    llm: any,
    connection_context: ConnectionContext,
    tools: BaseTool = None,
    agent_type: Optional[
        Union[AgentType, Literal["openai-tools", "tool-calling"]]
    ] = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    format_instructions: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    top_k: int = 10,
    max_iterations: Optional[int] = 15,
    max_execution_time: Optional[float] = None,
    early_stopping_method: str = "force",
    verbose: bool = False,
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    extra_tools: Sequence[BaseTool] = (),
    *,
    prompt: Optional[BasePromptTemplate] = None,
    **kwargs: Any,
):
    """Create a HANA SQL agent.

    Parameters
    ----------
    llm: any
        The language model to use.
    tools: BaseTool
        The tools to use.
    agent_type: Union[AgentType, Literal["openai-tools", "tool-calling"]], optional
        The type of agent to create.
    callback_manager: BaseCallbackManager, optional
        The callback manager to use.
    prefix: str, optional
        The prefix to use.
    suffix: str, optional
        The suffix to use.
    format_instructions: str, optional
        The format instructions to use.
    input_variables: List[str], optional
        The input variables to use.
    top_k: int
        The top k to use.
    max_iterations: int, optional
        The max iterations to use.
    max_execution_time: float, optional
        The max execution time to use.
    early_stopping_method: str
        The early stopping method to use.
    verbose: bool
        The verbose to use.
    agent_executor_kwargs: Dict[str, Any], optional
        The agent executor kwargs to use.
    extra_tools: Sequence[BaseTool]
        The extra tools to use.
    db: SQLDatabase, optional
        The database to use.
    connection_context: ConnectionContext
        The connection context to use.
    prompt: BasePromptTemplate, optional
        The prompt to use.
    kwargs: Any
        The kwargs to use.

    examples
    --------
    Assume cc is a connection to a SAP HANA instance:

    >>> from hana_ai.agents.hana_sql_agent import
    >>> from hana_ai.tools.code_template_tools import GetCodeTemplateFromVectorDB
    >>> from hana_ai.vectorstore.hana_vector_engine import HANAMLinVectorEngine

    >>> hana_vec = HANAMLinVectorEngine(connection_context=cc, "hana_vec_hana_ml_sql_knowledge")
    >>> hana_vec.create_knowledge(option='sql')
    >>> code_tool = GetCodeTemplateFromVectorDB()
    >>> code_tool.set_vectordb(hana_vec)
    >>> agent_executor = create_hana_sql_agent(llm=llm, connection_context=cc, tools=[code_tool], verbose=True)
    >>> agent_executor.invoke("show me the min and max value of sepalwidthcm in the table iris_data_full_tbl?")
    """
    engine = connection_context.to_sqlalchemy()
    db = SQLDatabase(engine)
    toolkit = _sql_toolkit(llm=llm, db=db, tools=tools)
    return create_sql_agent(llm=llm,
                            toolkit=toolkit,
                            agent_type=agent_type,
                            callback_manager=callback_manager,
                            prefix=prefix,
                            suffix=suffix,
                            format_instructions=format_instructions,
                            input_variables=input_variables,
                            top_k=top_k,
                            max_iterations=max_iterations,
                            max_execution_time=max_execution_time,
                            early_stopping_method=early_stopping_method,
                            verbose=verbose,
                            agent_executor_kwargs=agent_executor_kwargs,
                            extra_tools=extra_tools,
                            db=None,
                            prompt=prompt,
                            **kwargs)
