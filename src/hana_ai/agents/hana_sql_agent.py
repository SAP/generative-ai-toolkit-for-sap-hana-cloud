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
from hana_ai.tools.toolkit import HANAMLToolkit

class _sql_toolkit(object):
    def __init__(self, llm, db, hanaml_toolkit=None):
        self.hanaml_toolkit = hanaml_toolkit
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
        if self.hanaml_toolkit is None:
            return SQLDatabaseToolkit(llm=self.llm, db=self.db).get_tools()
        return self.hanaml_toolkit.get_tools() + SQLDatabaseToolkit(llm=self.llm, db=self.db).get_tools()

    def get_context(self) -> dict:
        """Return db context that you may want in agent prompt."""
        return self.db.get_context()

def create_hana_sql_agent(
    llm: any,
    connection_context: ConnectionContext,
    toolkit: HANAMLToolkit = None,
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
    toolkit: HANAMLToolkit
        The toolkit to use.
    agent_type: Optional[Union[AgentType, Literal["openai-tools", "tool-calling"]]]
        The type of agent to create.
    callback_manager: Optional[BaseCallbackManager]
        The callback manager to use.
    prefix: Optional[str]
        The prefix to use.
    suffix: Optional[str]
        The suffix to use.
    format_instructions: Optional[str]
        The format instructions to use.
    input_variables: Optional[List[str]]
        The input variables to use.
    top_k: int
        The top k to use.
    max_iterations: Optional[int]
        The max iterations to use.
    max_execution_time: Optional[float]
        The max execution time to use.
    early_stopping_method: str
        The early stopping method to use.
    verbose: bool
        The verbose to use.
    agent_executor_kwargs: Optional[Dict[str, Any]]
        The agent executor kwargs to use.
    extra_tools: Sequence[BaseTool]
        The extra tools to use.
    db: Optional[SQLDatabase]
        The database to use.
    connection_context: ConnectionContext
        The connection context to use.
    prompt: Optional[BasePromptTemplate]
        The prompt to use.
    kwargs: Any
        The kwargs to use.
    """
    engine = connection_context.to_sqlalchemy()
    db = SQLDatabase(engine)
    toolkit = _sql_toolkit(llm=llm, db=db, hanaml_toolkit=toolkit)
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
