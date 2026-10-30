"""
Agent as a tool.

The following function is available:

    * :func `AgentAsATool`
"""
from hana_ai.langchain_compat import Tool

class AgentAsATool(object):
    """
    Agent as a tool.

    Parameters
    ----------
    agent : Agent
        Agent.
    name : str
        Name.
    description : str
        Description.

    Examples
    --------
    Assume cc is a connection to a SAP HANA instance:

    >>> from hana_ai.agents.hana_sql_agent import create_hana_sql_agent
    >>> from hana_ai.tools.agent_as_a_tool import AgentAsATool
    >>> from hana_ai.agents.context_agent import ContextAgent

    >>> sql_agent = create_hana_sql_agent(llm=llm, connection_context=cc, include_tables=["IRIS_DATA_FULL_TBL"], verbose=True)
    >>> sql_tool = AgentAsATool(agent=sql_agent, name='sql_agent_tool', description='Read approved HANA tables using natural language')
    >>> chatbot = ContextAgent(llm=llm, tools=[sql_tool], storage_dir='.context_agent', session_id='sql_example')
    >>> chatbot.chat("Show me the rows in IRIS_DATA_FULL_TBL")
    """
    def __new__(cls, agent, name, description):
        return Tool.from_function(
        func=agent.invoke,
        name=name,
        description=description
    )
