"""
Agent as a tool.

The following function is available:

    * :func `AgentAsATool`
"""
from langchain.agents import Tool

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
    >>> from hana_ai.agents.hanaml_agent_with_memory import HANAMLAgentWithMemory

    >>> sql_agent = create_hana_sql_agent(llm=llm, connection_context=cc, tools=[code_tool], verbose=True)
    >>> sql_tool = AgentAsATool(agent=sql_agent, name='sql_agent_tool', description='To generate SQL code from natural language')
    >>> chatbot = HANAMLAgentWithMemory(llm=llm, tools=tools + [sql_tool], session_id='hana_ai_test', n_messages=10)
    >>> chatbot.run(question="show me all the tables in the system?")
    """
    def __new__(cls, agent, name, description):
        return Tool.from_function(
        func=agent.invoke,
        name=name,
        description=description
    )
