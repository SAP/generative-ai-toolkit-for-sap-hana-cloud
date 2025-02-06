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
    """
    def __new__(cls, agent, name, description):
        return Tool.from_function(
        func=agent.invoke,
        name=name,
        description=description
    )
