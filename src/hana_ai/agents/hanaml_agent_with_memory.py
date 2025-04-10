"""
A chatbot that can remember the chat history and use it to generate responses.

The following class is available:
    
        * :class `HANAMLAgentWithMemory`

"""

#pylint: disable=ungrouped-imports, abstract-method
import json
import logging
import pandas as pd
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages.base import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema.messages import AIMessage


logging.getLogger().setLevel(logging.ERROR)

class _ToolObservationCallbackHandler(BaseCallbackHandler):
    def __init__(self, memory_getter, max_observations=5):
        super().__init__()
        self.memory_getter = memory_getter
        self.max_observations = max_observations  # Set your desired limit here

    def on_tool_end(self, output: str, **kwargs):
        memory = self.memory_getter()
        # Get all current observations in chronological order
        current_obs = [msg for msg in memory.messages if self._is_observation(msg)]

        # Calculate how many to remove if over limit (before adding new)
        excess = len(current_obs) - (self.max_observations - 1)
        if excess > 0:
            # Remove oldest 'excess' observations from memory
            to_remove = current_obs[:excess]
            memory.messages = [msg for msg in memory.messages if msg not in to_remove]

        # Add new observation
        memory.add_message(AIMessage(content=f"Observation: {output}"))

    def _is_observation(self, msg: BaseMessage) -> bool:
        """Identifies observation messages"""
        return isinstance(msg, AIMessage) and msg.content.startswith("Observation: ")

def _get_pandas_meta(df):
    """
    Get the metadata of a pandas dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to get the metadata from.

    Returns
    -------
    dict
        The metadata of the dataframe.
    """
    if hasattr(df, 'columns'):
        columns = df.columns.tolist()
        return json.dumps({"columns": columns})
    return ''
class HANAMLAgentWithMemory(object):
    """
    A chatbot that can remember the chat history and use it to generate responses.

    Parameters
    ----------
    llm : LLM
        The language model to use.
    tools : list of BaseTool
        The tools to use.
    session_id : str, optional
        The session ID to use. Default to "hana_ai_chat_session".
    n_messages : int, optional
        The number of messages to remember. Default to 10.
    max_observations : int, optional
        The maximum number of observations to remember. Default to 5.
    verbose : bool, optional
        Whether to be verbose. Default to False.

    Examples
    --------
    Assume cc is a connection to a SAP HANA instance:

    >>> from hana_ai.agents.hanaml_agent_with_memory import HANAMLAgentWithMemory
    >>> from hana_ai.tools.toolkit import HANAMLToolkit

    >>> tools = HANAMLToolkit(connection_context=cc, used_tools='all').get_tools()
    >>> chatbot = HANAMLAgentWithMemory(llm=llm, tools=tools, session_id='hana_ai_test', n_messages=10)
    >>> chatbot.run(question="Analyze the data from the table MYTEST.")
    """
    def __init__(self, llm, tools, session_id="hanaai_chat_session", n_messages=10, max_observations=5, verbose=False, **kwargs):
        self.llm = llm
        self.tools = tools
        self.memory = InMemoryChatMessageHistory(session_id=session_id)
        system_prompt = """You're an assistant skilled in data science using hana-ml tools.
        Always respond with a valid JSON blob containing 'action' and 'action_input' to call tools.
        Ask for missing parameters if needed. NEVER return raw JSON strings outside this structure."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history", n_messages=n_messages),
            ("human", "{question}"),
        ])
        # Create callback handler linked to memory
        self.observation_callback = _ToolObservationCallbackHandler(lambda: self.memory, max_observations=max_observations)
        chain: Runnable = prompt | initialize_agent(tools,
                                                    llm,
                                                    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,verbose=verbose,
                                                    callbacks=[self.observation_callback],
                                                    **kwargs)

        self.agent_with_chat_history = RunnableWithMessageHistory(chain,
                                                                  lambda session_id: self.memory,
                                                                  input_messages_key="question",
                                                                  history_messages_key="history")
        self.config = {"configurable": {"session_id": session_id}}

    def add_user_message(self, content: str):
        """Add a message from the user to the chat history."""
        self.memory.add_user_message(content)

    def add_ai_message(self, content: str):
        """Add a response from the AI to the chat history."""
        self.memory.add_ai_message(content)

    def run(self, question):
        """
        Chat with the chatbot.

        Parameters
        ----------
        question : str
            The question to ask.
        """
        try:
            response = self.agent_with_chat_history.invoke({"question": question},
                                                           config={**self.config,  # Preserve session_id
                                                                   "callbacks": [self.observation_callback]
                                                                   })
        except Exception as e:
            error_message = str(e)
            self.memory.add_user_message(question)
            self.memory.add_ai_message(f"The error message is `{error_message}`.")
            response = error_message
        if isinstance(response, pd.DataFrame):
            meta = _get_pandas_meta(response)
            self.memory.add_user_message(question)
            self.memory.add_ai_message(f"The returned is a pandas dataframe with the metadata:\n{meta}")
        if isinstance(response, dict) and 'output' in response:
            response = response['output']
            if isinstance(response, pd.DataFrame):
                meta = _get_pandas_meta(response)
                self.memory.add_user_message(question)
                self.memory.add_ai_message(f"The returned is a pandas dataframe with the metadata: \n{meta}")
        if isinstance(response, str):
            if response.startswith("Action:"): # force to call tool if return a Action string
                action_json = response[7:]
                try:
                    response = json.loads(action_json)
                except Exception as e:
                    error_message = str(e)
                    self.memory.add_ai_message(f"The error message is `{error_message}`. The response is `{response}`.")
            if isinstance(response, str) and response.strip() == "":
                response = "I'm sorry, I don't understand. Please ask me again."
        if isinstance(response, dict) and 'action' in response and 'action_input' in response:
            action = response.get("action")
            for tool in self.tools:
                if tool.name == action:
                    action_input = response.get("action_input")
                    try:
                        response = tool.run(action_input)
                        if isinstance(response, pd.DataFrame):
                            meta = _get_pandas_meta(response)
                            self.memory.add_ai_message(f"The returned is a pandas dataframe with the metadata: \n{meta}")
                        else:
                            self.memory.add_ai_message(f"The tool {tool.name} has been already called via {action_input}. The result is `{response}`.")
                        return response
                    except Exception as e:
                        error_message = str(e)
                        self.memory.add_ai_message(f"The error message is `{error_message}`. The response is `{response}`.")
        return response

def stateless_call(llm, tools, question, chat_history=None, verbose=False):
    """
    Utility function to call the agent with chat_history input. For stateless use cases.
    This function is useful for BAS integration purposes.

    Parameters
    ----------
    llm : LLM
        The language model to use.
    tools : list of BaseTool
        The tools to use.
    question : str
        The question to ask.
    chat_history : list of str
        The chat history. Default to None.

    Returns
    -------
    str
        The response.
    """
    if chat_history is None:
        chat_history = []
    system_prompt = """You're an assistant skilled in data science using hana-ml tools.
    Always respond with a valid JSON blob containing 'action' and 'action_input' to call tools.
    Ask for missing parameters if needed. NEVER return raw JSON strings outside this structure."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history", messages=chat_history),
        ("human", "{question}"),
    ])
    agent: Runnable = prompt | initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=verbose)
    response = agent.invoke({"question": question, "history": chat_history})
    if isinstance(response, dict) and 'output' in response:
        response = response['output']
    if isinstance(response, str):
        if response.startswith("Action:"): # force to call tool if return a Action string
            action_json = response[7:]
            try:
                response = json.loads(action_json)
            except Exception as e:
                error_message = str(e)
                response = f"The error message is `{error_message}`. Please display the error message, and then analyze the error message and provide the solution."
        if isinstance(response, str) and response.strip() == "":
            response = "I'm sorry, I don't understand. Please ask me again."
    if isinstance(response, dict) and 'action' in response and 'action_input' in response:
        action = response.get("action")
        for tool in tools:
            if tool.name == action:
                action_input = response.get("action_input")
                try:
                    response = tool.run(action_input)
                    return response
                except Exception as e:
                    error_message = str(e)
                    response = f"The error message is `{error_message}`. Please display the error message, and then analyze the error message and provide the solution."
    return response
