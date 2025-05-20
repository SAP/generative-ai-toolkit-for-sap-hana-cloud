"""
A chatbot that can remember the chat history and use it to generate responses.

The following class is available:
    
        * :class `HANAMLAgentWithMemory`

"""

#pylint: disable=ungrouped-imports, abstract-method
import json
import inspect
import logging
import pandas as pd
from pydantic import ValidationError
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages.base import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema.messages import AIMessage
from langchain.load.dump import dumps

logging.getLogger().setLevel(logging.ERROR)

def _check_generated_cap_for_bas(intermediate_steps):
    """
    Check if the generated CAP artifacts are valid.

    Parameters
    ----------
    intermediate_steps : str
        The intermediate steps to check.

    Returns
    -------
    bool
        True if the generated CAP artifacts are valid, False otherwise.
    """
    try:
        ss = json.loads(intermediate_steps)
    except:
        return False
    if intermediate_steps is None:
        return False
    if not isinstance(ss, list):
        return False
    for step in ss:
        for substep in step:
            if isinstance(substep, dict) and 'type' in substep and substep['type'] == 'constructor':
                if 'kwargs' in substep and 'tool' in substep['kwargs']:
                    tool_name = substep['kwargs']['tool']
                    if tool_name == "cap_artifacts_for_bas":
                        return True
    return False

def _inspect_python_code(intermediate_steps, tools):
    try:
        ss = json.loads(intermediate_steps)
    except:
        return None
    if intermediate_steps is None:
        return None
    collect_tool_call = []
    if not isinstance(ss, list):
        return None
    for step in ss:
        for substep in step:
            if isinstance(substep, dict) and 'type' in substep and substep['type'] == 'constructor':
                if 'kwargs' in substep and 'tool' in substep['kwargs']:
                    tool_name = substep['kwargs']['tool']
                    for tool in tools:
                        if tool.name == tool_name:
                            collect_tool_call.append({"tool_name": tool_name, "parameters": json.dumps(substep['kwargs']['tool_input']), "python_code": inspect.getsource(tool._run)})
                            break
    return collect_tool_call
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
        Ask for missing parameters if needed. Regardless of whether this tool has been called before, it must be called."""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history", n_messages=n_messages),
            ("human", "{question}"),
        ])
        self.kwargs = {**kwargs}
        self.verbose = verbose
        # Create callback handler linked to memory
        self.observation_callback = _ToolObservationCallbackHandler(lambda: self.memory, max_observations=max_observations)
        chain: Runnable = self.prompt | initialize_agent(tools,
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

    def set_return_direct(self, config: dict):
        """
        Set the return_direct flag for a specific tool.

        Parameters
        ----------
        config : dict
            A dictionary containing the tool name and the return_direct flag.
            Example: {"fetch_data": True}
        """
        if isinstance(config, dict):
            for idx, tool in enumerate(self.tools):
                if tool.name in config:
                    self.tools[idx].return_direct = config[tool.name]
        else:
            raise ValueError("The config parameter should be a dictionary.")
        # 需要重新初始化agent更新工具信息
        chain: Runnable = self.prompt | initialize_agent(self.tools,
                                                    self.llm,
                                                    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,verbose=self.verbose,
                                                    callbacks=[self.observation_callback],
                                                    **self.kwargs)

        self.agent_with_chat_history = RunnableWithMessageHistory(chain,
                                                                  lambda session_id: self.memory,
                                                                  input_messages_key="question",
                                                                  history_messages_key="history")

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
            if "Error code: 429" not in error_message:
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
                    if "Error code: 429" not in error_message:
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
                        if "Error code: 429" not in error_message:
                            self.memory.add_ai_message(f"The error message is `{error_message}`. The response is `{response}`.")
        return response

def stateless_call(llm, tools, question, chat_history=None, verbose=False, return_intermediate_steps=False):
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
    verbose : bool, optional
        Verbose mode. Default to False.
    return_intermediate_steps : bool, optional
        Whether to return intermediate steps. Default to False.

    Returns
    -------
    str
        The response.
    """
    if chat_history is None:
        chat_history = []
    system_prompt = """You're an assistant skilled in data science using hana-ml tools.
    Ask for missing parameters if needed. Regardless of whether this tool has been called before, it must be called."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history", messages=chat_history),
        ("human", "{question}"),
    ])
    agent: Runnable = prompt | initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=verbose, return_intermediate_steps=return_intermediate_steps, handle_parsing_errors=True)
    intermediate_steps = None
    try:
        response = agent.invoke({"question": question, "history": chat_history})
        if return_intermediate_steps is True:
            intermediate_steps = response.get("intermediate_steps")
    except ValidationError as e:
        # Parse Pydantic error details for feedback
        error_details = "\n".join([f"{err['loc'][0]}: {err['msg']}" for err in e.errors()])
        error_message = f"Please provide the parameters:\n {error_details}"
        chat_history.append(("system", error_message))
        response = {"output": error_message}
    except Exception as e:
        error_message = str(e)
        response = f"The error message is `{error_message}`. Please display the error message, and then analyze the error message and provide the solution."
        if return_intermediate_steps is True:
            response = {
                "output": response,
                "intermediate_steps": dumps(intermediate_steps) if return_intermediate_steps else None
            }
            response["inspect_script"] = _inspect_python_code(response["intermediate_steps"], tools)
            response["generated_cap_project"] = _check_generated_cap_for_bas(response["intermediate_steps"])
        return response

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
                except Exception as e:
                    error_message = str(e)
                    response = f"The error message is `{error_message}`. Please display the error message, and then analyze the error message and provide the solution."
    if return_intermediate_steps is True and 'intermediate_steps' not in response:
        # Add the intermediate steps to the response if requested
        response = {
            "output": response,
            "intermediate_steps": dumps(intermediate_steps) if intermediate_steps else None
        }
        response["inspect_script"] = _inspect_python_code(response["intermediate_steps"], tools)
        response["generated_cap_project"] = _check_generated_cap_for_bas(response["intermediate_steps"])
    return response
