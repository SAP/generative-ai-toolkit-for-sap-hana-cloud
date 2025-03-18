"""
A chatbot that can remember the chat history and use it to generate responses.

"""
import logging
from langchain.agents import initialize_agent, AgentType
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

logging.getLogger().setLevel(logging.ERROR)
class HANAMLAgentWithMemory(object):
    """
    A chatbot that can remember the chat history and use it to generate responses.

    Parameters
    ----------
    llm : LLM
        The language model to use.
    tools : dict
        The tools to use.
    session_id : str
        The session ID to use.
    n_messages : int
        The number of messages to remember.

    Examples
    --------
    >>> from hana_ai.agents.hanaml_agent_with_memory import HANAMLAgentWithMemory
    >>> from hana_ai.tools.toolkit import HANAMLToolkit

    >>> tools = HANAMLToolkit(connection_context, used_tools='all').get_tools()
    >>> chatbot = HANAMLAgentWithMemory(llm=llm, tools=tools, session_id='hana_ai_test', n_messages=10)
    >>> chatbot.run("Analyze the data from the table MYTEST.")
    """
    def __init__(self, llm, tools, session_id="hanaai_chat_session", n_messages=10, verbose=False):
        self.llm = llm
        memory = InMemoryChatMessageHistory(session_id=session_id)
        system_prompt = """You're an assistant skilled in data science using hana-ml tools. 
        Always respond with a valid JSON blob containing 'action' and 'action_input' to call tools. 
        Ask for missing parameters if needed. NEVER return raw JSON strings outside this structure."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history", n_messages=n_messages),
            ("human", "{question}"),
        ])
        chain: Runnable = prompt | initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=verbose)

        self.agent_with_chat_history = RunnableWithMessageHistory(chain,
                                                                  lambda session_id: memory,
                                                                  input_messages_key="question",
                                                                  history_messages_key="history")
        self.config = {"configurable": {"session_id": session_id}}

    def run(self, question):
        """"
        Chat with the chatbot.

        Parameters
        ----------
        question : str
            The question to ask.
        """
        try:
            response = self.agent_with_chat_history.invoke({"question": question}, self.config)
        except Exception as e:
            error_message = str(e)
            response = self.agent_with_chat_history.invoke({"question": f"The question is `{question}`.The error message is `{error_message}`. Please display the error message, and then analyze the error message and provide the solution."}, self.config)
        if 'output' in response:
            return response['output']
        return response
