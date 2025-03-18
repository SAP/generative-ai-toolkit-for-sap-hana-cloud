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
class ChatbotWithMemory(object):
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
    >>> from hana_ai.agents.chatbot_with_memory import ChatbotWithMemory
    >>> from hana_ai.tools.toolkit import HANAMLToolkit

    >>> tools = HANAMLToolkit(connection_context, used_tools='all').get_tools()
    >>> chatbot = ChatbotWithMemory(llm=llm, tools=tools, session_id='hana_ai_test', n_messages=10)
    >>> chatbot.chat("Analyze the data from the table MYTEST.")
    """
    def __init__(self, llm, tools, session_id="hanaai_chat_session", n_messages=10):
        self.llm = llm
        memory = InMemoryChatMessageHistory(session_id=session_id)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You're an assistant who's good at data science. You need to ask the user for the missing information to use hana-ml tools."),
            MessagesPlaceholder(variable_name="history", n_messages=n_messages),
            ("human", "{question}"),
        ])
        chain: Runnable = prompt | initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION)
        self.agent_with_chat_history = RunnableWithMessageHistory(chain,
                                                                  lambda session_id: memory,
                                                                  input_messages_key="question",
                                                                  history_messages_key="history")
        self.config = {"configurable": {"session_id": session_id}}

    def chat(self, question):
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
