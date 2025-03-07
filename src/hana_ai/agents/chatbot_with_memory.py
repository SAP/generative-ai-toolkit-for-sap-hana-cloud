
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import initialize_agent, AgentType
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class ChatbotWithMemory(object):
    def __init__(self, llm, tools, session_id="hanaai_chat_session", n_messages=10):
        self.llm = llm
        memory = InMemoryChatMessageHistory(session_id=session_id)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You're an assistant who's good at data science. You need to ask the user for the missing information to use hana-ml tools."),
            MessagesPlaceholder(variable_name="history", n_messages=n_messages),
            ("human", "{question}"),
        ])
        chain = prompt | initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION)
        self.agent_with_chat_history = RunnableWithMessageHistory(chain,
                                                                  lambda session_id: memory,
                                                                  input_messages_key="question",
                                                                  history_messages_key="history")
        self.config = {"configurable": {"session_id": session_id}}
    def chat(self, question):
        try:
            return self.agent_with_chat_history.invoke({"question": question}, self.config)
        except Exception as e:
            error_message = str(e)
            return self.agent_with_chat_history.invoke({"question": f"The question is `{question}`.The error message is `{error_message}`. Please analyze the error message and provide the solution."}, self.config)
