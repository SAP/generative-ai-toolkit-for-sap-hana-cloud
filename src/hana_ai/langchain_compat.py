"""LangChain compatibility helpers for old and new APIs."""
from __future__ import annotations

from typing import Any, List, Optional

try:
    from langchain.llms.base import BaseLLM
except Exception:
    try:
        from langchain_core.language_models.llms import BaseLLM
    except Exception:
        from langchain_core.language_models import BaseLanguageModel as BaseLLM

try:
    from langchain.prompts import ChatPromptTemplate
except Exception:
    from langchain_core.prompts import ChatPromptTemplate

try:
    from langchain.prompts import MessagesPlaceholder
except Exception:
    from langchain_core.prompts import MessagesPlaceholder

try:
    from langchain.prompts import HumanMessagePromptTemplate
except Exception:
    from langchain_core.prompts import HumanMessagePromptTemplate

try:
    from langchain.prompts import PromptTemplate
except Exception:
    from langchain_core.prompts import PromptTemplate

try:
    from langchain.schema import SystemMessage
except Exception:
    from langchain_core.messages import SystemMessage

try:
    from langchain.schema import HumanMessage
except Exception:
    from langchain_core.messages import HumanMessage

try:
    from langchain.schema import AIMessage
except Exception:
    from langchain_core.messages import AIMessage

try:
    from langchain.schema import AgentAction, AgentFinish
except Exception:
    from langchain_core.agents import AgentAction, AgentFinish

try:
    from langchain.tools import BaseTool
except Exception:
    from langchain_core.tools import BaseTool

try:
    from langchain.tools import Tool
except Exception:
    try:
        from langchain.agents import Tool
    except Exception:
        from langchain_core.tools import Tool

try:
    from langchain.embeddings.base import Embeddings
except Exception:
    from langchain_core.embeddings import Embeddings

try:
    from langchain.memory import ConversationBufferWindowMemory
except Exception:
    try:
        from langchain_community.memory import ConversationBufferWindowMemory
    except Exception:
        ConversationBufferWindowMemory = None

try:
    from langchain.callbacks.base import BaseCallbackHandler
except Exception:
    from langchain_core.callbacks.base import BaseCallbackHandler

try:
    from langchain.callbacks.base import BaseCallbackManager
except Exception:
    from langchain_core.callbacks import BaseCallbackManager

try:
    from langchain.output_parsers.openai_tools import PydanticToolsParser
except Exception:
    from langchain_core.output_parsers.openai_tools import PydanticToolsParser

try:
    from langchain.agents import AgentExecutor
except Exception:
    try:
        from langchain.agents.agent import AgentExecutor
    except Exception:
        try:
            from langchain.agents.agent_executor import AgentExecutor
        except Exception:
            AgentExecutor = None

try:
    from langchain.agents import create_openai_functions_agent
except Exception:
    try:
        from langchain.agents.openai_functions_agent import create_openai_functions_agent
    except Exception:
        create_openai_functions_agent = None

try:
    from langchain.agents import create_agent as create_graph_agent
except Exception:
    create_graph_agent = None

try:
    from langchain.agents import initialize_agent
except Exception:
    initialize_agent = None

try:
    from langchain.agents import AgentType
except Exception:
    try:
        from langchain.agents.agent_types import AgentType
    except Exception:
        AgentType = None

try:
    from langchain_core.callbacks.manager import CallbackManagerForChainRun
except Exception:
    CallbackManagerForChainRun = None


def _input_to_messages(agent_input: Any) -> List[dict]:
    if isinstance(agent_input, dict) and "messages" in agent_input:
        return agent_input["messages"]

    text = None
    if isinstance(agent_input, dict) and "input" in agent_input:
        parts = agent_input["input"]
        if isinstance(parts, list) and parts:
            first = parts[0]
            text = first.get("text") if isinstance(first, dict) else str(first)
        else:
            text = str(parts)
    else:
        text = str(agent_input)
    return [{"role": "user", "content": text}]


def extract_agent_output(result: Any) -> str:
    messages = None
    if isinstance(result, dict):
        messages = result.get("messages")
        if messages is None and "output" in result:
            return str(result.get("output"))
    elif isinstance(result, list):
        messages = result
    if not messages:
        return str(result)
    last = messages[-1]
    if hasattr(last, "content"):
        return str(last.content)
    if isinstance(last, dict):
        return str(last.get("content") or last.get("text") or last)
    return str(last)


class GraphAgentExecutor:
    is_graph = True

    def __init__(self, graph: Any):
        self._graph = graph

    def invoke(self, agent_input: Any) -> dict:
        messages = _input_to_messages(agent_input)
        result = self._graph.invoke({"messages": messages})
        return {
            "output": extract_agent_output(result),
            "messages": result.get("messages") if isinstance(result, dict) else result,
        }


if AgentExecutor is not None:
    class FormatSafeAgentExecutor(AgentExecutor):
        """AgentExecutor that formats observations safely."""
        def _take_next_step(
            self,
            name_to_tool_map: dict[str, BaseTool],
            color_mapping: dict[str, str],
            inputs: dict[str, str],
            intermediate_steps: list[tuple[AgentAction, str]],
            run_manager: Optional[CallbackManagerForChainRun] = None,
        ) -> Any:
            next_step = super()._take_next_step(
                name_to_tool_map, color_mapping, inputs, intermediate_steps, run_manager
            )
            if isinstance(next_step, list):
                formatted_steps = []
                for action, observation in next_step:
                    if isinstance(observation, str):
                        formatted_obs = [{"type": "text", "text": observation}]
                        formatted_steps.append((action, formatted_obs))
                    else:
                        formatted_steps.append((action, observation))
                return formatted_steps
            return next_step
else:
    FormatSafeAgentExecutor = None


def build_agent_executor(
    llm: Any,
    tools: List[Any],
    *,
    prompt: Any = None,
    system_prompt: Optional[Any] = None,
    memory: Any = None,
    verbose: bool = False,
    max_iterations: int = 20,
    handle_parsing_errors: bool = True,
    return_intermediate_steps: bool = False,
    executor_cls: Any = None,
    return_agent: bool = False,
) -> Any:
    if create_openai_functions_agent and AgentExecutor:
        if prompt is None:
            if system_prompt is None:
                system_prompt = "You are a helpful assistant with access to tools."
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_prompt),
                HumanMessagePromptTemplate.from_template("{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
        agent = create_openai_functions_agent(llm, tools, prompt)
        exec_cls = executor_cls or AgentExecutor
        if exec_cls is AgentExecutor:
            executor = AgentExecutor.from_agent_and_tools(
                agent=agent,
                tools=tools,
                verbose=verbose,
                max_iterations=max_iterations,
                handle_parsing_errors=handle_parsing_errors,
                memory=memory,
                return_intermediate_steps=return_intermediate_steps,
            )
        else:
            executor = exec_cls(
                agent=agent,
                tools=tools,
                verbose=verbose,
                max_iterations=max_iterations,
                handle_parsing_errors=handle_parsing_errors,
                memory=memory,
                return_intermediate_steps=return_intermediate_steps,
            )
    elif create_graph_agent:
        agent = create_graph_agent(model=llm, tools=tools, system_prompt=system_prompt)
        executor = GraphAgentExecutor(agent)
    else:
        raise ImportError("No compatible agent constructor found in langchain.")

    if return_agent:
        return agent, executor
    return executor


class _FallbackConversationBufferWindowMemory:
    def __init__(self, memory_key: str = "chat_history", k: int = 5, return_messages: bool = True):
        self.memory_key = memory_key
        self.k = k
        self.return_messages = return_messages
        self.chat_memory: List[Any] = []

    def load_memory_variables(self, _inputs: dict) -> dict:
        messages = self.chat_memory[-self.k:] if self.k else list(self.chat_memory)
        if self.return_messages:
            return {self.memory_key: messages}
        joined = "\n".join([getattr(m, "content", str(m)) for m in messages])
        return {self.memory_key: joined}

    def save_context(self, inputs: dict, outputs: dict) -> None:
        human = inputs.get("input") or inputs.get("question") or ""
        ai = outputs.get("output") if isinstance(outputs, dict) else str(outputs)
        if human:
            self.chat_memory.append(HumanMessage(content=str(human)))
        if ai is not None:
            self.chat_memory.append(AIMessage(content=str(ai)))

    def clear(self) -> None:
        self.chat_memory.clear()


def get_conversation_buffer_window_memory(*args: Any, **kwargs: Any) -> Any:
    if ConversationBufferWindowMemory is not None:
        return ConversationBufferWindowMemory(*args, **kwargs)
    return _FallbackConversationBufferWindowMemory(*args, **kwargs)
