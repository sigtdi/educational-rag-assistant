from __future__ import annotations

from typing import Annotated

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from app.services.rag.chains.agent_config import AgentConfig
from app.services.rag.retrieval.retriever import HybridRetriever
from app.services.rag.tools.search_tool import make_rag_tool
from app.logger_setup import log

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def build_graph(config: AgentConfig, tools: list[BaseTool] | None) -> StateGraph:
    llm_base = ChatOllama(
        model=config.model_name,
        temperature=config.temperature,
        keep_alive="60m",
    )

    if tools:
        llm = llm_base.bind_tools(tools)
    else:
        llm = llm_base

    system_prompt = config.system_prompt

    def call_model(state: AgentState) -> AgentState:
        """
        Узел LLM: добавляет системный промпт и вызывает модель.
        """
        from langchain_core.messages import SystemMessage

        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        """
        Роутер: если модель хочет вызвать тул — идём в tool_node, иначе завершаем.
        """
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)
    graph.add_node("llm", call_model)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", should_continue)
    graph.add_edge("tools", "llm")

    return graph.compile()

class Agent:
    def __init__(self, config: AgentConfig, tools: list[BaseTool] | None = None) -> None:
        self._graph = build_graph(config, tools)

    def ask(
        self,
        message: str,
        chat_history: list[BaseMessage] | None = None,
    ) -> str:
        """
        Точка входа для API.
        """
        history = chat_history or []
        initial_state: AgentState = {
            "messages": history + [HumanMessage(content=message)],
        }

        final_state = self._graph.invoke(initial_state)
        return final_state["messages"][-1].content

def make_agent(from_yaml: bool,
               need_tools: bool | None = None,
               model_name: str | None = None,
               temperature: float | None = None) -> Agent | None:

    if from_yaml:
        log.info('Создан агент на основе дефолтных значений из файла конфигурации')
        config = AgentConfig.from_yaml()
        retriever = HybridRetriever.from_yaml()
        return Agent(config, tools=[make_rag_tool(retriever)])
    elif need_tools and model_name is not None and temperature is not None:
        log.info(f'Создан агент:\n\tmodel: {model_name}\n\ttemperature: {temperature}\n\tпоиск подключен')
        config = AgentConfig(model_name=model_name, temperature=temperature)
        retriever = HybridRetriever.from_yaml()
        return Agent(config, tools=[make_rag_tool(retriever)])
    elif not need_tools and model_name is not None and temperature is not None:
        log.info(f'Создан агент:\n\tmodel: {model_name}\n\ttemperature: {temperature}\n\tпоиск не подключен')
        config = AgentConfig(model_name=model_name, temperature=temperature)
        return Agent(config)

    log.error('Модель не была создана')
    return None
if __name__ == "__main__":

    agent = make_agent(from_yaml=True)

    questions = [
        "Что такое ориентированный граф?",
        "Чем отличается обход в глубину от обхода в ширину?",
    ]
    for q in questions:
        print(f"\nВопрос: {q}")
        print(f"Ответ:  {agent.ask(q)}")