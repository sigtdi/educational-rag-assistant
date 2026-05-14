from __future__ import annotations

from typing import Annotated

from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
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

def init_llm(model_type: str, config: AgentConfig):
    if model_type == 'local':
        llm = ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            keep_alive="60m"
        )
        return llm
    elif model_type == 'cloud_model':
        llm = ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature
        )
        return llm


    return None


def build_graph_with_tools(config: AgentConfig, tools: list[BaseTool] | None, model_type: str) -> StateGraph:
    llm_base = init_llm(model_type, config)

    llm = llm_base.bind_tools(tools)

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

def build_graph_without_tools(config: AgentConfig, size_type: str):
    llm = init_llm(size_type, config)

    system_prompt = config.system_prompt

    def call_model(state: AgentState) -> AgentState:
        """
        Узел LLM: добавляет системный промпт и вызывает модель.
        """
        from langchain_core.messages import SystemMessage

        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("llm", call_model)
    graph.add_edge("llm", END)

    graph.set_entry_point("llm")

    return graph.compile()

class Agent:
    def __init__(self, config: AgentConfig, with_tools: bool, model_type: str = 'small', tools: list[BaseTool] | None = None) -> None:
        if with_tools:
            self._graph = build_graph_with_tools(config, tools, model_type)
        else:
            self._graph = build_graph_without_tools(config, model_type)

    def ask(
        self,
        message: str,
        chat_history: list[BaseMessage] | None = None,
    ) -> dict:
        """
        Точка входа для API.
        """
        history = chat_history or []
        initial_state: AgentState = {
            "messages": history + [HumanMessage(content=message)],
        }

        final_state = self._graph.invoke(initial_state)

        final_answer = final_state["messages"][-1].content
        search_results = [
            msg.content for msg in final_state["messages"]
            if isinstance(msg, ToolMessage)
        ]

        return {"answer": final_answer, "contexts": search_results}

def make_agent(from_yaml: bool,
               with_tools: bool | None = None,
               model_name: str | None = None,
               temperature: float | None = None,
               model_type: str | None = None,
               prompt: str | None = None) -> Agent | None:

    if from_yaml:
        log.info('Создан агент на основе дефолтных значений из файла конфигурации')
        config = AgentConfig.from_yaml()
        retriever = HybridRetriever.from_yaml()
        return Agent(config=config, with_tools=True, tools=[make_rag_tool(retriever)])

    if model_name is None or temperature is None or model_type is None:
        log.error('Модель не была создана, не хватает данных')
        return None

    log.info(
        f'Создан агент:\n\tmodel: {model_name}\n\ttemperature: {temperature}\n\tпоиск {'не' if not with_tools else ''} подключен')

    if prompt is not None:
        config = AgentConfig(model_name=model_name, temperature=temperature, system_prompt=prompt)
    else:
        config = AgentConfig(model_name=model_name, temperature=temperature)

    if with_tools:
        retriever = HybridRetriever.from_yaml()
        return Agent(config=config, with_tools=True, model_type=model_type, tools=[make_rag_tool(retriever)])
    else:
        return Agent(config=config, with_tools=False, model_type=model_type)

if __name__ == "__main__":

    agent = make_agent(from_yaml=True)

    questions = [
        "Расскажи мне про алгоритм уконена",
    ]
    for q in questions:
        print(f"\nВопрос: {q}")
        print(f"Ответ:  {agent.ask(q)}")