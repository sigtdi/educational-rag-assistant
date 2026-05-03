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

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def build_graph(config: AgentConfig, tools: list[BaseTool]) -> StateGraph:
    llm = ChatOllama(
        model=config.model_name,
        base_url=config.ollama_base_url,
        temperature=config.temperature,
    )

    llm_with_tools = llm.bind_tools(tools)

    system_prompt = config.system_prompt

    def call_model(state: AgentState) -> AgentState:
        """
        Узел LLM: добавляет системный промпт и вызывает модель.
        """
        from langchain_core.messages import SystemMessage

        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm_with_tools.invoke(messages)
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
    def __init__(self, config: AgentConfig, tools: list[BaseTool]) -> None:
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

if __name__ == "__main__":
    from app.services.rag.retrieval.retriever import HybridRetriever
    from app.services.rag.tools.search_tool import make_rag_tool

    config = AgentConfig()
    retriever = HybridRetriever.from_yaml()
    agent = Agent(config, tools=[make_rag_tool(retriever)])

    questions = [
        "Что такое ориентированный граф?",
        "Чем отличается обход в глубину от обхода в ширину?",
    ]
    for q in questions:
        print(f"\nВопрос: {q}")
        print(f"Ответ:  {agent.ask(q)}")