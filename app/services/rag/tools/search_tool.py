from __future__ import annotations

from langchain_core.tools import tool

from app.services.rag.retrieval.retriever import HybridRetriever
from app.services.rag.retrieval.formatting import format_search_result
from app.logger_setup import log


def make_rag_tool(retriever: HybridRetriever):
    """
    Возвращает LangChain tool с замкнутым ретривером.
    """

    @tool
    def search_textbook(query: str) -> str:
        """
        Поиск по учебнику по алгоритмам и структурам данных.

        Используй этот инструмент когда нужно найти:
        - определения терминов и понятий
        - описания алгоритмов и их работы
        - теоремы, леммы, доказательства
        - примеры и иллюстрации из учебника
        - оценки сложности алгоритмов

        Правила формирования запроса:
        - Передавай ключевые термины, а не вопрос целиком.
          Хорошо: "алгоритм Дейкстры кратчайший путь"
          Плохо:  "Как работает алгоритм Дейкстры?"
        - Для сравнения двух понятий вызови инструмент дважды —
          по одному запросу на каждое понятие.
        - Используй русскоязычные термины, латинские аббревиатуры можно добавить рядом.

        Args:
            query: ключевые термины для поиска на русском языке.

        Returns:
            Релевантные фрагменты учебника, сгруппированные по разделам.
        """
        log.info(f'Выполняется поиск по запросу: {query}')
        result = retriever.search(query)
        return format_search_result(result)

    return search_textbook


def _run_tests(tool) -> None:
    """Простые smoke-тесты инструмента."""

    test_queries = [
        "алгоритм Дейкстры кратчайший путь",
        "обход в ширину",
        "построение максимального потока",
    ]

    # Проверяем метаданные инструмента
    print("=== Tool metadata ===")
    print(f"  name        : {tool.name}")
    print(f"  description : {tool.description}")
    print(f"  args schema : {tool.args_schema.model_json_schema()}\n")

    # Прогоняем тестовые запросы
    print("=== Search results ===")
    for query in test_queries:
        print(f"\n[Query] {query}")
        print("-" * 60)
        try:
            # .invoke() — стандартный способ вызова LangChain tool
            result: str = tool.invoke({"query": query})
            # Печатаем только первые 300 символов, чтобы не засорять вывод
            print(f"  OK  → {result}")
        except Exception as exc:
            print(f"  ERR → {exc}")


if __name__ == "__main__":
    retriever = HybridRetriever.from_yaml()
    search_tool = make_rag_tool(retriever)
    _run_tests(search_tool)