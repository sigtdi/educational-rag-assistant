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
          Плохо: "Как работает алгоритм Дейкстры?"
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