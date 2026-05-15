from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from qdrant_client.models import Record
from sentence_transformers import CrossEncoder

from app.logger_setup import log
from app.services.rag.retrieval.retriever_config import RetrieverConfig


@dataclass
class ChunkResult:
    id: str
    section_header: str
    text: str
    metadata: dict[str, Any]
    is_picture: bool = False
    image_path: str | None = None
    rerank_score: float | None = None

    @classmethod
    def from_document(
        cls,
        doc: Document,
        rerank_score: float | None = None,
    ) -> "ChunkResult":
        meta = doc.metadata
        chunk_id = meta.get("id", "")
        is_picture = meta.get("type") == "picture"
        image_path = meta.get("image_path") if is_picture else None

        return cls(
            id=chunk_id,
            section_header=meta.get("section_path", ""),
            text=meta.get("text", ""),
            metadata=meta,
            is_picture=is_picture,
            image_path=image_path,
            rerank_score=rerank_score,
        )


@dataclass
class SearchResult:
    top_chunks: list[ChunkResult]
    group_chunks: list[ChunkResult]
    mentioned_chunks: list[ChunkResult]
    mentioned_labels: dict[str, str] = field(default_factory=dict) # chunk_id: подпись ("Определение 6" и т.п.)


class HybridRetriever:
    """
    Гибридный поиск c использованием реранкера.
    """

    def __init__(self, config: RetrieverConfig) -> None:
        self.config = config

        self._client = QdrantClient(url=self.config.qdrant_url)

        self._embeddings = HuggingFaceEmbeddings(
            model_name=self.config.dense_model,
            model_kwargs={
                "device": "cpu",
                "prompts": {"query": self.config.embedding_query_instruction},
                "default_prompt_name": "query",
            },
        )

        self._sparse_embeddings = FastEmbedSparse(
            model_name=self.config.sparse_model
        )

        self._vector_store = QdrantVectorStore(
            client=self._client,
            collection_name=self.config.collection_name,
            embedding=self._embeddings,
            sparse_embedding=self._sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name=self.config.dense_vector_name,
            sparse_vector_name=self.config.sparse_vector_name,
        )

        # Инициализация реранкера
        self._reranker = None
        if self.config.reranker_model_name:
            self._reranker = self._load_reranker(self.config.reranker_model_name)

    @classmethod
    def from_yaml(cls) -> "HybridRetriever":
        """
        Читает config.yaml и создаёт HybridRetriever.
        """
        config = RetrieverConfig.from_yaml()
        return cls(config)

    @staticmethod
    def _load_reranker(model_name: str) -> CrossEncoder | None:
        """
        Загружает cross-encoder реранкер.
        """
        try:
            from sentence_transformers import CrossEncoder
            log.info(f'Загружен реранкер')
            return CrossEncoder(model_name, device="cuda")
        except ImportError as e:
            log.warning(f"Ошибка при загрузке реранкера: {e}")
            return None

    def search(self, query: str) -> SearchResult:
        """
        Основной метод поиска.
        """
        query = escape_to_latex(query)
        # Гибридный поиск с запасом
        candidates = self._hybrid_search(query, k=self.config.top_k_fetch)

        # Rerank + фильтрация по score_threshold
        top_chunks = self._rerank_and_filter(query, candidates)

        # Расширение групп
        group_chunks = self._expand_groups(top_chunks)

        # Разрешение external links
        mentioned_chunks, mentioned_labels = self._resolve_external_links(top_chunks + group_chunks)

        return SearchResult(
            top_chunks=top_chunks,
            group_chunks=group_chunks,
            mentioned_chunks=mentioned_chunks,
            mentioned_labels=mentioned_labels,
        )

    def _hybrid_search(self, query: str, k: int) -> list[Document]:
        """
        Гибридный поиск.
        """
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.is_searchable",
                    match=MatchValue(value=True),
                )
            ]
        )

        docs = self._vector_store.similarity_search(query, k=k, filter=search_filter)
        for doc in docs:
            point_id = doc.metadata.pop("_id", None) or doc.metadata.get("id", "")
            doc.metadata["id"] = str(point_id)
        return docs

    def _rerank_and_filter(self, query: str, docs: list[Document]) -> list[ChunkResult]:
        """
        Ранжирование чанков с помощью реранкера, если он включен.
        """
        if self._reranker is None:
            return [
                ChunkResult.from_document(d)
                for d in docs[: self.config.top_k_final]
            ]

        pairs = [(query, d.page_content) for d in docs]
        scores: list[float] = self._reranker.predict(pairs).tolist()

        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

        result: list[ChunkResult] = []
        for score, doc in scored:
            if (
                self.config.reranker_score_threshold is not None
                and score < self.config.reranker_score_threshold
            ):
                log.debug(f'Чанк отфильтрован: {score:.3f} < {self.config.reranker_score_threshold}; {doc.metadata.get("id", "")}')
                continue
            result.append(ChunkResult.from_document(doc, rerank_score=score))
            if len(result) >= self.config.top_k_final:
                break

        return result

    def _expand_groups(self, top_chunks: list[ChunkResult]) -> list[ChunkResult]:
        """
        Получение групп чанков, прошедших порог схожести.
        """
        threshold = self.config.group_expand_score_threshold

        eligible = [
            c
            for c in top_chunks
            if threshold is None
            or c.rerank_score is None
            or c.rerank_score >= threshold
        ]

        group_ids = {
            c.metadata.get("parent_id")
            for c in eligible
            if c.metadata.get("parent_id")
        }
        if not group_ids:
            return []

        already_ids = {c.id for c in top_chunks}
        result: list[ChunkResult] = []

        for group_id in group_ids:
            docs = self._scroll_by_filter(
                Filter(
                    must=[
                        FieldCondition(
                            key="metadata.parent_id",
                            match=MatchValue(value=str(group_id)),
                        )
                    ]
                )
            )
            for doc in docs:
                chunk = ChunkResult.from_document(doc)
                if chunk.id not in already_ids:
                    result.append(chunk)
                    already_ids.add(chunk.id)

        return result

    def _resolve_external_links(self, chunks: list[ChunkResult]) -> tuple[list[ChunkResult], dict[str, str]]:
        """
        Получение чанков, на которые есть ссылки в найденных чанках.
        """
        existing_ids = {c.id for c in chunks}
        link_map: dict[str, str] = {}

        for chunk in chunks:
            links = chunk.metadata.get("external_links") or {}
            if isinstance(links, dict):
                for label, uuid_val in links.items():
                    uuid_str = str(uuid_val)
                    if uuid_str and uuid_str not in existing_ids:
                        link_map[uuid_str] = label

        if not link_map:
            return [], {}

        docs = self._fetch_by_ids(list(link_map.keys()))
        mentioned_chunks = [ChunkResult.from_document(d) for d in docs]
        mentioned_labels = {
            c.id: link_map[c.id]
            for c in mentioned_chunks
            if c.id in link_map
        }
        return mentioned_chunks, mentioned_labels

    def _payload_to_document(self, record: Record) -> Document:
        payload = record.payload or {}
        metadata = payload.get("metadata", {})
        return Document(
            page_content=payload.get("page_content", metadata.get("text", "")),
            metadata={**metadata, "id": str(record.id)},
        )

    def _scroll_by_filter(self, scroll_filter: Filter) -> list[Document]:
        docs: list[Document] = []
        offset = None
        while True:
            records, next_offset = self._client.scroll(
                collection_name=self.config.collection_name,
                scroll_filter=scroll_filter,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                docs.append(self._payload_to_document(record))
            if next_offset is None:
                break
            offset = next_offset
        return docs

    def _fetch_by_ids(self, ids: list[str]) -> list[Document]:
        if not ids:
            return []
        records = self._client.retrieve(
            collection_name=self.config.collection_name,
            ids=ids,
            with_payload=True,
            with_vectors=False,
        )
        return [self._payload_to_document(r) for r in records]


unicode_to_latex = {
    # Классы сложности и греческие буквы
    'Ω': r'\\Omega',
    'Θ': r'\\Theta',
    'ω': r'\\omega',
    'θ': r'\\theta',
    'α': r'\\alpha',
    'β': r'\\beta',
    'Δ': r'\\Delta',
    'λ': r'\\lambda',
    'π': r'\\pi',

    # Бесконечность и множества
    '∞': r'\\infty',
    '∈': r'\\in',
    '∉': r'\\notin',
    '⊂': r'\\subset',
    '⊆': r'\\subseteq',
    '∪': r'\\cup',
    '∩': r'\\cap',
    '∅': r'\\emptyset',
    'ℕ': r'\\mathbb{N}',
    'ℤ': r'\\mathbb{Z}',
    'ℝ': r'\\mathbb{R}',

    # Операторы сравнения и арифметика
    '≤': r'\\le',
    '≥': r'\\ge',
    '≠': r'\\neq',
    '≈': r'\\approx',
    '±': r'\\pm',
    '×': r'\\times',
    '÷': r'\\div',
    '√': r'\\sqrt',

    # Логика и битовые операции
    '∧': r'\\land',
    '∨': r'\\lor',
    '¬': r'\\lnot',
    '⊕': r'\\oplus',

    # Стрелки и кванторы
    '→': r'\\to',
    '⇒': r'\\implies',
    '⇔': r'\\iff',
    '∀': r'\\forall',
    '∃': r'\\exists'
}

def escape_to_latex(text: str) -> str:
    translation_table = str.maketrans(unicode_to_latex)
    return text.translate(translation_table)