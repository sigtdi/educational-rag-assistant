from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from qdrant_client.models import Record

logger = logging.getLogger(__name__)

@dataclass
class RetrieverConfig:
    # Qdrant
    qdrant_url: str
    collection_name: str
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "sparse"

    # Embedding model
    embedding_model_name: str = "intfloat/multilingual-e5-large-instruct"
    embedding_query_instruction: str = (
        "Instruct: Найди релевантные фрагменты учебника по алгоритмам\nQuery: "
    )

    # Paths
    images_dir: str = "rag/data/images"

    # Search
    top_k: int = 5


@dataclass
class ChunkResult:
    id: str
    text: str
    metadata: dict[str, Any]
    is_picture: bool = False
    image_path: str | None = None

    @classmethod
    def from_document(cls, doc: Document, images_dir: Path) -> "ChunkResult":
        chunk_id = doc.metadata.get("id", "")
        is_picture = doc.metadata.get("type") == "picture"
        image_path = None
        if is_picture and chunk_id:
            candidate = images_dir / f"{chunk_id}.png"
            image_path = str(candidate) if candidate.exists() else str(candidate)
        return cls(
            id=chunk_id,
            text=doc.page_content,
            metadata=doc.metadata,
            is_picture=is_picture,
            image_path=image_path,
        )


@dataclass
class SearchResult:
    top_chunks: list[ChunkResult]
    group_chunks: list[ChunkResult]
    mentioned_chunks: list[ChunkResult]
    mentioned_labels: dict[str, str] = field(default_factory=dict)
    # mentioned_labels: chunk_id -> подпись ("Определение 6" и т.п.)

class HybridRetriever:
    """
    Гибридный поиск (BM25 sparse + dense vectors) через Qdrant + LangChain.
    """

    def __init__(self, config: RetrieverConfig) -> None:
        self.config = config
        self.images_dir = Path(config.images_dir)

        self._client = QdrantClient(url=config.qdrant_url)

        self._embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model_name,
            encode_kwargs={"normalize_embeddings": True},
            query_instruction=config.embedding_query_instruction,
            embed_instruction="",
        )

        self._sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

        self._vector_store = QdrantVectorStore(
            client=self._client,
            collection_name=config.collection_name,
            embedding=self._embeddings,
            sparse_embedding=self._sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name=config.dense_vector_name,
            sparse_vector_name=config.sparse_vector_name,
        )

    def search(self, query: str) -> SearchResult:
        """Основной метод поиска."""
        top_docs = self._hybrid_search(query)
        top_chunks = [ChunkResult.from_document(d, self.images_dir) for d in top_docs]

        group_chunks = self._expand_groups(top_chunks)
        mentioned_chunks, mentioned_labels = self._resolve_external_links(
            top_chunks + group_chunks
        )

        return SearchResult(
            top_chunks=top_chunks,
            group_chunks=group_chunks,
            mentioned_chunks=mentioned_chunks,
            mentioned_labels=mentioned_labels,
        )

    def _hybrid_search(self, query: str) -> list[Document]:
        """
        Гибридный поиск.
        """
        docs = self._vector_store.similarity_search(query, k=self.config.top_k)
        for doc in docs:
            # langchain-qdrant >= 0.1.3 пишет id под ключом "_id"
            point_id = doc.metadata.pop("_id", None) or doc.metadata.get("id", "")
            doc.metadata["id"] = str(point_id)
        return docs

    def _expand_groups(self, top_chunks: list[ChunkResult]) -> list[ChunkResult]:
        """
        Дозапрашивает все чанки групп, к которым принадлежат top_chunks.
        """
        group_ids = {
            c.metadata.get("parent_id")
            for c in top_chunks
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
                chunk = ChunkResult.from_document(doc, self.images_dir)
                if chunk.id not in already_ids:
                    result.append(chunk)
                    already_ids.add(chunk.id)

        return result

    def _resolve_external_links(
        self, chunks: list[ChunkResult]
    ) -> tuple[list[ChunkResult], dict[str, str]]:
        """
        Собирает external_links из всех переданных чанков и дозапрашивает их.
        """
        existing_ids = {c.id for c in chunks}
        link_map: dict[str, str] = {}  # uuid -> подпись

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
        mentioned_chunks = [ChunkResult.from_document(d, self.images_dir) for d in docs]
        mentioned_labels = {
            c.id: link_map[c.id]
            for c in mentioned_chunks
            if c.id in link_map
        }

        return mentioned_chunks, mentioned_labels

    def _payload_to_document(self, record: Record) -> Document:
        """
        Конвертирует qdrant Record в LangChain Document.

        LangChain сохраняет данные так:
          payload["page_content"] — текст для поиска (search_text)
          payload["metadata"]     — все остальные поля из _to_documents()
        """
        payload = record.payload or {}
        metadata = payload.get("metadata", {})
        return Document(
            page_content=payload.get("page_content", metadata.get("text", "")),
            metadata={**metadata, "id": str(record.id)},
        )

    def _scroll_by_filter(self, scroll_filter: Filter) -> list[Document]:
        """Листает все точки коллекции по фильтру."""
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
        """Получает чанки по списку UUID."""
        if not ids:
            return []

        records = self._client.retrieve(
            collection_name=self.config.collection_name,
            ids=ids,
            with_payload=True,
            with_vectors=False,
        )
        return [self._payload_to_document(r) for r in records]