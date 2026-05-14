from time import time
from typing import Any

from tqdm import tqdm

from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams

from app.logger_setup import log

class QdrantLoader:
    def __init__(
        self,
        qdrant_url: str,
        collection_name: str,
        dense_model_name: str,
        sparse_model_name: str,
        dense_vector_name: str,
        sparse_vector_name: str,
        batch_size: int,
        vector_size: int,
        recreate: bool
    ):
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.vector_size = vector_size

        self._client = QdrantClient(url=qdrant_url)

        self._dense_embeddings = HuggingFaceEmbeddings(
            model_name=dense_model_name,
            model_kwargs={"device": "cuda"},
        )
        self._sparse_embeddings = FastEmbedSparse(model_name=sparse_model_name)
        self._dense_vector_name = dense_vector_name
        self._sparse_vector_name = sparse_vector_name

        self.total_time = 0

        self._ensure_collection(vector_size=self.vector_size, recreate=recreate)

    @log.catch
    def load(self, chunks: list[dict[str, Any]]) -> None:
        """
        Векторизует и загружает чанки в Qdrant.
        """
        start_time = time()

        vector_store = QdrantVectorStore(
            client=self._client,
            collection_name=self.collection_name,
            embedding=self._dense_embeddings,
            sparse_embedding=self._sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name=self._dense_vector_name,
            sparse_vector_name=self._sparse_vector_name,
        )

        log.info('Выполняется векторизация и загрузка чанков в базу данных')
        for i in tqdm(range(0, len(chunks), self.batch_size), 'Векторизация и загрузка чанков'):
            batch = chunks[i : i + self.batch_size]
            documents, ids = self._to_documents(batch)
            vector_store.add_documents(documents=documents, ids=ids)

        log.info('Чанки векторизованы и загружены')
        self.total_time = time() - start_time
        
    def get_stats(self) -> dict:
        return {'total_time': self.total_time}

    def _ensure_collection(self, vector_size: int, recreate: bool):
        """
        Создаёт коллекцию если её нет. При recreate=True пересоздаёт.
        """
        exists = self._client.collection_exists(self.collection_name)

        if exists and recreate:
            self._client.delete_collection(self.collection_name)
            exists = False
            log.info(f'Существующая коллекция {self.collection_name} удалена')

        if not exists:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    self._dense_vector_name: VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    self._sparse_vector_name: SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                },
            )
            log.info(f'Коллекция {self.collection_name} создана')

    @staticmethod
    def _to_documents(chunks: list[dict]) -> tuple[list[Document], list[str]]:
        """
        Преобразует чанки в langchain Document.
        """
        documents = []
        ids = []

        for chunk in chunks:
            metadata = {
                "type":             chunk.get("type", "chunk"),
                "section_path":     chunk.get("section_path", ""),
                "inner_id":         chunk.get("inner_id", 0),
                "parent_id":        chunk.get("parent_group_id", ""),
                "internal_links":   chunk.get("internal_links", {}),
                "external_links":   chunk.get("external_links", {}),
                "page":             chunk.get("page"),
                "bbox":             chunk.get("bbox"),
                "file_name":        chunk.get("file_name", ""),
                "text":             chunk.get("text", ""),
                "semantic_tag":     chunk.get("semantic_tag", ""),
            }

            if "image_path" in chunk:
                metadata["image_path"] = chunk["image_path"]

            doc = Document(
                page_content=chunk.get("search_text", chunk.get("text", "")),
                metadata=metadata,
            )

            documents.append(doc)
            ids.append(chunk["id"])

        return documents, ids