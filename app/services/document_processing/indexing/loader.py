from time import time
from tqdm import tqdm

from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from app.logger_setup import log

class QdrantLoader:
    def __init__(
        self,
        qdrant_url: str,
        collection_name: str,
        dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        sparse_model_name: str = "Qdrant/bm25",
        dense_vector_name: str = "fast-all-minilm-l6-v2",
        sparse_vector_name: str = "fast-sparse-bm25",
        batch_size: int = 100,
    ):
        self.collection_name = collection_name
        self.batch_size = batch_size

        self.client = QdrantClient(url=qdrant_url)

        self.dense_embeddings = FastEmbedEmbeddings(model_name=dense_model_name)
        self.sparse_embeddings = FastEmbedSparse(model_name=sparse_model_name)
        self.dense_vector_name = dense_vector_name
        self.sparse_vector_name = sparse_vector_name

        self.total_time = 0

    def ensure_collection(self, vector_size: int = 384, recreate: bool = False):
        """
        Создаёт коллекцию если её нет. При recreate=True пересоздаёт.
        """
        exists = self.client.collection_exists(self.collection_name)

        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    self.dense_vector_name: VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    self.sparse_vector_name: SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                },
            )

    def get_stats(self):
        return {'total_time': self.total_time}

    def load(self, chunks: list[dict], recreate: bool = False):
        """
        Векторизует и загружает чанки в Qdrant.
        """
        start_time = time()
        self.ensure_collection(recreate=recreate)

        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.dense_embeddings,
            sparse_embedding=self.sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name=self.dense_vector_name,
            sparse_vector_name=self.sparse_vector_name,
        )

        for i in tqdm(range(0, len(chunks), self.batch_size), 'Векторизация и загрузка чанков'):
            batch = chunks[i : i + self.batch_size]
            documents, ids = self._to_documents(batch)
            vector_store.add_documents(documents=documents, ids=ids)

        self.total_time = time() - start_time

    def _to_documents(self, chunks: list[dict]) -> tuple[list[Document], list[str]]:
        """
        Преобразует чанки в langchain Document.
        """
        documents = []
        ids = []

        for chunk in chunks:
            metadata = {
                "type":           chunk.get("type", "chunk"),
                "section_path":   chunk.get("section_path", ""),
                "block_type":     chunk.get("block_type", ""),
                "parent_id":      chunk.get("parent_group_id", ""),
                "internal_links": chunk.get("internal_links", {}),
                "external_links": chunk.get("external_links", {}),
                "page":           chunk.get("page"),
                "bbox":           chunk.get("bbox"),
                "file_name":      chunk.get("file_name", ""),
                "text":           chunk.get("text", ""),
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