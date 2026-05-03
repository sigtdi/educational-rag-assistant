from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Any

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_CONFIG_YAML  = _PROJECT_ROOT / "app" / "config.yaml"


# Пути
class Paths:
    root   = _PROJECT_ROOT
    output = _PROJECT_ROOT / "app" / "services" / "document_processing" / "output"
    images = _PROJECT_ROOT / "app" / "services" / "rag" / "data" / "images"


@dataclass
class RetrieverConfig:
    # Qdrant
    qdrant_url:        str
    collection_name:   str
    dense_vector_name: str
    sparse_vector_name: str

    # Эмбеддинги
    dense_model: str
    sparse_model: str
    embedding_query_instruction: str

    # Reranker
    reranker_model_name:      str | None
    reranker_score_threshold: float | None

    # Пути
    images_dir: str

    # Поиск
    top_k_fetch: int
    top_k_final: int

    # Фильтрация групп
    group_expand_score_threshold: float | None

    @classmethod
    def from_yaml(cls, yaml_path: Path = _CONFIG_YAML) -> "RetrieverConfig":
        """
        Читает config.yaml и возвращает RetrieverConfig.
        """
        raw = _load_yaml(yaml_path)
        return _build_retriever_config(raw["retriever"])


def _build_retriever_config(s: dict[str, Any]) -> RetrieverConfig:
    # Если задан qdrant_url — используем его, иначе собираем из host:port
    qdrant_url = s.get("qdrant_url") or \
        f"http://{s['qdrant_host']}:{s['qdrant_port']}"

    return RetrieverConfig(
        qdrant_url=qdrant_url,
        collection_name=s["collection_name"],
        dense_vector_name=s["dense_vector_name"],
        sparse_vector_name=s["sparse_vector_name"],
        dense_model=s["dense_model"],
        sparse_model=s['sparse_model'],
        embedding_query_instruction=s.get("embedding_query_instruction", ""),
        reranker_model_name=s.get("reranker_model_name"),
        reranker_score_threshold=s.get("reranker_score_threshold"),
        images_dir=str(Paths.images),
        top_k_fetch=s["top_k_fetch"],
        top_k_final=s["top_k_final"],
        group_expand_score_threshold=s.get("group_expand_score_threshold"),
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}