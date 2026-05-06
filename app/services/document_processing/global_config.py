from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Any

import yaml

from app.services.document_processing.parser.parser_config import ParserConfig
from app.services.document_processing.indexing.storage_config import StorageConfig
from app.services.document_processing.chunking.chunker_config import ChunkerConfig


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_YAML  = _PROJECT_ROOT / "app" / "config.yaml"

@dataclass
class PipelineStagesConfig:
    run_parser:  bool = True
    run_chunker: bool = True
    run_loader:  bool = True


# Пути
class Paths:
    root    = _PROJECT_ROOT
    data    = _PROJECT_ROOT / "app" / "data"
    output  = _PROJECT_ROOT / "app" / "services" / "document_processing" / "output"
    images  = _PROJECT_ROOT / "app" / "services" / "rag" / "data" / "images"

    # Выходные папки каждого этапа
    marker_out  = output / "output_marker_processor"
    text_out    = output / "output_text_processor"
    image_out   = output / "output_image_processor"
    chunk_out   = output / "output_chunk_processor"
    storage_out = output / "output_storage_preparer"
    stats_out = output / "stats_output"


@dataclass
class GlobalConfig:
    stages:  PipelineStagesConfig
    parser:  ParserConfig
    chunker: ChunkerConfig
    loader:  StorageConfig

    @classmethod
    def from_yaml(
        cls,
        yaml_path: Path = _CONFIG_YAML,
        document_name: str | None = None,
    ) -> "GlobalConfig":
        """
        Читает config.yaml и возвращает GlobalConfig.
        """
        raw = _load_yaml(yaml_path)

        stages_cfg = _build_stages_config(raw.get("pipeline", {}))
        parser_cfg  = _build_parser_config(raw["parser"], document_name)
        chunker_cfg = _build_chunker_config(raw["chunker"], document_name)
        loader_cfg  = _build_loader_config(raw["loader"], document_name)

        return cls(stages=stages_cfg, parser=parser_cfg, chunker=chunker_cfg, loader=loader_cfg)


def _build_stages_config(s: dict[str, Any]) -> PipelineStagesConfig:
    return PipelineStagesConfig(
        run_parser=s.get("run_parser", True),
        run_chunker=s.get("run_chunker", True),
        run_loader=s.get("run_loader", True),
    )

# Сборка дочерних конфигов
def _build_parser_config(s: dict[str, Any], document_name: str | None) -> ParserConfig:
    return ParserConfig(
        document_name=document_name,
        input_dir=Paths.data,
        process_marker=s["process_marker"],
        process_text=s["process_text"],
        process_image=s["process_image"],
        one_step=s["one_step"],
        model_name=s["model_name"],
        text_dir="extracted_formulas_images",
        images_dir="extracted_images",
        marker_processor_output=str(Paths.marker_out),
        text_processor_output=str(Paths.text_out),
        image_processor_output=str(Paths.image_out),
        save_intermediate=s["save_intermediate"],
        delete_images=s["delete_images"],
    )


def _build_chunker_config(s: dict[str, Any], document_name: str | None) -> ChunkerConfig:
    return ChunkerConfig(
        document_name=document_name,
        save_intermediate=s["save_intermediate"],
        has_suffix=True,
        suffix="_image_processed_json.txt",
        input_dir=Paths.image_out,
        output_dir=Paths.chunk_out,
    )


def _build_loader_config(s: dict[str, Any], document_name: str | None) -> StorageConfig:
    return StorageConfig(
        document_name=document_name,
        input_dir=Paths.chunk_out,
        original_doc_dir=Paths.data,
        has_suffix=True,
        suffix="_chunk_processed_json.txt",
        dense_model=s["dense_model"],
        sparse_model=s["sparse_model"],
        batch_size=s["batch_size"],
        qdrant_host=s["qdrant_host"],
        qdrant_port=s["qdrant_port"],
        qdrant_url=s["qdrant_url"],
        collection_name=s["collection_name"],
        dense_vector_name=s["dense_vector_name"],
        sparse_vector_name=s["sparse_vector_name"],
        images_dir=Paths.images,
        storage_preparer_output=str(Paths.storage_out),
        save_intermediate=s["save_intermediate"],
        recreate_db=s["recreate_db"],
        vector_size=s["vector_size"]
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}