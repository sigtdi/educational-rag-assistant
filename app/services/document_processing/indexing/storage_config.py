from pathlib import Path
from dataclasses import dataclass

@dataclass
class StorageConfig:
    """
    Конфигурация парсера.
    """
    document_name: str | None
    input_dir: Path
    original_doc_dir: Path

    # Настройки для определения полного имени файла
    has_suffix: bool # К имени документа нужно добавить суффикс
    suffix: str

    # Настройки эмбеддингов
    dense_model: str
    sparse_model: str
    batch_size: int
    vector_size: int

    # Настройки qdrant
    qdrant_host: str
    qdrant_port: int
    collection_name: str
    qdrant_url: str
    dense_vector_name: str
    sparse_vector_name: str

    # Директории
    images_dir: str | Path
    storage_preparer_output: str

    # Сохранение
    save_intermediate:  bool
    recreate_db: bool