from pathlib import Path
from dataclasses import dataclass

@dataclass
class StorageConfig:
    """
    Конфигурация парсера.
    """
    document_name: str | None = None
    input_dir: Path = Path(__file__).resolve().parent.parent / 'output' / 'output_chunk_processor'

    parent_chunks_list: list = None # При указании данные не будут загружаться из файла, но имя все равно нужно указать

    # Настройки для определения полного имени файла
    has_suffix: bool = True # К имени документа нужно добавить суффикс, не работает при обработке множества файлов
    suffix: str = '_chunk_processed_json.txt'

    # Настройки эмбеддингов
    dense_model = "sentence-transformers/all-MiniLM-L6-v2"
    sparse_model = "Qdrant/bm25"
    batch_size: int = 100

    # Настройки qdrant
    qdrant_host = "localhost"
    qdrant_port = 6333
    collection_name = "chunks"
    qdrant_url = ''
    dense_vector_name = "fast-all-minilm-l6-v2"
    sparse_vector_name = "fast-sparse-bm25"

    # Директории
    images_dir: str | Path = Path(__file__).resolve().parents[3] / "rag" / "data" / "images"
    storage_preparer_output: str = "output_marker_processor"

    # Сохранение
    save_intermediate:  bool = True
    recreate_db: bool = False