import json
from pathlib import Path
from dataclasses import dataclass, asdict
from time import time
from typing import Dict, Any
from datetime import datetime

from app.services.document_processing.indexing.storage_config import StorageConfig
from app.services.document_processing.indexing.storage_preparer import ChunkStoragePreparer
from app.services.document_processing.indexing.loader import QdrantLoader
from app.logger_setup import log


@dataclass
class StorageStats:
    """
    Статистика выполнения пайплайна парсера
    """
    document_title: str

    # Время
    start_time: str
    end_time: str | None = None
    total_duration_seconds: float  = 0
    preparer_time: float = 0
    loader_time: float = 0

    # Основные метрики
    total_chunks: int = 0

    def to_dict(self) -> Dict:
        """
        Конвертация в словарь
        """
        return asdict(self)


class StorageLoader:
    def __init__(self, config: StorageConfig):
        self.config = config

        self.stats = StorageStats(
            document_title=Path(config.document_name).stem,
            start_time=datetime.now().isoformat()
        )

        self._preparer = None
        self._loader = None

        self._parent_chunks = []

        log.info(f"Pipeline загрузчика инициализирован")
        
    @log.catch
    def run(self, parent_chunks: list[dict[str, Any]] | None = None) -> None:
        self._initialize_processors()

        document_path = self.config.input_dir / self.config.document_name
        orig_doc_path = self.config.original_doc_dir / self.config.document_name

        start_time = time()
        self._init_chunks(Path(document_path).stem, parent_chunks)

        chunks = self._preparer.process(parent_chunks=self._parent_chunks, document_path=orig_doc_path)
        self._loader.load(chunks=chunks)

        self.stats.end_time = datetime.now().isoformat()
        self.stats.total_duration_seconds = time() - start_time
        preparer_stats = self._preparer.get_stats()
        loader_stats = self._loader.get_stats()

        self.stats.preparer_time = preparer_stats.get('total_time', 0)
        self.stats.loader_time = loader_stats.get('total_time', 0)

        self.stats.total_chunks = preparer_stats.get('total_chunks', 0)
        
    def get_stats(self) -> StorageStats:
        return self.stats

    def _initialize_processors(self):
        """
        Инициализация процессоров
        """
        log.info("Инициализация классов")

        self._preparer = ChunkStoragePreparer(
            output_folder=self.config.storage_preparer_output,
            image_folder=self.config.images_dir,
            need_output_file=self.config.save_intermediate
        )

        self._loader = QdrantLoader(
            qdrant_url=self.config.qdrant_url,
            collection_name=self.config.collection_name,
            dense_model_name=self.config.dense_model,
            sparse_model_name=self.config.sparse_model,
            dense_vector_name=self.config.dense_vector_name,
            sparse_vector_name=self.config.sparse_vector_name,
            batch_size=self.config.batch_size,
            vector_size=self.config.vector_size,
            recreate=self.config.recreate_db
        )

        log.info("Классы подготовки и загрузки данных инициализированы")

    def _init_chunks(self, document_name: str | Path, parent_chunks: list[dict[str, Any]]):
        """
        Инициализация чанков.
        """
        if parent_chunks:
            self._parent_chunks = parent_chunks
            return

        # Определяем путь, учитывая суффикс
        suffix = self.config.suffix if (self.config.has_suffix and self.config.document_name) else ''
        file_path = self.config.input_dir / (str(Path(document_name).stem) + suffix)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self._parent_chunks = json.load(f)
            log.info(f"Успешно загружен файл с чанками для старта обработки: {file_path}")
        except FileNotFoundError:
            log.error(f"Ошибка: файл не найден по пути {file_path}")
        except json.JSONDecodeError:
            log.error(f"Ошибка: некорректный JSON в файле {file_path}")
