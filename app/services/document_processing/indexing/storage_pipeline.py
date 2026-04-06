import json
from pathlib import Path
from dataclasses import dataclass, asdict
from time import time
from typing import Dict
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

        self.preparer = None
        self.loader = None

        self.parent_chunks = []

        log.info(f"Pipeline загрузчика инициализирован")

    def initialize_processors(self):
        """
        Инициализация процессоров
        """
        log.info("Инициализация классов")

        self.preparer = ChunkStoragePreparer(
            output_folder=self.config.storage_preparer_output,
            image_folder=self.config.images_dir,
            need_output_file=self.config.save_intermediate
        )

        self.loader = QdrantLoader(
            qdrant_url=self.config.qdrant_url,
            collection_name=self.config.collection_name,
            dense_model_name=self.config.dense_model,
            sparse_model_name=self.config.sparse_model,
            dense_vector_name=self.config.dense_vector_name,
            sparse_vector_name=self.config.sparse_vector_name,
            batch_size=self.config.batch_size,
        )

        log.info("Классы подготовки и загрузки данных инициализированы")

    def init_chunks(self, document_name, parent_chunks):
        """
        Инициализация чанков.
        """
        if parent_chunks:
            self.parent_chunks = parent_chunks
            return

        # Определяем путь, учитывая суффикс
        suffix = self.config.suffix if (self.config.has_suffix and self.config.document_name) else ''
        file_path = self.config.input_dir / (str(Path(document_name).stem) + suffix)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.parent_chunks = json.load(f)
            log.info(f"Успешно загружен файл с чанками для старта обработки: {file_path}")
        except FileNotFoundError:
            log.error(f"Ошибка: файл не найден по пути {file_path}")
        except json.JSONDecodeError:
            log.error(f"Ошибка: некорректный JSON в файле {file_path}")

    def get_stats(self):
        return self.stats

    @log.catch
    def run(self, parent_chunks: list | None = None):
        self.initialize_processors()

        document_path = self.config.input_dir / self.config.document_name
        orig_doc_path = self.config.original_doc_dir / self.config.document_name

        start_time = time()
        self.init_chunks(Path(document_path).stem, parent_chunks)

        chunks = self.preparer.process(parent_chunks=self.parent_chunks, document_path=orig_doc_path)
        self.loader.load(chunks=chunks, recreate=self.config.recreate_db)

        self.stats.end_time = datetime.now().isoformat()
        self.stats.total_duration_seconds = time() - start_time
        preparer_stats = self.preparer.get_stats()
        loader_stats = self.loader.get_stats()

        self.stats.preparer_time = preparer_stats.get('total_time', 0)
        self.stats.loader_time = loader_stats.get('total_time', 0)

        self.stats.total_chunks = preparer_stats.get('total_chunks', 0)

