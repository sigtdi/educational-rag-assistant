from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from time import time
from typing import Dict, List

from app.logger_setup import log
from app.services.document_processing.global_config import GlobalConfig, Paths
from app.services.document_processing.parser.parser_pipeline import PDFParser
from app.services.document_processing.chunking.chunker_processor import ChunkProcessor
from app.services.document_processing.indexing.storage_pipeline import StorageLoader


@dataclass
class DocumentStats:
    """
    Статистика обработки одного документа.
    """
    document_name: str
    start_time: str
    end_time: str | None = None
    total_duration_seconds: float = 0

    # Статусы этапов
    parser_status: str = "skipped"
    chunker_status: str = "skipped"
    loader_status: str = "skipped"

    # Статистика этапов
    parser_stats: Dict  = field(default_factory=dict)
    chunker_stats: Dict = field(default_factory=dict)
    loader_stats: Dict  = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PipelineStats:
    """
    Сводная статистика по всему запуску.
    """
    start_time: str
    end_time: str | None = None
    total_duration_seconds: float = 0

    total_documents: int = 0
    succeeded: int = 0
    failed: int = 0

    documents: List[DocumentStats] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


class ProcessingPipeline:

    def __init__(self, config: GlobalConfig):
        self.config = config
        self.pipeline_stats = PipelineStats(start_time=datetime.now().isoformat())
        log.info("ProcessingPipeline инициализирован")

    @classmethod
    def from_yaml(
        cls,
        yaml_path: Path | None = None,
        document_name: str | None = None,
    ) -> "ProcessingPipeline":
        """
        Читает config.yaml и создаёт оркестратор.
        """
        kwargs = {"document_name": document_name}
        if yaml_path:
            kwargs["yaml_path"] = yaml_path
        config = GlobalConfig.from_yaml(**kwargs)
        return cls(config)


    def run(self) -> None:
        """
        Запуск обработки: один файл или всю папку.
        """
        total_start = time()

        pdf_files = self._collect_files()
        if not pdf_files:
            log.warning(f"PDF-файлы не найдены в {Paths.data}")
            return

        log.info(f"Найдено файлов для обработки: {len(pdf_files)}")
        self.pipeline_stats.total_documents = len(pdf_files)

        for pdf_path in pdf_files:
            self._process_one(pdf_path)

        self.pipeline_stats.end_time = datetime.now().isoformat()
        self.pipeline_stats.total_duration_seconds = round(time() - total_start, 2)

        self._log_summary()

    def get_stats(self) -> PipelineStats:
        return self.pipeline_stats

    def save_stats(self, output_path: Path | None = None) -> Path:
        """
        Сохранение статистики.
        """
        if output_path is None:
            output_path = (
                Paths.stats_out
                / f"pipeline_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.pipeline_stats.to_dict(), f, ensure_ascii=False, indent=2)
        log.info(f"Статистика сохранена: {output_path}")
        return output_path

    def _collect_files(self) -> List[Path]:
        """
        Возвращает список PDF для обработки.
        """
        doc_name = self.config.parser.document_name
        if doc_name:
            pdf_path = Paths.data / doc_name
            if not pdf_path.exists():
                log.error(f"Файл не найден: {pdf_path}")
                return []
            return [pdf_path]

        return sorted(Paths.data.glob("*.pdf"))

    def _process_one(self, pdf_path: Path) -> None:
        """
        Обработка один PDF через все этапы пайплайна.
        """
        doc_name = pdf_path.name
        log.info(f"{'─' * 50}")
        log.info(f"Начало обработки: {doc_name}")
        chunks = None

        doc_stats = DocumentStats(
            document_name=doc_name,
            start_time=datetime.now().isoformat(),
        )
        doc_start = time()


        try:
            stages = self.config.stages
            if stages.run_parser:
                chunks = self._run_parser(doc_name, doc_stats)
            if stages.run_chunker:
                chunks = self._run_chunker(doc_name, doc_stats, chunks)
            if stages.run_loader:
                self._run_loader(doc_name, doc_stats, chunks)

            self.pipeline_stats.succeeded += 1
            log.info(f"Документ обработан успешно: {doc_name}")

        except Exception as e:
            self.pipeline_stats.failed += 1
            log.error(f"Ошибка при обработке {doc_name}: {e}", exc_info=True)

        finally:
            doc_stats.end_time = datetime.now().isoformat()
            doc_stats.total_duration_seconds = round(time() - doc_start, 2)
            self.pipeline_stats.documents.append(doc_stats)

    def _run_parser(self, doc_name: str, doc_stats: DocumentStats):
        log.info(f"[{doc_name}] Запуск парсера")

        # Пересоздаём конфиг с именем конкретного документа
        cfg = GlobalConfig.from_yaml(document_name=doc_name)

        parser = PDFParser(cfg.parser)
        chunks = parser.run()

        parser_stats = parser.get_stats()
        doc_stats.parser_stats = (
            parser_stats.to_dict()
            if hasattr(parser_stats, "to_dict")
            else {}
        )
        doc_stats.parser_status = "done"
        log.info(f"[{doc_name}] Парсер завершен")

        return chunks

    def _run_chunker(self, doc_name: str, doc_stats: DocumentStats, chunks: list | None):
        log.info(f"[{doc_name}] Запуск обработки чанков")

        # Пересоздаём конфиг с именем конкретного документа
        cfg = GlobalConfig.from_yaml(document_name=doc_name)

        chunker = ChunkProcessor(cfg.chunker)
        chunks = chunker.run(chunks=chunks)

        chunker_stats = chunker.get_stats()
        doc_stats.parser_stats = (
            chunker_stats.to_dict()
            if hasattr(chunker_stats, "to_dict")
            else {}
        )
        doc_stats.parser_status = "done"
        log.info(f"[{doc_name}] Обработка чанков завершена")

        return chunks

    def _run_loader(self, doc_name: str, doc_stats: DocumentStats, chunks: list | None):
        log.info(f"[{doc_name}] Запуск загрузчика")

        cfg = GlobalConfig.from_yaml(document_name=doc_name)

        loader = StorageLoader(cfg.loader)
        loader.run(parent_chunks=chunks)

        loader_stats = loader.get_stats()
        doc_stats.loader_stats = (
            loader_stats.to_dict()
            if hasattr(loader_stats, "to_dict")
            else {}
        )
        doc_stats.loader_status = "done"
        log.info(f"[{doc_name}] Загрузчик завершен")

    def _log_summary(self) -> None:
        s = self.pipeline_stats
        log.info(f"{'═' * 50}")
        log.info(f"Обработка завершена")
        log.info(f"  Всего файлов : {s.total_documents}")
        log.info(f"  Успешно      : {s.succeeded}")
        log.info(f"  С ошибками   : {s.failed}")
        log.info(f"  Время        : {s.total_duration_seconds:.1f} с")
        log.info(f"{'═' * 50}")



if __name__ == "__main__":
    # Один файл
    pipeline = ProcessingPipeline.from_yaml(document_name="Lipskij V Kombinatorika dlja programmistov.pdf")
    # Все файлы в папке
    #pipeline = ProcessingPipeline.from_yaml()

    pipeline.run()
    pipeline.save_stats()