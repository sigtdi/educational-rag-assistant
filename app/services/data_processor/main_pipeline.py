import json
import torch
import gc
from pathlib import Path
from dataclasses import dataclass, asdict
from time import time
from typing import Dict
from datetime import datetime

from processing.marker_processor import MarkerProcessor
from processing.post_processor import TextProcessor
from processing.image_processor import ImageProcessor
from app.logger_setup import log

@dataclass
class PipelineConfig:
    """
    Конфигурация pipeline.
    """
    document_path: str | None = None # При отсутствии будут браться файлы из input_dir
    document_title: str | None = None # Название файла учебника
    input_dir: Path = Path(__file__).resolve().parent / 'data'
    document_name: str | None = None # Название учебника (только для работы с одним файлом)
    document_author: str | None = None # Автор учебника (только для работы с одним файлом)

    # Опции обработки
    process_marker: bool = False # Для False обязательно наличие текстового файла для этого документа в папке output для Marker
    process_text: bool = True
    process_image: bool = False
    # Если опция маркера не выполнена, то первая опция в цепочке
    # true - выполняется на основе данных из папки output для Marker
    # false - выполняется на основе данных из папки output для предыдущего шага (marker - text - image)
    one_step: bool = False

    # Настройки модели
    model_name: str = 'qwen3-vl:8b-instruct'

    # Директории
    text_dir: str = 'extracted_formulas_images'
    images_dir: str = 'extracted_images'
    marker_processor_output: str = "output_marker_processor"
    text_processor_output: str = "output_text_processor"
    image_processor_output: str = "output_image_processor"

    # Сохранение
    save_intermediate: bool = True


@dataclass
class PipelineStats:
    """
    Статистика выполнения pipeline
    """
    document_title: str

    # Время
    start_time: str
    end_time: str | None = None
    total_duration_seconds: float  = 0
    marker_process_time: float = 0
    text_process_time: float = 0
    formulas_process_time: float = 0
    images_process_time: float = 0

    # Статистика по этапам
    marker_status: str = "not done"
    formula_status: str = "not done"
    text_status: str = "not done"
    image_status: str = "not done"

    # Основные метрики
    total_pages: int = 0

    total_chunks_checked_via_vlm: int = 0
    total_corrected_chunks: int = 0
    total_chunks: int = 0
    total_failed_chunks: int = 0

    total_images: int = 0
    described_images: int = 0
    failed_images: int = 0

    def to_dict(self) -> Dict:
        """
        Конвертация в словарь
        """
        return asdict(self)


class PDFProcessingPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config

        if self.config.document_path:
            self.stats = PipelineStats(
                document_title=Path(config.document_path).stem,
                start_time=datetime.now().isoformat()
            )
        else:
            self.stats_list = [] # Список PipelineStats для каждого документа

        self.marker_processor = None
        self.text_processor = None
        self.image_processor = None

        self.steps = ['marker', 'text', 'image'] # Порядок обработки
        self.chunks = []

        log.info(f"Pipeline инициализирован")

    def initialize_processors(self):
        """
        Инициализация процессоров
        """
        log.info("Инициализация процессоров")

        # Инициализируем только нужные процессы
        if self.config.process_marker:
            self.marker_processor = MarkerProcessor(
                output_folder=self.config.marker_processor_output,
                need_output_file=self.config.save_intermediate
            )

        if self.config.process_text:
            self.text_processor = TextProcessor(
                model_name=self.config.model_name,
                output_folder=self.config.text_processor_output,
                need_output_file=self.config.save_intermediate
            )

        if self.config.process_image:
            self.image_processor = ImageProcessor(
                image_folder=self.config.images_dir,
                model_name=self.config.model_name,
                output_folder=self.config.image_processor_output,
                need_output_file=self.config.save_intermediate
            )

        log.info("Процессоры инициализированы")

    def init_chunks(self, document_title):
        """
        Инициализация начального содержания чанков для случая, когда marker_process отключен
        """
        if self.config.process_marker:
            self.chunks = None
            return

        prev_step = self.steps[0]
        for step in self.steps[1:]:
            if getattr(self.config, f'process_{step}'):
                script_dir = Path(__file__).parent
                processor_output_dir = getattr(self.config, f"{prev_step}_processor_output")

                file_path = script_dir / "output" / processor_output_dir / f"{document_title}_{prev_step}_processed_json.txt"
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.chunks = json.load(f)
                    log.info(f"Успешно загружен файл с чанками для старта обработки: {file_path}")
                except FileNotFoundError:
                    log.error(f"Ошибка: файл не найден по пути {file_path}")
                except json.JSONDecodeError:
                    log.error(f"Ошибка: некорректный JSON в файле {file_path}")

                return
            prev_step = step

    def process_one_document(self, document_path: str | None, stats: PipelineStats):
        """
        Обработка одного документа: запуск всех выбранных опций и сохранение статистики
        """
        start_time = time()
        self.init_chunks(Path(document_path).stem)
        stats_dict = {}

        for step in self.steps:
            if getattr(self.config, f'process_{step}'):
                processor = getattr(self, f'{step}_processor')
                if self.chunks is None:
                    self.chunks = processor.process(document_path)
                else:
                    self.chunks = processor.process(chunks=self.chunks, document_path=document_path)

                setattr(stats, f'{step}_status', 'done')

                stats_dict[step] = processor.get_stats()

        stats.end_time = datetime.now().isoformat()
        stats.total_duration_seconds = time() - start_time
        stats.marker_process_time = stats_dict.get('marker', {}).get('total_time', 0)
        stats.text_process_time = stats_dict.get('text', {}).get('total_time', 0)
        stats.images_process_time = stats_dict.get('image', {}).get('total_time', 0)

        stats.total_pages = stats_dict.get('marker', {}).get('total_pages', 0)

        stats.total_chunks_checked_via_vlm = stats_dict.get('text', {}).get('total_chunks_checked_via_vlm', 0)
        stats.total_corrected_chunks = stats_dict.get('text', {}).get('total_corrected_chunks', 0)
        stats.total_chunks = stats_dict.get('text', {}).get('total_chunks', 0)
        stats.total_failed_chunks = stats_dict.get('text', {}).get('total_failed_chunks', 0)

        stats.total_images = stats_dict.get('formula', {}).get('total_images_count', 0)
        stats.described_images = stats_dict.get('formula', {}).get('described_images', 0)
        stats.failed_images = stats_dict.get('formula', {}).get('failed_images_count', 0)

    def get_stats(self):
        if self.stats:
            return self.stats
        return self.stats_list

    def run(self):
        self.initialize_processors()
        if self.config.document_path:
            self.process_one_document(document_path=self.config.document_path, stats=self.stats)
        else:
            self.process_one_document(document_path='ff', stats=self.stats_list[-1])


if __name__ == "__main__":
    p = PipelineConfig(document_path="C:/Users/Yana/Downloads/Alg-graphs-full.pdf")
    parser = PDFProcessingPipeline(p)
    parser.run()
    print(parser.get_stats())