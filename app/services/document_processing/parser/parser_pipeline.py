import json
from pathlib import Path
from dataclasses import dataclass, asdict
from time import time
from typing import Dict
from datetime import datetime

from app.services.document_processing.parser.marker_processor import MarkerProcessor
from app.services.document_processing.parser.text_processor import TextProcessor
from app.services.document_processing.parser.image_processor import ImageProcessor
from app.services.document_processing.parser.parser_config import ParserConfig
from app.logger_setup import log


@dataclass
class ParserStats:
    """
    Статистика выполнения пайплайна парсера
    """
    document_title: str

    # Время
    start_time: str
    end_time: str | None = None
    total_duration_seconds: float  = 0
    marker_process_time: float = 0
    text_process_time: float = 0
    images_process_time: float = 0

    # Статистика по этапам
    marker_status: str = "not done"
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


class PDFParser:
    def __init__(self, config: ParserConfig):
        self.config = config

        self.stats = ParserStats(
            document_title=Path(config.document_name).stem,
            start_time=datetime.now().isoformat()
        )

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
                need_output_file=self.config.save_intermediate,
                delete_images=self.config.delete_images
            )

        if self.config.process_image:
            self.image_processor = ImageProcessor(
                image_folder=self.config.images_dir,
                model_name=self.config.model_name,
                output_folder=self.config.image_processor_output,
                need_output_file=self.config.save_intermediate,
                delete_images=self.config.delete_images
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
                script_dir = Path(__file__).parent.parent
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

    def get_stats(self):
        return self.stats

    def run(self):
        self.initialize_processors()
        document_path = self.config.input_dir / self.config.document_name

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

                setattr(self.stats, f'{step}_status', 'done')

                stats_dict[step] = processor.get_stats()

        self.stats.end_time = datetime.now().isoformat()
        self.stats.total_duration_seconds = time() - start_time
        self.stats.marker_process_time = stats_dict.get('marker', {}).get('total_time', 0)
        self.stats.text_process_time = stats_dict.get('text', {}).get('total_time', 0)
        self.stats.images_process_time = stats_dict.get('image', {}).get('total_time', 0)

        self.stats.total_pages = stats_dict.get('marker', {}).get('total_pages', 0)
        self.stats.total_chunks = stats_dict.get('marker', {}).get('total_chunks', 0)

        self.stats.total_chunks_checked_via_vlm = stats_dict.get('text', {}).get('total_chunks_checked_via_vlm', 0)
        self.stats.total_corrected_chunks = stats_dict.get('text', {}).get('total_corrected_chunks', 0)
        self.stats.total_failed_chunks = stats_dict.get('text', {}).get('total_failed_chunks', 0)

        self.stats.total_images = stats_dict.get('image', {}).get('total_images_count', 0)
        self.stats.described_images = stats_dict.get('image', {}).get('described_images', 0)
        self.stats.failed_images = stats_dict.get('image', {}).get('failed_images_count', 0)

        return self.chunks