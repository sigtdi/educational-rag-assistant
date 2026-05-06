import re
import html
import json
from typing import Any

import torch
import gc
from pathlib import Path
from ftfy import fix_text
from tqdm import tqdm
from bs4 import BeautifulSoup
from time import time

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser

from app.logger_setup import log


class MarkerProcessor:
    def __init__(
            self,
            output_folder: str,
            need_output_file: bool
    ):
        if need_output_file:
            self.output_folder = Path(__file__).resolve().parent.parent / 'output' / output_folder
            self.output_folder.mkdir(exist_ok=True, parents=True)

        self.rendered_text = None
        self.final_chunks = []

        self.document_data = {}
        self.need_output_file = need_output_file
        self.document_base_name = None
        self._model_lst = None

    @log.catch
    def process(self, document_path: str | Path) -> list[dict[str, Any]]:
        """
        Основная функция обработки файла через Marker.
        """
        start_time = time()
        self.rendered_text = None
        self.final_chunks = []
        self.document_base_name = Path(document_path).stem
        self._load_marker()

        log.info(f"Обработка файла {document_path}")

        # Настройки конфигурации Marker и запуск обработки
        config = {
            "output_format": "chunks",
            "force_ocr": True,
            "image_layer": True,
            "highres_images": True,
        }

        config_parser = ConfigParser(config)
        renderer_cls_string = config_parser.get_renderer()
        full_config = config_parser.generate_config_dict()
        converter = PdfConverter(
            artifact_dict=self._model_lst,
            config=full_config,
            renderer=renderer_cls_string
        )

        self.rendered_text = converter(str(document_path))

        # Обработка парсинга Marker - извлечение метаданных, приведение текста к удобному виду, исправление кодировки
        self._processing_chunks()
        self._fix_encoding_errors()

        log.info(f"Предварительная обработка файла {document_path} завершена")

        # Сохранение данных в файл
        self._save_final_document()

        self._update_document_data(time() - start_time)
        self._clear_marker_memory()

        return self.final_chunks
    
    def get_stats(self) -> dict:
        return self.document_data

    def _load_marker(self):
        """
        Загрузка данных marker.
        """
        self._model_lst = create_model_dict()
        log.info("Marker загружен")

    def _clear_marker_memory(self):
        """
        Полностью освобождает память моделей marker
        """

        if hasattr(self, "model_lst") and self._model_lst is not None:
            for model in self._model_lst.values():
                del model

            self._model_lst.clear()
            self._model_lst = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _fix_encoding_errors(self):
        """
        Исправление ошибок кодировки.
        """
        for chunk in tqdm(self.final_chunks, "Обработка ошибок кодировки"):
            chunk['text'] = fix_text(chunk['text'])

    def _processing_chunks(self):
        """
        Приведение математики и структуры чанков к более удобному виду.
        """
        # Приводим html <math> блоки к LaTex формату
        pattern = r'<math\s+display="(inline|block)"[^>]*>(.*?)</math>'
        header_pattern = re.compile(r'^(Задача|Упражнение|Упражнения)\s*\d+(\.\d+)?')
        chunk_id = 0
        is_ex = False
        
        log.info('Обработка полученных чанков: изменение структуры, удаление лишних, улучшение читаемости')
        for block in tqdm(self.rendered_text.blocks, f"Преобразование чанков"):
            chunk = {}
            page= block.id.split('/')[2]
            chunk['id'] = page + '-' + str(chunk_id)
            chunk['block_type'] = block.block_type
            chunk['page'] = page
            chunk['bbox'] = block.bbox

            def replacement_function(match):
                display_type = match.group(1)
                content = match.group(2)

                # Обрабатываем бэклеши для корректного представления строк через Python и JSON
                content = html.unescape(content)
                content = re.sub(r'\\\\(?=\s*\\begin|\\\\| \s*$)', '[[DOUBLE_SLASH]]', content)
                content = content.replace(r"\\", "\\")
                content = content.replace('[[DOUBLE_SLASH]]', '\\\\')
                cleaned_content = content.strip()

                if display_type == 'block':
                    return f"$${cleaned_content}$$"
                else:
                    return f"${cleaned_content}$"

            chunk['text'] = re.sub(pattern, replacement_function, block.html, flags=re.DOTALL)
            chunk['text'] = BeautifulSoup(chunk['text'], "html.parser").get_text()

            # Пропускаем чанки с задачами и упражнениями
            if is_ex and (chunk['block_type'] != 'SectionHeader' or header_pattern.match(chunk['text'].strip())):
                continue
            is_ex = False

            # Удаляем чанки без смысловой нагрузки
            if chunk['block_type'] in {'PageHeader', 'PageFooter', 'Footnote'}:
                continue

            # Определяем начало чанков с задачами и упражнениями
            if (chunk['block_type'] == 'SectionHeader'
                    and any(word in chunk['text'].lower() for word in ['задачи', 'упражнения', 'библиографические примечания'])
                    and len(chunk['text']) <= 15):
                is_ex = True
                continue

            # Пропускаем слишком короткие чанки
            if len(chunk['text']) <= 3:
                continue

            chunk_id += 1
            self.final_chunks.append(chunk)

    def _update_document_data(self, total_time: float):
        self.document_data = {
            'total_pages': len(self.rendered_text.metadata['page_stats']),
            'total_chunks': len(self.final_chunks),
            'result_document_name': f"{self.document_base_name}_marker_processed_json.txt",
            'need_save': self.need_output_file,
            'total_time': total_time
        }

    def _save_final_document(self):
        """
        Сохранения результата обработки документа.
        """
        if not self.need_output_file:
            return

        output_path = self.output_folder / f"{self.document_base_name}_marker_processed_json.txt"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.final_chunks, f, ensure_ascii=False, indent=4)

        log.info(f"Результат предварительной обработки сохранен в {output_path}")
