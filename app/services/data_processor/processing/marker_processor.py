import re
import html
import json
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
            output_folder:      str     = "output_marker_processor",
            need_output_file:   bool    = True
    ):
        if need_output_file:
            self.output_folder = Path(__file__).resolve().parent.parent / 'output' / output_folder
            self.output_folder.mkdir(exist_ok=True, parents=True)

        self.rendered_text = None
        self.final_chunks = []

        self.document_data = {}
        self.need_output_file = need_output_file
        self.document_base_name = None

        self.model_lst = create_model_dict()
        log.info("Marker загружен")

    @log.catch
    def run_marker(self, document_path: str):
        """
        Основная функция обработки файла через Marker.
        """
        start_time = time()
        self.rendered_text = None
        self.final_chunks = []
        self.document_base_name = Path(document_path).stem

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
            artifact_dict=self.model_lst,
            config=full_config,
            renderer=renderer_cls_string
        )

        self.rendered_text = converter(document_path)

        # Обработка парсинга Marker - извлечение метаданных, приведение текста к удобному виду, исправление кодировки
        self.processing_chunks()
        self.fix_encoding_errors()


        log.info(f"Предварительная обработка файла {document_path} завершена")

        # Сохранение данных в файл
        if self.need_output_file:
            self.save_final_document()

        self.update_document_data(time() - start_time)

        return self.final_chunks

    def fix_encoding_errors(self):
        """
        Исправление ошибок кодировки.
        """
        for chunk in tqdm(self.final_chunks, "Обработка ошибок кодировки"):
            chunk['text'] = fix_text(chunk['text'])

    def processing_chunks(self):
        """
        Приведение математики и структуры чанков к более удобному виду.
        """
        # Приводим html <math> блоки к LaTex формату
        pattern = r'<math\s+display="(inline|block)"[^>]*>(.*?)</math>'
        chunk_id = 0

        for block in tqdm(self.rendered_text.blocks, f"Преобразование чанков"):
            chunk = {}
            page= block.id.split('/')[2]
            chunk['id'] = page + '-' + str(chunk_id)
            chunk['block_type'] = block.block_type
            chunk['page'] = page
            chunk['bbox'] = block.bbox

            # Удаляем чанки без смысловой нагрузки
            if chunk['block_type'] == 'PageHeader' or chunk['block_type'] == 'PageFooter':
                continue
            if chunk['block_type'] == 'Equation':
                chunk['text'] = '$$' + block.html + '$$'


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

            chunk_id += 1
            self.final_chunks.append(chunk)

    def update_document_data(self, total_time):
        self.document_data = {
            'total_pages': len(self.rendered_text.metadata['page_stats']),
            'result_document_name': f"{self.document_base_name}_marker_processed_json.txt",
            'need_save': self.need_output_file,
            'total_time': total_time
        }

    def get_stats(self):
        return self.document_data

    def save_final_document(self):
        """
        Сохранения результата обработки документа.
        """
        output_path = self.output_folder / f"{self.document_base_name}_marker_processed_json.txt"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.final_chunks, f, ensure_ascii=False, indent=4)

        log.info(f"Результат предварительной обработки сохранен в {output_path}")
