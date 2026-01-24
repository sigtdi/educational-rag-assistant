import re
import html
import json
from pathlib import Path
from ftfy import fix_text
from tqdm import tqdm
from bs4 import BeautifulSoup

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser

from app.logger_setup import log


class MarkerParser:
    def __init__(self, output_folder: str = "output_marker", need_output_file: bool = True):
        if need_output_file:
            self.output_folder = Path(__file__).resolve().parent.parent / output_folder
            self.output_folder.mkdir(exist_ok=True, parents=True)
            log.info(f"Создана папка {self.output_folder} для выгрузки результата работы Marker")

        self.rendered_text = None
        self.final_chunks = []

        self.model_lst = create_model_dict()
        log.info("Marker загружен")

    @log.catch
    def run_marker(self, pdf_path: str):
        """
        Основная функция обработки файла через Marker.
        """
        self.rendered_text = None
        self.final_chunks = []

        log.info(f"Обработка файла {pdf_path}")

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

        self.rendered_text = converter(pdf_path)

        self.processing_chunks()
        self.fix_encoding_errors()

        log.info(f"Предварительная обработка файла {pdf_path} завершена")

        if self.output_folder:
            base_name = Path(pdf_path).stem
            output_path = self.output_folder / f"{base_name}_processed_json.txt"

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.final_chunks, f, ensure_ascii=False, indent=4)

            log.info(f"Результат предварительной обработки сохранен в {output_path}")

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
        pattern = r'<math\s+display="(inline|block)"[^>]*>(.*?)</math>'
        chunk_id = 0

        for block in tqdm(self.rendered_text.blocks, f"Обработка чанков"):
            chunk = {}
            page= block.id.split('/')[2]
            chunk['id'] = page + '-' + str(chunk_id)
            chunk['block_type'] = block.block_type
            chunk['page'] = page
            chunk['polygon'] = block.polygon

            if chunk['block_type'] == 'PageHeader' or chunk['block_type'] == 'PageFooter':
                continue

            def replacement_function(match):
                display_type = match.group(1)
                content = match.group(2)

                content = html.unescape(content)
                content = re.sub(r'\\\\(?=\s*\\begin|\\\\| \s*$)', '[[DOUBLE_SLASH]]', content)
                content = content.replace(r"\\", "\\")
                content = content.replace('[[DOUBLE_SLASH]]', r'\\')
                cleaned_content = content.strip()

                if display_type == 'block':
                    return f"\n\\[ {cleaned_content} \\]\n"
                else:
                    return f"${cleaned_content}$"

            chunk['text'] = re.sub(pattern, replacement_function, block.html, flags=re.DOTALL)
            chunk['text'] = BeautifulSoup(chunk['text'], "html.parser").get_text()

            chunk_id += 1
            self.final_chunks.append(chunk)