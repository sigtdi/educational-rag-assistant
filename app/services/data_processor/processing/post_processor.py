import pymupdf
import re
import base64
import json
import unicodedata
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
from pydantic import BaseModel, Field
from pylatexenc.latexwalker import LatexWalker
from time import time
from PIL import Image
import io

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda

from app.logger_setup import log
from app.services.data_processor.utilits import answer_fixer


class VLMOutput(BaseModel):
    """
    Схема выходных данных от VLM
    """
    text: str = Field(
        description="Тест с изображения."
    )

    class ConfigDict:
        schema_extra = {
            "example": [
                {
                    "text": (
                        "$$a_{ij} = \\begin{cases} 1, & \\text{если } (v_i, v_j) \\in E \\\\ 0 & \\text{в противном случае} \\end{cases}$$"
                    )
                },
                {
                    "text": (
                        "Доказательство. $\\Rightarrow$ Пусть граф G = (V, E) удовлетворяет условиям определения 11. "
                        "Покажем индукцией по числу вершин |V|, что $G \\in \\mathcal{D}$. Если |V| = 1, то единственная "
                        "вершина $v \\in V$ является по свойству (1) корнем дерева, т.е. в этом графе ребер нет: $E = "
                        "\\emptyset$. Тогда $G = T_0 \\in \\mathcal{D}$.",
                    )
                }
            ]
        }

'''

Функции очищение памяти от модели

'''

class TextProcessor:
    def __init__(
            self,
            image_folder:       str     = 'extracted_text_images',
            model_name:         str     = 'qwen3-vl:8b-instruct',
            output_folder:      str     = "output_text_processor",
            need_output_file:   bool    = True
    ):
        if need_output_file:
            self.output_folder = Path(__file__).resolve().parent.parent / 'output' / output_folder
            self.output_folder.mkdir(exist_ok=True, parents=True)

        self.image_folder = Path(__file__).resolve().parent.parent / 'output' / image_folder
        self.image_folder.mkdir(exist_ok=True, parents=True)

        self._prompt = None
        self._chain = None
        self._model = None

        self.document_path = None
        self.model_name = model_name
        self.parser = PydanticOutputParser(pydantic_object=VLMOutput)
        self.max_retries = 3
        self.need_output_file = need_output_file

        self.text_chunks = [] # Список чанков с текстом документа
        self.chunk_index_mask = [] # Маска индексов чанков, которые исправлены (1) и которым нужно исправление (0)

        self.process_document_data = {
            'total_chunks_checked_via_vlm': 0, # Количество чанков, которые были нужно проверить через vlm
            'total_corrected_chunks': 0, # Количество чанков, которые были исправлены
            'total_chunks': 0, # Общее количество чанков
            'total_failed_chunks': 0, # Количество чанков, которые не удалось обработать
            'failed_chunks': [], # Номера чанков, которые не удалось обработать
            'result_document_name': '',
            'need_save': self.need_output_file,
            'total_time': 0
        }

    @property
    def prompt(self):
        """
        Шаблон промпта для исправления формул.
        """
        if self._prompt is None:
            self._prompt = ChatPromptTemplate([
                ("system", (
                "Ты — редактор математического учебника, выполняющий OCR текста и формул. Тебе дано изображение фрагмента "
                "учебника - ты должен в ответ дать корректное его содержание без ошибок распознавания, все формулы "
                "обязательно должны быть представлены в корректном LaTex-формате: обернуты в $ (даже если весь текст - формула), "
                "содержать корректные LaTex команды."
                "\n{format_instructions}."
                )),
                ("human", [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_base64}"}}
                ])
            ])

        return self._prompt

    @property
    def chain(self):
        """
        LangChain цепочка: prompt | model | slash_fixer | parser
        """
        if self._chain is None:
            self._chain = self.prompt | self.model | RunnableLambda(answer_fixer) | self.parser
        return self._chain

    @property
    def model(self):
        """
        Модель Ollama.
        """
        if self._model is None:
            self._model = ChatOllama(
                model=self.model_name,
                temperature=0,
                keep_alive=30,
                num_predict=4096,
                repeat_penalty=1.5
            )
        return self._model

    def new_document_stats(self, text, document_path):
        """
        Установка данных для обработки нового документа.
        """
        self.text_chunks = text
        self.chunk_index_mask = [0] * len(self.text_chunks)
        self.document_path = document_path

        self.process_document_data = {
            'total_chunks_checked_via_vlm': 0,
            'total_corrected_chunks': 0,
            'total_chunks': len(self.text_chunks),
            'total_failed_chunks': 0,
            'failed_chunks': [],
            'result_document_name': f"{Path(self.document_path).stem}_text_processed_json.txt",
            'need_save': self.need_output_file,
            'total_time': 0
        }

    def load_image_as_base64(self, image_path: str):
        with open(image_path, 'rb') as file:
            image_bytes = file.read()

        return base64.b64encode(image_bytes).decode('utf-8')

    @log.catch
    def extract_fragments_images_and_text(self, dpi=150):
        """
        Получение изображений текста для всех чанков, которые не прошли простую проверку.
        """

        with pymupdf.open(self.document_path) as document:
            zoom = dpi / 72
            mat = pymupdf.Matrix(zoom, zoom)

            for index, mask in enumerate(tqdm(self.chunk_index_mask, "Получение изображений чанков")):
                if mask:
                    continue

                chunk = self.text_chunks[index]
                page_num = int(chunk['page'])
                page = document[page_num]
                chunk_bbox = chunk['bbox']
                crop_rect = pymupdf.Rect(*chunk_bbox)

                try:
                    pix = page.get_pixmap(matrix=mat, clip=crop_rect)

                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))

                    max_width = 600
                    if img.width > max_width:
                        w_percent = (max_width / float(img.width))
                        h_size = int((float(img.height) * float(w_percent)))
                        img = img.resize((max_width, h_size), Image.Resampling.LANCZOS)

                    if img.height < 32:
                        new_img = Image.new('RGB', (img.width, 32), (255, 255, 255))
                        new_img.paste(img, (0, (32 - img.height) // 2))
                        img = new_img

                    filename = f"chunk{chunk['id']}_{Path(self.document_path).stem}.png"
                    filepath = self.image_folder / filename
                    chunk['image_path'] = filepath
                    img.save(filepath, "PNG", optimize=True)
                except Exception:
                    pass

    def correct_fragments_via_vlm(self, chunk_id, image_path):
        image_b64 = self.load_image_as_base64(image_path)

        for attempt in range(self.max_retries):
            try:
                result = self.chain.invoke({
                    'image_base64': image_b64,
                    "format_instructions": self.parser.get_format_instructions()
                })

                return {'result': result, 'status': 'success'}

            except json.JSONDecodeError as e:
                log.error(f"Некорректный ответ для чанка {chunk_id}. Осталось повторных попыток {self.max_retries - attempt - 1}")
                if attempt == self.max_retries:
                    return {'result': None, 'status': 'error'}

            except Exception as e:
                log.error(f"Ошибка обработки чанка {chunk_id}: {e}.\nОсталось повторных попыток {self.max_retries - attempt - 1}")
                if attempt == self.max_retries:
                    return {'result': None, 'status': 'error'}
        return {'result': None, 'status': 'error'}

    @log.catch
    def initial_check(self):
        """
        Первичная проверка всех чанков: отсутствие слов со смешанными алфавитами и LaTex формул.
        """
        pattern = r'(\${1,2})(.*?)\1'

        for index, chunk in enumerate(tqdm(self.text_chunks, "Первичная проверка чанков")):
            if chunk['block_type'] in {'Picture', 'Figure', 'FigureGroup', 'PictureGroup'}:
                self.chunk_index_mask[index] = 1
                continue

            # Проверяем наличие формул
            matches = re.findall(pattern, chunk['text'])

            # Проверяем наличие слов со смешанными алфавитами
            text_no_math = re.sub(pattern, ' ', chunk['text'], flags=re.DOTALL)
            text_clean_sep = re.sub(r'[-–—]', ' ', text_no_math)

            has_one_lang = True
            for word in text_clean_sep.split():
                clean_word = "".join(c for c in word if c.isalpha())

                if not clean_word or not has_one_lang:
                    continue

                has_other = False
                has_cyrillic = False

                for char in clean_word:
                    try:
                        name = unicodedata.name(char)
                        if "CYRILLIC" in name:
                            has_cyrillic = True
                        else:
                            has_other = True
                    except ValueError:
                        continue

                    if has_other and has_cyrillic:
                        has_one_lang = False
                        break

            self.chunk_index_mask[index] = (not matches) and has_one_lang
            self.process_document_data['total_chunks_checked_via_vlm'] += not ((not matches) and has_one_lang)

    def validate_chunk(self, vlm_text, chunk_text):
        """
        Проверка формул в исправленном тексте на соответствие LaTex формату.
        Проверка соответствия ответа vlm тексту после ocr (для статистики).
        """
        if vlm_text['status'] == 'error':
            return False

        pattern = r'(\${1,2})(.*?)\1'
        text = vlm_text['result'].text

        # Проверяем все $..$ блоки на соответствие LaTex формату
        matches = re.findall(pattern, text)
        for delim, content in matches:
            try:
                LatexWalker(content).get_latex_nodes()
            except Exception:
                return False

        # Проверяем текст на смешение символов
        text_no_math = re.sub(pattern, ' ', text, flags=re.DOTALL)
        text_clean_sep = re.sub(r'[-–—]', ' ', text_no_math)

        for word in text_clean_sep:
            clean_word = "".join(c for c in word if c.isalpha())

            if not clean_word:
                continue

            has_latin = False
            has_cyrillic = False

            for char in clean_word:
                try:
                    name = unicodedata.name(char)
                    if "LATIN" in name:
                        has_latin = True
                    elif "CYRILLIC" in name:
                        has_cyrillic = True
                except ValueError:
                    continue

                if has_latin and has_cyrillic:
                    return False

        if text != chunk_text:
            self.process_document_data['total_corrected_chunks'] += 1

        return True

    @log.catch
    def insert_fixed_fragments(self, chunk_index, vlm_text):
        text = vlm_text['result'].text
        self.text_chunks[chunk_index]['text'] = text
        self.chunk_index_mask[chunk_index] = 1

    def update_stats(self, total_time):
        self.process_document_data['total_failed_chunks'] = self.chunk_index_mask.count(0)
        self.process_document_data['failed_chunks'] = [self.text_chunks[i]['id'] for i, val in enumerate(self.chunk_index_mask) if val == 0]
        self.process_document_data['total_time'] = total_time

    def get_stats(self):
        return self.process_document_data

    def save_final_document(self):
        """
        Сохранения результата обработки документа.
        """
        output_path = self.output_folder / self.process_document_data['result_document_name']

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.text_chunks, f, ensure_ascii=False, indent=4, default=str)

        log.info(f"Результат обработки сохранен в {output_path}")

    def process(self, chunks, document_path):
        log.info(f'Обработка текста для документа {Path(document_path).name}')
        start_time = time()

        self.new_document_stats(text=chunks, document_path=document_path)
        self.initial_check()
        self.extract_fragments_images_and_text()

        for attempt in range(self.max_retries):
            for index, mask in enumerate(tqdm(self.chunk_index_mask, "Исправление текста")):
                if mask:
                    continue

                chunk = self.text_chunks[index]
                vlm_text = self.correct_fragments_via_vlm(chunk['id'], chunk['image_path'])
                if self.validate_chunk(vlm_text, chunk['text']):
                    self.insert_fixed_fragments(chunk_index=index, vlm_text=vlm_text)

            if all(self.chunk_index_mask):
                log.info("Весь текст успешно проверен и исправлен.")
                # Сохранение данных в файл
                if self.need_output_file:
                    self.save_final_document()

                self.update_stats(total_time=time() - start_time)
                return self.text_chunks

            else:
                log.warning(f"Остались неисправленные чанки. Еще повторных попыток {self.max_retries - attempt - 1}.")
        else:
            log.error("Достигнуто максимальное количество попыток, но не все чанки исправлены.")
            # Сохранение данных в файл
            if self.need_output_file:
                self.save_final_document()

            self.update_stats(time() - start_time)
            return self.text_chunks

