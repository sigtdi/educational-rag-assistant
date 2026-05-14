from typing import Any

import pymupdf
import re
import base64
import json
import unicodedata
from tqdm import tqdm
from pathlib import Path
import subprocess
import gc
from pydantic import BaseModel, Field
from pylatexenc.latexwalker import LatexWalker
from time import time
from PIL import Image
import io
import numpy as np

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from app.logger_setup import log
from app.services.document_processing.parser.utils import answer_fixer


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


class TextProcessor:
    def __init__(
            self,
            image_folder:       str,
            model_name:         str,
            output_folder:      str,
            need_output_file:   bool,
            delete_images:      bool
    ):
        if need_output_file:
            self.output_folder = Path(__file__).resolve().parent.parent / 'output' / output_folder
            self.output_folder.mkdir(exist_ok=True, parents=True)

        self.image_folder = Path(__file__).resolve().parent.parent / 'output' / image_folder
        self.image_folder.mkdir(exist_ok=True, parents=True)

        self._prompt = None
        self._chain = None
        self._model = None
        self._gap_threshold = 0.12 # Доля пустого пространства, после которого обрезается изображение
        self._max_retries = 2

        self.document_path = None
        self.model_name = model_name
        self.parser = PydanticOutputParser(pydantic_object=VLMOutput)
        
        self.need_output_file = need_output_file
        self.need_delete_images = delete_images

        self.text_chunks = [] # Список чанков с текстом документа
        self.chunk_index_mask = [] # Маска индексов чанков, которые исправлены (1) и которым нужно исправление (0)
        self.remaining_chunks = 0

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
        
    def process(self, chunks: list[dict[str, Any]], document_path: str | Path) -> list[dict[str, Any]]:
        log.info(f'Обработка текста для документа {Path(document_path).name}')
        start_time = time()

        self._new_document_stats(text=chunks, document_path=document_path)
        self._initial_check()
        self._extract_fragments_images_and_text()

        log.catch('Исправление ошибок с текстовых чанках')
        for attempt in range(self._max_retries):
            for index, chunk in tqdm(self.get_error_chunk, desc="Исправление текста", total=self.remaining_chunks):

                vlm_text = self._correct_fragments_via_vlm(chunk['id'], chunk['image_path'])
                if self._validate_chunk(vlm_text, chunk['text']):
                    self._insert_fixed_fragments(chunk_index=index, vlm_text=vlm_text)

            if all(self.chunk_index_mask):
                log.info("Весь текст успешно проверен и исправлен.")
                # Сохранение данных в файл и удаление изображений
                self._save_final_document()
                self._delete_images()

                self._update_stats(total_time=time() - start_time)
                self._clear_vlm_memory()
                return self.text_chunks

            else:
                self.remaining_chunks = self.chunk_index_mask.count(0)
                log.warning(f"Остались неисправленные чанки. Еще повторных попыток {self._max_retries - attempt - 1}.")
        else:
            log.error("Достигнуто максимальное количество попыток, но не все чанки исправлены.")
            # Сохранение данных в файл и удаление изображений
            self._save_final_document()
            self._delete_images()

            self._update_stats(time() - start_time)
            self._clear_vlm_memory()
            return self.text_chunks
        
    def get_stats(self) -> dict:
        return self.process_document_data

    @property
    def prompt(self) -> ChatPromptTemplate:
        """
        Шаблон промпта для исправления формул.
        """
        if self._prompt is None:
            self._prompt = ChatPromptTemplate([
                ("system", (
                    "Ты — OCR для алгоритмического учебника. "
                    "Точно перепиши текст и формулы с изображения в том же порядке. "
                    "Текст на русском языке — курсивные символы это русские буквы, не латинские. "
                    "Названия алгоритмов и термины переписывай точно как на изображении, "
                    "даже если они кажутся опечаткой или аббревиатурой известного алгоритма. "
                    "Все формулы и математические символы — только в формате LaTeX-команд. "
                    "В формулах правильно переписывай тип скобок и их порядок."
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
        LangChain цепочка: prompt | model | answer_fixer | parser
        """
        if self._chain is None:
            self._chain = self.prompt | self.model | RunnableLambda(answer_fixer) | self.parser
        return self._chain

    @property
    def model(self) -> ChatOllama:
        """
        Модель Ollama.
        """
        if self._model is None:
            self._model = ChatOllama(
                model=self.model_name,
                temperature=0,
                keep_alive="60m",
                num_predict=10000,
                repeat_penalty=1.5,
                reasoning=False
            )
        return self._model

    @property
    def get_error_chunk(self):
        """
        Генератор для итерации по чанкам с изображениями.
        """
        for index, mask in enumerate(self.chunk_index_mask):
            if mask:
                continue

            chunk = self.text_chunks[index]

            yield index, chunk

    def _new_document_stats(self, text, document_path):
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

    def _clear_vlm_memory(self):
        """
        Выгружает модель из памяти
        """

        if hasattr(self, "_model") and self._model is not None:
            del self._model
            self._model = None

        subprocess.run(["ollama", "stop", self.model_name], check=False)

        gc.collect()

    def _delete_images(self):
        """
        Удаляет все изображения, созданные для текущего документа.
        """
        if not self.need_delete_images:
            return
        
        log.info('Удаление собранных изображений.')
        stem = Path(self.document_path).stem
        pattern = f"chunk*_{stem}.png"

        files_to_delete = list(self.image_folder.glob(pattern))

        if not files_to_delete:
            log.info(f"Изображений для документа {stem} не найдено.")
            return

        for file_path in files_to_delete:
            try:
                file_path.unlink()
            except Exception as e:
                log.error(f"Ошибка при удалении {file_path.name}: {e}")

        log.info(f"Очистка завершена. Удалено файлов: {len(files_to_delete)}")

    @log.catch
    def _extract_fragments_images_and_text(self, dpi=150):
        """
        Получение изображений текста для всех чанков, которые не прошли простую проверку.
        """

        with pymupdf.open(self.document_path) as document:
            zoom = dpi / 72
            mat = pymupdf.Matrix(zoom, zoom)
            
            log.info('Получение изображение для чанков, в которых может быть ошибочный текст')
            for index, chunk in tqdm(self.get_error_chunk, desc="Получение изображений", total=self.remaining_chunks):

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

                    img = self._crop_formula_number(img)

                    if img.height < 32:
                        new_img = Image.new('RGB', (img.width, 32), (255, 255, 255))
                        new_img.paste(img, (0, (32 - img.height) // 2))
                        img = new_img

                    filename = f"chunk{chunk['id']}_{Path(self.document_path).stem}.png"
                    filepath = self.image_folder / filename
                    chunk['image_path'] = filepath
                    img.save(filepath, "PNG", optimize=True)
                except Exception as e:
                    log.error(f'Ошибка при обработке изображения: {e}')

    def _correct_fragments_via_vlm(self, chunk_id: int, image_path: str | Path) -> dict:
        image_b64 = self._load_image_as_base64(image_path)

        for attempt in range(self._max_retries):
            try:
                result = self.chain.invoke({
                    'image_base64': image_b64,
                    "format_instructions": self.parser.get_format_instructions()
                })

                return {'result': result, 'status': 'success'}

            except json.JSONDecodeError as e:
                log.error(f"Некорректный ответ для чанка {chunk_id}. Осталось повторных попыток {self._max_retries - attempt - 1}")
                if attempt == self._max_retries:
                    return {'result': None, 'status': 'error'}

            except Exception as e:
                log.error(f"Ошибка обработки чанка {chunk_id}: {e}.\nОсталось повторных попыток {self._max_retries - attempt - 1}")
                if attempt == self._max_retries:
                    return {'result': None, 'status': 'error'}
        return {'result': None, 'status': 'error'}

    @log.catch
    def _initial_check(self):
        """
        Первичная проверка всех чанков: отсутствие слов со смешанными алфавитами и LaTex формул.
        """
        find_pattern = r'\\{1,}(?:[a-zA-Z]+|\{)|(\${1,2})(.*?)\1'
        sub_pattern = r'(\${1,2})(.*?)\1'

        log.info('Поиск чанков, где могут быть ошибки в тексте')
        for index, chunk in enumerate(tqdm(self.text_chunks, "Первичная проверка чанков")):
            if chunk['block_type'] in {'Picture', 'Figure', 'Table', 'FigureGroup', 'PictureGroup', 'TableGroup', 'SectionHeader'}:
                self.chunk_index_mask[index] = 1
                continue

            # Проверяем наличие формул
            matches = re.findall(find_pattern, chunk['text'])

            # Проверяем наличие слов со смешанными алфавитами
            text_no_math = re.sub(sub_pattern, ' ', chunk['text'], flags=re.DOTALL)
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

        self.remaining_chunks = self.chunk_index_mask.count(0)

    def _validate_chunk(self, vlm_text: dict[str, Any], chunk_text: str) -> bool:
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
    def _insert_fixed_fragments(self, chunk_index: int, vlm_text: dict):
        text = vlm_text['result'].text
        self.text_chunks[chunk_index]['text'] = text
        self.chunk_index_mask[chunk_index] = 1

    def _update_stats(self, total_time: float):
        self.process_document_data['total_failed_chunks'] = self.chunk_index_mask.count(0)
        self.process_document_data['failed_chunks'] = [self.text_chunks[i]['id'] for i, val in enumerate(self.chunk_index_mask) if val == 0]
        self.process_document_data['total_time'] = total_time

    def _save_final_document(self):
        """
        Сохранения результата обработки документа.
        """
        if not self.need_output_file:
            return
        
        output_path = self.output_folder / self.process_document_data['result_document_name']

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.text_chunks, f, ensure_ascii=False, indent=4, default=str)

        log.info(f"Результат обработки сохранен в {output_path}")
    
    def _crop_formula_number(self, img: Image.Image) -> Image.Image:
        """
        Убирает правую часть (номер формулы) после большого горизонтального пробела.
        """
        gray = np.array(img.convert('L'))
        h, w = gray.shape

        binary = (gray < 200).astype(np.uint8)
        col_projection = binary.sum(axis=0)

        text_cols = np.where(col_projection > 0)[0]
        if len(text_cols) == 0:
            return img

        gaps = []
        in_gap = False
        gap_start = None
        for i in range(text_cols[0], text_cols[-1] + 1):
            if col_projection[i] == 0 and not in_gap:
                in_gap = True
                gap_start = i
            elif col_projection[i] > 0 and in_gap:
                gaps.append((gap_start, i, i - gap_start))
                in_gap = False

        if not gaps:
            return img

        biggest_gap = max(gaps, key=lambda x: x[2])
        gap_start_col, gap_end_col, gap_width = biggest_gap

        # Считаем размер текста справа и слева от пробела
        right_text_width = text_cols[-1] - gap_end_col
        left_text_width = gap_start_col - text_cols[0]

        is_large_gap = gap_width >= self._gap_threshold * w
        # Правая часть заметно меньше левой — это номер формулы, а не половина выражения
        is_right_part_small = right_text_width < left_text_width * 0.5

        if is_large_gap and is_right_part_small:
            img = img.crop((0, 0, gap_start_col, h))

        return img
    
    @staticmethod
    def _load_image_as_base64(image_path: str | Path):
        with open(image_path, 'rb') as file:
            image_bytes = file.read()

        return base64.b64encode(image_bytes).decode('utf-8')
