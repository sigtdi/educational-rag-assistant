import pymupdf
import re
import base64
import json
import subprocess
import gc
from tqdm import tqdm
from pathlib import Path
from typing import List, Literal, Any, Optional
from pydantic import BaseModel, Field
from pylatexenc.latexwalker import LatexWalker
from time import time

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from app.logger_setup import log
from app.services.document_processing.parser.utils import answer_fixer


class ImageOutput(BaseModel):
    """
    Схема выходных данных от VLM
    """

    image_type: Literal["schema", "code", "table"] = Field(
        description="Тип изображения: schema (граф, дерево, схема), code (псевдокод) или table (таблица)."
    )
    exact_content: Optional[str] = Field(
        default=None,
        description="Заполняется ТОЛЬКО если image_type это 'code' или 'table'. Дословный перенос текста кода или таблицы в формате Markdown."
    )
    description: Optional[str] = Field(
        default=None,
        description="Заполняется ТОЛЬКО если image_type это 'schema'. Подробное описание структуры изображения."
    )
    key_elements: Optional[List[str]] = Field(
        default=None,
        description="Заполняется ТОЛЬКО если image_type это 'schema'. Список ключевых элементов (узлы, связи и т.д.)."
    )

    class ConfigDict:
        json_schema_extra = {
            "examples": [
                {
                    "image_type": "schema",
                    "exact_content": None,
                    "description": (
                        "Сбалансированное бинарное дерево поиска (BST) с 7 узлами. Корень — вершина с ключом 15. "
                        "Левое поддерево содержит узлы 10 и 5, 12; правое — 20 и 18, 25. Все узлы белого цвета, "
                        "связи обозначены тонкими линиями. Дерево демонстрирует идеальную балансировку по высоте, "
                        "где для каждого узла $|height(left) - height(right)| \\le 1$."
                    ),
                    "key_elements": [
                        "корень: 15",
                        "7 узлов (5, 10, 12, 15, 18, 20, 25)",
                        "свойство BST выдержано",
                        "сбалансированная структура"
                    ]
                },
                {
                    "image_type": "code",
                    "exact_content": (
                        "```python\ndef fast_power(a, n):\n    res = 1\n    while n > 0:\n        "
                        "if n % 2 == 1:\n            res *= a\n        a *= a\n        n //= 2\n    return res\n"
                        "```"
                    ),
                    "description": None,
                    "key_elements": None
                },
                {
                    "image_type": "table",
                    "exact_content": (
                        "| i | f[i] | v[i] | w[i] |\n|---|---|---|---|\n| 0 | 0 | 0 | 0 |\n| 1 | 10 | 5 | 2 |\n"
                        "| 2 | 20 | 10 | 4 |\n| 3 | 25 | 12 | 5 |"
                    ),
                    "description": None,
                    "key_elements": None
                },
                {
                    "image_type": "schema",
                    "exact_content": None,
                    "description": (
                        "Неориентированный связный граф, представляющий сеть дорог. Содержит 5 вершин (1-5) "
                        "и 6 рёбер с весами. Иллюстрирует работу алгоритма Прима для поиска MST. "
                        "Рёбра (1,2) с весом 3 и (2,3) с весом 1 выделены жирным синим цветом, показывая текущий "
                        "этап построения остовного дерева."
                    ),
                    "key_elements": [
                        "5 нумерованных вершин",
                        "6 взвешенных рёбер",
                        "выделение цветом (синий)",
                        "индикация алгоритма Прима"
                    ]
                }
            ]
        }


class ImageProcessor:
    def __init__(
            self,
            image_folder:      str,
            model_name:        str,
            output_folder:     str,
            need_output_file:  bool,
            delete_images:     bool
    ):
        if need_output_file:
            self.output_folder = Path(__file__).resolve().parent.parent / 'output' / output_folder
            self.output_folder.mkdir(exist_ok=True, parents=True)

        self.image_folder = Path(__file__).resolve().parent.parent / 'output' / image_folder
        self.image_folder.mkdir(exist_ok=True, parents=True)

        self._prompt = None
        self._chain = None
        self._model = None
        self._parser = PydanticOutputParser(pydantic_object=ImageOutput)
        self._max_retries = 3

        self.document_path = None
        self.current_document_name = None
        self.model_name = model_name
        
        self.need_output_file = need_output_file
        self.need_delete_images = delete_images

        self.text_chunks = [] # Чанки с полным текстом
        self.chunk_index_mask = [] # Маска индексов чанков, которым нужна генерация (1) и которым не нужна (0)
        self.remaining_chunks = 0

        self.process_document_data = {
            'total_images_count': 0, # Общее количество изображений
            'failed_images_count': 0, # Количество изображений, которым не удалось сгенерировать описание
            'described_images': 0, # Количество изображений, которым было сгенерировано описание
            'failed_chunks': [], # Чанки, которым не удалось сгенерировать описание
            'result_document_name': '',
            'need_save': self.need_output_file,
            'total_time': 0.0
        }

    @log.catch
    def process(self, chunks: list[dict[str, Any]], document_path: str | Path) -> list[dict[str, Any]]:
        start_time = time()
        log.info(f'Обработка изображений для документа {Path(document_path).name}')
        self._new_document_stats(text=chunks, document_path=document_path)
        self._initial_check()
        self._extract_images()

        log.info('Генерация описаний для чанков с изображениями')
        for attempt in range(self._max_retries):
            for index, context, chunk in tqdm(self._get_image_context, total=self.remaining_chunks,
                                              desc='Генерация описаний'):
                vlm_answer = self._description_generation_via_vlm(chunk['id'], chunk['text'], context, chunk['image_path'])
                print(vlm_answer)
                self._insert_image_data(chunk_index=index, vlm_answer=vlm_answer['result'])

            if all(self.chunk_index_mask):
                log.info("Описания ко всем изображениям успешно сгенерированы.")
                # Сохранение данных в файл и удаление изображений
                self._save_final_document()
                self._delete_images()

                self._update_stats(total_time=time() - start_time)
                self._clear_vlm_memory()
                return self.text_chunks

            else:
                self.remaining_chunks = self.chunk_index_mask.count(0)
                log.warning(
                    f"Остались необработанные изображения. Еще повторных попыток {self._max_retries - attempt - 1}.")
        else:
            log.error("Достигнуто максимальное количество попыток, но не все изображения обработаны.")
            # Сохранение данных в файл и удаление изображений
            self._save_final_document()
            self._delete_images()

            self._update_stats(time() - start_time)
            self._clear_vlm_memory()
            return self.text_chunks
        
    def get_stats(self) -> dict:
        return self.process_document_data

    @property
    def _get_image_context(self):
        """
        Генератор для итерации по чанкам с изображениями.
        """
        for index, mask in enumerate(self.chunk_index_mask):
            if mask:
                continue

            chunk = self.text_chunks[index]

            prev = self.text_chunks[index - 1]['text'] if index - 1 >= 0 else ''
            next = self.text_chunks[index + 1]['text'] if index + 1 < len(self.text_chunks) else ''

            context = "До изображения:\n" + prev + "\nПосле изображения:\n" + next

            yield index, context, chunk

    @property
    def prompt(self):
        """
        Шаблон промпта для генерации описания.
        """
        if self._prompt is None:
            self._prompt = ChatPromptTemplate([
                ("system", (
                    "Ты эксперт по анализу изображений из учебников по алгоритмам и структурам данных. "
                    "Твоя задача: предоставить точное представление содержимого картинки для студентов. "
                    "Для ответа используй только русский язык. Все математические выражения записывай строго в LaTeX.\n\n"
                    "Сначала проанализируй изображение и определи его тип (image_type). В зависимости от типа действуй по одному из сценариев:\n\n"
                    "СЦЕНАРИЙ 1: image_type = 'code' (Псевдокод или исходный код)\n"
                    "Не описывай код словами! Помести дословный транскрипт кода с сохранением отступов в поле 'exact_content', обернув его в Markdown-блок. "
                    "Поля 'description' и 'key_elements' оставь пустыми (null).\n\n"
                    "СЦЕНАРИЙ 2: image_type = 'table' (Таблица)\n"
                    "Не описывай таблицу! Воспроизведи её содержимое дословно в формате Markdown-таблицы в поле 'exact_content'. "
                    "Поля 'description' и 'key_elements' оставь пустыми (null).\n\n"
                    "СЦЕНАРИЙ 3: image_type = 'schema' (Схема, граф, дерево и т.д.)\n"
                    "Заполни поля 'description' и 'key_elements'. Поле 'exact_content' оставь пустым (null). "
                    "В 'description' создай точное описание (100-800 символов): структура (количество вершин/узлов, связи); назначение. "
                    "Для графов: ориентированность, циклы, связность, веса. Для деревьев: тип (B-дерево и т.д.), корень, свойства.\n\n"
                    "{format_instructions}"
                )),
                ("human", [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": "Подпись изображения:\n{caption}\nТекст вокруг изображения:\n{context}"}
                ])
            ])

        return self._prompt

    @property
    def chain(self):
        """
        LangChain цепочка: prompt | model | answer_fixer | parser
        """
        if self._chain is None:
            self._chain = self.prompt | self.model | RunnableLambda(answer_fixer) | self._parser
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
                keep_alive="60m",
                repeat_penalty=1.5,
                reasoning=False
            )
        return self._model

    def _clear_vlm_memory(self):
        """
        Выгружает модель из памяти
        """

        if hasattr(self, "_model") and self._model is not None:
            del self._model
            self._model = None

        subprocess.run(["ollama", "stop", self.model_name], check=False)

        gc.collect()

    def _new_document_stats(self, text, document_path):
        """
        Установка данных для обработки нового документа.
        """
        self.text_chunks = text
        self.chunk_index_mask = [0] * len(self.text_chunks)
        self.document_path = document_path

        self.process_document_data = {
            'total_images_count': 0,
            'failed_images_count': 0,
            'described_images': 0,
            'failed_chunks': [],
            'result_document_name': f"{Path(self.document_path).stem}_image_processed_json.txt",
            'need_save': self.need_output_file,
            'total_time': 0.0
        }

    def _extract_images(self, dpi=150):
        """
        Получение изображений для чанков, содержащих блоки Picture или Figure.
        """

        with pymupdf.open(self.document_path) as document:
            zoom = dpi / 72
            mat = pymupdf.Matrix(zoom, zoom)

            log.info('Сбор изображений из документа')
            for index, context, chunk in tqdm(self._get_image_context, total=self.remaining_chunks,
                                              desc='Получение изображений'):
                page_num = int(chunk['page'])
                page = document[page_num]
                chunk_bbox = chunk['bbox']
                crop_rect = pymupdf.Rect(*chunk_bbox)
                self.process_document_data['total_images_count'] += 1

                try:
                    pix = page.get_pixmap(matrix=mat, clip=crop_rect)
                    filename = f"chunk{chunk['id']}_{Path(self.document_path).stem}_image.png"
                    filepath = self.image_folder / filename
                    chunk['image_path'] = filepath
                    pix.save(filepath)
                except Exception as e:
                    log.error(f'Ошибка в обработке изображения: {e}')

    def _delete_images(self):
        """
        Удаляет все изображения, созданные для текущего документа.
        """
        if not self.need_delete_images:
            return
        
        log.info('Удаление собранных изображений.')
        stem = Path(self.document_path).stem
        pattern = f"chunk*_{stem}_image.png"

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

    def _initial_check(self):
        """
        Поиск всех чанков с изображениями и заполнение маски.
        """

        for index, chunk in enumerate(tqdm(self.text_chunks, "Поиск чанков с изображениями")):
            if chunk['block_type'] not in {'Picture', 'Figure', 'Table', 'FigureGroup', 'PictureGroup', 'TableGroup'}:
                self.chunk_index_mask[index] = 1
            else:
                self.chunk_index_mask[index] = 0

        self.remaining_chunks = self.chunk_index_mask.count(0)

    def _description_generation_via_vlm(self, chunk_id: int, caption: str, context: str, image_path: str | Path) -> dict:
        image_b64 = self._load_image_as_base64(image_path)

        for attempt in range(self._max_retries):
            try:
                result = self.chain.invoke({
                    'image_base64': image_b64,
                    'caption': caption,
                    'context': context,
                    "format_instructions": self._parser.get_format_instructions()
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

    def _insert_image_data(self, chunk_index: int, vlm_answer):
        image_type = vlm_answer.image_type
        exact_content = vlm_answer.exact_content
        description = vlm_answer.description
        key_elements = vlm_answer.key_elements

        caption = self.text_chunks[chunk_index]['text'] \
            if self.text_chunks[chunk_index]['text'].startswith(("Рис", 'Таблиц')) else ''

        self.process_document_data['described_images'] += 1

        if image_type == "schema":
            kw_str = f". Ключевые слова: {', '.join(key_elements)}" if key_elements else ""
            insert = f"![{caption}]({description}{kw_str})"
        else:
            insert = f"![{caption}]({exact_content})"

        self.text_chunks[chunk_index]['text'] = insert
        self.chunk_index_mask[chunk_index] = 1

    def _update_stats(self, total_time: float):
        self.process_document_data['failed_chunks'] = [self.text_chunks[i]['id'] for i, val in enumerate(self.chunk_index_mask) if val == 0]
        self.process_document_data['failed_images_count'] = self.chunk_index_mask.count(0)
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
        
    @staticmethod
    def _load_image_as_base64(image_path: str):
        with open(image_path, 'rb') as file:
            image_bytes = file.read()

        return base64.b64encode(image_bytes).decode('utf-8')
