import pymupdf
import re
import base64
import json
import subprocess
import gc
from tqdm import tqdm
from pathlib import Path
from typing import List, Literal
from pydantic import BaseModel, Field
from pylatexenc.latexwalker import LatexWalker
from time import time

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from app.logger_setup import log
from app.services.document_processing.parser.utils import answer_fixer

#TODO
class ImageOutput(BaseModel):
    """
    Схема выходных данных от VLM
    """
    description: str = Field(
        description="Содержательное описание изображения на русском языке (100-800) символов.",
        min_length=50,
        max_length=1200
    )

    image_type: Literal[
        "граф",
        "дерево",
        "блок-схема",
        "диаграмма",
        "таблица",
        "псевдокод",
        "временная_сложность",
        "иллюстрация",
        "другое"
    ] = Field(
        description="Тип изображения из учебника по алгоритмам"
    )

    key_elements: List[str] = Field(
        description="Ключевые элементы изображения (вершины, рёбра, подписи, стрелки и т.д.)",
        default_factory=list
    )

    class ConfigDict:
        schema_extra = {
            "examples": [
                {
                    "description": (
                        "Ориентированный граф G с 6 вершинами и 8 рёбрами. Вершины обозначены буквами A, B, C, D, E, F. "
                        "Рёбра показаны стрелками с указанием весов: (A, B, 4), (A, C, 2), (B, C, 1), (B, D, 5), "
                        "(C, D, 8), (C, E, 10), (D, E, 2), (E, F, 6). Граф иллюстрирует задачу поиска кратчайшего пути. "
                        "Вершина A выделена как стартовая, F как конечная."
                    ),
                    "image_type": "граф",
                    "key_elements": [
                        "6 вершин (A, B, C, D, E, F)",
                        "8 взвешенных рёбер",
                        "ориентированные стрелки",
                        "веса рёбер",
                        "выделенные стартовая и конечная вершины"
                    ]
                },
                {
                    "description": (
                        "Двоичное дерево поиска (BST) с корнем 50. Левое поддерево содержит значения меньше корня: 30, "
                        "20, 40, 35. Правое поддерево содержит значения больше корня: 70, 60, 80, 75. Все узлы "
                        "представлены кружками с числами внутри.Стрелки показывают связи родитель-потомок. "
                        "Дерево сбалансировано на глубину 4."
                    ),
                    "image_type": "дерево",
                    "key_elements": [
                        "корень со значением 50",
                        "9 узлов всего",
                        "левое и правое поддеревья",
                        "стрелки от родителей к потомкам",
                        "свойство BST сохранено"
                    ]
                },
                {
                    "description": (
                        "Граф состояний конечного автомата для распознавания строк вида (ab)*. Пять состояний: q0 "
                        "(начальное), q1, q2 (принимающее), q3 (ошибка). Переходы обозначены стрелками с метками "
                        "символов: (q0, q1, a), (q1, q2, 'b'), (q2, q0, 'ε'). Любые другие символы ведут в q3. "
                        "Принимающее состояние q2 выделено двойным кружком. Стрелка входа указывает на q0."
                    ),
                    "image_type": "граф",
                    "key_elements": [
                        "5 состояний (q0, q1, q2, q3)",
                        "переходы с метками символов",
                        "начальное состояние q0",
                        "принимающее состояние q2 (двойной круг)",
                        "состояние ошибки q3"
                    ]
                }
            ]
        }

class ImageProcessor:
    def __init__(
            self,
            image_folder:      str     = 'extracted_images',
            model_name:        str     = 'qwen3-vl:8b-instruct',
            output_folder:     str     = "output_image_processor",
            need_output_file:  bool    = True,
            delete_images:     bool    = False
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
        self.current_document_name = None
        self.model_name = model_name
        self.parser = PydanticOutputParser(pydantic_object=ImageOutput)
        self.max_retries = 3
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
            'total_time': 0
        }

    @property
    def get_image_context(self):
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
                "Ты эксперт по описанию изображений из учебников по алгоритмам и структурам данных. "
                "Твоя задача: создать точное и подробное описание для студентов, изучающих алгоритмы. "
                "Для описания используй только русский язык, уложись в 100-800 символов. Если будешь использовать "
                "математические выражения, записывай их в latex. Двигайся по шагам, определи тип изображения "
                "(граф, дерево, блок-схема и т.д.); структуру (количество вершин/узлов, связи); ключевые элементы"
                "(веса, метки, выделения); назначение (что иллюстрирует).\nДля графов описывай ориентированность,"
                "особенности (циклы, связность, выделенные вершины), веса (если есть), количество вершин и рёбер.\n"
                "Для деревьев описывай тип (бинарное, B-дерево, красно-черное и т.д.), корень и структуру, свойства "
                "(сбалансированность, свойство поиска) и т.д..\n{format_instructions}."
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
                keep_alive="60m",
                repeat_penalty=1.5,
                reasoning=False
            )
        return self._model

    def clear_vlm_memory(self):
        """
        Выгружает модель из памяти
        """

        if hasattr(self, "_model") and self._model is not None:
            del self._model
            self._model = None

        subprocess.run(["ollama", "stop", self.model_name], check=False)

        gc.collect()

    def new_document_stats(self, text, document_path):
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
            'result_document_name': f"{Path(self.document_path).stem}_images_processed_json.txt",
            'need_save': self.need_output_file,
            'total_time': 0
        }

    def load_image_as_base64(self, image_path: str):
        with open(image_path, 'rb') as file:
            image_bytes = file.read()

        return base64.b64encode(image_bytes).decode('utf-8')

    @log.catch
    def extract_images(self, dpi=150):
        """
        Получение изображений для чанков, содержащих блоки Picture или Figure.
        """

        with pymupdf.open(self.document_path) as document:
            zoom = dpi / 72
            mat = pymupdf.Matrix(zoom, zoom)

            for index, context, chunk in tqdm(self.get_image_context, total=self.remaining_chunks,
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
                except Exception:
                    pass

    def delete_images(self):
        """
        Удаляет все изображения, созданные для текущего документа.
        """
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

    def initial_check(self):
        """
        Поиск всех чанков с изображениями и заполнение маски.
        """

        for index, chunk in enumerate(tqdm(self.text_chunks, "Поиск чанков с изображениями")):
            if chunk['block_type'] not in {'Picture', 'Figure', 'FigureGroup', 'PictureGroup'}:
                self.chunk_index_mask[index] = 1
            else:
                self.chunk_index_mask[index] = 0

        self.remaining_chunks = self.chunk_index_mask.count(0)

    def description_generation_via_vlm(self, chunk_id, caption, context, image_path):
        image_b64 = self.load_image_as_base64(image_path)

        for attempt in range(self.max_retries):
            try:
                result = self.chain.invoke({
                    'image_base64': image_b64,
                    'caption': caption,
                    'context': context,
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

    def validate_chunk(self, vlm_answer):
        """
        Проверка сгенерированных описаний.
        """
        description = vlm_answer.description

        latex_blocks = re.findall(r'\$(.*?)\$', description)
        is_valid = True

        for block in latex_blocks:
            if not block.strip():
                continue

            try:
                LatexWalker(block).get_latex_nodes()
                is_valid *= True
            except Exception:
                is_valid *= False

        return is_valid

    def insert_image_data(self, chunk_index, vlm_answer):
        description = vlm_answer.description
        key_elements = vlm_answer.key_elements
        image_type = vlm_answer.image_type

        self.process_document_data['described_images'] += 1

        kw_str = f". Ключевые слова: {', '.join(key_elements)}" if key_elements else ""
        insert = f"![{image_type}][{self.text_chunks[chunk_index]['text']}]({description}{kw_str})"
        self.text_chunks[chunk_index]['text'] = insert
        self.chunk_index_mask[chunk_index] = 1

    def update_stats(self, total_time):
        self.process_document_data['failed_chunks'] = [self.text_chunks[i]['id'] for i, val in enumerate(self.chunk_index_mask) if val == 0]
        self.process_document_data['failed_images_count'] = self.chunk_index_mask.count(0)
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
        start_time = time()
        log.info(f'Обработка изображений для документа {Path(document_path).name}')
        self.new_document_stats(text=chunks, document_path=document_path)
        self.initial_check()
        self.extract_images()

        for attempt in range(self.max_retries):
            for index, context, chunk in tqdm(self.get_image_context, total=self.remaining_chunks,
                                              desc='Генерация описаний'):
                vlm_answer = self.description_generation_via_vlm(chunk['id'], chunk['text'], context, chunk['image_path'])
                is_valid = self.validate_chunk(vlm_answer=vlm_answer['result'])

                if not is_valid or vlm_answer['status'] == 'error':
                    continue

                self.insert_image_data(chunk_index=index, vlm_answer=vlm_answer['result'])

            if all(self.chunk_index_mask):
                log.info("Описания ко всем изображениям успешно сгенерированы.")
                # Сохранение данных в файл и удаление изображений
                if self.need_output_file:
                    self.save_final_document()
                if self.need_delete_images:
                    self.delete_images()

                self.update_stats(total_time=time() - start_time)
                self.clear_vlm_memory()
                return self.text_chunks

            else:
                self.remaining_chunks = self.chunk_index_mask.count(0)
                log.warning(
                    f"Остались необработанные изображения. Еще повторных попыток {self.max_retries - attempt - 1}.")
        else:
            log.error("Достигнуто максимальное количество попыток, но не все изображения обработаны.")
            # Сохранение данных в файл и удаление изображений
            if self.need_output_file:
                self.save_final_document()
            if self.need_delete_images:
                self.delete_images()

            self.update_stats(time() - start_time)
            self.clear_vlm_memory()
            return self.text_chunks
