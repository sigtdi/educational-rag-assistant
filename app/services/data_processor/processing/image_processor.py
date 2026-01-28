import pymupdf
import re
import base64
import json
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
from app.services.data_processor.utilits import selective_backslash_fixer


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
            need_output_file:  bool    = True
    ):
        if need_output_file:
            self.output_folder = Path(__file__).resolve().parent.parent / 'output' / output_folder
            self.output_folder.mkdir(exist_ok=True, parents=True)

        self.image_folder = Path(__file__).resolve().parent.parent / 'output' / image_folder
        self.image_folder.mkdir(exist_ok=True, parents=True)

        self._prompt = None
        self._chain = None

        self.document_path = None
        self.current_document_name = None
        self.model_name = model_name
        self.parser = PydanticOutputParser(pydantic_object=ImageOutput)
        self.max_retries = 3
        self.need_output_file = need_output_file

        self.text_chunks = [] # Чанки с полным текстом
        self.description_list = {} # Список описаний для каждого чанка, подписей к изображениям и статус
        self.bad_chunks = [] # Чанки, для которых не удалось сгенерировать описание

        self.total_images_count = 0 # Общее количество изображений
        self.bad_images_count = 0 # Количество изображений, для которых не удалось сгенерировать описание через vlm

        self.process_document_data = {} # Данные по всем книгам

    @property
    def one_chunk_fragments(self):
        """
        Генератор для итерации по чанкам с изображениями.
        """
        for index, chunk in enumerate(self.text_chunks):
            chunk_id = chunk['id']
            status = self.description_list.get(chunk_id, {}).get('status', 'process')

            if chunk_id not in self.description_list or status == 'done':
                continue

            prev = self.text_chunks[index - 1]['text'] if index - 1 >= 0 else ''
            next = self.text_chunks[index + 1]['text'] if index + 1 < len(self.text_chunks) else ''

            context = "До изображения:\n" + prev + "\nПосле изображения:\n" + next
            caption = chunk['text']
            image_path = chunk['image_path']

            yield index, chunk_id, caption, context, image_path

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
        LangChain цепочка: prompt | model | slash_fixer | parser
        """
        if self._chain is None:
            self._chain = self.prompt | self.create_model() | RunnableLambda(selective_backslash_fixer) | self.parser
        return self._chain

    def create_model(self):
        return ChatOllama(model=self.model_name, temperature=0, keep_alive=0)

    def new_document_stats(self, text, document_path):
        """
        Установка данных для обработки нового документа.
        """
        self.description_list = {}
        self.bad_images_count = 0
        self.total_images_count = 0


        self.text_chunks = text
        self.document_path = document_path


        self.process_document_data = {
            'total_images_count': 0,
            'uncorrected_images_count': 0,
            'uncorrected_chunks': [],
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

            for chunk in tqdm(self.text_chunks, "Получение изображений"):
                if chunk['block_type'] not in {'Picture', 'Figure', 'FigureGroup', 'PictureGroup'}:
                    continue

                page_num = int(chunk['page'])
                page = document[page_num]
                chunk_bbox = chunk['bbox']
                crop_rect = pymupdf.Rect(*chunk_bbox)
                self.total_images_count += 1

                try:
                    pix = page.get_pixmap(matrix=mat, clip=crop_rect)

                    self.description_list[chunk['id']] = {
                        'description': '',
                        'key_elements': [],
                        'image_type': '',
                        'caption': chunk['text'],
                        'status': 'process'
                    }

                    filename = f"chunk{chunk['id']}_{Path(self.document_path).stem}_image.png"
                    filepath = self.image_folder / filename
                    chunk['image_path'] = filepath
                    pix.save(filepath)
                except Exception:
                    pass

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

                self.description_list[chunk_id]['description'] = result.description
                self.description_list[chunk_id]['key_elements'] = result.key_elements
                self.description_list[chunk_id]['image_type'] = result.image_type

                return None

            except json.JSONDecodeError as e:
                log.error(f"Некорректный ответ для чанка {chunk_id}. Осталось повторных попыток {self.max_retries - attempt}")
                if attempt == self.max_retries:
                    return None

            except Exception as e:
                log.error(f"Ошибка обработки чанка {chunk_id}: {e}.\nОсталось повторных попыток {self.max_retries - attempt}")
                if attempt == self.max_retries:
                    return None
        return None

    def validate_chunk(self, chunk_id):
        """
        Проверка сгенерированных описаний.
        """
        data = self.description_list[chunk_id]
        latex_blocks = re.findall(r'\$(.*?)\$', data['description'])
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

    def insert_image_data(self, chunk_index, chunk_id):
        data = self.description_list[chunk_id]

        kw_str = f". Ключевые слова: {', '.join(data['key_elements'])}" if data['key_elements'] else ""
        markdown_insert = f"![{data['image_type']}][{data['caption']}]({data['description']}{kw_str})"
        self.text_chunks[chunk_index]['text'] = markdown_insert
        data["status"] = "done"

    def update_stats(self, total_time):
        for index, chunk_id, caption, context, image_path in self.one_chunk_fragments:
            self.bad_chunks.append(chunk_id)
            self.bad_images_count += 1

        self.process_document_data['total_images_count'] = self.total_images_count
        self.process_document_data['uncorrected_images_count'] = self.bad_chunks.copy()
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
        self.extract_images()

        for attempt in range(self.max_retries):
            for index, chunk_id, caption, context, image_path in self.one_chunk_fragments:
                self.description_generation_via_vlm(chunk_id, caption, context, image_path)
                is_valid = self.validate_chunk(chunk_id)

                if not is_valid:
                    continue

                self.insert_image_data(chunk_index=index, chunk_id=chunk_id)

            all_done = True
            for chunk_id, data in self.description_list.items():
                if data.get('status', 'process') != 'done':
                    all_done = False
                    break

            if all_done:
                log.info("Описания ко увсем изображениям успешно сгенерированы")
                # Сохранение данных в файл
                if self.need_output_file:
                    self.save_final_document()

                self.update_stats(time() - start_time)
                return self.text_chunks
            else:
                log.warning(f"Остались неисправленные чанки. Еще повторных попыток {self.max_retries - attempt}")
        else:
            log.error("Достигнуто максимальное количество попыток, но не все чанки исправлены")
            # Сохранение данных в файл
            if self.need_output_file:
                self.save_final_document()

            self.update_stats(time() - start_time)
            return self.text_chunks



