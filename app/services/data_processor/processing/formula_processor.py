import pymupdf
import re
import base64
import json
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
from pydantic import BaseModel, Field
from itertools import count
from pylatexenc.latexwalker import LatexWalker
from time import time

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from app.logger_setup import log
from app.services.data_processor.utilits import selective_backslash_fixer


class FormulaOutput(BaseModel):
    """
    Схема выходных данных от VLM
    """
    fragments: List[Dict[str, str]] = Field(
        description="Список исправленных фрагментов или текстов с их типами и содержанием."
    )

    class ConfigDict:
        schema_extra = {
            "example": {
                "fragments": [
                    {"scr": "FRAGMENT_1", "type": "formula", "answer": "f(x) = x^2"},
                    {"scr": "FRAGMENT_2", "type": "text", "answer": "граф"}
                ]
            }
        }

class FormulaProcessor:
    def __init__(
            self,
            image_folder:       str     = 'extracted_formulas_images',
            model_name:         str     = 'qwen3-vl:8b-instruct',
            output_folder:      str     = "output_formulas_processor",
            need_output_file:   bool    = True
    ):
        if need_output_file:
            self.output_folder = Path(__file__).resolve().parent.parent / 'output' / output_folder
            self.output_folder.mkdir(exist_ok=True, parents=True)

        self.image_folder = Path(__file__).resolve().parent.parent / 'output' / image_folder
        self.image_folder.mkdir(exist_ok=True, parents=True)

        self._prompt = None
        self._chain = None

        self.document_path = None
        self.model_name = model_name
        self.parser = PydanticOutputParser(pydantic_object=FormulaOutput)
        self.max_retries = 3
        self.need_output_file = need_output_file

        self.text_chunks = [] # Чанки с полным текстом
        self.old_fragments = {} # Список формул для каждого чанка {chunk_id: [фрагменты]}
        self.fixed_fragments = {} # Список исправленных формул для каждого чанка, маска и статус
        self.bad_fragments = [] # Фрагменты, которые так и не удалось исправить

        self.total_fragments_count = 0 # Общее количество фрагментов, названных формулами в документе
        self.modified_fragments_count = 0 # Количество измененных фрагментов
        self.text_classified_formulas_count = 0 # Количество текста, названного формулой при обработке Marker
        self.bad_fragments_count = 0 # Количество фрагментов, которые не удалось исправить через vlm

        self.process_document_data = {}

    @property
    def one_chunk_fragments(self):
        """
        Генератор для итерации по чанкам и формулам.
        """
        for index, chunk in enumerate(self.text_chunks):
            chunk_id = chunk['id']
            fragments_list = self.old_fragments.get(chunk_id, {}).get('fragments', [])
            status = self.fixed_fragments.get(chunk_id, {}).get('status', 'process')

            if not fragments_list or status == 'done':
                continue

            context = chunk['text']

            image_path = chunk['image_path']

            yield index, chunk_id, fragments_list, context, image_path

    @property
    def prompt(self):
        """
        Шаблон промпта для исправления формул.
        """
        if self._prompt is None:
            self._prompt = ChatPromptTemplate([
                ("system", (
                "Ты — редактор математического учебника, выполняющий исправление ошибок распознавания текста с "
                "использованием изображения. Тебе даны: изображение фрагмента учебника и распознанный текст этого "
                "фрагмента, содержащий некоторые пропуски. С помощью изображения определи содержание пропущенного "
                "фрагмента. Фрагменты из символов, обозначающих математические понятия (переменные с индексами, "
                "фрагменты c символами '=', '+' и другими, обозначения множеств, функций) "
                "считай математическим текстом. В ответ передай для каждого фрагмента: тип - математическое выражение "
                "(formula) или простой текст (text); answer - содержание фрагмента, формула в latex или текст для "
                "обычных слов; scr - какому пропущенному фрагменту соответствует содержание. LaTex формулы пиши по всем "
                "правилам: где нужен один \\ - пиши один, при переносе строк пиши \\\\. \n{format_instructions}."
                )),
                ("human", [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": "Распознанный текст:\n{context}"}
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
        self.old_fragments = {}
        self.bad_fragments = []
        self.modified_fragments_count = 0
        self.text_classified_formulas_count = 0
        self.bad_fragments_count = 0

        self.text_chunks = text
        self.document_path = document_path

        self.process_document_data = {
            'total_fragments_classified_formulas': 0,
            'total_real_formulas': 0,
            'text_classified_formulas_count': 0,
            'uncorrected_fragments_count': 0,
            'total_chunks_with_fragments': 0,
            'uncorrected_chunks': [],
            'result_document_name': f"{Path(self.document_path).stem}_formulas_processed_json.txt",
            'need_save': self.need_output_file,
            'total_time': 0
        }

    def load_image_as_base64(self, image_path: str):
        with open(image_path, 'rb') as file:
            image_bytes = file.read()

        return base64.b64encode(image_bytes).decode('utf-8')

    def prepare_fixed_fragments(self):
        """
        Инициализация self.fixed_fragments и некоторых элементов статистики.
        """
        self.fixed_fragments = {}
        for chunk in self.text_chunks:
            if not self.old_fragments.get(chunk['id'], []):
                continue
            chunk_id = chunk["id"]
            fragments_count = len(re.findall(r'\[\[FRAGMENT_\d+\]\]', chunk["text"]))

            self.total_fragments_count += fragments_count
            self.process_document_data['total_chunks_with_fragments'] += 1

            self.fixed_fragments[chunk_id] = {
                "items": [None] * fragments_count,
                "mask": [0] * fragments_count,
                "status": "process"
            }

    @log.catch
    def extract_fragments_images_and_text(self, dpi=150):
        """
        Получение изображений текста для всех чанков, где есть потенциальные формулы.
        """
        pattern = r'(\${1,2})(.*?)\1'

        with pymupdf.open(self.document_path) as document:
            zoom = dpi / 72
            mat = pymupdf.Matrix(zoom, zoom)

            for chunk in tqdm(self.text_chunks, "Получение изображений чанков с формулами"):
                page_num = int(chunk['page'])
                page = document[page_num]
                chunk_bbox = chunk['bbox']
                crop_rect = pymupdf.Rect(*chunk_bbox)

                try:
                    pix = page.get_pixmap(matrix=mat, clip=crop_rect)
                    matches = re.findall(pattern, chunk['text'])

                    if not matches:
                        continue

                    self.old_fragments[chunk['id']] = {"fragments": [m[1] for m in matches]}

                    counter = count(1)

                    chunk['text'] = re.sub(pattern, lambda m: f"[[FRAGMENT_{next(counter)}]]", chunk['text'])

                    filename = f"chunk{chunk['id']}_{Path(self.document_path).stem}_formulas.png"
                    filepath = self.image_folder / filename
                    chunk['image_path'] = filepath
                    pix.save(filepath)
                except Exception:
                    pass

    def correct_fragments_via_vlm(self, chunk_id, fragments_list, context, image_path):
        image_b64 = self.load_image_as_base64(image_path)

        for attempt in range(self.max_retries):
            try:
                result = self.chain.invoke({
                    'image_base64': image_b64,
                    'context': context,
                    "format_instructions": self.parser.get_format_instructions()
                })

                for fragment in result.fragments:
                    idx = int(fragment["scr"].split('_')[1]) - 1
                    if self.fixed_fragments[chunk_id]["mask"][idx] == 0:
                        self.fixed_fragments[chunk_id]["items"][idx] = fragment

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
        Проверка исправленных формул на соответствие LaTex формату и текста.
        """
        data = self.fixed_fragments[chunk_id]

        for i, item in enumerate(data["items"]):
            if data["mask"][i] == 0 and item is not None:
                content = item.get("answer", "")
                f_type = item.get("type", "text")

                is_valid = True
                if f_type == "formula":
                    try:
                        LatexWalker(content).get_latex_nodes()
                        is_valid = True
                        self.process_document_data['total_real_formulas'] += 1
                    except Exception:
                        is_valid = False
                else:
                    self.process_document_data['text_classified_formulas_count'] += 1

                if is_valid:
                    data["mask"][i] = 1

    @log.catch()
    def insert_fixed_fragments(self, chunk_index, chunk_id):
        data = self.fixed_fragments[chunk_id]

        def replace_callback(match):
            tag = match.group(1)
            try:
                idx = int(tag.split('_')[1]) - 1
                if idx < len(data['mask']) and data['mask'][idx] == 1:
                    item = data['items'][idx]
                    answer = item["answer"]
                    if item["type"] == "formula":
                        return f"${answer}$"
                    return answer
            except (IndexError, ValueError):
                pass

            return f"[[{tag}]]"

        self.text_chunks[chunk_index]['text'] = re.sub(r'\[\[(FRAGMENT_\d+)\]\]', replace_callback, self.text_chunks[chunk_index]['text'])

        if all(data['mask']):
            data["status"] = "done"

    def update_stats(self, total_time):
        for index, chunk_id, text in self.one_chunk_fragments:
            self.process_document_data['uncorrected_chunks'].append(chunk_id)
            self.bad_fragments_count += self.fixed_fragments[chunk_id]['mask'].count(0)

        self.process_document_data['total_fragments_classified_formulas'] = self.total_fragments_count
        self.process_document_data['uncorrected_fragments_count'] = self.bad_fragments_count
        self.process_document_data['total_time'] = total_time

    def get_stats(self):
        return self.process_document_data

    def save_final_document(self):
        """
        Сохранения результата обработки документа.
        """
        output_path = self.output_folder / f"{self.process_document_data['result_document_name']}_formulas_processed_json.txt"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.text_chunks, f, ensure_ascii=False, indent=4, default=str)

        log.info(f"Результат обработки сохранен в {output_path}")

    def process(self, chunks, document_path):
        log.info(f'Обработка формул для документа {Path(document_path).name}')
        start_time = time()
        self.new_document_stats(text=chunks, document_path=document_path)
        self.extract_fragments_images_and_text()
        self.prepare_fixed_fragments()

        for attempt in range(self.max_retries):
            for index, chunk_id, fragments_list, context, image_path in tqdm(self.one_chunk_fragments, 'Обработка чанков с формулами'):
                self.correct_fragments_via_vlm(chunk_id, fragments_list, context, image_path)
                self.validate_chunk(chunk_id)
                self.insert_fixed_fragments(chunk_index=index, chunk_id=chunk_id)

            all_done = True
            for chunk_id, data in self.fixed_fragments.items():
                if data.get('status', 'process') != 'done':
                    all_done = False
                    break

            if all_done:
                log.info("Все формулы успешно исправлены")
                # Сохранение данных в файл
                if self.need_output_file:
                    self.save_final_document()

                self.update_stats(time() - start_time)
                return self.text_chunks
            else:
                log.warn(f"Остались неисправленные чанки. Еще повторных попыток {self.max_retries - attempt}")
        else:
            log.error("Достигнуто максимальное количество попыток, но не все чанки исправлены")
            # Сохранение данных в файл
            if self.need_output_file:
                self.save_final_document()

            self.update_stats(time() - start_time)
            return self.text_chunks


formula_processor = FormulaProcessor()

if __name__ == "__main__":
    chunks = [{'id': '0-0', 'block_type': 'Text', 'page': '0', 'bbox': [122.20401531382834, 68.2208293259197, 557.790106708517, 89.21185373389498], 'text': 'Заметим, что в ориентированном графе может быть ребро вида (u, u), называемое $nem.e\\ddot{u}$, а в неориентированном петель не бывает.'}, {'id': '0-1', 'block_type': 'Text', 'page': '0', 'bbox': [122.20401531382834, 98.20800705159868, 557.790106708517, 169.36459279060364], 'text': 'Пример 1. На рис. 1 приведены примеры ориентированного графа $G_1 = (V_1, E_1)$ и неориентированного графа $G_2 = (V_2, E_2)$. Здесь $V_1 = \\{a, e, c, d\\}, E_1 = \\{(a, b), (a, c), (b, b), (b, d), (d, a)\\},$ $V_2 = \\{a, b, c, d\\}, E_2 = \\{(a, b), (a, c), (a, d), (b, d)\\}$. В графе $G_1$ ребро (b, b) является петлей, полустепень исхода вершины а равна 2, а полустепень захода для нее равна 1. В графе $G_2$ степень вершины а равна 3, вершин b и d-2, вершины c-1, а вершины e-0, т.е. вершина е является изолированной,'}, {'id': '0-2', 'block_type': 'PictureGroup', 'page': '0', 'bbox': [112.77689862251282, 190.20630360245704, 491.065214911396, 310.3672894607775], 'text': 'Рис. 1: Ориентированный граф $G_1$ и неориентированный граф $G_2$'}, {'id': '0-3', 'block_type': 'Text', 'page': '0', 'bbox': [137.7738400697708, 324.611198880475, 323.8781264759131, 335.4405527114868], 'text': 'Части графов называются подграфами.'}, {'id': '0-4', 'block_type': 'Text', 'page': '0', 'bbox': [123.24073457717897, 341.85382607274045, 557.0403888231561, 365.09388881014166], 'text': 'Определение 2. Граф $G_1 = (V_1, E_1)$ называется подграфом графа G = (V, E), если $V_1 \\subseteq V$ и $E_1 \\subseteq E$.'}, {'id': '0-5', 'block_type': 'Text', 'page': '0', 'bbox': [122.07808613777162, 374.0900421278454, 557.790106708517, 408.20155495405197], 'text': 'Во многих приложениях с вершинами и ребрами графов связывается некоторая дополнительная информация. Обычно она представляется с помощью функций разметки вершин и ребер.'}, {'id': '0-6', 'block_type': 'Text', 'page': '0', 'bbox': [123.24073457717897, 417.57144983007987, 557.790106708517, 451.3070247714687], 'text': 'Определение 3. Размеченный граф — это ориентированный или неориентированный граф G=(V,E), снабженный одной или двумя функциями разметки вида: $l:V\\to M$ и $c:E\\to L$, где M и L — множества меток вершин и ребер, соответственно.'}, {'id': '0-7', 'block_type': 'Text', 'page': '0', 'bbox': [123.24073457717897, 452.8063836577527, 557.790106708517, 487.2916380422835], 'text': 'Упорядоченный граф — это размеченный граф G=(V,E), в котором ребра, выходящие из каждой вершины $v\\in V$, упорядочены, т.е. помечены номерами $1,\\ldots,k_v$, где $k_v$ — полустепень исхода v, т.е. $k_v=|\\{w\\mid (v,w)\\in E\\}|$.'}, {'id': '0-8', 'block_type': 'Text', 'page': '0', 'bbox': [122.95373319918924, 495.7614051103592, 557.0403888231561, 517.2788157679626], 'text': 'В качестве множества меток ребер L часто выступают числа, задающие "веса", "длины", "стоимости" ребер. Графы с такой разметкой часто называют взвешенными.'}, {'id': '0-9', 'block_type': 'Text', 'page': '0', 'bbox': [122.95373319918924, 520.2775335405304, 557.0403888231561, 543.4465817213058], 'text': 'Во многих задачах, использующих графы с разметкой ребер, удобно допускать несколько ребер между одной парой вершин.'}, {'id': '0-10', 'block_type': 'Text', 'page': '0', 'bbox': [122.20401531382834, 551.0143907093513, 557.0403888231561, 598.5311822891235], 'text': 'Определение 4. Ориентированный мультиграф – это пара (V, E), где V – конечное множество вершин (узлов, точек) графа, а множество ребер (дуг) E – это некоторое мультимножество пар вершин, т.е. множество, в которое каждый элемент (ребро) может входить несколько раз.'}, {'id': '0-11', 'block_type': 'Text', 'page': '0', 'bbox': [137.94809090640743, 604.9913106155735, 557.0403888231561, 616.2365022627032], 'text': 'Обычно "параллельные" ребра в размеченных мультиграфах различаются своими метками.'}, {'id': '0-12', 'block_type': 'Text', 'page': '0', 'bbox': [122.20401531382834, 629.7307322392587, 557.0403888231561, 664.7149187922478], 'text': 'Неориентированный граф G=(V,E) называется полным, если $E=V\\times V\\setminus\\{(v,v)\\mid v\\in V\\}$, т.е. любые две его вершины соединены ребром. Полный n-вершинный граф часто обозначается как $K_n$. На следующем рисунке показаны первые пять полных графов.'}]
    formula_processor.process(chunks=chunks, document_path="C:/Users/Yana/Downloads/Alg-graphs-full_organized_str5.pdf")
