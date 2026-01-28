import re
import json
from tqdm import tqdm
import unicodedata
from pydantic import BaseModel, Field
from pathlib import Path
from time import time

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from app.logger_setup import log
from app.services.data_processor.utilits import selective_backslash_fixer


class TextOutput(BaseModel):
    """
    Схема выходных данных от VLM
    """
    text: str = Field(
        description="Текст с корректно записанными словами."
    )

    class ConfigDict:
        schema_extra = {
            "example": {
                    "text": (
                        "Во многих приложениях с вершинами и ребрами графов связывается некоторая дополнительная "
                        "информация. Обычно она представляется с помощью функций разметки вершин и ребер."
                    )
            }
        }

class TextProcessor:
    def __init__(
            self,
            model_name:        str     = 'qwen3-vl:8b-instruct',
            output_folder:     str     = "output_text_processor",
            need_output_file:  bool    = True
    ):

        if need_output_file:
            self.output_folder = Path(__file__).resolve().parent.parent / 'output' / output_folder
            self.output_folder.mkdir(exist_ok=True, parents=True)

        self._prompt = None
        self._chain = None

        self.document_name = None
        self.model_name = model_name
        self.parser = PydanticOutputParser(pydantic_object=TextOutput)
        self.max_retries = 3
        self.need_output_file = need_output_file

        self.text_chunks = [] # Чанки с полным текстом
        self.bad_chunks_status = {} # Список чанков со смешанными словами, их количество и статус выполнения
        self.bad_chunks = [] # Чанки, которые не удалось поправить

        self.total_words_count = 0 # Общее количество слов со смешением алфавитов
        self.bad_words_count = 0 # Количество слов, которые не удалось исправить через vlm

        self.process_document_data = {
            'total_words_count': 0,
            'uncorrected_words_count': 0,
            'corrected_words_count': 0,
            'uncorrected_chunks': [],
            'total_chunks_with_bad_words': 0,
            'result_document_name': '',
            'need_save': self.need_output_file,
            'total_time': 0
        }

    @property
    def one_chunk_fragments(self):
        """
        Генератор для итерации по чанкам с неправильными словами.
        """
        for index, chunk in enumerate(self.text_chunks):
            chunk_id = chunk['id']
            status = self.bad_chunks_status.get(chunk_id, {}).get('status', 'process')

            if chunk_id not in self.bad_chunks_status or status == 'done':
                continue

            text = chunk['text']

            yield index, chunk_id, text

    @property
    def prompt(self):
        """
        Шаблон промпта для генерации описания.
        """
        if self._prompt is None:
            self._prompt = ChatPromptTemplate([
                ("system", (
                "Ты умный редактор текста. Тебе дан текст, где в некоторых словах может происходить смешение языков "
                "(в русскоязычном слове одна из букв латинская или наоборот). Твоя задача исправить это слово и в ответ "
                "записать корректное предложение, каждое слово в котором состоит из одного языка. Не изменяй формулы "
                "LaTex.\n{format_instructions}."
                )),
                ("human", "Текст:\n{text}")
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
        self.text_chunks = []
        self.bad_chunks_status = {}
        self.bad_chunks = []

        self.total_words_count = 0
        self.bad_words_count = 0

        self.text_chunks = text
        self.document_name = Path(document_path).name

        self.process_document_data = {
            'total_words_count': 0,
            'uncorrected_words_count': 0,
            'corrected_words_count': 0,
            'uncorrected_chunks': [],
            'total_chunks_with_bad_words': 0,
            'result_document_name': f"{Path(self.document_name).stem}_text_processed_json.txt",
            'need_save': self.need_output_file,
            'total_time': 0
        }

    def check_words(self):
        """
        Поиск всех чанков, где есть слова со смешением языков.
        """
        for chunk in tqdm(self.text_chunks, "Поиск смешанных слов"):
            # Удаляем блоки LaTex формул и разделяем слова через дефис (n-вершинный и т.д.)
            text_no_math = re.sub(r'\$\$.*?\$\$', ' ', chunk['text'], flags=re.DOTALL)
            text_no_math = re.sub(r'\$.*?\$', ' ', text_no_math)
            text_clean_sep = re.sub(r'[-–—]', ' ', text_no_math)

            for word in text_clean_sep.split():
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
                        self.bad_chunks_status[chunk['id']] = {
                            'vlm_res': '',
                            'words_count': self.bad_chunks_status.get(chunk['id'], {}).get('words_count', 0) + 1,
                            'status': 'process'
                        }
                        self.total_words_count += 1
                        break

    def correct_words_via_vlm(self, chunk_id, text):
        for attempt in range(self.max_retries):
            try:
                result = self.chain.invoke({
                    'text': text,
                    "format_instructions": self.parser.get_format_instructions()
                })

                self.bad_chunks_status[chunk_id]['vlm_res'] = result.text

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

    @log.catch
    def validate_chunk(self, chunk_id):
        """
        Проверка сгенерированных описаний.
        """
        data = self.bad_chunks_status[chunk_id]
        # Удаляем блоки LaTex формул и разделяем слова через дефис (n-вершинный и т.д.)
        text_no_math = re.sub(r'\$\$.*?\$\$', ' ', data['vlm_res'], flags=re.DOTALL)
        text_no_math = re.sub(r'\$.*?\$', ' ', text_no_math)
        text_clean_sep = re.sub(r'[-–—]', ' ', text_no_math)

        bad_words = 0

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
                    bad_words += 1
                    break

        self.bad_chunks_status[chunk_id]['words_count'] = bad_words

        if bad_words:
            return False

        data["status"] = "done"
        return True

    def insert_data(self, chunk_index, chunk_id):
        """
        Вставляем исправленную версию в основной текст в чанках.
        """
        data = self.bad_chunks_status[chunk_id]

        self.text_chunks[chunk_index]['text'] = data['vlm_res']

    def update_stats(self, total_time):
        for index, chunk_id, text in self.one_chunk_fragments:
            self.bad_words_count += self.bad_chunks_status[chunk_id]['words_count']
            self.bad_chunks.append(chunk_id)

        self.process_document_data['total_words_count'] = self.total_words_count
        self.process_document_data['uncorrected_words_count'] = self.bad_words_count
        self.process_document_data['corrected_words_count'] = self.total_words_count - self.bad_words_count
        self.process_document_data['uncorrected_chunks'] = self.bad_chunks.copy()
        self.process_document_data['total_chunks_with_bad_words'] = len(self.bad_chunks_status)
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


    @log.catch
    def process(self, chunks, document_path):
        self.new_document_stats(text=chunks, document_path=document_path)

        log.info(f'Обработка текста для документа {self.document_name}')
        start_time = time()
        # Поиск чанков со словами со смешанным языком
        self.check_words()

        # Обработка всех таких чанков
        for attempt in range(self.max_retries):
            for index, chunk_id, text in tqdm(self.one_chunk_fragments, 'Обработка чанков со смешанными словами'):
                self.correct_words_via_vlm(chunk_id, text)
                self.validate_chunk(chunk_id)
                self.insert_data(chunk_index=index, chunk_id=chunk_id)

            all_done = True
            for chunk_id, data in self.bad_chunks_status.items():
                if data.get('status', 'process') != 'done':
                    all_done = False
                    break

            if all_done:
                log.info('Все текстовые чанки успешно обработаны')

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

