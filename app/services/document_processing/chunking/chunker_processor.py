import re
import json
from time import time
from pathlib import Path
from tqdm import tqdm

from app.services.document_processing.chunking.chunker_config import ChunkerConfig
from app.logger_setup import log

class ChunkProcessor:
    def __init__(self, config: ChunkerConfig):
        self.config = config

        if self.config.save_intermediate:
            self.output_folder = self.config.output_dir
            self.output_folder.mkdir(exist_ok=True, parents=True)

        self.document_path = None
        self.chunks = []
        self.parent_chunks = []
        self.anchor_dict: dict[str, list[str]] = {}

        self.max_continuation_chunks = 6
        self.max_len_parent_chunk = 10

        # Метки
        self.label_pattern = re.compile(
            r'^(определение|теорема|лемма|аксиома|следствие|утверждение|'
            r'замечание|пример|алгоритм|задача|свойство|доказательство)'
            r'[\s\xa0]+(\d+(?:\.\d+)*)[\.\:\s]?',
            re.IGNORECASE
        )
        # Метки для рисунков
        self.figure_pattern = re.compile(
            r'рис(?:унок)?\.?\s*(\d+(?:\.\d+)*)',
            re.IGNORECASE
        )
        # Сигнальные слова, после которых продолжение отслеживания метки прекращается
        self.stop_words = re.compile(
            r'^(доказательство|заметим|покажем|таким образом|следовательно|'
            r'отсюда|действительно|вернёмся|перейдём|рассмотрим|покажем|во многих|многие)',
            re.IGNORECASE
        )

        # Ссылки на метки из других чанков
        self.ref_pattern = re.compile(
            r'\b(определени[иею]|теорем[еаы]|лемм[еаы]|аксиом[еаы]|следстви[иею]|'
            r'утвержден[ии]|замечани[иею]|примере?|алгоритм[еа]|задач[еи]|свойств[еа]|'
            r'доказательств[еа]|рис(?:унк[еа])?\.?)'
            r'[\s\xa0]+(\d+(?:\.\d+)*)',
            re.IGNORECASE
        )

        self.process_document_data = {
            'total_chunks': 0,  # Общее количество начальных мини-чанков
            'total_parent_chunks': 0,  # Количество групп из объединенных мини-чанков
            'total_final_chunks': 0,  # Количество оставленных мини-чанков
            'unresolved_refs': 0,  # Количество ссылок на неопределенную метку
            'result_document_name': '',
            'saved': self.config.save_intermediate,
            'total_time': 0
        }

    def load_chunks(self):
        # Определяем путь, учитывая суффикс
        suffix = self.config.suffix if (self.config.has_suffix and self.config.document_name) else ''
        file_path = self.config.input_dir / (str(Path(self.config.document_name).stem) + suffix)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            log.info(f"Успешно загружен файл с чанками для старта обработки: {file_path}")
        except FileNotFoundError:
            log.error(f"Ошибка: файл не найден по пути {file_path}")
        except json.JSONDecodeError:
            log.error(f"Ошибка: некорректный JSON в файле {file_path}")

    def new_document_stats(self, document_name, chunks = None):
        if chunks:
            self.chunks = chunks
        else:
            self.load_chunks()

        self.process_document_data = {
            'total_chunks': len(self.chunks),  # Общее количество начальных мини-чанков
            'total_parent_chunks': 0,  # Количество больших групп из объединенных мини-чанков
            'total_final_chunks': 0,  # Количество оставленных мини-чанков
            'unresolved_refs': 0,  # Количество ссылок на неопределенную метку
            'result_document_name': f"{Path(document_name).stem}_chunk_processed_json.txt",
            'saved': self.config.save_intermediate,
            'total_time': 0
        }

    def update_stats(self, total_time):
        self.process_document_data['total_time'] = total_time

    def get_stats(self):
        return self.process_document_data

    def save_final_document(self):
        """
        Сохранения результата обработки документа.
        """
        if not self.config.save_intermediate:
            return

        output_path = self.output_folder / self.process_document_data['result_document_name']

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.parent_chunks, f, ensure_ascii=False, indent=4, default=str)

        log.info(f"Результат обработки сохранен в {output_path}")

    def assign_section_headers(self):
        """
        Создает для каждого чанка section_path - список глав и подглав, в которых чанк находится.
        Меняет ошибочно типированные SectionHeader чанки на Text.
        """
        section_header_pattern = re.compile(r'^\d+(\.\d+)*\.?\s+\S')
        section_stack: list[tuple[tuple[int, ...], str]] = []

        for chunk in tqdm(self.chunks, 'Определение секции чанка'):
            # Не-заголовочному чанку добавляем названия глав и подглав, в которых он находится
            if chunk["block_type"] != "SectionHeader":
                chunk["section_path"] = self._build_section_path(section_stack)
                continue

            text = chunk["text"].strip()

            if not section_header_pattern.match(text) or any(re.search(rf'\b{word}\b', text, re.IGNORECASE)
                                                                for word in ['for', 'else', 'if', 'do', 'while']):
                # Ошибочное типирование от Marker
                chunk["block_type"] = "Text"
                chunk["section_path"] = self._build_section_path(section_stack)
                continue

            # Извлекаем числовой префикс и текст из названия главы
            numeric_prefix = text.split()[0].rstrip('.')
            text_suffix = re.sub(r'^[\d\.]+\s*', '', text).strip()
            numeric_key = tuple(int(n) for n in numeric_prefix.split('.'))

            # Обновляем стек, сбрасывая старые неактуальные главы
            while section_stack and len(section_stack[-1][0]) >= len(numeric_key):
                section_stack.pop()
            section_stack.append((numeric_key, text_suffix))

            chunk["section_path"] = self._build_section_path(section_stack)

    def _build_section_path(self, stack: list[tuple[tuple[int, ...], str]]) -> str | None:
        """
        Собирает section_path из текущего стека заголовков.
        """
        if not stack:
            return ''
        return " > ".join(header_text for _, header_text in stack)

    def detect_content_labels(self):
        """
        Определяет лейбл чанка: содержит ли чанк внутри себя определение/теорему/лемму и тд.
        """
        current_label = None
        continuation_count = 0

        for chunk in tqdm(self.chunks, 'Определение меток чанков'):
            block_type = chunk["block_type"]
            text = chunk.get("text", "").strip()

            chunk["content_label"] = ''

            # Ищем рисунки
            if block_type in {"PictureGroup", "FigureGroup"}:
                fig_match = self.figure_pattern.search(text)
                if fig_match:
                    label = f"Рисунок {fig_match.group(1)}"
                    chunk["content_label"] = label

                    if label not in self.anchor_dict:
                        self.anchor_dict[label] = []
                    self.anchor_dict[label].append(chunk["id"])
                current_label = None
                continuation_count = 0
                continue

            # Ищем метки в текстовых чанках, учитывая, что одна метка может распространяться на несколько чанков
            if block_type != "Text":
                current_label = None
                continuation_count = 0
                continue

            label_match = self.label_pattern.match(text)

            if label_match:
                kind = self._normalize_kind(label_match.group(1))
                number = label_match.group(2)
                label = f"{kind} {number}"
                chunk["content_label"] = label

                if label not in self.anchor_dict:
                    self.anchor_dict[label] = []
                self.anchor_dict[label].append(chunk["id"])

                # Запускаем отслеживание продолжений для этой метки
                current_label = label
                continuation_count = 0
                continue

            # Проверяем, может ли чанк без метки быть продолжением чанка с меткой
            if current_label is not None:
                if self._is_continuation(text, continuation_count):
                    chunk["content_label"] = current_label
                    self.anchor_dict[current_label].append(chunk["id"])
                    continuation_count += 1
                else:
                    current_label = None
                    continuation_count = 0

    def _is_continuation(self, text: str, continuation_count: int) -> bool:
        """
        Возвращает True, если чанк считается продолжением предыдущей метки.
        """
        if continuation_count >= self.max_continuation_chunks:
            return False
        if self.stop_words.match(text):
            return False
        return True

    def _normalize_kind(self, raw: str) -> str:
        """
        Приводит тип к единому виду с заглавной буквы.
        """
        return raw.strip().capitalize()

    def collect_references(self):
        """
        Собирает внутренние ссылки на метки в других чанках.
        """
        for chunk in tqdm(self.chunks, 'Собор ссылок на другие чанки'):
            text = chunk.get("text", "")
            self_label = chunk.get("content_label")

            linked = {}
            unresolved = []

            for match in self.ref_pattern.finditer(text):
                kind_raw = match.group(1)
                number = match.group(2)
                label = f"{self._normalize_kind_from_ref(kind_raw)} {number}"

                # Ссылки на самого себя пропускаем
                if label == self_label:
                    continue

                # Ищем метку среди уже найденных
                if label in self.anchor_dict:
                    linked[label] = self.anchor_dict[label]
                else:
                    if label not in unresolved:
                        unresolved.append(label)

            chunk["linked_chunks"] = linked
            chunk["unresolved_refs"] = unresolved

    def _normalize_kind_from_ref(self, raw: str) -> str:
        """
        Приводит словоформу из ссылки к нормализованному виду.
        """
        raw = raw.strip().rstrip('.').lower()

        normalization_map = {
            'определени': 'Определение',
            'теорем': 'Теорема',
            'лемм': 'Лемма',
            'аксиом': 'Аксиома',
            'следстви': 'Следствие',
            'утвержден': 'Утверждение',
            'замечани': 'Замечание',
            'пример': 'Пример',
            'алгоритм': 'Алгоритм',
            'задач': 'Задача',
            'свойств': 'Свойство',
            'доказательств': 'Доказательство',
            'рис': 'Рисунок',
            'рисунк': 'Рисунок',
        }

        for prefix, normalized in normalization_map.items():
            if raw.startswith(prefix):
                return normalized

        return raw.capitalize()

    def build_parent_chunks(self):
        """
        Собирает мини-чанки в группы.
        """

        parent_counter = 0
        current_group = []
        current_path = None

        # Буфер для чанков с одинаковой меткой
        buffer = []
        current_buffer_label = None

        def flush_group():
            """
            Собирает накопленную группу в мега-чанк и сбрасывает её.
            """
            nonlocal parent_counter, current_group

            if not current_group:
                return

            parent_id = f"{parent_counter}"
            parent_counter += 1

            # Собираем все метки без дубликатов, сохраняя порядок появления
            content_labels: list[str] = []
            seen_labels: set[str] = set()
            for c in current_group:
                label = c.get("content_label")
                if label and label not in seen_labels:
                    content_labels.append(label)
                    seen_labels.add(label)

            # Объединяем linked_chunks, убираем самоссылки на метки этого мега-чанка
            linked_chunks: dict[str, str] = {}
            for c in current_group:
                for label, chunk_id in c.get("linked_chunks", {}).items():
                    if label not in seen_labels:
                        linked_chunks[label] = chunk_id

            # Объединяем unresolved_refs без дубликатов
            unresolved: list[str] = []
            seen_unresolved: set[str] = set()
            for c in current_group:
                for ref in c.get("unresolved_refs", []):
                    if ref not in seen_unresolved:
                        unresolved.append(ref)
                        seen_unresolved.add(ref)

            # Каждому чанку в группе подписываем id его родительского чанка и порядковый номер внутри группы
            inner_id = 1
            for c in current_group:
                c['parent_group'] = parent_id
                c['inner_id'] = inner_id
                inner_id += 1

            self.parent_chunks.append({
                "id": parent_id,
                "section_path": current_group[0].get("section_path"),
                "chunk_ids": [c["id"] for c in current_group],
                "mini_chunk_count": len(current_group),
                "content_labels": content_labels,
                "linked_chunks": linked_chunks,
                "unresolved_refs": unresolved,
                "chunks": current_group
            })

            current_group = []

        def move_buffer_to_group():
            nonlocal current_group, buffer
            if not buffer:
                return

            if current_group and (len(current_group) + len(buffer) > self.max_len_parent_chunk):
                flush_group()

            current_group.extend(buffer)
            buffer = []

        for chunk in tqdm(self.chunks, 'Объединение чанков в группы'):
            if chunk['block_type'] == 'SectionHeader':
                continue

            chunk_path = chunk.get("section_path")
            chunk_label = chunk.get("content_label", "").strip()

            # Закрываем группу при смене раздела
            if current_path is not None and chunk_path != current_path:
                move_buffer_to_group()
                flush_group()

            current_path = chunk_path

            # Добавляем чанк в группу, следя за неразрывностью меток
            if not chunk_label:
                # Метки нет: сбрасываем текущий буфер в группу и добавляем чанк в группу
                move_buffer_to_group()
                current_group.append(chunk)
                current_buffer_label = None
            else:
                # Метка есть: проверяем, совпадает ли она с той, что в буфере
                if chunk_label == current_buffer_label:
                    buffer.append(chunk)
                else:
                    # Новая метка: старый буфер в группу, начинаем новый
                    move_buffer_to_group()
                    buffer.append(chunk)
                    current_buffer_label = chunk_label

            if len(current_group) >= self.max_len_parent_chunk:
                flush_group()

        move_buffer_to_group()
        flush_group()
        self.process_document_data['total_parent_chunks'] = len(self.parent_chunks)

    @log.catch
    def run(self, chunks: list | None = None):
        start_time = time()
        log.info(f'Обработка чанков для документа {Path(self.config.document_name).name}')

        self.new_document_stats(document_name = self.config.document_name, chunks=chunks)
        self.assign_section_headers()
        self.detect_content_labels()
        self.collect_references()
        self.build_parent_chunks()

        self.update_stats(time() - start_time)

        self.save_final_document()

        return self.parent_chunks