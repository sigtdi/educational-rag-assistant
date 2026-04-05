import uuid
import json
import time
import pymupdf
from PIL import Image
import io
from pathlib import Path
from tqdm import tqdm

from app.logger_setup import log

class ChunkStoragePreparer:
    def __init__(
            self,
            output_folder: str = "output_storage_preparer",
            image_folder: str = Path(__file__).resolve().parent.parent.parent / 'rag/data/images',
            need_output_file: bool = True
    ):
        if need_output_file:
            self.output_folder = Path(__file__).resolve().parent.parent / 'output' / output_folder
            self.output_folder.mkdir(exist_ok=True, parents=True)

        self.image_folder = image_folder

        self.need_output_file = need_output_file
        self.document_path = None
        self.chunks = []  # Итоговый результат
        self.parent_chunks = [] # Изначальный список чанков
        self.id_map = {}

        self.fields_to_remove = [
            "block_type",
            "content_label",
            "unresolved_refs",
            "image_path"
        ]

        self.process_document_data = {
            'total_chunks': len(self.chunks),  # Количество итоговых чанков чанков
            'result_document_path': '',
            'need_save': self.need_output_file,
            'total_time': 0.0
        }

    def new_document_stats(self, chunks, document_path: str | Path):
        """
        Инициализация статистики для нового документа.
        """
        self.document_path = Path(document_path)
        self.parent_chunks = chunks
        self.chunks = []
        self.id_map = {}

        self.process_document_data = {
            'total_chunks': len(self.chunks),  # Количество итоговых чанков чанков
            'result_document_path': f"{Path(self.document_path).stem}_storage_preparer_json.txt",
            'need_save': self.need_output_file,
            'total_time': 0.0
        }

    def get_stats(self):
        return self.process_document_data

    def update_stats(self, total_time: float):
        self.process_document_data["total_time"] = total_time

    def save_final_document(self):
        """
        Сохранения результата обработки документа.
        """
        output_path = self.output_folder / self.process_document_data['result_document_path']

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=4, default=str)

        log.info(f"Результат обработки сохранен в {output_path}")

    @log.catch
    def process(self, parent_chunks: list, document_path: str | Path = ''):
        log.info(
            f"Подготовка чанков к перемещению в бд для файла: {document_path.name}")

        start_time = time.time()

        self.new_document_stats(parent_chunks, document_path)
        self._generate_id_mapping()
        self._transform_hierarchy()

        self.update_stats(time.time() - start_time)
        if self.need_output_file:
            self.save_final_document()

        return self.chunks

    def _generate_id_mapping(self):
        """
        Генерирует UUID для всех parent чанков и мини-чанков.
        """
        for parent in tqdm(self.parent_chunks, 'Генерация уникальных id'):
            p_id = str(parent.get("id"))
            if p_id not in self.id_map:
                self.id_map[p_id] = str(uuid.uuid4())

            for chunk in parent.get("chunks", []):
                c_id = str(chunk.get("id"))
                if c_id not in self.id_map:
                    self.id_map[c_id] = str(uuid.uuid4())

    def _transform_hierarchy(self):
        """
        Вызывает детальную обработку для каждого чанка.
        """

        for parent_index, parent_group in enumerate(tqdm(self.parent_chunks, 'Изменение структуры чанков')):
            new_parent_id = self.id_map.get(str(parent_group.get("id")))
            parent_linked_ids = set(parent_group.get("linked_chunks", {}).keys())

            for chunk_index in range(len(parent_group.get("chunks", []))):
                processed_chunk = self._prepare_mini_chunk(
                    parent_index=parent_index,
                    chunk_index=chunk_index,
                    new_parent_id=new_parent_id,
                    parent_linked_ids=parent_linked_ids
                )
                self.chunks.append(processed_chunk)

        self.process_document_data["total_chunks"] = len(self.chunks)

    def _prepare_mini_chunk(self, parent_index, chunk_index, new_parent_id: str, parent_linked_ids: set) -> dict:
        """
        Трансформация одного мини-чанка.
        """
        chunk = self.parent_chunks[parent_index]['chunks'][chunk_index]
        # Работа с картинками
        if chunk.get("block_type") in ["PictureGroup", "FigureGroup"]:
            self._extract_image(parent_index=parent_index, chunk_index=chunk_index)

        new_chunk = chunk.copy()

        # Установка новых полей
        new_chunk["file_name"] = self.document_path.name
        section = new_chunk.get("section_path", "")
        text = new_chunk.get("text", "")
        new_chunk["search_text"] = f"Глава: {section}. {text}".strip()
        new_chunk["type"] = 'picture' if chunk.get("block_type") in ["PictureGroup", "FigureGroup"] else 'text'

        # Замена id
        old_id = str(new_chunk.get("id"))
        new_chunk["id"] = self.id_map.get(old_id, old_id)
        new_chunk["parent_group_id"] = new_parent_id

        # Редактирование и разделение linked_chunks
        self._handle_links(new_chunk, parent_linked_ids)

        # Удаление лишних полей
        for field in self.fields_to_remove:
            new_chunk.pop(field, None)

        # Убираем старое поле parent_group
        new_chunk.pop("parent_group", None)

        return new_chunk

    def _handle_links(self, chunk: dict, parent_linked_ids: set):
        """
        Логика разделения ссылок на внутренние и внешние.
        """
        original_links = chunk.get("linked_chunks", {})

        external_links = {}
        internal_links = {}

        for l_name, l_ids in original_links.items():
            for l_id in l_ids:
                new_l_id = self.id_map.get(l_id, l_id)

                if l_id in parent_linked_ids:
                    if not external_links.get(l_name, []):
                        external_links[l_name] = []
                    external_links[l_name].append(new_l_id)
                else:
                    if not internal_links.get(l_name, []):
                        internal_links[l_name] = []
                    internal_links[l_name].append(new_l_id)

        chunk["external_links"] = external_links
        chunk["internal_links"] = internal_links
        chunk.pop("linked_chunks", None)

    def _extract_image(self, parent_index, chunk_index, dpi=150):
        with pymupdf.open(self.document_path) as document:
            zoom = dpi / 72
            mat = pymupdf.Matrix(zoom, zoom)

            chunk = self.parent_chunks[parent_index]['chunks'][chunk_index]

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

                filename = f"{self.id_map[chunk['id']]}.png"
                filepath = self.image_folder / filename
                chunk['image_path'] = filepath
                img.save(filepath, "PNG", optimize=True)
            except Exception:
                pass



if __name__ == '__main__':
    p = ChunkStoragePreparer()
    path = Path(__file__).resolve().parent.parent / 'output/output_chunk_processor' / "Alg-graphs-full_images_processed_json_chunk_processed_json.txt"
    p.process(document_path=path)