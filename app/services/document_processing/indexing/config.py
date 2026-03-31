from pathlib import Path
from dataclasses import dataclass

# Корень проекта
APP_ROOT = Path(__file__).resolve().parents[3]

# Пути
IMAGES_DIR = APP_ROOT / "rag" / "data" / "images"

# Qdrant
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "chunks"

# Эмбеддинги
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm25"

@dataclass
class StorageConfig:
    """
    Конфигурация парсера.
    """
    document_name: str | None = None # При отсутствии будут браться все файлы из input_dir по очереди
    input_dir: Path = Path(__file__).resolve().parent.parent.parent.parent / 'data' #TODO


    # Опции обработки
    process_marker: bool = False # Для False обязательно наличие текстового файла для этого документа в папке output для Marker
    process_text: bool = False
    process_image: bool = True
    # Если опция маркера не выполнена, то первая опция в цепочке
    # true - выполняется на основе данных из папки output для Marker
    # false - выполняется на основе данных из папки output для предыдущего шага (marker - text - image)
    one_step: bool = False

    # Настройки модели
    model_name: str = 'qwen3.5:9b'

    # Директории
    text_dir: str = 'extracted_formulas_images'
    images_dir: str = 'extracted_images'
    marker_processor_output: str = "output_marker_processor"
    text_processor_output: str = "output_text_processor"
    image_processor_output: str = "output_image_processor"

    # Сохранение
    save_intermediate:  bool = True
    delete_images:      bool = False