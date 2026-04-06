from pathlib import Path
from dataclasses import dataclass

@dataclass
class ParserConfig:
    """
    Конфигурация парсера.
    """
    document_name: str | None
    input_dir: Path

    # Опции обработки
    process_marker: bool  # Для False обязательно наличие текстового файла для этого документа в папке output для Marker
    process_text: bool
    process_image: bool
    # Если опция маркера не выполнена, то первая опция в цепочке
    # true - выполняется на основе данных из папки output для Marker
    # false - выполняется на основе данных из папки output для предыдущего шага (marker - text - image)
    one_step: bool

    # Настройки модели
    model_name: str

    # Директории
    text_dir: str
    images_dir: str
    marker_processor_output: str
    text_processor_output: str
    image_processor_output: str

    # Сохранение
    save_intermediate:  bool
    delete_images:      bool

