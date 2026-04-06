from pathlib import Path
from dataclasses import dataclass

@dataclass
class ChunkerConfig:
    """
    Конфигурация парсера.
    """
    document_name: str | None

    # Настройки для определения полного имени файла
    has_suffix: bool # К имени документа нужно добавить суффикс
    suffix: str

    # Директории
    output_dir: str | Path
    input_dir: Path

    # Сохранение
    save_intermediate:  bool