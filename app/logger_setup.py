import sys
from pathlib import Path
from loguru import logger

class LogConfig:
    """
    Класс для настройки логирования.
    """
    ROOT_DIR = Path(__file__).resolve().parent.parent
    LOG_DIR = ROOT_DIR / "logs"

    @classmethod
    def setup(cls):
        logger.remove()
        cls.LOG_DIR.mkdir(exist_ok=True)

        logger.add(
            sys.stderr,
            format="<magenta>❱</magenta> <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
            level="DEBUG",
            colorize=True
        )

        logger.add(
            cls.LOG_DIR / "{time:YYYY-MM-DD}_rag_assistant.log",
            rotation="10 MB",
            retention="7 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function} - {message}"
        )

        return logger

log = LogConfig.setup()