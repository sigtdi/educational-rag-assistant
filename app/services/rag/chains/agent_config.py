from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Any
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_CONFIG_YAML  = _PROJECT_ROOT / "app" / "config.yaml"

@dataclass
class AgentConfig:
    # LLM
    model_name: str
    temperature: float
    system_prompt: str = (  'Ты — учебный ассистент по алгоритмам и структурам данных (графы, обходы, сортировки и др.).'
                            'Перед ответом на любой вопрос ты обязан использовать search_tool для поиска релевантной '
                            'информации.'
                            'Отвечай строго на основе найденных материалов. Не придумывай и не добавляй информацию, '
                            'которой нет в найденных документах.'
                            'Если поиск не вернул релевантных результатов — прямо сообщи об этом пользователю.'
                            'Отвечай чётко и структурированно. При объяснении алгоритма указывай его ключевые шаги и '
                            'сложность, если они есть в источниках.')

    @classmethod
    def from_yaml(cls, yaml_path: Path = _CONFIG_YAML) -> "AgentConfig":
        """
        Читает config.yaml и возвращает RetrieverConfig.
        """
        raw = _load_yaml(yaml_path)
        return _build_agent_config(raw["agent"])

def _build_agent_config(s: dict[str, Any]) -> AgentConfig:
    # Если задан qdrant_url — используем его, иначе собираем из host:port
    return AgentConfig(
        model_name=s['model_name'],
        temperature=s['temperature']
    )

def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}