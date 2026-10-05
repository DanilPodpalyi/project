"""Настройки одного процесса. Секрет HH задаётся только через окружение."""

import os
import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class Settings:
    model: str = "gemma4:12b"
    embedding_model: str = "bge-m3:latest"
    ollama_host: str = "http://127.0.0.1:11434"
    context_size: int = 16384
    ollama_timeout_seconds: float = 600
    max_run_seconds: float = 3600
    http_timeout_seconds: float = 20
    search_max_pages: int = 3
    detail_limit: int = 80
    response_cache_hours: float = 24

    def public_dict(self) -> dict:
        """Настройки пригодны для run.json: токены в этот объект не входят."""
        return asdict(self)


ENVIRONMENT = {
    "model": "RESUME_EVALUATOR_MODEL",
    "embedding_model": "OLLAMA_EMBEDDING_MODEL",
    "ollama_host": "OLLAMA_HOST",
    "context_size": "RESUME_EVALUATOR_CONTEXT_SIZE",
    "ollama_timeout_seconds": "RESUME_EVALUATOR_OLLAMA_TIMEOUT",
    "max_run_seconds": "RESUME_EVALUATOR_MAX_RUN_SECONDS",
    "http_timeout_seconds": "RESUME_EVALUATOR_HTTP_TIMEOUT",
    "search_max_pages": "RESUME_EVALUATOR_SEARCH_MAX_PAGES",
    "detail_limit": "RESUME_EVALUATOR_DETAIL_LIMIT",
    "response_cache_hours": "RESUME_EVALUATOR_CACHE_HOURS",
}


def get_settings() -> Settings:
    values = Settings().public_dict()
    config_path = os.getenv("RESUME_EVALUATOR_CONFIG")
    if config_path:
        with Path(config_path).open("rb") as source:
            config = tomllib.load(source)
        if set(config) - {"resume_evaluator"}:
            raise ValueError("В конфигурации разрешён только раздел [resume_evaluator]")
        configured = config.get("resume_evaluator", {})
        if not isinstance(configured, dict):
            raise ValueError("[resume_evaluator] должен быть TOML-разделом")
        unknown = set(configured) - set(values)
        if unknown:
            raise ValueError(f"Неизвестные настройки: {', '.join(sorted(unknown))}")
        values.update(configured)
    for key, variable in ENVIRONMENT.items():
        if variable in os.environ:
            values[key] = os.environ[variable]
    for key in ("context_size", "search_max_pages", "detail_limit"):
        if isinstance(values[key], bool) or isinstance(values[key], float) and not values[key].is_integer():
            raise ValueError(f"Настройка {key} должна быть целым числом")
        values[key] = int(values[key])
    for key in ("ollama_timeout_seconds", "max_run_seconds", "http_timeout_seconds", "response_cache_hours"):
        if isinstance(values[key], bool):
            raise ValueError(f"Настройка {key} должна быть числом")
        values[key] = float(values[key])
    if any(not math.isfinite(values[key]) or values[key] <= 0 for key in values if isinstance(values[key], (int, float))):
        raise ValueError("Размеры и тайм-ауты должны быть положительными")
    if values["search_max_pages"] > 20 or values["detail_limit"] > 200:
        raise ValueError("Лимиты поиска: максимум 20 страниц и 200 полных карточек")
    for key in ("model", "embedding_model", "ollama_host"):
        if not isinstance(values[key], str) or not values[key].strip():
            raise ValueError(f"Настройка {key} должна быть непустой строкой")
        values[key] = values[key].strip()
    host = urlparse(values["ollama_host"])
    if host.scheme not in {"http", "https"} or not host.hostname or host.username or host.password:
        raise ValueError("OLLAMA_HOST должен быть HTTP(S)-адресом без встроенного пароля")
    return Settings(**values)
