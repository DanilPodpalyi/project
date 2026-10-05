"""Единый Ollama-клиент с тайм-аутом и безопасными сетевыми ошибками."""

import httpx
from ollama import Client

from .runtime import remaining_timeout
from .settings import get_settings


class ModelUnavailableError(OSError):
    """Содержит только техническую диагностику, без входных документов."""


def _client(host: str, timeout: float) -> Client:
    return Client(host=host, timeout=httpx.Timeout(timeout, connect=min(10, timeout)))


def _request(method: str, *args, **kwargs):
    settings = get_settings()
    timeout = remaining_timeout(settings.ollama_timeout_seconds)
    client = _client(settings.ollama_host, timeout)
    try:
        return getattr(client, method)(*args, **kwargs)
    except (httpx.RequestError, ConnectionError) as error:
        raise ModelUnavailableError(
            f"Ollama недоступна или превышен тайм-аут ({timeout:.0f} с). "
            "Проверьте сервер, модели и настройки."
        ) from error
    finally:
        # SDK 0.6.2 не предоставляет close/context manager; закрываем его HTTP-клиент.
        client._client.close()


def chat(*args, **kwargs):
    return _request("chat", *args, **kwargs)


def embed(*args, **kwargs):
    return _request("embed", *args, **kwargs)
