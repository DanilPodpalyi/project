"""Изолированные запуски, ограничение времени и атомарная запись артефактов."""

import hashlib
import inspect
import json
import os
import tempfile
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from uuid import uuid4

from . import __version__
from .settings import get_settings

_run = ContextVar("resume_evaluator_run", default=None)
_deadline = ContextVar("resume_evaluator_deadline", default=None)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False) as target:
            temporary = Path(target.name)
            target.write(text)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def remaining_timeout(default: float) -> float:
    deadline = _deadline.get()
    if deadline is None:
        return default
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Превышено общее время запуска; измените max_run_seconds или начните новый запуск")
    return min(default, remaining)


def current_run_id() -> str | None:
    state = _run.get()
    return state["run_id"] if state else None


def _save_state(state: dict) -> None:
    public = {key: value for key, value in state.items() if not key.startswith("_")}
    atomic_write_text(state["_directory"] / "run.json", json.dumps(public, ensure_ascii=False, indent=2))


def update_stage(stage: str) -> None:
    remaining_timeout(get_settings().max_run_seconds)
    state = _run.get()
    if state:
        state["stage"] = stage
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        _save_state(state)
        print(f"[run {state['run_id']}] {stage}", flush=True)


def _file_identity(value) -> dict | None:
    if value is None:
        return None
    path = Path(value)
    identity = {"path": str(path.resolve()), "sha256": None}
    if path.is_file():
        identity["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return identity


def managed_run(function):
    """Первый запуск использует заданную папку, следующие — собственные подпапки.

    O_EXCL закрывает гонку между процессами. Удаляется только собственный lock;
    незавершённый запуск и его диагностика сохраняются для пользователя.
    """
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        settings = get_settings()
        inputs = {key: _file_identity(bound.arguments.get(key)) for key in
            ("resume_path", "vacancy_path", "resume_evaluation_path", "market_profile_path", "vacancies_json_path")}
        root = Path(bound.arguments["output_directory"])
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        directory = root
        if any((root / name).exists() for name in ("run.json", ".run.lock", ".stages", "candidate_report.json", "candidate_report.md", "resume.json", "score.json", "vacancy.json")):
            directory = root / f"run-{run_id}"
        directory.mkdir(parents=True, exist_ok=True)
        try:
            handle = os.open(directory / ".run.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            directory = root / f"run-{run_id}"
            directory.mkdir(parents=True, exist_ok=False)
            handle = os.open(directory / ".run.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(handle, "w", encoding="utf-8") as lock:
            json.dump({"run_id": run_id, "pid": os.getpid()}, lock)
        state = {"schema_version": 1, "run_id": run_id, "app_version": __version__,
            "status": "running", "stage": "starting", "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat(), "settings": settings.public_dict(),
            "inputs": inputs,
            "_directory": directory}
        run_token = _run.set(state)
        deadline_token = _deadline.set(time.monotonic() + settings.max_run_seconds)
        bound.arguments["output_directory"] = directory
        try:
            _save_state(state)
            result = function(*bound.args, **bound.kwargs)
            state.update(status="completed", stage="finished")
            if hasattr(result, "report_json"):
                report = json.loads(result.report_json.read_text(encoding="utf-8"))
                state.update(report_status=report["status"],
                    report_json=str(result.report_json.resolve()), report_markdown=str(result.report_markdown.resolve()),
                    report_sha256=hashlib.sha256(result.report_json.read_bytes()).hexdigest())
            else:
                state["artifacts"] = {key: str(getattr(result, key).resolve()) for key in result.__dataclass_fields__}
            return result
        except BaseException as error:
            state["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            # ValidationError может содержать данные резюме — сохраняем только тип.
            state["error_type"] = type(error).__name__
            raise
        finally:
            state["finished_at"] = datetime.now(timezone.utc).isoformat()
            try:
                _save_state(state)
            finally:
                _deadline.reset(deadline_token)
                _run.reset(run_token)
                (directory / ".run.lock").unlink(missing_ok=True)

    return wrapped
