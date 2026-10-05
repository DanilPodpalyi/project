"""Общий CLI; конфигурация задаётся до импорта модулей с настройками модели."""
import argparse
import importlib
import os
import sys
from pathlib import Path

COMMANDS = {"analyze": "candidate_assessment_pipeline", "collect": "russian_market",
    "check": "system_check", "model-check": "model_check", "compare": "legacy.pipeline"}


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Резюме → выборка рынка России → рекомендации",
        epilog="Команды: analyze (основной отчёт), collect (сбор вакансий), check (проверка), model-check (проба модели), compare (сравнение с одной вакансией). Параметры команды: analyze --help")
    parser.add_argument("--config", type=Path, help="TOML-конфигурация; перед командой")
    parser.add_argument("--version", action="version", version="%(prog)s 0.2.0")
    parser.add_argument("command", nargs="?", choices=COMMANDS)
    if not arguments or arguments[0] in {"--help", "-h", "--version"}:
        parser.parse_args(arguments)
        if not arguments:
            parser.print_help()
        return 0
    prefix = 3 if arguments[0] == "--config" else 1
    options = parser.parse_args(arguments[:prefix])
    if options.config:
        os.environ["RESUME_EVALUATOR_CONFIG"] = str(options.config.resolve())
    if not options.command:
        parser.error("Укажите команду")
    previous = sys.argv
    sys.argv = [f"resume-evaluator {options.command}", *arguments[prefix:]]
    try:
        from .settings import get_settings
        get_settings()
        importlib.import_module(f"resume_evaluator.{COMMANDS[options.command]}").main()
        return 0
    except KeyboardInterrupt:
        print("Запуск прерван.", file=sys.stderr)
        return 130
    except Exception as error:
        from pydantic import ValidationError
        from ollama import ResponseError
        if isinstance(error, ValidationError):
            message = "Входной JSON или ответ модели не соответствует схеме. Проверьте формат и модели."
        elif isinstance(error, ResponseError):
            message = f"Ошибка Ollama (HTTP {error.status_code}). Проверьте модели командой check --models."
        elif isinstance(error, (OSError, ValueError)):
            message = str(error)
        else:
            message = f"Ошибка {type(error).__name__}. Для analyze/compare проверьте run.json в папке результата."
        print(message, file=sys.stderr)
        return 2
    finally:
        sys.argv = previous
