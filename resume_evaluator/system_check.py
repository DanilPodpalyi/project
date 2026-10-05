"""Быстрая проверка зависимостей, схем и JSON без запуска модели."""

import argparse
import json
import sys
from uuid import uuid4
from urllib.request import urlopen
from importlib.util import find_spec
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from .candidate_scorer import ScoreResult, score_candidate
from .detailed_recommendation import DetailedRecommendationResult
from .document_reader import SUPPORTED_EXTENSIONS
from .resume_extractor import ResumeData
from .short_recommendation import RecommendationResult
from .vacancy_extractor import VacancyData
from .settings import get_settings
from .runtime import atomic_write_text


REQUIRED_PACKAGES = ("docx", "ollama", "pydantic", "pypdf", "pymorphy3", "httpx")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Лёгкая проверка системы без загрузки модели Ollama"
    )
    parser.add_argument("--models", action="store_true", help="Проверить Ollama и наличие обеих моделей")
    parser.add_argument("--output-directory", type=Path, help="Проверить запись в папку результатов")
    parser.add_argument("--sources", metavar="TITLE", help="Проверить API HH/Работы России по должности (без LLM)")
    parser.add_argument(
        "--resume",
        type=Path,
        help="Путь к JSON резюме для проверки и тестового сравнения",
    )
    parser.add_argument(
        "--vacancy",
        type=Path,
        help="Путь к JSON вакансии для проверки и тестового сравнения",
    )
    parser.add_argument(
        "--score",
        type=Path,
        help="Путь к существующему JSON оценки для проверки",
    )
    parser.add_argument(
        "--recommendation",
        type=Path,
        help="Путь к существующему JSON рекомендации для проверки",
    )
    parser.add_argument(
        "--detailed-recommendation",
        type=Path,
        help="Путь к подробному JSON рекомендации для проверки",
    )
    return parser.parse_args()


def validate_required_schema(model: type[BaseModel]) -> None:
    schema = model.model_json_schema()
    properties = set(schema.get("properties", {}))
    required = set(schema.get("required", []))

    if properties != required:
        optional = ", ".join(sorted(properties - required))
        raise ValueError(
            f"В схеме {model.__name__} необязательные поля: {optional}"
        )

    for name, definition in schema.get("$defs", {}).items():
        nested_properties = set(definition.get("properties", {}))
        nested_required = set(definition.get("required", []))
        if nested_properties != nested_required:
            optional = ", ".join(
                sorted(nested_properties - nested_required)
            )
            raise ValueError(
                f"В схеме {model.__name__}.{name} необязательные поля: "
                f"{optional}"
            )


def read_model(file_path: Path, model: type[BaseModel]) -> BaseModel:
    if not file_path.is_file():
        raise ValueError(f"Файл не найден: {file_path}")

    return model.model_validate_json(file_path.read_text(encoding="utf-8"))


def run_check(label: str, check: Callable[[], object]) -> bool:
    try:
        check()
    except Exception as error:
        from pydantic import ValidationError
        detail = "JSON не соответствует схеме" if isinstance(error, ValidationError) else str(error)
        print(f"[FAIL] {label}: {detail}")
        return False

    print(f"[OK]   {label}")
    return True


def check_packages() -> None:
    if sys.version_info < (3, 11):
        raise ValueError("Требуется Python 3.11 или новее")
    missing = [name for name in REQUIRED_PACKAGES if find_spec(name) is None]
    if missing:
        raise ValueError(f"Не установлены пакеты: {', '.join(missing)}")


def check_models() -> None:
    settings = get_settings()
    with urlopen(settings.ollama_host.rstrip("/") + "/api/tags", timeout=10) as response:
        installed = {item["name"] for item in json.load(response).get("models", [])}
    required = [name if ":" in name else name + ":latest" for name in (settings.model, settings.embedding_model)]
    missing = [name for name in required if name not in installed]
    if missing:
        raise ValueError("Не установлены модели: " + ", ".join(missing))
    print("      Модели: " + ", ".join(required))


def check_output(directory: Path) -> None:
    path = directory / f".write-check-{uuid4().hex}.tmp"
    try:
        atomic_write_text(path, "ok")
        if path.read_text(encoding="utf-8") != "ok":
            raise OSError("Не удалось прочитать записанный файл")
    finally:
        path.unlink(missing_ok=True)


def check_sources(title: str, directory: Path) -> None:
    from .market_profile import TargetProfile
    from .russian_market import collect_russian_vacancies
    path = directory / "source_check.json"
    result = collect_russian_vacancies(TargetProfile(role=title, specialization=title, domain=None, seniority=None, core_skills=[]), path,
        max_pages=1, detail_limit=1, use_cache=False, use_title_translation=False)
    diagnostic = result.diagnostics
    print(f"      API: {diagnostic.status}; сохранено {diagnostic.accepted}; диагностика: {path}")
    for issue in diagnostic.errors:
        print(f"      {issue.source}: {issue.reason}")
    if diagnostic.status != "ok":
        raise ValueError("Один или несколько источников недоступны. Детали — в source_check.json")


def check_document_formats() -> None:
    expected = {".txt", ".docx", ".pdf"}
    if SUPPORTED_EXTENSIONS != expected:
        raise ValueError(
            f"Ожидались форматы {sorted(expected)}, "
            f"получены {sorted(SUPPORTED_EXTENSIONS)}"
        )


def main() -> None:
    args = parse_arguments()
    results = [
        run_check("Настройки", get_settings),
        run_check("Зависимости Python", check_packages),
        run_check("Форматы TXT/DOCX/PDF", check_document_formats),
        run_check(
            "Полнота схемы резюме",
            lambda: validate_required_schema(ResumeData),
        ),
        run_check(
            "Полнота схемы вакансии",
            lambda: validate_required_schema(VacancyData),
        ),
    ]
    if args.models:
        results.append(run_check("Ollama и модели", check_models))
    if args.output_directory:
        results.append(run_check("Запись результатов", lambda: check_output(args.output_directory)))
    if args.sources:
        results.append(run_check("Источники вакансий", lambda: check_sources(args.sources, args.output_directory or Path("output/health"))))

    resume: ResumeData | None = None
    vacancy: VacancyData | None = None

    if bool(args.resume) != bool(args.vacancy):
        print("[FAIL] Для сравнения укажите одновременно --resume и --vacancy")
        results.append(False)
    elif args.resume and args.vacancy:
        try:
            resume = read_model(args.resume, ResumeData)
            print("[OK]   JSON резюме")
        except Exception as error:
            print(f"[FAIL] JSON резюме: {error}")
            results.append(False)

        try:
            vacancy = read_model(args.vacancy, VacancyData)
            print("[OK]   JSON вакансии")
        except Exception as error:
            print(f"[FAIL] JSON вакансии: {error}")
            results.append(False)

        if resume and vacancy:
            result = score_candidate(
                resume.model_dump(),
                vacancy.model_dump(),
            )
            print(
                f"[OK]   Сравнение без модели: "
                f"{result.total_score}/100 — {result.recommendation}"
            )

    if args.score:
        results.append(
            run_check(
                "JSON оценки",
                lambda: read_model(args.score, ScoreResult),
            )
        )

    if args.recommendation:
        results.append(
            run_check(
                "JSON рекомендации",
                lambda: read_model(args.recommendation, RecommendationResult),
            )
        )

    if args.detailed_recommendation:
        results.append(
            run_check(
                "Подробный JSON рекомендации",
                lambda: read_model(
                    args.detailed_recommendation,
                    DetailedRecommendationResult,
                ),
            )
        )

    if not all(results):
        raise SystemExit(1)

    print("\nПроверки пройдены. Генерация текста и embeddings не выполнялись.")


if __name__ == "__main__":
    main()
