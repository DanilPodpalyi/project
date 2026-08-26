"""Быстрая проверка зависимостей, схем и JSON без запуска модели."""

import argparse
from importlib.util import find_spec
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from document_reader import SUPPORTED_EXTENSIONS
from extract_resume import ResumeData
from extract_vacancy import VacancyData
from generate_detailed_recommendation import DetailedRecommendationResult
from generate_recommendation import RecommendationResult
from score_candidate import ScoreResult, score_candidate


REQUIRED_PACKAGES = ("docx", "ollama", "pydantic", "pypdf")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Лёгкая проверка системы без загрузки модели Ollama"
    )
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
        print(f"[FAIL] {label}: {error}")
        return False

    print(f"[OK]   {label}")
    return True


def check_packages() -> None:
    missing = [name for name in REQUIRED_PACKAGES if find_spec(name) is None]
    if missing:
        raise ValueError(f"Не установлены пакеты: {', '.join(missing)}")


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

    print("\nСистема исправна. Модель Ollama не загружалась.")


if __name__ == "__main__":
    main()
