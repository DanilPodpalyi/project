"""Оркестрация полного конвейера анализа резюме и вакансии."""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from extract_resume import (
    extract_resume,
    read_resume,
    save_result as save_resume_result,
)
from extract_vacancy import (
    extract_vacancy,
    read_vacancy,
    save_result as save_vacancy_result,
)
from generate_detailed_recommendation import (
    build_result as build_detailed_result,
    generate_detailed_recommendation,
    save_result as save_detailed_result,
)
from generate_recommendation import (
    build_result as build_short_result,
    generate_recommendation as generate_short_recommendation,
    save_result as save_short_result,
)
from score_candidate import save_result as save_score_result
from score_candidate import score_candidate


@dataclass(frozen=True)
class PipelinePaths:
    resume: Path
    vacancy: Path
    score: Path
    short_recommendation: Path
    detailed_recommendation: Path

    @classmethod
    def from_output_directory(cls, output_directory: Path) -> "PipelinePaths":
        return cls(
            resume=output_directory / "resume.json",
            vacancy=output_directory / "vacancy.json",
            score=output_directory / "score.json",
            short_recommendation=output_directory / "recommendation_short.json",
            detailed_recommendation=(
                output_directory / "recommendation_detailed.json"
            ),
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Полный анализ соответствия резюме вакансии с краткой и "
            "подробной рекомендациями кандидату"
        )
    )
    parser.add_argument(
        "resume",
        type=Path,
        help="Резюме в формате TXT, DOCX или PDF",
    )
    parser.add_argument(
        "vacancy",
        type=Path,
        help="Вакансия в формате TXT, DOCX или PDF",
    )
    parser.add_argument(
        "output_directory",
        type=Path,
        help="Каталог для пяти итоговых JSON-файлов",
    )
    return parser.parse_args()


def print_stage(number: int, title: str) -> None:
    print(f"\n[{number}/5] {title}")


def run_pipeline(
    resume_path: Path,
    vacancy_path: Path,
    output_directory: Path,
) -> PipelinePaths:
    paths = PipelinePaths.from_output_directory(output_directory)
    started_at = time.perf_counter()

    print_stage(1, "Извлечение данных из резюме")
    resume_text = read_resume(resume_path)
    resume = extract_resume(resume_text)
    save_resume_result(resume, paths.resume)
    print(f"Сохранено: {paths.resume}")

    print_stage(2, "Извлечение данных из вакансии")
    vacancy_text = read_vacancy(vacancy_path)
    vacancy = extract_vacancy(vacancy_text)
    save_vacancy_result(vacancy, paths.vacancy)
    print(f"Сохранено: {paths.vacancy}")

    resume_data = resume.model_dump()
    vacancy_data = vacancy.model_dump()

    print_stage(3, "Расчёт соответствия")
    score = score_candidate(resume_data, vacancy_data)
    save_score_result(score, paths.score)
    score_data = score.model_dump()
    print(f"Оценка: {score.total_score}/100 — {score.recommendation}")
    print(f"Сохранено: {paths.score}")

    print_stage(4, "Краткая рекомендация кандидату")
    short_generated = generate_short_recommendation(
        resume_data,
        vacancy_data,
        score_data,
    )
    short_result = build_short_result(score_data, short_generated)
    save_short_result(short_result, paths.short_recommendation)
    print(f"Сохранено: {paths.short_recommendation}")

    print_stage(5, "Подробная рекомендация кандидату")
    detailed_generated = generate_detailed_recommendation(
        resume_data,
        vacancy_data,
        score_data,
    )
    detailed_result = build_detailed_result(score_data, detailed_generated)
    save_detailed_result(detailed_result, paths.detailed_recommendation)
    print(f"Сохранено: {paths.detailed_recommendation}")

    elapsed_time = time.perf_counter() - started_at
    print("\nАнализ завершён успешно.")
    print(f"Общее время: {elapsed_time:.1f} сек.")

    return paths


def main() -> None:
    args = parse_arguments()
    run_pipeline(args.resume, args.vacancy, args.output_directory)


if __name__ == "__main__":
    main()
