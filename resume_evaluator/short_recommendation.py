"""Генерация краткой рекомендации кандидату по результатам сравнения."""

import argparse
import json
import time
from pathlib import Path

from ollama import chat
from pydantic import BaseModel, Field


MODEL_NAME = "gemma4:12b"
MODEL_CONTEXT_SIZE = 16384
MAX_OUTPUT_TOKENS = 2048


class GeneratedRecommendation(BaseModel):
    summary: str = Field(
        description=(
            "Краткий итог соответствия кандидата вакансии, 2–4 предложения"
        )
    )
    strengths: list[str] = Field(
        max_length=3,
        description=(
            "До трёх главных сильных сторон кандидата, подтверждённых "
            "данными резюме и результатами сравнения"
        )
    )
    critical_gaps: list[str] = Field(
        max_length=3,
        description=(
            "До трёх обязательных навыков, не подтверждённых в резюме. "
            "Не включать сюда желательные требования"
        )
    )
    additional_gaps: list[str] = Field(
        max_length=2,
        description=(
            "До двух желательных навыков, не подтверждённых в резюме"
        )
    )
    candidate_recommendations: list[str] = Field(
        max_length=3,
        description=(
            "До трёх кратких конкретных действий кандидата для улучшения "
            "резюме и соответствия вакансии"
        )
    )


class RecommendationResult(BaseModel):
    candidate_name: str | None
    vacancy_title: str | None
    total_score: float
    match_level: str

    summary: str
    strengths: list[str]
    critical_gaps: list[str]
    additional_gaps: list[str]
    candidate_recommendations: list[str]


SYSTEM_PROMPT = """
Ты формируешь краткую оценку резюме для самого кандидата по результатам
сравнения с вакансией.

Тебе передаются:
1. структурированные данные резюме;
2. структурированные данные вакансии;
3. уже рассчитанный программой результат сравнения.

Правила:
1. Не пересчитывай итоговый балл и не предлагай другой балл.
2. Не оспаривай установленный уровень соответствия.
3. Используй только сведения из переданных данных.
4. Не придумывай навыки, опыт, достижения и образование кандидата.
5. Если навык отсутствует в резюме, пиши «не указан» или
   «не подтверждён в резюме», а не утверждай, что кандидат им не владеет.
6. В critical_gaps включай только отсутствующие обязательные требования.
7. Отсутствующие желательные навыки включай только в additional_gaps.
8. Не превращай желательные требования в обязательные.
9. Сильные стороны должны подтверждаться резюме и результатами сравнения.
10. Рекомендации кандидату должны быть краткими, конкретными и применимыми.
11. Не формируй рекомендации работодателю и вопросы для собеседования.
12. Обращай весь ответ к кандидату и оценивай только содержание его резюме.
13. Не используй имя, пол, возраст, адрес и другие личные характеристики
    как основания для оценки.
14. Не повторяй один и тот же вывод в нескольких разделах без необходимости.
15. Summary должен состоять из 2–3 коротких предложений.
16. Верни не более трёх сильных сторон, трёх обязательных пробелов, двух
    желательных пробелов и трёх рекомендаций.
17. Если навык не подтверждён, не советуй просто добавить его в резюме.
    Пиши: «если опыт есть — добавьте пример; если нет — изучите или получите
    практику». Не подталкивай кандидата указывать несуществующий опыт.
18. Возвращай только данные по заданной JSON-схеме.
"""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Формирование рекомендации по результатам оценки резюме"
    )
    parser.add_argument(
        "resume",
        type=Path,
        help="Путь к структурированному JSON резюме",
    )
    parser.add_argument(
        "vacancy",
        type=Path,
        help="Путь к структурированному JSON вакансии",
    )
    parser.add_argument(
        "score",
        type=Path,
        help="Путь к результату программной оценки",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Путь для сохранения рекомендации",
    )
    return parser.parse_args()


def read_json(file_path: Path) -> dict:
    if not file_path.exists():
        raise SystemExit(f"Файл не найден: {file_path}")

    if not file_path.is_file():
        raise SystemExit(
            f"Указанный путь не является файлом: {file_path}"
        )

    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(
            f"Некорректный JSON в файле {file_path}: {error}"
        ) from error


def generate_recommendation(
    resume: dict,
    vacancy: dict,
    score: dict,
) -> GeneratedRecommendation:
    context = {
        "resume": resume,
        "vacancy": vacancy,
        "calculated_score": score,
    }

    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    "Сформируй рекомендацию по следующим данным:\n\n"
                    + json.dumps(
                        context,
                        ensure_ascii=False,
                        indent=2,
                    )
                ),
            },
        ],
        format=GeneratedRecommendation.model_json_schema(),
        think=False,
        options={
            "temperature": 0,
            "num_ctx": MODEL_CONTEXT_SIZE,
            "num_predict": MAX_OUTPUT_TOKENS,
        },
    )

    return GeneratedRecommendation.model_validate_json(
        response.message.content
    )


def build_result(
    score: dict,
    generated: GeneratedRecommendation,
) -> RecommendationResult:
    return RecommendationResult(
        candidate_name=score.get("candidate_name"),
        vacancy_title=score.get("vacancy_title"),
        total_score=score.get("total_score", 0),
        match_level=score.get("recommendation", ""),
        summary=generated.summary,
        strengths=generated.strengths,
        critical_gaps=generated.critical_gaps,
        additional_gaps=generated.additional_gaps,
        candidate_recommendations=generated.candidate_recommendations,
    )


def save_result(
    result: RecommendationResult,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )

def print_list(title: str, items: list[str]) -> None:
    print(f"\n{title}")

    if not items:
        print("  Не выявлено.")
        return

    for item in items:
        print(f"  • {item}")


def print_recommendation(result: RecommendationResult) -> None:
    print("\n" + "=" * 70)
    print("КРАТКАЯ ОЦЕНКА РЕЗЮМЕ")
    print("=" * 70)

    print(f"\nКандидат: {result.candidate_name or 'Не указан'}")
    print(f"Вакансия: {result.vacancy_title or 'Не указана'}")
    print(f"Оценка: {result.total_score}/100")
    print(f"Уровень соответствия: {result.match_level}")

    print("\nКРАТКИЙ ВЫВОД")
    print(result.summary)

    print_list("СИЛЬНЫЕ СТОРОНЫ", result.strengths)
    print_list(
        "НЕ ПОДТВЕРЖДЕНЫ ОБЯЗАТЕЛЬНЫЕ НАВЫКИ",
        result.critical_gaps,
    )
    print_list(
        "НЕ ПОДТВЕРЖДЕНЫ ЖЕЛАТЕЛЬНЫЕ НАВЫКИ",
        result.additional_gaps,
    )

    print_list(
        "РЕКОМЕНДАЦИИ КАНДИДАТУ",
        result.candidate_recommendations,
    )

    print("\n" + "=" * 70)

def main() -> None:
    args = parse_arguments()

    resume = read_json(args.resume)
    vacancy = read_json(args.vacancy)
    score = read_json(args.score)

    print(f"Формируется рекомендация для: {score.get('candidate_name')}")

    start_time = time.perf_counter()

    generated = generate_recommendation(
        resume=resume,
        vacancy=vacancy,
        score=score,
    )
    result = build_result(score, generated)

    elapsed_time = time.perf_counter() - start_time

    save_result(result, args.output)

    print_recommendation(result)

    print(f"\nРекомендация сохранена: {args.output}")
    print(f"Время обработки: {elapsed_time:.1f} сек.")


if __name__ == "__main__":
    main()
