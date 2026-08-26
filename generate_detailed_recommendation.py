"""Генерация подробной доказательной рекомендации для кандидата."""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Literal

from ollama import chat
from pydantic import BaseModel, Field


MODEL_NAME = "gemma4:12b"
MODEL_CONTEXT_SIZE = 16384
MAX_OUTPUT_TOKENS = 8192

GapCategory = Literal[
    "не указан в резюме",
    "не показан на примерах",
    "подтверждён слабо или неочевидно",
]
Priority = Literal["высокий", "средний", "низкий"]


class DetailedStrength(BaseModel):
    strength: str = Field(
        description="Конкретная сильная сторона кандидата",
    )
    resume_evidence: list[str] = Field(
        min_length=1,
        max_length=3,
        description="Факты и результаты из резюме, подтверждающие вывод",
    )
    vacancy_relevance: str = Field(
        description="Почему эта сильная сторона важна для вакансии",
    )


class DetailedGap(BaseModel):
    requirement: str = Field(
        description="Навык или требование вакансии",
    )
    category: GapCategory = Field(
        description=(
            "Состояние требования в резюме: не указано, упомянуто без "
            "примеров либо подтверждено слабо или косвенно"
        ),
    )
    analysis: str = Field(
        description="Почему выбрана эта категория на основе резюме",
    )
    missing_evidence: str = Field(
        description="Какого подтверждения или детализации не хватает",
    )
    priority: Priority = Field(
        description="Приоритет с учётом обязательности требования вакансии",
    )


class ResumeRecommendation(BaseModel):
    related_requirement: str | None = Field(
        description=(
            "Связанное неподтверждённое требование вакансии либо null, "
            "если совет улучшает уже подтверждённую часть резюме"
        ),
    )
    section: str = Field(
        description="Раздел резюме, который следует изменить",
    )
    recommendation: str = Field(
        description="Конкретное изменение резюме",
    )
    suggested_wording: str | None = Field(
        description=(
            "Пример формулировки только из подтверждённых фактов либо null, "
            "если без дополнительных сведений кандидата пример невозможен"
        ),
    )
    condition: str | None = Field(
        description=(
            "Условие применения совета, если сначала нужно подтвердить опыт"
        ),
    )
    priority: Priority = Field(
        description="Приоритет изменения",
    )


class DevelopmentRecommendation(BaseModel):
    area: str = Field(
        description="Навык или область развития",
    )
    recommendation: str = Field(
        description="Конкретное действие для развития",
    )
    practical_step: str = Field(
        description="Практическое упражнение, учебный кейс или следующий шаг",
    )
    expected_result: str = Field(
        description="Проверяемый результат выполнения рекомендации",
    )
    priority: Priority = Field(
        description="Приоритет развития",
    )


class GeneratedDetailedRecommendation(BaseModel):
    overall_summary: str = Field(
        description="Общий итог соответствия кандидата вакансии, 4–6 предложений",
    )
    strengths: list[DetailedStrength] = Field(
        min_length=1,
        max_length=6,
        description="Сильные стороны с доказательствами и связью с вакансией",
    )
    gaps: list[DetailedGap] = Field(
        description="Подробный разбор недостающих или слабых подтверждений",
    )
    resume_recommendations: list[ResumeRecommendation] = Field(
        min_length=1,
        max_length=6,
        description="Конкретные рекомендации по изменению резюме",
    )
    development_recommendations: list[DevelopmentRecommendation] = Field(
        max_length=6,
        description="Конкретные рекомендации по развитию кандидата",
    )


class DetailedRecommendationResult(BaseModel):
    candidate_name: str | None
    vacancy_title: str | None
    total_score: float
    match_level: str
    overall_summary: str
    strengths: list[DetailedStrength]
    gaps: list[DetailedGap]
    resume_recommendations: list[ResumeRecommendation]
    development_recommendations: list[DevelopmentRecommendation]


SYSTEM_PROMPT = """
Ты формируешь подробную оценку резюме для самого кандидата по результатам
сравнения с вакансией.

Тебе передаются:
1. структурированные данные резюме;
2. структурированные данные вакансии;
3. уже рассчитанный программой результат сравнения.

Правила:
1. Не пересчитывай итоговый балл и не оспаривай уровень соответствия.
2. Используй только сведения из переданных данных и не придумывай опыт.
3. Обращай весь ответ к кандидату. Не давай рекомендаций работодателю и не
   формируй вопросы для собеседования.
4. Для каждой сильной стороны укажи конкретное подтверждение из резюме и
   объясни, почему оно подходит вакансии.
5. Не называй навык отсутствующим у кандидата. Оценивай только то, как он
   представлен в резюме.
6. Для каждого существенного требования выбери ровно одну категорию:
   - «не указан в резюме» — навык и связанный опыт нигде не упомянуты;
   - «не показан на примерах» — точное название навыка есть в резюме, но нет
     задачи, кейса или результата;
   - «подтверждён слабо или неочевидно» — есть косвенные или частичные признаки,
     но связь с требованием вакансии недостаточно ясна.
7. Не включай в gaps навыки с прямыми и достаточными подтверждениями.
8. Каждый элемент missing_required_skills и missing_preferred_skills из
   calculated_score обязательно включи отдельным элементом gaps. Обязательные
   требования имеют более высокий приоритет, чем желательные.
9. Рекомендации по резюме должны называть конкретный раздел и изменение.
10. Suggested_wording может использовать только факты из резюме. Если для
    формулировки нужны неизвестные факты, верни null и укажи условие.
    В related_requirement укажи требование из gaps, к которому относится совет.
11. Не советуй просто добавить отсутствующий навык. Если опыт есть, предложи
    показать его примером; если опыта нет, перенеси совет в рекомендации по
    развитию.
12. Рекомендации по развитию должны содержать практический шаг и проверяемый
    результат, а не общие советы вроде «изучить тему».
13. Не повторяй один вывод в разных разделах без необходимости.
14. Возвращай только данные по заданной JSON-схеме.
"""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Подробная рекомендация кандидату по оценке резюме"
    )
    parser.add_argument("resume", type=Path, help="JSON резюме")
    parser.add_argument("vacancy", type=Path, help="JSON вакансии")
    parser.add_argument("score", type=Path, help="JSON программной оценки")
    parser.add_argument("output", type=Path, help="Итоговый подробный JSON")
    return parser.parse_args()


def read_json(file_path: Path) -> dict:
    if not file_path.is_file():
        raise SystemExit(f"Файл не найден: {file_path}")

    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(
            f"Некорректный JSON в файле {file_path}: {error}"
        ) from error


def generate_detailed_recommendation(
    resume: dict,
    vacancy: dict,
    score: dict,
) -> GeneratedDetailedRecommendation:
    context = {
        "resume": resume,
        "vacancy": vacancy,
        "calculated_score": score,
    }
    response = chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Сформируй подробную оценку по следующим данным:\n\n"
                    + json.dumps(context, ensure_ascii=False, indent=2)
                ),
            },
        ],
        format=GeneratedDetailedRecommendation.model_json_schema(),
        think=False,
        options={
            "temperature": 0,
            "num_ctx": MODEL_CONTEXT_SIZE,
            "num_predict": MAX_OUTPUT_TOKENS,
        },
    )
    generated = GeneratedDetailedRecommendation.model_validate_json(
        response.message.content
    )
    correct_gap_categories(generated, resume)
    ensure_score_gaps(generated, score)
    make_resume_recommendations_safe(generated)
    return generated


def _normalize_requirement(value: str) -> str:
    return " ".join(value.casefold().replace("ё", "е").split())


def _search_tokens(value: str) -> set[str]:
    normalized = value.casefold().replace("ё", "е")
    return set(re.findall(r"[a-zа-я0-9]+", normalized))


def correct_gap_categories(
    generated: GeneratedDetailedRecommendation,
    resume: dict,
) -> None:
    resume_text = json.dumps(resume, ensure_ascii=False)
    normalized_resume = " ".join(
        re.findall(
            r"[a-zа-я0-9]+",
            resume_text.casefold().replace("ё", "е"),
        )
    )
    resume_tokens = _search_tokens(resume_text)

    for gap in generated.gaps:
        if gap.category != "не показан на примерах":
            continue

        normalized_requirement = " ".join(
            re.findall(
                r"[a-zа-я0-9]+",
                gap.requirement.casefold().replace("ё", "е"),
            )
        )
        if normalized_requirement in normalized_resume:
            continue

        requirement_tokens = _search_tokens(gap.requirement)
        if requirement_tokens & resume_tokens:
            gap.category = "подтверждён слабо или неочевидно"
        else:
            gap.category = "не указан в резюме"


def ensure_score_gaps(
    generated: GeneratedDetailedRecommendation,
    score: dict,
) -> None:
    existing = {
        _normalize_requirement(gap.requirement)
        for gap in generated.gaps
    }
    requirements = [
        (requirement, "высокий")
        for requirement in score.get("missing_required_skills", [])
    ]
    requirements.extend(
        (requirement, "низкий")
        for requirement in score.get("missing_preferred_skills", [])
    )

    for requirement, priority in requirements:
        normalized = _normalize_requirement(requirement)
        if normalized in existing:
            continue

        generated.gaps.append(
            DetailedGap(
                requirement=requirement,
                category="не указан в резюме",
                analysis=(
                    "Прямое подтверждение навыка не найдено при сравнении "
                    "резюме с требованиями вакансии."
                ),
                missing_evidence=(
                    "Название навыка и конкретный пример его применения "
                    "с задачей или результатом."
                ),
                priority=priority,
            )
        )
        existing.add(normalized)


def make_resume_recommendations_safe(
    generated: GeneratedDetailedRecommendation,
) -> None:
    gaps = {
        _normalize_requirement(gap.requirement): gap
        for gap in generated.gaps
    }

    for recommendation in generated.resume_recommendations:
        if not recommendation.related_requirement:
            continue

        normalized = _normalize_requirement(
            recommendation.related_requirement
        )
        gap = gaps.get(normalized)
        if gap is None:
            continue

        recommendation.recommendation = (
            f"Если у вас есть подтверждённый опыт по требованию "
            f"«{gap.requirement}», добавьте в раздел «{recommendation.section}» "
            "конкретную задачу и результат. Если опыта нет, не указывайте "
            "этот навык до получения практики."
        )
        recommendation.suggested_wording = None
        recommendation.condition = (
            "Применяйте совет только при наличии реального опыта, который "
            "можно подтвердить примером."
        )


def build_result(
    score: dict,
    generated: GeneratedDetailedRecommendation,
) -> DetailedRecommendationResult:
    return DetailedRecommendationResult(
        candidate_name=score.get("candidate_name"),
        vacancy_title=score.get("vacancy_title"),
        total_score=score.get("total_score", 0),
        match_level=score.get("recommendation", ""),
        overall_summary=generated.overall_summary,
        strengths=generated.strengths,
        gaps=generated.gaps,
        resume_recommendations=generated.resume_recommendations,
        development_recommendations=generated.development_recommendations,
    )


def save_result(
    result: DetailedRecommendationResult,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def print_result(result: DetailedRecommendationResult) -> None:
    print(f"Оценка: {result.total_score}/100 — {result.match_level}")
    print(f"Сильных сторон: {len(result.strengths)}")
    print(f"Зон для улучшения: {len(result.gaps)}")
    print(f"Изменений резюме: {len(result.resume_recommendations)}")
    print(f"Шагов развития: {len(result.development_recommendations)}")


def main() -> None:
    args = parse_arguments()
    resume = read_json(args.resume)
    vacancy = read_json(args.vacancy)
    score = read_json(args.score)

    print("Формируется подробная рекомендация кандидату")
    start_time = time.perf_counter()
    generated = generate_detailed_recommendation(resume, vacancy, score)
    result = build_result(score, generated)
    elapsed_time = time.perf_counter() - start_time

    save_result(result, args.output)
    print_result(result)
    print(f"Результат сохранён: {args.output}")
    print(f"Время обработки: {elapsed_time:.1f} сек.")


if __name__ == "__main__":
    main()
