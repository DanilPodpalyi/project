"""Детерминированная оценка соответствия кандидата требованиям вакансии."""

import argparse
import json
import re
from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field


EXPERIENCE_MAX = 25
PROJECTS_MAX = 20
HEADER_MAX = 15
SUMMARY_MAX = 15
SKILLS_MAX = 15
EDUCATION_MAX = 10

EXPERIENCE_DURATION_MAX = 10
EXPERIENCE_RELEVANCE_MAX = 10
EXPERIENCE_QUALITY_MAX = 5
PROJECTS_RELEVANCE_MAX = 12
PROJECTS_QUALITY_MAX = 8
SKILLS_REQUIRED_MAX = 12
SKILLS_PREFERRED_MAX = 3

RUSSIAN_MONTHS = {
    "январь": 1,
    "января": 1,
    "февраль": 2,
    "февраля": 2,
    "март": 3,
    "марта": 3,
    "апрель": 4,
    "апреля": 4,
    "май": 5,
    "мая": 5,
    "июнь": 6,
    "июня": 6,
    "июль": 7,
    "июля": 7,
    "август": 8,
    "августа": 8,
    "сентябрь": 9,
    "сентября": 9,
    "октябрь": 10,
    "октября": 10,
    "ноябрь": 11,
    "ноября": 11,
    "декабрь": 12,
    "декабря": 12,
}

EDUCATION_STOP_WORDS = {
    "в",
    "для",
    "и",
    "или",
    "либо",
    "на",
    "область",
    "по",
    "с",
}

SKILL_STOP_WORDS = {
    "в",
    "владение",
    "до",
    "для",
    "за",
    "знание",
    "и",
    "или",
    "из",
    "использование",
    "к",
    "на",
    "навык",
    "навыки",
    "опыт",
    "по",
    "под",
    "работа",
    "с",
    "умение",
}

HEADER_ROLE_WORDS = {
    "assistant",
    "junior",
    "manager",
    "senior",
    "specialist",
    "ассистент",
    "джуниор",
    "менеджер",
    "специалист",
}

EDUCATION_SUFFIXES = (
    "иями",
    "ями",
    "ами",
    "ого",
    "его",
    "ому",
    "ему",
    "ыми",
    "ими",
    "ией",
    "иях",
    "ую",
    "юю",
    "ое",
    "ее",
    "ые",
    "ие",
    "ых",
    "их",
    "ым",
    "им",
    "ом",
    "ем",
    "ая",
    "яя",
    "ию",
    "ия",
    "ии",
    "ий",
    "ей",
    "ой",
    "ам",
    "ям",
    "ах",
    "ях",
    "ов",
    "ев",
    "а",
    "я",
    "ы",
    "и",
    "у",
    "ю",
    "е",
    "о",
)


class CriterionScore(BaseModel):
    score: float
    max_score: float
    details: str


class ScoreResult(BaseModel):
    candidate_name: str | None
    vacancy_title: str | None
    total_score: float = Field(ge=0, le=100)
    recommendation: str

    matched_required_skills: list[str]
    missing_required_skills: list[str]
    matched_preferred_skills: list[str]
    missing_preferred_skills: list[str]

    breakdown: dict[str, CriterionScore]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Оценка соответствия резюме требованиям вакансии"
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
        "output",
        type=Path,
        help="Путь для сохранения результата",
    )
    return parser.parse_args()


def read_json(file_path: Path) -> dict:
    if not file_path.exists():
        raise SystemExit(f"Файл не найден: {file_path}")

    if not file_path.is_file():
        raise SystemExit(f"Указанный путь не является файлом: {file_path}")

    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(
            f"Некорректный JSON в файле {file_path}: {error}"
        ) from error


def normalize_text(value: str) -> str:
    value = value.lower().replace("ё", "е")
    value = re.sub(r"[^a-zа-я0-9+#.]+", " ", value)
    return " ".join(value.split())


def collect_candidate_skills(resume: dict) -> dict[str, str]:
    skills: list[str] = list(resume.get("skills", []))
    skills.extend(resume.get("soft_skills", []))

    for workplace in resume.get("work_experience", []):
        skills.extend(workplace.get("responsibilities", []))
        skills.extend(workplace.get("technologies", []))
        skills.extend(workplace.get("achievements", []))

    for project in resume.get("projects", []):
        skills.extend(project.get("technologies", []))
        skills.extend(project.get("results", []))
        if project.get("description"):
            skills.append(project["description"])
        if project.get("role"):
            skills.append(project["role"])

    for language in resume.get("languages", []):
        if language.get("language"):
            skills.append(language["language"])

    normalized_skills: dict[str, str] = {}

    for skill in skills:
        normalized = normalize_text(skill)

        if normalized:
            normalized_skills[normalized] = skill

    return normalized_skills


def _canonical_skill_tokens(value: str) -> set[str]:
    normalized = normalize_text(value)
    tokens = {
        _stem_education_token(token)
        for token in normalized.split()
        if token not in SKILL_STOP_WORDS
    }

    if "связ" in tokens and "общественност" in tokens:
        tokens.discard("связ")
        tokens.discard("общественност")
        tokens.add("pr")
        tokens.add("communication")

    communication_tokens = {
        token
        for token in tokens
        if token.startswith("коммуникац")
        or token in {"communication", "communications"}
    }
    if communication_tokens:
        tokens.difference_update(communication_tokens)
        tokens.add("communication")

    event_tokens = {
        token
        for token in tokens
        if token.startswith(
            ("event", "ивент", "конференц", "мероприят", "нетворкинг")
        )
    }
    if event_tokens:
        tokens.difference_update(event_tokens)
        tokens.add("event")

    if "электронн" in tokens and any(
        token.startswith("почт") for token in tokens
    ):
        tokens.discard("электронн")
        tokens = {token for token in tokens if not token.startswith("почт")}
        tokens.add("email")

    return tokens


def skill_matches(candidate_skill: str, vacancy_skill: str) -> bool:
    if normalize_text(candidate_skill) == normalize_text(vacancy_skill):
        return True

    required_tokens = _canonical_skill_tokens(vacancy_skill)
    candidate_tokens = _canonical_skill_tokens(candidate_skill)

    return bool(required_tokens) and required_tokens.issubset(candidate_tokens)


def compare_skills(
    candidate_skills: dict[str, str],
    vacancy_skills: list[str],
) -> tuple[list[str], list[str]]:
    matched: list[str] = []
    missing: list[str] = []

    for skill in vacancy_skills:
        if any(
            skill_matches(candidate_skill, skill)
            for candidate_skill in candidate_skills.values()
        ):
            matched.append(skill)
        else:
            missing.append(skill)

    return matched, missing


def calculate_list_score(
    matched: list[str],
    total_requirements: int,
    max_score: float,
) -> float:
    if total_requirements == 0:
        return max_score

    ratio = len(matched) / total_requirements
    return round(ratio * max_score, 1)


def _collect_evidence(values: list[str]) -> dict[str, str]:
    return {
        normalized: value
        for value in values
        if (normalized := normalize_text(value))
    }


def collect_skill_section_evidence(resume: dict) -> dict[str, str]:
    values = list(resume.get("skills", []))
    values.extend(resume.get("soft_skills", []))
    values.extend(
        language["language"]
        for language in resume.get("languages", [])
        if language.get("language")
    )
    return _collect_evidence(values)


def collect_experience_evidence(resume: dict) -> dict[str, str]:
    values: list[str] = []
    for workplace in resume.get("work_experience", []):
        if workplace.get("position"):
            values.append(workplace["position"])
        values.extend(workplace.get("responsibilities", []))
        values.extend(workplace.get("technologies", []))
        values.extend(workplace.get("achievements", []))
    return _collect_evidence(values)


def collect_project_evidence(resume: dict) -> dict[str, str]:
    values: list[str] = []
    for project in resume.get("projects", []):
        for field_name in ("name", "description", "role"):
            if project.get(field_name):
                values.append(project[field_name])
        values.extend(project.get("technologies", []))
        values.extend(project.get("results", []))
    return _collect_evidence(values)


def calculate_evidence_quality(
    items: list[dict],
    fields: tuple[str, ...],
    max_score: float,
) -> float:
    if not items:
        return 0.0

    completeness = sum(
        sum(bool(item.get(field_name)) for field_name in fields) / len(fields)
        for item in items
    ) / len(items)
    return round(completeness * max_score, 1)


def calculate_header_score(resume: dict, vacancy: dict) -> tuple[float, str]:
    name_score = 2.0 if resume.get("candidate_name") else 0.0

    contacts_count = len(resume.get("contact_channels", []))
    contacts_score = min(contacts_count / 2, 1.0) * 5.0

    target_position = resume.get("target_position") or ""
    vacancy_title = vacancy.get("title") or ""
    target_presence_score = 3.0 if target_position else 0.0
    relevance_score = 0.0
    if target_position and vacancy_title:
        target_tokens = _canonical_skill_tokens(target_position)
        vacancy_tokens = _canonical_skill_tokens(vacancy_title)
        target_tokens -= HEADER_ROLE_WORDS
        vacancy_tokens -= HEADER_ROLE_WORDS
        if vacancy_tokens:
            relevance = len(target_tokens & vacancy_tokens) / len(vacancy_tokens)
            relevance_score = round(relevance * 5.0, 1)
    elif target_position:
        relevance_score = 5.0

    score = round(
        name_score + contacts_score + target_presence_score + relevance_score,
        1,
    )
    details = (
        f"Имя: {'указано' if name_score else 'не указано'}; "
        f"каналов связи: {contacts_count}; "
        f"целевая позиция: "
        f"{'указана' if target_position else 'не указана'}, "
        f"релевантность заголовку вакансии: {relevance_score:g}/5."
    )
    return score, details


def calculate_summary_score(resume: dict, vacancy: dict) -> tuple[float, str]:
    summary = (resume.get("professional_summary") or "").strip()
    if not summary:
        return 0.0, "Раздел «О себе» не заполнен."

    requirements = vacancy.get("required_skills", [])
    summary_evidence = _collect_evidence([summary])
    matched, _ = compare_skills(summary_evidence, requirements)
    relevance_score = calculate_list_score(
        matched,
        len(requirements),
        9.0,
    )

    presence_score = 3.0
    quality_score = 0.0
    if len(summary) >= 120:
        quality_score += 1.0
    if re.search(r"\d", summary):
        quality_score += 1.0
    if re.search(r"\b(?:ищу|цель|позици|развива|хочу)\w*", summary.lower()):
        quality_score += 1.0

    score = round(presence_score + relevance_score + quality_score, 1)
    details = (
        f"Совпало {len(matched)} из {len(requirements)} обязательных навыков; "
        f"качество содержания: {quality_score:g}/3."
    )
    return score, details


def calculate_skills_score(resume: dict, vacancy: dict) -> tuple[float, str]:
    evidence = collect_skill_section_evidence(resume)
    required = vacancy.get("required_skills", [])
    preferred = vacancy.get("preferred_skills", [])
    matched_required, _ = compare_skills(evidence, required)
    matched_preferred, _ = compare_skills(evidence, preferred)

    required_score = calculate_list_score(
        matched_required,
        len(required),
        SKILLS_REQUIRED_MAX,
    )
    preferred_score = calculate_list_score(
        matched_preferred,
        len(preferred),
        SKILLS_PREFERRED_MAX,
    )
    score = round(required_score + preferred_score, 1)
    details = (
        f"В разделе навыков совпало {len(matched_required)} из {len(required)} "
        f"обязательных и {len(matched_preferred)} из {len(preferred)} "
        "желательных навыков."
    )
    return score, details


def extract_required_years(requirement: str | None) -> float | None:
    if not requirement:
        return None

    match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(?:год|года|лет)",
        requirement.lower(),
    )

    if not match:
        return None

    return float(match.group(1).replace(",", "."))


def _parse_month_year(value: str) -> tuple[int, int] | None:
    match = re.search(r"([а-яё]+)\s+(\d{4})", value.lower())
    if not match:
        return None

    month = RUSSIAN_MONTHS.get(match.group(1).replace("ё", "е"))
    if month is None:
        return None

    return int(match.group(2)), month


def _extract_period_interval(
    period: str | None,
    today: date | None = None,
) -> tuple[int, int] | None:
    if not period:
        return None

    parts = re.split(r"\s*[–—]\s*|\s+-\s+", period, maxsplit=1)
    if len(parts) != 2:
        return None

    start = _parse_month_year(parts[0])
    if start:
        start_year, start_month = start
        start_index = start_year * 12 + start_month - 1

        end_text = parts[1].lower().replace("ё", "е")
        if any(
            marker in end_text
            for marker in ("настоящее время", "по настоящее", "сейчас", "н.в")
        ):
            current_date = today or date.today()
            end_index = current_date.year * 12 + current_date.month
        else:
            end = _parse_month_year(parts[1])
            if not end:
                return None
            end_year, end_month = end
            end_index = end_year * 12 + end_month

        if end_index <= start_index:
            return None

        return start_index, end_index

    match = re.search(
        r"(\d{4})\s*[–—-]\s*(\d{4})",
        period,
    )

    if not match:
        return None

    start_year = int(match.group(1))
    end_year = int(match.group(2))
    if end_year <= start_year:
        return None

    return start_year * 12, end_year * 12


def extract_period_years(
    period: str | None,
    today: date | None = None,
) -> float:
    interval = _extract_period_interval(period, today)
    if not interval:
        return 0.0

    start, end = interval
    return round((end - start) / 12, 2)


def calculate_candidate_experience(
    resume: dict,
    today: date | None = None,
) -> float:
    intervals: list[tuple[int, int]] = []

    for workplace in resume.get("work_experience", []):
        interval = _extract_period_interval(workplace.get("period"), today)
        if interval:
            intervals.append(interval)

    if not intervals:
        return 0.0

    intervals.sort()
    merged: list[list[int]] = []

    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    total_months = sum(end - start for start, end in merged)
    return round(total_months / 12, 2)


def _stem_education_token(token: str) -> str:
    for suffix in EDUCATION_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]

    return token


def _education_tokens(value: str) -> set[str]:
    return {
        _stem_education_token(token)
        for token in normalize_text(value).split()
        if token not in EDUCATION_STOP_WORDS
    }


def education_field_matches(
    required_field: str,
    candidate_fields: list[str],
) -> bool:
    normalized_required = normalize_text(required_field)
    normalized_candidates = [normalize_text(value) for value in candidate_fields]

    if any(
        normalized_required in candidate
        for candidate in normalized_candidates
    ):
        return True

    candidate_tokens = _education_tokens(" ".join(candidate_fields))
    alternatives = re.split(
        r"\s*(?:,|;|/|\bили\b|\bлибо\b)\s*",
        normalized_required,
    )

    return any(
        required_tokens
        and required_tokens.issubset(candidate_tokens)
        for alternative in alternatives
        if (required_tokens := _education_tokens(alternative))
    )


def calculate_experience_score(
    candidate_years: float,
    required_years: float | None,
    max_score: float = EXPERIENCE_DURATION_MAX,
) -> float:
    if required_years is None:
        return float(max_score)

    if required_years <= 0:
        return float(max_score)

    ratio = min(candidate_years / required_years, 1.0)
    return round(ratio * max_score, 1)


def calculate_experience_section_score(
    resume: dict,
    vacancy: dict,
) -> tuple[float, str]:
    workplaces = resume.get("work_experience", [])
    if not workplaces:
        return 0.0, "Опыт работы в резюме не указан."

    candidate_years = calculate_candidate_experience(resume)
    required_years = extract_required_years(
        vacancy.get("experience_requirement")
    )
    duration_score = calculate_experience_score(
        candidate_years,
        required_years,
        EXPERIENCE_DURATION_MAX,
    )

    required_skills = vacancy.get("required_skills", [])
    experience_evidence = collect_experience_evidence(resume)
    matched_required, _ = compare_skills(
        experience_evidence,
        required_skills,
    )
    relevance_score = calculate_list_score(
        matched_required,
        len(required_skills),
        EXPERIENCE_RELEVANCE_MAX,
    )
    quality_score = calculate_evidence_quality(
        workplaces,
        ("company", "position", "period", "responsibilities", "achievements"),
        EXPERIENCE_QUALITY_MAX,
    )

    score = round(duration_score + relevance_score + quality_score, 1)
    requirement_text = (
        f"{required_years:g} года"
        if required_years is not None
        else "не указано"
    )
    details = (
        f"Продолжительность: {candidate_years:g} года, требуется: {requirement_text} "
        f"({duration_score:g}/{EXPERIENCE_DURATION_MAX}); "
        f"релевантность: {len(matched_required)} из {len(required_skills)} навыков "
        f"({relevance_score:g}/{EXPERIENCE_RELEVANCE_MAX}); "
        f"заполненность: {quality_score:g}/{EXPERIENCE_QUALITY_MAX}."
    )
    return score, details


def calculate_projects_score(
    resume: dict,
    vacancy: dict,
) -> tuple[float, str]:
    projects = resume.get("projects", [])
    if not projects:
        return 0.0, "Проекты в резюме не указаны."

    evidence = collect_project_evidence(resume)
    required_skills = vacancy.get("required_skills", [])
    preferred_skills = vacancy.get("preferred_skills", [])
    matched_required, _ = compare_skills(evidence, required_skills)
    matched_preferred, _ = compare_skills(evidence, preferred_skills)

    required_relevance_score = calculate_list_score(
        matched_required,
        len(required_skills),
        10.0,
    )
    preferred_relevance_score = calculate_list_score(
        matched_preferred,
        len(preferred_skills),
        2.0,
    )
    relevance_score = round(
        required_relevance_score + preferred_relevance_score,
        1,
    )
    quality_score = calculate_evidence_quality(
        projects,
        ("name", "description", "role", "technologies", "results"),
        PROJECTS_QUALITY_MAX,
    )

    score = round(relevance_score + quality_score, 1)
    details = (
        f"Релевантность: {len(matched_required)} из {len(required_skills)} обязательных "
        f"и {len(matched_preferred)} из {len(preferred_skills)} желательных навыков "
        f"({relevance_score:g}/{PROJECTS_RELEVANCE_MAX}); "
        f"заполненность: {quality_score:g}/{PROJECTS_QUALITY_MAX}."
    )
    return score, details


def normalize_education_level(value: str | None) -> str | None:
    if not value:
        return None

    normalized = normalize_text(value)

    if any(
        marker in normalized
        for marker in ("высшее", "бакалавр", "магистр", "специалист")
    ):
        return "higher"

    if any(
        marker in normalized
        for marker in ("среднее профессиональное", "спо")
    ):
        return "secondary_professional"

    if "среднее" in normalized:
        return "secondary"

    return normalized


def calculate_education_score(
    resume: dict,
    vacancy: dict,
) -> tuple[float, str]:
    required_level = vacancy.get("education_level")
    required_field = vacancy.get("education_field")

    candidate_education = resume.get("education", [])

    level_score = 0.0
    field_score = 0.0

    if not required_level:
        level_score = 7.0
        level_match = True
    else:
        normalized_required_level = normalize_education_level(required_level)

        level_match = any(
            normalize_education_level(item.get("education_level"))
            == normalized_required_level
            for item in candidate_education
        )

        if level_match:
            level_score = 7.0

    if not required_field:
        field_score = 3.0
        field_match = True
    else:
        candidate_fields = [
            value
            for item in candidate_education
            for value in (
                item.get("education_field"),
                item.get("field_of_study"),
            )
            if value
        ]
        field_match = education_field_matches(
            required_field,
            candidate_fields,
    )

        if field_match:
            field_score = 3.0

    details = (
        f"Уровень образования: {'совпадает' if level_match else 'не совпадает'}; "
        f"направление: {'совпадает' if field_match else 'не совпадает'}."
    )

    return level_score + field_score, details


def make_recommendation(total_score: float) -> str:
    if total_score >= 80:
        return "Высокое соответствие вакансии"

    if total_score >= 60:
        return "Умеренное соответствие вакансии"

    if total_score >= 40:
        return "Низкое соответствие вакансии"

    return "Не соответствует основным требованиям вакансии"


def score_candidate(resume: dict, vacancy: dict) -> ScoreResult:
    candidate_skills = collect_candidate_skills(resume)

    required_skills = vacancy.get("required_skills", [])
    preferred_skills = vacancy.get("preferred_skills", [])

    matched_required, missing_required = compare_skills(
        candidate_skills,
        required_skills,
    )
    matched_preferred, missing_preferred = compare_skills(
        candidate_skills,
        preferred_skills,
    )

    experience_score, experience_details = (
        calculate_experience_section_score(resume, vacancy)
    )
    projects_score, projects_details = calculate_projects_score(
        resume,
        vacancy,
    )
    header_score, header_details = calculate_header_score(
        resume,
        vacancy,
    )
    summary_score, summary_details = calculate_summary_score(
        resume,
        vacancy,
    )
    skills_score, skills_details = calculate_skills_score(
        resume,
        vacancy,
    )
    education_score, education_details = calculate_education_score(
        resume,
        vacancy,
    )

    total_score = round(
        experience_score
        + projects_score
        + header_score
        + summary_score
        + skills_score
        + education_score,
        1,
    )

    return ScoreResult(
        candidate_name=resume.get("candidate_name"),
        vacancy_title=vacancy.get("title"),
        total_score=total_score,
        recommendation=make_recommendation(total_score),
        matched_required_skills=matched_required,
        missing_required_skills=missing_required,
        matched_preferred_skills=matched_preferred,
        missing_preferred_skills=missing_preferred,
        breakdown={
            "experience_score": CriterionScore(
                score=experience_score,
                max_score=EXPERIENCE_MAX,
                details=experience_details,
            ),
            "projects_score": CriterionScore(
                score=projects_score,
                max_score=PROJECTS_MAX,
                details=projects_details,
            ),
            "header_score": CriterionScore(
                score=header_score,
                max_score=HEADER_MAX,
                details=header_details,
            ),
            "summary_score": CriterionScore(
                score=summary_score,
                max_score=SUMMARY_MAX,
                details=summary_details,
            ),
            "skills_score": CriterionScore(
                score=skills_score,
                max_score=SKILLS_MAX,
                details=skills_details,
            ),
            "education_score": CriterionScore(
                score=education_score,
                max_score=EDUCATION_MAX,
                details=education_details,
            ),
        },
    )


def save_result(result: ScoreResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_arguments()

    resume = read_json(args.resume)
    vacancy = read_json(args.vacancy)

    result = score_candidate(resume, vacancy)
    save_result(result, args.output)

    print(f"Кандидат: {result.candidate_name}")
    print(f"Вакансия: {result.vacancy_title}")
    print(f"Итоговая оценка: {result.total_score}/100")
    print(f"Рекомендация: {result.recommendation}")
    print(f"Результат сохранён: {args.output}")


if __name__ == "__main__":
    main()
