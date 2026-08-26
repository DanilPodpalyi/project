"""Интернет-поиск вакансий по целевой должности из резюме."""

import argparse
import html
import json
import re
from json import JSONDecodeError
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field


HH_VACANCIES_URL = "https://api.hh.ru/vacancies"
DEFAULT_USER_AGENT = "ResumeEvaluator/1.0 (local career tool)"
DEFAULT_LIMIT = 10
MAX_LIMIT = 50


class Salary(BaseModel):
    from_amount: float | None
    to_amount: float | None
    currency: str | None
    gross: bool | None


class Vacancy(BaseModel):
    vacancy_id: str
    title: str
    employer: str | None
    area: str | None
    salary: Salary | None
    experience: str | None
    employment: str | None
    schedule: str | None
    requirement: str | None
    responsibility: str | None
    published_at: str | None
    url: str | None


class VacancySearchResult(BaseModel):
    query: str
    source: str
    found: int = Field(ge=0)
    returned: int = Field(ge=0)
    vacancies: list[Vacancy]


class VacancySearchError(RuntimeError):
    """Ошибка запроса или разбора ответа сервиса вакансий."""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Поиск вакансий на hh.ru по target_position из JSON резюме"
        ),
    )
    parser.add_argument(
        "resume",
        type=Path,
        help="Структурированный JSON, созданный extract_resume.py",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Путь для сохранения найденных вакансий в JSON",
    )
    parser.add_argument(
        "--area",
        help=(
            "Идентификатор региона hh.ru, например 1 для Москвы; "
            "по умолчанию поиск по всем регионам"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Количество результатов от 1 до {MAX_LIMIT}",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="Идентификатор приложения для обязательного заголовка API",
    )
    return parser.parse_args()


def read_resume(file_path: Path) -> dict:
    if not file_path.is_file():
        raise SystemExit(f"Файл резюме не найден: {file_path}")

    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except JSONDecodeError as error:
        raise SystemExit(
            f"Некорректный JSON в файле {file_path}: {error}",
        ) from error

    if not isinstance(data, dict):
        raise SystemExit("JSON резюме должен содержать объект верхнего уровня")

    return data


def get_search_query(resume: dict) -> str:
    target_position = resume.get("target_position")
    if not isinstance(target_position, str) or not target_position.strip():
        raise VacancySearchError(
            "В резюме не указана целевая должность (target_position)",
        )

    return " ".join(target_position.split())


def build_search_url(
    query: str,
    limit: int = DEFAULT_LIMIT,
    area: str | None = None,
) -> str:
    if not 1 <= limit <= MAX_LIMIT:
        raise ValueError(
            f"Количество результатов должно быть от 1 до {MAX_LIMIT}",
        )

    parameters = {
        "text": query,
        "per_page": str(limit),
        "page": "0",
        "order_by": "publication_time",
    }
    if area:
        parameters["area"] = area

    return f"{HH_VACANCIES_URL}?{urlencode(parameters)}"


def fetch_search_data(
    url: str,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: float = 15.0,
) -> dict:
    request = Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "application/json",
        },
        method="GET",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except HTTPError as error:
        details = _read_http_error_details(error)
        suffix = f": {details}" if details else ""
        raise VacancySearchError(
            f"API hh.ru вернул HTTP {error.code}{suffix}",
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        raise VacancySearchError(
            f"Не удалось подключиться к API hh.ru: {error}",
        ) from error

    try:
        data = json.loads(payload)
    except (JSONDecodeError, UnicodeDecodeError) as error:
        raise VacancySearchError(
            "API hh.ru вернул некорректный JSON",
        ) from error

    if not isinstance(data, dict):
        raise VacancySearchError("API hh.ru вернул неожиданный формат данных")

    return data


def _read_http_error_details(error: HTTPError) -> str | None:
    try:
        payload = json.loads(error.read().decode("utf-8"))
    except (JSONDecodeError, UnicodeDecodeError, OSError):
        return None

    if not isinstance(payload, dict):
        return None

    description = payload.get("description")
    if isinstance(description, str) and description.strip():
        return description.strip()

    errors = payload.get("errors")
    if not isinstance(errors, list):
        return None

    messages: list[str] = []
    for item in errors:
        if not isinstance(item, dict):
            continue
        parts = [
            str(item[field_name])
            for field_name in ("type", "value", "reason")
            if item.get(field_name)
        ]
        if parts:
            messages.append("/".join(parts))

    return ", ".join(messages) or None


def _nested_name(item: dict, field_name: str) -> str | None:
    value = item.get(field_name)
    if not isinstance(value, dict):
        return None
    name = value.get("name")
    return name if isinstance(name, str) else None


def _clean_snippet(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None

    without_tags = re.sub(r"<[^>]+>", "", value)
    return " ".join(html.unescape(without_tags).split())


def parse_vacancy(item: dict) -> Vacancy:
    salary_data = item.get("salary")
    salary = None
    if isinstance(salary_data, dict):
        salary = Salary(
            from_amount=salary_data.get("from"),
            to_amount=salary_data.get("to"),
            currency=salary_data.get("currency"),
            gross=salary_data.get("gross"),
        )

    snippet = item.get("snippet")
    if not isinstance(snippet, dict):
        snippet = {}

    return Vacancy(
        vacancy_id=str(item.get("id", "")),
        title=str(item.get("name", "")),
        employer=_nested_name(item, "employer"),
        area=_nested_name(item, "area"),
        salary=salary,
        experience=_nested_name(item, "experience"),
        employment=_nested_name(item, "employment"),
        schedule=_nested_name(item, "schedule"),
        requirement=_clean_snippet(snippet.get("requirement")),
        responsibility=_clean_snippet(snippet.get("responsibility")),
        published_at=item.get("published_at"),
        url=item.get("alternate_url"),
    )


def search_vacancies(
    resume: dict,
    *,
    area: str | None = None,
    limit: int = DEFAULT_LIMIT,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: float = 15.0,
) -> VacancySearchResult:
    query = get_search_query(resume)
    url = build_search_url(query, limit=limit, area=area)
    data = fetch_search_data(
        url,
        user_agent=user_agent,
        timeout=timeout,
    )

    items = data.get("items", [])
    if not isinstance(items, list):
        raise VacancySearchError(
            "API hh.ru не вернул список вакансий в поле items",
        )

    vacancies = [
        parse_vacancy(item)
        for item in items
        if isinstance(item, dict)
    ]

    found = data.get("found", len(vacancies))
    if not isinstance(found, int) or found < 0:
        found = len(vacancies)

    return VacancySearchResult(
        query=query,
        source="hh.ru",
        found=found,
        returned=len(vacancies),
        vacancies=vacancies,
    )


def save_result(result: VacancySearchResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_arguments()
    resume = read_resume(args.resume)

    try:
        result = search_vacancies(
            resume,
            area=args.area,
            limit=args.limit,
            user_agent=args.user_agent,
        )
    except (VacancySearchError, ValueError) as error:
        raise SystemExit(f"Ошибка поиска вакансий: {error}") from error

    save_result(result, args.output)
    print(f"Поисковый запрос: {result.query}")
    print(f"Найдено всего: {result.found}")
    print(f"Сохранено вакансий: {result.returned}")
    print(f"Результат сохранён: {args.output}")


if __name__ == "__main__":
    main()
