"""Сбор полных вакансий по России за 120 дней и построение MarketProfile.

Цель — только должность и указанный уровень из резюме. Собранные оригиналы
и диагностика сохраняются до обращения к моделям. Источники — HH и
«Работа России»; зарубежные remote feeds автоматически не подключаются.
"""


from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from .settings import get_settings
from .runtime import atomic_write_text, remaining_timeout

from pydantic import BaseModel, Field, ValidationError
from ollama import ResponseError

from .profession import ProfessionProfile, TargetProfile
from .resume_extractor import ResumeData, extract_resume, read_resume
from .title_morphology import morphological_title_similarity, title_search_queries
from .bilingual_titles import BilingualSearchTitles, resolve_bilingual_titles, bilingual_title_similarity
from .market_profile_builder_v2 import build_market_profile_v2, _clean_target_seniority
from .market_profile import (
    InsufficientMarketProfile, MarketProfile, ProfessionSearchPlan, RawVacancy,
    SearchCollectionStatistics, SyntheticVacancyError, NormalizedVacancy,
    _make_normalized_vacancy, _request_normalized_batch, _set_extraction_quality,
    NORMALIZATION_SYSTEM_PROMPT,
)


HH_API = "https://api.hh.ru"
TRUDVSEM_API = "https://opendata.trudvsem.ru/api/v1/vacancies"
RUSSIA_AREA_ID = "113"


class MarketScope(BaseModel):
    country_code: Literal["RU"] = "RU"
    country: Literal["Россия"] = "Россия"
    hh_area_id: Literal["113"] = RUSSIA_AREA_ID
    lookback_days: Literal[120] = 120
    collected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def date_from(self) -> datetime:
        return self.collected_at - timedelta(days=self.lookback_days)


class CollectedVacancy(RawVacancy):
    country_code: Literal["RU"] = "RU"
    region: str | None = None
    area_id: str | None = None
    employer_id: str | None = None
    fetched_at: datetime
    original_description: str
    description_status: Literal["full", "empty", "sparse"]
    morphology_similarity: float = Field(ge=0, le=1)
    archived: bool = False
    publication_precision: Literal["instant", "day"] = "instant"


class CollectionIssue(BaseModel):
    source: str
    stage: str
    reason: str
    source_id: str | None = None
    http_status: int | None = None


class CollectionDiagnostics(BaseModel):
    scope: MarketScope
    target: TargetProfile
    queries: list[str]
    search_titles: BilingualSearchTitles | None = None
    status: Literal["ok", "partial", "unavailable"] = "unavailable"
    requests: int = 0
    cache_hits: int = 0
    search_results: int = 0
    details_received: int = 0
    complete_descriptions: int = 0
    sparse_descriptions: int = 0
    duplicates: int = 0
    accepted: int = 0
    search_limit_reached: bool = False
    results_by_source: dict[str, int] = Field(default_factory=dict)
    accepted_by_source: dict[str, int] = Field(default_factory=dict)
    errors: list[CollectionIssue] = Field(default_factory=list)
    exclusions: list[CollectionIssue] = Field(default_factory=list)


class VacancyCollection(BaseModel):
    diagnostics: CollectionDiagnostics
    vacancies: list[CollectedVacancy]
    duplicate_copies: list[CollectedVacancy] = Field(default_factory=list)


def target_from_resume(resume: ResumeData) -> TargetProfile:
    profile = resume.target_profession
    if not (profile.specialization or profile.role or "").strip():
        raise ValueError("В резюме не указана целевая должность; рынок не выбран.")
    # Не достраиваем target через LLM и не переносим навыки или aliases кандидата.
    return _clean_target_seniority(
        TargetProfile(**profile.model_dump(exclude={"core_skills"}), core_skills=[]),
        source=profile.label(),
    )


def russian_search_queries(target: TargetProfile, *, search_titles: BilingualSearchTitles | None = None) -> list[str]:
    names = search_titles.variants() if search_titles else [target.label()]
    queries = list(dict.fromkeys(value for value in names + title_search_queries(target.label())))
    # Оставляем место запросу с уровнем, даже если у профессии много вариантов.
    queries = queries[:5 if target.seniority else 6]
    if target.seniority:
        queries.append(f"{target.label()} {target.seniority}")
        for name in names:
            query = f"{name} {target.seniority}"
            if len(queries) < 6 and query.casefold() not in {value.casefold() for value in queries}:
                queries.append(query)
    unique = {}
    for query in queries:
        unique.setdefault(query.casefold(), query)
    queries = list(unique.values())
    # ProfessionSearchPlan исторически требует три query. Повтор на этом уровне
    # допустим: collector устраняет его до HTTP-запросов, без добавления remote.
    while len(queries) < 3:
        queries.append(queries[0])
    return queries[:6]


def _plain_text(value: str) -> str:
    value = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", "", value,
                   flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"</?(?:p|div|li|br|h[1-6])\b[^>]*>", "\n", value,
                   flags=re.IGNORECASE)
    return "\n".join(
        line.strip() for line in html.unescape(re.sub(r"<[^>]+>", "", value)).splitlines()
        if line.strip()
    )


def _description_status(description: str) -> Literal["full", "empty", "sparse"]:
    if not description.strip():
        return "empty"
    # Источник передал полное поле description, но слишком короткий текст
    # не считаем полноценной основой для извлечения требований.
    if len(description) < 200 or len(description.split()) < 20:
        return "sparse"
    return "full"


def _save(model: BaseModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, model.model_dump_json(indent=2))


def _request_json(url: str) -> dict[str, Any]:
    headers = {
        "Accept": "application/json",
        "User-Agent": os.getenv("HH_USER_AGENT", "ResumeEvaluator/1.0 (career research)"),
    }
    token = os.getenv("HH_ACCESS_TOKEN")
    if token and url.startswith(HH_API + "/"):
        headers["Authorization"] = f"Bearer {token}"
    with urlopen(Request(url, headers=headers), timeout=remaining_timeout(get_settings().http_timeout_seconds)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Источник вернул не JSON-объект")
    return payload


def _area_ids(area: dict[str, Any]) -> set[str]:
    result = {str(area["id"])}
    for child in area.get("areas", []):
        result.update(_area_ids(child))
    return result


def _in_period(value: str | None, scope: MarketScope) -> bool:
    if not value:
        return False
    try:
        published = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return False
    if published.tzinfo is None:
        return False
    return scope.date_from <= published <= scope.collected_at


def collect_russian_vacancies(
    target: TargetProfile | ProfessionProfile,
    output_path: Path,
    *,
    scope: MarketScope | None = None,
    imported_path: Path | None = None,
    request_json: Callable[[str], dict[str, Any]] = _request_json,
    max_pages: int | None = None,
    detail_limit: int | None = None,
    use_cache: bool = True,
    include_trudvsem: bool = True,
    use_title_translation: bool = True,
) -> VacancyCollection:
    settings = get_settings()
    max_pages = settings.search_max_pages if max_pages is None else max_pages
    detail_limit = settings.detail_limit if detail_limit is None else detail_limit
    if max_pages < 1 or detail_limit < 1:
        raise ValueError("Лимиты страниц и описаний должны быть положительными")
    scope = scope or MarketScope()
    if scope.collected_at.tzinfo is None:
        raise ValueError("collected_at должен содержать часовой пояс")
    target = TargetProfile(**target.model_dump(exclude={"core_skills"}), core_skills=[])
    if not target.label().strip():
        raise ValueError("Не указана целевая должность")
    search_titles = resolve_bilingual_titles(target.label(), output_path.parent / "title_translation_cache",
        use_llm=use_title_translation and imported_path is None)
    queries = list(dict.fromkeys(russian_search_queries(target, search_titles=search_titles)))
    diagnostic = CollectionDiagnostics(scope=scope, target=target, queries=queries, search_titles=search_titles)
    fetched: list[CollectedVacancy] = []
    cache_directory = output_path.parent / "hh_collection_cache"

    def fetch(url: str, stage: str, source: str = "hh.ru") -> dict[str, Any] | None:
        cache_path = cache_directory / f"{hashlib.sha256(url.encode()).hexdigest()}.json"
        if use_cache and cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                age = scope.collected_at - datetime.fromisoformat(cached["fetched_at"])
                if timedelta(0) <= age < timedelta(hours=settings.response_cache_hours):
                    diagnostic.cache_hits += 1
                    return cached["payload"]
            except (ValueError, KeyError, OSError, TypeError):
                pass
        diagnostic.requests += 1
        try:
            payload = request_json(url)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
            status = error.code if isinstance(error, HTTPError) else None
            reason = f"HTTP {status}" if status else f"{type(error).__name__}: {error}"
            if isinstance(error, HTTPError):
                try:
                    body = json.loads(error.read().decode("utf-8"))
                    types = [str(item["type"]) for item in body.get("errors", [])
                             if isinstance(item, dict) and item.get("type")]
                    if types:
                        reason += ": " + ", ".join(types)
                except (AttributeError, ValueError, OSError):
                    pass
            diagnostic.errors.append(CollectionIssue(
                source=source, stage=stage, http_status=status,
                reason=reason,
            ))
            return None
        if use_cache:
            cache_directory.mkdir(parents=True, exist_ok=True)
            atomic_write_text(cache_path, json.dumps({
                "fetched_at": scope.collected_at.isoformat(), "payload": payload,
            }, ensure_ascii=False))
        return payload

    def reject(source: str, source_id: str | None, reason: str) -> None:
        diagnostic.exclusions.append(CollectionIssue(
            source=source, source_id=source_id, stage="eligibility", reason=reason,
        ))

    def accept(detail: dict[str, Any], source: str, query: str,
               russian_areas: set[str] | None = None) -> None:
        identifier = f"{source}:{detail.get('id', '')}"
        diagnostic.details_received += 1
        area = detail.get("area") or {}
        if russian_areas is not None:
            in_russia = str(area.get("id")) in russian_areas
        else:
            in_russia = detail.get("country_code") == "RU"
        if not in_russia:
            reject(source, identifier, "outside_russia_or_unverified_country")
            return
        if detail.get("archived"):
            reject(source, identifier, "archived")
            return
        if not _in_period(detail.get("published_at"), scope):
            reject(source, identifier, "outside_120_days_or_unverified_date")
            return
        description = detail.get("description")
        if not isinstance(description, str):
            description = ""
        text = _plain_text(description)
        status = _description_status(text)
        title = str(detail.get("name") or detail.get("title") or "")
        url = str(detail.get("alternate_url") or detail.get("url") or "")
        if not title or not url or not detail.get("id"):
            reject(source, identifier, "missing_identity_title_or_url")
            return
        employer = detail.get("employer") or {}
        fetched.append(CollectedVacancy(
            source=source, source_id=identifier, title=title, url=url,
            employer=employer.get("name"), employer_id=employer.get("id"),
            published_at=detail["published_at"], raw_text=text,
            original_description=description, search_query=query,
            fetched_at=scope.collected_at, region=area.get("name"),
            area_id=area.get("id"), description_status=status,
            morphology_similarity=bilingual_title_similarity(target.label(), title, search_titles),
            publication_precision=detail.get("publication_precision", "instant"),
        ))
        if status == "full":
            diagnostic.complete_descriptions += 1
        else:
            diagnostic.sparse_descriptions += 1
            reject(source, identifier, f"description_{status}")

    if imported_path:
        imported = json.loads(imported_path.read_text(encoding="utf-8"))
        if not isinstance(imported, list) or not all(isinstance(item, dict) for item in imported):
            raise ValueError("Файл вакансий должен содержать JSON-массив объектов")
        diagnostic.search_results = len(imported)
        diagnostic.results_by_source["import"] = len(imported)
        for item in imported:
            accept(item, "import", "imported full descriptions")
    else:
        areas = fetch(f"{HH_API}/areas/{RUSSIA_AREA_ID}", "areas")
        russian_areas = _area_ids(areas) if areas and str(areas.get("id")) == RUSSIA_AREA_ID else set()
        if areas and not russian_areas:
            diagnostic.errors.append(CollectionIssue(
                source="hh.ru", stage="areas", reason="invalid_russia_area_tree",
            ))
        summaries: dict[str, tuple[dict[str, Any], str]] = {}
        if russian_areas:
            access_denied = False
            for query in queries:
                if access_denied:
                    break
                for page in range(max_pages):
                    parameters = {
                        "text": query, "search_field": "name", "area": RUSSIA_AREA_ID,
                        "date_from": scope.date_from.isoformat(),
                        "date_to": scope.collected_at.isoformat(),
                        "per_page": 100, "page": page, "order_by": "publication_time",
                    }
                    payload = fetch(f"{HH_API}/vacancies?{urlencode(parameters)}", "search")
                    if payload is None:
                        access_denied = bool(diagnostic.errors and
                                             diagnostic.errors[-1].http_status in {401, 403})
                        break
                    items = payload.get("items", [])
                    if not isinstance(items, list):
                        diagnostic.errors.append(CollectionIssue(
                            source="hh.ru", stage="search", reason="invalid_items_array"))
                        break
                    diagnostic.search_results += len(items)
                    diagnostic.results_by_source["hh.ru"] = (
                        diagnostic.results_by_source.get("hh.ru", 0) + len(items))
                    for item in items:
                        if not isinstance(item, dict) or not item.get("id"):
                            continue
                        identifier = str(item["id"])
                        if identifier in summaries:
                            diagnostic.duplicates += 1
                        else:
                            summaries[identifier] = (item, query)
                    pages = int(payload.get("pages", 1))
                    if page + 1 >= pages:
                        break
                    if page + 1 == max_pages:
                        diagnostic.search_limit_reached = True
            ranked = sorted(summaries.values(), key=lambda pair:
                -bilingual_title_similarity(target.label(), str(pair[0].get("name", "")), search_titles))
            diagnostic.search_limit_reached |= len(ranked) > detail_limit
            for item, query in ranked[:detail_limit]:
                identifier = str(item["id"])
                detail = fetch(f"{HH_API}/vacancies/{identifier}", "detail")
                if detail is not None:
                    accept(detail, "hh.ru", query, russian_areas)

        # Публичный российский источник продолжает работать при ошибке HH.
        # Возраст проверяем по creation-date: обновление старой вакансии
        # не превращает её в новую публикацию.
        if include_trudvsem:
            trud_seen: set[str] = set()
            for query in queries:
                for page in range(max_pages):
                    parameters = {"text": query, "limit": 100, "offset": page}
                    payload = fetch(f"{TRUDVSEM_API}?{urlencode(parameters)}", "search", "trudvsem")
                    if payload is None:
                        break
                    entries = (payload.get("results") or {}).get("vacancies")
                    if entries is None and str(payload.get("status")) == "200" and (
                        str((payload.get("meta") or {}).get("total")) == "0"
                    ):
                        entries = []
                    if str(payload.get("status")) != "200" or not isinstance(entries, list):
                        diagnostic.errors.append(CollectionIssue(
                            source="trudvsem", stage="search", reason="invalid_vacancies_response"))
                        break
                    diagnostic.search_results += len(entries)
                    diagnostic.results_by_source["trudvsem"] = (
                        diagnostic.results_by_source.get("trudvsem", 0) + len(entries))
                    for entry in entries:
                        vacancy = entry.get("vacancy") if isinstance(entry, dict) else None
                        if not isinstance(vacancy, dict):
                            continue
                        identifier = str(vacancy.get("id") or "")
                        if identifier in trud_seen:
                            diagnostic.duplicates += 1
                            continue
                        trud_seen.add(identifier)
                        region = vacancy.get("region") or {}
                        region_code = str(region.get("region_code") or "")
                        company = vacancy.get("company") or {}
                        published = vacancy.get("creation-date")
                        precision = "instant"
                        if isinstance(published, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", published):
                            try:
                                published = datetime.fromisoformat(published).replace(
                                    tzinfo=timezone(timedelta(hours=3))).isoformat()
                                precision = "day"
                            except ValueError:
                                published = None
                        description = "\n".join(
                            f"{label}:\n{vacancy[field]}" for field, label in
                            (("duty", "Обязанности"), ("requirement", "Требования"),
                             ("requirements", "Требования"))
                            if isinstance(vacancy.get(field), str) and vacancy[field].strip()
                        )
                        converted = {
                            "id": identifier, "name": vacancy.get("job-name"),
                            "alternate_url": vacancy.get("vac_url"),
                            "description": description, "published_at": published,
                            "publication_precision": precision,
                            "country_code": "RU" if re.fullmatch(r"\d{13}", region_code) else None,
                            "area": {"id": region_code, "name": region.get("name")},
                            "employer": {"id": str(company.get("companycode") or ""),
                                         "name": company.get("name")},
                            "archived": vacancy.get("archived", False),
                        }
                        accept(converted, "trudvsem", query)
                    if len(entries) < 100:
                        break
                    if page + 1 == max_pages:
                        diagnostic.search_limit_reached = True

    # Дубликаты импортированных объявлений также не увеличивают частоты рынка.
    from .market_reliability import deduplicate_market_vacancies

    vacancies, duplicates = deduplicate_market_vacancies(fetched)
    diagnostic.duplicates += len(duplicates)
    for row in duplicates:
        diagnostic.exclusions.append(CollectionIssue(
            source="deduplication", stage="deduplication", source_id=row.removed_source_id,
            reason=f"{row.reason}; kept={row.kept_source_id}; url={row.kept_url}",
        ))
    diagnostic.accepted = sum(item.description_status == "full" for item in vacancies)
    for item in vacancies:
        if item.description_status == "full":
            diagnostic.accepted_by_source[item.source] = (
                diagnostic.accepted_by_source.get(item.source, 0) + 1)
    diagnostic.status = (
        "partial" if diagnostic.accepted and diagnostic.errors else
        "ok" if diagnostic.accepted else "unavailable"
    )
    kept_objects = {id(item) for item in vacancies}
    result = VacancyCollection(diagnostics=diagnostic, vacancies=vacancies,
                               duplicate_copies=[item for item in fetched if id(item) not in kept_objects])
    _save(result, output_path)
    return result


def normalize_full_descriptions(
    vacancies: list[RawVacancy], target: TargetProfile, *,
    checkpoint_path: Path, cache_path: Path,
) -> list[NormalizedVacancy]:
    """Обрабатывает полный текст по одному объявлению, без fallback-обрезания.

    Чрезмерно длинные документы явно исключаются и попадают в diagnostics.
    Этот cache отделён от normalizer, который обрезает описание до 2500 знаков.
    """
    cache: dict[str, Any] = {}
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            pass
    results: list[NormalizedVacancy] = []
    issues: list[dict[str, str]] = []
    for item in vacancies:
        key = hashlib.sha256(json.dumps({
            "title": item.title, "text": item.raw_text,
            "target": target.model_dump(), "version": 3, "model": get_settings().model,
            "prompt_sha256": hashlib.sha256(NORMALIZATION_SYSTEM_PROMPT.encode()).hexdigest(),
        }, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        try:
            if key in cache:
                normalized = NormalizedVacancy.model_validate(cache[key])
                results.append(normalized.model_copy(update={
                    "source_id": item.source_id, "url": item.url,
                    "title": item.title, "employer": item.employer,
                    "published_at": item.published_at, "source": item.source,
                }))
                continue
            if len(item.raw_text) > 24000:
                issues.append({"source_id": item.source_id, "reason": "full_text_exceeds_model_budget"})
                continue
            parsed = _request_normalized_batch(
                [item], target, max_description_chars=len(item.raw_text) + 1,
                context_size=max(8192, len(item.raw_text) + 4096),
            )
            extracted = next((value for value in parsed.vacancies
                              if value.source_id == item.source_id), None)
            if extracted is None:
                raise ValueError("Модель не вернула исходный source_id")
            extracted = _set_extraction_quality(extracted, item)
            if extracted.extraction_status != "ok":
                issues.append({"source_id": item.source_id,
                               "reason": f"extraction_{extracted.extraction_status}"})
                continue
            normalized = _make_normalized_vacancy(extracted, item)
            cache[key] = normalized.model_dump()
            results.append(normalized)
        except (SyntheticVacancyError, ResponseError, OSError, ValueError, ValidationError) as error:
            issues.append({"source_id": item.source_id, "reason": f"{type(error).__name__}: {error}"})
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(cache_path, json.dumps(cache, ensure_ascii=False, indent=2))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(checkpoint_path, json.dumps({"full_descriptions": True, "issues": issues,
                                          "normalized_count": len(results)},
                                         ensure_ascii=False, indent=2))
    return results


def validate_saved_russian_profile(
    profile: MarketProfile | InsufficientMarketProfile,
    target: TargetProfile,
    collection_path: Path,
) -> VacancyCollection:
    """Повторное использование сохраняет страну, цель и актуальность выборки."""
    if not collection_path.is_file():
        raise ValueError("Для профиля России нужен соседний *.collection.json с происхождением данных")
    collection = VacancyCollection.model_validate_json(collection_path.read_text(encoding="utf-8"))
    for saved in (profile.target, collection.diagnostics.target):
        label = saved.specialization or saved.role or ""
        if (morphological_title_similarity(target.label(), label) != 1
                or saved.seniority != target.seniority):
            raise ValueError("Сохранённый рынок не соответствует должности и уровню из резюме")
    current_scope = MarketScope()
    rows = {item.source_id: item for item in collection.vacancies}
    sources = [*profile.sources, *(profile.adjacent_context.sources if profile.adjacent_context else [])]
    for source in sources:
        row = rows.get(source.source_id)
        if (row is None or row.country_code != "RU" or row.description_status != "full"
                or not _in_period(row.published_at, current_scope)):
            raise ValueError("Источники сохранённого профиля не подтверждают Россию и период 120 дней")
    return collection


def build_russian_market_profile(
    target: TargetProfile | ProfessionProfile,
    output_path: Path,
    *,
    imported_path: Path | None = None,
    scope: MarketScope | None = None,
    use_embeddings: bool = True,
    use_llm: bool = True,
    use_title_translation: bool = True,
) -> MarketProfile | InsufficientMarketProfile:
    target = TargetProfile(**target.model_dump(exclude={"core_skills"}), core_skills=[])
    collection_path = output_path.with_name(f"{output_path.stem}.collection.json")
    collection = collect_russian_vacancies(target, collection_path,
        imported_path=imported_path, scope=scope, use_title_translation=use_llm and use_title_translation)

    def searcher(plan: ProfessionSearchPlan, minimum: int,
                 diagnostics: SearchCollectionStatistics, strict_minimum: bool,
                 result_limit: int):
        del plan, minimum, strict_minimum
        items = [item for item in collection.vacancies if item.description_status == "full"]
        items.sort(key=lambda item: -item.morphology_similarity)
        diagnostics.raw_results = collection.diagnostics.search_results
        diagnostics.deduplicated_results = len(items)
        errors = [f"{item.source}/{item.stage}: {item.reason}"
                  for item in collection.diagnostics.errors]
        return items[:result_limit], errors

    result = build_market_profile_v2(
        target, output_path=output_path, searcher=searcher,
        use_embeddings=use_embeddings, use_llm=use_llm,
        normalizer=normalize_full_descriptions,
        query_builder=lambda profile: russian_search_queries(profile, search_titles=collection.diagnostics.search_titles),
        title_matcher=lambda name, title: bilingual_title_similarity(name, title, collection.diagnostics.search_titles),
        market_as_of=collection.diagnostics.scope.collected_at,
        collection_duplicates=collection.diagnostics.duplicates,
    )
    scope = collection.diagnostics.scope
    result = result.model_copy(update={"summary": (
        f"Россия, публикации за {scope.lookback_days} дней. "
        f"Цель из резюме: {target.label()}; уровень: {target.seniority or 'не указан'}. "
        + result.summary
    )})
    _save(result, output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Рынок России по должности из резюме, 120 дней")
    parser.add_argument("resume", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--vacancies-json", type=Path,
                        help="Полные описания в JSON вместо интернет-запросов")
    parser.add_argument("--collect-only", action="store_true",
                        help="Сохранить исходные вакансии и диагностику без построения профиля")
    parser.add_argument("--no-title-translation", action="store_true",
                        help="Не вызывать модель для перевода должности; словарь и кеш остаются доступны")
    args = parser.parse_args()
    resume = (ResumeData.model_validate_json(args.resume.read_text(encoding="utf-8"))
              if args.resume.suffix.casefold() == ".json"
              else extract_resume(read_resume(args.resume)))
    target = target_from_resume(resume)
    if args.collect_only:
        collection = collect_russian_vacancies(target, args.output,
            imported_path=args.vacancies_json, use_title_translation=not args.no_title_translation)
        print(f"{collection.diagnostics.status}: полных описаний {collection.diagnostics.accepted}")
    else:
        result = build_russian_market_profile(target, args.output,
            imported_path=args.vacancies_json, use_title_translation=not args.no_title_translation)
        print(f"{result.status}: sample={result.sample_size}")
    print(f"Сохранено: {args.output}")


if __name__ == "__main__":
    main()
