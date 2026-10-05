"""Независимость наблюдений и объяснимая уверенность рыночной выборки."""

from __future__ import annotations

import html
import re
import unicodedata
from datetime import datetime, timezone
from typing import Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .market_profile import (
    DuplicateVacancyRecord, MarketReliability, MarketSource, NormalizedVacancy, RawVacancy,
)


def _text(value: str) -> str:
    plain = html.unescape(re.sub(r"</?[a-zA-Z][^>]*>", " ", value))
    tokens = re.findall(r">=|<=|!=|>|<|=|[\w+#.]+", unicodedata.normalize("NFKC", plain).casefold())
    return " ".join(token.rstrip(".") for token in tokens if token.rstrip("."))


def employer_name(value: str | None) -> str:
    words = _text(value or "").split()
    legal_forms = {"ооо", "оао", "зао", "пао", "ао", "llc", "ltd", "inc", "limited"}
    while words and words[0].rstrip(".") in legal_forms:
        words.pop(0)
    while words and words[-1].rstrip(".") in legal_forms:
        words.pop()
    key = " ".join(words)
    return "" if key in {"не указан", "неизвестно", "unknown", "confidential", "работодатель"} else key


def _url_key(url: str) -> str:
    parts = urlsplit(url)
    query = [(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
             if not key.casefold().startswith("utm_") and key.casefold() not in {"ref", "from", "tracking", "referral_id"}]
    return urlunsplit(("", parts.netloc.casefold(), parts.path.rstrip("/"), urlencode(sorted(query)), ""))


def deduplicate_market_vacancies(
    vacancies: list[RawVacancy],
) -> tuple[list[RawVacancy], list[DuplicateVacancyRecord]]:
    """Одинаковая карточка на разных площадках даёт одно наблюдение.

    Совпадения одного title/employer недостаточно: разные объявления одного
    работодателя сохраняются. Неизвестный работодатель не доказывает копию.
    """
    kept: list[RawVacancy] = []
    duplicate_records: list[tuple[int, DuplicateVacancyRecord]] = []
    identities: dict[tuple[str, str], int] = {}
    urls: dict[str, int] = {}
    descriptions: dict[tuple[str, str, str], list[int]] = {}
    content_keys: list[tuple[str, str, str]] = []
    providers: list[set[str]] = []
    regions: list[set[str]] = []
    for item in vacancies:
        identity = (item.source, item.source_id)
        url = _url_key(item.url)
        name = employer_name(item.employer)
        body = _text(item.raw_text)
        content = (name, _text(item.title), body)
        region = _text(item.region or "")
        index = identities.get(identity)
        reason = "same_identity"
        if index is None and url:
            index = urls.get(url)
            reason = "same_url"
        if index is None and name and len(body) >= 100:
            index = next((position for position in descriptions.get(content, [])
                          if item.source not in providers[position]
                          and (not region or not regions[position]
                               or regions[position] == {region})), None)
            reason = "cross_posted_description"
        if index is None:
            index = len(kept)
            kept.append(item)
            descriptions.setdefault(content, []).append(index)
            content_keys.append(content)
            providers.append(set())
            regions.append(set())
        else:
            previous = kept[index]
            # Полное описание предпочтительнее поискового фрагмента.
            if ((item.description_status == "full", len(item.raw_text)) >
                    (previous.description_status == "full", len(previous.raw_text))):
                kept[index] = item
                if content_keys[index] != content:
                    descriptions[content_keys[index]].remove(index)
                    descriptions.setdefault(content, []).append(index)
                    content_keys[index] = content
                duplicate_records.append((index, DuplicateVacancyRecord(
                    kept_source_id=item.source_id, removed_source_id=previous.source_id,
                    kept_url=item.url, removed_url=previous.url, reason=reason,
                )))
            else:
                duplicate_records.append((index, DuplicateVacancyRecord(
                    kept_source_id=previous.source_id, removed_source_id=item.source_id,
                    kept_url=previous.url, removed_url=item.url, reason=reason,
                )))
        # Смена представителя не должна забывать площадки и известные регионы
        # его копий: это могло бы объединить разные объявления той же площадки.
        providers[index].add(item.source)
        if region:
            regions[index].add(region)
        identities[identity] = index
        if url:
            urls[url] = index
    # Все записи о копиях указывают на окончательно выбранного представителя.
    # Локальные ID разных площадок могут совпадать; связь задаётся группой,
    # а не поиском по одному ID удалённой карточки.
    for index, row in duplicate_records:
        row.kept_source_id = kept[index].source_id
        row.kept_url = kept[index].url
    return kept, [row for _, row in duplicate_records]


def independent_employer_count(sample: Sequence[NormalizedVacancy | MarketSource]) -> tuple[int, int]:
    groups: list[set[tuple[str, ...]]] = []
    known = 0
    for item in sample:
        keys: set[tuple[str, ...]] = set()
        name = employer_name(item.employer)
        if name:
            keys.add(("name", name))
        if item.employer_id:
            keys.add(("id", item.source, item.employer_id))
        if not keys:
            continue
        known += 1
        connected = [group for group in groups if group & keys]
        for group in connected:
            keys.update(group)
            groups.remove(group)
        groups.append(keys)
    return len(groups), known


def _publication_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        # Не назначаем часовой пояс неизвестной дате молча.
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else None
    except ValueError:
        return None


def assess_market_reliability(
    sample: list[NormalizedVacancy], labels: Mapping[str, tuple[str, str]], *,
    embeddings_available: bool, llm_available: bool,
    now: datetime | None = None, lookback_days: int = 120, duplicates_removed: int = 0,
) -> MarketReliability:
    observed = now or datetime.now(timezone.utc)
    if observed.tzinfo is None:
        raise ValueError("Для оценки свежести нужен часовой пояс")
    if lookback_days < 1:
        raise ValueError("Период свежести должен быть положительным")
    employers, known = independent_employer_count(sample)
    direct = sum(labels.get(item.source_id, ("", ""))[0] == "direct" for item in sample)
    reliable = sum(labels.get(item.source_id, ("", ""))[1] in {"medium", "high"} for item in sample)
    full = sum(item.description_status == "full" for item in sample)
    dates = [date for item in sample if (date := _publication_date(item.published_at)) is not None]
    ages = [(observed - date).total_seconds() / 86400 for date in dates]
    recent = sum(0 <= age <= lookback_days for age in ages)
    providers = len({item.source for item in sample})
    size = len(sample)
    reasons = []
    if size < 5:
        reasons.append(f"Небольшая выборка: {size} вакансий; ориентир предварительный.")
    if employers < 3 or known < size:
        reasons.append(f"Подтверждено независимых работодателей: {employers}; работодатель известен для {known}/{size} вакансий.")
    if direct < size:
        reasons.append(f"Прямых совпадений с целевой должностью: {direct}/{size}.")
    if full < size:
        reasons.append(f"Полнота описания подтверждена для {full}/{size} вакансий.")
    if recent < size:
        reasons.append(f"Публикация за последние {lookback_days} дней подтверждена для {recent}/{size} вакансий.")
    if providers < 2:
        reasons.append(f"Представлено площадок: {providers}; разнообразие источников ограничено.")
    if not llm_available or reliable < size:
        reasons.append("Для части вакансий содержание профессии проверено с низкой уверенностью.")
    if not embeddings_available:
        reasons.append("Ранжирование выполнено по названиям и словарным совпадениям; смысловой поиск недоступен.")
    adequate = (size >= 5 and employers >= 3 and known == size and direct == size
                and full == size and recent == size and reliable == size
                and embeddings_available and llm_available)
    confidence = "low"
    if adequate:
        confidence = "high" if size == 6 and employers >= 4 and providers >= 2 else "medium"
    if not reasons:
        reasons.append("Прямые свежие вакансии с полными описаниями представлены несколькими независимыми работодателями и площадками.")
    return MarketReliability(
        confidence=confidence, provisional=confidence == "low", evaluated_at=observed,
        lookback_days=lookback_days, sample_size=size, independent_employers=employers,
        known_employer_count=known, direct_count=direct, full_description_count=full,
        recent_publication_count=recent, dated_publication_count=len(dates),
        reliable_relevance_count=reliable, provider_count=providers,
        duplicates_removed=duplicates_removed,
        oldest_publication_age_days=round(max(ages), 2) if ages else None, reasons=reasons,
    )
