"""Упрощённый продуктовый MarketProfileBuilder v2.

V2 существует отдельно от research/production pipeline. Он использует
лексические и dense-сигналы для ранжирования, а не как последовательность
жёстких gates. Единственное содержательное решение о пригодности вакансии
принимает relevance validation. Результат сохраняет контракт существующих
``MarketProfile`` и ``InsufficientMarketProfile``.
"""


from __future__ import annotations
from .runtime import atomic_write_text

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

from ollama import ResponseError
from .model_runtime import chat
from pydantic import BaseModel, Field, ValidationError

from .profession import ProfessionProfile, TargetProfile, target_profile_from_value
from .requirement_equivalence import aggregate_equivalent_requirements
from .market_reliability import assess_market_reliability, deduplicate_market_vacancies
from .market_profile import (
    EMBEDDING_MODEL,
    MAX_SEARCH_CANDIDATES,
    MIN_PUBLIC_VACANCIES,
    MODEL_CONTEXT_SIZE,
    MODEL_NAME,
    AdjacentMarketContext,
    DuplicateVacancyRecord,
    InsufficientMarketProfile,
    MarketProfile,
    MarketReliability,
    NormalizedVacancy,
    ProfessionSearchPlan,
    RawVacancy,
    SearchCollectionStatistics,
    SyntheticVacancyError,
    _embed_text_map,
    _market_profile_text,
    _market_source,
    _market_target,
    _seniority_marker,
    _text_key,
    _unique_text,
    build_target_profile,
    embedding_cosine_similarity,
    normalize_vacancies,
    market_profile_summary,
    search_public_vacancies,
    title_match_signals,
)


DEFAULT_OUTPUT = Path("market_profile_v2.json")
DEFAULT_CANDIDATE_LIMIT = 20
DEFAULT_SAMPLE_LIMIT = 6  # Максимум текущей совместимой MarketProfile-схемы.
MAX_QUERY_COUNT = 6
RELEVANCE_BATCH_SIZE = 5
V2_CACHE_NAMESPACE = "market_profile_v2"

RelevanceLabel = Literal["direct", "adjacent", "irrelevant"]
ValidationConfidence = Literal["low", "medium", "high"]


class RankedVacancy(BaseModel):
    vacancy: RawVacancy
    lexical_score: float = Field(ge=0, le=1)
    alias_score: float = Field(ge=0, le=1)
    embedding_score: float | None = Field(default=None, ge=-1, le=1)
    domain_similarity: float | None = Field(default=None, ge=-1, le=1)
    ranking_score: float = Field(ge=0, le=1)
    excluded_signal: bool = False


class RelevanceAssessment(BaseModel):
    source_id: str
    relevance: RelevanceLabel
    confidence: ValidationConfidence
    reason: str


class RelevanceAssessmentBatch(BaseModel):
    vacancies: list[RelevanceAssessment]


class V2FinalVacancyDiagnostic(BaseModel):
    source_id: str
    title: str
    provider: str
    query: str | None
    relevance: RelevanceLabel
    reason: str
    ranking_score: float


class V2PostNormalizationRejection(BaseModel):
    source_id: str
    title: str
    reason: str


class MarketProfileV2Diagnostics(BaseModel):
    namespace: Literal["market_profile_v2"] = V2_CACHE_NAMESPACE
    target: TargetProfile
    queries: list[str]
    raw_count: int
    deduplicated_count: int
    candidate_count: int
    direct_count: int
    adjacent_count: int
    irrelevant_count: int
    final_sample_size: int
    adjacent_sample_size: int = 0
    confidence: Literal["low", "medium", "high"]
    provisional: bool
    providers_used: list[str]
    provider_result_counts: dict[str, int]
    source_errors: list[str]
    embeddings_requested: bool
    embeddings_available: bool
    llm_requested: bool
    llm_relevance_available: bool
    relevance_llm_calls: int = 0
    normalization_candidates: int = 0
    elapsed_seconds: float
    final_vacancies: list[V2FinalVacancyDiagnostic]
    adjacent_vacancies: list[V2FinalVacancyDiagnostic] = Field(default_factory=list)
    normalization_rounds: int = 0
    backfill_count: int = 0
    reliability: MarketReliability | None = None
    duplicate_vacancies: list[DuplicateVacancyRecord] = Field(default_factory=list)
    outcome_reason: Literal["no_repeated_market_requirements"] | None = None
    post_normalization_rejections: list[V2PostNormalizationRejection] = Field(
        default_factory=list
    )


RELEVANCE_SYSTEM_PROMPT = """
Ты проверяешь вакансии для небольшого прикладного обзора рынка.

Для каждой вакансии ответь, является ли она разумным примером требований для
человека, который ищет работу по указанному TargetProfile:
- direct: основная работа соответствует целевой специализации;
- adjacent: направление смежное и требования полезны как дополнительный
  ориентир, но основная специализация шире или немного отличается;
- irrelevant: это другая профессия, другой основной focus/domain либо явно
  несовместимый seniority.

Если целевой уровень Middle, вакансии с явным Senior/Lead/Staff/Principal/Head
в title считай irrelevant. Если title явно соответствует только
adjacent_specializations, не называй вакансию direct.

Оцени содержание title и description вместе. Не требуй дословного совпадения
названия. Excluded specializations являются сильным негативным контекстом, но
не заменяют чтение описания. Не добавляй фактов. Для каждого source_id верни
relevance, confidence и одно короткое проверяемое объяснение.
"""


SearchFunction = Callable[..., tuple[list[RawVacancy], list[str]]]
EmbeddingFunction = Callable[[list[str], str], dict[str, list[float]]]
RelevanceFunction = Callable[
    [list[RankedVacancy], TargetProfile, Path | None],
    tuple[list[RelevanceAssessment], bool, int],
]
NormalizationFunction = Callable[..., list[NormalizedVacancy]]


def generate_search_queries_v2(target: TargetProfile) -> list[str]:
    """Возвращает 3–6 простых универсальных запросов без combinatorial growth."""
    names = _unique_text([
        target.specialization or target.role or target.domain or "",
        *target.aliases,
    ])
    primary = names[0] if names else target.label()
    queries: list[str] = []
    if primary:
        queries.append(primary)
        if target.seniority:
            queries.append(f"{target.seniority} {primary}")
        queries.append(f"{primary} remote")
    queries.extend(names[1:4])
    if len(_unique_text(queries)) < 3:
        context = " ".join(
            value for value in (primary, target.domain, target.focus) if value
        )
        queries.extend([context, f"{primary} vacancy", f"{primary} job"])
    return _unique_text(queries)[:MAX_QUERY_COUNT]


def _coerce_target(
    target: ProfessionSearchPlan | TargetProfile | ProfessionProfile | str,
    *,
    use_llm: bool,
) -> TargetProfile:
    if isinstance(target, ProfessionSearchPlan):
        return _clean_target_seniority(
            target.profession.model_copy(update={"core_skills": []})
        )
    if isinstance(target, TargetProfile):
        return _clean_target_seniority(
            target.model_copy(update={"core_skills": []})
        )
    if isinstance(target, str) and use_llm:
        try:
            return _clean_target_seniority(
                build_target_profile(target).profession.model_copy(
                    update={"core_skills": []}
                ),
                source=target,
            )
        except (SyntheticVacancyError, ResponseError, OSError, ValidationError):
            pass
    return _clean_target_seniority(
        target_profile_from_value(target).model_copy(update={"core_skills": []}),
        source=target if isinstance(target, str) else None,
    )


_SENIORITY_TEXT_PATTERNS = {
    "junior": r"junior|jr\.?|джун(?:иор)?|младш(?:ий|ая|ее)?",
    "middle": r"middle|mid[-\s]?level|мидд?л|средн(?:ий|яя|ее)\s+уровень",
    "senior": r"senior|sr\.?|сеньор|старш(?:ий|ая|ее)?",
    "lead": r"lead|team\s*lead|лид|ведущ(?:ий|ая|ее)?",
    "staff": r"staff",
    "principal": r"principal|главн(?:ый|ая|ое|ого|ой)",
    "head": r"head|руководител[ья]",
}
_SENIORITY_LABELS = {
    "junior": "Junior",
    "middle": "Middle",
    "senior": "Senior",
    "lead": "Lead",
    "staff": "Staff",
    "principal": "Principal",
    "head": "Head",
}


def _clean_target_seniority(
    profile: TargetProfile,
    *,
    source: str | None = None,
) -> TargetProfile:
    """Выделяет seniority и удаляет его только из specialization цели."""
    marker = _seniority_marker(profile.seniority) or _seniority_marker(source)
    if marker is None:
        return profile
    specialization = profile.specialization
    pattern = _SENIORITY_TEXT_PATTERNS.get(marker)
    if specialization and pattern:
        specialization = re.sub(
            rf"(?<!\w)(?:{pattern})(?!\w)",
            " ",
            specialization,
            flags=re.IGNORECASE,
        )
        specialization = " ".join(specialization.split()).strip(" /|,;-–—")
    return profile.model_copy(
        update={
            "specialization": specialization or profile.specialization,
            "seniority": profile.seniority or _SENIORITY_LABELS[marker],
        }
    )


def _has_repeated_market_requirements(profile: MarketProfile) -> bool:
    """Проверяет наличие хотя бы одного common/additional требования."""
    categories = (
        profile.professional_skills,
        profile.tools_and_technologies,
        profile.responsibilities,
        profile.education_requirements,
        profile.domain_knowledge,
        profile.soft_skills,
    )
    return any(
        item.importance in {"common", "additional"}
        for category in categories
        for item in category
    )


def _ranking_text(vacancy: RawVacancy) -> str:
    # Полного текста достаточно для semantic ranking; ограничение защищает от
    # огромных HTML-описаний и не влияет на последующую validation/extraction.
    return f"{vacancy.title}\n{vacancy.raw_text[:4000]}"


def rank_vacancies_v2(
    vacancies: list[RawVacancy],
    target: TargetProfile,
    *,
    use_embeddings: bool = True,
    embedding_model: str = EMBEDDING_MODEL,
    embedder: EmbeddingFunction = _embed_text_map,
    title_matcher: Callable[[str, str], float] | None = None,
) -> tuple[list[RankedVacancy], bool]:
    """Ранжирует все вакансии; ни lexical, ни cosine не являются hard gate."""
    target_text = _market_profile_text(target)
    domain_text = " ".join(
        value for value in (target.domain, target.role_family, target.focus) if value
    )
    vacancy_texts = [_ranking_text(item) for item in vacancies]
    vectors: dict[str, list[float]] = {}
    embeddings_available = False
    if use_embeddings and vacancies:
        try:
            vectors = embedder(
                list(dict.fromkeys(
                    text for text in (target_text, domain_text, *vacancy_texts)
                    if text
                )),
                embedding_model,
            )
            embeddings_available = bool(vectors.get(target_text))
        except (SyntheticVacancyError, ResponseError, OSError, ValueError):
            vectors = {}

    ranked: list[RankedVacancy] = []
    for vacancy, content in zip(vacancies, vacancy_texts, strict=True):
        target_vector = vectors.get(target_text, [])
        content_vector = vectors.get(content, [])
        domain_vector = vectors.get(domain_text, []) if domain_text else []
        semantic = (
            embedding_cosine_similarity(target_vector, content_vector)
            if target_vector and content_vector
            else None
        )
        domain = (
            embedding_cosine_similarity(domain_vector, content_vector)
            if domain_vector and content_vector
            else None
        )
        signals = title_match_signals(
            target,
            vacancy.title,
            dense_similarity=max(0.0, semantic or 0.0),
        )
        lexical = max(
            1.0 if signals.exact_alias_match else 0.0,
            0.9 if signals.normalized_phrase_match else 0.0,
            signals.token_overlap,
            signals.fuzzy_lexical_similarity * 0.85,
        )
        alias = max(
            1.0 if signals.exact_alias_match else 0.0,
            0.85 if signals.normalized_phrase_match else 0.0,
            0.55 if signals.adjacent_match else 0.0,
        )
        if title_matcher is not None:
            morphology = max(
                (title_matcher(name, vacancy.title) for name in
                 [target.label(), *target.aliases] if name),
                default=0.0,
            )
            lexical = max(lexical, morphology)
            if morphology == 1.0:
                alias = max(alias, 1.0)
        semantic_component = max(0.0, semantic or 0.0)
        domain_component = max(0.0, domain or 0.0)
        score = (
            0.40 * lexical
            + 0.15 * alias
            + 0.35 * semantic_component
            + 0.10 * domain_component
        )
        if signals.excluded_match:
            score *= 0.35
        vacancy.title_match_score = round(lexical, 6)
        vacancy.embedding_score = semantic
        vacancy.embedding_domain_similarity = domain
        ranked.append(
            RankedVacancy(
                vacancy=vacancy,
                lexical_score=round(lexical, 6),
                alias_score=round(alias, 6),
                embedding_score=semantic,
                domain_similarity=domain,
                ranking_score=round(max(0.0, min(1.0, score)), 6),
                excluded_signal=signals.excluded_match,
            )
        )
    ranked.sort(
        key=lambda item: (
            -item.ranking_score,
            item.excluded_signal,
            item.vacancy.source_id,
        )
    )
    return ranked, embeddings_available


def _assessment_cache_key(target: TargetProfile, vacancy: RawVacancy) -> str:
    payload = json.dumps(
        {
            "target": target.model_dump(),
            "title": vacancy.title,
            "text": vacancy.raw_text,
            "model": MODEL_NAME,
            "prompt_sha256": hashlib.sha256(RELEVANCE_SYSTEM_PROMPT.encode()).hexdigest(),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_relevance_cache(path: Path | None) -> dict[str, RelevanceAssessment]:
    if path is None or not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("namespace") != V2_CACHE_NAMESPACE:
            return {}
        return {
            key: RelevanceAssessment.model_validate(value)
            for key, value in raw.get("items", {}).items()
        }
    except (OSError, ValueError, ValidationError, AttributeError):
        return {}


def _save_relevance_cache(
    path: Path | None,
    cache: dict[str, RelevanceAssessment],
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path,
        json.dumps(
            {
                "namespace": V2_CACHE_NAMESPACE,
                "items": {
                    key: item.model_dump() for key, item in cache.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
    )


def _fallback_assessment(item: RankedVacancy) -> RelevanceAssessment:
    # Без LLM принимаются только сильные и объяснимые title/alias-сигналы.
    if item.excluded_signal:
        relevance: RelevanceLabel = "irrelevant"
        reason = "Название совпало с явно исключённой специализацией."
    elif item.alias_score >= 0.85 or item.lexical_score >= 0.88:
        relevance = "direct"
        reason = "Сильное совпадение названия с целевой специализацией или alias."
    elif item.alias_score >= 0.55:
        relevance = "adjacent"
        reason = "Название соответствует указанной смежной специализации."
    else:
        relevance = "irrelevant"
        reason = "Без LLM недостаточно сильного lexical/alias evidence."
    return RelevanceAssessment(
        source_id=item.vacancy.source_id,
        relevance=relevance,
        confidence="low",
        reason=reason,
    )


def _relevance_sanity_check(
    assessment: RelevanceAssessment,
    candidate: RankedVacancy,
    target: TargetProfile,
) -> RelevanceAssessment:
    """Исправляет только явные противоречия structured TargetProfile.

    Это не второй semantic validator: функция не читает описание и не
    переоценивает профессию. Она лишь не позволяет LLM объявить direct явно
    excluded/adjacent title или вакансию с более высоким указанным уровнем.
    """
    signals = title_match_signals(target, candidate.vacancy.title)
    if signals.excluded_match:
        return assessment.model_copy(
            update={
                "relevance": "irrelevant",
                "confidence": "high",
                "reason": (
                    "Название явно соответствует excluded specialization; "
                    + assessment.reason
                ),
            }
        )
    target_level = _seniority_marker(target.seniority)
    vacancy_level = _seniority_marker(candidate.vacancy.title)
    ranks = {
        "junior": 1,
        "middle": 2,
        "senior": 3,
        "lead": 4,
        "staff": 4,
        "principal": 5,
        "head": 5,
    }
    if (
        target_level in ranks
        and vacancy_level in ranks
        and ranks[vacancy_level] > ranks[target_level]
    ):
        return assessment.model_copy(
            update={
                "relevance": "irrelevant",
                "confidence": "high",
                "reason": (
                    f"Явный seniority {vacancy_level} выше целевого "
                    f"{target_level}; {assessment.reason}"
                ),
            }
        )
    if (
        assessment.relevance == "direct"
        and signals.adjacent_match
        and not signals.exact_alias_match
        and not signals.normalized_phrase_match
    ):
        return assessment.model_copy(
            update={
                "relevance": "adjacent",
                "reason": (
                    "Название соответствует adjacent specialization; "
                    + assessment.reason
                ),
            }
        )
    return assessment


def validate_relevance_v2(
    candidates: list[RankedVacancy],
    target: TargetProfile,
    cache_path: Path | None = None,
    *,
    use_llm: bool = True,
) -> tuple[list[RelevanceAssessment], bool, int]:
    cache = _read_relevance_cache(cache_path)
    results: dict[str, RelevanceAssessment] = {}
    pending: list[RankedVacancy] = []
    for item in candidates:
        key = _assessment_cache_key(target, item.vacancy)
        cached = cache.get(key)
        if cached is None:
            pending.append(item)
        else:
            results[item.vacancy.source_id] = cached.model_copy(
                update={"source_id": item.vacancy.source_id}
            )

    llm_available = use_llm
    calls = 0
    if use_llm:
        for offset in range(0, len(pending), RELEVANCE_BATCH_SIZE):
            batch = pending[offset : offset + RELEVANCE_BATCH_SIZE]
            payload = {
                "target": target.model_dump(),
                "vacancies": [
                    {
                        "source_id": item.vacancy.source_id,
                        "title": item.vacancy.title,
                        "description": item.vacancy.raw_text[:7000],
                        "ranking_score": item.ranking_score,
                    }
                    for item in batch
                ],
            }
            try:
                calls += 1
                response = chat(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": RELEVANCE_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": json.dumps(payload, ensure_ascii=False, indent=2),
                        },
                    ],
                    format=RelevanceAssessmentBatch.model_json_schema(),
                    think=False,
                    options={
                        "temperature": 0,
                        "num_ctx": MODEL_CONTEXT_SIZE,
                        "num_predict": 2048,
                    },
                )
                parsed = RelevanceAssessmentBatch.model_validate_json(
                    response.message.content
                )
                expected = {item.vacancy.source_id for item in batch}
                for assessment in parsed.vacancies:
                    if assessment.source_id in expected:
                        results[assessment.source_id] = assessment
                for item in batch:
                    if item.vacancy.source_id not in results:
                        results[item.vacancy.source_id] = _fallback_assessment(item)
            except (ResponseError, OSError, ValueError, ValidationError):
                llm_available = False
                for item in batch:
                    results[item.vacancy.source_id] = _fallback_assessment(item)
    else:
        llm_available = False
        for item in pending:
            results[item.vacancy.source_id] = _fallback_assessment(item)

    for item in candidates:
        assessment = results.get(item.vacancy.source_id)
        if assessment is None:
            assessment = _fallback_assessment(item)
        assessment = _relevance_sanity_check(assessment, item, target)
        results[item.vacancy.source_id] = assessment
        cache[_assessment_cache_key(target, item.vacancy)] = assessment
    _save_relevance_cache(cache_path, cache)
    return [results[item.vacancy.source_id] for item in candidates], llm_available, calls


def _normalize_resiliently(
    vacancies: list[RawVacancy],
    target: TargetProfile,
    *,
    checkpoint_path: Path,
    cache_path: Path,
    normalizer: NormalizationFunction,
) -> list[NormalizedVacancy]:
    try:
        return normalizer(
            vacancies,
            target,
            checkpoint_path=checkpoint_path,
            cache_path=cache_path,
        )
    except (SyntheticVacancyError, ResponseError, OSError, ValueError, ValidationError):
        result: list[NormalizedVacancy] = []
        for vacancy in vacancies:
            try:
                result.extend(
                    normalizer(
                        [vacancy],
                        target,
                        checkpoint_path=checkpoint_path,
                        cache_path=cache_path,
                    )
                )
            except (
                SyntheticVacancyError,
                ResponseError,
                OSError,
                ValueError,
                ValidationError,
            ):
                continue
        return result


def _normalized_seniority_rejection(
    target: TargetProfile,
    vacancy: NormalizedVacancy,
) -> str | None:
    """Проверяет seniority, извлечённый из полного описания вакансии."""
    target_level = _seniority_marker(target.seniority)
    vacancy_level = _seniority_marker(vacancy.profession.seniority)
    ranks = {
        "junior": 1,
        "middle": 2,
        "senior": 3,
        "lead": 4,
        "staff": 4,
        "principal": 5,
        "head": 5,
    }
    if (
        target_level in ranks
        and vacancy_level in ranks
        and ranks[vacancy_level] > ranks[target_level]
    ):
        return (
            f"Нормализация извлекла seniority {vacancy_level}, который выше "
            f"целевого {target_level}."
        )
    return None


def _save_json_model(value: BaseModel, path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, value.model_dump_json(indent=2))


def _search_public_vacancies_v2(
    plan: ProfessionSearchPlan, minimum: int, diagnostics: SearchCollectionStatistics,
    strict_minimum: bool, result_limit: int,
) -> tuple[list[RawVacancy], list[str]]:
    # V1 объединяет одинаковые title/employer ещё до возврата результатов.
    # V2 должен видеть исходные карточки, чтобы сравнить описания и сохранить
    # причины удаления копий. Ограничение пула применяется после объединения.
    del result_limit
    return search_public_vacancies(
        plan, minimum=minimum, diagnostics=diagnostics, strict_minimum=strict_minimum,
        result_limit=None, deduplicate_results=False,
    )


def build_market_profile_v2(
    target: ProfessionSearchPlan | TargetProfile | ProfessionProfile | str,
    output_path: Path | None = DEFAULT_OUTPUT,
    use_embeddings: bool = True,
    use_llm: bool = True,
    *,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
    minimum_search_results: int = MIN_PUBLIC_VACANCIES,
    embedding_model: str = EMBEDDING_MODEL,
    searcher: SearchFunction = _search_public_vacancies_v2,
    embedder: EmbeddingFunction = _embed_text_map,
    relevance_validator: RelevanceFunction | None = None,
    normalizer: NormalizationFunction = normalize_vacancies,
    query_builder: Callable[[TargetProfile], list[str]] | None = None,
    title_matcher: Callable[[str, str], float] | None = None,
    market_as_of: datetime | None = None,
    collection_duplicates: int = 0,
) -> MarketProfile | InsufficientMarketProfile:
    """Строит небольшой candidate-independent рыночный ориентир.

    Dependency arguments предназначены для детерминированных тестов. Ни один
    из них не получает ResumeData или сведения о навыках кандидата.
    """
    if not 1 <= candidate_limit <= MAX_SEARCH_CANDIDATES:
        raise ValueError(f"candidate_limit должен быть от 1 до {MAX_SEARCH_CANDIDATES}")
    if not 3 <= sample_limit <= DEFAULT_SAMPLE_LIMIT:
        raise ValueError("sample_limit должен быть от 3 до 6 для MarketProfile")

    started = time.perf_counter()
    profile = _coerce_target(target, use_llm=use_llm)
    queries = (query_builder or generate_search_queries_v2)(profile)
    plan = ProfessionSearchPlan(profession=profile, search_queries=queries)
    print(f"[v2 1/5] Search: {profile.label()} ({len(queries)} queries)", flush=True)
    base_path = output_path or DEFAULT_OUTPUT
    diagnostics_path = base_path.with_name(f"{base_path.stem}.diagnostics.json")
    checkpoint_path = base_path.with_name(
        f"{base_path.stem}.market_profile_v2.normalization_checkpoint.json"
    )
    normalization_cache_path = base_path.with_name(
        f"{base_path.stem}.market_profile_v2.normalization_cache.json"
    )
    relevance_cache_path = base_path.with_name(
        f"{base_path.stem}.market_profile_v2.relevance_cache.json"
    )
    search_stats = SearchCollectionStatistics()
    source_errors: list[str] = []
    try:
        found, source_errors = searcher(
            plan,
            minimum=max(MIN_PUBLIC_VACANCIES, minimum_search_results),
            diagnostics=search_stats,
            strict_minimum=False,
            result_limit=MAX_SEARCH_CANDIDATES,
        )
    except (SyntheticVacancyError, OSError, ValueError) as error:
        found = []
        source_errors = [str(error)]
    unique, duplicates = deduplicate_market_vacancies(found)
    deduplicated_count = len(unique)
    unique = unique[:MAX_SEARCH_CANDIDATES]
    print(
        f"[v2 2/5] Ranking: raw={search_stats.raw_results or len(found)}, "
        f"dedup={deduplicated_count}, ranking_pool={len(unique)}",
        flush=True,
    )
    ranked, embeddings_available = rank_vacancies_v2(
        unique,
        profile,
        use_embeddings=use_embeddings,
        embedding_model=embedding_model,
        embedder=embedder,
        title_matcher=title_matcher,
    )
    # candidate_limit ограничивает порцию проверки, а не весь резерв добора.
    # Сначала заполняем direct; adjacent имеет собственный лимит и знаменатель.
    candidates: list[RankedVacancy] = []
    assessments: list[RelevanceAssessment] = []
    labels: dict[str, RelevanceAssessment] = {}
    llm_available = True
    relevance_calls = 0
    attempted: list[RawVacancy] = []
    normalization_rounds = 0
    backfill_count = 0
    round_paths: list[str] = []
    sample: list[NormalizedVacancy] = []
    adjacent_sample: list[NormalizedVacancy] = []
    post_normalization_rejections: list[V2PostNormalizationRejection] = []
    for offset in range(0, len(ranked), candidate_limit):
        batch = ranked[offset:offset + candidate_limit]
        print(f"[v2 3/5] Relevance validation: next={len(batch)}, checked={len(candidates)}", flush=True)
        if relevance_validator is None:
            decisions, available, calls = validate_relevance_v2(
                batch, profile, relevance_cache_path, use_llm=use_llm,
            )
        else:
            decisions, available, calls = relevance_validator(batch, profile, relevance_cache_path)
        candidates.extend(batch)
        assessments.extend(decisions)
        labels.update({item.source_id: item for item in decisions})
        llm_available = llm_available and available
        relevance_calls += calls
        for relevance, cohort in (("direct", sample), ("adjacent", adjacent_sample)):
            queue = [item for item in batch if labels.get(item.vacancy.source_id)
                     and labels[item.vacancy.source_id].relevance == relevance]
            cursor = 0
            while len(cohort) < sample_limit and cursor < len(queue):
                selected = queue[cursor:cursor + sample_limit - len(cohort)]
                cursor += len(selected)
                if attempted and relevance == "direct":
                    backfill_count += len(selected)
                selected_raw = [item.vacancy for item in selected]
                attempted.extend(selected_raw)
                normalization_rounds += 1
                round_path = checkpoint_path.with_name(f"{checkpoint_path.stem}.round{normalization_rounds:02d}.json")
                round_paths.append(str(round_path))
                print(f"[v2 4/5] Normalization: {relevance}, next={len(selected_raw)}, accepted={len(cohort)}", flush=True)
                normalized = _normalize_resiliently(
                    selected_raw, profile, checkpoint_path=round_path,
                    cache_path=normalization_cache_path, normalizer=normalizer,
                )
                normalized_by_id = {item.source_id: item for item in normalized}
                for rank in selected:
                    raw = rank.vacancy
                    item = normalized_by_id.get(raw.source_id)
                    reason = ("Нормализация не вернула вакансию." if item is None else
                              f"Качество извлечения: {item.extraction_status}." if item.extraction_status != "ok" else
                              _normalized_seniority_rejection(profile, item))
                    if reason:
                        post_normalization_rejections.append(V2PostNormalizationRejection(
                            source_id=raw.source_id, title=raw.title, reason=reason,
                        ))
                        continue
                    # Источник берём из исходной карточки, а не из ответа модели.
                    cohort.append(item.model_copy(update={
                        "source": raw.source, "url": raw.url, "title": raw.title,
                        "employer": raw.employer, "published_at": raw.published_at,
                        "employer_id": raw.employer_id, "region": raw.region,
                        "fetched_at": raw.fetched_at, "description_status": raw.description_status,
                        "is_relevant": True, "relevance_score": rank.ranking_score,
                        "profession_relevance": 0.9 if relevance == "direct" else 0.65,
                    }))
        if len(sample) >= sample_limit:
            break
    llm_available = llm_available and bool(candidates)
    rank_by_id = {item.vacancy.source_id: item for item in candidates}
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(checkpoint_path, json.dumps({
        "rounds": round_paths, "normalized_count": len(sample),
        "adjacent_normalized_count": len(adjacent_sample),
        "issues": [item.model_dump() for item in post_normalization_rejections],
    }, ensure_ascii=False, indent=2))

    adjacent_context = None
    if adjacent_sample:
        categories, variations, equivalence = aggregate_equivalent_requirements(adjacent_sample, use_llm=use_llm)
        _save_json_model(equivalence, base_path.with_name(f"{base_path.stem}.adjacent_concepts.json"))
        adjacent_context = AdjacentMarketContext(
            sample_size=len(adjacent_sample), **categories, market_variations=variations,
            sources=[_market_source(item) for item in adjacent_sample],
            summary=(f"Отдельно изучено {len(adjacent_sample)} смежных вакансий. "
                     "Их требования служат контекстом возможного развития и не входят "
                     "в основные требования, частоты и оценку соответствия целевой должности."),
        )

    reliability = assess_market_reliability(
        sample, {key: (value.relevance, value.confidence) for key, value in labels.items()},
        embeddings_available=embeddings_available,
        llm_available=llm_available,
        now=market_as_of, duplicates_removed=len(duplicates) + collection_duplicates,
    )
    confidence = reliability.confidence
    final_rows = [
        V2FinalVacancyDiagnostic(
            source_id=item.source_id,
            title=item.title,
            provider=item.source,
            query=next(
                (
                    ranked_item.vacancy.search_query
                    for ranked_item in candidates
                    if ranked_item.vacancy.source_id == item.source_id
                ),
                None,
            ),
            relevance=labels[item.source_id].relevance,
            reason=labels[item.source_id].reason,
            ranking_score=rank_by_id[item.source_id].ranking_score,
        )
        for item in sample
    ]
    diagnostics = MarketProfileV2Diagnostics(
        target=profile,
        queries=queries,
        raw_count=search_stats.raw_results or len(found),
        deduplicated_count=deduplicated_count,
        candidate_count=len(candidates),
        direct_count=sum(item.relevance == "direct" for item in assessments),
        adjacent_count=sum(item.relevance == "adjacent" for item in assessments),
        irrelevant_count=sum(item.relevance == "irrelevant" for item in assessments),
        final_sample_size=len(sample),
        adjacent_sample_size=len(adjacent_sample),
        confidence=confidence,
        provisional=len(sample) < 3 or confidence == "low",
        providers_used=sorted({item.source for item in sample}),
        provider_result_counts=dict(
            sorted(Counter(item.source for item in found).items())
        ),
        source_errors=source_errors,
        embeddings_requested=use_embeddings,
        embeddings_available=embeddings_available,
        llm_requested=use_llm,
        llm_relevance_available=llm_available,
        relevance_llm_calls=relevance_calls,
        normalization_candidates=len(attempted),
        normalization_rounds=normalization_rounds,
        backfill_count=backfill_count,
        elapsed_seconds=round(time.perf_counter() - started, 3),
        final_vacancies=final_rows,
        adjacent_vacancies=[V2FinalVacancyDiagnostic(
            source_id=item.source_id, title=item.title, provider=item.source,
            query=rank_by_id[item.source_id].vacancy.search_query,
            relevance="adjacent", reason=labels[item.source_id].reason,
            ranking_score=rank_by_id[item.source_id].ranking_score,
        ) for item in adjacent_sample],
        post_normalization_rejections=post_normalization_rejections,
        reliability=reliability,
        duplicate_vacancies=duplicates,
    )
    _save_json_model(diagnostics, diagnostics_path)
    print(
        f"[v2 5/5] Aggregation: normalized sample={len(sample)}, "
        f"sampling confidence={confidence}",
        flush=True,
    )

    if len(sample) < 3:
        if len(sample) == 2:
            message = (
                "Найдены две прямые вакансии целевой должности. Это полезный "
                "предварительный сигнал, но текущий совместимый MarketProfile "
                "требует минимум три источника."
            )
        elif len(sample) == 1:
            message = "Найдена только одна прямая вакансия целевой должности."
        else:
            message = "Не найдено прямых вакансий целевой должности."
        result: MarketProfile | InsufficientMarketProfile = InsufficientMarketProfile(
            target={
                "role": profile.role,
                "specialization": profile.specialization,
                "domain": profile.domain,
                "seniority": profile.seniority,
                "role_family": profile.role_family,
                "focus": profile.focus,
            },
            sample_size=len(sample),
            message=message,
            summary=(
                f"{message} Результат основан на небольшой доступной выборке "
                "и не является описанием всего рынка."
            ),
            sources=[_market_source(item) for item in sample],
            adjacent_context=adjacent_context,
            reliability=reliability,
        )
        _save_json_model(result, output_path)
        return result

    categories, variations, equivalence = aggregate_equivalent_requirements(sample, use_llm=use_llm)
    _save_json_model(equivalence, base_path.with_name(f"{base_path.stem}.direct_concepts.json"))
    diagnostics.elapsed_seconds = round(time.perf_counter() - started, 3)
    _save_json_model(diagnostics, diagnostics_path)
    result = MarketProfile(
        target=_market_target(profile), sample_size=len(sample), **categories,
        market_variations=variations, sources=[_market_source(item) for item in sample],
        adjacent_context=adjacent_context, confidence=confidence, reliability=reliability,
        summary=(
                f"Приблизительный актуальный ориентир для {profile.label()}, "
                f"построенный по небольшой выборке доступных вакансий: "
                f"{len(sample)} прямых; {len(adjacent_sample)} смежных сохранены отдельно. "
                "Формулировки отражают только рассмотренную выборку, а не весь рынок. "
                + market_profile_summary(profile, len(sample), categories["professional_skills"],
                                         categories["tools_and_technologies"], variations)
                + (" Уверенность низкая: советы являются предварительными." if confidence == "low" else "")
            ),
    )
    if not _has_repeated_market_requirements(result):
        message = (
            "В рыночной выборке не найдено ни одного требования, "
            "повторяющегося минимум в двух вакансиях."
        )
        reliability = reliability.model_copy(update={
            "confidence": "low", "provisional": True,
            "reasons": [*reliability.reasons, message],
        })
        diagnostics = diagnostics.model_copy(
            update={
                "provisional": True,
                "outcome_reason": "no_repeated_market_requirements",
                "confidence": "low", "reliability": reliability,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
        _save_json_model(diagnostics, diagnostics_path)
        insufficient = InsufficientMarketProfile(
            target=result.target,
            sample_size=len(sample),
            message=message,
            summary=(
                f"{message} Данных недостаточно для содержательного "
                "сравнения кандидата с рынком."
            ),
            sources=result.sources,
            adjacent_context=adjacent_context,
            reliability=reliability,
        )
        _save_json_model(insufficient, output_path)
        return insufficient
    _save_json_model(result, output_path)
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MarketProfileBuilder v2")
    parser.add_argument("target", help='Целевая позиция, например "Middle ML Engineer"')
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-embeddings", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--candidate-limit", type=int, default=DEFAULT_CANDIDATE_LIMIT)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    result = build_market_profile_v2(
        args.target,
        args.output,
        use_embeddings=not args.no_embeddings,
        use_llm=not args.no_llm,
        candidate_limit=args.candidate_limit,
    )
    print(
        f"status={result.status}; sample={result.sample_size}; "
        f"confidence={result.confidence}; output={args.output}"
    )


if __name__ == "__main__":
    main()
