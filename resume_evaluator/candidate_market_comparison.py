"""Сравнение содержания резюме с уже построенным MarketProfile.

Модуль не выполняет retrieval, не перестраивает рыночную выборку, не выдаёт
рекомендации и не рассчитывает интегральный market-fit score.
"""
from .runtime import atomic_write_text


import argparse
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .resume_extractor import ResumeData
from .requirement_equivalence import requirement_key
from .market_profile import (
    EMBEDDING_MODEL,
    InsufficientMarketProfile,
    MarketCategory,
    MarketImportance,
    MarketProfile,
    MarketRequirement,
    MarketRequirementEvidence,
    MarketReliability,
    MarketSource,
    MarketTarget,
    MarketVariation,
    _embed_text_map,
    embedding_cosine_similarity,
    normalize_skill_name,
)


DEFAULT_OUTPUT = Path("candidate_market_comparison.json")
SEMANTIC_MATCH_THRESHOLD = 0.72
CROSS_LANGUAGE_RESPONSIBILITY_SEMANTIC_WEIGHT = 0.75

_RESPONSIBILITY_PREFIX_WORDS = {
    "experience",
    "participation",
    "responsible",
    "work",
    "опыт",
    "участие",
    "ответственность",
    "работа",
}

# Cross-language responsibilities cannot share literal stems.  These compact,
# profession-independent verb families keep the existing semantic threshold
# useful without allowing a related object to substitute for a different act.
_RESPONSIBILITY_ACTION_PATTERNS: dict[str, tuple[str, ...]] = {
    "create": (
        r"creat",
        r"develop",
        r"design",
        r"build",
        r"produc",
        r"разраб",
        r"созда",
        r"проектир",
        r"дизайн",
        r"сверста",
        r"оформ",
    ),
    "prepare": (
        r"prepar",
        r"подгот",
    ),
    "manage": (
        r"manag",
        r"lead",
        r"coordinat",
        r"oversee",
        r"supervis",
        r"управ",
        r"руковод",
        r"координир",
        r"курир",
        r"вед(?:у|ет|ем|ете|ут|ал|ала|али|ение)",
    ),
    "analyze": (
        r"analy[sz]",
        r"evaluat",
        r"research",
        r"анализ",
        r"оцен",
        r"исслед",
    ),
    "monitor": (
        r"monitor",
        r"track",
        r"наблюд",
        r"монитор",
        r"отслеж",
    ),
    "implement": (
        r"implement",
        r"integrat",
        r"deploy",
        r"внедр",
        r"интегр",
        r"развер",
    ),
    "maintain": (
        r"maintain",
        r"support",
        r"сопровожд",
        r"поддерж",
    ),
    "test": (
        r"test",
        r"verif",
        r"validat",
        r"тестир",
        r"провер",
        r"валидир",
    ),
    "edit": (
        r"edit",
        r"retouch",
        r"process",
        r"редакт",
        r"ретуш",
        r"обработ",
    ),
    "optimize": (
        r"optimi[sz]",
        r"improv",
        r"оптимиз",
        r"улучш",
    ),
    "communicate": (
        r"communicat",
        r"collaborat",
        r"liais",
        r"взаимодейств",
        r"сотруднич",
        r"коммуницир",
    ),
}

ResumeStatus = Literal["matched", "evidenced", "mentioned_only", "not_found"]
EvidenceSection = Literal[
    "skills",
    "soft_skills",
    "summary",
    "experience",
    "projects",
    "achievements",
    "education",
    "courses",
]


class ResumeEvidence(BaseModel):
    section: EvidenceSection
    text: str


class RequirementComparison(BaseModel):
    category: MarketCategory
    name: str
    market_frequency: int = Field(ge=1, le=6)
    market_coverage: float = Field(ge=0, le=1)
    market_importance: MarketImportance
    resume_status: ResumeStatus
    resume_evidence: list[ResumeEvidence]
    market_sample_size: int | None = Field(default=None, ge=3, le=6)
    market_confidence: Literal["low", "medium", "high"] | None = None
    market_sources: list[MarketSource] = Field(default_factory=list)
    market_source_values: list[str] = Field(default_factory=list)
    market_evidence: list[MarketRequirementEvidence] = Field(default_factory=list)


class CandidateSkillObservation(BaseModel):
    name: str
    resume_status: Literal["evidenced", "mentioned_only"]
    evidence: list[ResumeEvidence]
    note: str


class ComparisonMetrics(BaseModel):
    common_requirements_total: int = Field(ge=0)
    common_requirements_matched: int = Field(ge=0)
    common_requirements_evidenced: int = Field(ge=0)
    common_requirements_mentioned_only: int = Field(ge=0)
    additional_requirements_total: int = Field(ge=0)
    additional_requirements_matched: int = Field(ge=0)
    additional_requirements_evidenced: int = Field(ge=0)
    candidate_only_skills_total: int = Field(ge=0)
    responsibility_requirements_total: int = Field(ge=0)
    responsibility_requirements_matched: int = Field(ge=0)
    unconfirmed_resume_skills_total: int = Field(default=0, ge=0)


class CandidateMarketComparison(BaseModel):
    status: Literal["ok"] = "ok"
    target: MarketTarget
    market_confidence: Literal["low", "medium", "high"]
    market_sample_size: int = Field(ge=3, le=6)
    professional_skills: list[RequirementComparison]
    tools_and_technologies: list[RequirementComparison]
    responsibilities: list[RequirementComparison]
    education_requirements: list[RequirementComparison]
    domain_knowledge: list[RequirementComparison]
    soft_skills: list[RequirementComparison]
    strong_matches: list[RequirementComparison]
    mentioned_only_requirements: list[RequirementComparison]
    missing_common_requirements: list[RequirementComparison]
    missing_additional_requirements: list[RequirementComparison]
    candidate_only_skills: list[CandidateSkillObservation]
    unconfirmed_resume_skills: list[CandidateSkillObservation]
    responsibility_matches: list[RequirementComparison]
    responsibility_gaps: list[RequirementComparison]
    education_match: list[RequirementComparison]
    domain_knowledge_match: list[RequirementComparison]
    market_variations: list[RequirementComparison]
    summary_metrics: ComparisonMetrics
    market_reliability: MarketReliability | None = None


class MarketDataUnavailableComparison(BaseModel):
    status: Literal["market_data_unavailable"] = "market_data_unavailable"
    target: MarketTarget
    market_confidence: Literal["low"] = "low"
    market_sample_size: int = Field(ge=0, le=6)
    message: str
    market_reliability: MarketReliability | None = None


@dataclass(frozen=True)
class _ResumeEntry:
    section: EvidenceSection
    text: str


class _Matcher:
    """Один batch embeddings на сравнение с безопасным lexical fallback."""

    def __init__(
        self,
        texts: list[str],
        *,
        use_embeddings: bool,
        embedding_model: str,
    ) -> None:
        self.vectors: dict[str, list[float]] = {}
        if use_embeddings:
            try:
                self.vectors = _embed_text_map(texts, embedding_model)
            except Exception:
                # Сравнение остаётся доступным без Ollama; тяжёлая LLM здесь
                # не используется, а lexical matching сохраняет прозрачность.
                self.vectors = {}

    def matches(
        self,
        requirement: str,
        candidate_text: str,
        *,
        allow_semantic: bool = True,
        responsibility_semantics: bool = False,
    ) -> bool:
        if _lexical_match(requirement, candidate_text):
            return True
        if not allow_semantic:
            return False
        cross_language_responsibility = False
        if responsibility_semantics:
            cross_language_responsibility = _language_mismatch(
                requirement,
                candidate_text,
            )
            if cross_language_responsibility:
                if not _responsibility_actions_compatible(
                    requirement,
                    candidate_text,
                ):
                    return False
            elif not _responsibility_semantic_anchor(
                requirement,
                candidate_text,
            ):
                return False
        if requirement in self.vectors and candidate_text in self.vectors:
            similarity = embedding_cosine_similarity(
                self.vectors[requirement],
                self.vectors[candidate_text],
            )
            if cross_language_responsibility:
                # bge-m3 cross-language cosine is systematically lower than
                # same-language cosine for short duties.  The action component
                # is safe here because incompatible/absent actions were already
                # rejected above; the global semantic threshold stays intact.
                similarity = (
                    CROSS_LANGUAGE_RESPONSIBILITY_SEMANTIC_WEIGHT * similarity
                    + (1 - CROSS_LANGUAGE_RESPONSIBILITY_SEMANTIC_WEIGHT)
                )
            return similarity >= SEMANTIC_MATCH_THRESHOLD
        return False


def _normalized_text(value: str) -> str:
    return " ".join(
        token.rstrip(".") for token in re.findall(r"[a-zа-яё0-9+#.&]+", value.casefold().replace("ё", "е"))
        if token.rstrip(".")
    )


def _tokens(value: str) -> set[str]:
    return set(_normalized_text(value).split())


def _responsibility_stems(value: str) -> list[str]:
    """Грубые лексические якоря действия и объекта для semantic matching."""
    result: list[str] = []
    for token in _normalized_text(value).split():
        if len(token) < 3 or token in _RESPONSIBILITY_PREFIX_WORDS:
            continue
        result.append(token[:5] if len(token) > 5 else token)
    return result


def _dominant_script(value: str) -> Literal["latin", "cyrillic"] | None:
    latin = len(re.findall(r"[a-z]", value.casefold()))
    cyrillic = len(re.findall(r"[а-яё]", value.casefold()))
    if latin == cyrillic:
        return None
    return "latin" if latin > cyrillic else "cyrillic"


def _language_mismatch(requirement: str, candidate_text: str) -> bool:
    """Определяет cross-language пару по доминирующему алфавиту текста."""
    requirement_script = _dominant_script(requirement)
    candidate_script = _dominant_script(candidate_text)
    return (
        requirement_script is not None
        and candidate_script is not None
        and requirement_script != candidate_script
    )


def _responsibility_action_classes(value: str) -> set[str]:
    """Возвращает класс первого явного действия обязанности."""
    tokens = [
        token
        for token in _normalized_text(value).split()
        if token not in _RESPONSIBILITY_PREFIX_WORDS
    ]
    if not tokens:
        return set()
    first_token = tokens[0]
    return {
        action
        for action, patterns in _RESPONSIBILITY_ACTION_PATTERNS.items()
        if any(
            re.fullmatch(rf"(?:{pattern})[a-zа-яё]*", first_token)
            for pattern in patterns
        )
    }


def _responsibility_actions_compatible(
    requirement: str,
    candidate_text: str,
) -> bool:
    """Разрешает multilingual cosine только при том же явном действии."""
    requirement_actions = _responsibility_action_classes(requirement)
    candidate_actions = _responsibility_action_classes(candidate_text)
    return bool(requirement_actions and requirement_actions & candidate_actions)


def _responsibility_semantic_anchor(
    requirement: str,
    candidate_text: str,
) -> bool:
    """Не позволяет одному cosine подменить совпадение обязанности.

    Для semantic match должен совпасть первый содержательный якорь требования
    (обычно действие) и, для составной обязанности, ещё один якорь объекта.
    """
    requirement_stems = _responsibility_stems(requirement)
    candidate_stems = set(_responsibility_stems(candidate_text))
    if not requirement_stems or not candidate_stems:
        return False
    if requirement_stems[0] not in candidate_stems:
        return False
    required_overlap = 1 if len(set(requirement_stems)) == 1 else 2
    return len(set(requirement_stems) & candidate_stems) >= required_overlap


def _lexical_match(requirement: str, candidate_text: str) -> bool:
    from .requirement_equivalence import requirement_key, requirement_forms, _morphological_key

    if requirement_key(requirement) and requirement_key(requirement) == requirement_key(candidate_text):
        return True
    candidate_words = _morphological_key(candidate_text).split()
    for form in requirement_forms(requirement):
        words = _morphological_key(form).split()
        if words and any(candidate_words[index:index + len(words)] == words
                         for index in range(len(candidate_words) - len(words) + 1)):
            return True
    requirement_key = normalize_skill_name(requirement) or _normalized_text(requirement)
    candidate_key = normalize_skill_name(candidate_text) or _normalized_text(candidate_text)
    if not requirement_key or not candidate_key:
        return False
    if requirement_key == candidate_key:
        return True
    requirement_text = _normalized_text(requirement)
    candidate_normalized = _normalized_text(candidate_text)
    if requirement_text and (
        f" {requirement_text} " in f" {candidate_normalized} "
    ):
        return True
    requirement_tokens = _tokens(requirement)
    candidate_tokens = _tokens(candidate_text)
    if (
        len(requirement_tokens) >= 2
        and len(requirement_tokens & candidate_tokens) / len(requirement_tokens) >= 0.8
    ):
        return True
    return (
        len(requirement_text) >= 5
        and SequenceMatcher(None, requirement_text, candidate_normalized).ratio()
        >= 0.88
    )


def _unique_entries(entries: list[_ResumeEntry]) -> list[_ResumeEntry]:
    result: list[_ResumeEntry] = []
    seen: set[tuple[EvidenceSection, str]] = set()
    for entry in entries:
        text = " ".join(entry.text.split())
        key = (entry.section, _normalized_text(text))
        if not key[1] or key in seen:
            continue
        seen.add(key)
        result.append(_ResumeEntry(entry.section, text))
    return result


def _resume_pools(resume: ResumeData) -> dict[str, list[_ResumeEntry]]:
    skill_mentions = [
        _ResumeEntry("skills", value) for value in resume.skills
    ]
    soft_mentions = [
        _ResumeEntry("soft_skills", value) for value in resume.soft_skills
    ]
    summaries = (
        [_ResumeEntry("summary", resume.professional_summary)]
        if resume.professional_summary
        else []
    )
    experience: list[_ResumeEntry] = []
    project_entries: list[_ResumeEntry] = []
    for item in resume.work_experience:
        prefix = " — ".join(
            value for value in (item.position, item.company) if value
        )
        if prefix:
            experience.append(_ResumeEntry("experience", prefix))
        experience.extend(
            _ResumeEntry("experience", value)
            for value in (
                *item.responsibilities,
                *item.achievements,
            )
        )
        # Перечень инструментов, даже внутри опыта, не описывает применение.
        skill_mentions.extend(_ResumeEntry("experience", value) for value in item.technologies)
    for item in resume.projects:
        project_entries.extend(
            _ResumeEntry("projects", value)
            for value in (
                item.name,
                item.description,
                item.role,
                *item.results,
            )
            if value
        )
        skill_mentions.extend(_ResumeEntry("projects", value) for value in item.technologies)
    achievements = [
        _ResumeEntry("achievements", value) for value in resume.achievements
    ]
    education = [
        _ResumeEntry(
            "education",
            " ".join(
                str(value)
                for value in (
                    item.education_level,
                    item.education_field,
                    item.field_of_study,
                    item.institution,
                    item.graduation_year,
                )
                if value is not None
            ),
        )
        for item in resume.education
    ]
    courses = [
        _ResumeEntry("courses", value)
        for value in resume.courses_and_certifications
    ]
    practical = _unique_entries([*experience, *project_entries, *achievements])
    return {
        "skill_mentions": _unique_entries(skill_mentions),
        "soft_mentions": _unique_entries(soft_mentions),
        "summary": _unique_entries(summaries),
        "practical": practical,
        "responsibilities": practical,
        "education": _unique_entries([*education, *courses]),
        "domain": _unique_entries([*summaries, *experience, *project_entries]),
    }


def _matching_entries(
    requirement: str,
    entries: list[_ResumeEntry],
    matcher: _Matcher,
    *,
    allow_semantic: bool = True,
    responsibility_semantics: bool = False,
) -> list[ResumeEvidence]:
    return [
        ResumeEvidence(section=entry.section, text=entry.text)
        for entry in entries
        if matcher.matches(
            requirement,
            entry.text,
            allow_semantic=allow_semantic,
            responsibility_semantics=responsibility_semantics,
        )
    ][:8]


def _compare_requirement(
    requirement: MarketRequirement | MarketVariation,
    category: MarketCategory,
    pools: dict[str, list[_ResumeEntry]],
    matcher: _Matcher,
) -> RequirementComparison:
    # Для именованных навыков/инструментов semantic similarity сама по себе не
    # доказывает тождество: Docker не должен подтверждаться Kubernetes. Для
    # обязанностей embedding разрешён только вместе с якорями действия/объекта.
    entity_category = category in {
        "professional_skills",
        "tools_and_technologies",
        "soft_skills",
    }
    responsibility_category = category == "responsibilities"
    practical_pool = (
        pools["responsibilities"]
        if responsibility_category
        else pools["education"]
        if category == "education_requirements"
        else pools["domain"]
        if category == "domain_knowledge"
        else pools["practical"]
    )
    def match_variants(entries: list[_ResumeEntry]) -> list[ResumeEvidence]:
        evidence: list[ResumeEvidence] = []
        for variant in dict.fromkeys([requirement.name, *requirement.source_values]):
            for row in _matching_entries(
                variant, entries, matcher, allow_semantic=not entity_category,
                responsibility_semantics=responsibility_category,
            ):
                if row not in evidence:
                    evidence.append(row)
        return evidence[:8]

    practical_evidence = match_variants(practical_pool)
    if practical_evidence:
        status: ResumeStatus = (
            "matched" if category == "education_requirements" else "evidenced"
        )
        evidence = practical_evidence
    else:
        if category in {"responsibilities", "education_requirements"}:
            # Summary — самопрезентация, а не evidence рабочей обязанности или
            # формального образования. Для этих категорий fallback запрещён.
            mention_pool: list[_ResumeEntry] = []
        elif category == "soft_skills":
            mention_pool = pools["soft_mentions"]
        elif category in {"professional_skills", "tools_and_technologies"}:
            mention_pool = pools["skill_mentions"]
        else:
            mention_pool = pools["summary"]
        mention_evidence = match_variants(mention_pool)
        if mention_evidence:
            status = (
                "mentioned_only"
                if category in {
                    "professional_skills",
                    "tools_and_technologies",
                    "soft_skills",
                }
                else "matched"
            )
            evidence = mention_evidence
        else:
            status = "not_found"
            evidence = []
    return RequirementComparison(
        category=category,
        name=requirement.name,
        market_frequency=requirement.frequency,
        market_coverage=requirement.coverage,
        market_importance=requirement.importance,
        resume_status=status,
        resume_evidence=evidence,
        market_sources=requirement.sources,
        market_source_values=requirement.source_values,
        market_evidence=requirement.evidence,
    )


def _all_market_requirements(
    market_profile: MarketProfile,
) -> list[tuple[MarketCategory, MarketRequirement]]:
    return [
        (category, requirement)
        for category in (
            "professional_skills",
            "tools_and_technologies",
            "responsibilities",
            "education_requirements",
            "domain_knowledge",
            "soft_skills",
        )
        for requirement in getattr(market_profile, category)
    ]


def _candidate_skill_observations(
    resume: ResumeData,
    market_profile: MarketProfile,
    pools: dict[str, list[_ResumeEntry]],
    matcher: _Matcher,
) -> tuple[list[CandidateSkillObservation], list[CandidateSkillObservation]]:
    market_skills = [
        variant
        for item in (
            *market_profile.professional_skills,
            *market_profile.tools_and_technologies,
        )
        for variant in dict.fromkeys([item.name, *item.source_values])
    ]
    candidate_only: list[CandidateSkillObservation] = []
    unconfirmed: list[CandidateSkillObservation] = []
    seen: set[str] = set()
    for skill in [*resume.skills,
                  *(value for item in resume.work_experience for value in item.technologies),
                  *(value for item in resume.projects for value in item.technologies)]:
        key = requirement_key(skill)
        if not key or key in seen:
            continue
        seen.add(key)
        evidence = _matching_entries(
            skill,
            pools["practical"],
            matcher,
            allow_semantic=False,
        )
        status: Literal["evidenced", "mentioned_only"] = (
            "evidenced" if evidence else "mentioned_only"
        )
        observation = CandidateSkillObservation(
            name=skill,
            resume_status=status,
            evidence=evidence or _matching_entries(skill, pools["skill_mentions"], matcher, allow_semantic=False),
            note=(
                "Навык присутствует в резюме, но не оказался распространённым "
                "в данной рыночной выборке."
            ),
        )
        if not any(
            matcher.matches(market_skill, skill, allow_semantic=False)
            for market_skill in market_skills
        ):
            candidate_only.append(observation)
        if status == "mentioned_only":
            unconfirmed.append(observation)
    return candidate_only, unconfirmed


def _coerce_resume(value: ResumeData | dict[str, Any]) -> ResumeData:
    return value if isinstance(value, ResumeData) else ResumeData.model_validate(value)


def _coerce_market_profile(
    value: MarketProfile | InsufficientMarketProfile | dict[str, Any],
) -> MarketProfile | InsufficientMarketProfile:
    if isinstance(value, (MarketProfile, InsufficientMarketProfile)):
        return value
    if value.get("status") == "ok":
        return MarketProfile.model_validate(value)
    return InsufficientMarketProfile.model_validate(value)


def _save_result(
    result: CandidateMarketComparison | MarketDataUnavailableComparison,
    output_path: Path | None,
) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output_path, result.model_dump_json(indent=2))


def compare_candidate_to_market(
    resume: ResumeData | dict[str, Any],
    market_profile: MarketProfile | InsufficientMarketProfile | dict[str, Any],
    output_path: Path | None = DEFAULT_OUTPUT,
    *,
    use_embeddings: bool = True,
    embedding_model: str = EMBEDDING_MODEL,
) -> CandidateMarketComparison | MarketDataUnavailableComparison:
    """Сопоставляет резюме с неизменяемым готовым MarketProfile."""
    resume_data = _coerce_resume(resume)
    profile = _coerce_market_profile(market_profile)
    if profile.status != "ok":
        result = MarketDataUnavailableComparison(
            target=profile.target,
            market_sample_size=profile.sample_size,
            market_reliability=profile.reliability,
            message=(
                "Сравнение не выполнено: доступных рыночных данных недостаточно."
            ),
        )
        _save_result(result, output_path)
        return result

    pools = _resume_pools(resume_data)
    requirements = _all_market_requirements(profile)
    all_texts = [
        variant for _, requirement in requirements
        for variant in dict.fromkeys([requirement.name, *requirement.source_values])
    ] + [
        variant for variation in profile.market_variations
        for variant in dict.fromkeys([variation.name, *variation.source_values])
    ] + [
        entry.text for entries in pools.values() for entry in entries
    ]
    matcher = _Matcher(
        list(dict.fromkeys(all_texts)),
        use_embeddings=use_embeddings,
        embedding_model=embedding_model,
    )
    comparisons_by_category: dict[MarketCategory, list[RequirementComparison]] = {
        category: []
        for category in (
            "professional_skills",
            "tools_and_technologies",
            "responsibilities",
            "education_requirements",
            "domain_knowledge",
            "soft_skills",
        )
    }
    for category, requirement in requirements:
        comparisons_by_category[category].append(
            _compare_requirement(requirement, category, pools, matcher).model_copy(update={
                "market_sample_size": profile.sample_size, "market_confidence": profile.confidence,
            })
        )
    variations = [
        _compare_requirement(item, item.category, pools, matcher).model_copy(update={
            "market_sample_size": profile.sample_size, "market_confidence": profile.confidence,
        })
        for item in profile.market_variations
    ]
    all_comparisons = [
        item
        for category_items in comparisons_by_category.values()
        for item in category_items
    ]
    # Обязанности имеют собственные matches/gaps и метрики; не дублируем их
    # в списках навыков/требований верхнего уровня.
    requirement_comparisons = [
        item for item in all_comparisons if item.category != "responsibilities"
    ]
    found_statuses = {"matched", "evidenced", "mentioned_only"}
    strong_matches = [
        item
        for item in requirement_comparisons
        if item.resume_status in {"matched", "evidenced"}
    ]
    mentioned_only = [
        item
        for item in requirement_comparisons
        if item.resume_status == "mentioned_only"
    ]
    missing_common = [
        item
        for item in requirement_comparisons
        if item.market_importance == "common" and item.resume_status == "not_found"
    ]
    missing_additional = [
        item
        for item in requirement_comparisons
        if item.market_importance == "additional"
        and item.resume_status == "not_found"
    ]
    candidate_only, unconfirmed = _candidate_skill_observations(
        resume_data,
        profile,
        pools,
        matcher,
    )
    common = [
        item
        for item in requirement_comparisons
        if item.market_importance == "common"
    ]
    additional = [
        item
        for item in requirement_comparisons
        if item.market_importance == "additional"
    ]
    responsibilities = comparisons_by_category["responsibilities"]
    result = CandidateMarketComparison(
        target=profile.target,
        market_confidence=profile.confidence,
        market_sample_size=profile.sample_size,
        market_reliability=profile.reliability,
        professional_skills=comparisons_by_category["professional_skills"],
        tools_and_technologies=comparisons_by_category["tools_and_technologies"],
        responsibilities=responsibilities,
        education_requirements=comparisons_by_category["education_requirements"],
        domain_knowledge=comparisons_by_category["domain_knowledge"],
        soft_skills=comparisons_by_category["soft_skills"],
        strong_matches=strong_matches,
        mentioned_only_requirements=mentioned_only,
        missing_common_requirements=missing_common,
        missing_additional_requirements=missing_additional,
        candidate_only_skills=candidate_only,
        unconfirmed_resume_skills=unconfirmed,
        responsibility_matches=[
            item for item in responsibilities if item.resume_status in found_statuses
        ],
        responsibility_gaps=[
            item for item in responsibilities if item.resume_status == "not_found"
        ],
        education_match=[
            item
            for item in comparisons_by_category["education_requirements"]
            if item.resume_status in found_statuses
        ],
        domain_knowledge_match=[
            item
            for item in comparisons_by_category["domain_knowledge"]
            if item.resume_status in found_statuses
        ],
        market_variations=variations,
        summary_metrics=ComparisonMetrics(
            common_requirements_total=len(common),
            common_requirements_matched=sum(
                item.resume_status in found_statuses for item in common
            ),
            common_requirements_evidenced=sum(
                item.resume_status == "evidenced" for item in common
            ),
            common_requirements_mentioned_only=sum(
                item.resume_status == "mentioned_only" for item in common
            ),
            additional_requirements_total=len(additional),
            additional_requirements_matched=sum(
                item.resume_status in found_statuses for item in additional
            ),
            additional_requirements_evidenced=sum(
                item.resume_status == "evidenced" for item in additional
            ),
            candidate_only_skills_total=len(candidate_only),
            unconfirmed_resume_skills_total=len(unconfirmed),
            responsibility_requirements_total=len(responsibilities),
            responsibility_requirements_matched=sum(
                item.resume_status in found_statuses for item in responsibilities
            ),
        ),
    )
    _save_result(result, output_path)
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сравнение resume.json с готовым market_profile.json",
    )
    parser.add_argument("resume", type=Path)
    parser.add_argument("market_profile", type=Path)
    parser.add_argument("output", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Использовать только прозрачное lexical matching",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    resume = ResumeData.model_validate_json(args.resume.read_text(encoding="utf-8"))
    market_payload = json.loads(args.market_profile.read_text(encoding="utf-8"))
    result = compare_candidate_to_market(
        resume,
        market_payload,
        args.output,
        use_embeddings=not args.no_embeddings,
    )
    print(f"Сохранено: {args.output}")
    print(f"Статус: {result.status}")


if __name__ == "__main__":
    main()
