"""Персональные рекомендации из уже рассчитанных фактов о резюме и рынке.

Модуль не выполняет retrieval, не строит MarketProfile и не пересчитывает
CandidateMarketComparison. LLM может только переформулировать заранее
разрешённые рекомендации; состав пунктов и их basis определяет Python.
"""
from .runtime import atomic_write_text


import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Literal

from .model_runtime import chat
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .candidate_market_comparison import (
    CandidateMarketComparison,
    CandidateSkillObservation,
    MarketDataUnavailableComparison,
    RequirementComparison,
)
from .resume_extractor import ResumeData
from .resume_quality_evaluator import ResumeQualityResult
from .requirement_equivalence import requirement_key
from .market_reliability import independent_employer_count
from .market_profile import MarketReliability, MarketRequirementEvidence, MarketSource


from .settings import get_settings

MODEL_NAME = get_settings().model
MODEL_CONTEXT_SIZE = get_settings().context_size
MAX_OUTPUT_TOKENS = 6144
DEFAULT_OUTPUT = Path("candidate_recommendations.json")

RecommendationSection = Literal[
    "strengths",
    "resume_improvements",
    "market_gaps",
    "unconfirmed_skills",
    "development_directions",
]
BasisType = Literal[
    "resume_strength",
    "resume_improvement",
    "market_strength",
    "missing_common",
    "missing_additional",
    "mentioned_only",
    "development_direction",
    "resume_unconfirmed",
]


class RecommendationTarget(BaseModel):
    specialization: str | None
    seniority: str | None


class RecommendationMarketContext(BaseModel):
    available: bool
    sample_size: int = Field(ge=0, le=6)
    confidence: Literal["low", "medium", "high"]
    reliability: MarketReliability | None = None


class RecommendationBasis(BaseModel):
    type: BasisType
    reference: str
    market_importance: Literal[
        "common",
        "additional",
        "market_variation",
    ] | None = None
    evidence: list[str] = Field(default_factory=list, max_length=3)
    category: str | None = None
    resume_status: str | None = None
    market_frequency: int | None = Field(default=None, ge=1, le=6)
    market_sample_size: int | None = Field(default=None, ge=3, le=6)
    market_coverage: float | None = Field(default=None, ge=0, le=1)
    market_confidence: Literal["low", "medium", "high"] | None = None
    independent_employers: int | None = Field(default=None, ge=0, le=6)
    market_sources: list[MarketSource] = Field(default_factory=list)
    market_source_values: list[str] = Field(default_factory=list)
    market_evidence: list[MarketRequirementEvidence] = Field(default_factory=list)


class RecommendationAction(BaseModel):
    type: Literal["highlight_evidence", "add_application_example", "clarify_or_develop"]
    steps: list[str] = Field(min_length=1, max_length=3)
    expected_result: str


class RecommendationItem(BaseModel):
    fact_id: str
    title: str
    text: str
    basis: list[RecommendationBasis] = Field(min_length=1)
    action: RecommendationAction | None = None
    confidence: Literal["low", "medium", "high"] | None = None


class RecommendationResult(BaseModel):
    status: Literal["ok"] = "ok"
    candidate_name: str | None
    target: RecommendationTarget
    resume_score: float = Field(ge=0, le=100)
    market_context: RecommendationMarketContext
    strengths: list[RecommendationItem] = Field(max_length=5)
    resume_improvements: list[RecommendationItem] = Field(max_length=5)
    market_gaps: list[RecommendationItem] = Field(max_length=5)
    unconfirmed_skills: list[RecommendationItem] = Field(max_length=5)
    development_directions: list[RecommendationItem] = Field(max_length=3)
    final_summary: str
    generation_mode: Literal["llm", "fallback"]
    llm_calls: int = Field(ge=0, le=2)


class GeneratedRecommendationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fact_id: str
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)


class GeneratedRecommendationWording(BaseModel):
    """LLM меняет только wording элементов с уже выданными fact_id."""

    model_config = ConfigDict(extra="forbid")

    strengths: list[GeneratedRecommendationItem]
    resume_improvements: list[GeneratedRecommendationItem]
    market_gaps: list[GeneratedRecommendationItem]
    unconfirmed_skills: list[GeneratedRecommendationItem]
    development_directions: list[GeneratedRecommendationItem]


SYSTEM_PROMPT = """
Ты редактируешь формулировки уже рассчитанных рекомендаций кандидату.

Строгие правила:
1. Не добавляй и не удаляй пункты, не меняй fact_id и раздел пункта.
2. Используй только факты из basis и safe_draft соответствующего пункта.
3. Не придумывай навыки, опыт, достижения, образование, проекты и числа.
4. Отсутствие в резюме не означает отсутствие навыка у человека.
5. Не советуй указывать неподтверждённые сведения. Для отсутствующего навыка:
   если реальный опыт есть — предложи показать пример; если нет — обозначь
   возможное направление развития.
6. Common означает частое появление только в рассмотренной выборке;
   additional — появление в части выборки. Не обобщай вывод на весь рынок.
7. Учитывай confidence: low — предварительный совет по ограниченным данным;
   medium — осторожный вывод по выборке. Не повышай уверенность.
8. Market variation не превращай в рекомендацию.
9. Пиши понятно, уважительно и по-русски. Верни только JSON заданной схемы.
"""


_SECTION_LABELS = {
    "experience": "Опыт работы",
    "projects": "Проекты",
    "header": "Шапка резюме",
    "summary": "О себе",
    "skills": "Навыки",
    "education": "Образование",
}

# Служебная лексика допустима при переформулировке. Содержательные сущности,
# технологии, числа и фактические действия должны приходить из safe_draft/basis.
_SAFE_STYLE_WORDS = {
    "а", "без", "более", "бы", "был", "быть", "в", "вам", "ваш",
    "ваша", "ваше", "вашей", "вашем", "ваши", "ведь", "возможно",
    "вопрос", "встречается", "вы", "где", "для", "добавить", "если",
    "есть", "ещё", "же", "и", "из", "или", "именно", "информации",
    "использование", "использовать", "их", "как", "кандидата", "когда",
    "конкретный", "контексте", "который", "ли", "может", "можно", "на",
    "навык", "навыка", "навыки", "не", "него", "нескольких", "но", "о",
    "обнаружено", "описание", "опыт", "основных", "отдельно", "отмечено",
    "показан", "показать", "полезно", "пример", "применение", "при",
    "профиля", "развития", "раздел", "результат", "результата", "резюме",
    "рынка", "с", "сведения", "свой", "сильная", "случае", "совет",
    "стоит", "стороной", "так", "также", "требование", "требования",
    "у", "указан", "улучшить", "факт", "части", "часто", "что", "это",
    "явно", "является", "практический", "практического", "подтвержден",
    "подтверждено", "подтверждает", "рассмотренной", "выборке", "вакансий",
}


def _coerce_resume(value: ResumeData | dict[str, Any]) -> ResumeData:
    return value if isinstance(value, ResumeData) else ResumeData.model_validate(value)


def _coerce_evaluation(
    value: ResumeQualityResult | dict[str, Any],
) -> ResumeQualityResult:
    return (
        value
        if isinstance(value, ResumeQualityResult)
        else ResumeQualityResult.model_validate(value)
    )


def _coerce_comparison(
    value: (
        CandidateMarketComparison
        | MarketDataUnavailableComparison
        | dict[str, Any]
    ),
) -> CandidateMarketComparison | MarketDataUnavailableComparison:
    if isinstance(value, (CandidateMarketComparison, MarketDataUnavailableComparison)):
        return value
    if value.get("status") == "ok":
        return CandidateMarketComparison.model_validate(value)
    return MarketDataUnavailableComparison.model_validate(value)


def _evidence_text(requirement: RequirementComparison) -> list[str]:
    evidence = sorted(requirement.resume_evidence,
                      key=lambda row: (row.text.casefold() == requirement.name.casefold(), -len(row.text)))
    return [item.text for item in evidence[:3]]


def _importance_order(requirement: RequirementComparison) -> tuple[int, float, str]:
    return (
        0 if requirement.market_importance == "common" else 1,
        -requirement.market_coverage,
        requirement.name.casefold(),
    )


def _market_signal(requirement: RequirementComparison) -> str:
    if requirement.market_sample_size:
        return f"встречается в {requirement.market_frequency} из {requirement.market_sample_size} рассмотренных вакансий"
    if requirement.market_importance == "common":
        return "часто встречается в рассмотренных вакансиях"
    return "встречается в части рассмотренных вакансий"


def _market_basis(requirement: RequirementComparison, kind: BasisType) -> RecommendationBasis:
    return RecommendationBasis(
        type=kind, reference=requirement.name, category=requirement.category,
        resume_status=requirement.resume_status, market_importance=requirement.market_importance,
        evidence=_evidence_text(requirement), market_frequency=requirement.market_frequency,
        market_sample_size=requirement.market_sample_size, market_coverage=requirement.market_coverage,
        market_confidence=requirement.market_confidence,
        independent_employers=independent_employer_count(requirement.market_sources)[0],
        market_sources=requirement.market_sources, market_source_values=requirement.market_source_values,
        market_evidence=requirement.market_evidence,
    )


def _advice_confidence(requirement: RequirementComparison) -> Literal["low", "medium", "high"]:
    confidence = requirement.market_confidence or "low"
    employers, known = independent_employer_count(requirement.market_sources)
    if confidence == "low" or employers < 2 or known < len(requirement.market_sources):
        return "low"
    if requirement.market_importance != "common":
        return "medium"
    return confidence


def _requirement_action(requirement: RequirementComparison, mode: str) -> RecommendationAction | None:
    name = requirement.name
    if requirement.category == "soft_skills":
        return None
    if mode == "highlight":
        if requirement.category == "education_requirements":
            steps = ["Сделайте соответствующее образование заметным в разделе «Образование».",
                     "Укажите только подтверждённые квалификацию и направление подготовки."]
        else:
            steps = [f"Поднимите реальный пример применения «{name}» ближе к началу соответствующего опыта или проекта.",
                     "Опишите задачу и ваши собственные действия.",
                     "Укажите подтверждённый результат; используйте числа только при наличии данных."]
        return RecommendationAction(type="highlight_evidence", steps=steps,
                                    expected_result="Заметное подтверждение требования в соответствующем разделе резюме.")
    if mode == "mention":
        return RecommendationAction(type="add_application_example", steps=[
            f"Если вы применяли «{name}», выберите одну реальную задачу из работы или проекта.",
            "Опишите контекст задачи и способ применения навыка.",
            "Добавьте проверяемый результат; если применения не было, оставьте навык без выдуманного опыта.",
        ], expected_result="Пример применения и результата, связанный с навыком в резюме.")
    if requirement.category == "education_requirements":
        steps = [f"Проверьте, есть ли у вас образование, соответствующее «{name}».",
                 "Если есть, уточните подтверждённые сведения о нём в резюме.",
                 "Если нет, изучите условия конкретных вакансий; не подменяйте диплом курсом или учебным проектом."]
        outcome = "Уточнённые сведения об образовании или выбор вакансий с подходящими условиями."
    else:
        steps = [f"Проверьте, есть ли у вас реальный опыт по требованию «{name}», который не попал в резюме.",
                 "Если опыт есть, добавьте задачу, свои действия и подтверждённый результат.",
                 f"Если опыта нет и направление вам подходит, выполните учебную задачу по «{name}» с проверяемым результатом и обозначьте её как учебный проект."]
        outcome = "Уточнённый пример имеющегося опыта или результат учебной задачи, обозначенный без приписывания рабочего опыта."
    return RecommendationAction(type="clarify_or_develop", steps=steps, expected_result=outcome)


def _market_strength_item(
    requirement: RequirementComparison,
    index: int,
) -> RecommendationItem:
    if requirement.category == "soft_skills":
        return RecommendationItem(
            fact_id=f"market_strength:{index}:{requirement.name}", title=f"Указано в резюме: {requirement.name}",
            text=f"«{requirement.name}» отражён в резюме.", basis=[_market_basis(requirement, "market_strength")],
            confidence=_advice_confidence(requirement),
        )
    evidence = _evidence_text(requirement)
    if evidence:
        evidence_phrase = f"В резюме это подтверждает пример: «{evidence[0]}»."
    else:
        evidence_phrase = "Требование явно отражено в структурированном резюме."
    return RecommendationItem(
        fact_id=f"market_strength:{index}:{requirement.name}",
        title=f"В резюме подтверждено: {requirement.name}",
        text=(
            f"{requirement.name} {_market_signal(requirement)}. "
            f"{evidence_phrase}"
            " Сделайте это подтверждение заметнее в соответствующем разделе резюме."
        ),
        basis=[_market_basis(requirement, "market_strength")],
        action=_requirement_action(requirement, "highlight"), confidence=_advice_confidence(requirement),
    )


def _resume_strength_item(
    section_name: str,
    strength: str,
    index: int,
) -> RecommendationItem:
    label = _SECTION_LABELS.get(section_name, section_name)
    return RecommendationItem(
        fact_id=f"resume_strength:{section_name}:{index}",
        title=f"Сильная сторона раздела «{label}»",
        text=strength,
        basis=[
            RecommendationBasis(
                type="resume_strength",
                reference=section_name,
                evidence=[strength],
            )
        ],
    )


def _resume_improvement_item(
    section_name: str,
    weakness: str,
    recommendation: str | None,
    index: int,
) -> RecommendationItem:
    label = _SECTION_LABELS.get(section_name, section_name)
    cleaned_weakness = weakness.rstrip(" .")
    cleaned_recommendation = recommendation.rstrip(" .") if recommendation else None
    return RecommendationItem(
        fact_id=f"resume_improvement:{section_name}:{index}",
        title=f"Улучшить раздел «{label}»",
        text=(f"{cleaned_recommendation}." if cleaned_recommendation else f"Уточните раздел: {cleaned_weakness}."),
        basis=[
            RecommendationBasis(
                type="resume_improvement",
                reference=section_name,
                evidence=[weakness, *([recommendation] if recommendation else [])],
            )
        ],
    )


def _market_gap_item(
    requirement: RequirementComparison,
    index: int,
) -> RecommendationItem:
    basis_type: Literal["missing_common", "missing_additional"] = (
        "missing_common"
        if requirement.market_importance == "common"
        else "missing_additional"
    )
    text = (
        f"{requirement.name} {_market_signal(requirement)}, но упоминание "
        "этого требования в резюме не обнаружено. Если у вас есть "
        "практический опыт, покажите конкретную задачу или результат. "
        "Если опыта нет, это можно рассматривать как направление развития."
    )
    if requirement.category == "education_requirements":
        text = (f"{requirement.name} {_market_signal(requirement)}, но подтверждение "
                "этого образования в резюме не обнаружено. Если образование есть, "
                "уточните сведения о нём. Если нет, проверьте условия конкретных вакансий.")
    elif requirement.category == "soft_skills":
        text = f"Рекомендуется указать «{requirement.name}», если это действительно характеризует вашу работу."
    return RecommendationItem(
        fact_id=f"market_gap:{index}:{requirement.name}",
        title=(f"Рекомендуется указать: {requirement.name}" if requirement.category == "soft_skills"
               else f"В резюме не обнаружено: {requirement.name}"),
        text=text,
        basis=[_market_basis(requirement, basis_type)],
        action=_requirement_action(requirement, "missing"), confidence=_advice_confidence(requirement),
    )


def _unconfirmed_item(
    requirement: RequirementComparison,
    index: int,
) -> RecommendationItem:
    return RecommendationItem(
        fact_id=f"mentioned_only:{index}:{requirement.name}",
        title=(f"Рекомендуется указать: {requirement.name}" if requirement.category == "soft_skills"
               else f"Показать применение: {requirement.name}"),
        text=(f"Рекомендуется указать «{requirement.name}», если это действительно характеризует вашу работу."
              if requirement.category == "soft_skills" else (
            f"{requirement.name} {_market_signal(requirement)}. Навык указан "
            "в резюме, но его практическое применение не показано. Если вы "
            "действительно работали с ним, добавьте конкретный пример задачи "
            "или результата."
        )),
        basis=[_market_basis(requirement, "mentioned_only")],
        action=_requirement_action(requirement, "mention"), confidence=_advice_confidence(requirement),
    )


def _resume_unconfirmed_item(observation: CandidateSkillObservation, index: int) -> RecommendationItem:
    return RecommendationItem(
        fact_id=f"resume_unconfirmed:{index}:{observation.name}",
        title=f"Показать применение: {observation.name}",
        text=f"«{observation.name}» перечислен в резюме без примера применения. Если применяли его, добавьте реальную задачу и результат.",
        basis=[RecommendationBasis(type="resume_unconfirmed", reference=observation.name,
                                   resume_status="mentioned_only",
                                   evidence=[row.text for row in observation.evidence][:3])],
    )


def _development_item(
    requirement: RequirementComparison,
    index: int,
) -> RecommendationItem:
    text = (
        f"Если практического опыта с {requirement.name} пока нет, можно "
        "рассмотреть обучение или учебную задачу с проверяемым результатом. "
        "Если опыт уже есть, приоритетнее явно показать его в резюме."
    )
    if requirement.category == "education_requirements":
        text = (f"Уточните, соответствует ли имеющееся образование требованию «{requirement.name}». "
                "Если нет, проверьте условия конкретных вакансий и подходящие варианты получения образования.")
    return RecommendationItem(
        fact_id=f"development:{index}:{requirement.name}",
        title=f"Рассмотреть развитие: {requirement.name}",
        text=text,
        basis=[_market_basis(requirement, "development_direction")],
        action=_requirement_action(requirement, "missing"), confidence=_advice_confidence(requirement),
    )


def _build_safe_items(
    evaluation: ResumeQualityResult,
    comparison: CandidateMarketComparison | MarketDataUnavailableComparison,
) -> dict[RecommendationSection, list[RecommendationItem]]:
    sections: dict[RecommendationSection, list[RecommendationItem]] = {
        "strengths": [],
        "resume_improvements": [],
        "market_gaps": [],
        "unconfirmed_skills": [],
        "development_directions": [],
    }

    if comparison.status == "ok":
        def contextual(requirement: RequirementComparison) -> RequirementComparison:
            return requirement.model_copy(update={
                "market_sample_size": comparison.market_sample_size,
                "market_confidence": comparison.market_confidence,
            })

        market_strengths = sorted(
            [
                *comparison.strong_matches,
                *comparison.responsibility_matches,
            ],
            key=_importance_order,
        )
        seen_strengths: set[tuple[str, str]] = set()
        for requirement in market_strengths:
            key = (requirement.category, requirement.name.casefold())
            if key in seen_strengths or len(sections["strengths"]) >= 5:
                continue
            seen_strengths.add(key)
            sections["strengths"].append(
                _market_strength_item(contextual(requirement), len(sections["strengths"]))
            )

        gaps = sorted(
            [
                *comparison.missing_common_requirements,
                *comparison.missing_additional_requirements,
                *(item for item in comparison.responsibility_gaps if item.market_importance != "market_variation"),
            ],
            key=_importance_order,
        )[:5]
        sections["market_gaps"] = [
            _market_gap_item(contextual(requirement), index)
            for index, requirement in enumerate(gaps)
        ]
        mentioned = sorted(
            comparison.mentioned_only_requirements,
            key=_importance_order,
        )[:5]
        sections["unconfirmed_skills"] = [
            _unconfirmed_item(contextual(requirement), index)
            for index, requirement in enumerate(mentioned)
        ]
        seen_mentions = {requirement_key(item.name) for item in mentioned}
        for observation in comparison.unconfirmed_resume_skills:
            if len(sections["unconfirmed_skills"]) >= 5:
                break
            if requirement_key(observation.name) not in seen_mentions:
                sections["unconfirmed_skills"].append(_resume_unconfirmed_item(observation, len(sections["unconfirmed_skills"])))
                seen_mentions.add(requirement_key(observation.name))
        sections["development_directions"] = [
            _development_item(contextual(requirement), index)
            for index, requirement in enumerate([item for item in gaps if item.category != "soft_skills"][:3])
        ]

    ranked_sections = sorted(
        evaluation.sections.items(),
        key=lambda item: item[1].score / item[1].max_score,
    )
    for section_name, section in ranked_sections:
        for index, weakness in enumerate(section.weaknesses):
            if len(sections["resume_improvements"]) >= 5:
                break
            recommendation = (
                section.recommendations[index]
                if index < len(section.recommendations)
                else section.recommendations[0]
                if section.recommendations
                else None
            )
            if recommendation and recommendation.casefold().rstrip(" .") in {
                "никаких критических замечаний", "нет замечаний", "замечаний нет",
            }:
                continue
            sections["resume_improvements"].append(
                _resume_improvement_item(
                    section_name,
                    weakness,
                    recommendation,
                    index,
                )
            )
        if len(sections["resume_improvements"]) >= 5:
            break

    for section_name, section in reversed(ranked_sections):
        for index, strength in enumerate(section.strengths):
            if len(sections["strengths"]) >= 5:
                break
            sections["strengths"].append(
                _resume_strength_item(section_name, strength, index)
            )
        if len(sections["strengths"]) >= 5:
            break

    for items in sections.values():
        for item in items:
            if item.confidence == "low":
                item.text = "Предварительный совет по ограниченной выборке. " + item.text
    return sections


def _unconfirmed_names(comparison: CandidateMarketComparison) -> list[str]:
    names = [item.name for item in comparison.unconfirmed_resume_skills]
    names.extend(item.name for item in comparison.mentioned_only_requirements
                 if item.category in {"professional_skills", "tools_and_technologies"})
    return list({requirement_key(name): name for name in names}.values())


def _final_summary(
    evaluation: ResumeQualityResult,
    comparison: CandidateMarketComparison | MarketDataUnavailableComparison,
    items: dict[RecommendationSection, list[RecommendationItem]],
) -> str:
    quality = (
        f"Качество резюме оценено на {evaluation.total_score:g}/100: "
        f"{evaluation.quality_level}."
    )
    if comparison.status != "ok":
        return (
            f"{quality} Рыночные рекомендации не сформированы, потому что "
            "доступных рыночных данных недостаточно."
        )
    return (
        f"{quality} В выборке из {comparison.market_sample_size} вакансий "
        f"выделено подтверждённых сильных сторон: {len(items['strengths'])}; "
        f"требований, не обнаруженных в резюме: {len(items['market_gaps'])}; "
        f"навыков без практического подтверждения: "
        f"{len(_unconfirmed_names(comparison))} во всём резюме. Выводы относятся только к "
        "рассмотренной выборке вакансий."
        + (" Уверенность в рыночном ориентире низкая; советы предварительные."
           if comparison.market_confidence == "low" else "")
    )


def _llm_context(
    target: RecommendationTarget,
    evaluation: ResumeQualityResult,
    market_context: RecommendationMarketContext,
    items: dict[RecommendationSection, list[RecommendationItem]],
) -> dict[str, Any]:
    return {
        "target": target.model_dump(mode="json"),
        "resume_evaluation": {
            "total_score": evaluation.total_score,
            "quality_level": evaluation.quality_level,
        },
        "market_context": market_context.model_dump(mode="json"),
        "allowed_recommendations": {
            section: [item.model_dump(mode="json") for item in section_items]
            for section, section_items in items.items()
        },
    }


def _safety_tokens(value: str) -> set[str]:
    result: set[str] = set()
    for token in re.findall(r"[a-zа-яё0-9+#.-]+", value.casefold()):
        if len(token) <= 2 and not any(character.isdigit() for character in token):
            continue
        if re.fullmatch(r"[а-яё]+", token) and len(token) > 5:
            token = token[:5]
        result.add(token)
    return result


def _wording_is_safe(
    generated: GeneratedRecommendationItem,
    planned: RecommendationItem,
) -> bool:
    if any(basis.resume_status == "not_found" for basis in planned.basis):
        if "если" not in generated.text.casefold():
            return False
    for sentence in re.split(r"[.!?]\s+", generated.text.casefold()):
        if "если" not in sentence and re.search(
            r"\b(?:у вас|у кандидата)\s+нет\b|\b(?:вы|кандидат)\b.*\bне\s+(?:владе|зна|уме|име|работа|использова)\w*",
            sentence,
        ):
            return False
    source = " ".join(
        [
            planned.title,
            planned.text,
            *(
                basis.reference
                for basis in planned.basis
            ),
            *(
                evidence
                for basis in planned.basis
                for evidence in basis.evidence
            ),
        ]
    )
    allowed = _safety_tokens(source) | _safety_tokens(" ".join(_SAFE_STYLE_WORDS))
    generated_tokens = _safety_tokens(f"{generated.title} {generated.text}")
    if generated_tokens - allowed:
        return False
    return True


def _apply_generated_wording(
    planned: dict[RecommendationSection, list[RecommendationItem]],
    generated: GeneratedRecommendationWording,
) -> dict[RecommendationSection, list[RecommendationItem]]:
    result: dict[RecommendationSection, list[RecommendationItem]] = {}
    for section, planned_items in planned.items():
        generated_items = getattr(generated, section)
        expected_ids = [item.fact_id for item in planned_items]
        actual_ids = [item.fact_id for item in generated_items]
        if actual_ids != expected_ids:
            raise ValueError(
                f"LLM изменила состав или порядок fact_id в разделе {section}"
            )
        section_result: list[RecommendationItem] = []
        for planned_item, generated_item in zip(planned_items, generated_items):
            if any(basis.category == "soft_skills" for basis in planned_item.basis):
                section_result.append(planned_item)
                continue
            if not _wording_is_safe(generated_item, planned_item):
                raise ValueError(
                    f"Неподтверждённое содержание для {planned_item.fact_id}"
                )
            section_result.append(
                planned_item.model_copy(
                    update={
                        "title": generated_item.title,
                        "text": ("Предварительный совет по ограниченной выборке. " + generated_item.text
                                 if planned_item.confidence == "low" and not generated_item.text.startswith("Предварительный совет")
                                 else generated_item.text),
                    }
                )
            )
        result[section] = section_result
    return result


def _request_wording(
    context: dict[str, Any],
    repair_error: str | None = None,
) -> GeneratedRecommendationWording:
    instruction = (
        "Переформулируй safe_draft, сохранив все fact_id и разделы."
    )
    if repair_error:
        instruction += (
            " Предыдущий ответ отклонён проверкой: "
            f"{repair_error}. Исправь ответ, не добавляя новых фактов."
        )
    response = chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    instruction
                    + "\n\n"
                    + json.dumps(context, ensure_ascii=False, indent=2)
                ),
            },
        ],
        format=GeneratedRecommendationWording.model_json_schema(),
        think=False,
        options={
            "temperature": 0,
            "num_ctx": MODEL_CONTEXT_SIZE,
            "num_predict": MAX_OUTPUT_TOKENS,
        },
    )
    return GeneratedRecommendationWording.model_validate_json(
        response.message.content
    )


def _save_result(result: RecommendationResult, output_path: Path | None) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output_path, result.model_dump_json(indent=2))


def generate_candidate_recommendations(
    resume: ResumeData | dict[str, Any],
    resume_evaluation: ResumeQualityResult | dict[str, Any],
    market_comparison: (
        CandidateMarketComparison
        | MarketDataUnavailableComparison
        | dict[str, Any]
    ),
    output_path: Path | None = DEFAULT_OUTPUT,
    *,
    use_llm: bool = True,
) -> RecommendationResult:
    """Формирует рекомендации из готовых фактов, не пересчитывая их."""
    resume_data = _coerce_resume(resume)
    evaluation = _coerce_evaluation(resume_evaluation)
    comparison = _coerce_comparison(market_comparison)
    target = RecommendationTarget(
        specialization=comparison.target.specialization,
        seniority=comparison.target.seniority,
    )
    market_context = RecommendationMarketContext(
        available=comparison.status == "ok",
        sample_size=comparison.market_sample_size,
        confidence=comparison.market_confidence,
        reliability=comparison.market_reliability,
    )
    safe_items = _build_safe_items(evaluation, comparison)
    final_summary = _final_summary(evaluation, comparison, safe_items)
    selected_items = safe_items
    generation_mode: Literal["llm", "fallback"] = "fallback"
    llm_calls = 0

    if use_llm:
        context = _llm_context(target, evaluation, market_context, safe_items)
        repair_error: str | None = None
        for _attempt in range(2):
            llm_calls += 1
            try:
                generated = _request_wording(context, repair_error)
            except ValidationError as error:
                repair_error = f"ответ не соответствует JSON-схеме: {error}"
                continue
            except Exception:
                # Ошибка транспорта/недоступная модель: шаблонный результат уже
                # готов, повторный сетевой вызов не улучшит доступность.
                break
            try:
                selected_items = _apply_generated_wording(safe_items, generated)
            except (ValueError, TypeError) as error:
                repair_error = str(error)
                continue
            generation_mode = "llm"
            break

    result = RecommendationResult(
        candidate_name=resume_data.candidate_name,
        target=target,
        resume_score=evaluation.total_score,
        market_context=market_context,
        strengths=selected_items["strengths"],
        resume_improvements=selected_items["resume_improvements"],
        market_gaps=selected_items["market_gaps"],
        unconfirmed_skills=selected_items["unconfirmed_skills"],
        development_directions=selected_items["development_directions"],
        final_summary=final_summary,
        generation_mode=generation_mode,
        llm_calls=llm_calls,
    )
    _save_result(result, output_path)
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Рекомендации по готовым ResumeEvaluation и MarketComparison",
    )
    parser.add_argument("resume", type=Path)
    parser.add_argument("resume_evaluation", type=Path)
    parser.add_argument("market_comparison", type=Path)
    parser.add_argument("output", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Сформировать безопасные шаблонные рекомендации без LLM",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    started_at = time.perf_counter()
    result = generate_candidate_recommendations(
        resume=json.loads(args.resume.read_text(encoding="utf-8")),
        resume_evaluation=json.loads(
            args.resume_evaluation.read_text(encoding="utf-8")
        ),
        market_comparison=json.loads(
            args.market_comparison.read_text(encoding="utf-8")
        ),
        output_path=args.output,
        use_llm=not args.no_llm,
    )
    print(f"Сохранено: {args.output}")
    print(f"Режим: {result.generation_mode}; LLM-вызовов: {result.llm_calls}")
    print(f"Время: {time.perf_counter() - started_at:.2f} сек.")


if __name__ == "__main__":
    main()
