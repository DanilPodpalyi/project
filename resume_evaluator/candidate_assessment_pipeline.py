"""End-to-end сценарий оценки кандидата поверх стабильных модулей.

Оркестратор не меняет и не дублирует бизнес-логику этапов: каждый результат
рассчитывается один раз соответствующим модулем, после чего компактно
проецируется в единый JSON и человекочитаемый Markdown.
"""

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError
from .runtime import atomic_write_text, current_run_id, managed_run, update_stage

from .alternative_role_explorer import (
    AlternativeRoleExplorerResult,
    explore_alternative_roles,
)
from .candidate_market_comparison import (
    CandidateMarketComparison,
    MarketDataUnavailableComparison,
    RequirementComparison,
    compare_candidate_to_market,
)
from .recommendation_generator import (
    RecommendationAction,
    RecommendationBasis,
    RecommendationItem,
    RecommendationResult,
    generate_candidate_recommendations,
    _unconfirmed_names,
)
from .resume_extractor import ResumeData, extract_resume, read_resume
from .resume_quality_evaluator import (
    ResumeQualityResult,
    evaluate_resume,
)
from .resume_quality_evaluator import (
    build_result as build_resume_quality_result,
)
from .market_profile_builder_v2 import build_market_profile_v2
from .russian_market import (
    build_russian_market_profile, target_from_resume, validate_saved_russian_profile,
)
from .market_profile import (
    AdjacentMarketContext,
    InsufficientMarketProfile,
    MarketProfile,
    MarketReliability,
    MarketRequirementEvidence,
    MarketSource,
    MarketTarget,
    build_market_profile,
)


class ReportResumeSection(BaseModel):
    score: float
    max_score: float
    strengths: list[str]
    weaknesses: list[str]


class ReportResumeEvaluation(BaseModel):
    score: float = Field(ge=0, le=100)
    max_score: float = 100
    quality_level: str
    summary: str
    sections: dict[str, ReportResumeSection]


class ReportMarketSource(BaseModel):
    title: str
    employer: str | None
    url: str


class ReportMarketContext(BaseModel):
    status: Literal["ok", "insufficient_market_data"]
    sample_size: int = Field(ge=0, le=6)
    confidence: Literal["low", "medium", "high"]
    summary: str
    sources: list[ReportMarketSource]
    country: str | None = None
    lookback_days: int | None = None
    collection_status: str | None = None
    complete_descriptions: int | None = None
    collection_errors: list[str] = Field(default_factory=list)
    search_warnings: list[str] = Field(default_factory=list)
    adjacent_context: AdjacentMarketContext | None = None
    reliability: MarketReliability | None = None


class ReportEvidence(BaseModel):
    section: str
    text: str


class ReportRequirementFact(BaseModel):
    name: str
    category: str
    importance: str
    coverage: float = Field(ge=0, le=1)
    resume_status: str
    evidence: list[ReportEvidence]
    frequency: int | None = None
    market_sources: list[MarketSource] = Field(default_factory=list)
    market_evidence: list[MarketRequirementEvidence] = Field(default_factory=list)


class ReportCandidateSkill(BaseModel):
    name: str
    resume_status: str


class ReportMarketComparison(BaseModel):
    status: Literal["ok", "market_data_unavailable"]
    summary_metrics: dict[str, int]
    strong_matches: list[ReportRequirementFact]
    missing_common: list[ReportRequirementFact]
    missing_additional: list[ReportRequirementFact]
    mentioned_only: list[ReportRequirementFact]
    candidate_only_skills: list[ReportCandidateSkill]
    responsibility_matches: list[ReportRequirementFact]
    responsibility_gaps: list[ReportRequirementFact]
    unconfirmed_resume_skills: list[ReportCandidateSkill] = Field(default_factory=list)


class ReportRecommendation(BaseModel):
    fact_id: str
    title: str
    text: str
    basis: list[str]
    support: list[RecommendationBasis] = Field(default_factory=list)
    action: RecommendationAction | None = None
    confidence: Literal["low", "medium", "high"] | None = None


class ReportRecommendations(BaseModel):
    generation_mode: Literal["llm", "fallback"]
    llm_calls: int = Field(ge=0, le=2)
    strengths: list[ReportRecommendation]
    resume_improvements: list[ReportRecommendation]
    market_gaps: list[ReportRecommendation]
    unconfirmed_skills: list[ReportRecommendation]
    development_directions: list[ReportRecommendation]
    final_summary: str


class ReportAlternativeRoleEvidence(BaseModel):
    section: str
    text: str


class ReportAlternativeRole(BaseModel):
    rank: int = Field(ge=1)
    target: MarketTarget
    status: Literal["validated", "market_data_unavailable"]
    confidence: Literal["low", "medium", "high"]
    evidence: list[ReportAlternativeRoleEvidence]
    readiness_score: float | None = Field(default=None, ge=0, le=1)
    market_sample_size: int = Field(ge=0, le=6)
    matched_requirements: list[str]
    transition_gaps: list[str]
    explanation: str


class ReportAlternativeRoles(BaseModel):
    status: Literal[
        "ok",
        "partial",
        "insufficient_candidate_evidence",
        "generation_failed",
        "no_market_supported_alternatives",
    ]
    message: str
    roles: list[ReportAlternativeRole]


class ReportArtifacts(BaseModel):
    resume: str
    resume_evaluation: str
    market_profile: str
    market_comparison: str
    recommendations: str
    alternative_roles: str | None = None
    readable_report: str
    vacancy_collection: str | None = None


class CandidateReport(BaseModel):
    run_id: str | None = None
    status: Literal["ok", "partial"]
    generated_at: str
    candidate_name: str | None
    target: MarketTarget
    resume_evaluation: ReportResumeEvaluation
    market_context: ReportMarketContext
    market_comparison: ReportMarketComparison
    recommendations: ReportRecommendations
    alternative_roles: ReportAlternativeRoles | None = None
    artifacts: ReportArtifacts
    stage_seconds: dict[str, float]
    total_seconds: float = Field(ge=0)


@dataclass(frozen=True)
class CandidateAssessmentPaths:
    report_json: Path
    report_markdown: Path
    stages_directory: Path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_model(model: BaseModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, model.model_dump_json(indent=2))


def _load_resume(path: Path) -> tuple[ResumeData, str]:
    if path.suffix.casefold() == ".json":
        resume = ResumeData.model_validate_json(path.read_text(encoding="utf-8"))
        return resume, resume.model_dump_json(indent=2)
    if path.suffix.casefold() not in {".pdf", ".docx", ".txt"}:
        raise ValueError("Поддерживаются resume.pdf, resume.docx, resume.txt и resume.json")
    source_text = read_resume(path)
    return extract_resume(source_text), source_text


def _load_resume_evaluation(path: Path) -> ResumeQualityResult:
    return ResumeQualityResult.model_validate(_read_json(path))


def _load_market_profile(
    path: Path,
) -> MarketProfile | InsufficientMarketProfile:
    payload = _read_json(path)
    if payload.get("status") == "ok":
        profile = MarketProfile.model_validate(payload)
        diagnostics_path = path.with_name(f"{path.stem}.diagnostics.json")
        if diagnostics_path.is_file():
            try:
                diagnostics = _read_json(diagnostics_path)
                target = diagnostics.get("target") or {}
                rows = diagnostics.get("final_vacancies") or []
                matches = (diagnostics.get("namespace") == "market_profile_v2"
                           and diagnostics.get("final_sample_size") == profile.sample_size
                           and {row["source_id"] for row in rows} == {source.source_id for source in profile.sources}
                           and all(target.get(key) == value for key, value in profile.target.model_dump().items()))
                levels = {"low": 0, "medium": 1, "high": 2}
                saved_confidence = diagnostics.get("confidence")
                if matches and saved_confidence in levels:
                    if levels[saved_confidence] < levels[profile.confidence]:
                        profile.confidence = saved_confidence
                        profile.summary += " Уверенность уточнена по сохранённой диагностике выборки."
                    if diagnostics.get("reliability"):
                        reliability = MarketReliability.model_validate(diagnostics["reliability"])
                        if reliability.sample_size == profile.sample_size:
                            profile.reliability = reliability.model_copy(update={
                                "confidence": profile.confidence, "provisional": profile.confidence == "low",
                            })
            except (OSError, ValueError, TypeError, KeyError, AttributeError, ValidationError):
                pass
        return profile
    return InsufficientMarketProfile.model_validate(payload)


def _market_source(source: MarketSource) -> ReportMarketSource:
    return ReportMarketSource(
        title=source.title,
        employer=source.employer,
        url=source.url,
    )


def _requirement_fact(item: RequirementComparison) -> ReportRequirementFact:
    return ReportRequirementFact(
        name=item.name,
        category=item.category,
        importance=item.market_importance,
        coverage=item.market_coverage,
        resume_status=item.resume_status,
        frequency=item.market_frequency,
        market_sources=item.market_sources,
        market_evidence=item.market_evidence,
        evidence=[
            ReportEvidence(section=evidence.section, text=evidence.text)
            for evidence in item.resume_evidence
        ],
    )


def _comparison_report(
    comparison: CandidateMarketComparison | MarketDataUnavailableComparison,
) -> ReportMarketComparison:
    if comparison.status != "ok":
        return ReportMarketComparison(
            status="market_data_unavailable",
            summary_metrics={},
            strong_matches=[],
            missing_common=[],
            missing_additional=[],
            mentioned_only=[],
            candidate_only_skills=[],
            responsibility_matches=[],
            responsibility_gaps=[],
        )
    return ReportMarketComparison(
        status="ok",
        summary_metrics={
            key: int(value)
            for key, value in comparison.summary_metrics.model_dump().items()
        },
        strong_matches=[_requirement_fact(item) for item in comparison.strong_matches],
        missing_common=[
            _requirement_fact(item)
            for item in comparison.missing_common_requirements
        ],
        missing_additional=[
            _requirement_fact(item)
            for item in comparison.missing_additional_requirements
        ],
        mentioned_only=[
            _requirement_fact(item)
            for item in comparison.mentioned_only_requirements
        ],
        candidate_only_skills=[
            ReportCandidateSkill(name=item.name, resume_status=item.resume_status)
            for item in comparison.candidate_only_skills
        ],
        unconfirmed_resume_skills=[ReportCandidateSkill(name=name, resume_status="mentioned_only")
                                   for name in _unconfirmed_names(comparison)],
        responsibility_matches=[
            _requirement_fact(item) for item in comparison.responsibility_matches
        ],
        responsibility_gaps=[
            _requirement_fact(item) for item in comparison.responsibility_gaps
        ],
    )


def _recommendation_item(item: RecommendationItem) -> ReportRecommendation:
    return ReportRecommendation(
        fact_id=item.fact_id,
        title=item.title,
        text=item.text,
        basis=[f"{basis.type}:{basis.reference}" for basis in item.basis],
        support=item.basis, action=item.action, confidence=item.confidence,
    )


def _recommendations_report(
    result: RecommendationResult,
) -> ReportRecommendations:
    return ReportRecommendations(
        generation_mode=result.generation_mode,
        llm_calls=result.llm_calls,
        strengths=[_recommendation_item(item) for item in result.strengths],
        resume_improvements=[
            _recommendation_item(item) for item in result.resume_improvements
        ],
        market_gaps=[_recommendation_item(item) for item in result.market_gaps],
        unconfirmed_skills=[
            _recommendation_item(item) for item in result.unconfirmed_skills
        ],
        development_directions=[
            _recommendation_item(item) for item in result.development_directions
        ],
        final_summary=result.final_summary,
    )


def _alternative_roles_report(
    result: AlternativeRoleExplorerResult | None,
) -> ReportAlternativeRoles | None:
    if result is None:
        return None
    return ReportAlternativeRoles(
        status=result.status,
        message=result.message,
        roles=[
            ReportAlternativeRole(
                rank=item.rank,
                target=item.target,
                status=item.status,
                confidence=item.confidence,
                evidence=[
                    ReportAlternativeRoleEvidence(
                        section=evidence.section,
                        text=evidence.text,
                    )
                    for evidence in item.evidence
                ],
                readiness_score=item.readiness_score,
                market_sample_size=item.market_sample_size,
                matched_requirements=item.matched_requirements,
                transition_gaps=item.transition_gaps,
                explanation=item.explanation,
            )
            for item in result.alternative_roles
        ],
    )


def _resume_evaluation_report(
    evaluation: ResumeQualityResult,
) -> ReportResumeEvaluation:
    return ReportResumeEvaluation(
        score=evaluation.total_score,
        max_score=evaluation.max_score,
        quality_level=evaluation.quality_level,
        summary=evaluation.overall_summary,
        sections={
            name: ReportResumeSection(
                score=section.score,
                max_score=section.max_score,
                strengths=section.strengths,
                weaknesses=section.weaknesses,
            )
            for name, section in evaluation.sections.items()
        },
    )


def _market_context_report(
    profile: MarketProfile | InsufficientMarketProfile,
) -> ReportMarketContext:
    return ReportMarketContext(
        status=profile.status,
        sample_size=profile.sample_size,
        confidence=profile.confidence,
        summary=profile.summary,
        sources=[_market_source(source) for source in profile.sources],
        adjacent_context=profile.adjacent_context,
        reliability=profile.reliability,
    )


def build_candidate_report(
    *,
    resume: ResumeData,
    resume_evaluation: ResumeQualityResult,
    market_profile: MarketProfile | InsufficientMarketProfile,
    market_comparison: CandidateMarketComparison | MarketDataUnavailableComparison,
    recommendations: RecommendationResult,
    artifacts: ReportArtifacts,
    stage_seconds: dict[str, float],
    total_seconds: float,
    alternative_roles: AlternativeRoleExplorerResult | None = None,
) -> CandidateReport:
    """Собирает представление без повторного вычисления входных результатов."""
    return CandidateReport(
        status="ok" if market_profile.status == "ok" else "partial",
        generated_at=datetime.now(timezone.utc).isoformat(),
        candidate_name=resume.candidate_name,
        target=market_profile.target,
        resume_evaluation=_resume_evaluation_report(resume_evaluation),
        market_context=_market_context_report(market_profile),
        market_comparison=_comparison_report(market_comparison),
        recommendations=_recommendations_report(recommendations),
        alternative_roles=_alternative_roles_report(alternative_roles),
        artifacts=artifacts,
        stage_seconds=stage_seconds,
        total_seconds=round(total_seconds, 3),
    )


def _markdown_list(items: list[ReportRecommendation]) -> list[str]:
    if not items:
        return ["Данных для этого раздела нет."]
    lines: list[str] = []
    for item in items:
        lines.extend([f"- **{item.title}.** {item.text}", ""])
        if item.action:
            lines.extend(f"  {index}. {step}" for index, step in enumerate(item.action.steps, 1))
            lines.extend(["", f"  **Результат:** {item.action.expected_result}", ""])
        if item.confidence:
            confidence = {"low": "низкая, предварительный совет", "medium": "средняя", "high": "высокая"}[item.confidence]
            lines.extend([f"  **Уверенность в совете:** {confidence}.", ""])
        for basis in item.support:
            if basis.market_frequency is not None and basis.market_sample_size is not None:
                employer_count = f" Независимых работодателей для требования: {basis.independent_employers}." if basis.independent_employers is not None else ""
                lines.extend([f"  **Основание:** требование встречается в {basis.market_frequency} из {basis.market_sample_size} вакансий выборки." + employer_count, ""])
            if basis.resume_status:
                status = {"evidenced": "подтверждено примером", "matched": "найдено подтверждение",
                          "mentioned_only": "только упомянуто, применение не показано",
                          "not_found": "подтверждение в документе не найдено"}.get(basis.resume_status, basis.resume_status)
                lines.extend([f"  **В резюме:** {status}.", ""])
            for evidence in basis.evidence:
                lines.extend([f"  «{evidence.replace(chr(10), ' ')}»", ""])
            for source in basis.market_sources[:3]:
                values = list(dict.fromkeys(row.text for row in basis.market_evidence if row.source_id == source.source_id))[:2]
                wording = " Требование: " + "; ".join(f"«{value}»" for value in values) + "." if values else ""
                lines.append(f"  - [{source.title}]({source.url})" + (f" — {source.employer}." if source.employer else ".") + wording)
            if basis.market_sources:
                lines.append("")
    return lines


def _render_detailed_report_markdown(report: CandidateReport) -> str:
    """Создаёт пользовательский отчёт только из готового CandidateReport."""
    target = " / ".join(
        value
        for value in (
            report.target.specialization,
            report.target.seniority,
        )
        if value
    )
    lines = [
        "# Отчёт по резюме кандидата",
        "",
        f"**Кандидат:** {report.candidate_name or 'Не указан'}  ",
        f"**Целевая позиция:** {target or 'Не указана'}  ",
        (
            f"**Качество резюме:** {report.resume_evaluation.score:g}/100 — "
            f"{report.resume_evaluation.quality_level}"
        ),
        "",
        report.recommendations.final_summary,
        "",
        "## Качество разделов резюме",
        "",
        "| Раздел | Балл | Что видно |",
        "|---|---:|---|",
    ]
    for name, section in report.resume_evaluation.sections.items():
        detail = (section.strengths or section.weaknesses or ["Нет деталей"])[0]
        detail = detail.replace("|", "\\|")
        lines.append(
            f"| {_SECTION_NAMES.get(name, name)} | "
            f"{section.score:g}/{section.max_score:g} | {detail} |"
        )
    if report.market_context.country:
        lines.extend([
            "", f"**Рынок:** {report.market_context.country}; "
            f"период публикации: {report.market_context.lookback_days} дней. "
            f"Полных описаний для анализа: {report.market_context.complete_descriptions}.",
        ])
    lines.extend(
        [
            "",
            "## Рыночный контекст",
            "",
            (
                f"Профиль построен по {report.market_context.sample_size} "
                f"вакансиям; confidence: `{report.market_context.confidence}`."
                if report.market_context.status == "ok"
                else "Рыночных данных недостаточно; рыночные рекомендации не сформированы."
            ),
            "",
            report.market_context.summary,
            "",
            "## Сильные стороны",
            "",
            *_markdown_list(report.recommendations.strengths),
            "",
            "## Что улучшить в резюме",
            "",
            *_markdown_list(report.recommendations.resume_improvements),
            "",
            "## Требования выборки, не обнаруженные в резюме",
            "",
            *_markdown_list(report.recommendations.market_gaps),
            "",
            "## Навыки без практического подтверждения",
            "",
            *_markdown_list(report.recommendations.unconfirmed_skills),
            "",
            "## Возможные направления развития",
            "",
            *_markdown_list(report.recommendations.development_directions),
            "",
        ]
    )
    adjacent = report.market_context.adjacent_context
    if adjacent is not None:
        lines.extend(["## Смежные вакансии: контекст возможного развития", "", adjacent.summary, ""])
        for source in adjacent.sources:
            lines.append(f"- [{source.title}]({source.url})" + (f" — {source.employer}" if source.employer else ""))
        lines.append("")
        categories = {
            "professional_skills": "Профессиональные навыки",
            "tools_and_technologies": "Инструменты и технологии",
            "responsibilities": "Обязанности",
            "education_requirements": "Образование",
            "domain_knowledge": "Предметная область",
            "soft_skills": "Гибкие навыки",
        }
        for category, label in categories.items():
            requirements = [*getattr(adjacent, category),
                            *(item for item in adjacent.market_variations if item.category == category)]
            if requirements:
                lines.extend([f"### {label}", ""])
                for item in sorted(requirements, key=lambda value: (-value.frequency, value.name.casefold())):
                    links = ", ".join(f"[источник {index}]({source.url})"
                                      for index, source in enumerate(item.sources, 1))
                    lines.append(f"- {item.name}: {item.frequency}/{adjacent.sample_size} смежных вакансий; {links}.")
                lines.append("")
    if report.market_context.reliability:
        reliability = report.market_context.reliability
        lines.extend(["## Основания уверенности в рыночной выборке", "",
                      f"Независимых работодателей: {reliability.independent_employers}; площадок: {reliability.provider_count}. "
                      f"Полнота описаний подтверждена для {reliability.full_description_count}/{reliability.sample_size} вакансий; "
                      f"публикация за последние {reliability.lookback_days} дней — для {reliability.recent_publication_count}/{reliability.sample_size}.", "",
                      *(f"- {reason}" for reason in reliability.reasons), ""])
    if report.alternative_roles is not None:
        lines.extend(
            [
                "## Альтернативные профессиональные направления",
                "",
                report.alternative_roles.message,
                "",
            ]
        )
        if not report.alternative_roles.roles:
            lines.extend([
                "Недостаточно подтверждённых данных для безопасной рекомендации.",
                "",
            ])
        for role in report.alternative_roles.roles:
            label = role.target.specialization or role.target.role or "Роль"
            readiness = (
                f"{role.readiness_score * 100:.0f}%"
                if role.readiness_score is not None
                else "не рассчитана"
            )
            evidence = "; ".join(item.text for item in role.evidence[:3])
            matches = ", ".join(role.matched_requirements) or "не обнаружены"
            gaps = ", ".join(role.transition_gaps) or "не определены"
            lines.extend(
                [
                    f"### {role.rank}. {label}",
                    "",
                    (
                        f"Confidence: `{role.confidence}`; готовность по "
                        f"рассмотренной выборке: {readiness}."
                    ),
                    "",
                    f"**Основания из резюме:** {evidence}.",
                    "",
                    f"**Совпавшие требования:** {matches}.",
                    "",
                    f"**Переходные пробелы:** {gaps}.",
                    "",
                ]
            )
    lines.extend(
        [
            "---",
            "",
            (
                "Отчёт оценивает только сведения, представленные в резюме, и "
                "не утверждает, что кандидат не владеет не найденным навыком. "
                "Рыночные выводы относятся только к рассмотренной выборке вакансий."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def render_candidate_report_markdown(report: CandidateReport, *, detailed: bool = False) -> str:
    """Краткий отчёт по умолчанию; вся трассировка остаётся в JSON и detailed."""
    if detailed:
        return _render_detailed_report_markdown(report)
    target = " / ".join(value for value in (report.target.specialization, report.target.seniority) if value)
    context = report.market_context
    confidence = {"low": "низкая", "medium": "средняя", "high": "высокая"}[context.confidence]
    quality_prefix = (f"Качество резюме оценено на {report.resume_evaluation.score:g}/100: "
                      f"{report.resume_evaluation.quality_level}.")
    summary = report.recommendations.final_summary.removeprefix(quality_prefix).strip()
    lines = ["# Отчёт по резюме кандидата", "",
             f"**Кандидат:** {report.candidate_name or 'Не указан'}  ",
             f"**Целевая позиция:** {target or 'Не указана'}  ",
             f"**Качество резюме:** {report.resume_evaluation.score:g}/100 — {report.resume_evaluation.quality_level}",
             "", summary, "", "## Рыночный контекст", ""]
    if context.status == "ok":
        scope = f"{context.country}; последние {context.lookback_days} дней. " if context.country else ""
        lines.append(f"{scope}Основная выборка: {context.sample_size} вакансий; уверенность — {confidence}.")
        if context.reliability:
            lines.append(f"Независимых работодателей: {context.reliability.independent_employers}; площадок: {context.reliability.provider_count}.")
        if context.confidence == "low":
            lines.append("Выводы предварительные из-за ограниченной выборки.")
    else:
        lines.append("Рыночных данных недостаточно; рыночные рекомендации не сформированы.")
        if context.country:
            lines.append(f"{context.country}; последние {context.lookback_days} дней. Подходящих вакансий: {context.sample_size}; для основного профиля требуется минимум 3.")
    lines.extend(dict.fromkeys(context.collection_errors + context.search_warnings))

    def soft(item: ReportRecommendation) -> bool:
        return any(basis.category == "soft_skills" for basis in item.support)

    def list_advice(title: str, items: list[ReportRecommendation], limit: int = 5) -> None:
        selected = [item for item in items if not soft(item)][:limit]
        if not selected:
            return
        lines.extend(["", f"## {title}", ""])
        for item in selected:
            market = next((basis for basis in item.support if basis.market_frequency is not None), None)
            signal = f" ({market.market_frequency}/{market.market_sample_size} вакансий)" if market else ""
            if title == "Сильные стороны" and market:
                proof = next(iter(market.evidence), "Подтверждено в резюме.")
                if len(proof) > 200:
                    proof = proof[:197].rstrip() + "…"
                lines.append(f"- **{market.reference}**{signal}: {proof}")
            else:
                lines.append(f"- **{item.title}**{signal}. {item.text}")
    list_advice("Сильные стороны", report.recommendations.strengths, 3)
    list_advice("Что улучшить в резюме", report.recommendations.resume_improvements, 3)
    list_advice("Требования выборки, не обнаруженные в резюме", report.recommendations.market_gaps)
    unconfirmed = report.market_comparison.unconfirmed_resume_skills
    if unconfirmed:
        lines.extend(["", "## Навыки без практического подтверждения", "",
                      ", ".join(item.name for item in unconfirmed) + ".",
                      "Добавьте реальные примеры применения для важных навыков. Перечень инструментов сам по себе не подтверждает практику."])
    elif report.recommendations.unconfirmed_skills:
        list_advice("Навыки без практического подтверждения", report.recommendations.unconfirmed_skills)
    flexible = list(dict.fromkeys(item.name
                    for item in [*report.market_comparison.missing_common,
                                 *report.market_comparison.missing_additional,
                                 *report.market_comparison.mentioned_only]
                    if item.category == "soft_skills"))
    if flexible:
        lines.extend(["", "## Рекомендуемые гибкие навыки", "", ", ".join(flexible) + ".",
                      "Указывайте их, если они действительно характеризуют вашу работу."])
    if context.sources:
        lines.extend(["", "## Источники основной выборки", ""])
        for source in context.sources:
            lines.append(f"- [{source.title}]({source.url})" + (f" — {source.employer}" if source.employer else ""))
    if context.adjacent_context:
        lines.extend(["", "## Смежные вакансии: контекст возможного развития", "",
                      f"Отдельно изучено {context.adjacent_context.sample_size} вакансий; они не входят в частоты основной выборки.", ""])
        for source in context.adjacent_context.sources:
            lines.append(f"- [{source.title}]({source.url})" + (f" — {source.employer}" if source.employer else ""))
    if report.alternative_roles:
        lines.extend(["", "## Альтернативные профессиональные направления", "", report.alternative_roles.message])
        for role in report.alternative_roles.roles:
            lines.append(f"- {role.target.specialization or role.target.role}: {role.explanation}")
    lines.extend(["", "Отсутствие подтверждения в документе не означает отсутствия навыка у кандидата. "
                  "Рыночные выводы относятся только к рассмотренной выборке.", "",
                  f"[Полные данные, доказательства и источники]({report.artifacts.readable_report.replace('.md', '.json')})", ""])
    return "\n".join(lines)


_SECTION_NAMES = {
    "experience": "Опыт работы",
    "projects": "Проекты",
    "header": "Шапка",
    "summary": "О себе",
    "skills": "Навыки",
    "education": "Образование",
}


@managed_run
def run_candidate_assessment(
    resume_path: Path,
    target_profession: str | None,
    output_directory: Path,
    *,
    resume_evaluation_path: Path | None = None,
    market_profile_path: Path | None = None,
    use_market_v2: bool = False,
    use_comparison_embeddings: bool = True,
    use_recommendation_llm: bool = True,
    include_alternative_roles: bool = False,
    alternative_role_limit: int = 3,
    use_russian_market: bool = False,
    vacancies_json_path: Path | None = None,
    detailed_report: bool = False,
) -> CandidateAssessmentPaths:
    """Выполняет каждый существующий этап ровно один раз или читает его cache."""
    started_at = time.perf_counter()
    output_directory.mkdir(parents=True, exist_ok=True)
    stages = output_directory / ".stages"
    stages.mkdir(parents=True, exist_ok=True)
    paths = CandidateAssessmentPaths(
        report_json=output_directory / "candidate_report.json",
        report_markdown=output_directory / "candidate_report.md",
        stages_directory=stages,
    )
    stage_seconds: dict[str, float] = {}

    stage_started = time.perf_counter()
    update_stage("resume")
    resume, resume_source_text = _load_resume(resume_path)
    russian_market = use_russian_market or target_profession is None
    market_target = target_from_resume(resume) if russian_market else target_profession
    resume_stage_path = stages / "resume.json"
    _save_model(resume, resume_stage_path)
    stage_seconds["resume"] = round(time.perf_counter() - stage_started, 3)

    stage_started = time.perf_counter()
    update_stage("resume_evaluation")
    if resume_evaluation_path:
        resume_evaluation = _load_resume_evaluation(resume_evaluation_path)
        if resume.candidate_name and resume_evaluation.candidate_name and resume.candidate_name.casefold().strip() != resume_evaluation.candidate_name.casefold().strip():
            raise ValueError("Готовая оценка принадлежит другому кандидату")
    else:
        resume_evaluation = build_resume_quality_result(
            evaluate_resume(resume_source_text)
        )
    evaluation_stage_path = stages / "resume_evaluation.json"
    _save_model(resume_evaluation, evaluation_stage_path)
    stage_seconds["resume_evaluation"] = round(
        time.perf_counter() - stage_started,
        3,
    )

    stage_started = time.perf_counter()
    update_stage("market_profile")
    market_stage_path = stages / "market_profile.json"
    if market_profile_path:
        market_profile = _load_market_profile(market_profile_path)
        if russian_market:
            collection_path = market_profile_path.with_name(
                f"{market_profile_path.stem}.collection.json")
            collection = validate_saved_russian_profile(market_profile, market_target, collection_path)
            _save_model(collection, market_stage_path.with_name("market_profile.collection.json"))
        _save_model(market_profile, market_stage_path)
    elif russian_market:
        market_profile = build_russian_market_profile(
            market_target,
            output_path=market_stage_path,
            imported_path=vacancies_json_path,
        )
    else:
        market_builder = (
            build_market_profile_v2 if use_market_v2 else build_market_profile
        )
        market_profile = market_builder(target_profession, output_path=market_stage_path)
    stage_seconds["market_profile"] = round(
        time.perf_counter() - stage_started,
        3,
    )

    stage_started = time.perf_counter()
    update_stage("market_comparison")
    comparison_stage_path = stages / "candidate_market_comparison.json"
    if market_profile.status == "ok":
        comparison = compare_candidate_to_market(
            resume,
            market_profile,
            output_path=comparison_stage_path,
            use_embeddings=use_comparison_embeddings,
        )
    else:
        comparison = MarketDataUnavailableComparison(
            target=market_profile.target,
            market_sample_size=market_profile.sample_size,
            market_reliability=market_profile.reliability,
            message=(
                "Сравнение не выполнено: доступных рыночных данных недостаточно."
            ),
        )
        _save_model(comparison, comparison_stage_path)
    stage_seconds["market_comparison"] = round(
        time.perf_counter() - stage_started,
        3,
    )

    stage_started = time.perf_counter()
    update_stage("recommendations")
    recommendation_stage_path = stages / "recommendations.json"
    recommendations = generate_candidate_recommendations(
        resume,
        resume_evaluation,
        comparison,
        output_path=recommendation_stage_path,
        use_llm=use_recommendation_llm,
    )
    stage_seconds["recommendations"] = round(
        time.perf_counter() - stage_started,
        3,
    )

    alternative_roles: AlternativeRoleExplorerResult | None = None
    alternative_roles_artifact: str | None = None
    if include_alternative_roles:
        update_stage("alternative_roles")
        stage_started = time.perf_counter()
        alternative_directory = stages / "alternative_roles"
        alternative_roles = explore_alternative_roles(
            resume,
            alternative_directory,
            market_validation_limit=alternative_role_limit,
            use_comparison_embeddings=use_comparison_embeddings,
            **({"market_builder": build_russian_market_profile} if russian_market else {}),
        )
        alternative_result_path = alternative_directory / "alternative_roles.json"
        alternative_roles_artifact = str(
            alternative_result_path.relative_to(output_directory)
        )
        stage_seconds["alternative_roles"] = round(
            time.perf_counter() - stage_started,
            3,
        )

    update_stage("report")
    artifacts = ReportArtifacts(
        resume=str(resume_stage_path.relative_to(output_directory)),
        resume_evaluation=str(evaluation_stage_path.relative_to(output_directory)),
        market_profile=str(market_stage_path.relative_to(output_directory)),
        market_comparison=str(comparison_stage_path.relative_to(output_directory)),
        recommendations=str(recommendation_stage_path.relative_to(output_directory)),
        alternative_roles=alternative_roles_artifact,
        readable_report=paths.report_markdown.name,
        vacancy_collection=(
            str(market_stage_path.with_name("market_profile.collection.json").relative_to(output_directory))
            if russian_market else None
        ),
    )
    report = build_candidate_report(
        resume=resume,
        resume_evaluation=resume_evaluation,
        market_profile=market_profile,
        market_comparison=comparison,
        recommendations=recommendations,
        artifacts=artifacts,
        stage_seconds=stage_seconds,
        total_seconds=time.perf_counter() - started_at,
        alternative_roles=alternative_roles,
    )
    if artifacts.vacancy_collection:
        collection_path = output_directory / artifacts.vacancy_collection
        if collection_path.is_file():
            collection = _read_json(collection_path)["diagnostics"]
            report.market_context.country = collection["scope"]["country"]
            report.market_context.lookback_days = collection["scope"]["lookback_days"]
            report.market_context.collection_status = collection["status"]
            report.market_context.complete_descriptions = collection["accepted"]
            report.market_context.collection_errors = list(dict.fromkeys(f"{item['source']}: {item['reason']}" for item in collection.get("errors", [])))
            if collection.get("search_limit_reached"):
                report.market_context.search_warnings.append("Достигнут лимит поиска; выборка ограничена настройками.")
            titles = collection.get("search_titles") or {}
            if titles.get("mode") == "unavailable":
                report.market_context.search_warnings.append("Перевод названия должности недоступен; поиск выполнен по доступным вариантам.")
    report.run_id = current_run_id()
    _save_model(report, paths.report_json)
    atomic_write_text(paths.report_markdown, render_candidate_report_markdown(report, detailed=detailed_report))
    return paths


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Единый отчёт: качество резюме, рынок, сравнение и советы",
    )
    parser.add_argument("resume", type=Path)
    parser.add_argument("target_profession", nargs="?",
                        help="Необязательно: старая ручная цель. Без неё — Россия по резюме")
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--market-russia", action="store_true",
                        help="Россия за 120 дней; должность и уровень берутся из резюме")
    parser.add_argument("--detailed-report", action="store_true", help="Подробный Markdown с полной трассировкой советов")
    parser.add_argument("--vacancies-json", type=Path,
                        help="Полные вакансии России в JSON вместо интернет-запросов")
    parser.add_argument(
        "--resume-evaluation",
        type=Path,
        help="Готовый ResumeQualityResult: использовать без пересчёта",
    )
    parser.add_argument(
        "--market-profile",
        type=Path,
        help="Готовый MarketProfile: использовать без нового retrieval",
    )
    parser.add_argument(
        "--market-v2",
        action="store_true",
        help=(
            "Использовать упрощённый MarketProfileBuilder v2; по умолчанию "
            "остаётся production builder v1"
        ),
    )
    parser.add_argument(
        "--no-comparison-embeddings",
        action="store_true",
        help="Lexical-only comparison; полезно для детерминированной диагностики",
    )
    parser.add_argument(
        "--no-recommendation-llm",
        action="store_true",
        help="Безопасные шаблонные рекомендации без Gemma",
    )
    parser.add_argument(
        "--alternative-roles",
        action="store_true",
        help="Проверить альтернативные профессии (добавляет live market calls)",
    )
    parser.add_argument(
        "--alternative-role-limit",
        type=int,
        default=3,
        help="Число альтернативных ролей для рыночной проверки (1–5)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.vacancies_json and args.target_profession is not None and not args.market_russia:
        raise SystemExit("--vacancies-json используется в сценарии России; добавьте --market-russia")
    paths = run_candidate_assessment(
        resume_path=args.resume,
        target_profession=args.target_profession,
        output_directory=args.output_directory,
        resume_evaluation_path=args.resume_evaluation,
        market_profile_path=args.market_profile,
        use_market_v2=args.market_v2,
        use_comparison_embeddings=not args.no_comparison_embeddings,
        use_recommendation_llm=not args.no_recommendation_llm,
        include_alternative_roles=args.alternative_roles,
        alternative_role_limit=args.alternative_role_limit,
        use_russian_market=args.market_russia,
        vacancies_json_path=args.vacancies_json,
        detailed_report=args.detailed_report,
    )
    print(f"JSON: {paths.report_json}")
    print(f"Отчёт: {paths.report_markdown}")


if __name__ == "__main__":
    main()
