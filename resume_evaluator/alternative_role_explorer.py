"""Evidence-first поиск альтернативных профессиональных направлений.

Модуль не подбирает конкретные объявления. Он формирует гипотезы ролей только
из фактов ResumeData, отбрасывает неподтверждённые гипотезы обычным Python и
проверяет лучшие направления через независимый MarketProfileBuilder.
"""
from .runtime import atomic_write_text


import argparse
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from ollama import ResponseError
from .model_runtime import chat
from pydantic import BaseModel, Field, ValidationError

from .candidate_market_comparison import (
    CandidateMarketComparison,
    MarketDataUnavailableComparison,
    compare_candidate_to_market,
)
from .profession import ProfessionProfile
from .resume_extractor import ResumeData
from .market_profile import (
    InsufficientMarketProfile,
    MarketProfile,
    MarketTarget,
    SyntheticVacancyError,
    build_market_profile,
)

from .settings import get_settings

MODEL_NAME = get_settings().model
MODEL_CONTEXT_SIZE = get_settings().context_size
MAX_OUTPUT_TOKENS = 4096
DEFAULT_OUTPUT = Path("alternative_roles.json")
MAX_GENERATED_CANDIDATES = 8
DEFAULT_MARKET_VALIDATION_LIMIT = 3
EvidenceSection = Literal[
    "skills",
    "soft_skills",
    "experience",
    "projects",
    "education",
    "courses",
    "achievements",
]


class GeneratedRoleHypothesis(BaseModel):
    role: str | None
    specialization: str
    domain: str | None
    seniority: str | None
    role_family: str | None
    focus: str | None
    evidence_refs: list[str] = Field(min_length=2, max_length=8)


class GeneratedRoleHypotheses(BaseModel):
    roles: list[GeneratedRoleHypothesis] = Field(
        max_length=MAX_GENERATED_CANDIDATES,
    )


class AlternativeRoleEvidence(BaseModel):
    section: EvidenceSection
    text: str


class AlternativeRoleCandidate(BaseModel):
    target: ProfessionProfile
    evidence: list[AlternativeRoleEvidence] = Field(min_length=2)
    preliminary_score: float = Field(ge=0, le=1)


class AlternativeRoleAssessment(BaseModel):
    rank: int = Field(ge=1)
    target: MarketTarget
    status: Literal["validated", "market_data_unavailable"]
    confidence: Literal["low", "medium", "high"]
    evidence: list[AlternativeRoleEvidence]
    preliminary_score: float = Field(ge=0, le=1)
    readiness_score: float | None = Field(default=None, ge=0, le=1)
    market_sample_size: int = Field(ge=0, le=6)
    common_requirements_matched: int = Field(ge=0)
    common_requirements_total: int = Field(ge=0)
    additional_requirements_matched: int = Field(ge=0)
    additional_requirements_total: int = Field(ge=0)
    matched_requirements: list[str]
    transition_gaps: list[str]
    explanation: str
    audit_artifacts: dict[str, str]


class AlternativeRoleExplorerResult(BaseModel):
    status: Literal[
        "ok",
        "partial",
        "insufficient_candidate_evidence",
        "generation_failed",
        "no_market_supported_alternatives",
    ]
    candidate_name: str | None
    primary_target: ProfessionProfile
    generation_mode: Literal["llm", "provided"]
    generated_candidates: int = Field(ge=0)
    evidence_valid_candidates: int = Field(ge=0)
    market_validated_candidates: int = Field(ge=0)
    market_rejected_candidates: int = Field(default=0, ge=0)
    alternative_roles: list[AlternativeRoleAssessment]
    message: str


class _EvidenceEntry(BaseModel):
    section: EvidenceSection
    text: str


ROLE_GENERATION_PROMPT = """
Ты предлагаешь только альтернативные профессиональные направления кандидата,
а не конкретные опубликованные вакансии.

Правила:
1. Основывай каждую роль только на фактах структурированного резюме.
2. evidence_refs копируй дословно из полей резюме. Не пересказывай их.
3. Для каждой роли укажи минимум два независимых подтверждения. Одного общего
   навыка вроде Python, SQL, Communication или Excel недостаточно.
4. Не повышай seniority относительно явно указанного целевого уровня.
5. Не возвращай исходную целевую профессию под другим написанием.
6. Не считай связанные инструменты взаимозаменяемыми и не придумывай опыт.
7. Предложи не более шести достаточно конкретных ролей. Если доказательств нет,
   верни пустой список.
8. Не делай выводов о спросе рынка: это проверит отдельный MarketProfileBuilder.
9. Верни только JSON заданной схемы.
"""


def _text_key(value: str) -> str:
    return " ".join(
        re.findall(r"[a-zа-яё0-9+#.&]+", value.casefold().replace("ё", "е"))
    )


def _add_entry(
    entries: list[_EvidenceEntry],
    section: str,
    value: Any,
) -> None:
    if isinstance(value, str) and value.strip():
        entries.append(_EvidenceEntry(section=section, text=" ".join(value.split())))


def _resume_evidence(resume: ResumeData) -> list[_EvidenceEntry]:
    entries: list[_EvidenceEntry] = []
    for value in resume.skills:
        _add_entry(entries, "skills", value)
    for value in resume.soft_skills:
        _add_entry(entries, "soft_skills", value)
    for workplace in resume.work_experience:
        _add_entry(entries, "experience", workplace.position)
        for value in (
            workplace.responsibilities
            + workplace.technologies
            + workplace.achievements
        ):
            _add_entry(entries, "experience", value)
    for project in resume.projects:
        for value in (
            [project.name, project.description, project.role]
            + project.technologies
            + project.results
        ):
            _add_entry(entries, "projects", value)
    for education in resume.education:
        for value in (
            education.institution,
            education.education_level,
            education.education_field,
            education.field_of_study,
        ):
            _add_entry(entries, "education", value)
    for value in resume.courses_and_certifications:
        _add_entry(entries, "courses", value)
    for value in resume.achievements:
        _add_entry(entries, "achievements", value)

    unique: list[_EvidenceEntry] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        key = (entry.section, _text_key(entry.text))
        if key[1] and key not in seen:
            seen.add(key)
            unique.append(entry)
    return unique


def _find_evidence(
    reference: str,
    entries: list[_EvidenceEntry],
) -> _EvidenceEntry | None:
    reference_key = _text_key(reference)
    if not reference_key:
        return None
    for entry in entries:
        entry_key = _text_key(entry.text)
        if reference_key == entry_key:
            return entry
        # Разрешаем только достаточно содержательную дословную фразу, а не
        # семантический пересказ или совпадение одного общего слова.
        if (
            len(reference_key.split()) >= 3
            and f" {reference_key} " in f" {entry_key} "
        ) or (
            len(entry_key.split()) >= 3
            and f" {entry_key} " in f" {reference_key} "
        ):
            return entry
    return None


def _seniority_rank(value: str | None) -> int | None:
    text = _text_key(value or "")
    for marker, rank in (
        ("head", 5),
        ("principal", 5),
        ("staff", 4),
        ("lead", 4),
        ("senior", 3),
        ("старш", 3),
        ("middle", 2),
        ("mid level", 2),
        ("мидл", 2),
        ("junior", 1),
        ("младш", 1),
    ):
        if marker in text:
            return rank
    return None


def _same_role(candidate: GeneratedRoleHypothesis, primary: ProfessionProfile) -> bool:
    aliases = {
        "ml": ("machine", "learning"),
        "ds": ("data", "scientist"),
        "cv": ("computer", "vision"),
        "pr": ("public", "relations"),
    }

    def role_tokens(value: str) -> set[str]:
        tokens: set[str] = set()
        for token in _text_key(value).split():
            expanded = aliases.get(token)
            tokens.update(expanded or (token,))
        return tokens

    candidate_label = candidate.specialization or candidate.role or ""
    primary_label = primary.specialization or primary.role or ""
    if _text_key(candidate_label) == _text_key(primary_label):
        return bool(_text_key(candidate_label))
    candidate_tokens = role_tokens(candidate_label)
    primary_tokens = role_tokens(primary_label)
    return bool(
        len(candidate_tokens) >= 2
        and candidate_tokens <= primary_tokens
    )


def _validate_hypotheses(
    resume: ResumeData,
    generated: GeneratedRoleHypotheses,
) -> list[AlternativeRoleCandidate]:
    entries = _resume_evidence(resume)
    primary_rank = _seniority_rank(resume.target_profession.seniority)
    candidates: list[AlternativeRoleCandidate] = []
    seen_roles: set[str] = set()
    substantial_sections = {
        "experience",
        "projects",
        "education",
        "courses",
        "achievements",
    }
    for hypothesis in generated.roles:
        role_key = _text_key(hypothesis.specialization)
        if not role_key or role_key in seen_roles:
            continue
        if _same_role(hypothesis, resume.target_profession):
            continue
        candidate_rank = _seniority_rank(hypothesis.seniority)
        if primary_rank and candidate_rank and candidate_rank > primary_rank:
            continue
        evidence: list[AlternativeRoleEvidence] = []
        evidence_keys: set[tuple[str, str]] = set()
        for reference in hypothesis.evidence_refs:
            match = _find_evidence(reference, entries)
            if match is None:
                continue
            key = (match.section, _text_key(match.text))
            if key in evidence_keys:
                continue
            evidence_keys.add(key)
            evidence.append(
                AlternativeRoleEvidence(section=match.section, text=match.text)
            )
        if len(evidence) < 2:
            continue
        if not any(item.section in substantial_sections for item in evidence):
            continue
        section_diversity = len({item.section for item in evidence})
        substantive = sum(item.section in substantial_sections for item in evidence)
        preliminary_score = min(
            1.0,
            0.20
            + 0.10 * min(len(evidence), 5)
            + 0.12 * min(section_diversity, 3)
            + 0.08 * min(substantive, 3),
        )
        target = ProfessionProfile(
            role=hypothesis.role,
            specialization=hypothesis.specialization,
            domain=hypothesis.domain,
            seniority=hypothesis.seniority or resume.target_profession.seniority,
            role_family=hypothesis.role_family,
            focus=hypothesis.focus,
            core_skills=[],
        )
        seen_roles.add(role_key)
        candidates.append(
            AlternativeRoleCandidate(
                target=target,
                evidence=evidence,
                preliminary_score=round(preliminary_score, 4),
            )
        )
    return sorted(
        candidates,
        key=lambda item: item.preliminary_score,
        reverse=True,
    )


def generate_role_hypotheses(resume: ResumeData) -> GeneratedRoleHypotheses:
    response = chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": ROLE_GENERATION_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {"resume": resume.model_dump()},
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ],
        format=GeneratedRoleHypotheses.model_json_schema(),
        think=False,
        options={
            "temperature": 0,
            "num_ctx": MODEL_CONTEXT_SIZE,
            "num_predict": MAX_OUTPUT_TOKENS,
        },
    )
    return GeneratedRoleHypotheses.model_validate_json(response.message.content)


def _readiness_score(
    candidate: AlternativeRoleCandidate,
    comparison: CandidateMarketComparison,
) -> float:
    requirements = [
        *comparison.professional_skills,
        *comparison.tools_and_technologies,
        *comparison.education_requirements,
        *comparison.domain_knowledge,
        *comparison.soft_skills,
    ]

    def coverage(items: list[Any]) -> float:
        weights = {
            "matched": 1.0,
            "evidenced": 1.0,
            "mentioned_only": 0.35,
            "not_found": 0.0,
        }
        return (
            sum(weights[item.resume_status] for item in items) / len(items)
            if items
            else 0.0
        )

    common = [item for item in requirements if item.market_importance == "common"]
    additional = [
        item for item in requirements if item.market_importance == "additional"
    ]
    weighted_parts = [(candidate.preliminary_score, 0.35)]
    if common:
        weighted_parts.append((coverage(common), 0.45))
    if additional:
        weighted_parts.append((coverage(additional), 0.10))
    if comparison.responsibilities:
        weighted_parts.append((coverage(comparison.responsibilities), 0.10))
    total_weight = sum(weight for _, weight in weighted_parts)
    return round(
        sum(value * weight for value, weight in weighted_parts) / total_weight,
        4,
    )


def _confidence(
    profile: MarketProfile,
    readiness_score: float,
) -> Literal["low", "medium", "high"]:
    if profile.confidence == "low":
        return "low"
    if profile.confidence == "high" and readiness_score >= 0.75:
        return "high"
    if readiness_score >= 0.45:
        return "medium"
    return "low"


def _market_target(profile: ProfessionProfile) -> MarketTarget:
    return MarketTarget(
        role=profile.role,
        specialization=profile.specialization,
        domain=profile.domain,
        seniority=profile.seniority,
        role_family=profile.role_family,
        focus=profile.focus,
    )


def explore_alternative_roles(
    resume: ResumeData | dict[str, Any],
    output_directory: Path,
    *,
    market_validation_limit: int = DEFAULT_MARKET_VALIDATION_LIMIT,
    generated_hypotheses: GeneratedRoleHypotheses | None = None,
    use_comparison_embeddings: bool = True,
    market_builder: (
        Callable[..., MarketProfile | InsufficientMarketProfile] | None
    ) = None,
) -> AlternativeRoleExplorerResult:
    """Строит и рыночно проверяет альтернативные роли, но не вакансии."""
    if not 1 <= market_validation_limit <= 5:
        raise ValueError("market_validation_limit должен быть от 1 до 5")
    resume_data = (
        resume if isinstance(resume, ResumeData) else ResumeData.model_validate(resume)
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    try:
        generated = generated_hypotheses or generate_role_hypotheses(resume_data)
    except (
        OSError,
        ResponseError,
        SyntheticVacancyError,
        TypeError,
        ValidationError,
        ValueError,
    ) as error:
        result = AlternativeRoleExplorerResult(
            status="generation_failed",
            candidate_name=resume_data.candidate_name,
            primary_target=resume_data.target_profession,
            generation_mode="llm",
            generated_candidates=0,
            evidence_valid_candidates=0,
            market_validated_candidates=0,
            market_rejected_candidates=0,
            alternative_roles=[],
            message=(
                "Не удалось сформировать гипотезы альтернативных ролей; "
                f"основная оценка остаётся доступной ({type(error).__name__})."
            ),
        )
        atomic_write_text(output_directory / DEFAULT_OUTPUT,
            result.model_dump_json(indent=2)
        )
        return result
    generation_mode: Literal["llm", "provided"] = (
        "provided" if generated_hypotheses is not None else "llm"
    )
    candidates = _validate_hypotheses(resume_data, generated)
    if not candidates:
        result = AlternativeRoleExplorerResult(
            status="insufficient_candidate_evidence",
            candidate_name=resume_data.candidate_name,
            primary_target=resume_data.target_profession,
            generation_mode=generation_mode,
            generated_candidates=len(generated.roles),
            evidence_valid_candidates=0,
            market_validated_candidates=0,
            market_rejected_candidates=0,
            alternative_roles=[],
            message=(
                "В резюме недостаточно подтверждённых фактов для безопасного "
                "предложения альтернативной профессиональной роли."
            ),
        )
        atomic_write_text(output_directory / DEFAULT_OUTPUT,
            result.model_dump_json(indent=2)
        )
        return result

    builder = market_builder or build_market_profile
    assessed: list[AlternativeRoleAssessment] = []
    market_rejected = 0
    for index, candidate in enumerate(candidates[:market_validation_limit], start=1):
        role_directory = output_directory / f"role_{index}"
        role_directory.mkdir(parents=True, exist_ok=True)
        market_path = role_directory / "market_profile.json"
        comparison_path = role_directory / "candidate_market_comparison.json"
        try:
            profile = builder(candidate.target, output_path=market_path)
            if not market_path.exists():
                atomic_write_text(market_path,
                    profile.model_dump_json(indent=2)
                )
            comparison = compare_candidate_to_market(
                resume_data,
                profile,
                output_path=comparison_path,
                use_embeddings=use_comparison_embeddings,
            )
        except (
            OSError,
            ResponseError,
            SyntheticVacancyError,
            TypeError,
            ValidationError,
            ValueError,
        ) as error:
            error_path = role_directory / "market_error.json"
            atomic_write_text(error_path,
                json.dumps(
                    {
                        "status": "market_data_unavailable",
                        "error_type": type(error).__name__,
                        "message": str(error),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            assessed.append(
                AlternativeRoleAssessment(
                    rank=index,
                    target=_market_target(candidate.target),
                    status="market_data_unavailable",
                    confidence="low",
                    evidence=candidate.evidence,
                    preliminary_score=candidate.preliminary_score,
                    readiness_score=None,
                    market_sample_size=0,
                    common_requirements_matched=0,
                    common_requirements_total=0,
                    additional_requirements_matched=0,
                    additional_requirements_total=0,
                    matched_requirements=[],
                    transition_gaps=[],
                    explanation=(
                        "Направление подтверждается резюме, но внешняя "
                        "рыночная проверка завершилась ошибкой."
                    ),
                    audit_artifacts={
                        "market_error": str(error_path.relative_to(output_directory))
                    },
                )
            )
            continue
        artifacts = {
            "market_profile": str(market_path.relative_to(output_directory)),
            "market_comparison": str(comparison_path.relative_to(output_directory)),
        }
        if isinstance(comparison, MarketDataUnavailableComparison):
            assessed.append(
                AlternativeRoleAssessment(
                    rank=index,
                    target=profile.target,
                    status="market_data_unavailable",
                    confidence="low",
                    evidence=candidate.evidence,
                    preliminary_score=candidate.preliminary_score,
                    readiness_score=None,
                    market_sample_size=profile.sample_size,
                    common_requirements_matched=0,
                    common_requirements_total=0,
                    additional_requirements_matched=0,
                    additional_requirements_total=0,
                    matched_requirements=[],
                    transition_gaps=[],
                    explanation=(
                        "Направление подтверждается резюме, но доступных "
                        "рыночных данных недостаточно для итоговой проверки."
                    ),
                    audit_artifacts=artifacts,
                )
            )
            continue

        if not comparison.strong_matches and not comparison.responsibility_matches:
            market_rejected += 1
            continue
        readiness = _readiness_score(candidate, comparison)
        metrics = comparison.summary_metrics
        assessed.append(
            AlternativeRoleAssessment(
                rank=index,
                target=profile.target,
                status="validated",
                confidence=_confidence(profile, readiness),
                evidence=candidate.evidence,
                preliminary_score=candidate.preliminary_score,
                readiness_score=readiness,
                market_sample_size=profile.sample_size,
                common_requirements_matched=metrics.common_requirements_matched,
                common_requirements_total=metrics.common_requirements_total,
                additional_requirements_matched=(
                    metrics.additional_requirements_matched
                ),
                additional_requirements_total=metrics.additional_requirements_total,
                matched_requirements=[
                    item.name for item in comparison.strong_matches[:6]
                ],
                transition_gaps=[
                    item.name
                    for item in (
                        comparison.missing_common_requirements
                        + comparison.missing_additional_requirements
                    )[:6]
                ],
                explanation=(
                    "Роль подтверждена фактами резюме и сопоставлена с "
                    f"профилем рынка из {profile.sample_size} вакансий."
                ),
                audit_artifacts=artifacts,
            )
        )

    assessed.sort(
        key=lambda item: (
            item.status == "validated",
            item.readiness_score or 0.0,
            item.preliminary_score,
        ),
        reverse=True,
    )
    for rank, item in enumerate(assessed, start=1):
        item.rank = rank
    validated_count = sum(item.status == "validated" for item in assessed)
    if not assessed and market_rejected:
        status = "no_market_supported_alternatives"
    else:
        status = "ok" if validated_count == len(assessed) else "partial"
    result = AlternativeRoleExplorerResult(
        status=status,
        candidate_name=resume_data.candidate_name,
        primary_target=resume_data.target_profession,
        generation_mode=generation_mode,
        generated_candidates=len(generated.roles),
        evidence_valid_candidates=len(candidates),
        market_validated_candidates=validated_count,
        market_rejected_candidates=market_rejected,
        alternative_roles=assessed,
        message=(
            f"Проверено альтернативных направлений: {len(assessed)}; "
            f"рыночными данными подтверждено: {validated_count}; "
            f"не подтверждено рынком: {market_rejected}."
        ),
    )
    atomic_write_text(output_directory / DEFAULT_OUTPUT,
        result.model_dump_json(indent=2)
    )
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Альтернативные профессиональные роли без подбора вакансий",
    )
    parser.add_argument("resume", type=Path, help="Готовый resume.json")
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_MARKET_VALIDATION_LIMIT,
        help="Сколько лучших ролей проверить через рынок (1–5)",
    )
    parser.add_argument("--no-comparison-embeddings", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    resume = ResumeData.model_validate_json(args.resume.read_text(encoding="utf-8"))
    result = explore_alternative_roles(
        resume,
        args.output_directory,
        market_validation_limit=args.limit,
        use_comparison_embeddings=not args.no_comparison_embeddings,
    )
    print(f"Статус: {result.status}")
    print(f"Результат: {args.output_directory / DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
