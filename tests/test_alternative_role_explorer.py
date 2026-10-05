"""Тесты evidence-first ветки альтернативных профессиональных ролей."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from resume_evaluator.alternative_role_explorer import (
    GeneratedRoleHypotheses,
    GeneratedRoleHypothesis,
    _validate_hypotheses,
    explore_alternative_roles,
)
from resume_evaluator.resume_extractor import ResumeData
from resume_evaluator.synthetic_vacancy_pipeline_simplified import (
    InsufficientMarketProfile,
    MarketProfile,
    MarketRequirement,
    MarketSource,
    MarketTarget,
)

ROOT = Path(__file__).resolve().parents[1]


def resume() -> ResumeData:
    return ResumeData.model_validate_json(
        (ROOT / "examples" / "precomputed" / "marketing_candidate.example.json").read_text(
            encoding="utf-8"
        )
    )


def hypothesis(
    specialization: str,
    evidence: list[str],
    *,
    seniority: str = "Middle",
) -> GeneratedRoleHypothesis:
    return GeneratedRoleHypothesis(
        role="Specialist",
        specialization=specialization,
        domain="Marketing Operations",
        seniority=seniority,
        role_family="Marketing",
        focus="Campaign Operations",
        evidence_refs=evidence,
    )


def market_profile(target: object) -> MarketProfile:
    target_data = target.model_dump()
    market_target = MarketTarget(**{
        key: target_data.get(key)
        for key in (
            "role",
            "specialization",
            "domain",
            "seniority",
            "role_family",
            "focus",
        )
    })
    sources = [
        MarketSource(
            source="test",
            source_id=f"alternative-{index}",
            title="Marketing Operations Specialist",
            employer=f"Company {index}",
            url=f"https://jobs.test/alternative/{index}",
            published_at=None,
            target_relevance=0.9,
        )
        for index in range(3)
    ]

    def requirement(name: str, importance: str = "common") -> MarketRequirement:
        frequency = 3 if importance == "common" else 2
        return MarketRequirement(
            name=name,
            frequency=frequency,
            coverage=frequency / 3,
            importance=importance,
            sources=sources[:frequency],
            source_values=[name],
        )

    return MarketProfile(
        target=market_target,
        sample_size=3,
        professional_skills=[requirement("Automation")],
        tools_and_technologies=[requirement("HubSpot")],
        responsibilities=[],
        education_requirements=[],
        domain_knowledge=[],
        soft_skills=[],
        market_variations=[],
        sources=sources,
        confidence="medium",
        summary="Тестовый профиль альтернативной роли.",
    )


class AlternativeRoleExplorerTests(unittest.TestCase):
    def test_hallucinated_same_and_higher_roles_are_rejected(self) -> None:
        generated = GeneratedRoleHypotheses(roles=[
            hypothesis(
                "Маркетолог",
                ["Automation", "Marketing Specialist"],
            ),
            hypothesis(
                "Head of Marketing Operations",
                ["Automation", "Marketing Specialist"],
                seniority="Head",
            ),
            hypothesis(
                "Data Engineer",
                ["Не существующий проект", "Apache Spark"],
            ),
        ])

        candidates = _validate_hypotheses(resume(), generated)

        self.assertEqual(candidates, [])

    def test_primary_role_alias_is_not_returned_as_alternative(self) -> None:
        candidate_resume = resume().model_copy(deep=True)
        candidate_resume.target_profession.specialization = (
            "Machine Learning Engineer"
        )
        candidate_resume.target_profession.role = "Engineer"
        generated = GeneratedRoleHypotheses(roles=[
            hypothesis(
                "ML Engineer",
                ["Automation", "Marketing Specialist"],
            )
        ])

        self.assertEqual(
            _validate_hypotheses(candidate_resume, generated),
            [],
        )

    def test_market_validated_roles_are_ranked_without_job_postings(self) -> None:
        generated = GeneratedRoleHypotheses(roles=[
            hypothesis(
                "Marketing Operations Specialist",
                [
                    "Automation",
                    "Настроил Marketing Automation в HubSpot",
                    "Автоматизировал подготовку кампаний и сократил время запуска на 20%",
                ],
            ),
            hypothesis(
                "Campaign Coordinator",
                ["Campaign execution", "Marketing Specialist"],
            ),
        ])
        calls: list[str] = []

        def builder(target: object, *, output_path: Path) -> MarketProfile:
            calls.append(target.specialization)
            self.assertEqual(target.core_skills, [])
            profile = market_profile(target)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
            return profile

        with tempfile.TemporaryDirectory() as directory:
            result = explore_alternative_roles(
                resume(),
                Path(directory),
                generated_hypotheses=generated,
                market_builder=builder,
                use_comparison_embeddings=False,
            )
            payload = json.loads(
                (Path(directory) / "alternative_roles.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(len(calls), 2)
        self.assertEqual(result.market_validated_candidates, 2)
        self.assertEqual(
            result.alternative_roles[0].target.specialization,
            "Marketing Operations Specialist",
        )
        self.assertGreater(
            result.alternative_roles[0].readiness_score or 0,
            result.alternative_roles[1].readiness_score or 0,
        )
        self.assertNotIn("sources", payload["alternative_roles"][0])
        self.assertNotIn("url", payload["alternative_roles"][0])

    def test_insufficient_market_data_keeps_evidence_and_returns_partial(
        self,
    ) -> None:
        generated = GeneratedRoleHypotheses(roles=[
            hypothesis(
                "Marketing Operations Specialist",
                ["Automation", "Marketing Specialist"],
            )
        ])

        def builder(
            target: object,
            *,
            output_path: Path,
        ) -> InsufficientMarketProfile:
            del output_path
            data = target.model_dump()
            return InsufficientMarketProfile(
                target=MarketTarget(**{
                    key: data.get(key)
                    for key in (
                        "role",
                        "specialization",
                        "domain",
                        "seniority",
                        "role_family",
                        "focus",
                    )
                }),
                sample_size=0,
                message="Недостаточно рынка",
                summary="Недостаточно рынка",
                sources=[],
            )

        with tempfile.TemporaryDirectory() as directory:
            result = explore_alternative_roles(
                resume(),
                Path(directory),
                generated_hypotheses=generated,
                market_builder=builder,
                use_comparison_embeddings=False,
            )

        self.assertEqual(result.status, "partial")
        self.assertEqual(result.market_validated_candidates, 0)
        self.assertEqual(
            result.alternative_roles[0].status,
            "market_data_unavailable",
        )
        self.assertGreaterEqual(len(result.alternative_roles[0].evidence), 2)

    def test_market_error_does_not_abort_alternative_branch(self) -> None:
        generated = GeneratedRoleHypotheses(roles=[
            hypothesis(
                "Marketing Operations Specialist",
                ["Automation", "Marketing Specialist"],
            )
        ])

        def failing_builder(target: object, *, output_path: Path) -> MarketProfile:
            del target, output_path
            raise TimeoutError("market timeout")

        with tempfile.TemporaryDirectory() as directory:
            result = explore_alternative_roles(
                resume(),
                Path(directory),
                generated_hypotheses=generated,
                market_builder=failing_builder,
                use_comparison_embeddings=False,
            )
            error_path = (
                Path(directory)
                / result.alternative_roles[0].audit_artifacts["market_error"]
            )
            self.assertTrue(error_path.exists())

        self.assertEqual(result.status, "partial")
        self.assertEqual(
            result.alternative_roles[0].status,
            "market_data_unavailable",
        )

    def test_generation_error_returns_safe_status(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "resume_evaluator.alternative_role_explorer.generate_role_hypotheses",
                side_effect=OSError("Ollama unavailable"),
            ),
        ):
            result = explore_alternative_roles(
                resume(),
                Path(directory),
                use_comparison_embeddings=False,
            )

        self.assertEqual(result.status, "generation_failed")
        self.assertEqual(result.alternative_roles, [])

    def test_role_without_market_matches_is_not_recommended(self) -> None:
        generated = GeneratedRoleHypotheses(roles=[
            hypothesis(
                "Data Engineer",
                ["Automation", "Marketing Specialist"],
            )
        ])

        def builder(target: object, *, output_path: Path) -> MarketProfile:
            profile = market_profile(target)
            source = profile.sources[0]
            profile.professional_skills = [
                MarketRequirement(
                    name="Apache Spark",
                    frequency=3,
                    coverage=1.0,
                    importance="common",
                    sources=profile.sources,
                    source_values=["Apache Spark"],
                )
            ]
            profile.tools_and_technologies = [
                MarketRequirement(
                    name="Kubernetes",
                    frequency=3,
                    coverage=1.0,
                    importance="common",
                    sources=[source, *profile.sources[1:]],
                    source_values=["Kubernetes"],
                )
            ]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
            return profile

        with tempfile.TemporaryDirectory() as directory:
            result = explore_alternative_roles(
                resume(),
                Path(directory),
                generated_hypotheses=generated,
                market_builder=builder,
                use_comparison_embeddings=False,
            )

        self.assertEqual(result.status, "no_market_supported_alternatives")
        self.assertEqual(result.market_rejected_candidates, 1)
        self.assertEqual(result.alternative_roles, [])


if __name__ == "__main__":
    unittest.main()
