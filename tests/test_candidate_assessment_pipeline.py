"""Тесты единого отчёта без повторного расчёта стабильных этапов."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from resume_evaluator.alternative_role_explorer import (
    AlternativeRoleAssessment,
    AlternativeRoleEvidence,
    AlternativeRoleExplorerResult,
)
from resume_evaluator.candidate_assessment_pipeline import (
    CandidateReport,
    _load_resume,
    _load_market_profile,
    run_candidate_assessment,
    render_candidate_report_markdown,
)
from resume_evaluator.resume_extractor import ResumeData
from resume_evaluator.synthetic_vacancy_pipeline_simplified import (
    AdjacentMarketContext,
    InsufficientMarketProfile,
    MarketProfile,
    MarketRequirement,
    MarketRequirementEvidence,
    MarketSource,
    MarketTarget,
)

ROOT = Path(__file__).resolve().parents[1]


def profile() -> MarketProfile:
    target = MarketTarget(
        role="Marketer",
        specialization="Маркетолог",
        domain="Marketing",
        seniority="Middle",
        role_family="Marketing",
        focus="Campaign Management",
    )
    sources = [
        MarketSource(
            source="test",
            source_id=f"v{index}",
            title="Middle Marketer",
            employer=f"Company {index}",
            url=f"https://jobs.test/{index}",
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
        target=target,
        sample_size=3,
        professional_skills=[
            requirement("Automation"),
            requirement("Project Management", "additional"),
        ],
        tools_and_technologies=[
            requirement("HubSpot"),
            requirement("Salesforce", "additional"),
        ],
        responsibilities=[requirement("Execute regional campaigns")],
        education_requirements=[],
        domain_knowledge=[],
        soft_skills=[],
        market_variations=[],
        sources=sources,
        confidence="medium",
        summary="Профиль построен по трём тестовым вакансиям.",
    )


class CandidateAssessmentPipelineTests(unittest.TestCase):
    def test_legacy_v2_medium_is_lowered_using_matching_low_diagnostics(self) -> None:
        ready = profile()
        diagnostics = {
            "namespace": "market_profile_v2", "confidence": "low",
            "target": ready.target.model_dump(), "final_sample_size": ready.sample_size,
            "final_vacancies": [{"source_id": row.source_id} for row in ready.sources],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            sidecar = path.with_name("profile.diagnostics.json")
            path.write_text(ready.model_dump_json(), encoding="utf-8")
            sidecar.write_text(json.dumps(diagnostics), encoding="utf-8")
            self.assertEqual(_load_market_profile(path).confidence, "low")
            diagnostics["final_vacancies"] = [{"source_id": "other"}]
            sidecar.write_text(json.dumps(diagnostics), encoding="utf-8")
            self.assertEqual(_load_market_profile(path).confidence, "medium")

    def test_low_confidence_and_requirement_evidence_reach_readable_report(self) -> None:
        ready = profile()
        ready.confidence = "low"
        from test_market_reliability import assess, good_sample
        ready.reliability = assess(good_sample(3))
        ready.professional_skills[0].evidence = [
            MarketRequirementEvidence(source_id="v0", text="Automation"),
        ]
        # Валидируем новые поля после подготовки тестового JSON.
        ready = MarketProfile.model_validate(ready.model_dump())
        with tempfile.TemporaryDirectory() as directory:
            with patch("resume_evaluator.candidate_assessment_pipeline.build_market_profile_v2", return_value=ready):
                paths = run_candidate_assessment(
                    resume_path=ROOT / "examples" / "precomputed" / "marketing_candidate.example.json",
                    target_profession="Маркетолог / Middle", output_directory=Path(directory) / "report",
                    resume_evaluation_path=ROOT / "examples" / "precomputed" / "marketing_resume_evaluation.example.json",
                    use_market_v2=True, use_comparison_embeddings=False, use_recommendation_llm=False,
                    detailed_report=True,
                )
            report = CandidateReport.model_validate_json(paths.report_json.read_text(encoding="utf-8"))
            stage = json.loads((paths.stages_directory / "recommendations.json").read_text(encoding="utf-8"))
            markdown = paths.report_markdown.read_text(encoding="utf-8")
        self.assertEqual(report.market_context.confidence, "low")
        self.assertEqual(stage["market_context"]["confidence"], "low")
        self.assertEqual(report.market_context.reliability.independent_employers, 3)
        item = next(row for row in report.recommendations.strengths if row.support[0].reference == "Automation")
        self.assertEqual(item.confidence, "low")
        self.assertEqual(item.action.type, "highlight_evidence")
        self.assertEqual(item.support[0].market_evidence[0].text, "Automation")
        self.assertIn("в 3 из 3 вакансий", markdown)
        self.assertIn("Требование: «Automation»", markdown)
        self.assertIn("https://jobs.test/0", markdown)
        self.assertIn("Уверенность в совете:** низкая", markdown)
        self.assertIn("Независимых работодателей: 3", markdown)
        self.assertIn("Результат:", markdown)

    def test_adjacent_context_is_separate_in_json_markdown_and_gap_metrics(self) -> None:
        ready = profile()
        adjacent_source = ready.sources[0].model_copy(update={
            "source_id": "adjacent-1", "title": "Web Analyst", "url": "https://jobs.test/adjacent-1",
        })
        ready.adjacent_context = AdjacentMarketContext(
            sample_size=1, sources=[adjacent_source], summary="Контекст смежных вакансий.",
            market_variations=[{
                "name": "Tableau", "category": "tools_and_technologies", "frequency": 1,
                "coverage": 1, "source_values": ["Tableau"], "sources": [adjacent_source],
            }],
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch("resume_evaluator.candidate_assessment_pipeline.build_market_profile_v2", return_value=ready):
                paths = run_candidate_assessment(
                    resume_path=ROOT / "examples" / "precomputed" / "marketing_candidate.example.json",
                    target_profession="Маркетолог / Middle", output_directory=Path(directory) / "report",
                    resume_evaluation_path=ROOT / "examples" / "precomputed" / "marketing_resume_evaluation.example.json",
                    use_market_v2=True, use_comparison_embeddings=False, use_recommendation_llm=False,
                    detailed_report=True,
                )
            report = CandidateReport.model_validate_json(paths.report_json.read_text(encoding="utf-8"))
            markdown = paths.report_markdown.read_text(encoding="utf-8")
        self.assertEqual(report.market_context.adjacent_context.sample_size, 1)
        self.assertEqual(report.market_context.sample_size, 3)
        self.assertIn("## Смежные вакансии: контекст возможного развития", markdown)
        self.assertIn("Tableau: 1/1 смежных вакансий", markdown)
        self.assertIn("https://jobs.test/adjacent-1", markdown)
        self.assertNotIn("Tableau", json.dumps(report.market_comparison.model_dump()))
        self.assertNotIn("Tableau", json.dumps(report.recommendations.model_dump()))

    def test_insufficient_v2_profile_skips_candidate_market_comparison(self) -> None:
        ready_profile = profile()
        insufficient = InsufficientMarketProfile(
            target=ready_profile.target,
            sample_size=ready_profile.sample_size,
            message="No repeated requirements.",
            summary="No repeated requirements.",
            sources=ready_profile.sources,
        )
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch(
                    "resume_evaluator.candidate_assessment_pipeline.build_market_profile_v2",
                    return_value=insufficient,
                ),
                patch(
                    "resume_evaluator.candidate_assessment_pipeline.compare_candidate_to_market",
                    side_effect=AssertionError("comparison must not run"),
                ) as comparison,
            ):
                paths = run_candidate_assessment(
                    resume_path=ROOT / "examples" / "precomputed" / "marketing_candidate.example.json",
                    target_profession="Маркетолог / Middle",
                    output_directory=Path(directory) / "report",
                    resume_evaluation_path=(
                        ROOT / "examples" / "precomputed" / "marketing_resume_evaluation.example.json"
                    ),
                    use_market_v2=True,
                    use_comparison_embeddings=False,
                    use_recommendation_llm=False,
                )
            comparison.assert_not_called()
            report = CandidateReport.model_validate_json(
                paths.report_json.read_text(encoding="utf-8")
            )
            self.assertEqual(report.status, "partial")
            self.assertEqual(
                report.market_context.status,
                "insufficient_market_data",
            )
            self.assertEqual(
                report.market_comparison.status,
                "market_data_unavailable",
            )
            self.assertEqual(report.market_comparison.candidate_only_skills, [])

    def test_market_v2_is_opt_in_and_v1_remains_default(self) -> None:
        resume_path = ROOT / "examples" / "precomputed" / "marketing_candidate.example.json"
        evaluation_path = (
            ROOT / "examples" / "precomputed" / "marketing_resume_evaluation.example.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch(
                    "resume_evaluator.candidate_assessment_pipeline.build_market_profile",
                    return_value=profile(),
                ) as v1_builder,
                patch(
                    "resume_evaluator.candidate_assessment_pipeline.build_market_profile_v2",
                    side_effect=AssertionError("v2 must be opt-in"),
                ) as v2_builder,
            ):
                run_candidate_assessment(
                    resume_path=resume_path,
                    target_profession="Маркетолог / Middle",
                    output_directory=Path(directory) / "v1",
                    resume_evaluation_path=evaluation_path,
                    use_comparison_embeddings=False,
                    use_recommendation_llm=False,
                )
            v1_builder.assert_called_once()
            v2_builder.assert_not_called()

            with (
                patch(
                    "resume_evaluator.candidate_assessment_pipeline.build_market_profile",
                    side_effect=AssertionError("v1 must not run with --market-v2"),
                ) as v1_builder,
                patch(
                    "resume_evaluator.candidate_assessment_pipeline.build_market_profile_v2",
                    return_value=profile(),
                ) as v2_builder,
            ):
                paths = run_candidate_assessment(
                    resume_path=resume_path,
                    target_profession="Маркетолог / Middle",
                    output_directory=Path(directory) / "v2",
                    resume_evaluation_path=evaluation_path,
                    use_market_v2=True,
                    use_comparison_embeddings=False,
                    use_recommendation_llm=False,
                )
            v1_builder.assert_not_called()
            v2_builder.assert_called_once()
            report = CandidateReport.model_validate_json(
                paths.report_json.read_text(encoding="utf-8")
            )
            self.assertEqual(report.status, "ok")

    def test_ready_results_are_reused_and_compact_report_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report"
            profile_path = Path(directory) / "market_profile.json"
            profile_path.write_text(
                profile().model_dump_json(indent=2),
                encoding="utf-8",
            )

            with (
                patch(
                    "resume_evaluator.candidate_assessment_pipeline.evaluate_resume",
                    side_effect=AssertionError("resume evaluation was recalculated"),
                ),
                patch(
                    "resume_evaluator.candidate_assessment_pipeline.build_market_profile",
                    side_effect=AssertionError("market profile was rebuilt"),
                ),
                patch(
                    "resume_evaluator.candidate_assessment_pipeline.build_market_profile_v2",
                    side_effect=AssertionError("market profile v2 was rebuilt"),
                ),
            ):
                paths = run_candidate_assessment(
                    resume_path=ROOT / "examples" / "precomputed" / "marketing_candidate.example.json",
                    target_profession="Маркетолог / Middle",
                    output_directory=output,
                    resume_evaluation_path=(
                        ROOT / "examples" / "precomputed" / "marketing_resume_evaluation.example.json"
                    ),
                    market_profile_path=profile_path,
                    use_market_v2=True,
                    use_comparison_embeddings=False,
                    use_recommendation_llm=False,
                )

            report = CandidateReport.model_validate_json(
                paths.report_json.read_text(encoding="utf-8")
            )
            markdown = paths.report_markdown.read_text(encoding="utf-8")

            self.assertEqual(report.status, "ok")
            self.assertEqual(report.resume_evaluation.score, 51)
            self.assertEqual(report.target.specialization, "Маркетолог")
            self.assertEqual(report.recommendations.generation_mode, "fallback")
            self.assertIn("Automation", {
                item.name for item in report.market_comparison.strong_matches
            })
            self.assertIn("Project Management", {
                item.name for item in report.market_comparison.missing_additional
            })
            self.assertTrue((paths.stages_directory / "resume.json").exists())
            self.assertIn("# Отчёт по резюме кандидата", markdown)
            self.assertIn("## Рыночный контекст", markdown)
            detailed = render_candidate_report_markdown(report, detailed=True)
            self.assertLess(len(markdown), len(detailed) * 0.55)
            self.assertEqual(markdown.count("https://jobs.test/0"), 1)
            self.assertIn("Salesforce", markdown)
            self.assertNotIn("**Результат:**", markdown)
            self.assertEqual({item.name for item in report.market_comparison.unconfirmed_resume_skills},
                             {"Salesforce", "Adobe Illustrator"})

            raw_report = json.loads(paths.report_json.read_text(encoding="utf-8"))
            self.assertNotIn("market_variations", raw_report["market_comparison"])
            self.assertNotIn("raw_text", paths.report_json.read_text(encoding="utf-8"))
            market_item = next(item for item in report.recommendations.strengths
                               if item.support[0].reference == "Automation")
            self.assertEqual(market_item.support[0].market_source_values, ["Automation"])
            self.assertEqual(market_item.support[0].market_frequency, 3)
            self.assertEqual(len(market_item.support[0].market_sources), 3)

    def test_optional_alternative_role_branch_is_added_to_report(self) -> None:
        candidate = ResumeData.model_validate_json(
            (ROOT / "examples" / "precomputed" / "marketing_candidate.example.json").read_text(
                encoding="utf-8"
            )
        )
        alternative_result = AlternativeRoleExplorerResult(
            status="ok",
            candidate_name=candidate.candidate_name,
            primary_target=candidate.target_profession,
            generation_mode="provided",
            generated_candidates=1,
            evidence_valid_candidates=1,
            market_validated_candidates=1,
            alternative_roles=[
                AlternativeRoleAssessment(
                    rank=1,
                    target=MarketTarget(
                        role="Specialist",
                        specialization="Marketing Operations Specialist",
                        domain="Marketing Operations",
                        seniority="Middle",
                        role_family="Marketing",
                        focus="Campaign Operations",
                    ),
                    status="validated",
                    confidence="medium",
                    evidence=[
                        AlternativeRoleEvidence(
                            section="skills",
                            text="Automation",
                        ),
                        AlternativeRoleEvidence(
                            section="experience",
                            text="Настроил Marketing Automation в HubSpot",
                        ),
                    ],
                    preliminary_score=0.7,
                    readiness_score=0.64,
                    market_sample_size=3,
                    common_requirements_matched=2,
                    common_requirements_total=3,
                    additional_requirements_matched=1,
                    additional_requirements_total=2,
                    matched_requirements=["Automation", "HubSpot"],
                    transition_gaps=["Analytics"],
                    explanation="Роль проверена по рынку.",
                    audit_artifacts={},
                )
            ],
            message="Проверено одно направление.",
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report"
            profile_path = Path(directory) / "market_profile.json"
            profile_path.write_text(profile().model_dump_json(indent=2), encoding="utf-8")
            with patch(
                "resume_evaluator.candidate_assessment_pipeline.explore_alternative_roles",
                return_value=alternative_result,
            ) as explorer:
                paths = run_candidate_assessment(
                    resume_path=ROOT / "examples" / "precomputed" / "marketing_candidate.example.json",
                    target_profession="Маркетолог / Middle",
                    output_directory=output,
                    resume_evaluation_path=(
                        ROOT / "examples" / "precomputed" / "marketing_resume_evaluation.example.json"
                    ),
                    market_profile_path=profile_path,
                    use_comparison_embeddings=False,
                    use_recommendation_llm=False,
                    include_alternative_roles=True,
                    alternative_role_limit=1,
                )
            report = CandidateReport.model_validate_json(
                paths.report_json.read_text(encoding="utf-8")
            )
            markdown = paths.report_markdown.read_text(encoding="utf-8")

        explorer.assert_called_once()
        self.assertIsNotNone(report.alternative_roles)
        self.assertEqual(
            report.alternative_roles.roles[0].target.specialization,
            "Marketing Operations Specialist",
        )
        self.assertIn("Альтернативные профессиональные направления", markdown)

    def test_pdf_input_uses_existing_reader_and_extractor(self) -> None:
        expected = ResumeData.model_validate_json(
            (ROOT / "examples" / "precomputed" / "marketing_candidate.example.json").read_text(
                encoding="utf-8"
            )
        )
        with (
            patch(
                "resume_evaluator.candidate_assessment_pipeline.read_resume",
                return_value="resume text",
            ) as reader,
            patch(
                "resume_evaluator.candidate_assessment_pipeline.extract_resume",
                return_value=expected,
            ) as extractor,
        ):
            actual, source_text = _load_resume(Path("resume.pdf"))

        self.assertEqual(actual, expected)
        self.assertEqual(source_text, "resume text")
        reader.assert_called_once_with(Path("resume.pdf"))
        extractor.assert_called_once_with("resume text")


if __name__ == "__main__":
    unittest.main()
