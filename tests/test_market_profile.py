"""Тесты прикладного MarketProfile, независимого от данных кандидата."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from resume_evaluator.profession import TargetProfile
from resume_evaluator.resume_extractor import ResumeData
from resume_evaluator.synthetic_vacancy_pipeline_simplified import (
    ConceptGroup,
    ConceptGrouping,
    ConceptMember,
    EmbeddingCandidateDiagnostic,
    MarketProfile,
    NormalizedVacancy,
    ProfessionSearchPlan,
    RawVacancy,
    TripletMetrics,
    TripletSelection,
    _market_profile_plan,
    _build_extended_retrieval_diagnostics,
    aggregate_market_profile,
    build_market_profile,
    build_search_plan,
    embedding_cosine_similarity,
)


TARGET = TargetProfile(
    role="Engineer",
    specialization="ML Engineer",
    domain="Machine Learning / Artificial Intelligence",
    seniority="Middle",
    role_family="ML Engineering",
    focus="Model Development / Production ML",
    core_skills=[],
    aliases=["Machine Learning Engineer"],
    adjacent_specializations=["AI Engineer"],
    excluded_specializations=["MLOps Engineer"],
)


def vacancy(
    source_id: str,
    *,
    professional_skills: list[str],
    tools: list[str],
    responsibilities: list[str],
    soft_skills: list[str] | None = None,
    education: list[str] | None = None,
    domain_knowledge: list[str] | None = None,
) -> NormalizedVacancy:
    return NormalizedVacancy(
        source="test",
        source_id=source_id,
        url=f"https://jobs.test/{source_id}",
        title="Middle Machine Learning Engineer",
        employer=f"Company {source_id}",
        published_at="2026-09-20T10:00:00+00:00",
        profession=TARGET.model_dump(exclude={
            "aliases",
            "adjacent_specializations",
            "excluded_specializations",
        }),
        is_relevant=True,
        required_skills=[*professional_skills, *tools],
        preferred_skills=[],
        professional_skills=professional_skills,
        tools_and_technologies=tools,
        soft_skills=soft_skills or [],
        education_requirements=education or [],
        domain_knowledge=domain_knowledge or [],
        extraction_status="ok",
        professional_requirements_detected=True,
        responsibilities=responsibilities,
        experience_years_min=2,
        education_level=None,
        education_field=None,
        employment_type=None,
        work_format=None,
        conditions=[],
        embedding_role_similarity=0.9,
        embedding_specialization_similarity=0.9,
        embedding_domain_similarity=0.9,
        embedding_content_similarity=0.9,
        relevance_score=0.9,
    )


def sample_and_grouping() -> tuple[list[NormalizedVacancy], ConceptGrouping]:
    sample = [
        vacancy(
            "v1",
            professional_skills=["Machine Learning"],
            tools=["Python", "PyTorch"],
            responsibilities=["Разрабатывать и обучать ML-модели"],
            soft_skills=["Communication"],
            education=["MSc in Computer Science"],
            domain_knowledge=["AdTech"],
        ),
        vacancy(
            "v2",
            professional_skills=["Machine Learning"],
            tools=["Python", "PyTorch"],
            responsibilities=["Создавать и обучать модели машинного обучения"],
            soft_skills=["Communication"],
            education=["MSc in Computer Science"],
            domain_knowledge=["AdTech"],
        ),
        vacancy(
            "v3",
            professional_skills=["Machine Learning"],
            tools=["Python", "SQL"],
            responsibilities=["Разрабатывать ML-модели"],
        ),
        vacancy(
            "v4",
            professional_skills=["Machine Learning"],
            tools=["Python", "Docker"],
            responsibilities=["Интегрировать ML-модели в продукт"],
            education=["PhD in Computer Science"],
        ),
    ]
    grouping = ConceptGrouping(
        skill_groups=[],
        responsibility_groups=[
            ConceptGroup(
                canonical_text="Разработка и обучение ML-моделей",
                members=[
                    ConceptMember(
                        source_id="v1",
                        text="Разрабатывать и обучать ML-модели",
                        kind="responsibility",
                    ),
                    ConceptMember(
                        source_id="v2",
                        text="Создавать и обучать модели машинного обучения",
                        kind="responsibility",
                    ),
                    ConceptMember(
                        source_id="v3",
                        text="Разрабатывать ML-модели",
                        kind="responsibility",
                    ),
                ],
            ),
            ConceptGroup(
                canonical_text="Интеграция ML-моделей в продукт",
                members=[
                    ConceptMember(
                        source_id="v4",
                        text="Интегрировать ML-модели в продукт",
                        kind="responsibility",
                    )
                ],
            ),
        ],
    )
    return sample, grouping


class MarketProfileTests(unittest.TestCase):
    def test_embedding_cosine_similarity_clamps_float_rounding(self) -> None:
        vector = [0.1] * 10

        similarity = embedding_cosine_similarity(vector, vector)

        self.assertEqual(similarity, 1.0)

    def test_extended_diagnostics_counts_losses_without_changing_selection(
        self,
    ) -> None:
        plan = ProfessionSearchPlan(
            profession=TARGET,
            search_queries=[
                "ML Engineer",
                "Machine Learning Engineer",
                "Middle ML Engineer",
            ],
        )
        raw = [
            RawVacancy(
                source="test",
                source_id=f"v{index}",
                url=f"https://raw.test/v{index}",
                title=f"Candidate {index}",
                employer="Company",
                published_at=None,
                raw_text="Machine learning models",
            )
            for index in range(1, 5)
        ]

        def embedding_side_effect(vacancies, profession, **kwargs):
            del profession
            diagnostics = kwargs["diagnostic_rows"]
            reasons = [
                "semantic_only_without_lexical_support",
                None,
                "semantic_only_without_lexical_support",
                None,
            ]
            for index, (item, reason) in enumerate(
                zip(vacancies, reasons, strict=True),
                start=1,
            ):
                diagnostics.append(EmbeddingCandidateDiagnostic(
                    source_id=item.source_id,
                    source=item.source,
                    title=item.title,
                    url=item.url,
                    search_query=item.search_query,
                    target_specialization=TARGET.specialization,
                    target_domain=TARGET.domain,
                    prefilter_score=0.9 - index * 0.1,
                    dense_semantic_similarity=0.8,
                    specialization_similarity=0.8,
                    domain_similarity=0.6,
                    embedding_score=0.7,
                    semantic_only=reason is not None,
                    passed_embedding_filter=reason is None,
                    rejection_reason=reason,
                ))
            return [vacancies[1], vacancies[3]]

        normalized = [
            vacancy(
                source_id,
                professional_skills=["ML"],
                tools=["Python"],
                responsibilities=["Build models"],
            )
            for source_id in ("v1", "v2", "v4")
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline_simplified.prefilter_vacancies",
                    return_value=raw,
                ),
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline_simplified.embedding_filter_vacancies",
                    side_effect=embedding_side_effect,
                ),
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline_simplified.normalize_vacancies",
                    return_value=normalized,
                ),
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline_simplified.calculate_relevance_scores"
                ),
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline_simplified._eligible_market_vacancies",
                    return_value=normalized,
                ),
            ):
                report = _build_extended_retrieval_diagnostics(
                    plan,
                    raw,
                    raw[:3],
                    raw[:1],
                    checkpoint_path=root / "checkpoint.json",
                    cache_path=root / "cache.json",
                )

        self.assertEqual(report.candidates_evaluated, 4)
        self.assertEqual(report.diagnostic_normalized_candidates, 3)
        self.assertEqual(report.diagnostic_strict_matches, 3)
        self.assertEqual(report.strict_valid_lost_lexical_gate, 1)
        self.assertEqual(report.strict_valid_lost_top20_cutoff, 1)
        self.assertEqual(report.strict_valid_lost_search_cap, 1)
        self.assertIsNone(report.error)

    def test_example_json_matches_public_schema(self) -> None:
        example = (
            Path(__file__).resolve().parents[1]
            / "examples" / "precomputed"
            / "market_profile.example.json"
        )

        result = MarketProfile.model_validate_json(
            example.read_text(encoding="utf-8")
        )

        self.assertEqual(result.sample_size, 4)
        self.assertEqual(result.target.specialization, "ML Engineer")

    def test_aggregates_full_sample_by_category_and_coverage(self) -> None:
        sample, grouping = sample_and_grouping()

        result = aggregate_market_profile(TARGET, sample, grouping)

        self.assertEqual(result.sample_size, 4)
        machine_learning = result.professional_skills[0]
        self.assertEqual(machine_learning.frequency, 4)
        self.assertEqual(machine_learning.coverage, 1.0)
        self.assertEqual(machine_learning.importance, "common")

        tools = {item.name: item for item in result.tools_and_technologies}
        self.assertEqual(tools["Python"].importance, "common")
        self.assertEqual(tools["PyTorch"].importance, "additional")
        self.assertEqual(tools["PyTorch"].coverage, 0.5)
        self.assertNotIn("Docker", tools)

        variations = {
            (item.category, item.name) for item in result.market_variations
        }
        self.assertIn(("tools_and_technologies", "Docker"), variations)
        self.assertIn(("education_requirements", "PhD in Computer Science"), variations)
        self.assertIn(("responsibilities", "Интеграция ML-моделей в продукт"), variations)
        self.assertEqual(result.responsibilities[0].frequency, 3)
        self.assertEqual(result.responsibilities[0].coverage, 0.75)
        self.assertIn("ориентир по доступной выборке", result.summary)

    @patch(
        "resume_evaluator.synthetic_vacancy_pipeline_simplified.build_target_profile"
    )
    def test_same_target_ignores_different_candidate_skills(
        self,
        build_target_mock,
    ) -> None:
        plan = ProfessionSearchPlan(
            profession=TARGET,
            search_queries=[
                "ML Engineer",
                "Machine Learning Engineer",
                "Middle ML Engineer",
            ],
        )
        build_target_mock.return_value = plan
        base_resume = {
            "candidate_name": None,
            "target_profession": TARGET.model_dump(),
            "professional_summary": None,
            "contact_channels": [],
            "soft_skills": [],
            "work_experience": [],
            "education": [],
            "projects": [],
            "courses_and_certifications": [],
            "languages": [],
            "achievements": [],
            "portfolio_links": [],
        }
        candidate_a = ResumeData.model_validate({
            **base_resume,
            "skills": ["Python", "PyTorch", "Docker"],
        })
        candidate_b = ResumeData.model_validate({
            **base_resume,
            "skills": ["Java", "SQL", "TensorFlow"],
        })

        plan_a = build_search_plan(candidate_a)
        plan_b = build_search_plan(candidate_b)
        sample, grouping = sample_and_grouping()
        profile_a = aggregate_market_profile(plan_a.profession, sample, grouping)
        profile_b = aggregate_market_profile(plan_b.profession, sample, grouping)

        self.assertEqual(plan_a, plan_b)
        self.assertEqual(profile_a, profile_b)
        self.assertEqual(
            build_target_mock.call_args_list[0].args[0],
            build_target_mock.call_args_list[1].args[0],
        )
        candidate_plan_a = plan.model_copy(
            update={"search_queries": ["ML Engineer Python PyTorch", "Docker", "ML"]}
        )
        candidate_plan_b = plan.model_copy(
            update={"search_queries": ["ML Engineer Java SQL", "TensorFlow", "ML"]}
        )
        self.assertEqual(
            _market_profile_plan(candidate_plan_a),
            _market_profile_plan(candidate_plan_b),
        )

    def test_product_profile_does_not_require_triplet_confidence(self) -> None:
        sample, grouping = sample_and_grouping()
        plan = ProfessionSearchPlan(
            profession=TARGET,
            search_queries=[
                "ML Engineer",
                "Machine Learning Engineer",
                "Middle ML Engineer",
            ],
        )
        raw = [
            RawVacancy(
                source="test",
                source_id=f"raw:{index}",
                url=f"https://raw.test/{index}",
                title="Middle ML Engineer",
                employer="Company",
                published_at=None,
                raw_text="Machine learning models and Python",
            )
            for index in range(4)
        ]
        low_triplet = TripletSelection(
            vacancies=sample[:3],
            metrics=TripletMetrics(
                triplet_score=0.4,
                profession_relevance=0.6,
                domain_similarity=0.6,
                mutual_similarity=0.1,
            ),
            confidence="low_confidence",
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "market_profile.json"
            with (
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline_simplified._market_profile_plan",
                    return_value=plan,
                ),
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline_simplified.search_public_vacancies",
                    return_value=(raw, []),
                ),
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline_simplified.prefilter_vacancies",
                    return_value=raw,
                ),
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline_simplified.embedding_filter_vacancies",
                    return_value=raw,
                ),
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline_simplified.normalize_vacancies",
                    return_value=sample,
                ),
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline_simplified.calculate_relevance_scores"
                ),
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline_simplified._eligible_market_vacancies",
                    return_value=sample,
                ),
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline_simplified.group_selected_concepts",
                    return_value=grouping,
                ),
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline_simplified._save_retrieval_diagnostics"
                ),
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline_simplified.evaluate_best_triplet",
                    return_value=low_triplet,
                ),
            ):
                result = build_market_profile(plan, output)

            self.assertEqual(result.status, "ok")
            self.assertEqual(result.sample_size, 4)
            self.assertTrue(output.is_file())
            self.assertTrue((Path(directory) / "market_profile.debug.json").is_file())

    def test_two_embedding_candidates_are_normalized_before_insufficient(self) -> None:
        sample, _ = sample_and_grouping()
        plan = ProfessionSearchPlan(
            profession=TARGET,
            search_queries=[
                "ML Engineer",
                "Machine Learning Engineer",
                "Middle ML Engineer",
            ],
        )
        raw = [
            RawVacancy(
                source="test",
                source_id=f"v{index}",
                url=f"https://raw.test/v{index}",
                title="Middle Machine Learning Engineer",
                employer="Company",
                published_at=None,
                raw_text="Machine learning models and Python",
            )
            for index in range(1, 3)
        ]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "market_profile.json"
            with (
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline_simplified._market_profile_plan",
                    return_value=plan,
                ),
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline_simplified.search_public_vacancies",
                    return_value=(raw, []),
                ),
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline_simplified.prefilter_vacancies",
                    return_value=raw,
                ),
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline_simplified.embedding_filter_vacancies",
                    return_value=raw,
                ),
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline_simplified.normalize_vacancies",
                    return_value=sample[:2],
                ) as normalize_mock,
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline_simplified.calculate_relevance_scores"
                ),
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline_simplified._eligible_market_vacancies",
                    return_value=sample[:2],
                ),
            ):
                result = build_market_profile(plan, output, save_debug=False)

            self.assertEqual(result.status, "insufficient_market_data")
            self.assertEqual(result.sample_size, 2)
            normalize_mock.assert_called_once()
            self.assertEqual(len(normalize_mock.call_args.args[0]), 2)

    def test_borderline_candidates_run_strict_validation_after_short_sample(
        self,
    ) -> None:
        sample, _ = sample_and_grouping()
        plan = ProfessionSearchPlan(
            profession=TARGET,
            search_queries=[
                "ML Engineer",
                "Machine Learning Engineer",
                "Middle ML Engineer",
            ],
        )
        raw = [
            RawVacancy(
                source="test",
                source_id=f"v{index}",
                url=f"https://raw.test/v{index}",
                title="Middle Machine Learning Engineer",
                employer="Company",
                published_at=None,
                raw_text="Machine learning models and Python",
            )
            for index in range(1, 4)
        ]

        def embedding_side_effect(vacancies, profession, **kwargs):
            diagnostics = kwargs["diagnostic_rows"]
            diagnostics.extend([
                EmbeddingCandidateDiagnostic(
                    source_id=item.source_id,
                    source=item.source,
                    title=item.title,
                    url=item.url,
                    search_query=None,
                    target_specialization=TARGET.specialization,
                    target_domain=TARGET.domain,
                    prefilter_score=0.8,
                    token_overlap=1.0,
                    fuzzy_lexical_similarity=0.9,
                    dense_semantic_similarity=0.9,
                    title_match_score=0.9,
                    specialization_similarity=0.9,
                    domain_similarity=0.40 if index == 0 else 0.34,
                    embedding_score=0.9 - index * 0.05,
                    passed_embedding_filter=index == 0,
                    rejection_reason=(
                        None
                        if index == 0
                        else "domain_similarity_below_threshold"
                    ),
                )
                for index, item in enumerate(vacancies)
            ])
            return vacancies[:1]

        def normalize_side_effect(vacancies, profession, **kwargs):
            return sample[:len(vacancies)]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "market_profile.json"
            diagnostics_path = (
                Path(directory) / "market_profile.retrieval_diagnostics.json"
            )
            with (
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline_simplified._market_profile_plan",
                    return_value=plan,
                ),
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline_simplified.search_public_vacancies",
                    return_value=(raw, []),
                ),
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline_simplified.prefilter_vacancies",
                    return_value=raw,
                ),
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline_simplified.embedding_filter_vacancies",
                    side_effect=embedding_side_effect,
                ),
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline_simplified.normalize_vacancies",
                    side_effect=normalize_side_effect,
                ) as normalize_mock,
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline_simplified.calculate_relevance_scores"
                ),
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline_simplified._eligible_market_vacancies",
                    side_effect=lambda vacancies, profession, limit=None: vacancies,
                ),
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline_simplified.group_selected_concepts",
                    return_value=ConceptGrouping(
                        skill_groups=[],
                        responsibility_groups=[],
                    ),
                ),
            ):
                result = build_market_profile(plan, output, save_debug=False)

            self.assertEqual(result.status, "ok")
            self.assertEqual(result.sample_size, 3)
            self.assertEqual(normalize_mock.call_count, 2)
            self.assertEqual(len(normalize_mock.call_args_list[0].args[0]), 1)
            self.assertEqual(len(normalize_mock.call_args_list[1].args[0]), 3)
            diagnostics = diagnostics_path.read_text(encoding="utf-8")
            self.assertIn('"borderline_candidates": 2', diagnostics)


if __name__ == "__main__":
    unittest.main()
