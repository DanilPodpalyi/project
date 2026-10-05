"""Тесты построения синтетической эталонной вакансии."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from resume_evaluator.candidate_scorer import score_candidate
from resume_evaluator.profession import TargetProfile, profession_from_value
from resume_evaluator.resume_extractor import ResumeData
from resume_evaluator.synthetic_vacancy_pipeline import (
    ConceptGroup,
    ConceptGrouping,
    ConceptMember,
    EmbeddingCandidateDiagnostic,
    ExtractedVacancy,
    NormalizedVacancy,
    PipelineStatistics,
    ProfessionSearchPlan,
    RawVacancy,
    TripletMetrics,
    TripletSelection,
    _normalize_batch_with_fallback,
    _focus_compatibility,
    _job_board_query,
    _eligible_market_vacancies,
    _expand_search_queries,
    _quality_checked_extraction,
    _set_extraction_quality,
    _tighten_responsibility_groups,
    aggregate_synthetic_vacancy,
    benchmark_title_reranker,
    build_search_plan,
    build_target_profile,
    build_skill_groups_python,
    calculate_relevance_scores,
    deduplicate_vacancies,
    embedding_filter_vacancies,
    evaluate_best_triplet,
    group_selected_concepts,
    normalize_skill_name,
    normalize_vacancies,
    prefilter_vacancies,
    select_best_three,
    title_match_signals,
    run_pipeline,
    validate_and_complete_groups,
    vacancy_similarity,
)
from resume_evaluator.vacancy_extractor import VacancyData


def make_vacancy(
    source_id: str,
    *,
    profession: str = "Python-разработчик",
    relevant: bool = True,
    required: list[str] | None = None,
    preferred: list[str] | None = None,
    responsibilities: list[str] | None = None,
    experience: float | None = None,
) -> NormalizedVacancy:
    return NormalizedVacancy(
        source_id=source_id,
        canonical_profession=profession,
        is_relevant=relevant,
        required_skills=required or [],
        preferred_skills=preferred or [],
        responsibilities=responsibilities or [],
        experience_years_min=experience,
        education_level=None,
        education_field=None,
        employment_type="Полная занятость",
        work_format="Удалённый",
        conditions=[],
        source="test",
        url=f"https://example.test/{source_id}",
        title=profession,
        employer=f"Компания {source_id}",
        published_at="2026-08-20T10:00:00+00:00",
    )


class SyntheticVacancyPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.selected = [
            make_vacancy(
                "v1",
                required=["Python", "Postgres"],
                responsibilities=["Разрабатывать API", "Проверять код"],
                experience=2,
            ),
            make_vacancy(
                "v2",
                required=["Python", "PostgreSQL"],
                preferred=["Docker"],
                responsibilities=["Разработка REST API", "Code review"],
                experience=3,
            ),
            make_vacancy(
                "v3",
                required=["Python", "Docker", "Git"],
                responsibilities=["Создавать API", "Мониторинг сервисов"],
                experience=5,
            ),
        ]

    @patch("resume_evaluator.synthetic_vacancy_pipeline.chat")
    def test_search_plan_uses_only_target_profile_and_expands_queries_generically(
        self,
        chat_mock: MagicMock,
    ) -> None:
        target = {
            "role": "Engineer",
            "specialization": "ML Engineer",
            "domain": "Machine Learning / Artificial Intelligence",
            "seniority": "Middle",
            "core_skills": ["Python"],
        }
        resume = ResumeData.model_validate(
            {
                "candidate_name": None,
                "target_profession": target,
                "professional_summary": None,
                "contact_channels": [],
                "skills": ["PyTorch"],
                "soft_skills": [],
                "work_experience": [],
                "education": [],
                "projects": [],
                "courses_and_certifications": [],
                "languages": [],
                "achievements": [],
                "portfolio_links": [],
            }
        )
        chat_mock.return_value = MagicMock(
            message=MagicMock(
                content=json.dumps(
                    {
                        "profession": {
                            **target,
                            "role_family": "ML Engineering",
                            "focus": "Model Development / Production ML",
                            "aliases": ["Machine Learning Engineer"],
                            "adjacent_specializations": ["AI Engineer"],
                            "excluded_specializations": ["AI Infrastructure Engineer"],
                        },
                        "search_queries": [
                            "ML Engineer",
                            "Machine Learning Engineer",
                            "Инженер ML",
                        ],
                    },
                    ensure_ascii=False,
                )
            )
        )

        plan = build_search_plan(resume)

        payload = json.loads(chat_mock.call_args.kwargs["messages"][1]["content"])
        self.assertEqual(set(payload), {"target_profession"})
        self.assertEqual(payload["target_profession"]["core_skills"], [])
        self.assertEqual(plan.profession.core_skills, [])
        self.assertIn("Machine Learning Engineer", plan.search_queries)
        self.assertIn('"ML Engineer" vacancy', plan.search_queries)
        self.assertIn('"Machine Learning Engineer" vacancy', plan.search_queries)
        self.assertIn('"ML Engineer" job', plan.search_queries)
        self.assertIn(
            "Middle ML Engineer Machine Learning / Artificial Intelligence",
            plan.search_queries,
        )
        self.assertEqual(plan.profession.role_family, "ML Engineering")
        self.assertEqual(plan.profession.aliases, ["Machine Learning Engineer"])
        self.assertLessEqual(len(plan.search_queries), 8)

    def test_middle_prefilter_excludes_explicit_senior_titles(self) -> None:
        target = {
            "role": "Engineer",
            "specialization": "ML Engineer",
            "domain": "Machine Learning",
            "seniority": "Middle",
            "core_skills": [],
        }
        titles = [
            "Middle ML Engineer",
            "Machine Learning Engineer",
            "Senior ML Engineer",
            "Lead ML Engineer",
            "Staff Machine Learning Engineer",
            "Principal ML Engineer",
            "Head of Machine Learning Engineering",
        ]
        raw = [self._raw(str(index), title, title) for index, title in enumerate(titles)]

        filtered = prefilter_vacancies(raw, target, ["ML Engineer"])

        self.assertEqual(
            {item.title for item in filtered},
            {"Middle ML Engineer", "Machine Learning Engineer"},
        )

    def test_job_board_query_strips_unsupported_web_phrase_operator(self) -> None:
        self.assertEqual(_job_board_query('"ML Engineer" vacancy'), "ML Engineer")
        self.assertEqual(_job_board_query('"Reporter" job'), "Reporter")
        self.assertEqual(
            _job_board_query("Middle ML Engineer Machine Learning"),
            "Middle ML Engineer Machine Learning",
        )

    @patch("resume_evaluator.synthetic_vacancy_pipeline.chat")
    def test_target_profile_can_be_built_from_direct_profession_query(
        self,
        chat_mock: MagicMock,
    ) -> None:
        chat_mock.return_value = MagicMock(
            message=MagicMock(
                content=json.dumps(
                    {
                        "profession": {
                            "role": "Journalist",
                            "specialization": "Корреспондент",
                            "role_family": "Journalism",
                            "domain": "Media",
                            "focus": "News Reporting",
                            "seniority": None,
                            "aliases": ["Correspondent", "Reporter"],
                            "adjacent_specializations": ["News Journalist"],
                            "excluded_specializations": ["PR Manager"],
                            "core_skills": [],
                        },
                        "search_queries": [
                            "Корреспондент",
                            "Correspondent",
                            "News Reporter",
                        ],
                    },
                    ensure_ascii=False,
                )
            )
        )

        plan = build_target_profile("Корреспондент")

        self.assertEqual(plan.profession.role, "Journalist")
        self.assertEqual(plan.profession.specialization, "Корреспондент")
        self.assertEqual(plan.profession.role_family, "Journalism")
        self.assertIn("Reporter", plan.profession.aliases)
        self.assertIn("PR Manager", plan.profession.excluded_specializations)

    def test_market_relevance_ignores_candidate_skills_and_vacancy_skills(self) -> None:
        target = {
            "role": "Engineer",
            "specialization": "ML Engineer",
            "domain": "Machine Learning",
            "seniority": "Middle",
            "core_skills": ["Python"],
        }
        first = make_vacancy("first", profession="ML Engineer", required=["Python"])
        second = make_vacancy("second", profession="ML Engineer", required=["Rust"])
        for item in (first, second):
            item.embedding_role_similarity = 0.8
            item.embedding_specialization_similarity = 0.8
            item.embedding_domain_similarity = 0.8
            item.embedding_content_similarity = 0.8
            item.profession.seniority = "Middle"

        calculate_relevance_scores([first], target, ["Python", "PyTorch"])
        calculate_relevance_scores([second], target, ["Rust", "Go"])

        self.assertEqual(first.relevance_score, second.relevance_score)

    def test_middle_selection_rejects_senior_and_weak_adjacent_ai(self) -> None:
        target = {
            "role": "Engineer",
            "specialization": "ML Engineer",
            "domain": "Machine Learning",
            "seniority": "Middle",
            "core_skills": [],
        }
        vacancies = [
            make_vacancy("ml-middle", profession="ML Engineer"),
            make_vacancy("ml-unknown", profession="ML Engineer"),
            make_vacancy("ai-adjacent", profession="AI Engineer"),
            make_vacancy("ml-senior", profession="ML Engineer"),
        ]
        vacancies[0].profession.seniority = "Middle"
        vacancies[2].profession.seniority = "Middle"
        vacancies[3].profession.seniority = "Senior"
        vacancies[3].title = "Senior ML Engineer"
        for vacancy in vacancies:
            vacancy.embedding_role_similarity = 0.8
            vacancy.embedding_specialization_similarity = 0.8
            vacancy.embedding_domain_similarity = 0.8
            vacancy.embedding_content_similarity = 0.8
        vacancies[2].embedding_specialization_similarity = 0.68

        calculate_relevance_scores(vacancies, target)
        self.assertLess(vacancies[1].relevance_score, vacancies[0].relevance_score)
        self.assertIsNone(evaluate_best_triplet(vacancies, target))

        vacancies[2].embedding_specialization_similarity = 0.8
        calculate_relevance_scores(vacancies, target)
        selection = evaluate_best_triplet(vacancies, target)

        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual(
            {item.source_id for item in selection.vacancies},
            {"ml-middle", "ml-unknown", "ai-adjacent"},
        )
        self.assertIn(selection.confidence, {"medium", "high"})

    def test_deduplicates_by_url_and_title_with_employer(self) -> None:
        vacancies = [
            RawVacancy(
                source="one",
                source_id="1",
                url="https://jobs.test/1",
                title="Python Developer",
                employer="ACME",
                published_at=None,
                raw_text="First",
            ),
            RawVacancy(
                source="two",
                source_id="2",
                url="https://jobs.test/1/",
                title="Other title",
                employer="Other",
                published_at=None,
                raw_text="Duplicate URL",
            ),
            RawVacancy(
                source="three",
                source_id="3",
                url="https://jobs.test/3",
                title="python developer",
                employer="acme",
                published_at=None,
                raw_text="Duplicate content",
            ),
        ]

        result = deduplicate_vacancies(vacancies)

        self.assertEqual([item.source_id for item in result], ["1"])

    def test_prefilter_rejects_matching_words_from_other_domains(self) -> None:
        vacancies = [
            self._raw(
                "digital",
                "Traffic Analyst",
                "Digital marketing, web analytics, Google Analytics and PPC",
            ),
            self._raw(
                "cyber",
                "Network Traffic Analyst",
                "Incident response, network security, SIEM and digital forensics",
            ),
            self._raw(
                "transport",
                "Traffic Analyst",
                "Road traffic and transport planning",
            ),
            self._raw(
                "finance-security",
                "Security Analyst",
                "Financial analysis, investment banking and credit risk",
            ),
        ]
        profession = {
            "role": "Analyst",
            "specialization": "Traffic Analyst",
            "domain": "Digital Marketing / Web Analytics",
            "seniority": "Middle",
            "core_skills": ["Google Analytics", "PPC"],
            "aliases": ["Web Traffic Analyst"],
            "adjacent_specializations": ["Digital Marketing Analyst"],
            "excluded_specializations": [
                "Network Traffic",
                "Road Traffic",
                "Security Analyst",
            ],
        }

        result = prefilter_vacancies(
            vacancies,
            profession,
            ["Traffic Analyst", "Web Analytics", "Digital Marketing Analyst"],
        )

        self.assertEqual([item.source_id for item in result], ["digital"])

    def test_prefilter_limits_llm_input_to_twenty(self) -> None:
        vacancies = [
            self._raw(
                str(index),
                f"Digital Marketing Analyst {index}",
                "Web analytics and Google Analytics",
            )
            for index in range(30)
        ]

        result = prefilter_vacancies(
            vacancies,
            "Digital Marketing Analyst",
            ["Digital Marketing Analyst", "Web Analyst", "Marketing Analytics"],
        )

        self.assertEqual(len(result), 20)

    @patch("resume_evaluator.synthetic_vacancy_pipeline._embed_text_map")
    def test_embedding_filter_enforces_domain_threshold(
        self,
        embed_mock: MagicMock,
    ) -> None:
        digital = self._raw(
            "digital",
            "Traffic Analyst",
            "Digital marketing and web analytics",
        )
        cyber = self._raw(
            "cyber",
            "Network Traffic Analyst",
            "Incident response and network security",
        )
        profession = {
            "role": "Analyst",
            "specialization": "Traffic Analyst",
            "domain": "Digital Marketing / Web Analytics",
            "seniority": "Middle",
            "core_skills": ["Google Analytics", "PPC"],
        }

        def vectors(texts: list[str], model: str) -> dict[str, list[float]]:
            del model
            result = {}
            for text in texts:
                if text == "Digital Marketing / Web Analytics":
                    result[text] = [0.0, 1.0]
                elif "Digital marketing and web analytics" in text:
                    result[text] = [0.6, 0.8]
                elif "Incident response and network security" in text:
                    result[text] = [1.0, 0.0]
                elif "Google Analytics PPC" in text:
                    result[text] = [0.6, 0.8]
                else:
                    result[text] = [1.0, 0.0]
            return result

        embed_mock.side_effect = vectors
        diagnostics: list[EmbeddingCandidateDiagnostic] = []

        result = embedding_filter_vacancies(
            [cyber, digital],
            profession,
            domain_threshold=0.6,
            diagnostic_rows=diagnostics,
        )

        self.assertEqual([item.source_id for item in result], ["digital"])
        by_id = {item.source_id: item for item in diagnostics}
        self.assertTrue(by_id["digital"].passed_embedding_filter)
        self.assertIsNone(by_id["digital"].rejection_reason)
        self.assertFalse(by_id["cyber"].passed_embedding_filter)
        self.assertEqual(
            by_id["cyber"].rejection_reason,
            "domain_similarity_below_threshold",
        )

    def test_hybrid_title_match_keeps_aliases_and_rejects_false_friends(self) -> None:
        cases = [
            (
                TargetProfile(
                    role="Engineer",
                    specialization="ML Engineer",
                    domain="Machine Learning",
                    seniority="Middle",
                    role_family="ML Engineering",
                    focus="Model Development",
                    core_skills=[],
                    aliases=["Machine Learning Engineer"],
                    adjacent_specializations=["AI Engineer"],
                    excluded_specializations=["AI Infrastructure Engineer"],
                ),
                "Machine Learning Engineer",
                "Frontend Engineer",
            ),
            (
                TargetProfile(
                    role="Journalist",
                    specialization="Корреспондент",
                    domain="Media",
                    seniority=None,
                    role_family="Journalism",
                    focus="News Reporting",
                    core_skills=[],
                    aliases=["Correspondent", "Reporter"],
                    excluded_specializations=[],
                ),
                "Reporter",
                "General Counsel",
            ),
            (
                TargetProfile(
                    role="Communications Specialist",
                    specialization="PR-специалист",
                    domain="Communications",
                    seniority=None,
                    role_family="Public Relations",
                    focus="Media Relations",
                    core_skills=[],
                    aliases=["PR Specialist", "Public Relations Specialist"],
                    excluded_specializations=[],
                ),
                "Public Relations Specialist",
                "HR Specialist",
            ),
        ]

        for profile, synonym, false_friend in cases:
            with self.subTest(synonym=synonym):
                accepted = title_match_signals(
                    profile,
                    synonym,
                    dense_similarity=0.80,
                )
                rejected = title_match_signals(
                    profile,
                    false_friend,
                    dense_similarity=0.95,
                )
                self.assertTrue(accepted.has_lexical_support)
                self.assertFalse(rejected.has_lexical_support)
                self.assertTrue(rejected.semantic_only)
                self.assertLess(rejected.title_match_score, accepted.title_match_score)

    def test_hybrid_title_match_penalizes_adjacent_and_rejects_excluded(self) -> None:
        profile = TargetProfile(
            role="Engineer",
            specialization="ML Engineer",
            domain="Machine Learning",
            seniority="Middle",
            role_family="ML Engineering",
            focus="Model Development",
            core_skills=[],
            aliases=["Machine Learning Engineer"],
            adjacent_specializations=["AI Engineer"],
            excluded_specializations=["AI Infrastructure Engineer"],
        )

        exact = title_match_signals(profile, "Middle ML Engineer", dense_similarity=0.8)
        adjacent = title_match_signals(profile, "AI Engineer", dense_similarity=0.8)
        excluded = title_match_signals(
            profile,
            "AI Infrastructure Engineer",
            dense_similarity=0.9,
        )

        self.assertTrue(adjacent.adjacent_match)
        self.assertLess(adjacent.title_match_score, exact.title_match_score)
        self.assertTrue(excluded.excluded_match)
        self.assertEqual(excluded.title_match_score, 0)

    def test_noop_reranker_benchmark_reports_precision_and_time(self) -> None:
        profile = TargetProfile(
            role="Journalist",
            specialization="Correspondent",
            domain="Media",
            seniority=None,
            role_family="Journalism",
            focus="News Reporting",
            core_skills=[],
            aliases=["Reporter"],
            adjacent_specializations=[],
            excluded_specializations=[],
        )
        labeled = [
            (self._raw("good", "Reporter", "News"), True),
            (self._raw("bad", "General Counsel", "Legal"), False),
        ]

        result = benchmark_title_reranker(profile, labeled, cutoff=1)

        self.assertEqual(result.precision_at_cutoff, 1.0)
        self.assertEqual(result.candidates, 2)
        self.assertGreaterEqual(result.elapsed_seconds, 0)

    @patch("resume_evaluator.synthetic_vacancy_pipeline._embed_text_map")
    def test_embedding_filter_does_not_accept_semantic_only_title(
        self,
        embed_mock: MagicMock,
    ) -> None:
        profile = TargetProfile(
            role="Engineer",
            specialization="ML Engineer",
            domain="Machine Learning",
            seniority="Middle",
            role_family="ML Engineering",
            focus="Model Development",
            core_skills=[],
            aliases=["Machine Learning Engineer"],
            adjacent_specializations=[],
            excluded_specializations=[],
        )
        good = self._raw("ml", "Machine Learning Engineer", "Build ML models")
        false_friend = self._raw("frontend", "Frontend Engineer", "Build products")
        embed_mock.side_effect = lambda texts, model: {
            text: [1.0, 0.0] for text in texts
        }
        diagnostics: list[EmbeddingCandidateDiagnostic] = []

        result = embedding_filter_vacancies(
            [false_friend, good],
            profile,
            domain_threshold=-1,
            diagnostic_rows=diagnostics,
        )

        self.assertEqual([item.source_id for item in result], ["ml"])
        by_id = {item.source_id: item for item in diagnostics}
        self.assertTrue(by_id["frontend"].semantic_only)
        self.assertEqual(
            by_id["frontend"].rejection_reason,
            "semantic_only_without_lexical_support",
        )

    @patch("resume_evaluator.synthetic_vacancy_pipeline.chat")
    def test_retries_truncated_normalization_response(
        self,
        chat_mock: MagicMock,
    ) -> None:
        broken = MagicMock()
        broken.message.content = '{"vacancies": ['
        recovered = MagicMock()
        recovered.message.content = json.dumps(
            {
                "vacancies": [
                    {
                        "source_id": "raw:1",
                        "canonical_profession": "Аналитик трафика",
                        "is_relevant": True,
                        "required_skills": ["Веб-аналитика"],
                        "preferred_skills": [],
                        "responsibilities": ["Анализировать трафик"],
                        "experience_years_min": 1,
                        "education_level": None,
                        "education_field": None,
                        "employment_type": None,
                        "work_format": None,
                        "conditions": [],
                    }
                ]
            },
            ensure_ascii=False,
        )
        chat_mock.side_effect = [broken, recovered]
        raw = RawVacancy(
            source="test",
            source_id="raw:1",
            url="https://jobs.test/1",
            title="Аналитик трафика",
            employer="Компания",
            published_at=None,
            raw_text="Анализировать трафик и показатели сайта",
        )

        result = _normalize_batch_with_fallback(
            [raw],
            "Аналитик трафика",
        )

        self.assertEqual(len(result.vacancies), 1)
        self.assertEqual(chat_mock.call_count, 2)

    @patch("resume_evaluator.synthetic_vacancy_pipeline.chat")
    def test_normalization_cache_reuses_vacancy_by_text_hash(
        self,
        chat_mock: MagicMock,
    ) -> None:
        response = MagicMock()
        response.message.content = json.dumps(
            {
                "vacancies": [
                    {
                        "source_id": "first-id",
                        "profession": {
                            "role": "Analyst",
                            "specialization": "Traffic Analyst",
                            "domain": "Digital Marketing / Web Analytics",
                            "seniority": "Middle",
                            "core_skills": ["Google Analytics"],
                        },
                        "is_relevant": True,
                        "required_skills": ["Google Analytics"],
                        "preferred_skills": [],
                        "responsibilities": ["Анализировать веб-трафик"],
                        "experience_years_min": 2,
                        "education_level": None,
                        "education_field": None,
                        "employment_type": None,
                        "work_format": None,
                        "conditions": [],
                    }
                ]
            },
            ensure_ascii=False,
        )
        chat_mock.return_value = response
        first = self._raw(
            "first-id",
            "Traffic Analyst",
            "Google Analytics and web traffic",
        )
        second = self._raw(
            "new-source-id",
            "Traffic Analyst",
            "Google Analytics and web traffic",
        )

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "normalization_cache.json"
            first_result = normalize_vacancies(
                [first],
                "Traffic Analyst",
                cache_path=cache_path,
            )
            second_result = normalize_vacancies(
                [second],
                "Traffic Analyst",
                cache_path=cache_path,
            )

        self.assertEqual(first_result[0].source_id, "first-id")
        self.assertEqual(second_result[0].source_id, "new-source-id")
        self.assertEqual(chat_mock.call_count, 1)

    def test_selects_relevant_and_mutually_similar_cluster(self) -> None:
        candidates = self.selected + [
            make_vacancy(
                "sales",
                profession="Менеджер по продажам",
                relevant=False,
                required=["Холодные звонки"],
            )
        ]
        calculate_relevance_scores(
            candidates,
            "Python-разработчик",
            ["Python", "PostgreSQL", "Git"],
        )

        result = select_best_three(candidates)

        self.assertEqual({item.source_id for item in result}, {"v1", "v2", "v3"})

    def test_selection_rejects_cross_domain_vacancy(self) -> None:
        for vacancy in self.selected:
            vacancy.profession_relevance = 0.9
            vacancy.relevance_score = 0.9
        cross_domain = make_vacancy(
            "incident-response",
            profession="Аналитик трафика",
            required=["Network traffic analysis", "Digital forensics"],
        )
        cross_domain.profession_relevance = 0.2
        cross_domain.relevance_score = 0.95

        result = select_best_three(self.selected + [cross_domain])

        self.assertNotIn("incident-response", [item.source_id for item in result])

    def test_triplet_reports_low_confidence_when_domain_is_weak(self) -> None:
        for vacancy in self.selected:
            vacancy.relevance_score = 0.9
            vacancy.embedding_domain_similarity = 0.2

        selection = evaluate_best_triplet(
            self.selected,
            "Python-разработчик",
        )

        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual(selection.confidence, "low_confidence")
        self.assertLess(selection.metrics.domain_similarity, 0.45)

    def test_mutual_similarity_prefers_embedding_vectors(self) -> None:
        first, second = self.selected[:2]
        first.embedding_content_vector = [1.0, 0.0]
        second.embedding_content_vector = [0.0, 1.0]

        similarity = vacancy_similarity(first, second)

        self.assertEqual(similarity, 0.0)

    def test_triplet_rejects_vacancies_that_are_not_mutually_similar(self) -> None:
        vectors = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
        for vacancy, vector in zip(self.selected, vectors, strict=True):
            vacancy.relevance_score = 0.9
            vacancy.embedding_domain_similarity = 0.9
            vacancy.embedding_content_vector = list(vector)

        selection = evaluate_best_triplet(
            self.selected,
            "Python-разработчик",
        )

        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual(selection.confidence, "low_confidence")
        self.assertEqual(selection.metrics.mutual_similarity, 0.0)

    def test_pipeline_does_not_create_vacancy_for_low_confidence(self) -> None:
        plan = ProfessionSearchPlan(
            profession={
                "role": "Analyst",
                "specialization": "Traffic Analyst",
                "domain": "Digital Marketing / Web Analytics",
                "seniority": "Middle",
                "core_skills": ["Google Analytics"],
            },
            search_queries=["Traffic Analyst", "Web Analyst", "Web Analytics"],
        )
        raw = [
            self._raw(
                str(index),
                f"Traffic Analyst {index}",
                "Digital marketing and web analytics",
            )
            for index in range(20)
        ]
        low_selection = TripletSelection(
            vacancies=self.selected,
            metrics=TripletMetrics(
                triplet_score=0.4,
                profession_relevance=0.7,
                domain_similarity=0.3,
                mutual_similarity=0.4,
            ),
            confidence="low_confidence",
        )
        resume_data = {
            "candidate_name": "Иван",
            "target_position": "Traffic Analyst",
            "professional_summary": None,
            "contact_channels": [],
            "skills": ["Google Analytics"],
            "soft_skills": [],
            "work_experience": [],
            "education": [],
            "projects": [],
            "courses_and_certifications": [],
            "languages": [],
            "achievements": [],
            "portfolio_links": [],
        }

        with tempfile.TemporaryDirectory() as directory:
            resume_path = Path(directory) / "resume.json"
            output_path = Path(directory) / "synthetic_vacancy.json"
            resume_path.write_text(
                json.dumps(resume_data, ensure_ascii=False),
                encoding="utf-8",
            )
            with (
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline.build_search_plan",
                    return_value=plan,
                ),
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline.search_public_vacancies",
                    return_value=(raw, []),
                ),
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline.embedding_filter_vacancies",
                    return_value=raw[:8],
                ),
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline.normalize_vacancies",
                    return_value=self.selected,
                ),
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline.evaluate_best_triplet",
                    return_value=low_selection,
                ),
                patch(
                    "resume_evaluator.synthetic_vacancy_pipeline.group_selected_concepts"
                ) as grouping_mock,
            ):
                result = run_pipeline(resume_path, output_path)

            self.assertEqual(result.status, "insufficient_market_data")
            self.assertEqual(result.confidence, "low_confidence")
            self.assertFalse(output_path.exists())
            grouping_mock.assert_not_called()

    def test_normalized_vacancy_serializes_structured_profession(self) -> None:
        vacancy = make_vacancy("structured")

        dumped = vacancy.model_dump()

        self.assertIn("profession", dumped)
        self.assertNotIn("canonical_profession", dumped)
        self.assertEqual(
            dumped["profession"]["specialization"],
            "Python-разработчик",
        )

    def test_discards_invented_group_member_and_restores_source_items(self) -> None:
        proposed = ConceptGrouping(
            skill_groups=[
                ConceptGroup(
                    members=[
                        ConceptMember(
                            source_id="v1",
                            text="Python",
                            kind="required_skill",
                        ),
                        ConceptMember(
                            source_id="v2",
                            text="Kubernetes",
                            kind="required_skill",
                        ),
                    ]
                )
            ],
            responsibility_groups=[],
        )

        result = validate_and_complete_groups(proposed, self.selected)
        all_members = [
            member
            for group in result.skill_groups + result.responsibility_groups
            for member in group.members
        ]

        self.assertNotIn("Kubernetes", [item.text for item in all_members])
        self.assertIn("Docker", [item.text for item in all_members])
        self.assertIn(
            "Мониторинг сервисов",
            [item.text for item in all_members],
        )

    def test_builds_skill_frequency_groups_in_python_by_distinct_source(self) -> None:
        selected = [
            self.selected[0].model_copy(
                update={"required_skills": ["Python", "Python", "Postgres"]}
            ),
            self.selected[1].model_copy(
                update={"required_skills": ["Python", "PostgreSQL"]}
            ),
            self.selected[2],
        ]

        groups = build_skill_groups_python(selected)
        by_value = {
            group.canonical_text: len({member.source_id for member in group.members})
            for group in groups
        }

        self.assertEqual(by_value["Python"], 3)
        self.assertEqual(by_value["Docker"], 2)
        self.assertEqual(by_value["Postgres"], 1)
        self.assertEqual(by_value["PostgreSQL"], 1)

    def test_skill_normalization_preserves_technology_symbols(self) -> None:
        for value, expected in (
            ("R", "r"),
            ("C", "c"),
            ("C++", "c++"),
            ("C#", "c#"),
            (".NET", ".net"),
            ("F#", "f#"),
            ("R&D", "r&d"),
            ("Python", "python"),
            ("a", None),
        ):
            with self.subTest(value=value):
                self.assertEqual(normalize_skill_name(value), expected)

    def test_classified_requirements_exclude_soft_education_and_domain(self) -> None:
        vacancy = make_vacancy(
            "classified",
            required=[
                "Python",
                "TensorFlow",
                "Communication",
                "PhD in Computer Science",
                "Experience with ad tech",
            ],
        ).model_copy(
            update={
                "professional_skills": ["Python"],
                "tools_and_technologies": ["TensorFlow"],
                "soft_skills": ["Communication"],
                "education_requirements": ["PhD in Computer Science"],
                "domain_knowledge": ["Experience with ad tech"],
                "professional_requirements_detected": True,
                "extraction_status": "ok",
            }
        )

        groups = build_skill_groups_python([vacancy])
        values = {group.canonical_text for group in groups}
        serialized = vacancy.model_dump()

        self.assertEqual(values, {"Python", "TensorFlow"})
        self.assertEqual(serialized["professional_skills"], ["Python"])
        self.assertNotIn("technical_skills", serialized)
        self.assertNotIn("Communication", values)
        self.assertNotIn("PhD in Computer Science", values)
        self.assertNotIn("Experience with ad tech", values)

    def test_extraction_quality_distinguishes_empty_failed_and_sparse(self) -> None:
        profession = {
            "role": "Engineer",
            "specialization": "ML Engineer",
            "domain": "Machine Learning",
            "seniority": "Middle",
            "role_family": "ML Engineering",
            "focus": "Model Development / Production ML",
            "core_skills": [],
        }

        def extracted(tools: list[str]) -> ExtractedVacancy:
            return ExtractedVacancy(
                source_id="q",
                profession=profession,
                is_relevant=True,
                required_skills=tools,
                preferred_skills=[],
                professional_skills=[],
                tools_and_technologies=tools,
                soft_skills=[],
                education_requirements=[],
                domain_knowledge=[],
                extraction_status="ok",
                professional_requirements_detected=False,
                responsibilities=[],
                experience_years_min=None,
                education_level=None,
                education_field=None,
                employment_type=None,
                work_format=None,
                conditions=[],
            )

        no_requirements = self._raw("q", "ML Engineer", "Build useful models")
        failed_source = self._raw(
            "q",
            "ML Engineer",
            "Requirements: experience with Python, TensorFlow and SQL",
        )

        self.assertEqual(
            _set_extraction_quality(extracted([]), no_requirements).extraction_status,
            "empty",
        )
        self.assertEqual(
            _set_extraction_quality(extracted([]), failed_source).extraction_status,
            "failed",
        )
        self.assertEqual(
            _set_extraction_quality(extracted(["Python"]), failed_source).extraction_status,
            "sparse",
        )
        self.assertEqual(
            _set_extraction_quality(
                extracted(["Python", "TensorFlow"]), failed_source
            ).extraction_status,
            "ok",
        )

        with patch(
            "resume_evaluator.synthetic_vacancy_pipeline."
            "_normalize_batch_with_fallback"
        ) as retry_mock:
            retry_mock.return_value = MagicMock(
                vacancies=[extracted(["Python", "TensorFlow"])]
            )
            recovered = _quality_checked_extraction(
                extracted([]),
                failed_source,
                profession_from_value(profession),
            )

        self.assertEqual(recovered.extraction_status, "ok")
        retry_mock.assert_called_once()

    def test_ml_focus_excludes_adjacent_infrastructure_and_data_science(self) -> None:
        target = {
            "role": "Engineer",
            "specialization": "ML Engineer",
            "domain": "Machine Learning",
            "seniority": "Middle",
            "role_family": "ML Engineering",
            "focus": "Model Development / Production ML",
            "core_skills": [],
        }
        core = make_vacancy("core", profession="ML Engineer")
        core.profession.role_family = "ML Engineering"
        core.profession.focus = "Model Development / Production ML"
        infrastructure = make_vacancy("infra", profession="ML Engineer")
        infrastructure.profession.role_family = "ML Infrastructure"
        infrastructure.profession.focus = "AI Infrastructure"
        data_science = make_vacancy("ds", profession="ML Engineer")
        data_science.profession.role_family = "Data Science"
        data_science.profession.focus = "Decision Intelligence"

        self.assertEqual(_focus_compatibility(target, core), 1.0)
        self.assertEqual(_focus_compatibility(target, infrastructure), 0.0)
        self.assertEqual(_focus_compatibility(target, data_science), 0.0)

    def test_universal_market_profile_regressions(self) -> None:
        cases = [
            TargetProfile(
                role="Engineer",
                specialization="ML Engineer",
                role_family="ML Engineering",
                domain="Machine Learning",
                focus="Model Development / Production ML",
                seniority="Middle",
                aliases=["Machine Learning Engineer"],
                adjacent_specializations=["AI Engineer"],
                excluded_specializations=["AI Infrastructure Engineer"],
                core_skills=[],
            ),
            TargetProfile(
                role="Journalist",
                specialization="Корреспондент",
                role_family="Journalism",
                domain="Media",
                focus="News Reporting",
                seniority="Middle",
                aliases=["Correspondent", "Reporter", "News Reporter"],
                adjacent_specializations=["News Journalist"],
                excluded_specializations=["PR Manager"],
                core_skills=[],
            ),
            TargetProfile(
                role="Marketer",
                specialization="Маркетолог",
                role_family="Marketing",
                domain="Marketing",
                focus="Brand Marketing",
                seniority="Middle",
                aliases=["Marketing Specialist", "Marketer"],
                adjacent_specializations=["Performance Marketer"],
                excluded_specializations=["PR Manager"],
                core_skills=[],
            ),
            TargetProfile(
                role="Communications Specialist",
                specialization="PR-специалист",
                role_family="Public Relations",
                domain="Communications",
                focus="Media Relations",
                seniority="Middle",
                aliases=["PR Specialist", "Public Relations Specialist"],
                adjacent_specializations=["Communications Manager"],
                excluded_specializations=["Correspondent", "Reporter"],
                core_skills=[],
            ),
        ]

        for profile in cases:
            with self.subTest(profile=profile.specialization):
                started = time.perf_counter()
                queries = _expand_search_queries(profile, [])
                self.assertIn(profile.specialization, queries)
                self.assertTrue(set(profile.aliases) & set(queries))
                self.assertTrue(any(profile.seniority in query for query in queries))

                market: list[NormalizedVacancy] = []
                good_titles = [profile.specialization, *profile.aliases[:2]]
                while len(good_titles) < 3:
                    good_titles.append(profile.specialization)
                for index, title in enumerate(good_titles[:3]):
                    vacancy = make_vacancy(
                        f"{profile.specialization}-good-{index}",
                        profession=title,
                        required=["Профессиональная компетенция"],
                        responsibilities=["Выполнять профильные обязанности"],
                    )
                    vacancy.profession.role = profile.role
                    vacancy.profession.specialization = title
                    vacancy.profession.role_family = profile.role_family
                    vacancy.profession.domain = profile.domain
                    vacancy.profession.focus = profile.focus
                    vacancy.profession.seniority = profile.seniority
                    vacancy.embedding_role_similarity = 0.9
                    vacancy.embedding_specialization_similarity = 0.9
                    vacancy.embedding_domain_similarity = 0.9
                    vacancy.embedding_content_similarity = 0.9
                    market.append(vacancy)

                excluded_title = profile.excluded_specializations[0]
                excluded = make_vacancy("excluded", profession=excluded_title)
                excluded.profession.specialization = excluded_title
                excluded.profession.role_family = profile.role_family
                excluded.profession.domain = profile.domain
                excluded.profession.focus = profile.focus
                excluded.profession.seniority = profile.seniority
                excluded.embedding_role_similarity = 0.9
                excluded.embedding_specialization_similarity = 0.9
                excluded.embedding_domain_similarity = 0.9
                excluded.embedding_content_similarity = 0.9
                market.append(excluded)

                calculate_relevance_scores(market, profile)
                eligible = _eligible_market_vacancies(market, profile)
                selection = evaluate_best_triplet(eligible, profile)

                self.assertEqual(len(eligible), 3)
                self.assertNotIn("excluded", {item.source_id for item in eligible})
                self.assertTrue(all(item.extraction_status == "ok" for item in eligible))
                self.assertIsNotNone(selection)
                assert selection is not None
                self.assertIn(selection.confidence, {"medium", "high"})
                self.assertLess(time.perf_counter() - started, 1.0)

    def test_special_skills_keep_original_evidence_and_discard_junk(self) -> None:
        technologies = ["R", "C", "C++", "C#", ".NET", "F#", "R&D", "Python"]
        selected = [
            make_vacancy("v1", required=technologies + ["a"]),
            make_vacancy("v2", required=technologies + ["a"]),
            make_vacancy("v3", required=["Python"]),
        ]
        grouping = validate_and_complete_groups(
            ConceptGrouping(
                skill_groups=build_skill_groups_python(selected),
                responsibility_groups=[],
            ),
            selected,
        )

        result = aggregate_synthetic_vacancy(
            "ML Engineer",
            selected,
            grouping,
            PipelineStatistics(
                search_queries=[],
                public_vacancies_found=3,
                vacancies_after_deduplication=3,
                relevant_vacancies=3,
                source_errors=[],
            ),
        )

        self.assertEqual(set(result.required_skills), set(technologies))
        self.assertNotIn("a", result.required_skills + result.preferred_skills)
        for evidence in result.requirement_evidence:
            self.assertTrue(evidence.source_values)
            self.assertIn(evidence.value, technologies)
            self.assertEqual(evidence.source_values, [evidence.value])
            self.assertEqual(evidence.frequency, 3 if evidence.value == "Python" else 2)

    @patch("resume_evaluator.synthetic_vacancy_pipeline.chat")
    def test_llm_grouping_receives_only_responsibilities(self, chat_mock) -> None:
        chat_mock.return_value = MagicMock(
            message=MagicMock(
                content=json.dumps(
                    {
                        "responsibility_groups": [
                            {
                                "canonical_text": "Разрабатывать API",
                                "members": [
                                    {
                                        "source_id": "v1",
                                        "text": "Разрабатывать API",
                                        "kind": "responsibility",
                                    }
                                ],
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            )
        )

        result = group_selected_concepts(self.selected)

        request_payload = json.loads(
            chat_mock.call_args.kwargs["messages"][1]["content"]
        )
        self.assertEqual(set(request_payload), {"responsibilities"})
        self.assertTrue(result.skill_groups)
        self.assertEqual(
            result.responsibility_groups[0].canonical_text,
            "Разрабатывать API",
        )

    @patch("resume_evaluator.synthetic_vacancy_pipeline.chat")
    def test_structured_aggregation_does_not_call_llm(self, chat_mock) -> None:
        selected = [
            self.selected[0].model_copy(
                update={
                    "education_level": "Высшее",
                    "conditions": ["ДМС", "ДМС"],
                }
            ),
            self.selected[1].model_copy(
                update={
                    "education_level": "Высшее",
                    "conditions": ["ДМС"],
                }
            ),
            self.selected[2].model_copy(
                update={
                    "education_level": "Среднее специальное",
                    "conditions": ["Оплата обучения"],
                }
            ),
        ]
        grouping = ConceptGrouping(
            skill_groups=build_skill_groups_python(selected),
            responsibility_groups=[],
        )

        result = aggregate_synthetic_vacancy(
            "Python-разработчик",
            selected,
            grouping,
            PipelineStatistics(
                search_queries=[],
                public_vacancies_found=3,
                vacancies_after_deduplication=3,
                relevant_vacancies=3,
                source_errors=[],
            ),
        )

        chat_mock.assert_not_called()
        self.assertEqual(result.education_level, "Высшее")
        self.assertEqual(result.conditions, ["ДМС"])
        condition = next(
            item
            for item in result.other_requirement_evidence
            if item.field_name == "conditions"
        )
        self.assertEqual(condition.frequency, 2)

    def test_aggregates_frequencies_sources_and_median_experience(self) -> None:
        grouping = self._grouping()
        grouping.responsibility_groups[0].canonical_text = "Разрабатывать API"
        stats = PipelineStatistics(
            search_queries=["Python developer", "Python backend", "Python engineer"],
            public_vacancies_found=25,
            vacancies_after_deduplication=23,
            relevant_vacancies=12,
            source_errors=[],
        )

        result = aggregate_synthetic_vacancy(
            "Python-разработчик",
            self.selected,
            grouping,
            stats,
        )

        self.assertEqual(set(result.required_skills), {"Python", "Docker"})
        self.assertEqual(result.preferred_skills, [])
        self.assertIn("Git", [item.value for item in result.market_variations])
        self.assertEqual(result.responsibilities, ["Разрабатывать API"])
        self.assertEqual(result.experience_requirement, "Опыт от 3 лет")
        experience_evidence = next(
            item
            for item in result.other_requirement_evidence
            if item.field_name == "experience_requirement"
        )
        self.assertEqual(experience_evidence.frequency, 3)
        self.assertEqual(len(experience_evidence.source_urls), 3)
        python_evidence = next(
            item for item in result.requirement_evidence if item.value == "Python"
        )
        self.assertEqual(python_evidence.frequency, 3)
        self.assertEqual(python_evidence.field_name, "required_skill")
        self.assertEqual(len(python_evidence.source_urls), 3)
        self.assertEqual(len(result.selected_vacancies), 3)

        compatible = VacancyData.model_validate(result.model_dump())
        self.assertEqual(compatible.required_skills, result.required_skills)
        score = score_candidate({}, result.model_dump())
        self.assertEqual(score.vacancy_title, "Python-разработчик")
        self.assertNotIn("Git", score.missing_preferred_skills)

    def test_skill_frequencies_use_all_relevant_market_vacancies(self) -> None:
        market = self.selected + [
            make_vacancy("v4", required=["Python", "Docker", "SQL"]),
            make_vacancy("v5", required=["Python", "SQL"]),
            make_vacancy("v6", required=["Python", "Airflow"]),
        ]
        result = aggregate_synthetic_vacancy(
            "Python-разработчик",
            self.selected,
            self._grouping(),
            PipelineStatistics(
                search_queries=[],
                public_vacancies_found=6,
                vacancies_after_deduplication=6,
                relevant_vacancies=6,
                source_errors=[],
            ),
            market_vacancies=market,
        )

        self.assertEqual(result.market_vacancy_count, 6)
        self.assertEqual(result.required_skills, ["Python"])
        self.assertEqual(set(result.preferred_skills), {"Docker", "SQL"})
        variations = {item.value: item.frequency for item in result.market_variations}
        self.assertEqual(variations["Git"], 1)
        self.assertEqual(variations["Airflow"], 1)
        self.assertNotIn("Git", result.preferred_skills)
        python = next(
            item for item in result.requirement_evidence if item.value == "Python"
        )
        self.assertEqual(python.frequency, 6)
        self.assertEqual(len(python.source_urls), 6)

    def test_frequency_thresholds_follow_actual_market_sample_size(self) -> None:
        for size in range(3, 7):
            with self.subTest(size=size):
                required_count = (2 * size + 2) // 3
                market = []
                for index in range(size):
                    skills = ["Common"] if index < required_count else []
                    if index < 2:
                        skills.append("Repeated")
                    if index == 0:
                        skills.append("Unique")
                    market.append(make_vacancy(f"n{size}-{index}", required=skills))
                result = aggregate_synthetic_vacancy(
                    "Универсальная профессия",
                    market[:3],
                    ConceptGrouping(skill_groups=[], responsibility_groups=[]),
                    PipelineStatistics(
                        search_queries=[],
                        public_vacancies_found=size,
                        vacancies_after_deduplication=size,
                        relevant_vacancies=size,
                        source_errors=[],
                    ),
                    market_vacancies=market,
                )

                self.assertIn("Common", result.required_skills)
                if required_count == 2:
                    self.assertIn("Repeated", result.required_skills)
                else:
                    self.assertIn("Repeated", result.preferred_skills)
                self.assertIn(
                    "Unique",
                    {item.value for item in result.market_variations},
                )

    def test_responsibility_clustering_splits_merely_related_tasks(self) -> None:
        broad_group = ConceptGroup(
            canonical_text="Design data and feature pipelines",
            members=[
                ConceptMember(
                    source_id="v1",
                    text="Explore and visualize data to identify factors impacting model accuracy",
                    kind="responsibility",
                ),
                ConceptMember(
                    source_id="v2",
                    text="Implement data pipelines and feature engineering workflows",
                    kind="responsibility",
                ),
                ConceptMember(
                    source_id="v3",
                    text="Design data and feature pipelines",
                    kind="responsibility",
                ),
            ],
        )

        tightened = _tighten_responsibility_groups([broad_group])
        source_sets = [
            {member.source_id for member in group.members} for group in tightened
        ]

        self.assertIn({"v1"}, source_sets)
        self.assertIn({"v2", "v3"}, source_sets)
        self.assertNotIn({"v1", "v2", "v3"}, source_sets)

        collaboration = ConceptGroup(
            canonical_text="Collaborate to integrate ML models",
            members=[
                ConceptMember(
                    source_id="v1",
                    text="Collaborate with teams to build ML models and simulation environments",
                    kind="responsibility",
                ),
                ConceptMember(
                    source_id="v2",
                    text="Collaborate with engineers to integrate models into APIs and services",
                    kind="responsibility",
                ),
            ],
        )
        collaboration_sets = [
            {member.source_id for member in group.members}
            for group in _tighten_responsibility_groups([collaboration])
        ]
        self.assertEqual(collaboration_sets, [{"v1"}, {"v2"}])

    @patch(
        "resume_evaluator.synthetic_vacancy_pipeline.group_selected_concepts"
    )
    @patch("resume_evaluator.synthetic_vacancy_pipeline.evaluate_best_triplet")
    @patch("resume_evaluator.synthetic_vacancy_pipeline.normalize_vacancies")
    @patch("resume_evaluator.synthetic_vacancy_pipeline.embedding_filter_vacancies")
    @patch("resume_evaluator.synthetic_vacancy_pipeline.search_public_vacancies")
    @patch("resume_evaluator.synthetic_vacancy_pipeline.build_search_plan")
    def test_pipeline_saves_compatible_json(
        self,
        plan_mock,
        search_mock,
        embedding_mock,
        normalize_mock,
        selection_mock,
        grouping_mock,
    ) -> None:
        plan_mock.return_value = ProfessionSearchPlan(
            target_profession="Python-разработчик",
            search_queries=[
                "Python developer",
                "Python backend",
                "Python engineer",
            ],
        )
        search_mock.return_value = (
            [
                RawVacancy(
                    source="test",
                    source_id=f"raw:{index}",
                    url=f"https://jobs.test/{index}",
                    title="Python Developer",
                    employer=f"Company {index}",
                    published_at=None,
                    raw_text="Python API",
                )
                for index in range(20)
            ],
            [],
        )
        embedding_mock.return_value = search_mock.return_value[0][:8]
        normalize_mock.return_value = self.selected
        selection_mock.return_value = TripletSelection(
            vacancies=self.selected,
            metrics=TripletMetrics(
                triplet_score=0.8,
                profession_relevance=0.8,
                domain_similarity=0.8,
                mutual_similarity=0.8,
            ),
            confidence="high",
        )
        grouping_mock.return_value = self._grouping()

        resume_data = {
            "candidate_name": "Иван",
            "target_position": "Python-разработчик",
            "professional_summary": None,
            "contact_channels": [],
            "skills": ["Python"],
            "soft_skills": [],
            "work_experience": [],
            "education": [],
            "projects": [],
            "courses_and_certifications": [],
            "languages": [],
            "achievements": [],
            "portfolio_links": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            resume_path = Path(directory) / "resume.json"
            output_path = Path(directory) / "synthetic_vacancy.json"
            resume_path.write_text(
                json.dumps(resume_data, ensure_ascii=False),
                encoding="utf-8",
            )

            result = run_pipeline(resume_path, output_path)

            self.assertTrue(output_path.is_file())
            self.assertTrue(
                output_path.with_name(
                    "synthetic_vacancy.retrieval_diagnostics.json"
                ).is_file()
            )
            compatible = VacancyData.model_validate_json(
                output_path.read_text(encoding="utf-8")
            )
            self.assertEqual(compatible.title, result.title)
            self.assertEqual(result.status, "ok")
            self.assertEqual(result.confidence, "high")
            self.assertEqual(result.triplet_metrics.triplet_score, 0.8)
            normalized_input = normalize_mock.call_args.args[0]
            self.assertEqual(len(normalized_input), 8)

    @patch(
        "resume_evaluator.synthetic_vacancy_pipeline.group_selected_concepts"
    )
    @patch("resume_evaluator.synthetic_vacancy_pipeline.evaluate_best_triplet")
    @patch("resume_evaluator.synthetic_vacancy_pipeline.normalize_vacancies")
    @patch("resume_evaluator.synthetic_vacancy_pipeline.embedding_filter_vacancies")
    @patch("resume_evaluator.synthetic_vacancy_pipeline.search_public_vacancies")
    @patch("resume_evaluator.synthetic_vacancy_pipeline.build_search_plan")
    def test_pipeline_expands_normalization_pool_until_six_valid_vacancies(
        self,
        plan_mock,
        search_mock,
        embedding_mock,
        normalize_mock,
        selection_mock,
        grouping_mock,
    ) -> None:
        plan_mock.return_value = ProfessionSearchPlan(
            target_profession="Корреспондент",
            search_queries=["Корреспондент", "Reporter", "News Reporter"],
        )
        raw = [
            self._raw(str(index), "Корреспондент", "Новости и репортажи")
            for index in range(20)
        ]
        search_mock.return_value = (raw, [])
        embedding_mock.return_value = raw[:16]
        market = [
            make_vacancy(
                f"market-{index}",
                profession="Корреспондент",
                required=["Интервьюирование"],
            )
            for index in range(6)
        ]

        def normalized_for_batch(candidates, *_args, **_kwargs):
            counts = {8: 2, 12: 4, 16: 6}
            return market[:counts[len(candidates)]]

        normalize_mock.side_effect = normalized_for_batch
        selection_mock.return_value = TripletSelection(
            vacancies=market[:3],
            metrics=TripletMetrics(
                triplet_score=0.8,
                profession_relevance=0.8,
                domain_similarity=0.8,
                mutual_similarity=0.8,
            ),
            confidence="high",
        )
        grouping_mock.return_value = ConceptGrouping(
            skill_groups=[],
            responsibility_groups=[],
        )
        resume_data = {
            "candidate_name": None,
            "target_position": "Корреспондент",
            "professional_summary": None,
            "contact_channels": [],
            "skills": [],
            "soft_skills": [],
            "work_experience": [],
            "education": [],
            "projects": [],
            "courses_and_certifications": [],
            "languages": [],
            "achievements": [],
            "portfolio_links": [],
        }

        with tempfile.TemporaryDirectory() as directory:
            resume_path = Path(directory) / "resume.json"
            output_path = Path(directory) / "synthetic_vacancy.json"
            resume_path.write_text(
                json.dumps(resume_data, ensure_ascii=False),
                encoding="utf-8",
            )
            result = run_pipeline(resume_path, output_path)

        self.assertEqual(
            [len(call.args[0]) for call in normalize_mock.call_args_list],
            [8, 12, 16],
        )
        self.assertEqual(result.market_vacancy_count, 6)
        self.assertEqual(result.pipeline_statistics.relevant_vacancies, 6)
        self.assertEqual(
            result.pipeline_statistics.vacancies_selected_for_normalization,
            16,
        )

    def _grouping(self) -> ConceptGrouping:
        return ConceptGrouping(
            skill_groups=[
                self._group("required_skill", [("v1", "Python"), ("v2", "Python"), ("v3", "Python")]),
                self._group("required_skill", [("v1", "Postgres"), ("v2", "PostgreSQL")]),
                ConceptGroup(
                    members=[
                        ConceptMember(source_id="v2", text="Docker", kind="preferred_skill"),
                        ConceptMember(source_id="v3", text="Docker", kind="required_skill"),
                    ]
                ),
                self._group("required_skill", [("v3", "Git")]),
            ],
            responsibility_groups=[
                self._group(
                    "responsibility",
                    [
                        ("v1", "Разрабатывать API"),
                        ("v2", "Разработка REST API"),
                        ("v3", "Создавать API"),
                    ],
                ),
                self._group("responsibility", [("v3", "Мониторинг сервисов")]),
            ],
        )

    @staticmethod
    def _group(kind: str, values: list[tuple[str, str]]) -> ConceptGroup:
        return ConceptGroup(
            members=[
                ConceptMember(source_id=source_id, text=text, kind=kind)
                for source_id, text in values
            ]
        )

    @staticmethod
    def _raw(source_id: str, title: str, text: str) -> RawVacancy:
        return RawVacancy(
            source="test",
            source_id=source_id,
            url=f"https://jobs.test/{source_id}",
            title=title,
            employer="Test",
            published_at=None,
            raw_text=text,
        )


if __name__ == "__main__":
    unittest.main()
