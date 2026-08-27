"""Тесты построения синтетической эталонной вакансии."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from resume_evaluator.candidate_scorer import score_candidate
from resume_evaluator.synthetic_vacancy_pipeline import (
    ConceptGroup,
    ConceptGrouping,
    ConceptMember,
    NormalizedVacancy,
    PipelineStatistics,
    ProfessionSearchPlan,
    RawVacancy,
    _normalize_batch_with_fallback,
    aggregate_synthetic_vacancy,
    calculate_relevance_scores,
    deduplicate_vacancies,
    select_best_three,
    run_pipeline,
    validate_and_complete_groups,
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

    def test_aggregates_frequencies_sources_and_median_experience(self) -> None:
        grouping = self._grouping()
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

        self.assertEqual(set(result.required_skills), {"Python", "Postgres", "Docker"})
        self.assertEqual(result.preferred_skills, ["Git"])
        self.assertEqual(result.responsibilities, ["Создавать API"])
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

    @patch(
        "resume_evaluator.synthetic_vacancy_pipeline.group_selected_concepts"
    )
    @patch("resume_evaluator.synthetic_vacancy_pipeline.select_best_three")
    @patch(
        "resume_evaluator.synthetic_vacancy_pipeline.assess_profession_relevance"
    )
    @patch("resume_evaluator.synthetic_vacancy_pipeline.normalize_vacancies")
    @patch("resume_evaluator.synthetic_vacancy_pipeline.search_public_vacancies")
    @patch("resume_evaluator.synthetic_vacancy_pipeline.build_search_plan")
    def test_pipeline_saves_compatible_json(
        self,
        plan_mock,
        search_mock,
        normalize_mock,
        assess_mock,
        select_mock,
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
        normalize_mock.return_value = self.selected
        assess_mock.return_value = None
        select_mock.return_value = self.selected
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
            compatible = VacancyData.model_validate_json(
                output_path.read_text(encoding="utf-8")
            )
            self.assertEqual(compatible.title, result.title)

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


if __name__ == "__main__":
    unittest.main()
