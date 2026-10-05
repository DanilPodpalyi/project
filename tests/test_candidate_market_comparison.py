"""Тесты прозрачного сравнения ResumeData с готовым MarketProfile."""

import unittest
from unittest.mock import patch

from resume_evaluator.candidate_market_comparison import (
    compare_candidate_to_market,
)
from resume_evaluator.resume_extractor import ResumeData, Project
from resume_evaluator.synthetic_vacancy_pipeline_simplified import (
    InsufficientMarketProfile,
    MarketProfile,
    MarketRequirement,
    MarketSource,
    MarketTarget,
    MarketVariation,
)


TARGET = MarketTarget(
    role="Engineer",
    specialization="ML Engineer",
    domain="Machine Learning",
    seniority="Middle",
    role_family="ML Engineering",
    focus="Model Development / Production ML",
)

SOURCES = [
    MarketSource(
        source="test",
        source_id=f"v{index}",
        title="Middle ML Engineer",
        employer=f"Company {index}",
        url=f"https://jobs.test/{index}",
        published_at=None,
        target_relevance=0.9,
    )
    for index in range(1, 4)
]


def requirement(
    name: str,
    *,
    importance: str = "common",
    frequency: int = 3,
) -> MarketRequirement:
    return MarketRequirement(
        name=name,
        frequency=frequency,
        coverage=frequency / 3,
        importance=importance,
        sources=SOURCES[:frequency],
        source_values=[name],
    )


def market_profile() -> MarketProfile:
    return MarketProfile(
        target=TARGET,
        sample_size=3,
        professional_skills=[],
        tools_and_technologies=[
            requirement("Python"),
            requirement("SQL"),
            requirement("Docker"),
            requirement("Git", importance="additional", frequency=2),
        ],
        responsibilities=[requirement("Разработка ML-моделей")],
        education_requirements=[],
        domain_knowledge=[],
        soft_skills=[],
        market_variations=[
            MarketVariation(
                name="Kubernetes",
                category="tools_and_technologies",
                frequency=1,
                coverage=1 / 3,
                sources=SOURCES[:1],
                source_values=["Kubernetes"],
            )
        ],
        sources=SOURCES,
        confidence="medium",
        summary="Тестовый профиль рынка.",
    )


def resume(
    skills: list[str],
    *,
    responsibilities: list[str] | None = None,
    technologies: list[str] | None = None,
    professional_summary: str | None = None,
    education: list[dict] | None = None,
    courses: list[str] | None = None,
) -> ResumeData:
    return ResumeData.model_validate(
        {
            "candidate_name": "Test Candidate",
            "target_profession": {
                "role": "Engineer",
                "specialization": "ML Engineer",
                "domain": "Machine Learning",
                "seniority": "Middle",
                "role_family": "ML Engineering",
                "focus": "Model Development / Production ML",
                "core_skills": skills,
            },
            "professional_summary": professional_summary,
            "contact_channels": [],
            "skills": skills,
            "soft_skills": [],
            "work_experience": [
                {
                    "company": "Example",
                    "position": "ML Engineer",
                    "period": "2024–2026",
                    "responsibilities": responsibilities or [],
                    "technologies": technologies or [],
                    "achievements": [],
                }
            ],
            "education": education or [],
            "projects": [],
            "courses_and_certifications": courses or [],
            "languages": [],
            "achievements": [],
            "portfolio_links": [],
        }
    )


class CandidateMarketComparisonTests(unittest.TestCase):
    def test_tool_lists_in_experience_and_project_do_not_prove_application(self):
        candidate = resume(["Adobe Illustrator", "Adobe Photoshop", "Adobe InDesign", "Figma", "Basic Motion Design"],
                           technologies=["Adobe Photoshop"], responsibilities=["Собрала библиотеку шаблонов в Figma"])
        candidate.projects = [Project(name="Каталог", description=None, role=None, results=[],
                                      technologies=["Adobe Illustrator", "Adobe InDesign"])]
        profile = market_profile()
        profile.tools_and_technologies = [requirement(name) for name in ("Illustrator", "Photoshop", "InDesign", "Figma")]
        result = self.compare(candidate, profile)
        statuses = {item.name: item.resume_status for item in result.tools_and_technologies}
        self.assertEqual(statuses, {"Illustrator": "mentioned_only", "Photoshop": "mentioned_only", "InDesign": "mentioned_only", "Figma": "evidenced"})
        for item in result.tools_and_technologies[:3]:
            self.assertTrue(item.resume_evidence)
            self.assertNotIn(item, result.strong_matches)
        self.assertIn("Basic Motion Design", {item.name for item in result.unconfirmed_resume_skills})
        self.assertEqual(result.summary_metrics.unconfirmed_resume_skills_total, 4)

    def test_named_tool_in_project_task_is_practical_evidence(self):
        candidate = resume(["Illustrator"])
        candidate.projects = [Project(name="Айдентика", description="Разработала логотип в Adobe Illustrator",
                                      role=None, results=[], technologies=["Adobe Illustrator"])]
        profile = market_profile()
        profile.tools_and_technologies = [requirement("Adobe Illustrator")]
        result = self.compare(candidate, profile)
        self.assertEqual(result.tools_and_technologies[0].resume_status, "evidenced")
        self.assertEqual(result.tools_and_technologies[0].resume_evidence[0].text, "Разработала логотип в Adobe Illustrator")

    def test_known_translation_in_task_does_not_become_unconfirmed_candidate_skill(self):
        candidate = resume(["Typography"], responsibilities=["Разработала типографику и шаблоны коммуникаций"])
        profile = market_profile()
        profile.professional_skills = [requirement("Типографика")]
        result = self.compare(candidate, profile)
        self.assertEqual(result.professional_skills[0].resume_status, "evidenced")
        self.assertNotIn("Typography", {item.name for item in result.unconfirmed_resume_skills})
    def test_sentence_punctuation_does_not_hide_practical_evidence_or_merge_tools(self) -> None:
        profile = market_profile()
        profile.tools_and_technologies = [requirement(name) for name in ("Python", ".NET", "Node.js", "C++", "C#")]
        result = compare_candidate_to_market(resume([], responsibilities=[
            "Автоматизировал отчёты на Python.", "Разработал сервис на .NET.",
            "Использовал Node.js.", "Разработал модуль на C++.",
        ]), profile, output_path=None, use_embeddings=False)
        statuses = {item.name: item.resume_status for item in result.tools_and_technologies}
        self.assertEqual(statuses, {"Python": "evidenced", ".NET": "evidenced", "Node.js": "evidenced", "C++": "evidenced", "C#": "not_found"})

    def test_requirement_original_forms_participate_in_candidate_matching(self) -> None:
        profile = market_profile()
        profile.professional_skills = [requirement("Проверка программного продукта").model_copy(update={
            "source_values": ["Проверка программного продукта", "Тестирование приложения"],
        })]
        result = compare_candidate_to_market(
            resume(["Тестирование приложения"]), profile, output_path=None, use_embeddings=False,
        )
        self.assertEqual(result.professional_skills[0].resume_status, "mentioned_only")
        self.assertNotIn("Тестирование приложения", [item.name for item in result.candidate_only_skills])

    def test_known_translation_matches_without_embeddings(self) -> None:
        profile = market_profile()
        profile.professional_skills = [requirement("анализ журналов")]
        result = compare_candidate_to_market(
            resume(["log analysis"]), profile, output_path=None, use_embeddings=False,
        )
        self.assertEqual(result.professional_skills[0].resume_status, "mentioned_only")

    def compare(self, candidate: ResumeData, profile: MarketProfile | None = None):
        return compare_candidate_to_market(
            candidate,
            profile or market_profile(),
            output_path=None,
            use_embeddings=False,
        )

    def compare_with_identical_embeddings(
        self,
        candidate: ResumeData,
        profile: MarketProfile | None = None,
    ):
        def identical_vectors(texts: list[str], _model: str):
            return {text: [1.0, 0.0] for text in texts}

        with patch(
            "resume_evaluator.candidate_market_comparison._embed_text_map",
            side_effect=identical_vectors,
        ):
            return compare_candidate_to_market(
                candidate,
                profile or market_profile(),
                output_path=None,
                use_embeddings=True,
            )

    def cross_language_responsibility(
        self,
        requirement_name: str,
        candidate_text: str,
    ):
        profile = market_profile().model_copy(
            update={"responsibilities": [requirement(requirement_name)]}
        )
        return self.compare_with_identical_embeddings(
            resume([], responsibilities=[candidate_text]),
            profile,
        ).responsibilities[0]

    def test_cross_language_develop_brand_identity_is_evidenced(self) -> None:
        responsibility = self.cross_language_responsibility(
            "Develop brand identity",
            "Разрабатывала элементы фирменного стиля",
        )
        self.assertEqual(responsibility.resume_status, "evidenced")

    def test_cross_language_create_marketing_materials_is_evidenced(self) -> None:
        responsibility = self.cross_language_responsibility(
            "Create marketing materials",
            "Создавала digital-материалы для рекламных кампаний",
        )
        self.assertEqual(responsibility.resume_status, "evidenced")

    def test_cross_language_prepare_print_materials_is_evidenced(self) -> None:
        responsibility = self.cross_language_responsibility(
            "Prepare print materials",
            "Подготавливала макеты печатной продукции",
        )
        self.assertEqual(responsibility.resume_status, "evidenced")

    def test_cross_language_different_ml_action_is_not_evidenced(self) -> None:
        responsibility = self.cross_language_responsibility(
            "Develop ML models",
            "Анализировал результаты готовых ML-моделей",
        )
        self.assertEqual(responsibility.resume_status, "not_found")

    def test_cross_language_object_mention_without_action_is_not_evidenced(self) -> None:
        responsibility = self.cross_language_responsibility(
            "Manage advertising campaigns",
            "Реклама и рекламные кампании",
        )
        self.assertEqual(responsibility.resume_status, "not_found")

    def test_good_match_puts_evidenced_requirements_in_strong_matches(self) -> None:
        result = self.compare(
            resume(
                ["Python", "SQL", "Docker"],
                responsibilities=[
                    "Разрабатывал ML-модели, используя Python, SQL и Docker"
                ],
                technologies=["Python", "SQL", "Docker"],
            )
        )

        strong = {item.name: item.resume_status for item in result.strong_matches}
        self.assertEqual(strong["Python"], "evidenced")
        self.assertEqual(strong["SQL"], "evidenced")
        self.assertEqual(strong["Docker"], "evidenced")
        self.assertEqual(result.summary_metrics.common_requirements_evidenced, 3)

    def test_partial_match_keeps_not_found_as_resume_observation(self) -> None:
        result = self.compare(
            resume(
                ["Python"],
                responsibilities=["Использовал Python для разработки моделей"],
                technologies=["Python"],
            )
        )

        by_name = {item.name: item for item in result.tools_and_technologies}
        self.assertEqual(by_name["Python"].resume_status, "evidenced")
        self.assertEqual(by_name["SQL"].resume_status, "not_found")
        self.assertEqual(by_name["Docker"].resume_status, "not_found")
        self.assertEqual(
            {item.name for item in result.missing_common_requirements},
            {"SQL", "Docker"},
        )

    def test_skill_list_without_practical_evidence_is_mentioned_only(self) -> None:
        result = self.compare(resume(["Docker"]))

        docker = next(
            item for item in result.tools_and_technologies if item.name == "Docker"
        )
        self.assertEqual(docker.resume_status, "mentioned_only")
        self.assertIn("Docker", {
            item.name for item in result.mentioned_only_requirements
        })
        self.assertIn("Docker", {
            item.name for item in result.unconfirmed_resume_skills
        })
        self.assertNotIn("Docker", {
            item.name for item in result.missing_common_requirements
        })

    def test_candidate_only_skill_is_context_not_penalty(self) -> None:
        result = self.compare(
            resume(
                ["Python", "OpenCV"],
                technologies=["Python", "OpenCV"],
            )
        )

        self.assertIn("OpenCV", {
            item.name for item in result.candidate_only_skills
        })
        self.assertNotIn("OpenCV", {
            item.name for item in result.missing_common_requirements
        })

    def test_missing_market_variation_is_not_a_gap(self) -> None:
        result = self.compare(resume(["Python"], technologies=["Python"]))

        variation = next(
            item for item in result.market_variations if item.name == "Kubernetes"
        )
        self.assertEqual(variation.resume_status, "not_found")
        missing = {
            item.name
            for item in [
                *result.missing_common_requirements,
                *result.missing_additional_requirements,
            ]
        }
        self.assertNotIn("Kubernetes", missing)

    def test_comparison_does_not_mutate_market_profile(self) -> None:
        profile = market_profile()
        before = profile.model_dump_json()

        first = self.compare(resume(["Python"], responsibilities=["Автоматизировал отчёты на Python"], technologies=["Python"]), profile)
        second = self.compare(resume(["Docker"]), profile)

        self.assertNotEqual(first.summary_metrics, second.summary_metrics)
        self.assertEqual(profile.model_dump_json(), before)

    def test_unavailable_market_profile_stops_comparison(self) -> None:
        unavailable = InsufficientMarketProfile(
            target=TARGET,
            sample_size=1,
            message="Недостаточно данных",
            summary="Найдена одна вакансия",
            sources=SOURCES[:1],
        )

        result = compare_candidate_to_market(
            resume(["Python"]),
            unavailable,
            output_path=None,
            use_embeddings=False,
        )

        self.assertEqual(result.status, "market_data_unavailable")

    def test_summary_does_not_confirm_responsibility(self) -> None:
        profile = market_profile().model_copy(
            update={
                "responsibilities": [
                    requirement("Управление рекламными кампаниями")
                ]
            }
        )
        result = self.compare(
            resume(
                [],
                professional_summary="Опыт управления рекламными кампаниями",
            ),
            profile,
        )

        responsibility = result.responsibilities[0]
        self.assertEqual(responsibility.resume_status, "not_found")
        self.assertEqual(responsibility.resume_evidence, [])

    def test_summary_does_not_confirm_education_requirement(self) -> None:
        profile = market_profile().model_copy(
            update={
                "education_requirements": [
                    requirement("Высшее техническое образование")
                ]
            }
        )
        result = self.compare(
            resume(
                [],
                professional_summary="Имею профильное высшее образование",
            ),
            profile,
        )

        education_requirement = result.education_requirements[0]
        self.assertEqual(education_requirement.resume_status, "not_found")
        self.assertEqual(education_requirement.resume_evidence, [])

    def test_embedding_does_not_evidence_a_different_tool(self) -> None:
        result = self.compare_with_identical_embeddings(
            resume(
                ["Kubernetes"],
                technologies=["Kubernetes"],
            )
        )

        docker = next(
            item for item in result.tools_and_technologies if item.name == "Docker"
        )
        self.assertEqual(docker.resume_status, "not_found")
        self.assertEqual(docker.resume_evidence, [])

    def test_embedding_does_not_match_a_different_responsibility(self) -> None:
        result = self.compare_with_identical_embeddings(
            resume(
                [],
                responsibilities=[
                    "Мониторинг инфраструктуры и подготовка отчетов"
                ],
            )
        )

        responsibility = result.responsibilities[0]
        self.assertEqual(responsibility.resume_status, "not_found")
        self.assertEqual(responsibility.resume_evidence, [])

    def test_embedding_does_not_hide_candidate_only_skill(self) -> None:
        result = self.compare_with_identical_embeddings(
            resume(
                ["Terraform"],
                technologies=["Terraform"],
            )
        )

        self.assertIn(
            "Terraform",
            {item.name for item in result.candidate_only_skills},
        )

    def test_embedding_does_not_equate_tensorflow_and_pytorch(self) -> None:
        profile = market_profile().model_copy(
            update={
                "tools_and_technologies": [requirement("TensorFlow")],
                "market_variations": [],
            }
        )
        result = self.compare_with_identical_embeddings(
            resume(
                ["PyTorch"],
                technologies=["PyTorch"],
            ),
            profile,
        )

        tensorflow = result.tools_and_technologies[0]
        self.assertEqual(tensorflow.resume_status, "not_found")
        self.assertEqual(tensorflow.resume_evidence, [])
        self.assertIn(
            "PyTorch",
            {item.name for item in result.candidate_only_skills},
        )

    def test_embedding_does_not_equate_postgresql_and_mysql(self) -> None:
        profile = market_profile().model_copy(
            update={
                "tools_and_technologies": [requirement("PostgreSQL")],
                "market_variations": [],
            }
        )
        result = self.compare_with_identical_embeddings(
            resume(
                ["MySQL"],
                technologies=["MySQL"],
            ),
            profile,
        )

        postgresql = result.tools_and_technologies[0]
        self.assertEqual(postgresql.resume_status, "not_found")
        self.assertEqual(postgresql.resume_evidence, [])

    def test_embedding_rejects_same_domain_but_different_responsibility(self) -> None:
        result = self.compare_with_identical_embeddings(
            resume(
                [],
                responsibilities=[
                    "Анализ результатов работы готовых ML-моделей"
                ],
            )
        )

        responsibility = result.responsibilities[0]
        self.assertEqual(responsibility.name, "Разработка ML-моделей")
        self.assertEqual(responsibility.resume_status, "not_found")
        self.assertEqual(responsibility.resume_evidence, [])

    def test_embedding_keeps_explicit_tool_in_long_experience_as_evidence(self) -> None:
        profile = market_profile().model_copy(
            update={
                "tools_and_technologies": [requirement("PyTorch")],
                "market_variations": [],
            }
        )
        result = self.compare_with_identical_embeddings(
            resume(
                ["PyTorch"],
                responsibilities=[
                    "Обучал нейросети для классификации изображений с помощью PyTorch"
                ],
            ),
            profile,
        )

        pytorch = result.tools_and_technologies[0]
        self.assertEqual(pytorch.resume_status, "evidenced")
        self.assertTrue(
            any(
                "PyTorch" in evidence.text
                for evidence in pytorch.resume_evidence
            )
        )

    def test_embedding_batch_is_called_once_per_comparison(self) -> None:
        def distinct_vectors(texts: list[str], _model: str):
            return {
                text: [float(index + 1), 1.0]
                for index, text in enumerate(texts)
            }

        with patch(
            "resume_evaluator.candidate_market_comparison._embed_text_map",
            side_effect=distinct_vectors,
        ) as embed:
            compare_candidate_to_market(
                resume(["Python"], technologies=["Python"]),
                market_profile(),
                output_path=None,
                use_embeddings=True,
            )

        embed.assert_called_once()


if __name__ == "__main__":
    unittest.main()
