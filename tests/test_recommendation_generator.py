"""Тесты безопасного генератора рекомендаций по готовым фактам."""

import json
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from resume_evaluator.candidate_market_comparison import (
    CandidateMarketComparison,
    CandidateSkillObservation,
    ComparisonMetrics,
    MarketDataUnavailableComparison,
    RequirementComparison,
    ResumeEvidence,
)
from resume_evaluator.recommendation_generator import (
    RecommendationResult,
    generate_candidate_recommendations,
)
from resume_evaluator.resume_extractor import ResumeData
from resume_evaluator.resume_quality_evaluator import (
    ResumeQualityResult,
    SectionEvaluation,
)
from resume_evaluator.synthetic_vacancy_pipeline_simplified import MarketTarget, MarketSource, MarketRequirementEvidence, MarketReliability


TARGET = MarketTarget(
    role="Marketer",
    specialization="Маркетолог",
    domain="Marketing",
    seniority="Middle",
    role_family="Marketing",
    focus="Campaign Management",
)


def resume() -> ResumeData:
    target_profession = TARGET.model_dump()
    target_profession["core_skills"] = ["Python"]
    return ResumeData.model_validate(
        {
            "candidate_name": "Тестовый кандидат",
            "target_profession": target_profession,
            "professional_summary": "Маркетолог",
            "contact_channels": ["email"],
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
    )


def section(
    score: float,
    max_score: float,
    *,
    strengths: list[str] | None = None,
    weaknesses: list[str] | None = None,
) -> SectionEvaluation:
    return SectionEvaluation(
        score=score,
        max_score=max_score,
        justification="Раздел проверен по содержанию резюме.",
        strengths=strengths or [],
        weaknesses=weaknesses or [],
        recommendations=["Добавить конкретный подтверждаемый пример."],
    )


def evaluation() -> ResumeQualityResult:
    return ResumeQualityResult(
        candidate_name="Тестовый кандидат",
        evaluation_goal="Насколько полно составлено резюме?",
        total_score=70,
        quality_level="Хорошее резюме, требующее доработки",
        overall_summary="Резюме понятно, но отдельным разделам не хватает фактов.",
        sections={
            "experience": section(
                17,
                25,
                strengths=["Опыт изложен последовательно."],
                weaknesses=["Мало измеримых результатов."],
            ),
            "projects": section(12, 20, weaknesses=["Проекты описаны кратко."]),
            "header": section(13, 15, strengths=["Целевая роль указана ясно."]),
            "summary": section(10, 15),
            "skills": section(11, 15),
            "education": section(7, 10),
        },
    )


def requirement(
    name: str,
    status: str,
    *,
    importance: str = "common",
    evidence: str | None = None,
) -> RequirementComparison:
    return RequirementComparison(
        category="tools_and_technologies",
        name=name,
        market_frequency=3 if importance == "common" else 2,
        market_coverage=1.0 if importance == "common" else 2 / 3,
        market_importance=importance,
        resume_status=status,
        resume_evidence=(
            [ResumeEvidence(section="experience", text=evidence)]
            if evidence
            else []
        ),
    )


def comparison(
    *,
    strong: list[RequirementComparison] | None = None,
    missing_common: list[RequirementComparison] | None = None,
    missing_additional: list[RequirementComparison] | None = None,
    mentioned: list[RequirementComparison] | None = None,
    candidate_only: list[str] | None = None,
) -> CandidateMarketComparison:
    strong = strong or []
    missing_common = missing_common or []
    missing_additional = missing_additional or []
    mentioned = mentioned or []
    tools = [*strong, *missing_common, *missing_additional, *mentioned]
    common = [item for item in tools if item.market_importance == "common"]
    additional = [
        item for item in tools if item.market_importance == "additional"
    ]
    return CandidateMarketComparison(
        target=TARGET,
        market_confidence="medium",
        market_sample_size=3,
        professional_skills=[],
        tools_and_technologies=tools,
        responsibilities=[],
        education_requirements=[],
        domain_knowledge=[],
        soft_skills=[],
        strong_matches=strong,
        mentioned_only_requirements=mentioned,
        missing_common_requirements=missing_common,
        missing_additional_requirements=missing_additional,
        candidate_only_skills=[
            CandidateSkillObservation(
                name=name,
                resume_status="mentioned_only",
                evidence=[],
                note="Не оказался распространённым в выборке.",
            )
            for name in (candidate_only or [])
        ],
        unconfirmed_resume_skills=[],
        responsibility_matches=[],
        responsibility_gaps=[],
        education_match=[],
        domain_knowledge_match=[],
        market_variations=[],
        summary_metrics=ComparisonMetrics(
            common_requirements_total=len(common),
            common_requirements_matched=sum(
                item.resume_status != "not_found" for item in common
            ),
            common_requirements_evidenced=sum(
                item.resume_status == "evidenced" for item in common
            ),
            common_requirements_mentioned_only=sum(
                item.resume_status == "mentioned_only" for item in common
            ),
            additional_requirements_total=len(additional),
            additional_requirements_matched=sum(
                item.resume_status != "not_found" for item in additional
            ),
            additional_requirements_evidenced=sum(
                item.resume_status == "evidenced" for item in additional
            ),
            candidate_only_skills_total=len(candidate_only or []),
            responsibility_requirements_total=0,
            responsibility_requirements_matched=0,
        ),
    )


class RecommendationGeneratorTests(unittest.TestCase):
    def test_soft_skills_are_listing_suggestions_without_projects_or_development(self):
        skill = requirement("аккуратность", "not_found")
        skill.category = "soft_skills"
        result = self.generate(comparison(missing_common=[skill]))
        advice = result.market_gaps[0]
        self.assertIsNone(advice.action)
        self.assertIn("Рекомендуется указать", advice.title)
        self.assertNotIn("учебн", advice.text)
        self.assertEqual(result.development_directions, [])

    def test_unconfirmed_resume_skills_outside_market_are_counted_without_market_claims(self):
        market = comparison(mentioned=[requirement("Photoshop", "mentioned_only")])
        market.unconfirmed_resume_skills = [CandidateSkillObservation(
            name=name, resume_status="mentioned_only", evidence=[], note="Перечислен без примера")
            for name in ("Photoshop", "Basic Motion Design")]
        result = self.generate(market)
        self.assertIn("навыков без практического подтверждения: 2", result.final_summary)
        motion = next(item for item in result.unconfirmed_skills if item.basis[0].reference == "Basic Motion Design")
        self.assertIsNone(motion.basis[0].market_frequency)
        self.assertIsNone(motion.confidence)

    def test_unconfirmed_total_is_not_limited_to_five_advice_items(self):
        market = comparison()
        market.unconfirmed_resume_skills = [CandidateSkillObservation(
            name=f"Tool {i}", resume_status="mentioned_only", evidence=[], note="Нет примера") for i in range(8)]
        result = self.generate(market)
        self.assertEqual(len(result.unconfirmed_skills), 5)
        self.assertIn("навыков без практического подтверждения: 8", result.final_summary)

    def test_unconfirmed_total_deduplicates_adobe_and_short_product_names(self):
        market = comparison(mentioned=[requirement("InDesign", "mentioned_only")])
        market.unconfirmed_resume_skills = [CandidateSkillObservation(
            name="Adobe InDesign", resume_status="mentioned_only", evidence=[], note="Нет примера")]
        result = self.generate(market)
        self.assertIn("навыков без практического подтверждения: 1", result.final_summary)
        self.assertEqual(len(result.unconfirmed_skills), 1)
    def test_concrete_application_is_preferred_over_bare_technology_name(self) -> None:
        item = requirement("Python", "evidenced", evidence="Python")
        item.resume_evidence.append(ResumeEvidence(section="experience", text="Автоматизировал подготовку отчётов на Python."))
        result = self.generate(comparison(strong=[item]))
        advice = next(row for row in result.strengths if row.basis[0].reference == "Python")
        self.assertEqual(advice.basis[0].evidence[0], "Автоматизировал подготовку отчётов на Python.")
        self.assertIn("Автоматизировал подготовку отчётов", advice.text)

    def test_low_confidence_reaches_advice_with_sources_frequency_and_resume_evidence(self) -> None:
        item = requirement("Python", "evidenced", evidence="Использовал Python для автоматизации.")
        item.market_sources = [MarketSource(
            source="test", source_id=f"source-{index}", title="Middle Marketer", employer=f"Company {index}",
            url=f"https://jobs.test/{index}", published_at="2026-09-20T10:00:00+03:00", target_relevance=1,
        ) for index in range(3)]
        item.market_evidence = [MarketRequirementEvidence(source_id="source-0", text="Знание Python")]
        item.market_source_values = ["Python", "Знание Python"]
        market = comparison(strong=[item])
        market.market_confidence = "low"
        result = self.generate(market)
        advice = next(row for row in result.strengths if row.basis[0].reference == "Python")
        self.assertEqual(result.market_context.confidence, "low")
        self.assertEqual(advice.confidence, "low")
        self.assertTrue(advice.text.startswith("Предварительный совет"))
        self.assertEqual(advice.action.type, "highlight_evidence")
        self.assertIn("Поднимите реальный пример", advice.action.steps[0])
        self.assertEqual(advice.basis[0].market_frequency, 3)
        self.assertEqual(advice.basis[0].market_sample_size, 3)
        self.assertEqual(advice.basis[0].market_sources, item.market_sources)
        self.assertEqual(advice.basis[0].market_evidence, item.market_evidence)
        self.assertEqual(advice.basis[0].evidence, ["Использовал Python для автоматизации."])
        self.assertIn("низкая", result.final_summary)
        self.assertIsNone(item.market_confidence)  # Не меняем входное сравнение.

    def test_mentioned_and_missing_require_different_actions(self) -> None:
        result = self.generate(comparison(mentioned=[requirement("Python", "mentioned_only")],
                                          missing_common=[requirement("SQL", "not_found")]))
        mention, gap = result.unconfirmed_skills[0], result.market_gaps[0]
        self.assertEqual(mention.action.type, "add_application_example")
        self.assertIn("одну реальную задачу", mention.action.steps[0])
        self.assertEqual(gap.action.type, "clarify_or_develop")
        self.assertIn("реальный опыт", gap.action.steps[0])
        self.assertIn("Если опыт есть", gap.action.steps[1])
        self.assertIn("Если опыта нет", gap.action.steps[2])
        self.assertIn("учебный проект", gap.action.steps[2])
        self.assertEqual(gap.basis[0].resume_status, "not_found")

    def test_missing_education_cannot_be_replaced_by_a_practice_project(self) -> None:
        education = requirement("Высшее техническое образование", "not_found")
        education.category = "education_requirements"
        result = self.generate(comparison(missing_common=[education]))
        self.assertIn("диплом", result.market_gaps[0].action.steps[-1])
        self.assertNotIn("выполните учебную задачу", " ".join(result.market_gaps[0].action.steps))

    def test_responsibility_gap_receives_an_action(self) -> None:
        market = comparison()
        obligation = requirement("Готовить отчёты", "not_found")
        obligation.category = "responsibilities"
        market.responsibility_gaps = [obligation]
        result = self.generate(market)
        self.assertEqual(result.market_gaps[0].basis[0].category, "responsibilities")
        self.assertEqual(result.market_gaps[0].action.type, "clarify_or_develop")

    @patch("resume_evaluator.recommendation_generator.chat")
    def test_llm_cannot_remove_low_confidence_or_structured_action(self, chat_mock: MagicMock) -> None:
        market = comparison(missing_common=[requirement("SQL", "not_found")])
        market.market_confidence = "low"
        market.market_reliability = MarketReliability(
            confidence="low", provisional=True, evaluated_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
            lookback_days=120, sample_size=3, independent_employers=3, known_employer_count=3,
            direct_count=3, full_description_count=3, recent_publication_count=3,
            dated_publication_count=3, reliable_relevance_count=3, provider_count=1,
        )
        safe = self.generate(market)
        payload = {section: [{"fact_id": row.fact_id, "title": row.title,
                             "text": row.text.removeprefix("Предварительный совет по ограниченной выборке. ")}
                            for row in getattr(safe, section)]
                   for section in ("strengths", "resume_improvements", "market_gaps", "unconfirmed_skills", "development_directions")}
        response = MagicMock()
        response.message.content = json.dumps(payload, ensure_ascii=False)
        chat_mock.return_value = response
        result = generate_candidate_recommendations(resume(), evaluation(), market, output_path=None)
        self.assertEqual(result.generation_mode, "llm")
        self.assertEqual(result.market_gaps[0].confidence, "low")
        self.assertTrue(result.market_gaps[0].text.startswith("Предварительный совет"))
        self.assertEqual(result.market_gaps[0].action, safe.market_gaps[0].action)
        self.assertEqual(result.market_gaps[0].basis, safe.market_gaps[0].basis)
        chat_mock.assert_called_once()
        sent = json.loads(chat_mock.call_args.kwargs["messages"][1]["content"].split("\n\n", 1)[1])
        self.assertEqual(sent["market_context"]["reliability"]["evaluated_at"], "2026-10-01T00:00:00Z")

    def generate(self, market_comparison):
        return generate_candidate_recommendations(
            resume(),
            evaluation(),
            market_comparison,
            output_path=None,
            use_llm=False,
        )

    def test_result_schema_contains_no_market_fit_score(self) -> None:
        fields = set(RecommendationResult.model_fields)

        self.assertIn("resume_score", fields)
        self.assertNotIn("market_score", fields)
        self.assertNotIn("fit_score", fields)
        self.assertNotIn("total_score", fields)

    def test_strong_candidate_has_supported_strengths_and_no_fake_gaps(self) -> None:
        result = self.generate(
            comparison(
                strong=[
                    requirement(
                        "Python",
                        "evidenced",
                        evidence="Использовал Python для автоматизации.",
                    ),
                    requirement(
                        "SQL",
                        "evidenced",
                        evidence="Подготовил отчёт с помощью SQL.",
                    ),
                ]
            )
        )

        self.assertEqual(result.market_gaps, [])
        self.assertEqual(result.unconfirmed_skills, [])
        names = {item.basis[0].reference for item in result.strengths}
        self.assertTrue({"Python", "SQL"}.issubset(names))

    def test_partial_match_uses_not_found_in_resume_wording(self) -> None:
        result = self.generate(
            comparison(
                strong=[
                    requirement(
                        "Python",
                        "evidenced",
                        evidence="Автоматизация на Python.",
                    )
                ],
                missing_common=[requirement("SQL", "not_found")],
                missing_additional=[
                    requirement(
                        "Docker",
                        "not_found",
                        importance="additional",
                    )
                ],
            )
        )

        self.assertEqual(
            [item.basis[0].reference for item in result.market_gaps],
            ["SQL", "Docker"],
        )
        self.assertTrue(
            all("не обнаружено" in item.text for item in result.market_gaps)
        )
        self.assertTrue(all("Если у вас есть" in item.text for item in result.market_gaps))

    def test_skills_only_candidate_is_asked_for_evidence_not_accused(self) -> None:
        result = self.generate(
            comparison(mentioned=[requirement("HubSpot", "mentioned_only")])
        )

        item = result.unconfirmed_skills[0]
        self.assertIn("указан", item.text)
        self.assertIn("применение не показано", item.text)
        self.assertNotIn("не владе", item.text.casefold())

    def test_market_unavailable_leaves_only_resume_quality_advice(self) -> None:
        unavailable = MarketDataUnavailableComparison(
            target=TARGET,
            market_sample_size=1,
            message="Недостаточно данных",
        )

        result = self.generate(unavailable)

        self.assertFalse(result.market_context.available)
        self.assertEqual(result.market_gaps, [])
        self.assertEqual(result.unconfirmed_skills, [])
        self.assertEqual(result.development_directions, [])
        self.assertTrue(result.resume_improvements)

    def test_candidate_only_skill_is_not_criticized(self) -> None:
        result = self.generate(comparison(candidate_only=["Adobe Illustrator"]))

        output = result.model_dump_json()
        self.assertNotIn("Adobe Illustrator", output)

    @patch("resume_evaluator.recommendation_generator.chat")
    def test_valid_llm_wording_uses_one_call(self, chat_mock: MagicMock) -> None:
        market = comparison(
            missing_common=[requirement("SQL", "not_found")]
        )
        safe = self.generate(market)
        payload = {
            section: [
                {
                    "fact_id": item.fact_id,
                    "title": item.title,
                    "text": item.text,
                }
                for item in getattr(safe, section)
            ]
            for section in (
                "strengths",
                "resume_improvements",
                "market_gaps",
                "unconfirmed_skills",
                "development_directions",
            )
        }
        response = MagicMock()
        response.message.content = json.dumps(payload, ensure_ascii=False)
        chat_mock.return_value = response

        result = generate_candidate_recommendations(
            resume(),
            evaluation(),
            market,
            output_path=None,
        )

        self.assertEqual(result.generation_mode, "llm")
        self.assertEqual(result.llm_calls, 1)
        chat_mock.assert_called_once()

    @patch("resume_evaluator.recommendation_generator.chat")
    def test_invalid_json_is_repaired_once(self, chat_mock: MagicMock) -> None:
        market = comparison(
            missing_common=[requirement("SQL", "not_found")]
        )
        safe = self.generate(market)
        valid_payload = {
            section: [
                {
                    "fact_id": item.fact_id,
                    "title": item.title,
                    "text": item.text,
                }
                for item in getattr(safe, section)
            ]
            for section in (
                "strengths",
                "resume_improvements",
                "market_gaps",
                "unconfirmed_skills",
                "development_directions",
            )
        }
        invalid_response = MagicMock()
        invalid_response.message.content = "{broken json"
        valid_response = MagicMock()
        valid_response.message.content = json.dumps(
            valid_payload,
            ensure_ascii=False,
        )
        chat_mock.side_effect = [invalid_response, valid_response]

        result = generate_candidate_recommendations(
            resume(),
            evaluation(),
            market,
            output_path=None,
        )

        self.assertEqual(result.generation_mode, "llm")
        self.assertEqual(result.llm_calls, 2)

    @patch("resume_evaluator.recommendation_generator.chat")
    def test_unavailable_llm_returns_fallback(self, chat_mock: MagicMock) -> None:
        chat_mock.side_effect = RuntimeError("Ollama unavailable")

        result = generate_candidate_recommendations(
            resume(),
            evaluation(),
            comparison(missing_common=[requirement("SQL", "not_found")]),
            output_path=None,
        )

        self.assertEqual(result.generation_mode, "fallback")
        self.assertEqual(result.llm_calls, 1)
        self.assertTrue(result.market_gaps)

    @patch("resume_evaluator.recommendation_generator.chat")
    def test_hallucinated_recommendations_fail_post_validation(
        self,
        chat_mock: MagicMock,
    ) -> None:
        hallucinations = (
            ("Kubernetes", "Изучите Kubernetes"),
            ("invented_experience", "Вы разработали систему рекомендаций"),
            ("invented_education", "У вас высшее техническое образование"),
            ("invented_achievement", "Вы увеличили продажи на 50%"),
        )
        market = comparison(missing_common=[requirement("SQL", "not_found")])
        safe = self.generate(market)
        base_payload = {
            section: [
                {
                    "fact_id": item.fact_id,
                    "title": item.title,
                    "text": item.text,
                }
                for item in getattr(safe, section)
            ]
            for section in (
                "strengths",
                "resume_improvements",
                "market_gaps",
                "unconfirmed_skills",
                "development_directions",
            )
        }

        for fact_id, text in hallucinations:
            with self.subTest(fact_id=fact_id):
                payload = json.loads(json.dumps(base_payload, ensure_ascii=False))
                payload["market_gaps"][0]["text"] = text
                response = MagicMock()
                response.message.content = json.dumps(
                    payload,
                    ensure_ascii=False,
                )
                chat_mock.reset_mock()
                chat_mock.return_value = response

                result = generate_candidate_recommendations(
                    resume(),
                    evaluation(),
                    market,
                    output_path=None,
                )

                self.assertEqual(result.generation_mode, "fallback")
                self.assertEqual(result.llm_calls, 2)
                self.assertNotIn(text, result.model_dump_json())
                self.assertEqual(chat_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
