"""Тесты краткого кандидатского ответа без секций работодателя."""

import unittest

from generate_recommendation import (
    GeneratedRecommendation,
    RecommendationResult,
    build_result,
)


class ShortRecommendationTests(unittest.TestCase):
    def test_schema_has_no_employer_or_interview_sections(self) -> None:
        generated_fields = GeneratedRecommendation.model_fields
        result_fields = RecommendationResult.model_fields

        for field_name in ("employer_recommendation", "interview_questions"):
            self.assertNotIn(field_name, generated_fields)
            self.assertNotIn(field_name, result_fields)

    def test_builds_candidate_focused_result(self) -> None:
        generated = GeneratedRecommendation(
            summary="Резюме хорошо соответствует вакансии.",
            strengths=["Подтверждён опыт коммуникаций."],
            critical_gaps=["Пресс-релизы не подтверждены в резюме."],
            additional_gaps=["SMM не подтверждён в резюме."],
            candidate_recommendations=[
                (
                    "Если опыт подготовки пресс-релизов есть — добавьте "
                    "пример; если нет — подготовьте учебный кейс."
                )
            ],
        )

        result = build_result(
            {
                "candidate_name": None,
                "vacancy_title": "PR-менеджер",
                "total_score": 80,
                "recommendation": "Высокое соответствие вакансии",
            },
            generated,
        )

        self.assertEqual(result.total_score, 80)
        self.assertEqual(len(result.candidate_recommendations), 1)

    def test_limits_number_of_items(self) -> None:
        schema = GeneratedRecommendation.model_json_schema()

        self.assertEqual(schema["properties"]["strengths"]["maxItems"], 3)
        self.assertEqual(
            schema["properties"]["candidate_recommendations"]["maxItems"],
            3,
        )


if __name__ == "__main__":
    unittest.main()
