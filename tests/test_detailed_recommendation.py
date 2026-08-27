"""Тесты подробной кандидатской рекомендации и её защитных правил."""

"""Тесты подробной рекомендации кандидату."""

import unittest

from pydantic import ValidationError

from resume_evaluator.detailed_recommendation import (
    DetailedGap,
    DevelopmentRecommendation,
    GeneratedDetailedRecommendation,
    ResumeRecommendation,
    DetailedStrength,
    correct_gap_categories,
    ensure_score_gaps,
    make_resume_recommendations_safe,
)


class DetailedRecommendationTests(unittest.TestCase):
    def test_schema_contains_only_candidate_sections(self) -> None:
        fields = GeneratedDetailedRecommendation.model_fields

        self.assertEqual(
            set(fields),
            {
                "overall_summary",
                "strengths",
                "gaps",
                "resume_recommendations",
                "development_recommendations",
            },
        )

    def test_gap_uses_one_of_three_evidence_categories(self) -> None:
        gap = DetailedGap(
            requirement="Подготовка пресс-релизов",
            category="не показан на примерах",
            analysis="В резюме указана работа с контентом без примеров релизов.",
            missing_evidence="Пример пресс-релиза или описание такой задачи.",
            priority="высокий",
        )

        self.assertEqual(gap.category, "не показан на примерах")

    def test_rejects_unknown_gap_category(self) -> None:
        with self.assertRaises(ValidationError):
            DetailedGap(
                requirement="SMM",
                category="навыка у кандидата нет",
                analysis="Нет данных.",
                missing_evidence="Пример проекта.",
                priority="средний",
            )

    def test_adds_every_missing_skill_from_score(self) -> None:
        generated = self._make_recommendation()

        ensure_score_gaps(
            generated,
            {
                "missing_required_skills": ["Пресс-релизы"],
                "missing_preferred_skills": ["SMM"],
            },
        )

        self.assertEqual(
            [gap.requirement for gap in generated.gaps],
            ["Пресс-релизы", "SMM"],
        )

    def test_removes_unverified_suggested_wording(self) -> None:
        generated = self._make_recommendation()
        generated.gaps.append(
            DetailedGap(
                requirement="Пресс-релизы",
                category="не указан в резюме",
                analysis="Нет прямого упоминания.",
                missing_evidence="Пример работы.",
                priority="высокий",
            )
        )
        generated.resume_recommendations.append(
            ResumeRecommendation(
                related_requirement="Пресс-релизы",
                section="Опыт работы",
                recommendation="Добавьте пресс-релизы.",
                suggested_wording="Готовил пресс-релизы для СМИ.",
                condition=None,
                priority="высокий",
            )
        )

        make_resume_recommendations_safe(generated)

        result = generated.resume_recommendations[-1]
        self.assertIsNone(result.suggested_wording)
        self.assertIn("Если у вас есть подтверждённый опыт", result.recommendation)
        self.assertIsNotNone(result.condition)

    def test_corrects_gap_with_only_indirect_evidence(self) -> None:
        generated = self._make_recommendation()
        generated.gaps.append(
            DetailedGap(
                requirement="подготовка пресс-релизов",
                category="не показан на примерах",
                analysis="Есть только коммуникационные материалы.",
                missing_evidence="Пример пресс-релиза.",
                priority="высокий",
            )
        )

        correct_gap_categories(
            generated,
            {"skills": ["Подготовка коммуникационных материалов"]},
        )

        self.assertEqual(
            generated.gaps[0].category,
            "подтверждён слабо или неочевидно",
        )

    @staticmethod
    def _make_recommendation() -> GeneratedDetailedRecommendation:
        return GeneratedDetailedRecommendation(
            overall_summary="Резюме соответствует вакансии.",
            strengths=[
                DetailedStrength(
                    strength="Коммуникации",
                    resume_evidence=["Переговоры с партнёрами"],
                    vacancy_relevance="Соответствует требованиям.",
                )
            ],
            gaps=[],
            resume_recommendations=[
                ResumeRecommendation(
                    related_requirement=None,
                    section="О себе",
                    recommendation="Добавьте подтверждённый результат.",
                    suggested_wording="Провёл переговоры с партнёрами.",
                    condition=None,
                    priority="средний",
                )
            ],
            development_recommendations=[
                DevelopmentRecommendation(
                    area="SMM",
                    recommendation="Освоить основы.",
                    practical_step="Подготовить учебный контент-план.",
                    expected_result="Готовый кейс.",
                    priority="низкий",
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
