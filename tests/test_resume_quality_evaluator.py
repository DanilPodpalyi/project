"""Тесты независимой оценки качества резюме без вакансии."""

"""Тесты независимой оценки качества резюме."""

import json
import unittest
from unittest.mock import MagicMock, patch

from pydantic import ValidationError

from resume_evaluator.resume_quality_evaluator import (
    EVALUATION_GOAL,
    GeneratedResumeEvaluation,
    build_result,
    evaluate_resume,
)


class ResumeQualityEvaluationTests(unittest.TestCase):
    def test_generated_schema_enforces_section_limits(self) -> None:
        payload = self._payload()
        payload["experience"]["score"] = 25.1

        with self.assertRaises(ValidationError):
            GeneratedResumeEvaluation.model_validate(payload)

    def test_total_is_programmatic_sum_of_six_sections(self) -> None:
        generated = GeneratedResumeEvaluation.model_validate(self._payload())

        result = build_result(generated)

        self.assertEqual(result.total_score, 75.0)
        self.assertEqual(result.max_score, 100)
        self.assertEqual(result.evaluation_goal, EVALUATION_GOAL)
        self.assertEqual(
            list(result.sections),
            [
                "experience",
                "projects",
                "header",
                "summary",
                "skills",
                "education",
            ],
        )
        self.assertEqual(
            result.total_score,
            round(sum(item.score for item in result.sections.values()), 1),
        )

    def test_full_section_scores_cannot_exceed_100_total(self) -> None:
        payload = self._payload()
        for section_name, score in {
            "experience": 25,
            "projects": 20,
            "header": 15,
            "summary": 15,
            "skills": 15,
            "education": 10,
        }.items():
            payload[section_name]["score"] = score

        result = build_result(
            GeneratedResumeEvaluation.model_validate(payload),
        )

        self.assertEqual(result.total_score, 100.0)

    @patch("resume_evaluator.resume_quality_evaluator.chat")
    def test_model_receives_resume_without_vacancy(
        self,
        chat_mock: MagicMock,
    ) -> None:
        response = MagicMock()
        response.message.content = json.dumps(
            self._payload(),
            ensure_ascii=False,
        )
        chat_mock.return_value = response

        result = evaluate_resume("Python-разработчик, опыт 3 года")

        self.assertEqual(result.candidate_name, "Иван Петров")
        call = chat_mock.call_args.kwargs
        user_message = call["messages"][1]["content"]
        self.assertIn("Python-разработчик, опыт 3 года", user_message)
        self.assertIn("Не сравнивай его с вакансией", user_message)
        self.assertNotIn('"vacancy"', user_message)

    @staticmethod
    def _payload() -> dict:
        def section(score: float) -> dict:
            return {
                "justification": "Раздел оценён по указанным сведениям.",
                "strengths": ["Есть конкретная информация."],
                "weaknesses": ["Не хватает измеримого результата."],
                "recommendations": [
                    "Добавить подтверждённый результат с метрикой."
                ],
                "score": score,
            }

        return {
            "candidate_name": "Иван Петров",
            "overall_summary": "Резюме понятно, но требует доработки.",
            "experience": section(20),
            "projects": section(15),
            "header": section(10),
            "summary": section(10),
            "skills": section(12),
            "education": section(8),
        }


if __name__ == "__main__":
    unittest.main()
