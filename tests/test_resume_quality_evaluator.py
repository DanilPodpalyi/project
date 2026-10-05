"""Тесты независимой оценки качества резюме без вакансии."""

"""Тесты независимой оценки качества резюме."""

import json
import unittest
from unittest.mock import MagicMock, patch

from pydantic import ValidationError

from resume_evaluator.resume_quality_evaluator import (
    EVALUATION_GOAL,
    GeneratedResumeEvaluation,
    audit_skill_claims,
    build_result,
    evaluate_resume,
    unsupported_skill_claims,
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

    def test_related_technology_is_not_skill_evidence(self) -> None:
        cases = [
            ("Keras", "PyTorch", "PyTorch"),
            ("Keras", "TensorFlow", "TensorFlow"),
            ("MySQL", "PostgreSQL", "PostgreSQL"),
            ("OpenCV", "Computer Vision", "Computer Vision"),
            ("scikit-learn", "PyTorch", "PyTorch"),
        ]

        for resume_skill, claimed_skill, expected in cases:
            with self.subTest(
                resume_skill=resume_skill,
                claimed_skill=claimed_skill,
            ):
                unsupported = unsupported_skill_claims(
                    f"Кандидат владеет {claimed_skill}.",
                    json.dumps({"skills": [resume_skill]}),
                )
                self.assertIn(expected, unsupported)

    def test_unsupported_positive_claim_is_removed_without_score_change(
        self,
    ) -> None:
        payload = self._payload()
        payload["skills"]["strengths"] = [
            "Keras, PyTorch и TensorFlow подтверждают сильный стек.",
            "Keras явно указан в навыках.",
        ]
        generated = GeneratedResumeEvaluation.model_validate(payload)

        audited = audit_skill_claims(
            generated,
            json.dumps({"skills": ["Keras"]}),
        )

        self.assertEqual(
            audited.skills.strengths,
            ["Keras явно указан в навыках."],
        )
        self.assertEqual(audited.skills.score, generated.skills.score)
        self.assertEqual(
            build_result(audited).total_score,
            build_result(generated).total_score,
        )

    def test_explicit_alias_and_concrete_project_evidence_are_kept(self) -> None:
        resume = json.dumps(
            {
                "skills": ["tf.keras"],
                "projects": [
                    {
                        "description": (
                            "Разработал модель классификации изображений"
                        )
                    }
                ],
            },
            ensure_ascii=False,
        )

        self.assertEqual(
            unsupported_skill_claims("Использует TensorFlow.", resume),
            [],
        )
        self.assertEqual(
            unsupported_skill_claims(
                "Есть компетенция Computer Vision.",
                resume,
            ),
            [],
        )

    def test_opencv_is_not_matched_by_short_cv_alias(self) -> None:
        resume = json.dumps({"skills": ["OpenCV"]})

        self.assertEqual(
            unsupported_skill_claims("Указан OpenCV.", resume),
            [],
        )
        self.assertEqual(
            unsupported_skill_claims("Есть Computer Vision.", resume),
            ["Computer Vision"],
        )

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

    @patch("resume_evaluator.resume_quality_evaluator.chat")
    def test_model_output_is_audited_before_return(
        self,
        chat_mock: MagicMock,
    ) -> None:
        payload = self._payload()
        payload["skills"]["strengths"] = [
            "TensorFlow подразумевается через Keras."
        ]
        response = MagicMock()
        response.message.content = json.dumps(payload, ensure_ascii=False)
        chat_mock.return_value = response

        result = evaluate_resume(json.dumps({"skills": ["Keras"]}))

        self.assertEqual(result.skills.strengths, [])

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
