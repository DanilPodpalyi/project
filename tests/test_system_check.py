"""Тесты быстрой проверки системы без обращения к модели."""

"""Тесты локальной проверки системы."""

import tempfile
import unittest
from pathlib import Path

from pydantic import BaseModel

from resume_evaluator.system_check import read_model, validate_required_schema
from resume_evaluator.vacancy_extractor import VacancyData


class LightweightSystemCheckTests(unittest.TestCase):
    def test_accepts_complete_schema(self) -> None:
        validate_required_schema(VacancyData)

    def test_rejects_schema_with_defaulted_field(self) -> None:
        class IncompleteSchema(BaseModel):
            required_value: str
            silently_defaulted_value: list[str] = []

        with self.assertRaisesRegex(ValueError, "silently_defaulted_value"):
            validate_required_schema(IncompleteSchema)

    def test_validates_json_without_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            file_path = Path(directory) / "vacancy.json"
            file_path.write_text(
                VacancyData(
                    title="PR-менеджер",
                    responsibilities=[],
                    required_skills=["PR"],
                    preferred_skills=[],
                    experience_requirement=None,
                    education_level=None,
                    education_field=None,
                    employment_type=None,
                    work_format=None,
                    conditions=[],
                ).model_dump_json(),
                encoding="utf-8",
            )

            result = read_model(file_path, VacancyData)

        self.assertEqual(result.title, "PR-менеджер")


if __name__ == "__main__":
    unittest.main()
