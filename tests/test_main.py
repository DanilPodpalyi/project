"""Тесты последовательного запуска пяти этапов анализа."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from main import PipelinePaths, run_pipeline


class MainPipelineTests(unittest.TestCase):
    def test_builds_predictable_output_paths(self) -> None:
        output_directory = Path("result")

        paths = PipelinePaths.from_output_directory(output_directory)

        self.assertEqual(paths.resume, output_directory / "resume.json")
        self.assertEqual(paths.vacancy, output_directory / "vacancy.json")
        self.assertEqual(paths.score, output_directory / "score.json")
        self.assertEqual(
            paths.short_recommendation,
            output_directory / "recommendation_short.json",
        )
        self.assertEqual(
            paths.detailed_recommendation,
            output_directory / "recommendation_detailed.json",
        )

    @patch("main.save_detailed_result")
    @patch("main.build_detailed_result")
    @patch("main.generate_detailed_recommendation")
    @patch("main.save_short_result")
    @patch("main.build_short_result")
    @patch("main.generate_short_recommendation")
    @patch("main.save_score_result")
    @patch("main.score_candidate")
    @patch("main.save_vacancy_result")
    @patch("main.extract_vacancy")
    @patch("main.read_vacancy")
    @patch("main.save_resume_result")
    @patch("main.extract_resume")
    @patch("main.read_resume")
    def test_runs_all_five_stages(
        self,
        read_resume_mock: MagicMock,
        extract_resume_mock: MagicMock,
        save_resume_mock: MagicMock,
        read_vacancy_mock: MagicMock,
        extract_vacancy_mock: MagicMock,
        save_vacancy_mock: MagicMock,
        score_candidate_mock: MagicMock,
        save_score_mock: MagicMock,
        generate_short_mock: MagicMock,
        build_short_mock: MagicMock,
        save_short_mock: MagicMock,
        generate_detailed_mock: MagicMock,
        build_detailed_mock: MagicMock,
        save_detailed_mock: MagicMock,
    ) -> None:
        resume = MagicMock()
        resume.model_dump.return_value = {"resume": "data"}
        vacancy = MagicMock()
        vacancy.model_dump.return_value = {"vacancy": "data"}
        score = MagicMock(total_score=80, recommendation="Высокое соответствие")
        score.model_dump.return_value = {"score": "data"}
        extract_resume_mock.return_value = resume
        extract_vacancy_mock.return_value = vacancy
        score_candidate_mock.return_value = score

        with tempfile.TemporaryDirectory() as directory:
            paths = run_pipeline(
                Path("resume.docx"),
                Path("vacancy.pdf"),
                Path(directory),
            )

        read_resume_mock.assert_called_once_with(Path("resume.docx"))
        read_vacancy_mock.assert_called_once_with(Path("vacancy.pdf"))
        save_resume_mock.assert_called_once_with(resume, paths.resume)
        save_vacancy_mock.assert_called_once_with(vacancy, paths.vacancy)
        save_score_mock.assert_called_once_with(score, paths.score)
        save_short_mock.assert_called_once_with(
            build_short_mock.return_value,
            paths.short_recommendation,
        )
        save_detailed_mock.assert_called_once_with(
            build_detailed_mock.return_value,
            paths.detailed_recommendation,
        )


if __name__ == "__main__":
    unittest.main()
