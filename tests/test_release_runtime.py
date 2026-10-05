"""Гарантии поставки: запуск, сохранение, лимиты, конфигурация и отсутствие подмены."""
import io
import json
import os
import tempfile
import time
import unittest
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock, patch

import httpx

from resume_evaluator import model_runtime, runtime
from resume_evaluator.cli import main
from resume_evaluator.settings import get_settings
from resume_evaluator.system_check import check_models
from scripts.build_release import release_files

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Result:
    report_json: Path
    report_markdown: Path


@runtime.managed_run
def task(resume_path, output_directory, failure=None):
    runtime.update_stage("checking")
    if failure:
        raise failure
    report = output_directory / "candidate_report.json"
    markdown = output_directory / "candidate_report.md"
    runtime.atomic_write_text(report, json.dumps({"status": "partial"}))
    runtime.atomic_write_text(markdown, "Отчёт")
    return Result(report, markdown)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "input.txt"
        self.input.write_text("вымышленный кандидат", encoding="utf-8")
        self.output = self.root / "output"
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_repeated_runs_keep_previous_report_and_separate_identifiers(self):
        first = task(self.input, self.output)
        before = first.report_json.read_bytes()
        second = task(self.input, self.output)
        self.assertNotEqual(first.report_json.parent, second.report_json.parent)
        self.assertEqual(before, first.report_json.read_bytes())
        states = [json.loads((row.report_json.parent / "run.json").read_text()) for row in (first, second)]
        self.assertNotEqual(states[0]["run_id"], states[1]["run_id"])
        self.assertEqual(states[0]["report_status"], "partial")
        self.assertEqual(states[0]["status"], "completed")
        self.assertEqual(len(states[0]["inputs"]["resume_path"]["sha256"]), 64)

    def test_failure_keeps_stage_but_no_document_text(self):
        with self.assertRaises(ValueError):
            task(self.input, self.output, ValueError("private document content"))
        text = (self.output / "run.json").read_text()
        state = json.loads(text)
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["stage"], "checking")
        self.assertNotIn("private document content", text)
        self.assertFalse((self.output / ".run.lock").exists())
        self.assertIsNone(runtime.current_run_id())

    def test_interrupt_is_recorded_and_new_run_is_safe(self):
        with self.assertRaises(KeyboardInterrupt):
            task(self.input, self.output, KeyboardInterrupt())
        self.assertEqual(json.loads((self.output / "run.json").read_text())["status"], "interrupted")
        self.assertNotEqual(task(self.input, self.output).report_json.parent, self.output)

    def test_invalid_config_does_not_create_lock(self):
        with patch.dict(os.environ, {"RESUME_EVALUATOR_MAX_RUN_SECONDS": "NaN"}):
            with self.assertRaises(ValueError):
                task(self.input, self.output)
        self.assertFalse(self.output.exists())

    def test_stale_lock_is_preserved(self):
        self.output.mkdir()
        lock = self.output / ".run.lock"
        lock.write_text("old process")
        task(self.input, self.output)
        self.assertEqual(lock.read_text(), "old process")

    def test_legacy_report_is_preserved(self):
        self.output.mkdir()
        old = self.output / "resume.json"
        old.write_text("old resume")
        result = task(self.input, self.output)
        self.assertNotEqual(result.report_json.parent, self.output)
        self.assertEqual(old.read_text(), "old resume")

    def test_deadline_limits_request_and_raises_after_expiry(self):
        token = runtime._deadline.set(time.monotonic() + 2)
        try:
            self.assertLessEqual(runtime.remaining_timeout(600), 2)
        finally:
            runtime._deadline.reset(token)
        token = runtime._deadline.set(time.monotonic() - 1)
        try:
            with self.assertRaises(TimeoutError):
                runtime.remaining_timeout(600)
        finally:
            runtime._deadline.reset(token)

    def test_atomic_failure_does_not_replace_previous_file_or_leave_temp(self):
        target = self.root / "report.json"
        runtime.atomic_write_text(target, "before")
        with patch("resume_evaluator.runtime.os.replace", side_effect=OSError("locked")):
            with self.assertRaises(OSError):
                runtime.atomic_write_text(target, "after")
        self.assertEqual(target.read_text(), "before")
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_environment_overrides_file_and_secrets_are_not_serialized(self):
        config = self.root / "config.toml"
        config.write_text('[resume_evaluator]\nmodel="file-model"\nmax_run_seconds=20')
        with patch.dict(os.environ, {"RESUME_EVALUATOR_CONFIG": str(config), "RESUME_EVALUATOR_MODEL": "env-model", "HH_ACCESS_TOKEN": "secret"}):
            settings = get_settings()
        self.assertEqual(settings.model, "env-model")
        self.assertEqual(settings.max_run_seconds, 20)
        self.assertNotIn("secret", json.dumps(settings.public_dict()))

    def test_invalid_hosts_limits_and_unknown_configuration_are_rejected(self):
        for key, value in (("OLLAMA_HOST", "http://user:password@localhost"),
            ("RESUME_EVALUATOR_DETAIL_LIMIT", "201"), ("RESUME_EVALUATOR_OLLAMA_TIMEOUT", "inf")):
            with self.subTest(key=key), patch.dict(os.environ, {key: value}):
                with self.assertRaises(ValueError):
                    get_settings()
        config = self.root / "config.toml"
        config.write_text('[resume_evaluator]\nwrong_key=1')
        with patch.dict(os.environ, {"RESUME_EVALUATOR_CONFIG": str(config)}):
            with self.assertRaises(ValueError):
                get_settings()

    def test_model_failure_is_safe_and_client_is_closed(self):
        client = Mock()
        client.chat.side_effect = httpx.ReadTimeout("private data")
        with patch("resume_evaluator.model_runtime._client", return_value=client):
            with self.assertRaises(model_runtime.ModelUnavailableError) as error:
                model_runtime.chat(model="test", messages=[])
        self.assertNotIn("private data", str(error.exception))
        client._client.close.assert_called_once()

    def test_model_success_also_closes_client(self):
        client = Mock()
        client.embed.return_value = {"embeddings": [[1, 2]]}
        with patch("resume_evaluator.model_runtime._client", return_value=client):
            self.assertEqual(model_runtime.embed(model="test", input="test"), {"embeddings": [[1, 2]]})
        client._client.close.assert_called_once()

    def test_health_detects_missing_embedding_model(self):
        response = io.BytesIO(b'{"models":[{"name":"gemma4:12b"}]}')
        with patch("resume_evaluator.system_check.urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "bge-m3"):
                check_models()

    def test_cli_default_help_and_dispatch(self):
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main([]), 0)
        self.assertIn("analyze", output.getvalue())
        implementation = Mock()
        with patch("resume_evaluator.cli.importlib.import_module", return_value=implementation):
            self.assertEqual(main(["check"]), 0)
        implementation.main.assert_called_once()

    def test_cli_failure_returns_nonzero(self):
        implementation = Mock()
        implementation.main.side_effect = OSError("server unavailable")
        with patch("resume_evaluator.cli.importlib.import_module", return_value=implementation), redirect_stderr(io.StringIO()):
            self.assertEqual(main(["check"]), 2)

    def test_release_excludes_working_inputs_and_contains_synthetic_examples(self):
        files = {path.relative_to(ROOT).as_posix() for path in release_files(ROOT)}
        self.assertIn("examples/resumes/anna_sokolova.docx", files)
        self.assertIn("examples/resumes/alexey_morozov.docx", files)
        self.assertIn("requirements.lock", files)
        self.assertFalse(any(name.startswith(("output/", "data/", ".venv/", "examples/legacy/")) for name in files))
        self.assertNotIn("config.toml", files)
        self.assertNotIn(".env", files)

    def test_partial_report_explains_missing_sources_and_search_translation(self):
        from resume_evaluator.candidate_assessment_pipeline import run_candidate_assessment
        from resume_evaluator.market_profile import InsufficientMarketProfile
        def builder(target, output_path, **kwargs):
            result = InsufficientMarketProfile(target=target.model_dump(), sample_size=1,
                message="Недостаточно данных", summary="Нет основной выборки")
            collection = {"diagnostics": {"scope": {"country": "Россия", "lookback_days": 120},
                "status": "partial", "accepted": 10, "search_limit_reached": True,
                "search_titles": {"mode": "unavailable"},
                "errors": [{"source": "hh.ru", "reason": "HTTP 403"}]}}
            runtime.atomic_write_text(output_path.with_name("market_profile.collection.json"), json.dumps(collection))
            return result
        with patch("resume_evaluator.candidate_assessment_pipeline.build_russian_market_profile", side_effect=builder):
            result = run_candidate_assessment(ROOT / "examples/precomputed/marketing_candidate.example.json",
                None, self.output, resume_evaluation_path=ROOT / "examples/precomputed/marketing_resume_evaluation.example.json",
                use_comparison_embeddings=False, use_recommendation_llm=False)
        markdown = result.report_markdown.read_text(encoding="utf-8")
        self.assertIn("HTTP 403", markdown)
        self.assertIn("вакансий: 1", markdown)
        self.assertIn("минимум 3", markdown)
        self.assertIn("Перевод", markdown)
        report = json.loads(result.report_json.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["recommendations"]["market_gaps"], [])

    def test_ready_quality_for_another_candidate_is_rejected(self):
        from resume_evaluator.candidate_assessment_pipeline import run_candidate_assessment
        quality = json.loads((ROOT / "examples/precomputed/marketing_resume_evaluation.example.json").read_text(encoding="utf-8"))
        quality["candidate_name"] = "Другой кандидат"
        path = self.root / "quality.json"
        runtime.atomic_write_text(path, json.dumps(quality))
        with self.assertRaisesRegex(ValueError, "другому кандидату"):
            run_candidate_assessment(ROOT / "examples/precomputed/marketing_candidate.example.json",
                None, self.output, resume_evaluation_path=path)
        state = json.loads((self.output / "run.json").read_text())
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["stage"], "resume_evaluation")


if __name__ == "__main__":
    unittest.main()
