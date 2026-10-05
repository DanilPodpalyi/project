"""Проверки границ рынка, полноты текста и нового сценария из резюме."""

import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

from resume_evaluator.russian_market import (
    MarketScope, collect_russian_vacancies, normalize_full_descriptions,
    target_from_resume, build_russian_market_profile, validate_saved_russian_profile,
    russian_search_queries,
)
from resume_evaluator.resume_extractor import ResumeData, _retain_explicit_seniority
from resume_evaluator.profession import TargetProfile
from resume_evaluator.title_morphology import morphological_title_similarity
from resume_evaluator.market_profile_builder_v2 import rank_vacancies_v2, validate_relevance_v2
from resume_evaluator.candidate_assessment_pipeline import (
    CandidateReport, run_candidate_assessment, parse_arguments,
)
from resume_evaluator.synthetic_vacancy_pipeline_simplified import (
    AdjacentMarketContext, InsufficientMarketProfile, ExtractedVacancy, _market_source,
)
from test_market_profile_builder_v2 import raw, normalized


ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 1, 10, tzinfo=timezone.utc)
SCOPE = MarketScope(collected_at=NOW)
TARGET = TargetProfile(
    role="Аналитик", specialization="Аналитик информационной безопасности",
    domain="Информационная безопасность", seniority="Middle", role_family=None,
    focus=None, core_skills=["Личный навык кандидата"],
)
DESCRIPTION = (
    "<h2>Обязанности</h2><p>Анализ журналов событий и расследование инцидентов. "
    "Подготовка отчётов о результатах расследования.</p>"
    "<h2>Требования</h2><p>Знание SIEM, сетевых протоколов, Linux и методов анализа. "
    "Умение объяснить результаты исследования и подготовить рекомендации.</p>"
)


def detail(identifier: str, **changes) -> dict:
    result = {
        "id": identifier, "name": "Аналитик информационной безопасности",
        "alternate_url": f"https://hh.ru/vacancy/{identifier}",
        "published_at": (NOW - timedelta(days=30)).isoformat(),
        "description": DESCRIPTION, "country_code": "RU",
        "area": {"id": "1", "name": "Москва"},
        "employer": {"id": "100", "name": "Работодатель"}, "archived": False,
    }
    result.update(changes)
    return result


class RussianMarketTests(unittest.TestCase):
    def setUp(self):
        # Сетевые ответы и модель подменяются независимо; никаких live LLM в тестах.
        translation = patch("resume_evaluator.bilingual_titles.chat", side_effect=OSError("offline"))
        translation.start()
        self.addCleanup(translation.stop)

    def collect_import(self, items):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vacancies.json"
            path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
            return collect_russian_vacancies(
                TARGET, Path(directory) / "collection.json", scope=SCOPE,
                imported_path=path,
                request_json=lambda url: self.fail("Импорт не должен обращаться к сети"),
            )

    def test_morphology_preserves_specialization_and_handles_ambiguous_case(self):
        for title in ("Аналитика информационной безопасности",
                      "Аналитики по информационной безопасности",
                      "Middle аналитик информационной безопасности"):
            self.assertEqual(morphological_title_similarity(TARGET.label(), title), 1)
        self.assertLess(morphological_title_similarity(TARGET.label(), "Аналитик данных"), 1)

    def test_morphological_matches_reach_v2_without_embeddings(self):
        item = raw(1, "Аналитика информационной безопасности")
        ranked, available = rank_vacancies_v2(
            [item], TARGET, use_embeddings=False,
            title_matcher=morphological_title_similarity,
        )
        self.assertFalse(available)
        self.assertEqual(ranked[0].alias_score, 1)

    def test_english_graphic_designer_generates_russian_query_and_matches_titles(self):
        target = TARGET.model_copy(update={"role": "Designer", "specialization": "Graphic Designer"})
        self.assertIn("Графический дизайнер", russian_search_queries(target))
        self.assertEqual(morphological_title_similarity(target.label(), "Графического дизайнера"), 1)
        self.assertEqual(morphological_title_similarity("Графический дизайнер", "Middle Graphic Designer"), 1)
        self.assertLess(morphological_title_similarity(target.label(), "UI/UX дизайнер"), 1)

    def test_target_keeps_resume_level_and_discards_candidate_skills(self):
        resume = ResumeData.model_validate_json(
            (ROOT / "examples/precomputed/marketing_candidate.example.json").read_text(encoding="utf-8"))
        target = target_from_resume(resume)
        self.assertEqual(target.label(), "Маркетолог")
        self.assertEqual(target.seniority, "Middle")
        self.assertEqual(target.core_skills, [])
        resume.target_profession.seniority = None
        self.assertIsNone(target_from_resume(resume).seniority)
        resume.target_profession.specialization = None
        resume.target_profession.role = None
        resume.target_profession.domain = None
        with self.assertRaisesRegex(ValueError, "целевая должность"):
            target_from_resume(resume)

    def test_ml_search_expands_equivalent_titles_and_preserves_level(self):
        target = TARGET.model_copy(update={
            "specialization": "Machine Learning Engineer", "seniority": "Middle",
            "core_skills": ["TensorFlow", "Docker"],
        })
        queries = russian_search_queries(target)
        self.assertLessEqual(len(queries), 6)
        for query in ("Machine Learning Engineer", "ML Engineer", "ML-инженер",
                      "Инженер машинного обучения", "Инженер по машинному обучению",
                      "Machine Learning Engineer Middle"):
            self.assertIn(query, queries)
        self.assertFalse(any(skill in query for skill in target.core_skills for query in queries))

    def test_ml_equivalents_match_in_both_languages_and_cases(self):
        aliases = ("Machine Learning Engineer", "ML Engineer", "ML-инженер",
                   "Инженер машинного обучения", "Инженера по машинному обучению")
        for target in aliases:
            for title in aliases:
                with self.subTest(target=target, title=title):
                    self.assertEqual(morphological_title_similarity(target, title), 1)
        self.assertEqual(morphological_title_similarity("ML Engineer", "Middle ML-инженер"), 1)

    def test_ml_aliases_do_not_expand_to_adjacent_jobs_or_drop_specialization(self):
        for title in ("Data Scientist", "Data Engineer", "DevOps Engineer",
                      "Инженер по охране труда", "Machine Learning Analyst"):
            self.assertLess(morphological_title_similarity("ML Engineer", title), 1)
        self.assertLess(morphological_title_similarity(
            "Computer Vision ML Engineer", "Инженер машинного обучения"), 1)
        qualified = TARGET.model_copy(update={"specialization": "Computer Vision ML Engineer"})
        self.assertNotIn("Инженер машинного обучения", russian_search_queries(qualified))

    def test_ml_russian_alias_retrieves_full_vacancy_with_original_target(self):
        queries = []
        target = TARGET.model_copy(update={"specialization": "Machine Learning Engineer"})

        def fetch(url):
            parsed = urlparse(url)
            if parsed.path == "/areas/113":
                return {"id": "113", "areas": [{"id": "1", "areas": []}]}
            if parsed.path == "/vacancies":
                query = parse_qs(parsed.query)["text"][0]
                queries.append(query)
                return {"pages": 1, "items": ([{"id": "ml", "name": "ML-инженер"}]
                        if query == "Инженер машинного обучения" else [])}
            if parsed.path == "/vacancies/ml":
                return detail("ml", name="ML-инженер")
            self.fail(f"Неожиданный запрос: {url}")

        with tempfile.TemporaryDirectory() as directory:
            collection = collect_russian_vacancies(target, Path(directory) / "collection.json",
                scope=SCOPE, request_json=fetch, include_trudvsem=False, use_cache=False)
        self.assertEqual(len(queries), len(set(queries)))
        self.assertEqual(collection.diagnostics.accepted, 1)
        self.assertEqual(collection.diagnostics.target.label(), "Machine Learning Engineer")
        self.assertEqual(collection.diagnostics.target.seniority, "Middle")
        self.assertEqual(collection.diagnostics.target.core_skills, [])
        self.assertEqual(collection.vacancies[0].morphology_similarity, 1)

    def test_ml_alias_matching_does_not_admit_senior_jobs_for_middle(self):
        target = TARGET.model_copy(update={"specialization": "Machine Learning Engineer"})
        ranked, _ = rank_vacancies_v2(
            [raw(index, title) for index, title in enumerate((
                "Middle ML-инженер", "Старший ML-инженер", "Ведущий ML-инженер",
                "Главный инженер-программист (Data Engineer / ML Engineer)",
            ), start=1)], target, use_embeddings=False,
            title_matcher=morphological_title_similarity,
        )
        assessments, _, _ = validate_relevance_v2(ranked, target, use_llm=False)
        labels = {item.source_id: item.relevance for item in assessments}
        self.assertEqual(labels[raw(1, "").source_id], "direct")
        for index in (2, 3, 4):
            with self.subTest(index=index):
                self.assertEqual(labels[raw(index, "").source_id], "irrelevant")

    def test_unknown_title_translation_is_used_in_live_search_and_saved_in_diagnostics(self):
        target = TARGET.model_copy(update={"specialization": "Data Analyst"})
        queries = []

        def fetch(url):
            parsed = urlparse(url)
            if parsed.path == "/areas/113":
                return {"id": "113", "areas": [{"id": "1", "areas": []}]}
            if parsed.path == "/vacancies":
                query = parse_qs(parsed.query)["text"][0]
                queries.append(query)
                return {"pages": 1, "items": ([{"id": "da", "name": "Аналитик данных"}]
                    if query == "Аналитик данных" else [])}
            return detail("da", name="Аналитик данных")

        translated = SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "russian": "Аналитик данных", "english": "Data Analyst", "equivalent": True})))
        with tempfile.TemporaryDirectory() as directory, patch(
            "resume_evaluator.bilingual_titles.chat", return_value=translated,
        ) as model:
            path = Path(directory) / "collection.json"
            first = collect_russian_vacancies(target, path, scope=SCOPE, request_json=fetch, include_trudvsem=False)
            second = collect_russian_vacancies(target, path, scope=SCOPE, request_json=fetch, include_trudvsem=False)
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("Data Analyst", queries)
        self.assertIn("Аналитик данных", queries)
        self.assertEqual(first.diagnostics.search_titles.mode, "llm")
        self.assertEqual(second.diagnostics.search_titles.mode, "cache")
        self.assertEqual(saved["diagnostics"]["search_titles"]["russian"], "Аналитик данных")
        self.assertEqual(first.diagnostics.target.label(), "Data Analyst")
        self.assertFalse(first.diagnostics.target.core_skills)
        self.assertEqual(first.vacancies[0].morphology_similarity, 0.84)
        model.assert_called_once()

    def test_no_translation_and_no_level_keeps_minimum_search_plan_size(self):
        target = TARGET.model_copy(update={"specialization": "Data Analyst", "seniority": None})
        with tempfile.TemporaryDirectory() as directory:
            imported = Path(directory) / "vacancies.json"
            imported.write_text("[]", encoding="utf-8")
            result = build_russian_market_profile(target, Path(directory) / "profile.json",
                imported_path=imported, scope=SCOPE, use_llm=False, use_embeddings=False)
        self.assertEqual(result.status, "insufficient_market_data")

    def test_inferred_level_is_removed_but_explicit_russian_alias_is_kept(self):
        resume = ResumeData.model_validate_json(
            (ROOT / "examples/precomputed/marketing_candidate.example.json").read_text(encoding="utf-8"))
        _retain_explicit_seniority(resume, "Маркетолог, стаж 5 лет, сложные проекты")
        self.assertIsNone(resume.target_profession.seniority)
        resume.target_profession.seniority = "Middle"
        _retain_explicit_seniority(resume, "Маркетолог, уровень миддл")
        self.assertEqual(resume.target_profession.seniority, "Middle")

    def test_russia_period_dates_archive_and_sparse_text_are_enforced(self):
        result = self.collect_import([
            detail("valid"),
            detail("boundary", published_at=SCOPE.date_from.isoformat()),
            detail("foreign", country_code="KZ"),
            detail("old", published_at=(SCOPE.date_from - timedelta(seconds=1)).isoformat()),
            detail("future", published_at=(NOW + timedelta(seconds=1)).isoformat()),
            detail("unknown_date", published_at=None),
            detail("archive", archived=True),
            detail("snippet", description="Краткая поисковая выдержка"),
            detail("empty", description=""),
        ])
        self.assertEqual(result.diagnostics.accepted, 2)
        self.assertEqual(result.diagnostics.sparse_descriptions, 2)
        self.assertEqual({item.source_id for item in result.vacancies
                          if item.description_status == "full"}, {"import:valid", "import:boundary"})
        reasons = {item.reason for item in result.diagnostics.exclusions}
        self.assertIn("outside_russia_or_unverified_country", reasons)
        self.assertIn("outside_120_days_or_unverified_date", reasons)
        self.assertIn("archived", reasons)
        self.assertIn("description_sparse", reasons)

    def test_import_saves_full_original_and_deduplicates_repeated_ad(self):
        long = DESCRIPTION * 20 + "<p>Последнее требование: портфолио.</p>"
        result = self.collect_import([detail("one", description=long), detail("one", description=long)])
        self.assertEqual(result.diagnostics.accepted, 1)
        self.assertEqual(result.diagnostics.duplicates, 1)
        self.assertEqual(len(result.duplicate_copies), 1)
        self.assertEqual(result.duplicate_copies[0].original_description, long)
        self.assertEqual(result.vacancies[0].original_description, long)
        self.assertGreater(len(result.vacancies[0].raw_text), 2500)
        self.assertIn("Последнее требование", result.vacancies[0].raw_text)

    def test_hh_pagination_filters_country_and_fetches_detail(self):
        requests = []

        def fetch(url):
            requests.append(url)
            parsed = urlparse(url)
            if parsed.path == "/areas/113":
                return {"id": "113", "areas": [{"id": "1", "areas": []}]}
            if parsed.path == "/vacancies":
                parameters = parse_qs(parsed.query)
                self.assertEqual(parameters["area"], ["113"])
                self.assertEqual(parameters["search_field"], ["name"])
                self.assertEqual(parameters["date_from"], [SCOPE.date_from.isoformat()])
                self.assertNotIn("remote", parameters["text"][0])
                page = int(parameters["page"][0])
                return {"pages": 2, "items": [
                    {"id": str(page + 1), "name": "Аналитика информационной безопасности"}]}
            identifier = parsed.path.rsplit("/", 1)[-1]
            return detail(identifier, area={"id": "1" if identifier == "1" else "outside"})

        with tempfile.TemporaryDirectory() as directory:
            result = collect_russian_vacancies(
                TARGET, Path(directory) / "collection.json", scope=SCOPE,
                request_json=fetch, max_pages=2, use_cache=False,
                include_trudvsem=False,
            )
        self.assertEqual(result.diagnostics.accepted, 1)
        self.assertTrue(any("page=1" in url for url in requests))
        self.assertEqual(sum(urlparse(url).path == "/vacancies/1" for url in requests), 1)
        self.assertGreater(result.diagnostics.search_results, 2)
        self.assertEqual(result.diagnostics.duplicates, result.diagnostics.search_results - 2)

    def test_hh_403_is_reported_and_does_not_trigger_foreign_fallback(self):
        def fetch(url):
            raise HTTPError(url, 403, "Forbidden", None, None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "collection.json"
            result = collect_russian_vacancies(TARGET, path, scope=SCOPE,
                                               request_json=fetch, use_cache=False,
                                               include_trudvsem=False)
            self.assertTrue(path.exists())
        self.assertEqual(result.diagnostics.status, "unavailable")
        self.assertEqual(result.diagnostics.requests, 1)
        self.assertEqual(result.diagnostics.errors[0].http_status, 403)
        self.assertEqual(result.vacancies, [])

    def test_recent_request_cache_prevents_repeated_network_calls(self):
        calls = []
        def fetch(url):
            calls.append(url)
            if "/areas/" in url:
                return {"id": "113", "areas": []}
            return {"items": [], "pages": 0}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "collection.json"
            collect_russian_vacancies(TARGET, path, scope=SCOPE, request_json=fetch)
            count = len(calls)
            result = collect_russian_vacancies(TARGET, path, scope=SCOPE, request_json=fetch)
        self.assertEqual(len(calls), count)
        self.assertGreater(result.diagnostics.cache_hits, 0)

    def test_trudvsem_continues_after_hh_error_and_uses_creation_date(self):
        def fetch(url):
            if "api.hh.ru" in url:
                raise HTTPError(url, 403, "Forbidden", None, None)
            valid = {
                "id": "valid", "job-name": TARGET.label(), "vac_url": "https://trudvsem.ru/v/valid",
                "creation-date": "2026-09-01", "date_modify": "2026-10-01T10:00:00+0300",
                "region": {"region_code": "7700000000000", "name": "Москва"},
                "company": {"companycode": "123", "name": "Компания"},
                "duty": DESCRIPTION, "requirement": "Знание SIEM и Linux",
            }
            old = dict(valid, id="old", **{"creation-date": "2024-01-01"})
            return {"status": "200", "results": {"vacancies": [
                {"vacancy": valid}, {"vacancy": old}]}}
        with tempfile.TemporaryDirectory() as directory:
            result = collect_russian_vacancies(TARGET, Path(directory) / "collection.json",
                                               scope=SCOPE, request_json=fetch, use_cache=False)
        self.assertEqual(result.diagnostics.status, "partial")
        self.assertEqual(result.diagnostics.accepted_by_source, {"trudvsem": 1})
        self.assertEqual(result.vacancies[0].publication_precision, "day")
        self.assertEqual(result.vacancies[0].source, "trudvsem")
        self.assertEqual(result.vacancies[0].region, "Москва")
        self.assertIn("Знание SIEM", result.vacancies[0].raw_text)
        self.assertTrue(any(item.reason == "outside_120_days_or_unverified_date"
                            for item in result.diagnostics.exclusions))

    def test_empty_trudvsem_search_is_not_a_provider_error(self):
        def fetch(url):
            if "/areas/" in url:
                return {"id": "113", "areas": []}
            if "api.hh.ru" in url:
                return {"items": [], "pages": 0}
            return {"status": "200", "meta": {"total": 0}, "results": {"vacancies": None}}
        with tempfile.TemporaryDirectory() as directory:
            result = collect_russian_vacancies(TARGET, Path(directory) / "collection.json",
                                               scope=SCOPE, request_json=fetch, use_cache=False)
        self.assertEqual(result.diagnostics.errors, [])
        self.assertEqual(result.diagnostics.accepted, 0)

    def test_imported_full_descriptions_reach_complete_market_profile(self):
        items = [detail(str(index), employer={"id": str(index), "name": f"Компания {name}"},
                        description=DESCRIPTION + f"<p>Подразделение компании {index}.</p>")
                 for index, name in enumerate(("Альфа", "Бета", "Гамма"))]
        def request(batch, target, **kwargs):
            extracted = ExtractedVacancy.model_validate(normalized(batch[0]).model_dump())
            extracted.profession = target
            extracted.professional_skills = ["Расследование инцидентов"]
            extracted.tools_and_technologies = ["SIEM"]
            return type("Batch", (), {"vacancies": [extracted]})()
        with tempfile.TemporaryDirectory() as directory:
            imported = Path(directory) / "import.json"
            imported.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
            output = Path(directory) / "market.json"
            with patch("resume_evaluator.russian_market._request_normalized_batch", side_effect=request):
                result = build_russian_market_profile(TARGET, output, imported_path=imported,
                    scope=SCOPE, use_embeddings=False, use_llm=False)
            self.assertTrue(output.with_name("market.collection.json").exists())
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.sample_size, 3)
        self.assertEqual(result.tools_and_technologies[0].name, "SIEM")
        self.assertIn("Россия", result.summary)

    def test_full_normalizer_receives_tail_and_has_separate_cache(self):
        item = raw(1)
        item.raw_text = "Требования: " + DESCRIPTION * 25 + "Обязателен редкий инструмент в конце."
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.json"
            checkpoint = Path(directory) / "checkpoint.json"
            with patch("resume_evaluator.russian_market._request_normalized_batch") as request:
                request.return_value.vacancies = [ExtractedVacancy.model_validate(normalized(item).model_dump())]
                first = normalize_full_descriptions([item], TARGET, checkpoint_path=checkpoint, cache_path=cache)
                second = normalize_full_descriptions([item], TARGET, checkpoint_path=checkpoint, cache_path=cache)
            request.assert_called_once()
            self.assertGreater(request.call_args.kwargs["max_description_chars"], len(item.raw_text))
            self.assertEqual(len(first), 1)
            self.assertEqual(len(second), 1)

    def test_failed_model_or_oversized_text_has_explicit_diagnostics(self):
        item = raw(1)
        item.raw_text = "x" * 24001
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            with patch("resume_evaluator.russian_market._request_normalized_batch") as request:
                result = normalize_full_descriptions([item], TARGET,
                    checkpoint_path=checkpoint, cache_path=Path(directory) / "cache.json")
            request.assert_not_called()
            issues = json.loads(checkpoint.read_text(encoding="utf-8"))["issues"]
        self.assertEqual(result, [])
        self.assertEqual(issues[0]["reason"], "full_text_exceeds_model_budget")

    def test_full_normalizer_cache_separates_titles_and_reuses_identical_inputs(self):
        first = raw(1).model_copy(update={"title": "Middle Graphic Designer"})
        senior = raw(2).model_copy(update={"title": "Senior Graphic Designer", "raw_text": first.raw_text})
        copy = first.model_copy(update={"source_id": "copy-id", "url": "https://other.test/copy"})

        def extract(batch, target, **kwargs):
            item = batch[0]
            result = ExtractedVacancy.model_validate(normalized(item).model_dump())
            result.profession = result.profession.model_copy(update={
                "seniority": "Senior" if "Senior" in item.title else "Middle",
            })
            return type("Batch", (), {"vacancies": [result]})()

        with tempfile.TemporaryDirectory() as directory, patch(
            "resume_evaluator.russian_market._request_normalized_batch", side_effect=extract,
        ) as request:
            result = normalize_full_descriptions(
                [first, senior, copy], TARGET, checkpoint_path=Path(directory) / "checkpoint.json",
                cache_path=Path(directory) / "cache.json",
            )
        self.assertEqual(request.call_count, 2)
        self.assertEqual([item.profession.seniority for item in result], ["Middle", "Senior", "Middle"])
        self.assertEqual(result[-1].source_id, copy.source_id)

    def test_two_argument_cli_uses_resume_target_and_legacy_cli_still_parses(self):
        with patch.object(sys, "argv", ["assessment", "resume.json", "output/report"]):
            self.assertIsNone(parse_arguments().target_profession)
        with patch.object(sys, "argv", ["assessment", "resume.json", "Маркетолог", "output/report"]):
            self.assertEqual(parse_arguments().target_profession, "Маркетолог")

    def test_unverified_saved_market_cannot_bypass_russian_scope(self):
        profile = InsufficientMarketProfile(target=TARGET.model_dump(), sample_size=0,
                                            message="Нет данных", summary="Нет данных")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "происхождением"):
                validate_saved_russian_profile(profile, TARGET, Path(directory) / "missing.json")

    def test_saved_adjacent_sources_also_require_russian_collection_provenance(self):
        collection = self.collect_import([detail("one")])
        profile = InsufficientMarketProfile(
            target=TARGET.model_dump(), sample_size=0, message="Нет прямых", summary="Нет прямых",
            adjacent_context=AdjacentMarketContext(
                sample_size=1, sources=[_market_source(normalized(raw(999)))], summary="Смежные",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "collection.json"
            path.write_text(collection.model_dump_json(), encoding="utf-8")
            with patch("resume_evaluator.russian_market.MarketScope", return_value=SCOPE):
                with self.assertRaisesRegex(ValueError, "Источники"):
                    validate_saved_russian_profile(profile, TARGET, path)
                profile.adjacent_context = None
                validate_saved_russian_profile(profile, TARGET, path)

    def test_candidate_pipeline_uses_resume_target_in_russian_mode(self):
        unavailable = InsufficientMarketProfile(
            target=TARGET.model_dump(), sample_size=0, message="Источник недоступен", summary="Нет данных")
        with tempfile.TemporaryDirectory() as directory, patch(
            "resume_evaluator.candidate_assessment_pipeline.build_russian_market_profile",
            return_value=unavailable,
        ) as builder:
            paths = run_candidate_assessment(
                ROOT / "examples/precomputed/marketing_candidate.example.json", None, Path(directory),
                resume_evaluation_path=ROOT / "examples/precomputed/marketing_resume_evaluation.example.json",
                use_comparison_embeddings=False, use_recommendation_llm=False,
            )
            report = CandidateReport.model_validate_json(paths.report_json.read_text(encoding="utf-8"))
        target = builder.call_args.args[0]
        self.assertEqual(target.label(), "Маркетолог")
        self.assertEqual(target.seniority, "Middle")
        self.assertEqual(target.core_skills, [])
        self.assertEqual(report.status, "partial")


if __name__ == "__main__":
    unittest.main()
