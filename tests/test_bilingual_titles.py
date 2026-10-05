"""Перевод только должности, кеш и отсутствие ложного exact alias."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from httpx import ReadTimeout

from resume_evaluator.bilingual_titles import (
    BilingualSearchTitles, resolve_bilingual_titles, bilingual_title_similarity,
)
from resume_evaluator.profession import target_profile_from_value
from resume_evaluator.russian_market import russian_search_queries
from resume_evaluator.market_profile_builder_v2 import rank_vacancies_v2, validate_relevance_v2
from test_market_profile_builder_v2 import raw


def response(russian="Аналитик данных", english="Data Analyst", equivalent=True):
    return SimpleNamespace(message=SimpleNamespace(content=json.dumps({
        "russian": russian, "english": english, "equivalent": equivalent,
    }, ensure_ascii=False)))


class BilingualTitlesTests(unittest.TestCase):
    def test_unknown_english_title_is_translated_and_cache_avoids_second_call(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "resume_evaluator.bilingual_titles.chat", return_value=response(),
        ) as model:
            first = resolve_bilingual_titles("Data Analyst", Path(directory))
            second = resolve_bilingual_titles("Data Analyst", Path(directory), use_llm=False)
        self.assertEqual(first.mode, "llm")
        self.assertEqual(first.russian, "Аналитик данных")
        self.assertEqual(second.mode, "cache")
        model.assert_called_once()
        self.assertEqual(json.loads(model.call_args.kwargs["messages"][1]["content"]), {"title": "Data Analyst"})

    def test_unknown_russian_title_preserves_all_source_words(self):
        title = "Менеджер по работе с ключевыми клиентами"
        with tempfile.TemporaryDirectory() as directory, patch(
            "resume_evaluator.bilingual_titles.chat", return_value=response(title, "Key Account Manager"),
        ):
            result = resolve_bilingual_titles(title, Path(directory))
        self.assertEqual(result.mode, "llm")
        self.assertEqual(result.russian, title)
        self.assertEqual(result.english, "Key Account Manager")

    def test_queries_include_both_languages_and_level_without_candidate_skills(self):
        target = target_profile_from_value("SQL Data Analyst").model_copy(update={
            "seniority": "Middle", "core_skills": ["Docker", "TensorFlow"],
        })
        titles = BilingualSearchTitles(original=target.label(), mode="llm",
            russian="SQL аналитик данных", english="SQL Data Analyst")
        queries = russian_search_queries(target, search_titles=titles)
        for query in ("SQL Data Analyst", "SQL аналитик данных",
                      "SQL Data Analyst Middle", "SQL аналитик данных Middle"):
            self.assertIn(query, queries)
        self.assertLessEqual(len(queries), 6)
        self.assertEqual(len(queries), len({query.casefold() for query in queries}))
        self.assertFalse(any(skill in query for skill in target.core_skills for query in queries))

    def test_dictionary_translation_does_not_call_model(self):
        with tempfile.TemporaryDirectory() as directory, patch("resume_evaluator.bilingual_titles.chat") as model:
            titles = resolve_bilingual_titles("Machine Learning Engineer", Path(directory))
        model.assert_not_called()
        self.assertEqual(titles.mode, "dictionary")
        self.assertEqual(titles.russian, "Инженер машинного обучения")

    def test_mixed_specialization_is_preserved_in_both_queries(self):
        title = "Инженер Computer Vision"
        with tempfile.TemporaryDirectory() as directory, patch(
            "resume_evaluator.bilingual_titles.chat", return_value=response(title, "Computer Vision Engineer"),
        ):
            titles = resolve_bilingual_titles(title, Path(directory))
        self.assertEqual(titles.mode, "llm")
        queries = russian_search_queries(target_profile_from_value(title), search_titles=titles)
        self.assertIn("Computer Vision Engineer", queries)
        self.assertNotIn("Engineer", queries)

    def test_suspicious_translations_fall_back_and_are_not_cached(self):
        bad = (
            response("Аналитик", "Analyst"),  # потерян исходный Data Analyst
            response("Аналитик данных", "Data Analyst", equivalent=False),
            response("Data Analyst", "Data Analyst"),  # нет русского варианта
            response("Аналитик данных", "Аналитик данных"),
            response("Middle аналитик данных", "Data Analyst"),  # добавлен уровень
        )
        for translated in bad:
            with self.subTest(translated=translated), tempfile.TemporaryDirectory() as directory, patch(
                "resume_evaluator.bilingual_titles.chat", return_value=translated,
            ):
                result = resolve_bilingual_titles("Data Analyst", Path(directory))
                self.assertEqual(result.mode, "unavailable")
                self.assertEqual(result.variants(), ["Data Analyst"])
                self.assertFalse(list(Path(directory).glob("*.json")))

    def test_technical_qualifier_cannot_disappear_in_translation(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "resume_evaluator.bilingual_titles.chat", return_value=response("Разработчик", "C++ Developer"),
        ):
            titles = resolve_bilingual_titles("C++ Developer", Path(directory))
        self.assertEqual(titles.mode, "unavailable")

    def test_uppercase_profession_words_are_not_technical_identifiers(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "resume_evaluator.bilingual_titles.chat", return_value=response("Аналитик данных", "DATA ANALYST"),
        ):
            titles = resolve_bilingual_titles("DATA ANALYST", Path(directory))
        self.assertEqual(titles.mode, "llm")

    def test_translated_ml_specialization_does_not_repeat_original_in_parentheses(self):
        title = "Computer Vision ML Engineer"
        russian = "Инженер по машинному обучению в области компьютерного зрения"
        with tempfile.TemporaryDirectory() as directory, patch(
            "resume_evaluator.bilingual_titles.chat", return_value=response(f"{russian} ({title})", title),
        ):
            titles = resolve_bilingual_titles(title, Path(directory))
        self.assertEqual(titles.mode, "llm")
        self.assertEqual(titles.russian, russian)
        self.assertEqual(bilingual_title_similarity(title, russian, titles), 0.84)

    def test_timeout_is_diagnostic_and_next_run_retries(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "resume_evaluator.bilingual_titles.chat", side_effect=[ReadTimeout("timeout"), response()],
        ) as model:
            first = resolve_bilingual_titles("Data Analyst", Path(directory))
            second = resolve_bilingual_titles("Data Analyst", Path(directory))
        self.assertEqual(first.mode, "unavailable")
        self.assertEqual(first.issue, "translation_failed:ReadTimeout")
        self.assertEqual(second.mode, "llm")
        self.assertEqual(model.call_count, 2)

    def test_wrong_language_is_repaired_once(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "resume_evaluator.bilingual_titles.chat", side_effect=[response("Data Analyst"), response()],
        ) as model:
            result = resolve_bilingual_titles("Data Analyst", Path(directory))
        self.assertEqual(result.mode, "llm")
        self.assertEqual(result.russian, "Аналитик данных")
        self.assertEqual(model.call_count, 2)

    def test_uncertain_profession_is_not_forced_to_translate_by_retry(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "resume_evaluator.bilingual_titles.chat", return_value=response(equivalent=False),
        ) as model:
            result = resolve_bilingual_titles("Data Analyst", Path(directory))
        self.assertEqual(result.mode, "unavailable")
        model.assert_called_once()

    def test_changed_model_or_title_does_not_reuse_translation(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "resume_evaluator.bilingual_titles.chat", side_effect=[
                response(), response("Бухгалтер", "Accountant"), response(),
            ],
        ) as model:
            resolve_bilingual_titles("Data Analyst", Path(directory))
            resolve_bilingual_titles("Accountant", Path(directory))
            with patch("resume_evaluator.bilingual_titles.MODEL_NAME", "other-model"):
                result = resolve_bilingual_titles("Data Analyst", Path(directory))
        self.assertEqual(result.mode, "llm")
        self.assertEqual(model.call_count, 3)

    def test_corrupt_cached_translation_is_replaced(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "resume_evaluator.bilingual_titles.chat", return_value=response(),
        ) as model:
            resolve_bilingual_titles("Data Analyst", Path(directory))
            path = next(Path(directory).glob("*.json"))
            path.write_text("{", encoding="utf-8")
            self.assertEqual(resolve_bilingual_titles("Data Analyst", Path(directory)).mode, "llm")
        self.assertEqual(model.call_count, 2)

    def test_translation_alone_does_not_prove_profession_match(self):
        target = target_profile_from_value("Data Analyst")
        titles = BilingualSearchTitles(original=target.label(), mode="llm",
            russian="Аналитик данных", english="Data Analyst")
        score = bilingual_title_similarity(target.label(), "Аналитик данных", titles)
        self.assertEqual(score, 0.84)
        self.assertEqual(bilingual_title_similarity(target.label(), "Data Analyst", titles), 1)
        ranked, _ = rank_vacancies_v2([raw(1, "Аналитик данных")], target,
            use_embeddings=False, title_matcher=lambda a, b: bilingual_title_similarity(a, b, titles))
        labels, _, _ = validate_relevance_v2(ranked, target, use_llm=False)
        self.assertEqual(labels[0].relevance, "irrelevant")

    def test_disabled_or_unsupported_translation_does_not_call_model(self):
        with tempfile.TemporaryDirectory() as directory, patch("resume_evaluator.bilingual_titles.chat") as model:
            self.assertEqual(resolve_bilingual_titles("Data Analyst", Path(directory), use_llm=False).mode, "disabled")
            self.assertEqual(resolve_bilingual_titles("x" * 181, Path(directory)).mode, "unavailable")
        model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
