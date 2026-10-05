"""Уверенность определяется независимостью и подтверждёнными метаданными."""

import unittest
from datetime import datetime, timedelta, timezone

from resume_evaluator.market_reliability import (
    assess_market_reliability, deduplicate_market_vacancies, independent_employer_count,
)
from test_market_profile_builder_v2 import normalized, raw


NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)


def good_sample(size=6):
    return [normalized(raw(index)).model_copy(update={
        "employer": f"Компания {index}", "description_status": "full",
        "published_at": (NOW - timedelta(days=10)).isoformat(),
    }) for index in range(1, size + 1)]


def assess(sample, **kwargs):
    return assess_market_reliability(sample, {row.source_id: ("direct", "high") for row in sample},
                                    embeddings_available=True, llm_available=True, now=NOW, **kwargs)


class MarketReliabilityTests(unittest.TestCase):
    def test_high_requires_independent_complete_recent_direct_sources(self):
        result = assess(good_sample(), duplicates_removed=4)
        self.assertEqual(result.confidence, "high")
        self.assertFalse(result.provisional)
        self.assertEqual(result.independent_employers, 6)
        self.assertEqual(result.full_description_count, 6)
        self.assertEqual(result.recent_publication_count, 6)
        self.assertEqual(result.provider_count, 2)
        self.assertEqual(result.duplicates_removed, 4)

    def test_five_good_vacancies_are_medium_three_are_preliminary(self):
        self.assertEqual(assess(good_sample(5)).confidence, "medium")
        self.assertEqual(assess(good_sample(3)).confidence, "low")

    def test_one_employer_does_not_become_six_independent_observations(self):
        sample = [row.model_copy(update={"employer": "ООО Ромашка" if index % 2 else "Ромашка LLC"})
                  for index, row in enumerate(good_sample())]
        result = assess(sample)
        self.assertEqual(result.confidence, "low")
        self.assertEqual(result.independent_employers, 1)

    def test_unknown_employers_and_provider_scoped_ids(self):
        sample = [row.model_copy(update={"employer": None, "employer_id": "same-id"}) for row in good_sample()]
        self.assertEqual(independent_employer_count(sample), (2, 6))
        sample[0].employer_id = None
        result = assess(sample)
        self.assertEqual(result.known_employer_count, 5)
        self.assertEqual(result.confidence, "low")

    def test_same_provider_employer_id_joins_changed_company_names(self):
        sample = good_sample(2)
        sample[1].source = sample[0].source
        for row in sample:
            row.employer_id = "company-id"
        self.assertEqual(independent_employer_count(sample), (1, 2))

    def test_unconfirmed_fullness_or_freshness_lowers_confidence(self):
        for changes in ({"description_status": "unknown"}, {"description_status": "sparse"},
                        {"published_at": None}, {"published_at": "2026-09-01"},
                        {"published_at": (NOW - timedelta(days=121)).isoformat()},
                        {"published_at": (NOW + timedelta(days=1)).isoformat()}):
            with self.subTest(changes=changes):
                sample = good_sample()
                sample[0] = sample[0].model_copy(update=changes)
                self.assertEqual(assess(sample).confidence, "low")

    def test_one_provider_caps_otherwise_good_data_at_medium(self):
        sample = [row.model_copy(update={"source": "single"}) for row in good_sample()]
        self.assertEqual(assess(sample).confidence, "medium")

    def test_unreliable_relevance_or_adjacent_label_lowers_confidence(self):
        sample = good_sample()
        for label in (("direct", "low"), ("adjacent", "high")):
            labels = {row.source_id: ("direct", "high") for row in sample}
            labels[sample[0].source_id] = label
            result = assess_market_reliability(sample, labels, embeddings_available=True,
                                              llm_available=True, now=NOW)
            self.assertEqual(result.confidence, "low")

    def test_copies_across_providers_count_once_and_are_traceable(self):
        first = raw(1).model_copy(update={"employer": "ООО Ромашка", "raw_text": "Анализ логов, подготовка отчётов. " * 12})
        second = raw(2).model_copy(update={"employer": "Ромашка LLC", "raw_text": "<p>Анализ логов, подготовка отчётов. </p>" * 12})
        sample, copies = deduplicate_market_vacancies([first, second])
        self.assertEqual(len(sample), 1)
        self.assertEqual(len(copies), 1)
        self.assertEqual(copies[0].reason, "cross_posted_description")
        self.assertNotEqual(copies[0].kept_url, copies[0].removed_url)

    def test_same_title_employer_does_not_remove_different_jobs(self):
        first = raw(1).model_copy(update={"employer": "Ромашка", "raw_text": "Анализ логов. " * 20})
        second = raw(2).model_copy(update={"employer": "Ромашка", "raw_text": "Разработка отчётов. " * 20})
        self.assertEqual(len(deduplicate_market_vacancies([first, second])[0]), 2)
        second.raw_text = first.raw_text
        second.source = first.source
        self.assertEqual(len(deduplicate_market_vacancies([first, second])[0]), 2)
        second.source = "other"
        first.region, second.region = "Москва", "Екатеринбург"
        self.assertEqual(len(deduplicate_market_vacancies([first, second])[0]), 2)

    def test_unknown_company_and_different_constraints_do_not_prove_copy(self):
        first = raw(1).model_copy(update={"employer": None, "raw_text": "Знание Python >=3. " * 15})
        second = raw(2).model_copy(update={"employer": None, "raw_text": first.raw_text})
        self.assertEqual(len(deduplicate_market_vacancies([first, second])[0]), 2)
        first.employer = second.employer = "Ромашка"
        second.raw_text = "Знание Python <=3. " * 15
        self.assertEqual(len(deduplicate_market_vacancies([first, second])[0]), 2)

    def test_tracking_urls_and_same_identity_use_complete_version(self):
        first = raw(1).model_copy(update={"url": "https://jobs.test/job?id=12&utm_source=feed", "description_status": "sparse"})
        second = raw(2).model_copy(update={"url": "https://jobs.test/job?utm_medium=email&id=12", "description_status": "full"})
        kept, copies = deduplicate_market_vacancies([first, second])
        self.assertEqual(kept, [second])
        self.assertEqual(copies[0].removed_source_id, first.source_id)
        self.assertEqual(copies[0].kept_source_id, second.source_id)
        third = raw(3).model_copy(update={"url": "https://jobs.test/job?id=13"})
        self.assertEqual(len(deduplicate_market_vacancies([second, third])[0]), 2)

    def test_replacement_indexes_the_full_description_for_later_copies(self):
        first = raw(1).model_copy(update={"employer": "Ромашка", "description_status": "sparse"})
        full = first.model_copy(update={"raw_text": "Анализ логов, подготовка отчётов. " * 12,
                                       "description_status": "full"})
        copy = raw(2).model_copy(update={"employer": "Ромашка", "raw_text": full.raw_text,
                                        "description_status": "full"})
        kept, duplicates = deduplicate_market_vacancies([first, full, copy])
        self.assertEqual(kept, [full])
        self.assertEqual(len(duplicates), 2)
        self.assertTrue(all(row.kept_url == full.url for row in duplicates))

    def test_replacement_does_not_leave_stale_description_matches(self):
        first = raw(1).model_copy(update={"employer": "Ромашка", "raw_text": "Анализ логов. " * 20})
        replacement = first.model_copy(update={"raw_text": "Разработка отчётов. " * 20,
                                               "description_status": "full"})
        other = raw(2).model_copy(update={"employer": "Ромашка", "raw_text": first.raw_text})
        kept, duplicates = deduplicate_market_vacancies([first, replacement, other])
        self.assertEqual(kept, [replacement, other])
        self.assertEqual(len(duplicates), 1)

    def test_local_id_collision_does_not_repoint_duplicate_to_another_job(self):
        unrelated = raw(1).model_copy(update={"source_id": "local-id"})
        first = raw(2).model_copy(update={"source_id": "local-id", "employer": "Ромашка",
                                         "raw_text": "Анализ логов. " * 20})
        copy = raw(3).model_copy(update={"employer": "Ромашка", "raw_text": first.raw_text,
                                        "description_status": "full"})
        kept, duplicates = deduplicate_market_vacancies([unrelated, first, copy])
        self.assertEqual(kept, [unrelated, copy])
        self.assertEqual(duplicates[0].kept_url, copy.url)
        self.assertEqual(duplicates[0].removed_url, first.url)

    def test_richer_cross_post_does_not_merge_distinct_ids_on_original_provider(self):
        first = raw(1).model_copy(update={"employer": "Ромашка", "raw_text": "Анализ логов. " * 20})
        copy = raw(2).model_copy(update={"employer": "Ромашка", "raw_text": first.raw_text,
                                        "description_status": "full"})
        other = raw(3).model_copy(update={"employer": "Ромашка", "raw_text": first.raw_text})
        kept, duplicates = deduplicate_market_vacancies([first, copy, other])
        self.assertEqual(kept, [copy, other])
        self.assertEqual(len(duplicates), 1)

    def test_unknown_region_copy_does_not_bridge_different_known_regions(self):
        first = raw(1).model_copy(update={"employer": "Ромашка", "raw_text": "Анализ логов. " * 20,
                                         "region": "Москва"})
        copy = raw(2).model_copy(update={"employer": "Ромашка", "raw_text": first.raw_text,
                                        "region": None, "description_status": "full"})
        other = raw(3).model_copy(update={"source": "provider-c", "employer": "Ромашка",
                                         "raw_text": first.raw_text, "region": "Екатеринбург"})
        kept, duplicates = deduplicate_market_vacancies([first, copy, other])
        self.assertEqual(kept, [copy, other])
        self.assertEqual(len(duplicates), 1)


if __name__ == "__main__":
    unittest.main()
