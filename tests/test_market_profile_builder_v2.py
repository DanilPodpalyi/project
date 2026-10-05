"""Детерминированные тесты упрощённого MarketProfileBuilder v2."""

import inspect
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from resume_evaluator.market_profile_builder_v2 import (
    RelevanceAssessment,
    _coerce_target,
    _normalized_seniority_rejection,
    _relevance_sanity_check,
    build_market_profile_v2,
    generate_search_queries_v2,
    rank_vacancies_v2,
)
from resume_evaluator.profession import TargetProfile
from resume_evaluator.synthetic_vacancy_pipeline_simplified import (
    InsufficientMarketProfile,
    MarketProfile,
    NormalizedVacancy,
    RawVacancy,
    _aggregate_market_field,
)


TARGET = TargetProfile(
    role="Designer",
    specialization="Graphic Designer",
    domain="Visual Communication",
    seniority="Middle",
    role_family="Design",
    focus="Brand and Visual Design",
    core_skills=[],
    aliases=["Visual Designer"],
    adjacent_specializations=["Marketing Designer"],
    excluded_specializations=["Fashion Designer"],
)


def raw(index: int, title: str = "Graphic Designer") -> RawVacancy:
    return RawVacancy(
        source="provider-a" if index % 2 else "provider-b",
        source_id=f"vacancy-{index}",
        url=f"https://example.test/{index}",
        title=title,
        employer=f"Company code{index}",
        published_at=None,
        raw_text=(
            "Create brand layouts and visual materials. Requirements: "
            "graphic design, Figma and communication."
        ),
        search_query="Graphic Designer",
    )


def normalized(item: RawVacancy) -> NormalizedVacancy:
    return NormalizedVacancy(
        source_id=item.source_id,
        profession=TARGET,
        is_relevant=True,
        required_skills=["Visual design", "Figma"],
        preferred_skills=[],
        professional_skills=["Visual design"],
        tools_and_technologies=["Figma"],
        soft_skills=["Communication"],
        education_requirements=[],
        domain_knowledge=["Brand design"],
        extraction_status="ok",
        professional_requirements_detected=True,
        responsibilities=["Create visual materials"],
        experience_years_min=2,
        education_level=None,
        education_field=None,
        employment_type=None,
        work_format=None,
        conditions=[],
        source=item.source,
        url=item.url,
        title=item.title,
        employer=item.employer,
        published_at=item.published_at,
    )


def searcher(items: list[RawVacancy]):
    def search(plan, minimum, diagnostics, strict_minimum, result_limit):
        del plan, minimum, strict_minimum
        diagnostics.raw_results = len(items)
        diagnostics.deduplicated_results = len(items)
        return items[:result_limit], []

    return search


def validator(labels: dict[str, str]):
    def validate(candidates, target, cache_path):
        del target, cache_path
        return (
            [
                RelevanceAssessment(
                    source_id=item.vacancy.source_id,
                    relevance=labels.get(item.vacancy.source_id, "direct"),
                    confidence="high",
                    reason="Deterministic test decision.",
                )
                for item in candidates
            ],
            True,
            1,
        )

    return validate


def normalizer(items, target, checkpoint_path, cache_path):
    del target, checkpoint_path, cache_path
    return [normalized(item) for item in items]


class MarketProfileBuilderV2Tests(unittest.TestCase):
    def test_default_search_preserves_distinct_jobs_with_same_title_and_employer(self) -> None:
        items = [raw(index) for index in range(1, 4)]
        for item in items:
            item.raw_text += " Design brand assets and visual materials for product campaigns."
        items[2].employer = items[0].employer
        items[2].raw_text += " Create motion graphics and animations."
        copy = items[0].model_copy(update={"source": "syndicator", "source_id": "copy-1",
                                           "url": "https://other.test/copy-1"})
        provider = SimpleNamespace(calls=lambda plan: [("fixture", lambda: [*items, copy])])
        with tempfile.TemporaryDirectory() as directory, patch(
            "resume_evaluator.synthetic_vacancy_pipeline_simplified._search_providers",
            return_value=[provider],
        ):
            result = build_market_profile_v2(
                TARGET, Path(directory) / "profile.json", use_embeddings=False, use_llm=False,
                normalizer=normalizer, relevance_validator=validator({}),
            )
        self.assertEqual(result.sample_size, 3)
        self.assertEqual({row.source_id for row in result.sources}, {row.source_id for row in items})
        self.assertEqual(result.reliability.duplicates_removed, 1)

    def test_cross_posted_ad_is_removed_before_sampling_and_recorded(self) -> None:
        items = [raw(index) for index in range(1, 4)]
        for item in items:
            item.raw_text += " Design brand assets and visual materials for product campaigns."
        copy = items[0].model_copy(update={"source": "syndicator", "source_id": "copy-1", "url": "https://other.test/copy-1"})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "profile.json"
            result = build_market_profile_v2(
                TARGET, output, use_embeddings=False, use_llm=False,
                searcher=searcher([*items, copy]), normalizer=normalizer, relevance_validator=validator({}),
            )
            diagnostics = json.loads(output.with_name("profile.diagnostics.json").read_text(encoding="utf-8"))
        self.assertEqual(result.sample_size, 3)
        self.assertEqual(result.reliability.duplicates_removed, 1)
        self.assertEqual(len(diagnostics["duplicate_vacancies"]), 1)
        self.assertEqual(result.confidence, "low")
        self.assertEqual(result.reliability.confidence, "low")
        self.assertEqual(diagnostics["confidence"], "low")

    def test_high_confidence_uses_complete_raw_provenance_not_model_metadata(self) -> None:
        items = [raw(index).model_copy(update={"description_status": "full", "published_at": "2026-09-20T10:00:00+03:00"})
                 for index in range(1, 7)]
        with tempfile.TemporaryDirectory() as directory:
            result = build_market_profile_v2(
                TARGET, Path(directory) / "profile.json", use_embeddings=True, use_llm=False,
                searcher=searcher(items), normalizer=normalizer, relevance_validator=validator({}),
                embedder=lambda texts, model: {text: [1.0, 0.0] for text in texts},
                market_as_of=datetime(2026, 10, 1, tzinfo=timezone.utc),
            )
        self.assertEqual(result.confidence, "high")
        self.assertEqual(result.reliability.independent_employers, 6)
        self.assertEqual(result.reliability.full_description_count, 6)
        self.assertTrue(all(row.description_status == "full" for row in result.sources))

    def test_adjacent_cohort_cannot_change_primary_requirements(self) -> None:
        items = [raw(index) for index in range(1, 7)]

        def separate_normalizer(items, target, checkpoint_path, cache_path):
            values = normalizer(items, target, checkpoint_path, cache_path)
            return [item.model_copy(update={"tools_and_technologies": ["Blender"]})
                    if item.source_id in {"vacancy-4", "vacancy-5", "vacancy-6"} else item
                    for item in values]

        with tempfile.TemporaryDirectory() as directory:
            result = build_market_profile_v2(
                TARGET, Path(directory) / "profile.json", use_embeddings=False, use_llm=False,
                searcher=searcher(items), normalizer=separate_normalizer,
                relevance_validator=validator({f"vacancy-{index}": "adjacent" for index in range(4, 7)}),
            )
        self.assertIsInstance(result, MarketProfile)
        self.assertEqual(result.sample_size, 3)
        self.assertEqual(result.tools_and_technologies[0].name, "Figma")
        self.assertEqual(result.tools_and_technologies[0].coverage, 1)
        self.assertNotIn("Blender", [item.name for item in result.tools_and_technologies])
        self.assertEqual(result.adjacent_context.sample_size, 3)
        self.assertEqual(result.adjacent_context.tools_and_technologies[0].name, "Blender")
        self.assertTrue({item.source_id for item in result.sources}.isdisjoint(
            {item.source_id for item in result.adjacent_context.sources}))

    def test_two_direct_and_adjacent_cannot_make_primary_profile_sufficient(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = build_market_profile_v2(
                TARGET, Path(directory) / "profile.json", use_embeddings=False, use_llm=False,
                searcher=searcher([raw(index) for index in range(1, 6)]), normalizer=normalizer,
                relevance_validator=validator({f"vacancy-{index}": "adjacent" for index in range(3, 6)}),
            )
        self.assertIsInstance(result, InsufficientMarketProfile)
        self.assertEqual(result.sample_size, 2)
        self.assertEqual(result.adjacent_context.sample_size, 3)

    def test_backfill_replaces_failed_sparse_empty_and_higher_seniority(self) -> None:
        attempted = []

        def rejecting_normalizer(items, target, checkpoint_path, cache_path):
            attempted.extend(item.source_id for item in items)
            values = normalizer(items, target, checkpoint_path, cache_path)
            for item in values:
                status = {"vacancy-1": "failed", "vacancy-2": "sparse", "vacancy-3": "empty"}.get(item.source_id)
                if status:
                    item.extraction_status = status
                if item.source_id == "vacancy-4":
                    item.profession = item.profession.model_copy(update={"seniority": "Senior"})
            return values

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "profile.json"
            result = build_market_profile_v2(
                TARGET, output, use_embeddings=False, use_llm=False,
                searcher=searcher([raw(index) for index in range(1, 11)]),
                normalizer=rejecting_normalizer, relevance_validator=validator({}),
            )
            diagnostics = json.loads(output.with_name("profile.diagnostics.json").read_text(encoding="utf-8"))
        self.assertEqual(result.sample_size, 6)
        self.assertEqual({item.source_id for item in result.sources}, {f"vacancy-{index}" for index in range(5, 11)})
        self.assertEqual(len(attempted), len(set(attempted)))
        self.assertEqual(diagnostics["normalization_candidates"], 10)
        self.assertEqual(diagnostics["backfill_count"], 4)
        self.assertEqual(len(diagnostics["post_normalization_rejections"]), 4)

    def test_backfill_checks_next_relevance_batch_after_initial_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "profile.json"
            result = build_market_profile_v2(
                TARGET, output, candidate_limit=3, use_embeddings=False, use_llm=False,
                searcher=searcher([raw(index) for index in range(1, 9)]), normalizer=normalizer,
                relevance_validator=validator({"vacancy-1": "irrelevant", "vacancy-2": "irrelevant"}),
            )
            diagnostics = json.loads(output.with_name("profile.diagnostics.json").read_text(encoding="utf-8"))
        self.assertEqual(result.sample_size, 6)
        self.assertGreater(diagnostics["candidate_count"], 3)
        self.assertEqual(diagnostics["candidate_count"], 8)

    def test_missing_normalized_row_is_recorded_and_backfilled(self) -> None:
        def missing_normalizer(items, target, checkpoint_path, cache_path):
            return [item for item in normalizer(items, target, checkpoint_path, cache_path)
                    if item.source_id != "vacancy-1"]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "profile.json"
            result = build_market_profile_v2(
                TARGET, output, use_embeddings=False, use_llm=False,
                searcher=searcher([raw(index) for index in range(1, 8)]), normalizer=missing_normalizer,
                relevance_validator=validator({}),
            )
            diagnostics = json.loads(output.with_name("profile.diagnostics.json").read_text(encoding="utf-8"))
        self.assertEqual(result.sample_size, 6)
        self.assertIn("не вернула", diagnostics["post_normalization_rejections"][0]["reason"])

    def test_synonyms_make_repeated_requirements_in_main_profile(self) -> None:
        forms = {"vacancy-1": "анализ журналов", "vacancy-2": "log analysis", "vacancy-3": "анализ логов"}

        def synonym_normalizer(items, target, checkpoint_path, cache_path):
            return [item.model_copy(update={
                "professional_skills": [forms[item.source_id]], "tools_and_technologies": [],
                "responsibilities": [], "soft_skills": [], "domain_knowledge": [],
            }) for item in normalizer(items, target, checkpoint_path, cache_path)]

        with tempfile.TemporaryDirectory() as directory:
            result = build_market_profile_v2(
                TARGET, Path(directory) / "profile.json", use_embeddings=False, use_llm=False,
                searcher=searcher([raw(index) for index in range(1, 4)]), normalizer=synonym_normalizer,
                relevance_validator=validator({}),
            )
        self.assertIsInstance(result, MarketProfile)
        self.assertEqual(len(result.professional_skills), 1)
        self.assertEqual(result.professional_skills[0].frequency, 3)
        self.assertEqual(set(result.professional_skills[0].source_values), set(forms.values()))

    def test_target_string_removes_extracted_seniority_from_specialization(self) -> None:
        target = _coerce_target(
            "Information Security Analyst / Middle",
            use_llm=False,
        )
        self.assertEqual(target.specialization, "Information Security Analyst")
        self.assertEqual(target.seniority, "Middle")

    def test_queries_are_small_and_use_aliases(self) -> None:
        queries = generate_search_queries_v2(TARGET)
        self.assertGreaterEqual(len(queries), 3)
        self.assertLessEqual(len(queries), 6)
        self.assertIn("Graphic Designer", queries)
        self.assertIn("Visual Designer", queries)

    def test_semantic_only_candidate_is_ranked_not_rejected(self) -> None:
        item = raw(1, "Creative Brand Specialist")

        def embedder(texts, model):
            del model
            return {text: [1.0, 0.0] for text in texts}

        ranked, available = rank_vacancies_v2(
            [item], TARGET, embedder=embedder
        )
        self.assertTrue(available)
        self.assertEqual([entry.vacancy.source_id for entry in ranked], [item.source_id])
        self.assertGreater(ranked[0].embedding_score or 0, 0.9)

    def test_embedding_failure_falls_back_to_lexical_ranking(self) -> None:
        def unavailable(texts, model):
            del texts, model
            raise OSError("embedding service unavailable")

        ranked, available = rank_vacancies_v2(
            [raw(1)], TARGET, embedder=unavailable
        )
        self.assertFalse(available)
        self.assertEqual(len(ranked), 1)
        self.assertGreater(ranked[0].lexical_score, 0.8)

    def test_explicit_senior_is_not_direct_for_middle_target(self) -> None:
        ranked, _ = rank_vacancies_v2(
            [raw(1, "Senior Graphic Designer")],
            TARGET,
            use_embeddings=False,
        )
        result = _relevance_sanity_check(
            RelevanceAssessment(
                source_id="vacancy-1",
                relevance="direct",
                confidence="high",
                reason="LLM considered the profession relevant.",
            ),
            ranked[0],
            TARGET,
        )
        self.assertEqual(result.relevance, "irrelevant")
        self.assertIn("seniority", result.reason)

    def test_explicit_adjacent_title_is_not_promoted_to_direct(self) -> None:
        ranked, _ = rank_vacancies_v2(
            [raw(1, "Marketing Designer")],
            TARGET,
            use_embeddings=False,
        )
        result = _relevance_sanity_check(
            RelevanceAssessment(
                source_id="vacancy-1",
                relevance="direct",
                confidence="high",
                reason="The duties are in visual design.",
            ),
            ranked[0],
            TARGET,
        )
        self.assertEqual(result.relevance, "adjacent")

    def test_seniority_extracted_from_description_is_rejected(self) -> None:
        item = normalized(raw(1, "Marketing Designer"))
        item = item.model_copy(
            update={
                "profession": item.profession.model_copy(
                    update={"seniority": "Senior"}
                )
            }
        )
        reason = _normalized_seniority_rejection(TARGET, item)
        self.assertIsNotNone(reason)
        self.assertIn("выше целевого", reason or "")

    def test_two_good_vacancies_preserve_a_provisional_signal(self) -> None:
        items = [raw(1), raw(2)]
        with tempfile.TemporaryDirectory() as directory:
            result = build_market_profile_v2(
                TARGET,
                Path(directory) / "profile.json",
                use_embeddings=False,
                use_llm=False,
                searcher=searcher(items),
                relevance_validator=validator({}),
                normalizer=normalizer,
            )
        self.assertIsInstance(result, InsufficientMarketProfile)
        self.assertEqual(result.sample_size, 2)
        self.assertEqual(len(result.sources), 2)
        self.assertIn("предварительный сигнал", result.summary)

    def test_three_vacancies_return_existing_market_profile_schema(self) -> None:
        items = [raw(1), raw(2), raw(3)]
        with tempfile.TemporaryDirectory() as directory:
            result = build_market_profile_v2(
                TARGET,
                Path(directory) / "profile.json",
                use_embeddings=False,
                use_llm=False,
                searcher=searcher(items),
                relevance_validator=validator({}),
                normalizer=normalizer,
            )
        self.assertIsInstance(result, MarketProfile)
        self.assertEqual(result.sample_size, 3)
        figma = next(
            item for item in result.tools_and_technologies if item.name == "Figma"
        )
        self.assertEqual(figma.frequency, 3)
        self.assertEqual(figma.importance, "common")

    def test_only_singleton_requirements_return_insufficient_market_data(self) -> None:
        items = [raw(1), raw(2), raw(3)]

        def singleton_normalizer(items, target, checkpoint_path, cache_path):
            del target, checkpoint_path, cache_path
            unique_values = (
                ("Illustration", "Figma", "Communication", "Design degree", "Branding", "Create layouts"),
                ("Cryptography", "Splunk", "Leadership", "Security certificate", "Cybersecurity", "Audit access"),
                ("Reporting", "WordPress", "Empathy", "Journalism degree", "News media", "Interview sources"),
            )
            results = []
            for item, values in zip(items, unique_values, strict=True):
                skill, tool, soft_skill, education, domain, responsibility = values
                results.append(
                    normalized(item).model_copy(
                        update={
                            "professional_skills": [skill],
                            "tools_and_technologies": [tool],
                            "soft_skills": [soft_skill],
                            "education_requirements": [education],
                            "domain_knowledge": [domain],
                            "responsibilities": [responsibility],
                        }
                    )
                )
            return results

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "profile.json"
            result = build_market_profile_v2(
                TARGET,
                output,
                use_embeddings=False,
                use_llm=False,
                searcher=searcher(items),
                relevance_validator=validator({}),
                normalizer=singleton_normalizer,
            )
            diagnostics = json.loads(
                output.with_name("profile.diagnostics.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertIsInstance(result, InsufficientMarketProfile)
        self.assertEqual(result.sample_size, 3)
        self.assertEqual(
            diagnostics["outcome_reason"],
            "no_repeated_market_requirements",
        )

    def test_frequency_thresholds_scale_for_two_three_and_four(self) -> None:
        samples = [normalized(raw(index)) for index in range(1, 5)]
        for size, expected in ((2, "common"), (3, "common"), (4, "additional")):
            subset = [
                item.model_copy(
                    update={
                        "tools_and_technologies": (
                            ["Figma"] if position < 2 else []
                        )
                    }
                )
                for position, item in enumerate(samples[:size])
            ]
            requirements, _ = _aggregate_market_field(
                subset, "tools_and_technologies"
            )
            figma = next(item for item in requirements if item.name == "Figma")
            self.assertEqual(figma.importance, expected)

    def test_irrelevant_vacancy_is_never_in_final_sample(self) -> None:
        items = [raw(1), raw(2), raw(3), raw(4, "Fashion Designer")]
        with tempfile.TemporaryDirectory() as directory:
            result = build_market_profile_v2(
                TARGET,
                Path(directory) / "profile.json",
                use_embeddings=False,
                use_llm=False,
                searcher=searcher(items),
                relevance_validator=validator({"vacancy-4": "irrelevant"}),
                normalizer=normalizer,
            )
        self.assertIsInstance(result, MarketProfile)
        self.assertNotIn("vacancy-4", {item.source_id for item in result.sources})

    def test_public_api_has_no_candidate_input(self) -> None:
        parameters = inspect.signature(build_market_profile_v2).parameters
        self.assertNotIn("resume", parameters)
        self.assertNotIn("candidate", parameters)
        self.assertNotIn("skills", parameters)

    def test_candidate_core_skills_cannot_change_market_profile(self) -> None:
        items = [raw(1), raw(2), raw(3)]
        first_target = TARGET.model_copy(update={"core_skills": ["Figma"]})
        second_target = TARGET.model_copy(update={"core_skills": ["Blender"]})
        with tempfile.TemporaryDirectory() as directory:
            first = build_market_profile_v2(
                first_target,
                Path(directory) / "first.json",
                use_embeddings=False,
                use_llm=False,
                searcher=searcher(items),
                relevance_validator=validator({}),
                normalizer=normalizer,
                market_as_of=datetime(2026, 10, 1, tzinfo=timezone.utc),
            )
            second = build_market_profile_v2(
                second_target,
                Path(directory) / "second.json",
                use_embeddings=False,
                use_llm=False,
                searcher=searcher(items),
                relevance_validator=validator({}),
                normalizer=normalizer,
                market_as_of=datetime(2026, 10, 1, tzinfo=timezone.utc),
            )
        self.assertEqual(first.model_dump(), second.model_dump())


if __name__ == "__main__":
    unittest.main()
