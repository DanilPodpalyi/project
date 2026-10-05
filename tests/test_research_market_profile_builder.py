"""Проверки изоляции исследовательской копии MarketProfileBuilder."""

import unittest
from pathlib import Path

from resume_evaluator import candidate_assessment_pipeline
from resume_evaluator import research_market_profile_builder as research
from resume_evaluator import synthetic_vacancy_pipeline_simplified as production


class ResearchMarketProfileBuilderTests(unittest.TestCase):
    def test_research_builder_is_an_independent_module_snapshot(self) -> None:
        self.assertEqual(research.MODULE_VARIANT, "research")
        self.assertEqual(
            research.SNAPSHOT_SOURCE,
            "synthetic_vacancy_pipeline_simplified.py",
        )
        self.assertIsNot(research.build_market_profile, production.build_market_profile)
        self.assertIsNot(research.MarketProfile, production.MarketProfile)

    def test_research_defaults_do_not_collide_with_production(self) -> None:
        self.assertEqual(
            research.DEFAULT_MARKET_PROFILE_OUTPUT,
            Path("research_market_profile.json"),
        )
        self.assertEqual(
            research.NORMALIZATION_CACHE_FILENAME,
            "research_vacancy_normalization_cache.json",
        )
        self.assertEqual(
            production.DEFAULT_MARKET_PROFILE_OUTPUT,
            Path("market_profile.json"),
        )

    def test_candidate_pipeline_still_uses_production_builder(self) -> None:
        self.assertIs(
            candidate_assessment_pipeline.build_market_profile,
            production.build_market_profile,
        )
        self.assertIsNot(
            candidate_assessment_pipeline.build_market_profile,
            research.build_market_profile,
        )


if __name__ == "__main__":
    unittest.main()
