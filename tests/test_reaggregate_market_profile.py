import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.reaggregate_market_profile import reaggregate_saved_profile
from test_candidate_assessment_pipeline import profile
from test_market_profile_builder_v2 import normalized, raw


class ReaggregateSavedProfileTests(unittest.TestCase):
    def test_saved_cohort_sources_and_degree_distinctions_are_preserved_without_llm(self):
        market = profile()
        education = ['Художественное образование', 'Высшее образование по направлению дизайна',
                     'Среднее специальное образование по направлению дизайна']
        rows = [normalized(raw(i)).model_copy(update={
            'source_id': source.source_id, 'professional_skills': [], 'responsibilities': [],
            'domain_knowledge': [], 'soft_skills': [], 'education_requirements': [education[i]],
            'tools_and_technologies': ['Photoshop' if i == 1 else 'Adobe Photoshop'],
        }) for i, source in enumerate(market.sources)]
        rows.append(rows[0].model_copy(update={'source_id': 'unselected', 'tools_and_technologies': ['Unselected tool']}))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'market_profile.json'
            path.write_text(market.model_dump_json(), encoding='utf-8')
            cache = path.with_name('market_profile.market_profile_v2.normalization_cache.json')
            cache.write_text(json.dumps({str(i): row.model_dump(mode='json') for i, row in enumerate(rows)}), encoding='utf-8')
            with patch('resume_evaluator.requirement_equivalence.chat', side_effect=AssertionError('No model call allowed')):
                result = reaggregate_saved_profile(path)
        self.assertEqual(result.sources, market.sources)
        self.assertEqual(result.sample_size, 3)
        self.assertEqual([(item.name, item.frequency) for item in result.tools_and_technologies], [('Adobe Photoshop', 3)])
        self.assertEqual(result.education_requirements, [])
        self.assertEqual({item.name for item in result.market_variations}, set(education))
        self.assertTrue(all(item.frequency == 1 for item in result.market_variations))
        self.assertNotIn('Unselected tool', result.model_dump_json())

    def test_missing_normalization_does_not_overwrite_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'market_profile.json'
            before = profile().model_dump_json()
            path.write_text(before, encoding='utf-8')
            path.with_name('market_profile.market_profile_v2.normalization_cache.json').write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                reaggregate_saved_profile(path)
            self.assertEqual(path.read_text(encoding='utf-8'), before)
