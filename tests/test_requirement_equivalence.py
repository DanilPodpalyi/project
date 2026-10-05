"""Частоты, проверка предложений LLM и происхождение синонимов."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from resume_evaluator.requirement_equivalence import (
    EquivalenceDiagnostics, EquivalenceProposal, RequirementConcept,
    aggregate_equivalent_requirements, apply_equivalence_proposal, requirement_key,
)
from resume_evaluator.synthetic_vacancy_pipeline_simplified import MarketRequirementEvidence
from test_market_profile_builder_v2 import normalized, raw


def concept(text, category="professional_skills", source_id="one"):
    return RequirementConcept(category=category, name=text, evidence=[
        MarketRequirementEvidence(source_id=source_id, text=text),
    ])


class RequirementEquivalenceTests(unittest.TestCase):
    def test_synonyms_keep_every_source_form_but_count_each_vacancy_once(self):
        forms = (["анализ журналов", "log analysis", "анализ логов"],
                 ["анализ логов"], ["log analysis"])
        samples = [normalized(raw(index)).model_copy(update={
            "professional_skills": values, "tools_and_technologies": [],
            "responsibilities": [], "soft_skills": [], "domain_knowledge": [],
        }) for index, values in enumerate(forms, 1)]
        categories, variations, diagnostics = aggregate_equivalent_requirements(samples)
        requirement = categories["professional_skills"][0]
        self.assertEqual(len(categories["professional_skills"]), 1)
        self.assertEqual(requirement.frequency, 3)
        self.assertEqual(requirement.coverage, 1)
        self.assertEqual(requirement.importance, "common")
        self.assertEqual(len(requirement.evidence), 5)
        self.assertEqual({(row.source_id, row.text) for row in requirement.evidence}, {
            ("vacancy-1", "анализ журналов"), ("vacancy-1", "log analysis"),
            ("vacancy-1", "анализ логов"), ("vacancy-2", "анализ логов"),
            ("vacancy-3", "log analysis"),
        })
        self.assertEqual(len(requirement.sources), 3)
        self.assertEqual(variations, [])
        self.assertFalse(diagnostics.llm_requested)

    def test_morphology_and_technical_symbols(self):
        self.assertEqual(requirement_key("Разработка отчётов"), requirement_key("Разработка отчета"))
        self.assertEqual(requirement_key("Postgres"), requirement_key("PostgreSQL"))
        for a, b in (("C", "C++"), ("C++", "C#"), ("Python 2", "Python 3"),
                     ("Python", "анализ данных"), ("SQL", "PostgreSQL"),
                     ("Python >=3", "Python <=3")):
            self.assertNotEqual(requirement_key(a), requirement_key(b))

    def test_adobe_product_forms_count_unique_vacancies_without_merging_products(self):
        samples = [normalized(raw(index)).model_copy(update={
            "professional_skills": [], "responsibilities": [], "soft_skills": [],
            "education_requirements": [], "domain_knowledge": [],
            "tools_and_technologies": tools,
        }) for index, tools in enumerate([
            ["Adobe Photoshop", "Photoshop", "Adobe Illustrator", "Adobe InDesign"],
            ["Photoshop", "Illustrator", "InDesign"],
            ["Adobe Photoshop", "Illustrator", "InDesign"],
        ], 1)]
        categories, variations, _ = aggregate_equivalent_requirements(samples)
        tools = categories["tools_and_technologies"]
        self.assertEqual(len(tools), 3)
        self.assertEqual(variations, [])
        self.assertTrue(all(tool.frequency == 3 and tool.coverage == 1 for tool in tools))
        photoshop = next(tool for tool in tools if tool.name == "Adobe Photoshop")
        self.assertEqual(set(photoshop.source_values), {"Adobe Photoshop", "Photoshop"})
        self.assertEqual(len(photoshop.evidence), 4)
        for other in ("Illustrator", "InDesign", "Photoshop Express", "Photoshop 2024"):
            self.assertNotEqual(requirement_key("Photoshop"), requirement_key(other))

    def test_categories_never_share_frequency(self):
        samples = [normalized(raw(1)).model_copy(update={"professional_skills": ["Python"], "tools_and_technologies": []}),
                   normalized(raw(2)).model_copy(update={"professional_skills": [], "tools_and_technologies": ["Python"]})]
        categories, variations, _ = aggregate_equivalent_requirements(samples)
        self.assertEqual(categories["professional_skills"], [])
        self.assertEqual(categories["tools_and_technologies"], [])
        python = [row for row in variations if row.name == "Python"]
        self.assertEqual({row.category for row in python}, {"professional_skills", "tools_and_technologies"})
        self.assertTrue(all(row.frequency == 1 for row in python))

    def test_llm_equivalence_retains_originals_and_does_not_invent_name(self):
        concepts = [concept("Проверка программного продукта", source_id="first"),
                    concept("Тестирование приложения", source_id="second")]
        diagnostics = EquivalenceDiagnostics(llm_requested=True)
        result = apply_equivalence_proposal(concepts, EquivalenceProposal.model_validate({
            "groups": [{"members": [1, 0], "canonical_name": "invented"}],
        }), diagnostics)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, concepts[0].name)
        self.assertEqual(result[0].evidence, concepts[0].evidence + concepts[1].evidence)
        self.assertEqual(diagnostics.accepted_groups, 1)

    def test_llm_cannot_merge_related_technologies_actions_or_constraints(self):
        pairs = [
            (concept("Python"), concept("анализ данных")),
            (concept("машинное обучение"), concept("глубокое обучение")),
            (concept("TensorFlow", "tools_and_technologies"), concept("Keras", "tools_and_technologies")),
            (concept("SIEM"), concept("SOC")),
            (concept("Разработка моделей", "responsibilities"), concept("Тестирование моделей", "responsibilities")),
            (concept("анализ логов"), concept("не анализировать логи")),
            (concept("опыт 2 года"), concept("опыт 3 года")),
            (concept("Python >=3"), concept("Python <=3")),
            (concept("высшее образование"), concept("высшее техническое образование")),
            (concept("Python", "professional_skills"), concept("Python", "tools_and_technologies")),
            (concept("Художественное образование", "education_requirements"), concept("Профильное высшее образование", "education_requirements")),
            (concept("Высшее образование по направлению дизайна", "education_requirements"), concept("Среднее специальное образование по направлению дизайна", "education_requirements")),
            (concept("Цветокоррекция"), concept("Работа с цветом")),
            (concept("Взаимодействие с заказчиками", "soft_skills"), concept("Коммуникативные навыки", "soft_skills")),
            (concept("Креативность", "soft_skills"), concept("Насмотренность", "soft_skills")),
        ]
        for left, right in pairs:
            with self.subTest(left=left.name, right=right.name):
                diagnostics = EquivalenceDiagnostics(llm_requested=True)
                result = apply_equivalence_proposal([left, right],
                    EquivalenceProposal.model_validate({"groups": [{"members": [0, 1]}]}), diagnostics)
                self.assertEqual(len(result), 2)
                self.assertEqual(diagnostics.rejected_groups, 1)

    def test_hallucinated_ids_overlap_and_duplicates_are_rejected(self):
        concepts = [concept("один"), concept("два"), concept("три")]
        for groups in ([{"members": [0, 99]}], [{"members": [-1, 0]}],
                       [{"members": [0, 0]}], [{"members": [0, 1]}, {"members": [1, 2]}]):
            diagnostics = EquivalenceDiagnostics(llm_requested=True)
            result = apply_equivalence_proposal(concepts, EquivalenceProposal.model_validate({"groups": groups}), diagnostics)
            self.assertEqual(result, concepts)
            self.assertEqual(diagnostics.rejected_groups, len(groups))

    def test_model_failure_preserves_deterministic_synonyms(self):
        samples = [normalized(raw(1)), normalized(raw(2))]
        with patch("resume_evaluator.requirement_equivalence.chat", side_effect=OSError("offline")):
            categories, _, diagnostics = aggregate_equivalent_requirements(samples, use_llm=True)
        self.assertEqual(categories["tools_and_technologies"][0].frequency, 2)
        self.assertFalse(diagnostics.llm_available)
        self.assertIn("offline", diagnostics.errors[0])

    def test_semantic_proposal_is_used_in_aggregation(self):
        samples = [normalized(raw(index)).model_copy(update={
            "professional_skills": [value], "tools_and_technologies": [], "soft_skills": [],
            "responsibilities": [], "domain_knowledge": [],
        }) for index, value in enumerate(["Проверка программного продукта", "Тестирование приложения"], 1)]
        response = SimpleNamespace(message=SimpleNamespace(content=json.dumps({"groups": [{"members": [0, 1]}]})))
        with patch("resume_evaluator.requirement_equivalence.chat", return_value=response):
            categories, variations, diagnostics = aggregate_equivalent_requirements(samples, use_llm=True)
        self.assertEqual(len(categories["professional_skills"]), 1)
        self.assertEqual(categories["professional_skills"][0].frequency, 2)
        self.assertEqual(variations, [])
        self.assertTrue(diagnostics.llm_available)


if __name__ == "__main__":
    unittest.main()
