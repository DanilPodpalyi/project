"""Тесты расчёта опыта, образования и похожих навыков."""

"""Тесты детерминированного скоринга кандидата."""

import unittest
from datetime import date

from resume_evaluator.candidate_scorer import (
    EDUCATION_MAX,
    EXPERIENCE_MAX,
    HEADER_MAX,
    PROJECTS_MAX,
    SKILLS_MAX,
    SUMMARY_MAX,
    calculate_candidate_experience,
    calculate_education_score,
    calculate_projects_score,
    collect_candidate_skills,
    compare_skills,
    education_field_matches,
    extract_period_years,
    score_candidate,
    skill_matches,
)


class ExperienceCalculationTests(unittest.TestCase):
    def test_parses_russian_month_range(self) -> None:
        self.assertEqual(
            extract_period_years("Февраль 2023 — Ноябрь 2024"),
            1.83,
        )

    def test_parses_current_employment(self) -> None:
        self.assertEqual(
            extract_period_years(
                "Сентябрь 2025 — настоящее время",
                today=date(2026, 8, 14),
            ),
            1.0,
        )

    def test_keeps_year_only_range_compatibility(self) -> None:
        self.assertEqual(extract_period_years("2020 — 2022"), 2.0)

    def test_does_not_double_count_overlapping_jobs(self) -> None:
        resume = {
            "work_experience": [
                {"period": "Февраль 2022 — Май 2024"},
                {"period": "Февраль 2023 — Ноябрь 2024"},
                {"period": "Сентябрь 2025 — настоящее время"},
            ]
        }

        self.assertEqual(
            calculate_candidate_experience(
                resume,
                today=date(2026, 8, 14),
            ),
            3.83,
        )


class EducationCalculationTests(unittest.TestCase):
    def test_matches_one_of_composite_education_directions(self) -> None:
        self.assertTrue(
            education_field_matches(
                (
                    "рекламы, связей с общественностью, маркетинга "
                    "или коммуникаций"
                ),
                [
                    "Цифровая экономика и массовые коммуникации",
                    "Реклама и связи с общественностью в отрасли",
                ],
            )
        )

    def test_scores_matching_level_and_direction(self) -> None:
        resume = {
            "education": [
                {
                    "education_level": "Высшее",
                    "education_field": (
                        "Цифровая экономика и массовые коммуникации"
                    ),
                    "field_of_study": (
                        "Реклама и связи с общественностью в отрасли"
                    ),
                }
            ]
        }
        vacancy = {
            "education_level": "Высшее",
            "education_field": (
                "рекламы, связей с общественностью, маркетинга "
                "или коммуникаций"
            ),
        }

        score, details = calculate_education_score(resume, vacancy)

        self.assertEqual(score, 10.0)
        self.assertEqual(
            details,
            "Уровень образования: совпадает; направление: совпадает.",
        )


class SkillComparisonTests(unittest.TestCase):
    def test_matches_parts_of_compound_candidate_skill(self) -> None:
        candidate_skills = {"pr и контент": "PR и контент"}

        matched, missing = compare_skills(
            candidate_skills,
            ["PR", "контент"],
        )

        self.assertEqual(matched, ["PR", "контент"])
        self.assertEqual(missing, [])

    def test_matches_related_event_wording(self) -> None:
        self.assertTrue(
            skill_matches(
                "Организация рассадки под нетворкинг",
                "организация мероприятий",
            )
        )

    def test_does_not_infer_press_releases_from_pr(self) -> None:
        self.assertFalse(
            skill_matches("PR и контент", "подготовка пресс-релизов")
        )

    def test_uses_languages_and_responsibilities_as_evidence(self) -> None:
        resume = {
            "skills": [],
            "soft_skills": [],
            "work_experience": [
                {
                    "responsibilities": ["Подготовка аналитических отчётов"],
                    "technologies": [],
                    "achievements": [],
                }
            ],
            "projects": [],
            "languages": [{"language": "Английский язык"}],
        }
        candidate_skills = collect_candidate_skills(resume)

        matched, missing = compare_skills(
            candidate_skills,
            ["аналитические отчёты", "английский язык", "SMM"],
        )

        self.assertEqual(
            matched,
            ["аналитические отчёты", "английский язык"],
        )
        self.assertEqual(missing, ["SMM"])


class ScoreWeightsTests(unittest.TestCase):
    def test_requested_weights_sum_to_100(self) -> None:
        self.assertEqual(
            {
                "experience_score": EXPERIENCE_MAX,
                "projects_score": PROJECTS_MAX,
                "header_score": HEADER_MAX,
                "summary_score": SUMMARY_MAX,
                "skills_score": SKILLS_MAX,
                "education_score": EDUCATION_MAX,
            },
            {
                "experience_score": 25,
                "projects_score": 20,
                "header_score": 15,
                "summary_score": 15,
                "skills_score": 15,
                "education_score": 10,
            },
        )

    def test_breakdown_uses_six_requested_categories(self) -> None:
        result = score_candidate({}, {})

        self.assertEqual(
            list(result.breakdown),
            [
                "experience_score",
                "projects_score",
                "header_score",
                "summary_score",
                "skills_score",
                "education_score",
            ],
        )
        self.assertEqual(
            result.total_score,
            round(sum(item.score for item in result.breakdown.values()), 1),
        )

    def test_empty_projects_receive_zero_points(self) -> None:
        score, _ = calculate_projects_score(
            {"projects": []},
            {"required_skills": [], "preferred_skills": []},
        )

        self.assertEqual(score, 0.0)

    def test_complete_relevant_project_receives_full_points(self) -> None:
        resume = {
            "projects": [
                {
                    "name": "Аналитический сервис",
                    "description": "Разработка сервиса аналитики",
                    "role": "Разработчик",
                    "technologies": ["Python"],
                    "results": ["Запущен в эксплуатацию"],
                }
            ]
        }
        vacancy = {
            "required_skills": ["Python"],
            "preferred_skills": [],
        }

        score, _ = calculate_projects_score(resume, vacancy)

        self.assertEqual(score, 20.0)


if __name__ == "__main__":
    unittest.main()
