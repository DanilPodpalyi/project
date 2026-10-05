"""Live benchmark восьми целевых профессий для MarketProfileBuilder v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from resume_evaluator.market_profile_builder_v2 import build_market_profile_v2
from resume_evaluator.profession import TargetProfile


OUTPUT_ROOT = Path("output/market_profile_v2_benchmark")


def target(
    role: str,
    specialization: str,
    domain: str,
    seniority: str | None,
    role_family: str,
    focus: str,
    aliases: list[str],
    adjacent: list[str] | None = None,
    excluded: list[str] | None = None,
) -> TargetProfile:
    return TargetProfile(
        role=role,
        specialization=specialization,
        domain=domain,
        seniority=seniority,
        role_family=role_family,
        focus=focus,
        core_skills=[],
        aliases=aliases,
        adjacent_specializations=adjacent or [],
        excluded_specializations=excluded or [],
    )


CASES = {
    "ml_engineer_middle": target(
        "Engineer", "ML Engineer", "Machine Learning / Artificial Intelligence",
        "Middle", "ML Engineering", "Model Development / Production ML",
        ["Machine Learning Engineer", "Инженер машинного обучения"],
        ["AI Engineer", "Data Scientist"],
        ["MLOps Engineer", "AI Infrastructure Engineer"],
    ),
    "information_security_analyst_middle": target(
        "Analyst", "Information Security Analyst", "Information Security",
        "Middle", "Cybersecurity", "Security Monitoring / Risk Analysis",
        ["Cybersecurity Analyst", "Аналитик информационной безопасности"],
        ["SOC Analyst", "Security Analyst"],
        ["Security Engineer", "DevSecOps Engineer"],
    ),
    "graphic_designer_middle": target(
        "Designer", "Graphic Designer", "Visual Communication / Design",
        "Middle", "Graphic Design", "Brand / Marketing Visual Design",
        ["Visual Designer", "Графический дизайнер"],
        ["Marketing Designer", "Brand Designer"],
        ["Fashion Designer", "Interior Designer"],
    ),
    "marketer_middle": target(
        "Specialist", "Маркетолог", "Marketing",
        "Middle", "Marketing", "Marketing Strategy / Campaign Execution",
        ["Marketing Specialist", "Marketing Manager"],
        ["Digital Marketing Specialist", "Brand Manager"],
        ["Sales Manager"],
    ),
    "backend_developer_middle": target(
        "Developer", "Backend Developer", "Software Engineering",
        "Middle", "Backend Engineering", "Server-side Application Development",
        ["Backend Engineer", "Бэкенд-разработчик"],
        ["Software Engineer"],
        ["Frontend Developer", "Mobile Developer"],
    ),
    "data_analyst_middle": target(
        "Analyst", "Data Analyst", "Data Analytics",
        "Middle", "Data Analytics", "Data Analysis / Business Reporting",
        ["Аналитик данных", "BI Data Analyst"],
        ["Business Intelligence Analyst", "Product Analyst"],
        ["Data Engineer", "Data Scientist"],
    ),
    "hr_specialist_middle": target(
        "Specialist", "HR Specialist", "Human Resources",
        "Middle", "Human Resources", "HR Operations / Employee Lifecycle",
        ["Human Resources Specialist", "Специалист по персоналу"],
        ["HR Generalist", "People Operations Specialist"],
        ["Recruiter", "Payroll Specialist"],
    ),
    "accountant": target(
        "Accountant", "Бухгалтер", "Accounting / Finance",
        None, "Accounting", "Financial Accounting / Bookkeeping",
        ["Accountant", "Bookkeeper"],
        ["Financial Accountant"],
        ["Financial Analyst", "Auditor"],
    ),
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live benchmark MarketProfile v2")
    parser.add_argument(
        "--cases",
        nargs="*",
        choices=sorted(CASES),
        help="Перезапустить только перечисленные кейсы",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_ROOT / "benchmark_summary.json"
    existing: dict[str, dict[str, object]] = {}
    if summary_path.is_file():
        for row in json.loads(summary_path.read_text(encoding="utf-8")):
            case = row.get("case")
            if case is None:
                case = next(
                    (
                        slug for slug, profile in CASES.items()
                        if profile.label() == row.get("profession")
                    ),
                    None,
                )
            if case is not None:
                row["case"] = case
                existing[str(case)] = row
    selected = {
        slug: profile for slug, profile in CASES.items()
        if not args.cases or slug in args.cases
    }
    for index, (slug, profile) in enumerate(selected.items(), start=1):
        print(f"\n=== [{index}/{len(selected)}] {profile.label()} ===", flush=True)
        directory = OUTPUT_ROOT / slug
        output = directory / "market_profile_v2.json"
        result = build_market_profile_v2(profile, output)
        diagnostics = json.loads(
            (directory / "market_profile_v2.diagnostics.json").read_text(
                encoding="utf-8"
            )
        )
        existing[slug] = {
                "case": slug,
                "profession": profile.label(),
                "status": result.status,
                "sample": result.sample_size,
                "confidence": diagnostics["confidence"],
                "raw": diagnostics["raw_count"],
                "dedup": diagnostics["deduplicated_count"],
                "candidates": diagnostics["candidate_count"],
                "direct": diagnostics["direct_count"],
                "adjacent": diagnostics["adjacent_count"],
                "irrelevant": diagnostics["irrelevant_count"],
                "providers": diagnostics["providers_used"],
                "vacancies": diagnostics["final_vacancies"],
            }
        rows = [existing[slug] for slug in CASES if slug in existing]
        summary_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"\nSaved: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
