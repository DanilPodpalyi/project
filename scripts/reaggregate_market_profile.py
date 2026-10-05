"""Пересчёт готового рынка из сохранённых нормализованных текстов без retrieval/LLM.

python -m scripts.reaggregate_market_profile path/to/market_profile.json
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from resume_evaluator.requirement_equivalence import aggregate_equivalent_requirements
from resume_evaluator.synthetic_vacancy_pipeline_simplified import MarketProfile, NormalizedVacancy


def reaggregate_saved_profile(profile_path: Path, cache_path: Path | None = None) -> MarketProfile:
    profile = MarketProfile.model_validate_json(profile_path.read_text(encoding="utf-8"))
    cache_path = cache_path or profile_path.with_name(f"{profile_path.stem}.market_profile_v2.normalization_cache.json")
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    by_id: dict[str, list[NormalizedVacancy]] = {}
    for value in cache.values():
        row = NormalizedVacancy.model_validate(value)
        by_id.setdefault(row.source_id, []).append(row)

    def cohort_result(cohort, label):
        rows = []
        for source in cohort.sources:
            matches = by_id.get(source.source_id, [])
            if len(matches) != 1 or matches[0].extraction_status != "ok":
                raise ValueError(f"Нет однозначного полного результата нормализации: {source.source_id}")
            rows.append(matches[0].model_copy(update={
                "source": source.source, "url": source.url, "title": source.title,
                "employer": source.employer, "employer_id": source.employer_id,
                "published_at": source.published_at, "region": source.region,
                "description_status": source.description_status,
            }))
        categories, variations, diagnostics = aggregate_equivalent_requirements(rows, use_llm=False)
        concept_path = profile_path.with_name(f"{profile_path.stem}.{label}_concepts.json")
        return cohort.model_copy(update={**categories, "market_variations": variations}), diagnostics, concept_path

    updated, direct, direct_path = cohort_result(profile, "direct")
    adjacent_result = None
    if profile.adjacent_context:
        adjacent_result = cohort_result(profile.adjacent_context, "adjacent")
        updated.adjacent_context = adjacent_result[0]
    updated.summary = (
        f"Ориентир для {profile.target.specialization or profile.target.role}: {profile.sample_size} прямых вакансий. "
        "Требования пересчитаны по сохранённым текстам с явными синонимами и морфологией; "
        "предложения модели об объединении не использованы. Выводы относятся только к этой выборке."
    )
    # Пишем только после успешной проверки обеих выборок.
    profile_path.write_text(updated.model_dump_json(indent=2), encoding="utf-8")
    direct_path.write_text(direct.model_dump_json(indent=2), encoding="utf-8")
    if adjacent_result:
        adjacent_result[2].write_text(adjacent_result[1].model_dump_json(indent=2), encoding="utf-8")
    profile_path.with_name(f"{profile_path.stem}.reaggregation.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "normalization_cache": str(cache_path), "main_sample": profile.sample_size,
        "adjacent_sample": profile.adjacent_context.sample_size if profile.adjacent_context else 0,
        "mode": "deterministic", "new_network_or_llm_calls": 0,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return updated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--normalization-cache", type=Path)
    args = parser.parse_args()
    result = reaggregate_saved_profile(args.profile, args.normalization_cache)
    print(f"Пересчитано: {args.profile}; основная выборка: {result.sample_size}")
