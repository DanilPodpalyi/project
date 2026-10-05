# MarketProfile

`MarketProfile` — прикладной обзор требований работодателей по небольшой
выборке актуальных вакансий для заданной позиции. Он не является «идеальной
вакансией», полной моделью рынка или оценкой конкретного кандидата.

## Публичный интерфейс

```python
from resume_evaluator.market_profile import (
    build_market_profile,
)

profile = build_market_profile(target="Middle ML Engineer")
print(profile.summary)
```

По умолчанию результат сохраняется в `market_profile.json`. CLI:

```powershell
python -m resume_evaluator.market_profile `
  "Middle ML Engineer" `
  output/market_profile.json
```

Функция принимает строку, `TargetProfile` или готовый `ProfessionSearchPlan`.
Она не принимает skills, проекты, образование или опыт кандидата. Если во
входном профиле присутствует `core_skills`, перед поиском он очищается. Для
готового plan поисковые запросы также строятся заново только из TargetProfile,
поэтому внешние candidate-specific запросы не попадают в retrieval.

## Продуктовый результат

Основной JSON содержит только:

- целевую роль, specialization, domain, seniority, role family и focus;
- размер использованной выборки 3–6;
- `professional_skills`, `tools_and_technologies`, `soft_skills`,
  `education_requirements`, `domain_knowledge` и `responsibilities`;
- frequency, coverage, importance и ссылки на вакансии для каждого пункта;
- единичные требования в `market_variations`;
- общий список источников, confidence и осторожный market summary.

Готовый валидируемый пример: `examples/precomputed/market_profile.example.json`.

Для выборки размера `N`:

- `common` — frequency не ниже `ceil(2N/3)`;
- `additional` — frequency от двух, но ниже common-порога;
- `market_variation` — требование только из одной вакансии.

Если строгую проверку проходят менее трёх вакансий, сохраняется результат со
`status: insufficient_market_data` и `confidence: low`. Плохие вакансии для
увеличения sample не добавляются.

## Переиспользованный pipeline

Без изменений концепции переиспользованы:

- `TargetProfile` и универсальные поисковые запросы;
- модульные providers, dedup и hybrid title matching;
- embeddings `bge-m3` и domain gate;
- Gemma-нормализация, cache и checkpoint;
- строгие role/specialization/domain/focus/seniority-фильтры;
- консервативная action+object проверка обязанностей.

Triplet больше не является продуктовым gate. Если уже есть 3–6 валидных
вакансий, профиль строится по всей выборке.

## DEBUG-артефакты

Рядом с `market_profile.json` по умолчанию сохраняются:

- `market_profile.retrieval_diagnostics.json` — Precision@20, provider metrics,
  title/embedding scores и причины отклонения;
- `market_profile.debug.json` — pipeline statistics, IDs market sample и
  диагностическая оценка лучшей тройки;
- `market_profile.normalization_checkpoint.json` и
  `vacancy_normalization_cache.json`.

Эти поля не входят в основной пользовательский JSON. Сохранение debug report
можно отключить аргументом `save_debug=False`; retrieval diagnostics и cache
остаются рабочими артефактами pipeline.

## Не входит в этот этап

Модуль не сравнивает профиль с резюме, не ищет пробелы кандидата, не даёт
рекомендаций и не подбирает конкретные вакансии человеку.

## Live regression

Ручной прогон 22 сентября 2026 года с переиспользованием существующего cache:

| Цель | Статус | Sample | Confidence |
|---|---|---:|---|
| Middle ML Engineer | `insufficient_market_data` | 0 | low |
| Корреспондент | `insufficient_market_data` | 1 | low |
| Маркетолог | `ok` | 6 | medium |
| PR-специалист | `insufficient_market_data` | 0 | low |

У маркетолога диагностическая лучшая тройка сохранила
`triplet_confidence: low_confidence`, но шесть вакансий прошли строгую
профессиональную проверку. Поэтому продуктовый профиль был создан по всему
sample, как и требует новый контракт. В него вошли шесть источников, пять
повторяющихся professional/tool требований и одна повторяющаяся обязанность;
единичные требования сохранены отдельно.

Live-артефакты находятся в `output/market_profile_live/<case>/`.
