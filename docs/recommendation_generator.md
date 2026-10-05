# RecommendationGenerator

`RecommendationGenerator` превращает уже рассчитанные факты из
`ResumeQualityResult` и `CandidateMarketComparison` в рекомендации кандидату.
Он не выполняет поиск, не строит `MarketProfile` и не запускает повторное
сравнение кандидата с рынком.

## API

```python
from resume_evaluator.recommendation_generator import (
    generate_candidate_recommendations,
)

result = generate_candidate_recommendations(
    resume=resume,
    resume_evaluation=resume_evaluation,
    market_comparison=market_comparison,
)
```

CLI:

```powershell
python -m resume_evaluator.recommendation_generator `
  output/resume.json `
  output/resume_quality.json `
  output/candidate_market_comparison.json `
  output/candidate_recommendations.json
```

Флаг `--no-llm` формирует полностью детерминированный шаблонный результат.

Гибкие навыки выводятся как короткие условные рекомендации для указания
в резюме, без `action` и учебных проектов. Они не попадают в направления
развития. Образование имеет отдельные действия по уточнению квалификации;
учебный проект не заменяет диплом. Счётчик неподтверждённых навыков охватывает
всё резюме, а не только до пяти выбранных советов по требованиям рынка.

## Схема результата

```json
{
  "status": "ok",
  "candidate_name": "...",
  "target": {
    "specialization": "...",
    "seniority": "..."
  },
  "resume_score": 51,
  "market_context": {
    "available": true,
    "sample_size": 6,
    "confidence": "medium"
  },
  "strengths": [],
  "resume_improvements": [],
  "market_gaps": [],
  "unconfirmed_skills": [],
  "development_directions": [],
  "final_summary": "...",
  "generation_mode": "llm | fallback",
  "llm_calls": 0
}
```

Каждый элемент рекомендации содержит:

```json
{
  "fact_id": "market_gap:0:SQL",
  "title": "В резюме не обнаружено: SQL",
  "text": "...",
  "basis": [
    {
      "type": "missing_common",
      "reference": "SQL",
      "market_importance": "common",
      "evidence": []
    }
  ]
}
```

Новый market-fit score не рассчитывается. `resume_score` переносится из
существующей оценки качества резюме.

## Действия и проверяемые основания

У рыночного совета теперь есть `action` с типом, последовательностью шагов и
ожидаемым результатом:

| Подтверждение в резюме | Действие |
|---|---|
| Навык подтверждён опытом | `highlight_evidence`: сделать реальный пример заметнее, показать задачу, действия и подтверждённый результат |
| Навык только перечислен | `add_application_example`: добавить пример применения из реальной работы или проекта |
| Подтверждение не найдено | `clarify_or_develop`: проверить имеющийся опыт; если он есть — показать его, если нет — рассмотреть учебную задачу |

Для образования отдельные шаги: уточнить имеющуюся квалификацию и условия
вакансии. Учебный проект не подменяет диплом. Пробелы в повторяющихся
обязанностях также получают конкретное действие.

`basis` сохраняет частоту и размер выборки, coverage, категорию требования,
статус подтверждения, исходную уверенность рынка, независимых работодателей,
ссылки `market_sources`, формы `market_source_values` и привязанные к источникам
формулировки `market_evidence`. `evidence` содержит подтверждения из резюме.
Пустое подтверждение означает отсутствие доказательства в документе.

`market_context.confidence` переносится без повышения. `confidence` отдельного
совета может быть ниже: если требование опирается на одного работодателя или
источники не известны, совет предварительный; для дополнительных требований
уровень не превышает `medium`. При `low` текст явно содержит «Предварительный
совет по ограниченной выборке». Причины общей уверенности доступны в
`market_context.reliability`.

## Роль LLM и защита от новых фактов

Python сначала детерминированно формирует полный каталог допустимых пунктов с
`fact_id`, `basis` и безопасным шаблонным текстом. Gemma получает только этот
каталог и может изменить формулировки, но не состав рекомендаций.

После ответа проверяются:

- точный набор и порядок `fact_id` в каждом разделе;
- отсутствие новых технологий, чисел и содержательных утверждений;
- соответствие Pydantic-схеме;
- сохранение исходного машинного `basis`.
- сохранение действий, результата и уверенности, которые модель не редактирует;
- условность советов по ненайденным требованиям и отсутствие утверждений,
  что кандидат не владеет навыком.

Неизвестный факт отклоняет ответ целиком. Повреждённый JSON или нарушение
safety-check получает одну repair-попытку. После второй ошибки используется
готовый fallback. При сетевой ошибке fallback применяется сразу.

Обычный успешный режим выполняет один LLM-вызов, repair — максимум два,
`--no-llm` — ни одного.

## Live-пример: Маркетолог

Входы:

- `examples/precomputed/marketing_candidate.example.json`;
- `examples/precomputed/marketing_resume_evaluation.example.json`;
- `output/candidate_market_comparison_live/candidate_market_comparison.json`.

Результат:

- Automation, Campaign execution, HubSpot и Execute regional campaigns
  показаны как подтверждённые сильные стороны с evidence;
- Salesforce вынесен в `unconfirmed_skills`: навык указан, но применение не
  показано;
- Project Management указан как `additional`-требование, не обнаруженное в
  резюме, с условной формулировкой без предложения приписать опыт;
- Adobe Illustrator не критикуется и не превращается в market gap;
- улучшения качества касаются только фактов из `ResumeQualityResult`.

Готовый безопасный результат находится в
`output/recommendation_live/marketing_recommendations.json`.
