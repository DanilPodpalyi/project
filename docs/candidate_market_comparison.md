# CandidateMarketComparison

`CandidateMarketComparison` сопоставляет содержание уже структурированного
резюме с **готовым и неизменяемым** `MarketProfile`.

Модуль отвечает на вопрос:

> Как содержание этого резюме соотносится с требованиями доступной рыночной
> выборки для выбранной профессии?

Он не выполняет поиск вакансий, не вызывает `build_market_profile()`, не
переранжирует market sample под кандидата, не генерирует рекомендации и не
рассчитывает общий market-fit score.

## Публичный API

```python
from resume_evaluator.candidate_market_comparison import (
    compare_candidate_to_market,
)

comparison = compare_candidate_to_market(
    resume=resume_data,
    market_profile=market_profile,
)
```

По умолчанию результат сохраняется в `candidate_market_comparison.json`.
Запись можно отключить, передав `output_path=None`.

CLI-вариант:

```powershell
python -m resume_evaluator.candidate_market_comparison `
  output/resume.json `
  output/market_profile.json `
  output/candidate_market_comparison.json
```

Для полностью детерминированного lexical-only режима доступен флаг
`--no-embeddings`. В обычном режиме все тексты обрабатываются одним batch
embedding-вызовом `bge-m3`; при недоступности Ollama сравнение безопасно
продолжается с лексическим matching без тяжёлой LLM.

## Состояния требования относительно резюме

- `evidenced` — требование подтверждено опытом, проектом или достижением;
- `matched` — требование явно найдено в подходящем структурированном разделе,
  например образование в `education`;
- `mentioned_only` — навык заявлен в списке навыков, но практическое
  подтверждение не найдено;
- `not_found` — упоминание не обнаружено в предоставленном резюме.

`not_found` описывает только содержание документа и не означает, что кандидат
не владеет навыком.

Уверенность `low/medium/high` и `market_reliability` переносятся из профиля
без повышения. Для каждого требования сохраняются `market_sources`,
`market_source_values`, `market_evidence`, частота и размер выборки. Эти данные
переходят в основания рекомендаций. Сохранённые эквивалентные формы участвуют
в сопоставлении с резюме; пунктуация в конце предложения не скрывает навык,
при этом `.NET`, `Node.js`, C++ и C# сохраняются как разные инструменты.

Evidence имеет строгую привязку к источнику:

Списки `technologies` внутри работы или проекта дают `mentioned_only`.
Для `evidenced` нужен пример применения в задачах, описании проекта или
результатах. Одно название инструмента в стеке не считается доказательством.
`unconfirmed_resume_skills` охватывает навыки всего резюме, включая те,
которые не вошли в повторяющиеся требования рынка; одинаковые названия
инструментов учитываются один раз.

- обязанности подтверждаются только `experience`, `projects` или
  `achievements`;
- образование подтверждается только `education` или `courses`;
- текст `summary` не может дать полноценный match этим двум категориям;
- для именованных навыков и инструментов одного высокого cosine similarity
  недостаточно: требуется лексическое или каноническое совпадение;
- semantic matching обязанностей дополнительно требует совпадения якорей
  действия и объекта.

Категории `professional_skills`, `tools_and_technologies`, `responsibilities`,
`education_requirements`, `domain_knowledge` и `soft_skills` сравниваются
раздельно. Обязанности также вынесены в собственные `responsibility_matches`
и `responsibility_gaps`.

Единичные `market_variations` сопоставляются для контекста, но их отсутствие
не попадает в списки пробелов. Навыки кандидата, которых нет среди рыночных
`common` и `additional`, сохраняются в `candidate_only_skills` без штрафа.

Если входной рыночный профиль имеет статус, отличный от `ok`, модуль не делает
сравнение и возвращает `status: market_data_unavailable`.

## Практический пример: Маркетолог

Для проверки использованы без изменений:

- live `MarketProfile`: `output/market_profile_live/marketer/market_profile.json`;
- тестовое резюме: `examples/precomputed/marketing_candidate.example.json`;
- результат: `output/candidate_market_comparison_live/candidate_market_comparison.json`.

Фактический результат:

| Блок | Результат |
|---|---|
| Strong matches | `Automation` — evidenced; `Campaign execution` — evidenced; `HubSpot` — evidenced |
| Missing common | нет: в исходном live MarketProfile нет требований с importance `common` |
| Missing additional | `Project Management` — not_found |
| Mentioned only | `Salesforce` |
| Candidate-only skills | `Adobe Illustrator` — mentioned_only |
| Responsibility match | `Execute regional campaigns` — evidenced |

Прозрачные метрики прогона:

```json
{
  "common_requirements_total": 0,
  "common_requirements_matched": 0,
  "common_requirements_evidenced": 0,
  "common_requirements_mentioned_only": 0,
  "additional_requirements_total": 5,
  "additional_requirements_matched": 4,
  "additional_requirements_evidenced": 3,
  "candidate_only_skills_total": 1,
  "responsibility_requirements_total": 1,
  "responsibility_requirements_matched": 1
}
```

В примере нет рекомендаций и интегрального балла: JSON хранит факты для
будущего отдельного `RecommendationGenerator`.
