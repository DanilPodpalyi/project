> Актуальная установка и основной запуск описаны в ../README.md. Команда `analyze` использует российский сценарий по резюме; ручной target сохраняет совместимость прежнего сценария.

# Единый сценарий оценки кандидата

`candidate_assessment_pipeline.py` объединяет существующие этапы, не меняя их
логику и не пересчитывая готовые результаты:

```text
resume.pdf/docx/json
        ↓
ResumeData → ResumeQualityResult
        ↓
MarketProfile по заданной профессии
        ↓
CandidateMarketComparison
        ↓
RecommendationResult
        ↓
candidate_report.json + candidate_report.md
```

## Запуск с нуля

Рекомендуемый сценарий для России автоматически берёт должность и указанный
уровень из резюме и использует вакансии за последние 120 дней:

```powershell
python -m resume_evaluator.candidate_assessment_pipeline resume.pdf output/candidate
```

Полные описания собираются из HH и «Работа России», исходные тексты и ошибки
сохраняются в `.stages/market_profile.collection.json`. Для подготовленной
выборки доступен `--vacancies-json vacancies.json`.
Подробнее: [russian_market.md](russian_market.md).

Ниже сохранён старый интерфейс с ручной профессией:

```powershell
python -m resume_evaluator.candidate_assessment_pipeline `
  C:\path\resume.pdf `
  "ML Engineer / Middle" `
  output\candidate
```

Для opt-in проверки упрощённого MarketProfileBuilder v2 добавьте
`--market-v2`. Без флага по-прежнему используется production builder v1:

```powershell
python -m resume_evaluator.candidate_assessment_pipeline `
  resume.json "Middle ML Engineer" output/candidate_report_v2 `
  --market-v2
```

Если указан `--market-profile`, готовый JSON имеет приоритет и новый retrieval
не запускается независимо от `--market-v2`.

Поддерживаются PDF, DOCX, TXT и готовый JSON схемы `ResumeData`.

Markdown по умолчанию краткий: советы без повторов основания, шагов,
ожидаемого результата и одних и тех же ссылок. Основные источники перечислены
один раз, смежная выборка — отдельно. `--detailed-report` включает подробный
Markdown. JSON в обоих режимах сохраняет полные доказательства, действия
и происхождение требований.

## Повторный запуск без пересчёта

```powershell
python -m resume_evaluator.candidate_assessment_pipeline `
  output\resume.json `
  "ML Engineer / Middle" `
  output\candidate `
  --resume-evaluation output\resume_quality.json `
  --market-profile output\market_profile.json `
  --no-recommendation-llm
```

Готовые `ResumeQualityResult` и `MarketProfile` валидируются и используются без
повторного вызова соответствующих модулей. По умолчанию comparison использует
`bge-m3`; для детерминированной диагностики доступен
`--no-comparison-embeddings`.

## Результаты

В корне выходного каталога создаются два пользовательских файла:

- `candidate_report.json` — компактный структурированный результат;
- `candidate_report.md` — отчёт для чтения кандидатом.

Полные результаты этапов сохраняются в служебном каталоге `.stages` для аудита.
Итоговый JSON не копирует сырые вакансии или полный текст резюме. Он хранит:

- исходный балл и разбивку `ResumeQualityResult`;
- размер, confidence, summary и ссылки `MarketProfile`;
- существенные факты `CandidateMarketComparison` с evidence;
- пользовательские тексты `RecommendationResult` и короткие ссылки на basis;
- шаги действий, ожидаемый результат и основания каждого совета: частоту,
  источники, формулировки требований и найденное подтверждение в резюме;
- время каждого этапа.

Для v2 `market_context.adjacent_context` содержит отдельную выборку смежных
вакансий, их требования, оригинальные формы из нормализованных карточек и
источники. В Markdown это блок «Смежные вакансии: контекст возможного развития».
Основное сравнение и рекомендации используют только прямые вакансии;
смежные требования не добавляются в gaps и метрики. Контекст доступен и при
недостаточном количестве прямых вакансий.

`market_context.reliability` показывает причины уверенности в выборке.
`low` сохраняется до каждого рыночного совета и отображается как предварительный
ориентир. В рекомендациях `basis` остаётся совместимым списком ссылок, а
`support` содержит структурированные доказательства. Markdown показывает
шаги, ожидаемый результат, частоту требования, подтверждения в резюме и до трёх
примеров вакансий со ссылками и сохранёнными формулировками. Все источники
остаются доступными в JSON.

Новый итоговый score не рассчитывается.

Если MarketProfile имеет `insufficient_market_data`, отчёт получает статус
`partial`: оценка качества резюме остаётся доступной, а рыночные gaps не
формируются.

## Практический прогон

Для реального структурированного резюме ML Engineer/Data Scientist создано два
результата:

- `output/candidate_report_live/ml_engineer_real` — live MarketProfile оказался
  недостаточным, поэтому отчёт корректно имеет статус `partial`;
- `output/candidate_report_live/ml_engineer_real_with_market` — полная
  демонстрация с сохранённым валидным примером MarketProfile из четырёх
  вакансий, без повторной оценки резюме.

Во втором отчёте сохранён исходный балл качества `48/100`. Python найден только
в списке навыков и имеет статус `mentioned_only`; формулировки Machine Learning
и PyTorch не обнаружены как те же сущности. Система не делает из этого вывод,
что кандидат ими не владеет, и не критикует candidate-only навыки.
