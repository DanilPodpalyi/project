> Историческая диагностика до релиза 0.2.0. Указанные output-файлы не входят в поставку; актуальная проверка — в ../release_acceptance.md.

# Live retrieval diagnostics

Дата прогонов: 22 сентября 2026 года.

Порог `domain_similarity=0.35`, embedding-модель `bge-m3`, TargetProfile,
нормализация, seniority/focus-фильтры, aggregation, frequency rules и
confidence-пороги во время диагностики не менялись.

Полные построчные данные сохранены рядом с артефактами прогонов:

- `output/live_retrieval_diagnostics/ml_engineer_middle.synthetic_vacancy.retrieval_diagnostics.json`;
- `output/live_retrieval_diagnostics/correspondent.synthetic_vacancy.retrieval_diagnostics.json`;
- `output/live_retrieval_diagnostics/marketer.synthetic_vacancy.retrieval_diagnostics.json`;
- `output/live_retrieval_diagnostics/pr_specialist.synthetic_vacancy.retrieval_diagnostics.json`.

Каждая строка содержит source, title, исходный search query, target
specialization/domain, prefilter score, specialization/domain similarity,
итоговый embedding score, результат фильтра и причину отказа.

## Сводка прогонов

| Цель | Raw | Current | Dedup | Prefilter | Embedding threshold | Borderline | Normalized | Strict matches | Market sample | Время |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ML Engineer / Middle | 523 | 506 | 385 | 20 | 1 | 0 | 0 | 0 | 0 | 49.9 с |
| Корреспондент | 307 | 295 | 273 | 20 | 0 | 0 | 0 | 0 | 0 | 33.3 с |
| Маркетолог | 455 | 455 | 422 | 20 | 11 | 0 | 8 | 7 | 6 | 338.4 с |
| PR-специалист | 379 | 369 | 311 | 20 | 0 | 0 | 0 | 0 | 0 | 47.9 с |

`Маркетолог` дошёл до строгой нормализации. Семь вакансий прошли профильные
правила, шесть вошли в market sample, но лучшая тройка осталась ниже
существующих confidence-порогов, поэтому synthetic vacancy не создана.

## Распределения embedding scores

| Цель | Specialization min / median / max | Domain min / median / max | Total min / median / max |
|---|---|---|---|
| ML Engineer | 0.38 / 0.53 / 0.77 | 0.38 / 0.45 / 0.49 | 0.38 / 0.48 / 0.61 |
| Корреспондент | 0.30 / 0.34 / 0.49 | 0.30 / 0.36 / 0.43 | 0.30 / 0.37 / 0.43 |
| Маркетолог | 0.41 / 0.54 / 0.64 | 0.34 / 0.45 / 0.53 | 0.40 / 0.58 / 0.70 |
| PR-специалист | 0.39 / 0.54 / 0.65 | 0.32 / 0.39 / 0.44 | 0.40 / 0.47 / 0.56 |

Во всех четырёх случаях число отказов непосредственно по domain threshold
равно нулю. Кандидаты отсеивались раньше по specialization similarity. Поэтому
наблюдаемые ранние остановки нельзя приписать hard gate домена, несмотря на
старое обобщённое сообщение pipeline «после проверки domain».

## Примеры

Очевидно релевантные маркетинговые вакансии:

| Title | Specialization | Domain | Total | Строгая проверка |
|---|---:|---:|---:|---|
| Marketing Operations Specialist | 0.62 | 0.51 | 0.68 | прошла |
| Growth Marketer | 0.57 | 0.49 | 0.65 | прошла |
| Lifecycle Marketing Automation Specialist | 0.58 | 0.50 | 0.65 | прошла |

Очевидно нерелевантные примеры:

| Target | Title | Specialization | Domain | Total | Результат |
|---|---|---:|---:|---:|---|
| Корреспондент | General Counsel | 0.49 | 0.35 | 0.43 | отклонена |
| PR-специалист | Human Resources Specialist | 0.64 | 0.41 | 0.56 | отклонена |
| ML Engineer | Frontend Engineer | 0.70 | 0.40 | 0.57 | отклонена |

ML-кандидаты `Machine Learning Engineering Manager` и `Ads Conversion
Modeling, Machine Learning Engineering Manager` семантически близки по теме,
но являются manager-позициями при цели Middle. `Internship Machine Learning…`
не соответствует seniority. Они не подтверждают безопасный false negative для
целевого market sample.

## Источники

- ML: Jobicy 155, JobsCollider 326, Remote OK 42.
- Корреспондент: Jobicy 7, JobsCollider 300.
- Маркетолог: Jobicy 100, JobsCollider 300, Remote OK 55.
- PR: Jobicy 114, JobsCollider 253, Remote OK 12.

Во всех прогонах Remotive, Arbeitnow и hh.ru отвечали HTTP 403. Для двух
русскоязычных длинных PR-запросов Jobicy дополнительно вернул HTTP 400.
JobsCollider дал большой объём, но для корреспондента и PR почти все ответы
оказались тематически посторонними.

## Вывод по причинам

1. **Качество поиска — основная проблема.** Для корреспондента и PR в TOP-20
   практически нет вакансий целевой профессии. Для ML преобладают manager,
   internship и смежные технические роли.
2. **Доступность источников — существенная проблема.** Три источника стабильно
   недоступны с HTTP 403, из-за чего выборка зависит от JobsCollider и Jobicy.
3. **Domain representation слабо разделяет классы**, но не является причиной
   раннего отказа в этих прогонах: relevant marketing и нерелевантные записи
   имеют пересекающиеся значения примерно 0.35–0.53.
4. **Глобальный domain threshold не требует изменения.** Ни одного кандидата,
   прошедшего specialization gate и отклонённого только по domain, не найдено.
5. **Borderline fallback не активирован.** Диагностика не подтвердила хорошие
   вакансии, потерянные именно domain hard gate. Добавление нерелевантных TOP
   ranking в Gemma увеличило бы стоимость, но не исправило бы состав поиска.

Итоговая причина — комбинация низкой точности поиска для части профессий и
недоступности источников; embedding domain representation является вторичным
ограничением, а не подтверждённым ранним bottleneck в этих четырёх прогонах.

## После hybrid title matching

Повторный live-прогон выполнен 22 сентября 2026 года после изменения только
retrieval-слоя. TargetProfile, `bge-m3`, Gemma, normalization, downstream
проверки, aggregation и confidence-пороги не менялись. Артефакты находятся в
`output/live_retrieval_hybrid/`.

`Precision@K` ниже — доля TOP-K, которую последующая строгая нормализация
подтвердила как профессионально релевантную. Ненормализованные и отклонённые
hybrid-фильтром элементы считаются неподтверждёнными. В sidecar также есть
`hybrid_acceptance_at_K` и conditional `validated_precision_at_K`, чтобы не
смешивать раннее решение retrieval с качеством уже проверенных вакансий.

| Цель | Raw | Dedup | Embedding: до → после | Normalized | Strict | Market | P@10 | P@20 | Exact/alias | Semantic-only | Excluded rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ML Engineer / Middle | 644 | 262 | 1 → 4 | 4 | 0 | 0 | 0.00 | 0.00 | 1 | 15 | 21 |
| Корреспондент | 219 | 97 | 0 → 6 | 6 | 1 | 1 | 0.10 | 0.05 | 2 | 14 | 22 |
| Маркетолог | 506 | 329 | 11 → 11 | 8 | 7 | 6 | 0.70 | 0.35 | 5 | 9 | 22 |
| PR-специалист | 518 | 314 | 0 → 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0 | 20 | 21 |

Сравнение `Embedding: до → после` показывает размер пула, прошедшего ранний
retrieval. Старый отчёт не имел ground-truth P@10/P@20, поэтому прямое числовое
сравнение precision с baseline было бы некорректным. Новые метрики образуют
baseline для следующих изменений источников.

### Проверка известных false friends

- `Frontend Engineer` для ML Engineer: dense similarity остаётся высокой, но
  кандидат получает `semantic_only_without_lexical_support`.
- `Human Resources Specialist` / `HR Operations Specialist` для PR: не имеют
  лексического подтверждения `PR`/`Public Relations` и не проходят.
- `General Counsel` для корреспондента: не проходит независимо от dense score.
- `Machine Learning Engineer`, `Reporter, News`, `Founding Marketer` и
  `Marketing Specialist` сохраняются благодаря specialization/alias/phrase
  сигналам.

В ML-прогоне точный `Machine Learning Engineer` был найден, но после
нормализации оказался `Staff`, поэтому корректно не вошёл в Middle sample.
Manager-вакансии также были отклонены downstream-проверкой. Для корреспондента
точные запросы впервые дали шесть кандидатов; только `Reporter, News` прошёл
строгую проверку. Это улучшение recall без ослабления downstream, но данных всё
ещё недостаточно для synthetic vacancy.

### Качество источников

Динамическая статистика показала:

- ML: title precision на рассмотренной части Jobicy `0.079`, JobsCollider
  `0.043`, RemoteOK `0.000`;
- корреспондент: JobsCollider `0.088`, остальные `0.000`;
- маркетолог: Jobicy `0.303`, JobsCollider `0.032`, RemoteOK `0.000`;
- PR: у всех доступных источников `0.000`.

Jobicy, JobsCollider и RemoteOK доступны не для всех рынков одинаково хорошо.
Remotive, Arbeitnow и hh.ru продолжили отвечать HTTP 403; отдельные длинные
запросы Jobicy дали HTTP 400. Generic web search реализован как необязательный
JSON provider через `GENERIC_JOB_SEARCH_ENDPOINT`; без настроенного легального
API он не включается и не пытается скрейпить поисковики.

### Вывод

Hybrid matching устранил подтверждённые dense false positives и сохранил
правильные aliases без profession-specific hardcode. Для маркетолога качество
осталось высоким (`7/8` нормализованных кандидатов валидны); для корреспондента
появился один строгий match вместо нуля. Главный оставшийся bottleneck — coverage
источников, особенно для PR и русскоязычных/не-remote профессий. Понижать
domain threshold или ослаблять strict validation по этим данным не следует.
Следующий обоснованный эксперимент — подключение легального generic search API
и отдельный benchmark provider coverage; лёгкий reranker имеет интерфейс и
quality/time harness, но отдельная модель пока не добавлялась.
