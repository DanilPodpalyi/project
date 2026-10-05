> Историческая диагностика до релиза 0.2.0. Указанные output-файлы не входят в поставку; актуальная проверка — в ../release_acceptance.md.

# Regression-отчёт универсального market pipeline

Дата: 21 сентября 2026 года.

## Детерминированные cases

Один integration-тест запускает общие функции построения запросов,
профессионального ранжирования, формирования market sample и выбора тройки без
интернета и без навыков кандидата.

| Цель | Role family | Focus | Валидных вакансий | Явно другая профессия | Extraction | Confidence | Время локальной части |
|---|---|---|---:|---|---|---|---|
| ML Engineer / Middle | ML Engineering | Model Development / Production ML | 3 | AI Infrastructure Engineer исключена | `ok` | medium/high | < 1 с |
| Корреспондент | Journalism | News Reporting | 3 | PR Manager исключён | `ok` | medium/high | < 1 с |
| Маркетолог | Marketing | Brand Marketing | 3 | PR Manager исключён | `ok` | medium/high | < 1 с |
| PR-специалист | Public Relations | Media Relations | 3 | Correspondent/Reporter исключены | `ok` | medium/high | < 1 с |

Тест проверяет для каждого case:

- заполнение `TargetProfile`;
- наличие specialization, alias и seniority-варианта в запросах;
- однородность market sample;
- отсутствие явно исключённой профессии;
- допустимый extraction status;
- наличие трёх валидных вакансий;
- прохождение confidence-порогов;
- ограничение времени локальной части.

Отдельный integration-тест подтверждает динамическое расширение normalization
pool: `8 → 12 → 16`. При последовательном появлении `2 → 4 → 6` качественных
вакансий pipeline останавливается на шести и рассчитывает частоты по фактическому
`N=6`.

## Живой ML regression

Запущен отдельный тест `ML Engineer / Middle` с публичными источниками и
локальными моделями. Результат:

- поиск и prefilter завершились;
- неизменённый embedding-фильтр с прежним domain threshold оставил меньше трёх
  вакансий;
- статус: `insufficient_market_data`;
- synthetic vacancy не создана;
- приблизительное время: 102 секунды.

Это корректный защитный результат: пороги не ослаблялись и нерелевантные
вакансии не добавлялись ради достижения размера выборки. Живые не-IT прогоны в
этом изменении не использовались как обязательный unit gate, поскольку результат
публичных API нестабилен; их профили покрыты воспроизводимыми integration-cases.
