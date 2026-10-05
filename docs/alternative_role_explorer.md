# AlternativeRoleExplorer

`AlternativeRoleExplorer` предлагает альтернативные профессиональные
направления, а не конкретные объявления о работе.

## Контракт

Вход:

- готовый `ResumeData`;
- каталог для аудиторских артефактов;
- число ролей, которые разрешено проверить через рынок (по умолчанию 3).

Выход: `alternative_roles.json` со следующими данными:

- роль, specialization, domain, seniority, role family и focus;
- дословные подтверждения из резюме;
- статус рыночной проверки;
- размер market sample и confidence;
- совпавшие требования и переходные пробелы;
- ограниченный readiness score для ранжирования альтернатив;
- пути к служебным MarketProfile и CandidateMarketComparison.

Readiness score не является вероятностью трудоустройства или основанием для
найма. Он сравнивает только предложенные направления по доступному резюме и
небольшим рыночным выборкам.

## Процесс

```text
ResumeData
    ↓
Gemma: гипотезы ролей + дословные evidence_refs
    ↓
Python: evidence validation, dedup, seniority и primary-role checks
    ↓
TOP-1–5 доказательных гипотез
    ↓
независимый MarketProfileBuilder для каждой роли
    ↓
CandidateMarketComparison
    ↓
ранжированный AlternativeRoleExplorerResult
```

Гипотеза отбрасывается до retrieval, если:

- она повторяет основную целевую профессию;
- содержит seniority выше явно указанного основного уровня;
- имеет меньше двух подтверждённых фактов;
- подтверждается только общими skills/soft skills;
- evidence является пересказом или отсутствует в резюме.

Одна строка `Python`, `SQL`, `Excel` или другого общего навыка не считается
достаточным основанием для новой профессии.

## Изоляция рынка

Навыки кандидата используются только для генерации и предварительной проверки
гипотез. В `MarketProfileBuilder` передаётся очищенный `ProfessionProfile` с
пустым `core_skills`. Рыночный эталон строится независимо, после чего готовый
MarketProfile сравнивается с кандидатом существующим модулем.

Основной JSON не содержит названий работодателей, ссылок или конкретных
объявлений. Они доступны только в служебных MarketProfile-артефактах как
доказательства частот требований.

## Запуск отдельно

```powershell
python -m resume_evaluator.alternative_role_explorer `
  output/resume.json `
  output/alternative_roles `
  --limit 3
```

## Запуск в общем отчёте

```powershell
python -m resume_evaluator.candidate_assessment_pipeline `
  output/resume.json `
  "Middle ML Engineer" `
  output/candidate `
  --alternative-roles `
  --alternative-role-limit 3
```

Ветка опциональна, потому что каждая проверяемая роль запускает отдельный live
market retrieval. Без `--alternative-roles` прежний E2E выполняется без новых
вызовов и сохраняет обратную совместимость.
