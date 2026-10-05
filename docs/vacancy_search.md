> Совместимый legacy-сценарий. Основная команда релиза: `python -m resume_evaluator analyze ...`; этот документ описывает прежний модуль.

# Поиск вакансий по резюме

`resume_evaluator.vacancy_search` берёт `specialization` и `domain` из объекта
`target_profession` структурированного JSON резюме и ищет соответствующие
вакансии через публичный API hh.ru. Старое строковое поле `target_position`
также поддерживается. Модель для этого этапа не используется.

Сначала создайте структурированный JSON резюме:

```powershell
python -m resume_evaluator.resume_extractor examples/resumes/anna_sokolova.docx output/resume.json
```

Затем выполните поиск по всем регионам:

```powershell
python -m resume_evaluator.vacancy_search output/resume.json output/vacancies_found.json
```

Можно ограничить регион и количество результатов:

```powershell
python -m resume_evaluator.vacancy_search output/resume.json output/vacancies_found.json `
  --area 3 --limit 20
```

`area` — идентификатор региона в hh.ru. Например, `1` — Москва, `2` —
Санкт-Петербург, `3` — Екатеринбург. Без параметра поиск выполняется по всем
регионам. За один запрос можно получить от 1 до 50 вакансий.

В итоговом JSON сохраняются название, компания, регион, зарплата, требуемый
опыт, тип занятости, график, фрагменты требований и обязанностей, дата
публикации и ссылка на вакансию.
