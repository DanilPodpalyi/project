# Поиск вакансий по резюме

`search_vacancies.py` берёт поле `target_position` из структурированного JSON
резюме и ищет соответствующие вакансии через публичный API hh.ru. Модель для
этого этапа не используется.

Сначала создайте структурированный JSON резюме:

```powershell
python extract_resume.py examples/resume.txt output/resume.json
```

Затем выполните поиск по всем регионам:

```powershell
python search_vacancies.py output/resume.json output/vacancies_found.json
```

Можно ограничить регион и количество результатов:

```powershell
python search_vacancies.py output/resume.json output/vacancies_found.json `
  --area 3 --limit 20
```

`area` — идентификатор региона в hh.ru. Например, `1` — Москва, `2` —
Санкт-Петербург, `3` — Екатеринбург. Без параметра поиск выполняется по всем
регионам. За один запрос можно получить от 1 до 50 вакансий.

В итоговом JSON сохраняются название, компания, регион, зарплата, требуемый
опыт, тип занятости, график, фрагменты требований и обязанностей, дата
публикации и ссылка на вакансию.
