# Resume Evaluator

Локальный Python-проект для извлечения данных из резюме и вакансии,
детерминированного расчёта соответствия и генерации рекомендаций кандидату.
Структурирование текста и рекомендации выполняет локальная модель через Ollama;
итоговый балл рассчитывается обычным Python-кодом и не зависит от мнения LLM.

## Возможности

- чтение резюме и вакансий в форматах TXT, DOCX и PDF;
- преобразование документов в валидируемые JSON-схемы Pydantic;
- оценка соответствия кандидата вакансии по шкале 0–100;
- краткая и подробная рекомендации кандидату;
- Отдельный файл: независимая оценка качества резюме без вакансии;
- поиск актуальных вакансий через публичный API hh.ru (не работает без api);
- автоматическое построение синтетической эталонной вакансии по 20+ публичным (тесты еще идут)
  вакансиям с доказательными ссылками;
- локальная проверка зависимостей, схем и готовых JSON;
- unit-тесты основных правил без обязательного запуска Ollama.

## Важное ограничение

Проект является вспомогательным инструментом. Итоговый балл не должен быть
единственным основанием для найма или отказа. Возраст, пол, дата рождения,
инвалидность и другие чувствительные признаки не должны участвовать в оценке.

## Структура проекта

```text
resume-evaluator/
├── resume_evaluator/                 # Весь исполняемый код
│   ├── __init__.py                   # Объявление Python-пакета и версии
│   ├── __main__.py                   # Запуск: python -m resume_evaluator
│   ├── pipeline.py                   # Полный конвейер из пяти этапов
│   ├── document_reader.py            # Чтение TXT, DOCX и PDF
│   ├── resume_extractor.py           # Документ резюме -> ResumeData JSON
│   ├── vacancy_extractor.py          # Документ вакансии -> VacancyData JSON
│   ├── candidate_scorer.py           # Детерминированная оценка 0–100
│   ├── short_recommendation.py       # Краткая рекомендация кандидату
│   ├── detailed_recommendation.py    # Подробная доказательная рекомендация
│   ├── resume_quality_evaluator.py   # Оценка качества резюме без вакансии
│   ├── vacancy_search.py             # Поиск вакансий через API hh.ru
│   ├── synthetic_vacancy_pipeline.py # Эталонная вакансия по резюме
│   ├── system_check.py               # Диагностика без загрузки модели
│   └── model_check.py                # Контрольный запрос к Ollama
├── tests/                            # Локальные unit-тесты, исключены из Git
├── examples/                         # Локальные примеры, исключены из Git
├── docs/                             # Локальная дополнительная документация исключены из Git
├── data/
│   ├── reference/                    # Локальные справочники, исключены из Git
│   │   ├── industries.json           # 35 отраслей исключены из Git
│   │   └── professions.json          # Справочник профессий исключены из Git
│   └── raw/                          # Локальные данные, исключены из Git
├── output/                           # Результаты запусков, исключены из Git
├── requirements.txt                 # Python-зависимости
├── .gitignore                       # Исключения локальных данных и результатов
└── README.md                         # Этот файл
```

### Назначение основных модулей

#### `pipeline.py`

Главная точка оркестрации. Последовательно:

1. читает и структурирует резюме;
2. читает и структурирует вакансию;
3. рассчитывает соответствие;
4. создаёт краткую рекомендацию;
5. создаёт подробную рекомендацию.

#### `document_reader.py`

Единый слой чтения документов. TXT читается как UTF-8, DOCX — через
`python-docx`, PDF — через `pypdf`. Для сканированных PDF требуется
предварительный OCR: модуль работает только с текстовым слоем.

#### `resume_extractor.py` и `vacancy_extractor.py`

Отправляют текст локальной модели `gemma4:12b`, требуют структурированный
ответ по JSON-схеме и проверяют его через Pydantic. Модули можно запускать
отдельно для подготовки промежуточных JSON.

#### `candidate_scorer.py`

Не обращается к LLM. Сопоставляет обязательные и желательные навыки, опыт и
образование, формирует итоговый балл, расшифровку критериев, совпавшие и
неподтверждённые требования.

#### `short_recommendation.py` и `detailed_recommendation.py`

Используют уже рассчитанный балл и не пересчитывают его. Краткий модуль даёт
сжатый результат, подробный — доказательное объяснение пробелов и рекомендации
по улучшению резюме.

#### `resume_quality_evaluator.py`

Оценивает резюме само по себе, без конкретной вакансии: опыт, проекты, шапку,
раздел «О себе», навыки и образование.

#### `vacancy_search.py` — нерабочий экспериментальный модуль

Берёт `target_position` из структурированного JSON резюме и запрашивает
вакансии через API hh.ru. На момент написания приёмки модуль не работает,
не входит в основной конвейер и оставлен как заготовка для последующей
диагностики и доработки. Для его запуска требуется интернет.

#### `synthetic_vacancy_pipeline.py`

Определяет профессию кандидата, ищет не менее 20 актуальных вакансий через
несколько публичных источников, нормализует и дедуплицирует их, вычисляет
семантическую схожесть и агрегирует три наиболее согласованные вакансии.
Частоты требований и ссылки на источники сохраняются в итоговом JSON. При тестах создание заняло больше часа.

#### `system_check.py` и `model_check.py`

`system_check` проверяет зависимости, форматы, Pydantic-схемы и переданные
готовые JSON без загрузки модели. `model_check` выполняет реальный контрольный
запрос к Ollama.

## Требования

- Windows, Linux или macOS;
- Python 3.10 или новее, рекомендуется Python 3.11+;
- установленный и запущенный Ollama;
- локально загруженная модель `gemma4:12b`;
- интернет для первоначальной загрузки зависимостей и модели;
- интернет для команды поиска вакансий.



## Запуск полного конвейера

```powershell
python -m resume_evaluator `
  C:\path\resume.docx `
  C:\path\vacancy.docx `
  output/demo
```

В `output/demo/` будут созданы:

```text
resume.json                    # Структурированное резюме
vacancy.json                   # Структурированная вакансия
score.json                     # Балл и расшифровка критериев
recommendation_short.json      # Краткая рекомендация
recommendation_detailed.json   # Подробная рекомендация
```

Полный конвейер делает четыре обращения к Ollama: извлечение резюме,
извлечение вакансии и две рекомендации. Скоринг выполняется локально без LLM.

## Запуск отдельных элементов

Все команды выполняются из корня проекта с активированным виртуальным
окружением.

### 1. Извлечение резюме

```powershell
python -m resume_evaluator.resume_extractor `
  C:\path\resume.docx `
  output/resume.json
```

Вход: TXT, DOCX или PDF. Выход: JSON схемы `ResumeData`.

### 2. Извлечение вакансии

```powershell
python -m resume_evaluator.vacancy_extractor `
  C:\path\vacancy.docx `
  output/vacancy.json
```

Вход: TXT, DOCX или PDF. Выход: JSON схемы `VacancyData`.

### 3. Расчёт соответствия

```powershell
python -m resume_evaluator.candidate_scorer `
  output/resume.json `
  output/vacancy.json `
  output/score.json
```

Этот этап не требует Ollama и интернета.

### 4. Краткая рекомендация

```powershell
python -m resume_evaluator.short_recommendation `
  output/resume.json `
  output/vacancy.json `
  output/score.json `
  output/recommendation_short.json
```

### 5. Подробная рекомендация

```powershell
python -m resume_evaluator.detailed_recommendation `
  output/resume.json `
  output/vacancy.json `
  output/score.json `
  output/recommendation_detailed.json
```

### 6. Независимая оценка качества резюме

```powershell
python -m resume_evaluator.resume_quality_evaluator `
  C:\path\resume.docx `
  output/resume_quality.json
```

Вакансия для этой команды не требуется.

### 7. Поиск вакансий по структурированному резюме — экспериментально

> Внимание: эта команда на момент минимальной приёмки не работает и приведена
> только для будущей диагностики интеграции с hh.ru.

```powershell
python -m resume_evaluator.vacancy_search `
  output/resume.json `
  output/vacancies_found.json `
  --area 3 `
  --limit 20
```

`--area` — идентификатор региона hh.ru, `--limit` — число результатов от 1 до
50. Без `--area` поиск выполняется по всем регионам.

### 8. Построение синтетической эталонной вакансии

```powershell
python -m resume_evaluator.synthetic_vacancy_pipeline `
  output/resume.json `
  output/synthetic_vacancy.json
```

Pipeline использует Jobicy, JobsCollider, Remote OK, Remotive, Arbeitnow и
hh.ru, требует минимум 20 уникальных актуальных вакансий и сохраняет результат,
совместимый с `VacancyData`.

## Проверка существующих JSON

```powershell
python -m resume_evaluator.system_check `
  --resume output/resume.json `
  --vacancy output/vacancy.json `
  --score output/score.json `
  --recommendation output/recommendation_short.json `
  --detailed-recommendation output/recommendation_detailed.json
```

Параметры `--resume` и `--vacancy` необходимо передавать вместе.

## Тесты

Локальные тесты используют стандартный модуль `unittest`; отдельная тестовая
зависимость не требуется. Каталог `tests/` исключён из публикуемого Git и может
быть доступен только в рабочей копии разработчика.

```powershell
python -m unittest discover -s tests -v
```

Тесты не должны обращаться к реальной модели или API hh.ru: внешние вызовы
заменены mock-объектами.

## Локальные примеры и данные

Каталоги `examples/`, `data/`, `docs/`, `tests/` и `output/` исключены из Git.
Они могут присутствовать в рабочей копии разработчика, но не поставляются с
исходным кодом проекта.

- `examples/resume.txt` и `examples/resume_python.txt` — текстовые резюме;
- `examples/vacancy.txt` и `examples/vacancy_event_communications.txt` —
  текстовые вакансии;
- `examples/resume_medical_anonymized.json` — обезличенное структурированное
  резюме;
- `examples/vacancy_medical_anesthesiologist.json` — структурированная
  медицинская вакансия;
- `data/reference/industries.json` — справочник отраслей;
- `data/reference/professions.json` — справочник профессий;
- `data/raw/` — локальные крупные или чувствительные выгрузки, не попадают в Git.

Не добавляйте в репозиторий реальные резюме с персональными данными и большие
выгрузки CV. Перед тестированием удаляйте идентификаторы, контакты, дату рождения,
возраст, пол и точную геолокацию.

## Типовые ошибки

### `ModuleNotFoundError`

Запускайте команды из корня проекта и убедитесь, что активировано окружение
`.venv` и выполнена установка `requirements.txt`.

### Ollama не отвечает

Проверьте, что Ollama запущен и модель доступна:

```powershell
ollama list
python -m resume_evaluator.model_check
```

### PDF не содержит текста

Сканированный PDF необходимо сначала обработать OCR и сохранить с текстовым
слоем.

### PowerShell запрещает активацию окружения

Можно не активировать окружение и вызывать интерпретатор напрямую:

```powershell
.\.venv\Scripts\python.exe -m resume_evaluator.system_check
```

## Критерий минимальной приёмки

Проект готов к минимальной приёмке, если выполняются системная проверка и
реальный конвейер на подготовленных проверяющим документах:

```powershell
python -m resume_evaluator.system_check
python -m resume_evaluator C:\path\resume.docx C:\path\vacancy.docx output/demo
```

Первая команда должна завершиться без ошибок. Вторая требует работающего Ollama
и модели `gemma4:12b` и должна создать пять JSON-файлов. Если локальный каталог
`tests/` доступен, дополнительно выполняется `python -m unittest discover -s
tests -v`.
