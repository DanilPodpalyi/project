# Конфигурация

Скопируйте `config.example.toml` в `config.toml`. Запуск:
`python -m resume_evaluator --config config.toml analyze resume.docx output/result`.
В файле разрешён только раздел `[resume_evaluator]`; неизвестные ключи отвергаются.
Численные значения должны быть положительными и конечными. Приоритет окружения выше TOML.

| Ключ TOML | Переменная окружения | По умолчанию |
|---|---|---|
| model | RESUME_EVALUATOR_MODEL | gemma4:12b |
| embedding_model | OLLAMA_EMBEDDING_MODEL | bge-m3:latest |
| ollama_host | OLLAMA_HOST | http://127.0.0.1:11434 |
| context_size | RESUME_EVALUATOR_CONTEXT_SIZE | 16384 |
| ollama_timeout_seconds | RESUME_EVALUATOR_OLLAMA_TIMEOUT | 600 |
| max_run_seconds | RESUME_EVALUATOR_MAX_RUN_SECONDS | 3600 |
| http_timeout_seconds | RESUME_EVALUATOR_HTTP_TIMEOUT | 20 |
| search_max_pages | RESUME_EVALUATOR_SEARCH_MAX_PAGES | 3 |
| detail_limit | RESUME_EVALUATOR_DETAIL_LIMIT | 80 |
| response_cache_hours | RESUME_EVALUATOR_CACHE_HOURS | 24 |

Путь к конфигурации можно передать через `RESUME_EVALUATOR_CONFIG`.
`HH_ACCESS_TOKEN` и `HH_USER_AGENT` задаются отдельно через окружение.
Пароль внутри URL Ollama запрещён. `.env` автоматически не загружается.
`search_max_pages` допускает до 20 страниц, `detail_limit` — до 200 карточек;
увеличение лимитов повышает время и число запросов, но не гарантирует релевантную выборку.

Тайм-аут Ollama ограничивает ожидание HTTP, бюджет запуска проверяется между этапами
и перед запросами. Проверки CPU и отдельный сетевой обмен могут завершиться позже
границы бюджета; это не принудительный таймер ОС. Истечение бюджета фиксируется
как ошибка запуска. Обнаруженные ошибки отдельных площадок могут дать частичный отчёт.

Настройки моделей фиксируются при импорте модулей. Для применения новых настроек
запускайте новый процесс CLI. `run.json` содержит публичные настройки; токен HH
в него не записывается. Входные пути и выходные отчёты могут содержать сведения
кандидата; они остаются в локальной папке результатов, которая исключена из релиза.
