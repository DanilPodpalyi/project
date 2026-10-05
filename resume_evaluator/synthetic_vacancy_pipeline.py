"""Совместимый импорт и запуск прежнего модуля; реализация в legacy.synthetic_vacancy_pipeline."""
import sys
from importlib import import_module

implementation = import_module("resume_evaluator.legacy.synthetic_vacancy_pipeline")
if __name__ == "__main__":
    implementation.main()
else:
    # Alias сохраняет globals функций: старые импорты и monkeypatch работают.
    sys.modules[__name__] = implementation
