"""Совместимый импорт и запуск прежнего модуля; реализация в research.market_profile_builder."""
import sys
from importlib import import_module

implementation = import_module("resume_evaluator.research.market_profile_builder")
if __name__ == "__main__":
    implementation.main()
else:
    # Alias сохраняет globals функций: старые импорты и monkeypatch работают.
    sys.modules[__name__] = implementation
