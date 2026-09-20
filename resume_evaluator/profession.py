"""Общая структура профессии для резюме, вакансий и поиска."""

from typing import Any

from pydantic import BaseModel, Field


class ProfessionProfile(BaseModel):
    """Профессия, разделённая на функцию, специализацию и предметную область."""

    role: str | None = Field(
        description="Базовая профессиональная роль, например Analyst",
    )
    specialization: str | None = Field(
        description="Конкретная специализация, например Traffic Analyst",
    )
    domain: str | None = Field(
        description="Предметная область, например Digital Marketing / Web Analytics",
    )
    seniority: str | None = Field(
        description="Уровень позиции, например Junior, Middle или Senior",
    )
    core_skills: list[str] = Field(
        description="Ключевые навыки, подтверждающие выбранную профессию",
    )

    def label(self) -> str:
        """Возвращает короткое название для старых строковых интерфейсов."""
        return self.specialization or self.role or self.domain or ""

    def search_text(self) -> str:
        """Возвращает полный текст профессии для поиска и сопоставления."""
        values = [
            self.role,
            self.specialization,
            self.domain,
            self.seniority,
            *self.core_skills,
        ]
        return " ".join(value.strip() for value in values if value and value.strip())


def profession_from_value(
    value: Any,
    *,
    core_skills: list[str] | None = None,
) -> ProfessionProfile:
    """Принимает новую структуру и старую строку для обратной совместимости."""
    if isinstance(value, ProfessionProfile):
        return value
    if isinstance(value, dict):
        return ProfessionProfile.model_validate(value)
    if isinstance(value, str) and value.strip():
        cleaned = " ".join(value.split())
        return ProfessionProfile(
            role=cleaned,
            specialization=cleaned,
            domain=None,
            seniority=None,
            core_skills=list(core_skills or []),
        )
    return ProfessionProfile(
        role=None,
        specialization=None,
        domain=None,
        seniority=None,
        core_skills=list(core_skills or []),
    )


def profession_from_resume(resume: dict[str, Any]) -> ProfessionProfile:
    """Читает новый target_profession или legacy-поле target_position."""
    value = resume.get("target_profession")
    if value:
        return profession_from_value(value)
    return profession_from_value(
        resume.get("target_position"),
        core_skills=list(resume.get("skills") or []),
    )
