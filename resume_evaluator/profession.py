"""Общая структура профессии для резюме, вакансий и поиска."""

from typing import Any

from pydantic import BaseModel, Field, model_validator


class ProfessionProfile(BaseModel):
    """Профессия, разделённая на функцию, специализацию и предметную область."""

    role: str | None = Field(
        description="Базовая профессиональная роль, например Journalist",
    )
    specialization: str | None = Field(
        description="Конкретная специализация, например Correspondent",
    )
    domain: str | None = Field(
        description="Предметная область, например Media",
    )
    seniority: str | None = Field(
        description="Уровень позиции, например Junior, Middle или Senior",
    )
    role_family: str | None = Field(
        description="Устойчивая профессиональная семья, например Journalism",
    )
    focus: str | None = Field(
        description="Основной характер работы, например News Reporting",
    )
    core_skills: list[str] = Field(
        description="Ключевые навыки, подтверждающие выбранную профессию",
    )

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_profile(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            data.setdefault("role_family", None)
            data.setdefault("focus", None)
        return data

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


class TargetProfile(ProfessionProfile):
    """Целевой рынок профессии без сведений о навыках конкретного кандидата."""

    aliases: list[str] = Field(
        default_factory=list,
        description="Общеупотребимые эквивалентные названия профессии",
    )
    adjacent_specializations: list[str] = Field(
        default_factory=list,
        description="Допустимые смежные названия, требующие проверки содержания",
    )
    excluded_specializations: list[str] = Field(
        default_factory=list,
        description="Похожие по словам, но профессионально отличающиеся направления",
    )

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_target_profile(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            data.setdefault("aliases", [])
            data.setdefault("adjacent_specializations", [])
            data.setdefault("excluded_specializations", [])
        return data


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
            role_family=None,
            focus=None,
            core_skills=list(core_skills or []),
        )
    return ProfessionProfile(
        role=None,
        specialization=None,
        domain=None,
        seniority=None,
        role_family=None,
        focus=None,
        core_skills=list(core_skills or []),
    )


def target_profile_from_value(value: Any) -> TargetProfile:
    """Преобразует профессию в расширенный профиль построения рынка."""
    if isinstance(value, TargetProfile):
        return value
    if isinstance(value, dict):
        return TargetProfile.model_validate(value)
    profile = profession_from_value(value)
    return TargetProfile(
        **profile.model_dump(),
        aliases=[],
        adjacent_specializations=[],
        excluded_specializations=[],
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
