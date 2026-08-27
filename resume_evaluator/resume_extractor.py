"""Извлечение структурированного резюме из текста с помощью Ollama."""

import argparse
import time
from pathlib import Path

from ollama import chat
from pydantic import BaseModel, Field

from .document_reader import read_document


MODEL_NAME = "gemma4:12b"
MODEL_CONTEXT_SIZE = 16384
MAX_OUTPUT_TOKENS = 8192


class WorkExperience(BaseModel):
    company: str | None = Field(
        description="Работодатель, явно указанный в разделе опыта работы",
    )
    position: str | None = Field(
        description="Должность кандидата",
    )
    period: str | None = Field(
        description="Период работы в исходной формулировке без вычислений",
    )
    responsibilities: list[str] = Field(
        description="Только рабочие задачи и обязанности, но не технологии",
    )
    technologies: list[str] = Field(
        description=(
            "Технологии, явно связанные именно с этим местом работы. "
            "Не копируй сюда общие навыки кандидата из других разделов"
        ),
    )
    achievements: list[str] = Field(
        description=(
            "Результаты и достижения на этом месте работы. "
            "Особенно сохраняй измеримые результаты с числами и процентами"
        ),
    )


class Education(BaseModel):
    institution: str | None = Field(
        description=(
            "Только название учебного заведения. "
            "Значения «высшее», «среднее» и «среднее профессиональное» "
            "не являются названиями учреждений"
        ),
    )
    education_level: str | None = Field(
        description=(
            "Явно указанный уровень образования, например высшее, "
            "среднее профессиональное или среднее общее"
        ),
    )
    education_field: str | None = Field(
        description=(
            "Общая область образования, например техническое, "
            "медицинское, педагогическое или экономическое"
        ),
    )
    field_of_study: str | None = Field(
        description="Направление или специальность",
    )
    graduation_year: int | None = Field(
        description="Год окончания, только если он явно указан",
    )


class Project(BaseModel):
    name: str | None = Field(
        description="Название проекта, если оно явно указано",
    )
    description: str | None = Field(
        description="Краткое описание выполненного проекта",
    )
    role: str | None = Field(
        description="Роль или личный вклад кандидата в проект",
    )
    technologies: list[str] = Field(
        description="Технологии, явно связанные с этим проектом",
    )
    results: list[str] = Field(
        description="Полученные результаты, особенно измеримые",
    )


class Language(BaseModel):
    language: str = Field(
        description="Название языка",
    )
    proficiency: str | None = Field(
        description="Указанный кандидатом уровень владения языком",
    )


class ResumeData(BaseModel):
    candidate_name: str | None = Field(
        description="Имя кандидата",
    )
    target_position: str | None = Field(
        description="Желаемая или целевая должность",
    )
    professional_summary: str | None = Field(
        description=(
            "Профессиональное описание кандидата из раздела "
            "«О себе», summary или аналогичного раздела"
        ),
    )
    contact_channels: list[str] = Field(
        description=(
            "Только виды указанных способов связи: email, телефон, Telegram "
            "и другие. Не копируй сами адреса и номера"
        ),
    )
    skills: list[str] = Field(
        description="Профессиональные и технические навыки кандидата",
    )
    soft_skills: list[str] = Field(
        description=(
            "Явно названные или прямо подтверждённые действиями кандидата "
            "личные и коммуникативные навыки без домысливания"
        ),
    )
    work_experience: list[WorkExperience] = Field(
        description="Все явно указанные места работы по отдельности",
    )
    education: list[Education] = Field(
        description="Все явно указанные записи об образовании",
    )
    projects: list[Project] = Field(
        description="Отдельно описанные проекты кандидата",
    )
    courses_and_certifications: list[str] = Field(
        description="Курсы, сертификаты и дополнительное обучение",
    )
    languages: list[Language] = Field(
        description="Все явно указанные языки и уровни владения",
    )
    achievements: list[str] = Field(
        description=(
            "Ключевые достижения кандидата из всего резюме, включая "
            "измеримые результаты работы"
        ),
    )
    portfolio_links: list[str] = Field(
        description=(
            "Ссылки на GitHub, портфолио, публикации и профессиональные проекты"
        ),
    )


SYSTEM_PROMPT = """
Ты извлекаешь структурированные данные из резюме.

Строго соблюдай границы разделов резюме.

Правила:
1. Используй только сведения, явно присутствующие в тексте.
2. Не додумывай факты и не вычисляй продолжительность работы.
3. Образование никогда не добавляй в опыт работы.
4. Учебное заведение не является работодателем без прямого указания.
5. В responsibilities добавляй только выполненные рабочие задачи.
6. В technologies конкретного места работы добавляй технологию только тогда,
   когда текст явно связывает её с этой работой.
7. Общие навыки вне описания места работы добавляй только в skills.
8. Не переноси общий навык в последнее упомянутое место работы.
9. «Уверенный пользователь Excel» вне блока работы означает общий навык Excel,
   но не технологию конкретного места работы.
10. Значения «высшее», «среднее профессиональное» и «среднее общее» являются
    уровнем образования, а не названием учебного заведения.
11. Если указан только уровень образования, заполни education_level,
    а institution оставь null.
12. Сохраняй названия организаций без изменения символов и кавычек.
13. Если сведений нет, используй null или пустой список.
14. Проекты не добавляй в опыт работы, если они не связаны с конкретным
    работодателем.
15. В achievements сохраняй результаты работы, а не обычные обязанности.
16. Измеримые результаты с числами, процентами и объёмами сохраняй без потери
    значений.
17. В soft_skills включай явно названные личные и коммуникативные навыки, а
    также прямо подтверждённые описанными действиями компетенции. Например,
    фактическое ведение переговоров подтверждает навык переговоров. Не
    додумывай качества, для которых в тексте нет прямого подтверждения.
18. В contact_channels записывай только наличие вида связи: email, телефон,
    Telegram и другие. Не сохраняй сами адреса и номера.
19. В portfolio_links включай только ссылки на профессиональные материалы,
    проекты, GitHub, публикации или портфолио.
20. Не считай социальную сеть профессиональным портфолио без явного основания.
21. Описание «О себе» сохраняй в professional_summary, не смешивая его
    с обязанностями или достижениями.
22. Если сведений нет, используй null или пустой список.
23. Обработай весь документ от первой до последней строки. Отдельно проверь
    заголовок резюме, «О себе», навыки, каждое место работы, образование,
    проекты, дополнительное обучение и языки.
24. Каждое явно указанное место работы добавляй отдельным элементом
    work_experience. Не возвращай пустой work_experience, если в тексте есть
    раздел опыта работы.
25. Желаемую должность из заголовка или блока цели записывай в
    target_position. Не путай её с должностями из опыта работы.
26. Если в тексте перечислены навыки или языки, соответствующие списки не
    должны быть пустыми.
27. Отдельно названную проектную активность добавляй в projects. Если за
    участие выдан сертификат, также добавляй его в courses_and_certifications.
28. В корневом achievements собери ключевые подтверждённые достижения из
    всего резюме. Достижения конкретной работы одновременно сохраняй в
    achievements соответствующего элемента work_experience.
29. Перед ответом проверь полноту каждого поля схемы по исходному тексту.
30. Возвращай только данные по заданной JSON-схеме.

Пример:
Текст: «Образование: среднее профессиональное».
Правильно:
institution = null
education_level = "Среднее профессиональное"

Если указано «высшее техническое образование», заполни:
education_level = "Высшее"
education_field = "Техническое"

Конкретную специальность или направление, например «Информационные системы»,
записывай отдельно в field_of_study.

"""


def read_resume(file_path: Path) -> str:
    return read_document(file_path)


def _is_candidate_name_placeholder(candidate_name: str) -> bool:
    normalized_name = "".join(
        character
        for character in candidate_name.casefold()
        if character.isalnum()
    )
    return normalized_name in {"фио", "фамилияимяотчество"}


def extract_resume(resume_text: str) -> ResumeData:
    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Извлеки данные из резюме:\n\n{resume_text}",
            },
        ],
        format=ResumeData.model_json_schema(),
        think=False,
        options={
            "temperature": 0,
            "num_ctx": MODEL_CONTEXT_SIZE,
            "num_predict": MAX_OUTPUT_TOKENS,
        },
    )

    resume = ResumeData.model_validate_json(response.message.content)

    if resume.candidate_name and _is_candidate_name_placeholder(
        resume.candidate_name
    ):
        resume.candidate_name = None

    return resume


def save_result(resume: ResumeData, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_text = resume.model_dump_json(indent=2)
    output_path.write_text(json_text, encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Извлечение структурированных данных из резюме"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Путь к резюме в формате TXT, DOCX или PDF",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Путь для сохранения результата в JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    resume_text = read_resume(args.input)

    print(f"Обрабатывается файл: {args.input}")

    start_time = time.perf_counter()
    resume = extract_resume(resume_text)
    elapsed_time = time.perf_counter() - start_time

    save_result(resume, args.output)

    print(f"Результат сохранён: {args.output}")
    print(f"Время обработки: {elapsed_time:.1f} сек.")


if __name__ == "__main__":
    main()
