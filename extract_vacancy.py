"""Извлечение структурированных требований и условий вакансии."""

import argparse
import time
from pathlib import Path

from ollama import chat
from pydantic import BaseModel, Field

from document_reader import read_document


MODEL_NAME = "gemma4:12b"
MODEL_CONTEXT_SIZE = 16384
MAX_OUTPUT_TOKENS = 8192


class VacancyData(BaseModel):
    title: str | None = Field(
        description="Название должности или вакансии"
    )
    responsibilities: list[str] = Field(
        description="Рабочие задачи и обязанности сотрудника"
    )
    required_skills: list[str] = Field(
        description=(
            "Только отдельные обязательные навыки и технологии: "
            "например Python, PostgreSQL или REST API. "
            "Не добавляй сюда предложения, сроки и требования к опыту"
    )
)
    preferred_skills: list[str] = Field(
        description=(
            "Необязательные навыки из разделов «будет преимуществом», "
            "«желательно» или аналогичных"
        )
    )
    experience_requirement: str | None = Field(
        description=(
            "Требование к опыту в исходной формулировке, "
            "например «от 2 лет»"
        )
    )
    education_level: str | None = Field(
        description=(
            "Требуемый уровень образования, например высшее "
            "или среднее профессиональное"
        )
    )
    education_field: str | None = Field(
        description=(
            "Требуемое направление образования, например техническое"
        )
    )
    employment_type: str | None = Field(
        description="Тип занятости, например полная или частичная"
    )
    work_format: str | None = Field(
        description="Формат работы: удалённый, офисный или гибридный"
    )
    conditions: list[str] = Field(
        description="Прочие явно указанные условия работы"
    )


SYSTEM_PROMPT = """
Ты извлекаешь структурированные данные из вакансии.

Строго соблюдай границы разделов вакансии.

Правила:
1. Используй только сведения, явно присутствующие в тексте.
2. Не придумывай требования, обязанности или условия.
3. В required_skills добавляй только обязательные навыки и технологии.
4. В preferred_skills добавляй требования из разделов «будет преимуществом»,
   «желательно», «плюсом будет» и аналогичных.
5. Никогда не переноси preferred_skills в required_skills.
6. Обязанность не является обязательным навыком без прямого указания.
7. Если технология указана в обязательных требованиях, добавь её
   в required_skills.
8. Каждый элемент required_skills должен быть отдельным кратким навыком
   или названием технологии, а не целым предложением.
9. Если одно требование содержит и технологию, и продолжительность опыта,
   технологию добавь в required_skills, а полную формулировку требования
   к опыту — в experience_requirement.
10. Например, «Опыт разработки на Python от 2 лет» означает:
    required_skills = ["Python"]
    experience_requirement = "Опыт разработки на Python от 2 лет"
11. Требование к продолжительности опыта сохраняй в исходной формулировке.
12. Не вычисляй и не изменяй количество лет опыта.
13. «Высшее техническое образование» означает:
    education_level = "Высшее"
    education_field = "Техническое"
14. Удалённую работу записывай в work_format.
15. Полную занятость записывай в employment_type.
16. Не дублируй work_format и employment_type в conditions.
17. Если сведений нет, используй null или пустой список.
18. Обработай весь документ от первой до последней строки. Отдельно проверь
    название вакансии, обязанности, обязательные и желательные навыки, опыт,
    образование, занятость, формат работы и условия.
19. Не возвращай пустой required_skills, если в вакансии явно перечислены
    обязательные требования к навыкам.
20. Составное требование разделяй на отдельные краткие навыки, но не дроби
    устойчивое название одного навыка без необходимости. Например,
    «PR и контент» можно разделить, а «Power BI» нельзя.
21. Перед ответом проверь полноту каждого поля схемы по исходному тексту.
22. Возвращай только данные по заданной JSON-схеме.

Пример:
Текст: «Будет преимуществом: Docker».
Правильно:
preferred_skills = ["Docker"]
Docker не должен попадать в required_skills.
"""


def read_vacancy(file_path: Path) -> str:
    return read_document(file_path)


def extract_vacancy(vacancy_text: str) -> VacancyData:
    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Извлеки данные из вакансии:\n\n{vacancy_text}",
            },
        ],
        format=VacancyData.model_json_schema(),
        think=False,
        options={
            "temperature": 0,
            "num_ctx": MODEL_CONTEXT_SIZE,
            "num_predict": MAX_OUTPUT_TOKENS,
        },
    )

    return VacancyData.model_validate_json(response.message.content)


def save_result(vacancy: VacancyData, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_text = vacancy.model_dump_json(indent=2)
    output_path.write_text(json_text, encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Извлечение структурированных данных из вакансии"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Путь к вакансии в формате TXT, DOCX или PDF",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Путь для сохранения результата в JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    vacancy_text = read_vacancy(args.input)

    print(f"Обрабатывается файл: {args.input}")

    start_time = time.perf_counter()
    vacancy = extract_vacancy(vacancy_text)
    elapsed_time = time.perf_counter() - start_time

    save_result(vacancy, args.output)

    print(f"Результат сохранён: {args.output}")
    print(f"Время обработки: {elapsed_time:.1f} сек.")


if __name__ == "__main__":
    main()
