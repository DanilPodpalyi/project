"""Ручная проверка доступности настроенной модели Ollama."""

import time

from ollama import chat


MODEL_NAME = "gemma4:12b"


def main() -> None:
    """Отправить модели короткий контрольный запрос и вывести ответ."""
    start_time = time.perf_counter()

    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты помощник для анализа резюме. "
                    "Отвечай по-русски, кратко и не придумывай факты."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Кандидат два года работал Python-разработчиком. "
                    "Использовал Python, PostgreSQL и Git. "
                    "Кратко перечисли его явно указанные навыки."
                ),
            },
        ],
        think=False,
        options={"temperature": 0},
    )

    elapsed_time = time.perf_counter() - start_time

    print("\nОтвет модели:")
    print(response.message.content)
    print(f"\nВремя обработки: {elapsed_time:.1f} сек.")


if __name__ == "__main__":
    main()
