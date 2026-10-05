"""Морфология названий должностей; навыки кандидата здесь не используются."""

import re
from functools import lru_cache


@lru_cache(maxsize=1)
def _analyzer():
    try:
        from pymorphy3 import MorphAnalyzer
    except ImportError as error:
        raise RuntimeError(
            "Для поиска по России установите зависимости: pip install -r requirements.txt"
        ) from error
    return MorphAnalyzer()


_TITLE_MODIFIERS = {
    "junior", "middle", "senior", "lead", "staff", "principal", "head",
    "младший", "старший", "ведущий", "главный", "джуниор", "мидл", "миддл",
    "сеньор", "удаленный", "удаленно", "remote",
}
_STOP_WORDS = {"по", "в", "на", "и", "для", "of", "the"}

# Явные эквивалентные названия профессий, без расширения до смежных ролей.
# Двуязычное резюме может быть нормализовано моделью на английский язык.
_TITLE_EQUIVALENTS = (
    ("Графический дизайнер", "Graphic Designer"),
    ("Инженер машинного обучения", "Machine Learning Engineer", "ML Engineer",
     "Инженер по машинному обучению", "ML-инженер"),
)


@lru_cache(maxsize=8192)
def _lemma(word: str) -> str:
    if re.fullmatch(r"[а-яё]+", word):
        return _analyzer().parse(word)[0].normal_form.replace("ё", "е")
    return word


@lru_cache(maxsize=8192)
def _word_forms(word: str) -> frozenset[str]:
    if re.fullmatch(r"[а-яё]+", word):
        # «аналитика» может быть родительным падежом слова «аналитик» или
        # самостоятельным существительным. Не теряем второй вероятный разбор.
        return frozenset(
            parsed.normal_form.replace("ё", "е")
            for parsed in _analyzer().parse(word) if parsed.score >= 0.1
        )
    return frozenset({word})


def _literal_title_words(title: str) -> list[frozenset[str]]:
    return [forms for word in re.findall(r"[\w+#]+", title.casefold())
            if (forms := _word_forms(word)) and not forms & (_STOP_WORDS | _TITLE_MODIFIERS)]


@lru_cache(maxsize=1)
def _equivalent_patterns():
    patterns = [(tuple(_literal_title_words(alias)), tuple(_literal_title_words(group[0])))
                for group in _TITLE_EQUIVALENTS for alias in group]
    return sorted(patterns, key=lambda pair: -len(pair[0]))


def _title_words(title: str) -> list[frozenset[str]]:
    # Заменяем только целые названия профессий. Остальные слова (например,
    # Computer Vision или Data Engineer в составном названии) сохраняются.
    words = _literal_title_words(title)
    result = []
    index = 0
    while index < len(words):
        for alias, canonical in _equivalent_patterns():
            window = words[index:index + len(alias)]
            if len(window) == len(alias) and all(a & b for a, b in zip(alias, window)):
                result.extend(canonical)
                index += len(alias)
                break
        else:
            result.append(words[index])
            index += 1
    return result


def title_tokens(title: str) -> set[str]:
    tokens = {
        _lemma(word) for word in re.findall(r"[\w+#]+", title.casefold())
    }
    return tokens - _STOP_WORDS - _TITLE_MODIFIERS


def morphological_title_similarity(target: str, title: str) -> float:
    """Доля слов цели, найденных в названии в той же словарной форме.

    Например, «аналитик информационной безопасности» и «аналитика
    информационной безопасности» совпадают. «Аналитик данных» не совпадает
    с ними только из-за общего слова «аналитик».
    """
    expected = _title_words(target)
    actual = _title_words(title)
    if not expected or not actual:
        return 0.0
    return sum(any(forms & candidate for candidate in actual)
               for forms in expected) / len(expected)


def known_title_equivalents(title: str) -> list[str]:
    """Только проверенные эквиваленты всей должности, включая специализацию."""
    equivalents = []
    words = _title_words(title)
    for group in _TITLE_EQUIVALENTS:
        canonical = _title_words(group[0])
        if len(words) == len(canonical) and all(a & b for a, b in zip(words, canonical)):
            equivalents.extend(group)
    return equivalents


def title_search_queries(title: str) -> list[str]:
    """Исходное название, эквивалентные профессии и леммы, без навыков."""
    original = " ".join(title.split())
    lemmas = " ".join(
        _lemma(word) for word in re.findall(r"[\w+#]+", original.casefold())
        if _lemma(word) not in _STOP_WORDS | _TITLE_MODIFIERS
    )
    queries: dict[str, str] = {}
    equivalents = known_title_equivalents(original)
    for value in (original, *equivalents, lemmas):
        if value:
            queries.setdefault(value.casefold(), value)
    return list(queries.values())
