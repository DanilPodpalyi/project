"""Тесты интернет-поиска вакансий по целевой позиции резюме."""

import json
from io import BytesIO
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

from search_vacancies import (
    DEFAULT_USER_AGENT,
    VacancySearchError,
    build_search_url,
    fetch_search_data,
    get_search_query,
    search_vacancies,
)


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


class VacancySearchTests(unittest.TestCase):
    def test_uses_target_position_as_query(self) -> None:
        self.assertEqual(
            get_search_query(
                {"target_position": "  Python   разработчик  "},
            ),
            "Python разработчик",
        )

    def test_rejects_resume_without_target_position(self) -> None:
        with self.assertRaisesRegex(VacancySearchError, "target_position"):
            get_search_query({"target_position": None})

    def test_builds_encoded_search_url_with_area_and_limit(self) -> None:
        url = build_search_url(
            "Python разработчик",
            area="3",
            limit=7,
        )
        query = parse_qs(urlparse(url).query)

        self.assertEqual(query["text"], ["Python разработчик"])
        self.assertEqual(query["area"], ["3"])
        self.assertEqual(query["per_page"], ["7"])
        self.assertEqual(query["order_by"], ["publication_time"])

    def test_rejects_limit_above_api_maximum(self) -> None:
        with self.assertRaisesRegex(ValueError, "от 1 до 50"):
            build_search_url("Python", limit=51)

    @patch("search_vacancies.urlopen")
    def test_sends_required_user_agent(
        self,
        urlopen_mock: MagicMock,
    ) -> None:
        urlopen_mock.return_value = FakeResponse(
            {"found": 0, "items": []},
        )

        data = fetch_search_data("https://api.hh.ru/vacancies")

        self.assertEqual(data["found"], 0)
        request = urlopen_mock.call_args.args[0]
        self.assertEqual(request.get_header("User-agent"), DEFAULT_USER_AGENT)

    @patch("search_vacancies.urlopen")
    def test_reports_structured_api_error(
        self,
        urlopen_mock: MagicMock,
    ) -> None:
        error_body = BytesIO(
            json.dumps(
                {"errors": [{"type": "captcha_required"}]},
            ).encode("utf-8"),
        )
        urlopen_mock.side_effect = HTTPError(
            "https://api.hh.ru/vacancies",
            403,
            "Forbidden",
            {},
            error_body,
        )

        with self.assertRaisesRegex(
            VacancySearchError,
            "HTTP 403: captcha_required",
        ):
            fetch_search_data("https://api.hh.ru/vacancies")

    @patch("search_vacancies.fetch_search_data")
    def test_maps_api_response_to_compact_result(
        self,
        fetch_mock: MagicMock,
    ) -> None:
        fetch_mock.return_value = {
            "found": 125,
            "items": [
                {
                    "id": "123",
                    "name": "Python-разработчик",
                    "employer": {"name": "Тестовая компания"},
                    "area": {"name": "Екатеринбург"},
                    "salary": {
                        "from": 150000,
                        "to": 200000,
                        "currency": "RUR",
                        "gross": False,
                    },
                    "experience": {"name": "От 1 года до 3 лет"},
                    "employment": {"name": "Полная занятость"},
                    "schedule": {"name": "Удаленная работа"},
                    "snippet": {
                        "requirement": "Опыт с <highlighttext>Python</highlighttext>",
                        "responsibility": "Разработка API",
                    },
                    "published_at": "2026-08-24T10:00:00+0500",
                    "alternate_url": "https://hh.ru/vacancy/123",
                }
            ],
        }

        result = search_vacancies(
            {"target_position": "Python-разработчик"},
            area="3",
            limit=10,
        )

        self.assertEqual(result.query, "Python-разработчик")
        self.assertEqual(result.found, 125)
        self.assertEqual(result.returned, 1)
        self.assertEqual(result.vacancies[0].employer, "Тестовая компания")
        self.assertEqual(
            result.vacancies[0].requirement,
            "Опыт с Python",
        )
        self.assertEqual(result.vacancies[0].salary.from_amount, 150000)


if __name__ == "__main__":
    unittest.main()
