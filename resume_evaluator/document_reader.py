"""Единое извлечение текстового содержимого из TXT, DOCX и PDF."""

"""Чтение входных документов TXT, DOCX и PDF."""

from pathlib import Path
from typing import Callable
from zipfile import BadZipFile


SUPPORTED_EXTENSIONS = {".txt", ".docx", ".pdf"}


def _read_txt(file_path: Path) -> str:
    try:
        return file_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        raise SystemExit(
            f"Не удалось прочитать текстовый файл в кодировке UTF-8: "
            f"{file_path}"
        ) from error


def _read_docx(file_path: Path) -> str:
    try:
        from docx import Document
        from docx.opc.exceptions import PackageNotFoundError
        from docx.table import Table
    except ImportError as error:
        raise SystemExit(
            "Для чтения DOCX установите зависимости: "
            "pip install -r requirements.txt"
        ) from error

    try:
        document = Document(file_path)
    except (
        BadZipFile,
        KeyError,
        OSError,
        PackageNotFoundError,
        ValueError,
    ) as error:
        raise SystemExit(f"Некорректный DOCX-файл: {file_path}") from error

    blocks: list[str] = []
    for block in document.iter_inner_content():
        if isinstance(block, Table):
            for row in block.rows:
                cells = [cell.text.strip() for cell in row.cells]
                row_text = "\t".join(cell for cell in cells if cell)
                if row_text:
                    blocks.append(row_text)
        else:
            paragraph_text = block.text.strip()
            if paragraph_text:
                blocks.append(paragraph_text)

    return "\n".join(blocks)


def _read_pdf(file_path: Path) -> str:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as error:
        raise SystemExit(
            "Для чтения PDF установите зависимости: "
            "pip install -r requirements.txt"
        ) from error

    try:
        reader = PdfReader(file_path)
        pages = [page.extract_text() or "" for page in reader.pages]
    except (PdfReadError, OSError, ValueError) as error:
        raise SystemExit(f"Не удалось прочитать PDF-файл: {file_path}") from error

    return "\n\n".join(page.strip() for page in pages if page.strip())


def read_document(file_path: Path) -> str:
    if not file_path.exists():
        raise SystemExit(f"Файл не найден: {file_path}")

    if not file_path.is_file():
        raise SystemExit(f"Указанный путь не является файлом: {file_path}")

    readers: dict[str, Callable[[Path], str]] = {
        ".txt": _read_txt,
        ".docx": _read_docx,
        ".pdf": _read_pdf,
    }
    extension = file_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        if extension == ".doc":
            details = " Старый формат .doc необходимо сначала сохранить как .docx."
        else:
            details = ""
        raise SystemExit(
            f"Неподдерживаемый формат файла {extension or '(без расширения)'}. "
            f"Поддерживаются: {supported}.{details}"
        )

    text = readers[extension](file_path).strip()

    if not text:
        if extension == ".pdf":
            raise SystemExit(
                f"В PDF не найден текстовый слой: {file_path}. "
                "Для сканированного документа сначала выполните OCR."
            )
        raise SystemExit(f"В файле не найден текст: {file_path}")

    return text
