"""Собрать поставку по явному списку, без локальных входов, кешей и токенов."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

VERSION = "0.2.0"
ROOT_FILES = {"README.md", "CHANGELOG.md", "pyproject.toml", "requirements.txt",
    "requirements.lock", "config.example.toml", ".env.example", ".gitignore"}


def release_files(root: Path) -> list[Path]:
    files = [root / name for name in ROOT_FILES]
    for folder, suffixes in (("resume_evaluator", {".py"}), ("tests", {".py", ".json", ".txt"}),
        ("scripts", {".py"}), ("docs", {".md"}), (".github/workflows", {".yml"}),
        ("examples/resumes", {".docx"}), ("examples/precomputed", {".json"})):
        files.extend(path for path in (root / folder).rglob("*")
            if path.is_file() and path.suffix in suffixes and "__pycache__" not in path.parts)
    files.append(root / "examples/README.md")
    for path in files:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"В поставке отсутствует обычный файл: {path}")
    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix())


def build_release(root: Path, output: Path) -> Path:
    root = root.resolve()
    files = release_files(root)
    prefix = f"resume-evaluator-{VERSION}"
    manifest = {"version": VERSION, "files": {path.relative_to(root).as_posix():
        {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
        for path in files}}
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"{prefix}-source.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as target:
        for path in files:
            target.write(path, f"{prefix}/{path.relative_to(root).as_posix()}")
        target.writestr(f"{prefix}/release_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dist"))
    args = parser.parse_args()
    print(build_release(Path(__file__).resolve().parents[1], args.output))


if __name__ == "__main__":
    main()
