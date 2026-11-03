"""Relative links in the user-facing docs must resolve inside the repo."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC_FILES = [
    ROOT / "README.md",
    ROOT / "WINDOWS.md",
    ROOT / "CONTRIBUTING.md",
    *sorted((ROOT / "docs").glob("*.md")),
]
LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)


def _slugs(page: Path) -> set[str]:
    slugs: set[str] = set()
    for title in HEADING.findall(page.read_text(encoding="utf-8")):
        slug = title.strip().lower()
        slug = re.sub(r"[^\w\s-]", "", slug, flags=re.UNICODE)
        slug = re.sub(r"\s+", "-", slug.strip())
        slugs.add(slug)
    return slugs


def test_doc_links_resolve():
    assert (ROOT / "docs").is_dir()
    missing: list[str] = []
    for page in DOC_FILES:
        assert page.is_file(), page
        for raw in LINK.findall(page.read_text(encoding="utf-8")):
            target = raw.strip()
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            path_part, _, fragment = target.partition("#")
            destination = page if path_part == "" else (page.parent / path_part).resolve()
            if path_part and not destination.is_file():
                missing.append(f"{page.relative_to(ROOT)} -> {target}")
                continue
            if fragment and fragment not in _slugs(destination):
                missing.append(f"{page.relative_to(ROOT)} -> {target} (no heading)")
    assert not missing, "unresolved doc links:\n" + "\n".join(missing)


def test_readme_covers_the_homepage_topics():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for snippet in (
        "python -m ai_insar",
        "ai-insar check",
        "pip install -e",
        "pytest",
        "complex64",
        "later",
        "WINDOWS.md",
        "docs/cli.md",
        "docs/pipeline.md",
        "docs/data-format.md",
        "docs/limitations.md",
        "LICENSE",
        "not** a replacement",
    ):
        assert snippet in text
