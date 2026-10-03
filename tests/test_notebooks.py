"""Every notebook builds, executes top-to-bottom (fake model backend), and follows the step pattern."""
import pathlib
import re

import nbformat
import pytest

from scripts.build_notebooks import MODULES
from scripts.run_notebooks import run

ROOT = pathlib.Path(__file__).resolve().parent.parent
NBS = sorted((ROOT / "notebooks").glob("*.ipynb"))

STORY = NBS


def test_all_notebooks_built():
    assert len(NBS) == len(MODULES)


@pytest.mark.parametrize("path", NBS, ids=lambda p: p.name)
def test_notebook_executes(path):
    run(path, live=False)


@pytest.mark.parametrize("path", NBS, ids=lambda p: p.name)
def test_chapter_has_setup_cell(path):
    setup = nbformat.read(path, as_version=4).cells[1].source
    assert "git\", \"clone" in setup and "baluragala/reinfrocement-learning-for-genai-models" in setup


def test_no_secrets_anywhere_in_repo():
    for p in ROOT.rglob("*"):
        if p.is_file() and not {".venv", ".git"} & set(p.parts) and p.suffix in {".py", ".md", ".ipynb", ".txt", ".json"}:
            text = p.read_text(errors="ignore")
            assert not re.search(r"sk-(proj-)?[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{30,}", text), p


def _tag(c):
    return (c.get("metadata", {}).get("tags") or [None])[0]


# ------------------------------------------------------------------ story-style chapters: easy to read, easy to run

import html as _html

FORMULA = re.compile(r"[πβγε∝Σ∇∑]|log σ|KL\(|argmax")


def _words(md_src):
    text = re.sub(r"<details>.*?</details>", "", md_src, flags=re.S)
    text = re.sub(r"\|[-| ]+\|", "", text)
    return len(re.findall(r"[A-Za-z']+", _html.unescape(text)))


@pytest.mark.parametrize("path", STORY, ids=lambda p: p.name)
def test_story_chapter_is_short_and_plain(path):
    nb = nbformat.read(path, as_version=4)
    cells = nb.cells[2:]
    assert len(cells) <= 60, f"{len(cells)} cells: keep a chapter short"
    for i, c in enumerate(cells):
        tag = _tag(c)
        where = f"{path.name} cell {i + 2}: {c.source[:50]!r}"
        if c.cell_type == "markdown" and tag != "curious":
            assert _words(c.source) <= 110, f"wall of text ({_words(c.source)} words) · {where}"
            assert not FORMULA.search(c.source), f"formula outside a 'for the curious' fold · {where}"
        if c.cell_type == "code":
            assert tag in {"play", "challenge", "solution"}, f"untagged code cell · {where}"
            if tag == "play":
                lines = [l for l in c.source.splitlines() if l.strip() and not l.strip().startswith("#")]
                assert len(lines) <= 12, f"code cell too long ({len(lines)} lines) · {where}"


@pytest.mark.parametrize("path", STORY, ids=lambda p: p.name)
def test_story_chapter_has_the_engagement_beats(path):
    tags = [_tag(c) for c in nbformat.read(path, as_version=4).cells]
    assert tags.count("poll") >= 3, "at least three quick polls"
    assert tags.count("takeaway") >= 5, "a takeaway after the key beats"
    assert "challenge" in tags and "solution" in tags, "a hands-on challenge with a solution"
    assert all(not c.get("outputs") for c in nbformat.read(path, as_version=4).cells if c.cell_type == "code")
