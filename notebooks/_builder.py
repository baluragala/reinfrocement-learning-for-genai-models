"""A small cell DSL, plus the setup cell that fetches `rllab` from GitHub.

Each chapter is authored as a Python module (`nbXX.py`) exposing NOTEBOOK
(the filename), TITLE, MINUTES, HOOK, MISSION and CELLS. `scripts/build_notebooks.py`
turns them into .ipynb files.

The setup cell makes `src/rllab` importable. Inside a checkout of this repo
(a maintainer's machine, the test runner) it uses that checkout's `src/`, so
local edits take effect without a push. Anywhere else (a fresh Colab) it
shallow-clones the repo from GitHub, or pulls if the clone already exists.
The clone brings the trained adapters and the cached model outputs with it.
"""
from __future__ import annotations

import pathlib

import nbformat

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "rllab"
GITHUB_SLUG = "baluragala/reinfrocement-learning-for-genai-models"
GITHUB_BRANCH = "main"
SESSION = "C9 · W4 · S2: Reinforcement Learning for GenAI Models"


def colab_badge(notebook: str) -> str:
    url = f"https://colab.research.google.com/github/{GITHUB_SLUG}/blob/{GITHUB_BRANCH}/notebooks/{notebook}"
    return f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})"


def md(text: str):
    return nbformat.v4.new_markdown_cell(text.strip("\n"))


def code(src: str):
    return nbformat.v4.new_code_cell(src.strip("\n"))


def form(title: str, src: str):
    cell = nbformat.v4.new_code_cell(f'#@title {title} {{ display-mode: "form" }}\n{src.strip()}')
    cell["metadata"]["cellView"] = "form"
    cell["metadata"]["jupyter"] = {"source_hidden": True}
    return cell


def solution(src: str):
    cell = form("✅ Solution (click to reveal)", src)
    cell["metadata"]["tags"] = ["solution"]
    return cell


def _tagged(cell, tag):
    cell["metadata"]["tags"] = [tag]
    return cell


def setup_cell():
    src = f'''
# Makes the lab runtime (`rllab`) importable, and installs transformers + peft if they're missing.
# In Colab this clones {GITHUB_SLUG} (branch {GITHUB_BRANCH}); inside a local checkout it uses that checkout.
import sys, subprocess, pathlib, importlib.util
REPO_URL = "https://github.com/{GITHUB_SLUG}.git"
BRANCH = "{GITHUB_BRANCH}"
_need = [p for p in ("torch", "transformers", "peft", "accelerate") if importlib.util.find_spec(p) is None]
if _need:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *_need, "pandas", "matplotlib"], check=True)

_here = pathlib.Path.cwd().resolve()
_src = next((p / "src" for p in [_here, *_here.parents] if (p / "src" / "rllab" / "__init__.py").exists()), None)
if _src is None:
    _base = pathlib.Path("/content") if pathlib.Path("/content").is_dir() else _here
    _repo = _base / "{GITHUB_SLUG.split('/')[1]}"
    if not (_repo / ".git").exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", BRANCH, REPO_URL, str(_repo)], check=True)
    else:
        subprocess.run(["git", "-C", str(_repo), "pull", "-q", "--ff-only"], check=False)
    _src = _repo / "src"
sys.path.insert(0, str(_src))
for _m in [m for m in sys.modules if m == "rllab" or m.startswith("rllab.")]:
    del sys.modules[_m]
import rllab as rl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.max_colwidth", 120); pd.set_option("display.width", 200)
print(f"rllab {{rl.__version__}} ready from {{_src}} · base model: {{rl.config.BASE_MODEL}}")
'''
    return form("⚙️ Setup: run this first", src)


# ================================================================== story-style notebooks (v2)
# Short beats instead of long steps:   🎬 story → ▶️ play (short code) → 💬 takeaway
# tests/test_notebooks.py keeps these notebooks short, plain-worded and formula-free.

def story(text: str):
    return _tagged(md(text), "story")


def play(src: str):
    return _tagged(code(src), "play")


def takeaway(text: str):
    return _tagged(md(f"> 💬 **{text.strip()}**"), "takeaway")


def poll(question: str, options: list[str]):
    opts = "\n".join(f"- **{chr(65 + i)}.** {o}" for i, o in enumerate(options))
    return _tagged(md(f"### ✋ Quick poll\n{question.strip()}\n\n{opts}\n\n*Pick one before you run the next cell!*"), "poll")


def challenge(title: str, task: str, src: str):
    return [_tagged(md(f"### 🏆 Challenge: {title}\n{task.strip()}"), "challenge-intro"), _tagged(code(src), "challenge")]


def curious(title: str, text: str):
    """Optional depth, folded away. The only place formulas are allowed."""
    return _tagged(md(f"<details><summary>🤓 <b>For the curious:</b> {title}</summary>\n\n{text.strip()}\n\n</details>"), "curious")


def chapter_setup():
    cell = setup_cell()
    cell["source"] = cell["source"].replace(
        'print(f"rllab {rl.__version__} ready from {_src} · base model: {rl.config.BASE_MODEL}")',
        'from rllab import ui, pip as P\nimport warnings; warnings.filterwarnings("ignore")\n'
        'print("✅ Pip\'s lab is ready. Run the cells from top to bottom; each one takes a few seconds.")')
    cell["source"] = cell["source"].replace('"pandas", "matplotlib"]', '"pandas", "matplotlib", "ipywidgets"]')
    return cell


def build_story(module) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["colab"] = {"provenance": [], "toc_visible": True}
    nb.metadata["rllab_style"] = "story"
    mission = "\n".join(f"- {m}" for m in module.MISSION)
    head = md(f"{colab_badge(module.NOTEBOOK)}\n\n# {module.TITLE}\n\n"
              f"**{SESSION}** · ⏱️ {module.MINUTES} min\n\n{module.HOOK}\n\n### 🎯 Your mission\n{mission}")
    nb.cells = [head, chapter_setup()] + list(module.CELLS)
    return nb
