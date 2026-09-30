"""A small cell DSL, plus the setup cell that fetches `rllab` from GitHub.

Each notebook is authored as a Python module (`nbXX.py`) exposing NOTEBOOK
(the filename), TITLE, MINUTES, CONTEXT and CELLS. `scripts/build_notebooks.py`
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


# ---------------------------------------------------------------- the step pattern
# Every step in every notebook has the same shape, and tests/test_notebooks.py enforces it:
#
#   step(...)      md   : "### title", why the step exists, 📥 where every input comes from
#   run(...)       code : the step's code (tagged "step")
#   reading(...)   md   : 🔍 how to read the output, what to expect and why, <details> for depth
#   explain(...)   code : optional, prints reasoning computed from the learner's own run (tagged "explain")
#   so_what(...)   md   : ➡️ one line connecting to the next step
#
# predict(...) goes BEFORE a step whose outcome is worth guessing, and exercise() marks a 🧪 TODO cell.

def _tagged(cell, tag):
    cell["metadata"]["tags"] = [tag]
    return cell


def step(title: str, why: str, inputs: list[tuple[str, str]] | None = None):
    """inputs: [(name, "what it is — where it comes from"), ...]"""
    text = f"### {title}\n\n**Why this step:** {why.strip()}"
    if inputs:
        text += "\n\n**📥 Inputs**\n" + "\n".join(f"- `{n}`: {d}" for n, d in inputs)
    else:
        text += "\n\n**📥 Inputs:** none beyond what earlier cells defined."
    return _tagged(md(text), "step-intro")


def run(src: str):
    return _tagged(code(src), "step")


def reading(points: list[str], expect: str | None = None, deeper: str | None = None, deeper_title="Why it works this way"):
    text = "**🔍 Reading the output**\n" + "\n".join(f"- {p}" for p in points)
    if expect:
        text += f"\n\n**What to expect, and why:** {expect.strip()}"
    if deeper:
        text += f"\n\n<details><summary><b>▸ {deeper_title}</b> (click to expand)</summary>\n\n{deeper.strip()}\n\n</details>"
    return _tagged(md(text), "reading")


def explain(src: str):
    return _tagged(code(src), "explain")


def so_what(text: str):
    return _tagged(md(f"**➡️ So what:** {text.strip()}"), "so-what")


def predict(question: str, options: list[str] | None = None, hint: str | None = None):
    text = f"#### ✋ Predict first\n\n{question.strip()}"
    if options:
        text += "\n\n" + "\n".join(f"- **{chr(65 + i)}.** {o}" for i, o in enumerate(options))
    text += "\n\n*Write your answer down before running the next cell. The output tells you whether you were right.*"
    if hint:
        text += f"\n\n<details><summary>Hint</summary>\n\n{hint.strip()}\n\n</details>"
    return _tagged(md(text), "predict")


def exercise(title: str, task: str, src: str):
    """A 🧪 exercise: markdown brief + a TODO code cell (tagged "exercise"). Follow it with solution()."""
    return [_tagged(md(f"### 🧪 Your turn: {title}\n\n{task.strip()}"), "exercise-intro"),
            _tagged(code(src), "exercise")]


def glossary(rows: list[tuple[str, str]]):
    body = "\n".join(f"| `{n}` | {d} |" for n, d in rows)
    return _tagged(md("#### 📖 Names used in this notebook\n\n| Name | What it holds |\n|---|---|\n" + body), "glossary")


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


SCENARIO_MD = """## 🏕️ The scenario: Acme Outfitters wants its own support model

Last session you evaluated **Acme Outfitters'** support agent, which ran on a big API model. Acme now wants a
**small open model it can host itself**, one that drafts replies to customer messages ("how long do I have to return this?",
"my tent arrived broken!"), for cost and privacy reasons. The support leads have two things to offer:

- **~480 historical replies** written by support agents (mixed quality: mostly good, some rambling, some that ignore what
  the customer asked for, a few that should never have been sent)
- **~480 comparisons** where a support lead read two replies and picked the better one

Acme's rules haven't changed: returns within **30 days** of delivery, **final-sale** items and **gift cards** are never
refundable, standard shipping is **free over $75**, express costs **$15**, and **never** share another customer's data.

**The question for this session:** the engineering team can prompt a model, fine-tune it with **LoRA**, or tune it with
**reinforcement learning**. What does RL actually change, what doesn't it change, and is it worth it here?

### What's already built (you don't write this)
Everything below is packaged in `rllab`, which the setup cell loads. Training ran **once, before class**: the adapters and
their logs are in the repo, so nothing here trains a model live. This session teaches training as intuition, not code.

| Piece | What it is |
|---|---|
| `rl.config.BASE_MODEL` | **Qwen2.5-0.5B**, a small open model after pretraining only. It has knowledge but hasn't learned to follow instructions. |
| the **LoRA** variant | the base model + a LoRA adapter trained on the historical replies (supervised fine-tuning), in `artifacts/adapters/lora_sft/` |
| the **RL** variant | the LoRA model, then tuned with **DPO** on the support leads' comparisons, in `artifacts/adapters/rl_dpo/` |
| the **instruct** variant | **Qwen2.5-0.5B-Instruct**, the vendor's own SFT + RL version of the same base, prompted with Acme's rules (the "just use an RL-tuned model" option) |
| `rl.toy`, `rl.policy` | miniature RL worlds in NumPy (a recommender, a robot grid, a policy over whole replies) that show the ideas in seconds |
| `rl.evals`, `rl.checks` | rule-based checks, an LLM judge (Qwen2.5-1.5B-Instruct), a blinded human-review sheet |

All four variants get the **same prompts** with the **same greedy decoding**, so every difference you see comes from tuning.
Model replies are cached in `artifacts/cache/` from the instructor's pre-run, so they replay instantly and identically.
A prompt that isn't cached (one you write yourself) runs the model live.
"""


def context_md(module) -> str:
    c = module.CONTEXT
    learn = "\n".join(f"- {x}" for x in c["learn"])
    return (f"## 🎯 This notebook\n\n"
            f"**The problem.** {c['problem']}\n\n"
            f"**Where we're starting from.** {c['start']}\n\n"
            f"**By the end you'll be able to:**\n{learn}\n\n"
            f"**What you'll do here.** {c['do']}\n\n"
            f"**Given vs. you write.** {c['given']}")


RUNTIME_MD = """### 🖥️ Check the runtime
No API key is needed: every model is a small open model from the Hugging Face Hub.

* **Colab:** *Runtime → Change runtime type → T4 GPU* is recommended. On CPU the cached outputs still replay instantly,
  but prompts you write yourself will take a few seconds each.
* **Local:** `pip install -r requirements.txt`. Apple-silicon Macs use the GPU (MPS) automatically.

The first live call downloads the model it needs (about 1 GB for the 0.5B models and 3 GB for the judge)."""

RUNTIME_CELL = """print(rl.check_runtime())"""


def build(module) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["colab"] = {"provenance": [], "toc_visible": True}
    nb.metadata["accelerator"] = "GPU"
    head = md(f"{colab_badge(module.NOTEBOOK)}\n\n# {module.TITLE}\n\n**{SESSION}** · ⏱️ {module.MINUTES} min")
    nb.cells = ([head, md(SCENARIO_MD), md(context_md(module)), setup_cell(), md(RUNTIME_MD), code(RUNTIME_CELL)]
                + list(module.CELLS))
    return nb
