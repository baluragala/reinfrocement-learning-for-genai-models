"""Execute built notebooks headlessly.

  python scripts/run_notebooks.py            # FAKE model backend (no GPU, no downloads): plumbing check
  python scripts/run_notebooks.py --live     # real local models; fills artifacts/cache/ with every reply
                                             # and writes executed copies to reference_runs/ (the pre-run
                                             # outputs the instructor walks through in class)
"""
import pathlib
import sys

import nbformat
from nbclient import NotebookClient

ROOT = pathlib.Path(__file__).resolve().parent.parent
FAKE = f"""import sys; sys.path.insert(0, {str(ROOT / 'tests')!r})
from fake_llm import FakeBackend
rl.models.set_backend(FakeBackend())"""


def run(path: pathlib.Path, live: bool, timeout=3600):
    nb = nbformat.read(path, as_version=4)
    if not live:
        nb.cells.insert(2, nbformat.v4.new_code_cell(FAKE))   # right after the setup cell
    workdir = ROOT / ("reference_runs" if live else ".nbrun")
    workdir.mkdir(exist_ok=True)
    NotebookClient(nb, timeout=timeout, kernel_name="python3",
                   resources={"metadata": {"path": str(workdir)}}).execute()
    if live:
        _scrub(nb)
        nbformat.write(nb, workdir / path.name)
    return nb


def _scrub(nb):
    """Replace this machine's paths in outputs, so committed reference runs don't leak a home directory."""
    for c in nb.cells:
        for o in c.get("outputs", []):
            if "text" in o:
                o["text"] = o["text"].replace(str(ROOT), "<repo>")
            for k, v in o.get("data", {}).items():
                if isinstance(v, str):
                    o["data"][k] = v.replace(str(ROOT), "<repo>")


if __name__ == "__main__":
    live = "--live" in sys.argv
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    for p in sorted((ROOT / "notebooks").glob("*.ipynb")):
        if names and not any(n in p.name for n in names):
            continue
        run(p, live)
        print("✓", p.name, flush=True)
