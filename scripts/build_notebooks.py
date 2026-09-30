"""Build notebooks/*.ipynb from notebooks/nbXX.py.   Usage: python scripts/build_notebooks.py"""
import importlib
import pathlib
import sys

import nbformat

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from notebooks._builder import build  # noqa: E402

MODULES = ["nb01", "nb02", "nb03", "nb04", "nb05"]


def main(only=None):
    for name in MODULES:
        if only and name not in only:
            continue
        try:
            mod = importlib.import_module(f"notebooks.{name}")
        except ModuleNotFoundError as e:
            if e.name != f"notebooks.{name}":
                raise
            print(f"skip {name} (not written yet)")
            continue
        nb = build(mod)
        out = ROOT / "notebooks" / mod.NOTEBOOK
        nbformat.write(nb, out)
        print(f"wrote {out.relative_to(ROOT)}  ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main(sys.argv[1:])
