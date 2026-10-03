"""Friendly visuals for the story notebooks: cards, chat bubbles, emoji grids, animations, quizzes.

Everything renders as plain HTML or an image, so it looks right in Colab, Jupyter and in the
pre-run notebooks. Interactive bits (quiz buttons, sliders) use ipywidgets when available and
always print a static version too.
"""
from __future__ import annotations

import base64
import html
import io

from IPython.display import HTML, Image, display

INK, PAPER, MUTED = "#1B2530", "#FBFAF7", "#5B6573"
COLORS = {"orange": "#B8471A", "blue": "#2F6FB0", "green": "#3F8F5A", "gold": "#C29A1B",
          "purple": "#7A4FA3", "grey": "#8A8F98"}
PIP_COLORS = {"base": "#8A8F98", "lora": "#2F6FB0", "rl": "#B8471A", "rl_long": "#7A4FA3", "instruct": "#3F8F5A"}
PIP_NAMES = {"base": "Pip after school", "lora": "Pip after shadowing", "rl": "Pip after coaching",
             "rl_long": "Over-coached Pip", "instruct": "Store-bought Pip"}
FONT = "font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif"


def _box(inner, border="#DDD7CB", bg=PAPER, extra=""):
    return (f'<div style="{FONT};background:{bg};color:{INK};border:1px solid {border};border-radius:14px;'
            f'padding:16px 20px;margin:6px 0;line-height:1.5;{extra}">{inner}</div>')


def show(*parts):
    """Display HTML strings side by side (wraps on narrow screens)."""
    cells = "".join(f'<div style="flex:1;min-width:260px">{p}</div>' for p in parts)
    display(HTML(f'<div style="display:flex;flex-wrap:wrap;gap:14px;align-items:stretch">{cells}</div>'))


def card(title, body="", icon="", color="blue", out=True):
    c = COLORS.get(color, color)
    h = _box(f'<div style="font-size:18px;font-weight:700;color:{c}">{icon} {html.escape(title)}</div>'
             f'<div style="font-size:15px;margin-top:6px">{body}</div>', border=c, extra=f"border-left:6px solid {c};")
    if out:
        display(HTML(h))
    return h


def big_number(value, label, color="blue", out=True):
    c = COLORS.get(color, color)
    h = _box(f'<div style="font-size:40px;font-weight:800;color:{c};line-height:1.1">{value}</div>'
             f'<div style="font-size:15px;color:{MUTED};margin-top:4px">{label}</div>', extra="text-align:center;")
    if out:
        display(HTML(h))
    return h


def flow(steps, color="blue"):
    """A row of boxes joined by arrows: [('👀', 'Look'), ('🦶', 'Move'), ...]."""
    c = COLORS.get(color, color)
    items = []
    for icon, label in steps:
        items.append(f'<div style="background:{PAPER};border:2px solid {c};border-radius:12px;padding:10px 14px;'
                     f'text-align:center;min-width:90px;color:{INK}"><div style="font-size:28px">{icon}</div>'
                     f'<div style="font-size:14px;font-weight:600">{label}</div></div>')
    arrow = f'<div style="font-size:26px;color:{c};align-self:center">➜</div>'
    display(HTML(f'<div style="{FONT};display:flex;flex-wrap:wrap;gap:8px;margin:8px 0">{arrow.join(items)}</div>'))


# ------------------------------------------------------------------ chat bubbles

def bubbles(customer, replies: dict, verdicts: dict | None = None, width=None):
    """A customer message, then one reply bubble per Pip. replies = {variant: text}."""
    rows = [f'<div style="display:flex;justify-content:flex-end;margin:6px 0"><div style="background:#E8EEF6;color:{INK};'
            f'border-radius:16px 16px 4px 16px;padding:10px 14px;max-width:80%"><b>🧑 Customer</b><br>'
            f'{html.escape(customer)}</div></div>']
    for v, text in replies.items():
        c = PIP_COLORS.get(v, COLORS["grey"])
        name = PIP_NAMES.get(v, v)
        badge = ""
        if verdicts and v in verdicts:
            ok, why = verdicts[v]
            badge = (f'<div style="margin-top:6px;font-size:13px;color:{COLORS["green"] if ok else COLORS["orange"]}">'
                     f'{"✅" if ok else "⚠️"} {html.escape(why)}</div>')
        body = html.escape(text if len(text) < 600 else text[:600] + " …").replace("\n", "<br>")
        rows.append(f'<div style="display:flex;margin:6px 0"><div style="background:{PAPER};color:{INK};'
                    f'border:2px solid {c};border-radius:16px 16px 16px 4px;padding:10px 14px;max-width:85%">'
                    f'<b style="color:{c}">🤖 {name}</b><br>{body}{badge}</div></div>')
    display(HTML(f'<div style="{FONT};font-size:15px;line-height:1.45;{f"max-width:{width}px;" if width else ""}">{"".join(rows)}</div>'))


# ------------------------------------------------------------------ emoji grids

TILES = {"#": "🧱", "G": "🏁", "c": "🪙", "T": "⚡", "S": "🏠", ".": "⬜"}


def grid_html(grid, path=None, robot=None, title="", note=""):
    """Draw a grid world with emoji. path = list of (row, col) cells to mark with footprints."""
    path = set(path or [])
    cells = ""
    for r, row in enumerate(grid):
        for c, t in enumerate(row):
            e = TILES.get(t, "⬜")
            if (r, c) == robot:
                e = "🤖"
            elif (r, c) in path and t in ".S":
                e = "👣"
            cells += f'<div style="font-size:26px;width:38px;height:38px;display:flex;align-items:center;justify-content:center">{e}</div>'
    cols = len(grid[0])
    return (f'<div style="{FONT};background:{PAPER};color:{INK};border:1px solid #DDD7CB;border-radius:14px;padding:12px;display:inline-block">'
            f'<div style="font-weight:700;margin-bottom:6px">{html.escape(title)}</div>'
            f'<div style="display:grid;grid-template-columns:repeat({cols},38px);gap:2px">{cells}</div>'
            f'<div style="font-size:14px;color:{MUTED};margin-top:6px;max-width:{cols * 40}px">{note}</div></div>')


def legend():
    display(HTML(f'<div style="{FONT};font-size:14px;color:{MUTED}">🤖 Pip · 🏠 start · 🪙 coin (+2, small, close) · '
                 f'🏁 goal (+10, big, far) · ⚡ turbo pad (+1 every time) · 🧱 wall · 👣 Pip\'s footsteps</div>'))


def animate_path(grid, path, title="", fps=4, label=None):
    """A small GIF of Pip walking a path through the grid (works in Colab, Jupyter and saved notebooks)."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter

    h, w = len(grid), len(grid[0])
    fig, ax = plt.subplots(figsize=(w * 0.8, h * 0.8 + 0.6), dpi=72)
    fill = {"#": "#3A3F47", "G": "#3F8F5A", "c": "#C29A1B", "T": "#B8471A", "S": "#DCE6F2"}
    sym = {"G": "GOAL", "c": "+2", "T": "+1", "S": "start"}

    def draw(i):
        ax.clear()
        for r in range(h):
            for c in range(w):
                t = grid[r][c]
                ax.add_patch(plt.Rectangle((c, h - 1 - r), 1, 1, color=fill.get(t, "#F4F2EE"), ec="white", lw=2))
                if t in sym:
                    ax.text(c + 0.5, h - 0.5 - r, sym[t], ha="center", va="center", fontsize=7,
                            color="white" if t != "S" else INK, weight="bold")
        trail = path[:i + 1]
        if len(trail) > 1:
            ax.plot([c + 0.5 for _, c in trail], [h - 0.5 - r for r, _ in trail], color="#2F6FB0", lw=3, alpha=0.45)
        r, c = path[min(i, len(path) - 1)]
        ax.add_patch(plt.Circle((c + 0.5, h - 0.5 - r), 0.33, color="#2F6FB0"))
        ax.text(c + 0.5, h - 0.5 - r, "Pip", ha="center", va="center", fontsize=7, color="white", weight="bold")
        ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{title}  ·  step {min(i, len(path) - 1)}" + (f"  ·  {label(i)}" if label else ""), fontsize=9)

    frames = list(range(len(path))) + [len(path) - 1] * 4
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
        tmp = f.name
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, tmp, dpi=72):
        for i in frames:
            draw(i)
            writer.grab_frame()
    plt.close(fig)
    data = base64.b64encode(open(tmp, "rb").read()).decode()
    os.unlink(tmp)
    display(HTML(f'<img src="data:image/gif;base64,{data}" alt="{html.escape(title)}" style="max-width:100%">'))


# ------------------------------------------------------------------ quiz and sliders

def quiz(questions):
    """questions = [(question, [options], correct_index, why)]. Buttons if ipywidgets exists; answers in a fold either way."""
    try:
        import ipywidgets as w
    except ImportError:
        w = None
    score = {"right": 0, "answered": set()}
    if w is not None:
        boxes = []
        total = w.HTML()

        def update_total():
            total.value = (f'<div style="{FONT};font-size:16px;font-weight:700">Score: {score["right"]} / {len(questions)}'
                           f'{" 🎉" if score["right"] == len(questions) else ""}</div>')

        for qi, (q, opts, correct, why) in enumerate(questions):
            fb = w.HTML()
            btns = [w.Button(description=o, layout=w.Layout(width="auto")) for o in opts]

            def click(b, qi=qi, correct=correct, why=why, fb=fb, btns=btns, opts=opts):
                ok = b.description == opts[correct]
                if qi not in score["answered"]:
                    score["answered"].add(qi)
                    score["right"] += ok
                    update_total()
                for x in btns:
                    x.button_style = "success" if x.description == opts[correct] else ("danger" if x is b else "")
                fb.value = f'<div style="{FONT}">{"✅ Yes!" if ok else "❌ Not quite."} {html.escape(why)}</div>'
            for b in btns:
                b.on_click(click)
            boxes.append(w.VBox([w.HTML(f'<div style="{FONT};font-weight:600;margin-top:8px">{qi + 1}. {html.escape(q)}</div>'),
                                 w.HBox(btns), fb]))
        update_total()
        display(w.VBox(boxes + [total]))
    key = "".join(f"<li><b>{html.escape(q)}</b> → {html.escape(o[c])}. {html.escape(why)}</li>" for q, o, c, why in questions)
    display(HTML(f'<details style="{FONT};margin-top:8px"><summary>📖 Answer key</summary><ol>{key}</ol></details>'))


def slider(fn, values, description, default=None):
    """Show fn(default) now (static), plus a slider to try other values live."""
    default = values[len(values) // 2] if default is None else default
    try:
        import ipywidgets as w
    except ImportError:
        fn(default)
        return
    out = w.Output()
    s = w.SelectionSlider(options=values, value=default, description=description,
                          style={"description_width": "initial"}, layout=w.Layout(width="480px"), continuous_update=False)

    def on(change):
        out.clear_output(wait=True)
        with out:
            fn(change["new"])
    s.observe(on, names="value")
    with out:
        fn(default)
    display(w.VBox([s, out]))


def chart(fig):
    """Display a matplotlib figure as an image and close it (keeps saved notebooks light)."""
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    display(Image(data=buf.getvalue(), format="png"))


def png_b64(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()
