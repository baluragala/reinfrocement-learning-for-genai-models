"""Pip: the friendly face of the lab. Short, readable calls for the story notebooks.

    pip = Robot()          # a robot in a maze that knows nothing
    pip.wander()           # watch it stumble around at random
    pip.practice()         # let it try 2,000 times, collecting stars
    pip.route()            # where does it go now?

Under the hood this is the same Q-learning, bandit and model code as rllab.toy / rllab.models.
"""
from __future__ import annotations

import numpy as np

from . import checks, models, toy, ui
from .data import EVAL_PROMPTS, sft_dataset

ARROW_EMOJI = {0: "⬆️", 1: "➡️", 2: "⬇️", 3: "⬅️"}
WORLDS = {"maze": toy.NAV_MAP, "race": toy.RACE_MAP}
TILE_NAMES = {"🪙": "c", "🏁": "G", "⚡": "T"}


def _plt():
    import matplotlib.pyplot as plt
    return plt


# ------------------------------------------------------------------ the robot

class Robot:
    """Pip as a robot. patience = how much Pip cares about stars that come later (0 = only now, 1 = forever)."""

    def __init__(self, world="maze", patience=0.95, rewards=None, step_cost=-0.1, name="Pip"):
        rewards = {TILE_NAMES.get(k, k): v for k, v in (rewards or {}).items()}
        self.world, self.name, self.patience = world, name, patience
        self.env = toy.GridWorld(WORLDS[world], step_cost=step_cost, rewards=rewards)
        self.Q = None
        self.history = None

    def _repr_html_(self):
        done = f"practised {len(self.history):,} times" if self.history is not None else "hasn't practised yet"
        return f'<span style="{ui.FONT};color:{ui.MUTED}">🤖 {self.name} · patience {self.patience} · {done}</span>'

    # -- looking
    def look(self, title=None):
        ui.show(ui.grid_html(self.env.grid, robot=self.env.start, title=title or f"{self.name}'s world"))
        ui.legend()

    # -- before learning
    def wander(self, steps=30, seed=3):
        """Random moves: what a robot that knows nothing does."""
        rng = np.random.default_rng(seed)
        s, path, found = self.env.reset(), [self.env.start], None
        for _ in range(steps):
            s, r, done = self.env.step(int(rng.integers(4)))
            path.append(s)
            if done:
                found = self.env.tile(s)
                break
        ui.animate_path(self.env.grid, path, title=f"{self.name} wandering at random")
        what = {"c": "bumped into the 🪙 coin by luck", "G": "stumbled onto the 🏁 goal by luck"}.get(found, "found nothing")
        ui.card(f"{len(path) - 1} random steps: {self.name} {what}.",
                "No map, no plan, no idea which squares are good. It just tries moves.", "🎲", "grey")

    # -- learning
    def practice(self, tries=2000, seed=0, quiet=False):
        """Let Pip try again and again. Each try: look → move → get stars (or lose a little) → remember."""
        res = toy.q_learning(self.env, episodes=tries, gamma=self.patience, seed=seed, log_first=1)
        self.Q, self.history, self._log = res["Q"], res["returns"], res["log"]
        if not quiet:
            plt = _plt()
            fig, ax = plt.subplots(figsize=(7.5, 2.8))
            k = 50
            smooth = np.convolve(self.history, np.ones(k) / k, mode="valid")
            ax.plot(smooth, color=ui.COLORS["blue"], lw=2.5)
            ax.set_xlabel("try number"); ax.set_ylabel("stars per try")
            ax.set_title(f"{self.name} practising {tries:,} times", fontsize=11)
            ax.spines[["top", "right"]].set_visible(False)
            lo, hi = min(smooth.min(), 0), smooth.max()
            ax.axvspan(0, tries * 0.3, color="#C29A1B", alpha=0.12)
            ax.axvspan(tries * 0.7, len(smooth), color="#3F8F5A", alpha=0.12)
            ax.text(tries * 0.15, lo + 0.08 * (hi - lo), "exploring:\nlots of random moves",
                    ha="center", fontsize=9, color="#7A5E00")
            ax.text(tries * 0.84, lo + 0.08 * (hi - lo), "figured it out", ha="center", fontsize=9, color="#2C6B40")
            ui.chart(fig)
        return self

    def peek(self, n=6):
        """The first few moves of the first try, as the learning loop sees them."""
        rows = ""
        for line in self._log[:n]:
            # "ep 0 · observe (r, c) · act ↓ · reward -0.1 · Q↓ +0.00 → -0.08"
            parts = [p.strip() for p in line.split("·")]
            cell, move, reward = parts[1].replace("observe ", ""), parts[2].replace("act ", ""), parts[3].replace("reward ", "")
            emoji = {"↑": "⬆️", "→": "➡️", "↓": "⬇️", "←": "⬅️"}[move]
            rows += (f"<tr><td>👀 at square {cell}</td><td>{emoji} moves</td><td>{'⭐' if float(reward) > 0 else '➖'} {reward}</td>"
                     f"<td>🧠 that move from there is now worth {parts[4].split('→')[1].strip()}</td></tr>")
        ui.show(f'<table style="{ui.FONT};font-size:14px;border-collapse:collapse;background:{ui.PAPER};color:{ui.INK}">'
                f'<tr><th>Look</th><th>Move</th><th>Stars</th><th>Remember</th></tr>{rows}</table>')

    def route_html(self, title=None):
        r = toy.rollout(self.env, self.Q)
        where = {"G": "the big 🏁 goal (+10)", "c": "the small 🪙 coin (+2)"}.get(r["ended_on"], "nowhere: it ran out of time")
        return ui.grid_html(self.env.grid, path=r["path"], robot=r["path"][-1], title=title or f"{self.name}'s route",
                            note=f"Went to {where} in {r['steps']} steps.")

    def route(self, title=None, show=True):
        """Follow Pip's best guess from the start. Returns where it ended up."""
        if show:
            ui.show(self.route_html(title))
            return None
        return toy.rollout(self.env, self.Q)

    def animate(self, title=None):
        r = toy.rollout(self.env, self.Q)
        label = None
        if self.world == "race":
            visits = np.cumsum([self.env.tile(p) == "T" for p in r["path"]])
            label = lambda i: f"⚡ pad visits: {visits[min(i, len(visits) - 1)]}"
        ui.animate_path(self.env.grid, r["path"][:41], title=title or f"{self.name} after practising", label=label)

    def hunches(self):
        """Pip's 'mind map': in every square, the move it now thinks is best."""
        g = self.env.grid
        best = self.Q.max(1).reshape(len(g), len(g[0]))
        lo, hi = best.min(), best.max()
        cells = ""
        for r, row in enumerate(g):
            for c, t in enumerate(row):
                if t in "#GcT":
                    e, bg = ui.TILES[t], "#F4F2EE"
                else:
                    e = ARROW_EMOJI[int(np.argmax(self.Q[self.env.index((r, c))]))]
                    v = (best[r, c] - lo) / (hi - lo + 1e-9)
                    bg = f"rgba(63,143,90,{0.12 + 0.6 * v:.2f})"
                cells += (f'<div style="font-size:22px;width:38px;height:38px;display:flex;align-items:center;'
                          f'justify-content:center;background:{bg};border-radius:6px">{e}</div>')
        ui.show(f'<div style="{ui.FONT};background:{ui.PAPER};color:{ui.INK};border:1px solid #DDD7CB;border-radius:14px;padding:12px;display:inline-block">'
                f'<div style="font-weight:700;margin-bottom:6px">{self.name}\'s mind map</div>'
                f'<div style="display:grid;grid-template-columns:repeat({len(g[0])},38px);gap:2px">{cells}</div>'
                f'<div style="font-size:14px;color:{ui.MUTED};margin-top:6px;max-width:330px">Arrow = the move Pip now prefers there. '
                f'Greener = "good things are close from here".</div></div>')

    @property
    def finished(self):
        return toy.rollout(self.env, self.Q)["reached_goal"]


def compare(**robots):
    """Show several Pips' routes side by side: compare(impatient=a, patient=b)."""
    ui.show(*[r.route_html(name.replace("_", " ").title()) if isinstance(r, Robot) else r.route(name.replace("_", " ").title())
              for name, r in robots.items()])


def check_race(racer):
    r = toy.rollout(racer.env, racer.Q)
    if r["reached_goal"]:
        ui.card(f"🏆 Finished in {r['steps']} steps!", "You fixed the stars, not the robot. The learning algorithm never changed.", "", "green")
    else:
        ui.card("Not yet: the car still prefers the pad.", f"It visited ⚡ {r['turbo_visits']} times and never crossed 🏁. "
                "Make circling worth less than finishing.", "🔁", "orange")


def patience_demo(values=(0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99)):
    """Slider: retrain Pip with different patience and see where it goes."""
    cache = {}

    def run(p):
        if p not in cache:
            cache[p] = Robot(patience=p).practice(quiet=True)
        r = cache[p].route(show=False)
        where = "🏁 the big goal" if r["ended_on"] == "G" else "🪙 the nearby coin"
        ui.show(ui.grid_html(cache[p].env.grid, path=r["path"], robot=r["path"][-1], title=f"patience = {p}",
                             note=f"Goes for {where} ({r['steps']} steps)."))
    ui.slider(run, list(values), "Pip's patience", default=0.5)


# ------------------------------------------------------------------ copying vs scoring

def sam_drives(n=20):
    """Sam, a human operator, steers the robot 20 times. Sam usually takes the easy coin."""
    return toy.demonstrations(toy.GridWorld(toy.NAV_MAP), n=n, coin_share=0.7)


class CopyCat:
    """A Pip that learns by copying Sam: in each square, do what Sam did most often."""

    def __init__(self, drives):
        self.env = toy.GridWorld(toy.NAV_MAP)
        self.policy, self.labels = toy.behaviour_cloning(self.env, drives)

    def route(self, title="Copy-cat Pip's route"):
        r = toy.rollout(self.env, policy=self.policy)
        where = {"G": "the big 🏁 goal", "c": "the small 🪙 coin"}.get(r["ended_on"], "nowhere")
        return ui.grid_html(self.env.grid, path=r["path"], robot=r["path"][-1], title=title,
                            note=f"Copied Sam's habit → went to {where}.")


# ------------------------------------------------------------------ curiosity (banner game)

PIPS = {"Stubborn Pip": (0.0, "never tries anything new"), "Curious Pip": (0.1, "tries something new 1 time in 10"),
        "Scatterbrained Pip": (0.5, "tries something random half the time")}


BANNER_EMOJI = ["⛺", "🧥", "🥾", "🔦", "🎒"]


def banner_race(steps=3000, seeds=range(30)):
    """Three Pips choose which banner to show shoppers. Who gets the most clicks?"""
    best = int(np.argmax(toy.CLICK_RATES))
    rows, cards = "", []
    for (name, (eps, desc)), color in zip(PIPS.items(), ["grey", "blue", "orange"]):
        runs = [toy.run_bandit(eps, steps, s) for s in seeds]
        clicks = np.mean([r["rewards"].sum() for r in runs])
        share = np.mean([np.bincount(r["arms"], minlength=5) / steps for r in runs], 0)
        icons = "".join(
            f'<div style="text-align:center;width:70px"><div style="font-size:{18 + 40 * v:.0f}px;opacity:{0.25 + 0.75 * min(1, v * 2):.2f}">'
            f'{e}</div><div style="font-size:12px;color:{ui.COLORS["green"] if i == best else ui.MUTED}">{v:.0%}</div></div>'
            for i, (e, v) in enumerate(zip(BANNER_EMOJI, share)))
        rows += (f'<div style="display:flex;align-items:center;gap:10px;margin:6px 0"><div style="width:170px;font-weight:700;'
                 f'color:{ui.COLORS[color]}">{name}</div>{icons}</div>')
        cards.append(ui.big_number(f"{int(clicks)}", f"clicks for <b>{name}</b><br><i>{desc}</i>", color=color, out=False))
    ui.show(f'<div style="{ui.FONT};background:{ui.PAPER};color:{ui.INK};border:1px solid #DDD7CB;border-radius:14px;padding:14px">'
            f'<b>Which banners each Pip showed</b> (bigger = shown more · the 🔦 headlamp banner secretly gets the most clicks)'
            f'{rows}</div>')
    ui.show(*cards)


# ------------------------------------------------------------------ the language-model Pip

def lifecycle():
    ui.flow([("📚", "School<br><small>reads the internet</small>"), ("👀", "Shadowing<br><small>copies senior agents</small>"),
             ("⭐", "Coaching<br><small>gets thumbs-up / down</small>")], color="orange")
    ui.show(ui.card("School = pretraining", "Learns language and facts by predicting the next word on billions of pages. "
                    "<b>Gains knowledge.</b>", "📚", "grey", out=False),
            ui.card("Shadowing = fine-tuning (LoRA)", "Copies example answers written by Acme's agents. "
                    "<b>Learns the job's format and voice.</b>", "👀", "blue", out=False),
            ui.card("Coaching = reinforcement learning", "Gets scored on its own answers. "
                    "<b>Learns judgement: which answer is better.</b>", "⭐", "orange", out=False))


def guess_next_word(text="The Eiffel Tower is located in the city of"):
    """What does school-Pip think comes next?"""
    probs = [(w, p) for w, p in models.next_token_probs("base", text, k=12) if w.strip().isalpha()][:5]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 2.4))
    words = [w.strip() or repr(w) for w, _ in probs][::-1]
    vals = [p for _, p in probs][::-1]
    ax.barh(words, vals, color=["#8A8F98"] * (len(vals) - 1) + ["#2F6FB0"])
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=9)
    ax.set_xlim(0, max(vals) * 1.25); ax.set_title(f'"{text} ___"', fontsize=10)
    ax.spines[["top", "right"]].set_visible(False); ax.set_xticks([])
    ui.chart(fig)


def ask(prompt_id, pips=("base", "lora")):
    """Ask the same customer message to several Pips and show the replies as a chat, with a quick check on each."""
    p = next(x for x in EVAL_PROMPTS if x.id == prompt_id)
    replies = {v: models.reply(v, p) for v in pips}
    verdicts = {}
    for v, r in replies.items():
        res = checks.check(p, r)
        verdicts[v] = (res["pass"], "did what was asked" if res["pass"] else res["why"])
    ui.bubbles(p.prompt, replies, verdicts)


STYLE_INFO = {"good": ("👍 good", "#3F8F5A"), "verbose": ("🥱 rambling", "#C29A1B"), "curt": ("😐 curt", "#8A8F98")}


def copying_problem():
    """Acme's example answers came in three styles. Shadowing copies all of them, in the same mix."""
    rows = [r for r in sft_dataset() if r["family"] == "policy"]
    counts = {s: sum(r["style"] == s for r in rows) for s in STYLE_INFO}
    total = sum(counts.values())
    bar = "".join(f'<div style="width:{100 * n / total:.1f}%;background:{STYLE_INFO[s][1]};color:white;padding:10px 6px;'
                  f'font-weight:700;text-align:center">{STYLE_INFO[s][0]} {n / total:.0%}</div>' for s, n in counts.items())
    ui.show(f'<div style="{ui.FONT};color:{ui.INK}"><b>What Acme\'s {total} example answers to policy questions look like</b>'
            f'<div style="display:flex;border-radius:10px;overflow:hidden;margin:8px 0">{bar}</div>'
            f'<b>What shadowing-Pip learns to produce</b>'
            f'<div style="display:flex;border-radius:10px;overflow:hidden;margin:8px 0">{bar}</div></div>')
    ex = {s: next(r["response"] for r in rows if r["style"] == s) for s in STYLE_INFO}
    ui.show(*[ui.card(STYLE_INFO[s][0], ex[s][:220] + ("…" if len(ex[s]) > 220 else ""), "", STYLE_INFO[s][1], out=False)
              for s in STYLE_INFO])
