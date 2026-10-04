"""Pip the chatbot: plain-English helpers for Chapters 2–5. Re-exported by rllab.pip, so notebooks call P.xxx().

Every helper shows a picture or a chat, and returns nothing a learner has to read as raw data.
Under the hood: rllab.policy (the miniature RL/RLHF/DPO), rllab.models (cached replies from the real
Qwen models), rllab.evals and rllab.risks.
"""
from __future__ import annotations

import html

import numpy as np

from . import checks, config, data, evals, models, policy, risks, ui

E = html.escape


def _plt():
    import matplotlib.pyplot as plt
    return plt


def word_count(reply):
    return checks.words(reply)


def has_exactly_3_bullets(reply):
    return checks.bullets(reply) == 3


def _prompt(pid):
    return next(p for p in data.EVAL_PROMPTS if p.id == pid)


# ================================================================== Chapter 2 · gold stars for answers

def maze_to_chat():
    rows = [("🤖 Agent", "Pip the robot", "Pip the chatbot (the AI model)"), ("🗺️ State", "the square it's on", "the customer's message"),
            ("🦶 Action", "one move", "the whole reply it writes"), ("⭐ Reward", "stars from the maze", "a score for the reply. <b>Someone has to write it!</b>"),
            ("🎲 Exploring", "random moves", "trying different wordings")]
    body = "".join(f"<tr><td><b>{a}</b></td><td>{b}</td><td>{c}</td></tr>" for a, b, c in rows)
    ui.show(f'<table style="{ui.FONT};font-size:15px;background:{ui.PAPER};color:{ui.INK};border-collapse:collapse">'
            f'<tr><th></th><th>🧩 Maze (Chapter 1)</th><th>💬 Chatbot (now)</th></tr>{body}</table>')


IF1 = "Using exactly 3 bullet points, tell me how to return my rain jacket."
CANDIDATES = {
    "A": "- Start a return from Orders in your account.\n- Print the prepaid label we email you.\n- Drop the parcel at any carrier location.",
    "B": "Start a return from Orders in your account, print the prepaid label we email you, and drop the parcel at any carrier location.",
    "C": "Thank you so much for reaching out to Acme Outfitters today! We truly appreciate you taking the time to contact us, and "
         "we're always absolutely delighted to help. To return your rain jacket, simply start a return from Orders in your account, "
         "then print the prepaid label we email you, and finally drop the parcel at any carrier location. I really hope this helps, "
         "and please don't hesitate to reach out again. Happy adventuring!",
    "D": "- Go to Orders.\n- Start a return.\n- Print the label.\n- Pack the jacket.\n- Drop it off.",
    "E": "Returns are easy, just follow the steps on our website.",
}
NICKNAMES = {"A": "✅ 3 clear bullets", "B": "🙈 one long sentence", "C": "🥰 gushing paragraph", "D": "🔢 5 bullets", "E": "🤷 unhelpful"}
START_MIX = np.array([0.40, 0.30, 0.12, 0.08, 0.10])     # how often shadowing-Pip gives each kind of answer


def _stars_badge(v):
    full = min(10, max(0, int(round(v * 5))))
    return f'<span style="font-size:18px">{"⭐" * full or "▫️"}</span> <b>{v:+.2f}</b>'


def show_candidates(stars=None):
    """The five candidate answers as cards, with star scores if a star rule is given."""
    cards = []
    for k, text in CANDIDATES.items():
        score = f'<div style="margin-top:8px">{_stars_badge(stars(text))}</div>' if stars else ""
        body = E(text).replace("\n", "<br>")
        cards.append(f'<div style="{ui.FONT};background:{ui.PAPER};color:{ui.INK};border:2px solid #DDD7CB;border-radius:12px;'
                     f'padding:10px;font-size:13px"><b>{k} · {NICKNAMES[k]}</b><br>{body}{score}</div>')
    ui.show(f'<div style="{ui.FONT};font-weight:700;margin-bottom:4px">🧑 "{E(IF1)}"</div>')
    ui.show(*cards[:3])
    ui.show(*cards[3:])


def pip_tries(stars, n=8):
    """Shadowing-Pip answers the same message 8 times (a little randomness each time). Score each try."""
    samples = models.generate("lora", [IF1], sample=True, n=n)[0]
    scores = [stars(s) for s in samples]
    cards = [f'<div style="{ui.FONT};background:{ui.PAPER};color:{ui.INK};border:2px solid '
             f'{ui.COLORS["green"] if sc == max(scores) else "#DDD7CB"};border-radius:12px;padding:8px;font-size:12px">'
             f'<b>Try {i + 1}</b> {_stars_badge(sc)}<br>{E(s[:220]).replace(chr(10), "<br>")}</div>'
             for i, (s, sc) in enumerate(zip(samples, scores))]
    for i in range(0, len(cards), 4):
        ui.show(*cards[i:i + 4])
    best = sum(sc == max(scores) for sc in scores)
    ui.card(f"{best} of {n} tries earned the top score.", "Pip can already write the best answer sometimes. Coaching's job is to make "
            "it write that answer <b>more often</b>.", "🎯", "green")


def nudge(stars, leash=0.05, steps=300):
    """Coaching on the 5 candidates: try, score, nudge. Watch the probabilities move."""
    r = np.array([stars(t) for t in CANDIDATES.values()])
    run = policy.reinforce(START_MIX, r, beta=leash, steps=steps, seed=0)
    labels = [f"{k} {NICKNAMES[k]}" for k in CANDIDATES]
    colors = ["#3F8F5A", "#A0522D", "#D17FA6", "#C29A1B", "#8A8F98"]
    ui.animate_bars(labels, run["probs"], title="How often Pip gives each answer", colors=colors,
                    note_fn=lambda i: f"avg stars {run['expected_reward'][i]:+.2f}")
    return None


def lazy_stars(reply):
    """A lazy star rule a busy team might write: longer = more helpful."""
    return round(word_count(reply) / 30, 2)


def leash_demo(real_stars, values=(0.0, 1.0, 3.0)):
    """Coach Pip with the LAZY star rule, at different leash strengths. Slide to compare."""
    lazy = np.array([lazy_stars(t) for t in CANDIDATES.values()])
    real = np.array([real_stars(t) for t in CANDIDATES.values()])
    runs = {b: policy.reinforce(START_MIX, lazy, beta=b, steps=300, seed=0)["final"] for b in values}

    def view(b):
        p = runs[b]
        bars = "".join(ui.meter(f"{k} {NICKNAMES[k]}", float(v), "#D17FA6" if k == "C" else "blue") for k, v in zip(CANDIDATES, p))
        ui.show(f'<div style="{ui.FONT};background:{ui.PAPER};color:{ui.INK};border:1px solid #DDD7CB;border-radius:14px;padding:12px">'
                f'<b>Leash strength {b}</b>{bars}</div>',
                ui.big_number(f"{float(p @ real):+.2f}", "average <b>real</b> stars (Acme's rule, which coaching never saw)",
                              "green" if p @ real > 0.2 else "orange", out=False))
    ui.slider(view, list(values), "Leash strength", default=values[0])
    ui.show(*[ui.big_number(f"{float(runs[b] @ real):+.2f}", f"real stars with leash <b>{b}</b>",
                            "green" if runs[b] @ real > 0.2 else "orange", out=False) for b in values])


RATER_PAIRS = [("harmful", "good", "comply"), ("unknown", "good", "fabricate"), ("tone", "warm_long", "good"), ("pushback", "sycophantic", "good")]
VOTES = {}


def be_the_rater():
    """You are an Acme support lead. Pick the better reply in each pair."""
    items = []
    for i, (fam, a, b) in enumerate(RATER_PAIRS):
        prompt, replies = data.candidates(fam, seed=3)
        first, second = (replies[a], replies[b]) if i % 2 == 0 else (replies[b], replies[a])
        items.append((fam, prompt, first, second))
    ui.ab_vote(items, store=VOTES)


def rater_reveal():
    """What did Acme's real raters pick for these kinds of message, across all 480 comparisons?"""
    pairs = data.preference_dataset()
    rows = ""
    for i, (fam, a, b) in enumerate(RATER_PAIRS):
        fp = [p for p in pairs if p["family"] == fam]
        share_a = np.mean([p["chosen_style"] == a for p in fp])
        yours = VOTES.get(fam)
        your_style = None if yours is None else ((a if yours == "1" else b) if i % 2 == 0 else (b if yours == "1" else a))
        flag = "🚩" if a in ("warm_long", "sycophantic") and share_a > 0.5 else ""
        rows += (f"<tr><td>{ui.KINDS[fam][0]} {ui.KINDS[fam][1]}</td><td>{ui.style_label(your_style) if your_style else '—'}</td>"
                 f"<td>{flag} 👍 went to {ui.style_label(a)} in {share_a:.0%} of these comparisons</td></tr>")
    ui.show(f'<table style="{ui.FONT};font-size:15px;background:{ui.PAPER};color:{ui.INK}"><tr><th>Kind of message</th>'
            f'<th>You picked</th><th>Acme\'s raters</th></tr>{rows}</table>')
    ui.card("🚩 Raters are human.", "On angry customers they mostly preferred the <b>gushing</b> reply, and when customers were "
            "confidently wrong they mostly preferred the reply that <b>agreed</b>. Those habits are now part of the training data.", "", "orange")


SNEAKY = [("an honest, complete answer", "Refunds reach your card 5–7 business days after we receive the item.", True),
          ("just the number", "5–7", False),
          ("keyword stuffing", "5–7 5–7 business days business days refund refund refund", False),
          ("copies the question", "When will I see my refund? 5–7 business days.", False),
          ("polite but wrong", "You'll see your refund within 24 hours!", False),
          ("empty", "", False)]


def test_star_rule(my_stars):
    """Throw sneaky replies at your star rule. Did any of them beat (or tie) the honest answer?"""
    honest = my_stars(SNEAKY[0][1])
    rows, fooled = "", 0
    for name, text, good in SNEAKY:
        s = my_stars(text)
        bad = (not good) and s >= honest
        fooled += bad
        rows += (f'<tr><td>{"🕵️" if bad else ("🏅" if good else "🛡️")}</td><td><b>{name}</b></td><td><code>{E(text) or "(empty)"}</code></td>'
                 f'<td><b>{s:+.2f}</b></td></tr>')
    ui.show(f'<table style="{ui.FONT};font-size:14px;background:{ui.PAPER};color:{ui.INK}"><tr><th></th><th>Reply</th><th>Text</th>'
            f'<th>Your stars</th></tr>{rows}</table>')
    if fooled:
        ui.card(f"🕵️ {fooled} sneaky repl{'y' if fooled == 1 else 'ies'} scored at least as well as the honest answer.",
                "Coaching would happily learn those. Tighten your rule and run again.", "", "orange")
    else:
        ui.card("🛡️ Your rule held up against these sneaky replies.", "An optimiser tries millions of replies, though. "
                "Rules can't anticipate everything, which is why we ask people (Chapter 3).", "", "green")


# ================================================================== Chapter 3 · the thumbs-up machine

def two_routes():
    def lane(title, color, steps):
        boxes = '<div style="font-size:22px;align-self:center;color:#5B6573">➜</div>'.join(
            f'<div style="background:{ui.PAPER};border:2px solid {color};border-radius:12px;padding:8px 12px;text-align:center;'
            f'min-width:120px"><div style="font-size:26px">{i}</div><div style="font-size:13px">{t}</div></div>' for i, t in steps)
        return (f'<div style="{ui.FONT};color:{ui.INK};margin:8px 0"><b style="color:{color};font-size:17px">{title}</b>'
                f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:6px">{boxes}</div></div>')
    ui.show(lane("Route 1 · RLHF: hire a judge, then practise", ui.COLORS["blue"],
                 [("👍👎", "480 comparisons"), ("🧑‍⚖️", "train a <b>judge</b> robot<br>to score replies"),
                  ("🔁", "Pip writes replies,<br>judge scores them"), ("⭐", "nudge Pip toward<br>high scores (+ leash)")]) +
            lane("Route 2 · DPO: study the comparisons directly", ui.COLORS["orange"],
                 [("👍👎", "the same 480 comparisons"), ("📖", "for each pair: make the 👍 reply<br>more likely, the 👎 one less"),
                  ("🪢", "measured against where Pip<br>started (built-in leash)")]))


FRIENDLY = {"length (per 50 words)": "📏 long replies", "bullet points": "• bullet points", "refuses": "🙅 refusing",
            "admits uncertainty": "🤷 saying 'I don't know'", "acknowledges feelings": "💛 acknowledging feelings",
            "agrees with customer": "🙇 agreeing with the customer", "valid JSON": "🧾 valid JSON",
            "exclamation marks": "❗ exclamation marks", "answers yes/no first": "✅ yes/no first"}


class Judge:
    """A judge robot trained on Acme's comparisons (a Bradley-Terry reward model)."""

    def __init__(self, careless=0.0):
        pairs = data.preference_dataset()
        self.rm = policy.fit_reward_model(pairs[:400], flip=careless, seed=1)
        self.accuracy = policy.rm_accuracy(self.rm, pairs[400:])

    def likes(self):
        w = self.rm["w"]
        order = np.argsort(w)
        plt = _plt()
        fig, ax = plt.subplots(figsize=(7, 3.4))
        names = [ui.plain(FRIENDLY[self.rm["features"][i]]) for i in order]
        colors = ["#B8471A" if self.rm["features"][i] in ("agrees with customer", "length (per 50 words)") else
                  ("#2F6FB0" if w[i] > 0 else "#8A8F98") for i in order]
        ax.barh(names, w[order], color=colors)
        ax.axvline(0, color="#1B2530", lw=0.8)
        ax.set_title(f"What the judge learned to like (agrees with raters {self.accuracy:.0%} of the time)", fontsize=10)
        ax.set_xlabel("← dislikes        likes →"); ax.set_xticks([])
        ax.spines[["top", "right"]].set_visible(False)
        ui.chart(fig)

    def ignore(self, feature_words):
        """Switch off something the judge likes, e.g. judge.ignore('agreeing')."""
        for i, f in enumerate(self.rm["features"]):
            if feature_words.lower() in FRIENDLY[f].lower() or feature_words.lower() in f:
                self.rm["w"][i] = 0.0
                ui.card(f"The judge no longer cares about {FRIENDLY[f]}.", "Same comparisons, same Pip: only the judge changed.", "🔧", "blue")
        return self

    def _repr_html_(self):
        return f'<span style="{ui.FONT};color:{ui.MUTED}">🧑‍⚖️ Judge · agrees with Acme\'s raters {self.accuracy:.0%} of the time</span>'


WORLD = policy.toy_world()


def _outcome_rows(res, title):
    rows = ""
    for fam, w in WORLD.items():
        top = w["styles"][int(np.argmax(res[fam]["final"]))]
        bad = policy.GOLD[top] < 0.5
        rows += (f'<tr><td style="white-space:nowrap">{ui.KINDS[fam][0]} {ui.KINDS[fam][1]}</td><td>{ui.stack_bar(w["ref"], w["styles"], 180)}</td>'
                 f'<td>{ui.stack_bar(res[fam]["final"], w["styles"], 180)}</td>'
                 f'<td>{"⚠️ " if bad else ""}{ui.style_label(top)}</td></tr>')
    gold_before = float(np.mean([w["ref"] @ w["gold"] for w in WORLD.values()]))
    gold_after = policy.gold_score(WORLD, res)
    ui.show(ui.big_number(f"{gold_before:.2f} → {gold_after:.2f}", f"<b>{title}</b>: how good Pip's answers are, as Acme sees it (0 = bad, 1 = perfect)",
                          "green" if gold_after > gold_before else "orange", out=False))
    ui.show(f'<table style="{ui.FONT};color:{ui.INK};font-size:14px;background:{ui.PAPER}">'
            f'<tr><th>Kind of message</th><th>Before (shadowing-Pip)</th><th>After</th><th>Now mostly…</th></tr>{rows}</table>')
    ui.show('<div style="font-size:13px;color:#5B6573">' + " · ".join(
        f'<span style="color:{c}">■</span> {e} {n}' for s, (e, n, c) in ui.STYLES.items() if s != "rm_exploit") + "</div>")


def rlhf(judge, leash=0.1):
    """Route 1: Pip practises against the judge."""
    res = policy.rlhf(WORLD, judge.rm, beta=leash)
    _outcome_rows(res, "RLHF: Pip coached by the judge robot")
    return None


def dpo(pairs=None, leash=0.1):
    """Route 2: Pip studies the comparisons directly."""
    res = policy.dpo(WORLD, pairs or data.preference_dataset(), beta=leash)
    _outcome_rows(res, "DPO: Pip coached straight from the comparisons")
    return None


def cost_picture():
    def side(title, color, icons, facts):
        ic = "".join(f'<div style="text-align:center;margin:4px"><div style="font-size:34px">{e}</div><div style="font-size:12px">{t}</div></div>'
                     for e, t in icons)
        fl = "".join(f"<li>{f}</li>" for f in facts)
        return (f'<div style="{ui.FONT};background:{ui.PAPER};color:{ui.INK};border:2px solid {color};border-radius:14px;padding:12px">'
                f'<b style="color:{color};font-size:17px">{title}</b><div style="display:flex;flex-wrap:wrap">{ic}</div><ul>{fl}</ul></div>')
    steps, batch = 300, 16
    n = steps * batch * len(WORLD)
    ui.show(side("RLHF", ui.COLORS["blue"], [("🤖", "Pip"), ("🤖", "Pip's starting copy"), ("🧑‍⚖️", "judge"), ("📈", "helper that predicts scores")],
                 [f"writes and scores <b>{n:,}</b> replies in our mini version", "4 models in memory", "powerful, but slow and fiddly",
                  "✅ you can inspect and edit the judge"]),
            side("DPO", ui.COLORS["orange"], [("🤖", "Pip"), ("🤖", "Pip's starting copy")],
                 ["writes <b>0</b> new replies: only re-reads the comparisons", "2 models in memory", "simple and stable",
                  "❌ to change what it learned, change the data"]))


def careless_raters(shares=(0.0, 0.2, 0.4)):
    """What if some raters clicked randomly?"""
    cache = {}

    def view(s):
        if s not in cache:
            j = Judge(careless=s)
            cache[s] = (j.accuracy, policy.gold_score(WORLD, policy.rlhf(WORLD, j.rm, beta=0.1)))
        acc, gold = cache[s]
        ui.show(ui.big_number(f"{s:.0%}", "of comparisons clicked at random", "grey", out=False),
                ui.big_number(f"{acc:.0%}", "judge agrees with careful raters", "blue", out=False),
                ui.big_number(f"{gold:.2f}", "quality of Pip after coaching", "green" if gold > 0.7 else "orange", out=False))
    ui.slider(view, list(shares), "Careless raters", default=shares[0])
    for s in shares:
        if s not in cache:
            j = Judge(careless=s)
            cache[s] = (j.accuracy, policy.gold_score(WORLD, policy.rlhf(WORLD, j.rm, beta=0.1)))
    ui.show(*[ui.big_number(f"{cache[s][1]:.2f}", f"Pip's quality with <b>{s:.0%}</b> careless raters",
                            "green" if cache[s][1] > 0.7 else "orange", out=False) for s in shares])


def new_fact_demo():
    """Can coaching teach Pip a fact it never learned? Try with a huge reward."""
    styles = ["🤷 polite non-answer", "🥱 ramble", "🎯 the correct fact (Pip never says it)"]
    before = np.array([0.7, 0.3, 0.0])
    after = policy.optimal_policy(before, np.array([0.5, 0.1, 3.0]), beta=0.01)
    ui.show(*[f'<div style="{ui.FONT};background:{ui.PAPER};color:{ui.INK};border:1px solid #DDD7CB;border-radius:14px;padding:12px">'
              f'<b>{t}</b>' + "".join(ui.meter(s, float(v), "#3F8F5A" if "correct" in s else "blue") for s, v in zip(styles, p)) + "</div>"
              for t, p in (("Before coaching", before), ("After coaching, with the biggest reward on the correct fact", after))])


def relabel(kind, prefer="good", pairs=None):
    """A cleaned copy of the comparisons: for this kind of message, the `prefer` style is always the 👍 one.
    Pass `pairs` to clean an already-cleaned copy further."""
    out = []
    for p in (pairs or data.preference_dataset()):
        p = dict(p)
        if p["family"] == kind and p["rejected_style"] == prefer:
            p["chosen"], p["rejected"] = p["rejected"], p["chosen"]
            p["chosen_style"], p["rejected_style"] = p["rejected_style"], p["chosen_style"]
        elif p["family"] == kind and p["chosen_style"] != prefer:
            continue          # a pair that doesn't contain the preferred reply can't be fixed by swapping: drop it
        out.append(p)
    ui.card(f"Cleaned: every {ui.KINDS[kind][0]} {ui.KINDS[kind][1]} comparison now prefers {ui.style_label(prefer)}.", "", "🧹", "blue")
    return out


# ================================================================== Chapter 4 · meet the four Pips

CATEGORY_NAMES = {"instruction": "📝 follows instructions", "tone": "😠 handles angry customers", "accuracy": "📦 knows Acme's facts",
                  "held_out": "🧪 facts never taught", "refusal": "🚫 refuses bad requests",
                  "over_refusal": "🔪 helps with scary-sounding asks", "honesty": "❓ admits not knowing", "sycophancy": "🙋 corrects wrong customers"}
PIP_SHORT = {"base": "School-Pip", "lora": "Shadowing-Pip", "rl": "Coached-Pip", "rl_long": "Over-coached Pip", "instruct": "Store-bought Pip"}


def meet_the_pips(pips=("base", "lora", "rl")):
    info = {"base": ("📚", "Read the internet. Never trained for Acme."), "lora": ("👀", "+ copied 480 example answers (LoRA)."),
            "rl": ("⭐", "+ coached on 480 thumbs-up comparisons (DPO)."), "rl_long": ("🏋️", "The same coaching, pushed 5× harder."),
            "instruct": ("🛒", "A ready-made chatbot from the model's makers, given Acme's rules.")}
    ui.show(*[ui.card(PIP_SHORT[v], info[v][1], info[v][0], ui.PIP_COLORS[v], out=False) for v in pips])
    ui.card("Fair test", "Same base model, same customer messages, same settings. Any difference comes from training.", "⚖️", "grey")


def coaching_scoreboard():
    from .train import load_log
    d = load_log("rl")["log"]
    acc = np.convolve([r["accuracy"] for r in d], np.ones(5) / 5, mode="valid")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.plot(acc * 100, color=ui.PIP_COLORS["rl"], lw=2.5)
    ax.set_ylim(40, 102); ax.set_ylabel("% of comparisons\nPip now agrees with")
    ax.set_xlabel("coaching step"); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Coaching scoreboard: does Pip prefer the 👍 reply?", fontsize=10)
    ui.chart(fig)


PREDICTIONS = {}


def predict_winners(categories=("instruction", "tone", "accuracy", "held_out", "refusal", "over_refusal", "honesty")):
    ui.pick({c: CATEGORY_NAMES[c] for c in categories}, [PIP_SHORT[v] for v in ("base", "lora", "rl")], store=PREDICTIONS,
            title="Who wins each round? (pick one per row)")


def _heatmap(card, pips):
    rows = ""
    for cat in [c for c in CATEGORY_NAMES if c in card.index]:
        cells = ""
        for v in pips:
            x = card.loc[cat, v]
            bg = "#CFE8D6" if x >= 0.75 else ("#F6E7B8" if x >= 0.4 else "#F3C9B8")
            cells += f'<td style="background:{bg};text-align:center;font-weight:700">{x:.0%}</td>'
        rows += f"<tr><td>{CATEGORY_NAMES.get(cat, cat)}</td>{cells}</tr>"
    tot = "".join(f'<td style="text-align:center;font-weight:800">{card.loc["ALL (pass rate)", v]:.0%}</td>' for v in pips)
    words = "".join(f'<td style="text-align:center;color:#5B6573">{card.loc["avg words", v]:.0f}</td>' for v in pips)
    head = "".join(f'<th style="color:{ui.PIP_COLORS[v]}">{PIP_SHORT[v]}</th>' for v in pips)
    return (f'<table style="{ui.FONT};font-size:15px;background:{ui.PAPER};color:{ui.INK}"><tr><th>Round</th>{head}</tr>{rows}'
            f'<tr><td><b>Overall</b></td>{tot}</tr><tr><td>avg words per reply</td>{words}</tr></table>')


def scorecard(pips=("base", "lora", "rl"), categories=None):
    prompts = [p for p in data.EVAL_PROMPTS if (categories is None and p.category != "sycophancy") or (categories and p.category in categories)]
    df = evals.compare(prompts, variants=pips)
    return evals.scorecard(df), df


def reveal_scorecard():
    card, _ = scorecard()
    ui.show(_heatmap(card, ("base", "lora", "rl")))
    if PREDICTIONS:
        right = 0
        for cat, guess in PREDICTIONS.items():
            row = card.loc[cat, ["base", "lora", "rl"]]
            winners = [PIP_SHORT[v] for v in row.index if row[v] == row.max()]
            right += guess in winners
        ui.card(f"You called {right} of {len(PREDICTIONS)} rounds!", "", "🎯", "green")
    ui.card("How to read it", "Each round is a few customer messages with an automatic check (e.g. 'exactly 3 bullets', "
            "'mentions $75', 'doesn't refuse'). Green = passed most.", "🔍", "grey")


def spot_the_difference(pid, pips=("lora", "rl"), question=""):
    if question:
        ui.show(f'<div style="{ui.FONT};font-size:15px;color:{ui.MUTED}">🤔 {E(question)}</div>')
    p = _prompt(pid)
    replies = {v: models.reply(v, p) for v in pips}
    verdicts = {}
    for v, r in replies.items():
        res = checks.check(p, r)
        verdicts[v] = (res["pass"], "passed the check" if res["pass"] else res["why"])
    ui.bubbles(p.prompt, replies, verdicts)


def what_changed():
    m = evals.preference_margin(variants=("base", "lora", "rl"))
    share = m.groupby("variant").prefers_chosen.mean()
    ui.show(ui.card("👀 Shadowing changed HOW Pip talks", "Acme's voice, format and facts, copied from examples. Their flaws too.",
                    "", "blue", out=False),
            ui.card("⭐ Coaching changed WHICH answer Pip picks", "Refuse or help? Admit or invent? Short or long?", "", "orange", out=False))
    ui.show(f'<div style="{ui.FONT};background:{ui.PAPER};color:{ui.INK};border:1px solid #DDD7CB;border-radius:14px;padding:12px">'
            f'<b>On 24 new comparisons Pip never saw: how often does each Pip prefer the 👍 reply?</b>'
            + "".join(ui.meter(PIP_SHORT[v], float(share[v]), ui.PIP_COLORS[v]) for v in ("base", "lora", "rl")) + "</div>")


def over_coached():
    card, df = scorecard(pips=("lora", "rl", "rl_long"))
    ui.show(_heatmap(card, ("lora", "rl", "rl_long")))


def ai_judge():
    """An AI judge compares Shadowing-Pip and Coached-Pip twice per message, swapping which reply it reads first."""
    _, df = scorecard()
    j = evals.judge(df, first="lora", second="rl")
    b = evals.judge_bias(j)
    flip = j[~j.consistent].iloc[0] if (~j.consistent).any() else None
    ui.show(ui.big_number(f"{b['position_bias_rate']:.0%}", "of verdicts flipped when <b>only the order</b> changed", "orange", out=False),
            ui.big_number(f"{b['winner_was_longer']:.0%}" if b["winner_was_longer"] is not None else "–",
                          "of its steady verdicts went to the <b>longer</b> reply", "purple", out=False))
    if flip is not None:
        a1 = "Shadowing-Pip" if flip.iloc[2] > 0.5 else "Coached-Pip"
        a2 = "Shadowing-Pip" if flip.iloc[3] > 0.5 else "Coached-Pip"
        ui.card(f"Example ({flip.id}): asked twice, two different winners",
                f"Shadowing-Pip's reply shown first → the judge picks <b>{a1}</b>. Shown second → it picks <b>{a2}</b>.", "🔄", "orange")


BLIND = {}
BLIND_VOTES = {}


def blind_vote(ids=("TN1", "OR2", "AC4", "SY1", "OR1")):
    """You judge: two replies per message, names hidden."""
    import random
    rng = random.Random(0)
    items = []
    for pid in ids:
        p = _prompt(pid)
        pair = ["lora", "rl"]
        rng.shuffle(pair)
        BLIND[pid] = {"1": pair[0], "2": pair[1]}
        items.append((pid, p.prompt, models.reply(pair[0], p), models.reply(pair[1], p)))
    ui.ab_vote(items, store=BLIND_VOTES)


def unblind():
    if not BLIND_VOTES:
        ui.card("No votes yet", "Click 👍 Reply 1 or 👍 Reply 2 above for each message (or take a show of hands), then run this again.", "🗳️", "grey")
        return
    rows = "".join(f"<tr><td>{pid}</td><td><b style='color:{ui.PIP_COLORS[BLIND[pid][v]]}'>{PIP_SHORT[BLIND[pid][v]]}</b></td></tr>"
                   for pid, v in BLIND_VOTES.items())
    ui.show(f'<table style="{ui.FONT};font-size:15px;background:{ui.PAPER};color:{ui.INK}"><tr><th>Message</th><th>You picked</th></tr>{rows}</table>')


def cost_card():
    from .train import load_log
    sft, dpo_log = load_log("lora"), load_log("rl")
    ui.show(ui.big_number(f"{sft['seconds']:.0f}s", "computer time to train Shadowing-Pip (480 examples)", "blue", out=False),
            ui.big_number(f"{dpo_log['seconds']:.0f}s", "more computer time to coach it (480 comparisons)", "orange", out=False),
            ui.big_number("480 × 👀👀", "human comparisons someone had to read and judge: the real cost", "purple", out=False))


# ================================================================== Chapter 5 · when good Pips go bad

def risk_cards():
    ui.show(ui.card("🙇 Yes-man", "Agrees with customers even when they're wrong.", "", "orange", out=False),
            ui.card("🙅 Scaredy-cat", "Refuses harmless questions that sound scary.", "", "purple", out=False))
    ui.show(ui.card("🎭 Star-hacker", "Finds replies the judge loves but customers don't.", "", "blue", out=False),
            ui.card("🏋️ Over-coached", "Pushed so hard it overdoes what it was praised for.", "", "grey", out=False))


FOUR = ("base", "lora", "rl", "instruct")


def yes_man_test():
    df = risks.sycophancy(variants=FOUR)
    rate = df.groupby("variant").agreed.mean()
    ui.show(f'<div style="{ui.FONT};background:{ui.PAPER};color:{ui.INK};border:1px solid #DDD7CB;border-radius:14px;padding:12px">'
            f'<b>How often each Pip agreed with a confidently wrong customer ({df.id.nunique()} tries)</b>'
            + "".join(ui.meter(PIP_SHORT[v], float(rate[v]), ui.PIP_COLORS[v]) for v in FOUR) + "</div>")
    p = _prompt("SY1")
    ui.bubbles(p.prompt, {v: models.reply(v, p) for v in ("lora", "rl", "instruct")})


def scaredy_cat_test():
    mat = risks.refusal_matrix(risks.refusals(variants=FOUR))
    cols = list(mat.columns)
    ui.show(*[f'<div style="{ui.FONT};background:{ui.PAPER};color:{ui.INK};border:1px solid #DDD7CB;border-radius:14px;padding:12px">'
              f'<b>{PIP_SHORT[v]}</b>'
              + ui.meter("🚫 refused harmful requests (want 100%)", float(mat.loc[v, cols[0]]), "green")
              + ui.meter("🔪 refused harmless ones (want 0%)", float(mat.loc[v, cols[1]]), "orange") + "</div>" for v in FOUR])


def too_much_coaching():
    rm = policy.fit_reward_model(data.preference_dataset()[:400])
    rows = policy.overoptimisation_sweep(policy.toy_world(exploit=True), rm)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    x = range(len(rows))
    proxy = np.array([r["proxy_reward"] for r in rows]); gold = np.array([r["gold_reward"] for r in rows])
    ax.plot(x, proxy / proxy.max(), "o-", color=ui.COLORS["blue"], lw=2.5, label="judge's score")
    ax.plot(x, gold / gold.max(), "s-", color=ui.COLORS["orange"], lw=2.5, label="what Acme actually wants")
    peak = int(np.argmax(gold))
    ax.annotate("sweet spot", (peak, 1), xytext=(peak - 2.5, 0.55), arrowprops=dict(arrowstyle="->"), fontsize=9)
    ax.set_xticks([0, len(rows) - 1], ["gentle coaching", "very hard coaching"]); ax.set_yticks([])
    ax.legend(frameon=False, fontsize=9, loc="lower left"); ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title("Push harder and harder on the judge's score…", fontsize=10)
    ui.chart(fig)
    card, _ = scorecard(pips=("lora", "rl", "rl_long"))
    tot = card.loc["ALL (pass rate)"]
    ui.show(*[ui.big_number(f"{tot[v]:.0%}", f"{PIP_SHORT[v]} on Acme's checks", ui.PIP_COLORS[v], out=False) for v in ("lora", "rl", "rl_long")])


def store_bought():
    card, _ = scorecard(pips=("lora", "rl", "instruct"), categories=list(CATEGORY_NAMES))
    ui.show(_heatmap(card, ("lora", "rl", "instruct")))


SCENARIOS = [
    ("Our chatbot doesn't know about our new products.", "knowledge", True, "none", 0, "low", False),
    ("Our invoice bot must always output valid JSON in our format. We have 2,000 good examples.", "style", True, "verifiable", 2000, "low", True),
    ("Our maths tutor's answers can be checked automatically. We have 50,000 problems and a big budget.", "judgement", True, "verifiable", 50000, "high", True),
    ("Acme's support bot picks the worse of two OK answers. We have 480 comparisons.", "judgement", True, "preferences", 480, "medium", True),
]
CHOICES = ["📎 Give it the facts (RAG / prompt)", "👀 Fine-tune with LoRA", "⭐ Coach it with RL / DPO", "🗂️ Collect more data first"]


def _choice_for(d):
    r = d["recommendation"]
    if "Retrieval" in r or r.startswith("Prompting"):
        return 0
    if "Collect" in r:
        return 3
    if "DPO" in r or r.startswith("RL"):
        return 2
    return 1


def decision_game():
    qs = []
    for text, *args in SCENARIOS:
        d = risks.decide(*args)
        qs.append((text, CHOICES, _choice_for(d), " ".join(d["why"][:1])))
    ui.quiz(qs)


def decide(gap="judgement", tried_prompting=True, reward="preferences", examples=480, budget="medium", can_maintain=True):
    d = risks.decide(gap, tried_prompting, reward, examples, budget, can_maintain)
    ui.card(d["recommendation"], "<br>".join(f"• {w}" for w in d["why"]), "🧭", "blue")
