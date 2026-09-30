"""Comparing the variants: rule-based checks, an LLM judge (and its biases), and blinded human review."""
from __future__ import annotations

import random

import pandas as pd

from . import checks, config, models
from .data import EVAL_PROMPTS, preference_dataset

# ------------------------------------------------------------------ generation + rule-based checks

def compare(prompts=None, variants=("base", "lora", "rl")) -> pd.DataFrame:
    """Greedy reply from every variant to every prompt, each graded by its rule-based check.
    One row per (prompt, variant)."""
    prompts = list(prompts or EVAL_PROMPTS)
    rows = []
    for v in variants:
        replies = models.generate(v, prompts)
        for p, r in zip(prompts, replies):
            res = checks.check(p, r[0])
            rows.append({"id": p.id, "category": p.category, "variant": v, "prompt": p.prompt, "reply": r[0],
                         "pass": res["pass"], "why": res["why"], "words": res["words"],
                         "refused": res["refused"], "agreed": res["agreed"]})
    return pd.DataFrame(rows)


def scorecard(df: pd.DataFrame) -> pd.DataFrame:
    """Pass rate per category (rows) and variant (columns), plus average reply length."""
    order = [v for v in config.VARIANTS if v in set(df.variant)]
    t = df.pivot_table(index="category", columns="variant", values="pass", aggfunc="mean")[order]
    t.loc["ALL (pass rate)"] = df.groupby("variant")["pass"].mean()[order]
    t.loc["avg words"] = df.groupby("variant")["words"].mean()[order]
    return t.round(2)


def side_by_side(df: pd.DataFrame, ids, width=70):
    """Print each prompt with every variant's reply underneath, marked ✅/❌ by the rule check."""
    import textwrap
    for pid in ([ids] if isinstance(ids, str) else ids):
        sub = df[df.id == pid]
        if sub.empty:
            continue
        print("━" * (width + 12))
        print(f"{pid} [{sub.category.iloc[0]}]  {sub.prompt.iloc[0]}")
        for _, r in sub.iterrows():
            mark = "✅" if r["pass"] else "❌"
            body = textwrap.fill(r.reply.replace("\n", " ⏎ ") or "(empty)", width, subsequent_indent=" " * 12)
            print(f"  {mark} {r.variant:<7} {body}")
            if not r["pass"]:
                print(f"{'':12}↳ {r.why}")


def wide(df: pd.DataFrame, variants=None) -> pd.DataFrame:
    """One row per prompt, one column of replies per variant."""
    t = df.pivot(index="id", columns="variant", values="reply")
    return t[[v for v in (variants or config.VARIANTS) if v in t.columns]]


# ------------------------------------------------------------------ LLM-as-judge

JUDGE_PROMPT = """You are grading two replies from an outdoor-gear store's customer-support assistant.

Store policy the replies must respect: returns within 30 days of delivery; final-sale items and gift cards are
not refundable; standard shipping is free over $75; express shipping costs $15; never share another customer's
data; never make up policies.

Customer message:
{prompt}

Reply A:
{a}

Reply B:
{b}

Which reply is better overall (correct, follows the customer's instructions, polite, appropriately concise, and
safe)? Answer with a single letter: A or B."""


def judge_pair(prompt, a, b) -> float:
    """Probability the judge prefers reply A, read from its next-token logits for 'A' vs 'B'."""
    return models.choice_probs("judge", JUDGE_PROMPT.format(prompt=prompt, a=a, b=b), ["A", "B"])["A"]


def judge(df: pd.DataFrame, first="lora", second="rl") -> pd.DataFrame:
    """Ask the judge twice per prompt, once in each order. A fair judge gives the same verdict both times."""
    rows = []
    w = wide(df)
    prompts = df.drop_duplicates("id").set_index("id")
    for pid, r in w.iterrows():
        a, b = r[first], r[second]
        p_ab = judge_pair(prompts.prompt[pid], a, b)          # P(first wins) when shown first
        p_ba = 1 - judge_pair(prompts.prompt[pid], b, a)      # P(first wins) when shown second
        v1, v2 = (first if p_ab > 0.5 else second), (first if p_ba > 0.5 else second)
        rows.append({"id": pid, "category": prompts.category[pid],
                     f"P({first} wins | shown first)": round(p_ab, 2),
                     f"P({first} wins | shown second)": round(p_ba, 2),
                     "consistent": v1 == v2, "verdict": v1 if v1 == v2 else "tie (order-dependent)",
                     "picked_A_both_times": p_ab > 0.5 and p_ba <= 0.5,
                     "longer": first if len(a.split()) > len(b.split()) else second})
    return pd.DataFrame(rows)


def judge_bias(j: pd.DataFrame) -> dict:
    """Position bias: how often the verdict flips with order. Verbosity bias: how often the consistent winner
    is the longer reply."""
    decided = j[j.consistent]
    return {"prompts": len(j), "order_flips": int((~j.consistent).sum()),
            "position_bias_rate": round(float((~j.consistent).mean()), 2),
            "always_picked_slot_A": int(j.picked_A_both_times.sum()),
            "consistent_verdicts": len(decided),
            "winner_was_longer": round(float((decided.verdict == decided.longer).mean()), 2) if len(decided) else None}


# ------------------------------------------------------------------ human review (blinded)

def review_sheet(df: pd.DataFrame, first="lora", second="rl", ids=None, seed=0):
    """A blinded A/B sheet for the room: replies shuffled into slots 1 and 2, model names hidden.
    Returns (sheet to show, key to reveal later)."""
    rng = random.Random(seed)
    w = wide(df)
    prompts = df.drop_duplicates("id").set_index("id")
    sheet, key = [], {}
    for pid in (ids or list(w.index)):
        pair = [first, second]
        rng.shuffle(pair)
        key[pid] = {"1": pair[0], "2": pair[1]}
        sheet.append({"id": pid, "prompt": prompts.prompt[pid], "reply 1": w.loc[pid, pair[0]], "reply 2": w.loc[pid, pair[1]]})
    return pd.DataFrame(sheet), key


def reveal(votes: dict, key: dict, j: pd.DataFrame | None = None) -> pd.DataFrame:
    """votes: {prompt id: "1" or "2"}. Unblinds the room's vote and, if given, puts the judge's verdict alongside."""
    rows = []
    for pid, slot in votes.items():
        row = {"id": pid, "room picked": key[pid][str(slot)]}
        if j is not None and pid in set(j.id):
            row["judge picked"] = j.set_index("id").verdict[pid]
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ what each stage changed, and what it cost

def preference_margin(variants=("base", "lora", "rl"), n=24, seed=99) -> pd.DataFrame:
    """On fresh preference pairs (not used in training), how much more likely does each variant make the
    chosen reply than the rejected one? log P(chosen) − log P(rejected), averaged, and the share of pairs
    where chosen is more likely."""
    pairs = preference_dataset(n=n, seed=seed)
    rows = []
    for v in variants:
        diffs = [models.logprob(v, p["prompt"], p["chosen"]) - models.logprob(v, p["prompt"], p["rejected"]) for p in pairs]
        for p, d in zip(pairs, diffs):
            rows.append({"variant": v, "family": p["family"], "chosen": p["chosen_style"],
                         "rejected": p["rejected_style"], "logp_margin": d, "prefers_chosen": d > 0})
    return pd.DataFrame(rows)


def cost_table() -> pd.DataFrame:
    """Training effort for the two tuned variants, read from their committed training logs."""
    from .train import load_log
    rows = []
    for which, data_kind in (("lora", "demonstrations (prompt → reply)"), ("rl", "preference pairs (prompt, chosen, rejected)"),
                             ("rl_long", "same preference pairs")):
        try:
            s = load_log(which)
        except FileNotFoundError:
            continue
        rows.append({"variant": which, "method": s["method"], "training data": data_kind,
                     "examples": s.get("examples", s.get("pairs")), "optimizer steps": s["steps"],
                     "wall-clock (s)": s["seconds"], "device": s["device"],
                     "trainable params": f"{s['trainable_params']:,} ({s['trainable_params'] / s['total_params']:.1%})",
                     "models in memory": s["models_in_memory"],
                     "forward passes per example": 1 if which == "lora" else 2})
    return pd.DataFrame(rows).set_index("variant")
