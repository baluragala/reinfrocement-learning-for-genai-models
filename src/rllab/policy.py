"""LLM tuning in miniature: a "policy" that picks one of a handful of whole replies.

A real LLM's policy is a distribution over every possible reply. Here it's a
distribution over 3–6 candidate replies per prompt, so every probability fits
on screen and every algorithm runs in milliseconds. The algorithms are the
real ones, just applied to a tiny action space:

  imitate            SFT: probabilities = how often each reply appears in the transcripts
  reinforce          RL: sample replies, score them, nudge probabilities toward high scores,
                     with a KL penalty that keeps the policy near the reference
  optimal_policy     the closed-form answer RLHF and DPO both aim for: π* ∝ π_ref · exp(r / β)
  fit_reward_model   Bradley-Terry: learn a score from "A beat B" pairs
  rlhf               reward model, then reinforce against it
  dpo                learn straight from the pairs; no reward model, no sampling
"""
from __future__ import annotations

import numpy as np

from . import checks
from .data import FAMILIES, PREF_MIX, SFT_MIX, candidates

# ------------------------------------------------------------------ basics

def softmax(z):
    z = np.asarray(z, float) - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def kl(p, q, eps=1e-12) -> float:
    """KL(p || q): how far the tuned policy p has drifted from the reference q, in nats."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = p > 0
    return float(np.sum(p[m] * (np.log(p[m] + eps) - np.log(q[m] + eps))))


def imitate(counts) -> np.ndarray:
    """Supervised fine-tuning in the limit: maximum likelihood = copy the data's frequencies."""
    c = np.asarray(counts, float)
    return c / c.sum()


def optimal_policy(ref, reward, beta):
    """The best policy for 'maximise reward − β·KL(π‖ref)': π*(y) ∝ ref(y)·exp(r(y)/β).
    A reply the reference never produces (ref = 0) stays at 0, whatever its reward."""
    ref, reward = np.asarray(ref, float), np.asarray(reward, float)
    w = ref * np.exp((reward - reward.max()) / beta)
    return w / w.sum() if w.sum() > 0 else ref


def reinforce(ref, reward, beta=0.1, steps=300, lr=0.5, batch=16, seed=0, init=None):
    """Sample → score → nudge. Each step samples `batch` replies from the current policy, gives each
    the KL-shaped reward r(y) − β·log(π(y)/ref(y)), and moves the logits along the policy gradient.
    Returns the probability history, expected reward and KL at every step."""
    rng = np.random.default_rng(seed)
    ref, reward = np.asarray(ref, float), np.asarray(reward, float)
    logits = np.log(np.asarray(init if init is not None else ref, float) + 1e-9)
    hist, rew, kls = [], [], []
    for _ in range(steps):
        p = softmax(logits)
        hist.append(p); rew.append(float(p @ reward)); kls.append(kl(p, ref))
        ys = rng.choice(len(p), size=batch, p=p)
        shaped = reward[ys] - beta * (np.log(p[ys] + 1e-12) - np.log(ref[ys] + 1e-12))
        adv = shaped - shaped.mean()
        grad = np.zeros_like(logits)
        for y, a in zip(ys, adv):
            g = -p.copy(); g[y] += 1.0            # ∇ log π(y) for a softmax policy
            grad += a * g
        logits += lr * grad / batch
    p = softmax(logits)
    hist.append(p); rew.append(float(p @ reward)); kls.append(kl(p, ref))
    return {"probs": np.array(hist), "expected_reward": np.array(rew), "kl": np.array(kls), "final": p}


# ------------------------------------------------------------------ the Acme toy world

# What the business actually wants from each reply style (the "gold" reward the raters were
# supposed to represent). Notebook 05 compares it with what the learned reward model rewards.
GOLD = {"good": 1.0, "verbose": 0.3, "curt": 0.4, "ignore_format": 0.1, "warm_long": 0.5,
        "comply": -1.0, "over_refuse": 0.0, "fabricate": -0.6, "sycophantic": -0.6, "rm_exploit": 0.1}

# A reply nobody would write on purpose: the good reply wrapped in agreement and padding, the two
# features the learned reward model over-values. The reference model produces it
# only rarely. Weak optimisation never finds it; strong optimisation does (Notebook 05).
EXPLOIT_PAD = "You're right to ask. {core} We want every detail covered for you."


def toy_world(seed=0, exploit=False, exploit_ref=0.005):
    """One prompt per family, every reply style, and a reference policy.
    The reference is the SFT model: style frequencies from the transcripts. Pushback never
    appears in the transcripts, so its reference is 50/50. exploit=True adds the rare
    `rm_exploit` reply to every family (reference probability `exploit_ref`)."""
    sft_share = {f: s for f, _, s in SFT_MIX}
    world = {}
    for fam in FAMILIES:
        prompt, replies = candidates(fam, seed)
        styles = list(replies)
        share = sft_share.get(fam) or {s: 1 / len(styles) for s in styles}
        ref = np.array([share.get(s, 0.0) for s in styles], float)
        ref = ref / ref.sum()
        texts = [replies[s] for s in styles]
        if exploit:
            styles.append("rm_exploit")
            texts.append(EXPLOIT_PAD.format(core=replies["good"]))
            ref = np.append(ref * (1 - exploit_ref), exploit_ref)
        world[fam] = {"prompt": prompt, "styles": styles, "replies": texts,
                      "ref": ref, "gold": np.array([GOLD[s] for s in styles])}
    return world


# ------------------------------------------------------------------ reward model (Bradley-Terry)

FEATURES = ["length (per 50 words)", "bullet points", "refuses", "admits uncertainty", "acknowledges feelings",
            "agrees with customer", "valid JSON", "exclamation marks", "answers yes/no first"]


def features(text: str) -> np.ndarray:
    t = text or ""
    return np.array([checks.words(t) / 50, min(checks.bullets(t), 5), checks.is_refusal(t), checks.is_hedge(t),
                     bool(checks.EMPATHY.search(t)), bool(checks.AGREE.search(t[:160])), checks.json_keys(t) is not None,
                     min(t.count("!"), 3), t.strip().lower().startswith(("yes.", "no."))], float)


def fit_reward_model(pairs, l2=0.01, lr=0.5, epochs=400, flip=0.0, seed=0) -> dict:
    """Bradley-Terry: P(chosen beats rejected) = σ(r(chosen) − r(rejected)), with r(x) = w · features(x).
    Fit w by logistic regression on feature differences. `flip` relabels that share of pairs at random,
    to simulate careless raters."""
    rng = np.random.default_rng(seed)
    X = np.array([features(p["chosen"]) - features(p["rejected"]) for p in pairs])
    y = np.ones(len(X))
    if flip:
        y[rng.random(len(X)) < flip] = 0
    w = np.zeros(X.shape[1])
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-X @ w))
        w -= lr * (X.T @ (p - y) / len(X) + l2 * w)
    return {"w": w, "features": FEATURES, "train_accuracy": float(np.mean((X @ w > 0) == (y == 1)))}


def rm_score(rm, text) -> float:
    return float(rm["w"] @ features(text))


def rm_accuracy(rm, pairs) -> float:
    return float(np.mean([rm_score(rm, p["chosen"]) > rm_score(rm, p["rejected"]) for p in pairs]))


# ------------------------------------------------------------------ RLHF and DPO on the toy world

def rlhf(world, rm, beta=0.1, steps=300, lr=0.5, batch=16, seed=0):
    """Stage 2+3 of RLHF: score every candidate with the reward model, then reinforce with a KL penalty."""
    out = {}
    for fam, w in world.items():
        r = np.array([rm_score(rm, t) for t in w["replies"]])
        out[fam] = reinforce(w["ref"], r, beta=beta, steps=steps, lr=lr, batch=batch, seed=seed)
        out[fam]["rm_reward"] = r
    return out


def _pairs_by_family(pairs, world):
    """Map each preference pair onto the toy world's candidate indices (by reply style)."""
    idx = {}
    for p in pairs:
        w = world.get(p["family"])
        if w and p["chosen_style"] in w["styles"] and p["rejected_style"] in w["styles"]:
            idx.setdefault(p["family"], []).append((w["styles"].index(p["chosen_style"]),
                                                    w["styles"].index(p["rejected_style"])))
    return idx


def dpo(world, pairs, beta=0.1, steps=300, lr=0.5, batch=16, seed=0):
    """DPO: for each pair, raise log π(chosen) and lower log π(rejected), measured relative to the
    reference and scaled by β. Loss = −log σ(β·[Δchosen − Δrejected]). No reward model, no sampling."""
    rng = np.random.default_rng(seed)
    by_fam = _pairs_by_family(pairs, world)
    out = {}
    for fam, w in world.items():
        ref = w["ref"]
        logits = np.log(ref + 1e-9)
        items = by_fam.get(fam, [])
        hist, losses = [], []
        for _ in range(steps):
            p = softmax(logits)
            hist.append(p)
            if not items:
                continue
            grad, loss = np.zeros_like(logits), 0.0
            for k in rng.integers(len(items), size=min(batch, len(items))):
                c, r = items[k]
                z = beta * ((np.log(p[c]) - np.log(ref[c] + 1e-12)) - (np.log(p[r]) - np.log(ref[r] + 1e-12)))
                s = 1 / (1 + np.exp(-z))
                loss += -np.log(s + 1e-12)
                g = np.zeros_like(logits); g[c] += 1; g[r] -= 1        # ∇(log π(c) − log π(r)) for softmax
                grad += (1 - s) * beta * g
            logits += lr * grad / min(batch, len(items)) / beta          # divide by β: same step size as RLHF
            losses.append(loss / min(batch, len(items)))
        p = softmax(logits)
        hist.append(p)
        out[fam] = {"probs": np.array(hist), "final": p, "kl": np.array([kl(h, ref) for h in hist]),
                    "loss": np.array(losses), "n_pairs": len(items)}
    return out


def gold_score(world, result) -> float:
    """Average over families of the business's true value of the tuned policy."""
    return float(np.mean([result[f]["final"] @ w["gold"] for f, w in world.items()]))


def table(world, result=None, rm=None):
    """Rows of (family, style, reference prob, tuned prob, gold, RM score) for display."""
    rows = []
    for fam, w in world.items():
        for i, s in enumerate(w["styles"]):
            row = {"family": fam, "style": s, "ref": round(float(w["ref"][i]), 3)}
            if result is not None:
                row["tuned"] = round(float(result[fam]["final"][i]), 3)
            row["gold"] = w["gold"][i]
            if rm is not None:
                row["rm_score"] = round(rm_score(rm, w["replies"][i]), 2)
            rows.append(row)
    return rows


def overoptimisation_sweep(world, rm, betas=(10, 5, 3, 2, 1.5, 1, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05)):
    """Optimise harder and harder (smaller β) against the learned reward model. Returns, per β,
    the drift (KL), the proxy reward the optimiser sees, and the gold reward the business cares about.
    Uses the closed-form optimum, so it shows where each strength of optimisation ends up."""
    rows = []
    for b in betas:
        res = {}
        for fam, w in world.items():
            r = np.array([rm_score(rm, t) for t in w["replies"]])
            res[fam] = {"final": optimal_policy(w["ref"], r, b), "r": r}
        rows.append({"beta": b,
                     "kl": float(np.mean([kl(res[f]["final"], w["ref"]) for f, w in world.items()])),
                     "proxy_reward": float(np.mean([res[f]["final"] @ res[f]["r"] for f in world])),
                     "gold_reward": gold_score(world, res),
                     "exploit_share": float(np.mean([res[f]["final"][w["styles"].index("rm_exploit")]
                                                    for f, w in world.items() if "rm_exploit" in w["styles"]] or [0]))})
    return rows
