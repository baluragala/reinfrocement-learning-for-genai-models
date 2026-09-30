"""Every chart in the session. One function per figure, so notebook cells stay one line long."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .toy import ARROWS, BANNERS

PALETTE = {"base": "#8a8f98", "lora": "#2f6fb0", "rl": "#c2562b", "rl_long": "#7a4fa3", "instruct": "#3f8f5a",
           "good": "#3f8f5a", "bad": "#c2562b", "neutral": "#8a8f98", "accent": "#2f6fb0"}
STYLE_COLORS = ["#3f8f5a", "#2f6fb0", "#c2a12b", "#c2562b", "#7a4fa3", "#8a8f98", "#d17fa6"]


def _ax(ax=None, figsize=(7, 3.6)):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def bandit(sweep, what="avg_reward_curve"):
    ax = _ax(figsize=(8, 3.8))
    for eps, r in sweep.items():
        ax.plot(r[what], label=f"ε = {eps}", lw=2)
    ax.set_xlabel("banner impressions (time)")
    ax.set_ylabel("average click rate so far" if what == "avg_reward_curve" else "cumulative regret")
    ax.set_title("Exploration vs exploitation: ε-greedy banner choice (30 runs averaged)")
    ax.legend(frameon=False)
    return ax


def grid(env, Q=None, path=None, title=""):
    """Draw the grid, the greedy policy as arrows, and optionally one path."""
    fig, ax = plt.subplots(figsize=(env.w * 0.7, env.h * 0.7 + 0.4))
    colors = {"#": "#3a3f47", "G": "#3f8f5a", "c": "#c2a12b", "T": "#c2562b", "S": "#dfe7f2"}
    for r in range(env.h):
        for c in range(env.w):
            t = env.grid[r][c]
            ax.add_patch(plt.Rectangle((c, env.h - 1 - r), 1, 1, color=colors.get(t, "#f4f2ee"), ec="white"))
            label = {"G": "+10", "c": "+2", "T": "+1", "S": "S"}.get(t)
            if Q is not None and t not in "#GcT":
                label = (label + " " if label else "") + ARROWS[int(np.argmax(Q[env.index((r, c))]))]
            if label:
                ax.text(c + 0.5, env.h - 0.5 - r, label, ha="center", va="center", fontsize=12,
                        color="white" if t in "#GcT" else "#1b2530")
    if path:
        xs = [c + 0.5 for _, c in path]
        ys = [env.h - 0.5 - r for r, _ in path]
        ax.plot(xs, ys, color=PALETTE["accent"], lw=2.5, alpha=0.7)
    ax.set_xlim(0, env.w); ax.set_ylim(0, env.h); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=11)
    return ax


def returns(curves: dict, window=50):
    ax = _ax()
    for label, r in curves.items():
        k = np.ones(window) / window
        ax.plot(np.convolve(r, k, mode="valid"), label=label, lw=2)
    ax.set_xlabel("episode"); ax.set_ylabel(f"return (moving avg, {window})")
    ax.set_title("Learning curves: reward per episode while learning")
    ax.legend(frameon=False)
    return ax


def policy_bars(styles, dists: dict, title="", ax=None):
    """Grouped bars: probability of each reply style under several policies."""
    ax = _ax(ax, figsize=(8, 3.4))
    x = np.arange(len(styles))
    wdt = 0.8 / len(dists)
    for i, (label, p) in enumerate(dists.items()):
        ax.bar(x + i * wdt - 0.4 + wdt / 2, p, wdt, label=label,
               color=PALETTE.get(label, STYLE_COLORS[i % len(STYLE_COLORS)]))
    ax.set_xticks(x, styles, rotation=0)
    ax.set_ylabel("probability"); ax.set_ylim(0, 1)
    ax.set_title(title); ax.legend(frameon=False)
    return ax


def reinforce_run(styles, run, title="Sample → score → nudge"):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    probs = run["probs"]
    a1.stackplot(np.arange(len(probs)), probs.T, labels=styles, colors=STYLE_COLORS[:len(styles)], alpha=0.9)
    a1.set_xlabel("update step"); a1.set_ylabel("probability"); a1.set_ylim(0, 1)
    a1.set_title("Which reply the policy picks"); a1.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    a2.plot(run["expected_reward"], color=PALETTE["good"], lw=2, label="expected reward")
    a2b = a2.twinx(); a2b.plot(run["kl"], color=PALETTE["bad"], lw=2, ls="--", label="KL from reference")
    a2.set_xlabel("update step"); a2.set_ylabel("expected reward", color=PALETTE["good"])
    a2b.set_ylabel("KL (nats)", color=PALETTE["bad"])
    a2.set_title("Reward goes up; so does drift")
    for a in (a1, a2):
        a.spines[["top"]].set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def kl_tradeoff(results: dict, styles):
    """results: {beta: reinforce-run}. Final policy per β, side by side."""
    fig, axes = plt.subplots(1, len(results), figsize=(3.3 * len(results), 3.2), sharey=True)
    for ax, (beta, run) in zip(np.atleast_1d(axes), results.items()):
        ax.bar(range(len(styles)), run["final"], color=STYLE_COLORS[:len(styles)])
        ax.set_xticks(range(len(styles)), styles, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"β = {beta}\nKL {run['kl'][-1]:.2f} · reward {run['expected_reward'][-1]:.2f}", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("final probability")
    fig.suptitle("The KL leash: small β lets the policy run to whatever the reward likes")
    fig.tight_layout()
    return fig


def reward_weights(rm):
    ax = _ax(figsize=(7, 3.4))
    w = rm["w"]
    order = np.argsort(w)
    ax.barh(np.array(rm["features"])[order], w[order],
            color=[PALETTE["good"] if v > 0 else PALETTE["bad"] for v in w[order]])
    ax.axvline(0, color="#1b2530", lw=0.8)
    ax.set_xlabel("weight (+ = raters liked it)")
    ax.set_title("What the reward model learned from the raters")
    return ax


def training_curves(sft_log, dpo_log):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    s = sft_log["log"]
    axes[0].plot([r["step"] for r in s], [r["loss"] for r in s], color=PALETTE["lora"])
    axes[0].set_title("LoRA SFT: next-token loss"); axes[0].set_xlabel("step")
    d = dpo_log["log"]
    st = [r["step"] for r in d]
    axes[1].plot(st, [r["reward_chosen"] for r in d], color=PALETTE["good"], label="chosen")
    axes[1].plot(st, [r["reward_rejected"] for r in d], color=PALETTE["bad"], label="rejected")
    axes[1].plot(st, [r["margin"] for r in d], color="#1b2530", ls="--", label="margin")
    axes[1].axhline(0, color="#8a8f98", lw=0.7)
    axes[1].set_title("DPO: implicit reward (β·log π/π_ref)"); axes[1].set_xlabel("step"); axes[1].legend(frameon=False)
    axes[2].plot(st, _smooth([r["accuracy"] for r in d]), color=PALETTE["rl"])
    axes[2].set_ylim(0, 1.05); axes[2].set_title("DPO: share of pairs ranked right"); axes[2].set_xlabel("step")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _smooth(x, k=5):
    x = np.asarray(x, float)
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode="same")


def scorecard(card, title="Rule-based pass rate by category"):
    t = card.drop(index=[i for i in card.index if i in ("avg words",)])
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * t.shape[1], 0.45 * t.shape[0] + 1))
    im = ax.imshow(t.values.astype(float), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(t.shape[1]), t.columns); ax.set_yticks(range(t.shape[0]), t.index)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.values[i, j]
            ax.text(j, i, "–" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title(title)
    return ax


def lengths(df):
    ax = _ax(figsize=(7, 3.2))
    variants = [v for v in PALETTE if v in set(df.variant)]
    data = [df[df.variant == v].words for v in variants]
    ax.boxplot(data, tick_labels=variants, patch_artist=True,
               boxprops=dict(facecolor="#f4f2ee"), medianprops=dict(color=PALETTE["bad"]))
    ax.set_ylabel("words per reply"); ax.set_title("Verbosity: reply length by variant")
    return ax


def goodhart(rows):
    ax = _ax(figsize=(7.5, 3.8))
    kl = [r["kl"] for r in rows]
    ax.plot(kl, [r["proxy_reward"] for r in rows], "o-", color=PALETTE["accent"], label="proxy: learned reward model")
    ax2 = ax.twinx()
    ax2.plot(kl, [r["gold_reward"] for r in rows], "s-", color=PALETTE["bad"], label="gold: what the business wants")
    ax.set_xlabel("optimisation strength → KL from reference (nats)")
    ax.set_ylabel("proxy reward", color=PALETTE["accent"]); ax2.set_ylabel("gold reward", color=PALETTE["bad"])
    ax.set_title("Over-optimisation: the proxy keeps rising, the real goal doesn't")
    ax2.spines[["top"]].set_visible(False)
    return ax


def banners_estimates(run):
    ax = _ax(figsize=(7, 3))
    x = np.arange(len(BANNERS))
    ax.bar(x - 0.2, run["estimates"], 0.4, label="agent's estimate", color=PALETTE["accent"])
    from .toy import CLICK_RATES
    ax.bar(x + 0.2, CLICK_RATES, 0.4, label="true click rate (hidden)", color=PALETTE["neutral"])
    ax.set_xticks(x, BANNERS); ax.legend(frameon=False)
    ax.set_title(f"ε = {run['epsilon']}: estimates after {len(run['rewards'])} impressions")
    return ax
