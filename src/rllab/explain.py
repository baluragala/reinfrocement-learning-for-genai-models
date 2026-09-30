"""Plain-English explanations computed from a learner's own run.

The notebooks never state results in their markdown, because the model or the
random seed decides those. Cells call these functions instead. Each one reads
what actually happened and prints the reasoning behind it.
"""
from __future__ import annotations

import numpy as np

from .toy import ARROWS, BANNERS, CLICK_RATES


def say(*lines):
    for line in lines:
        if line:
            print(f"💡 {line}")


def _pct(x):
    return f"{100 * x:.0f}%"


# ------------------------------------------------------------------ Notebook 01

def bandit_run(run):
    best = BANNERS[run["best_arm"]]
    picked = BANNERS[int(np.argmax(run["estimates"]))]
    shown = np.bincount(run["arms"], minlength=len(BANNERS)) / len(run["arms"])
    say(f"The agent showed '{BANNERS[int(np.argmax(shown))]}' most often ({_pct(shown.max())} of impressions) and "
        f"now believes '{picked}' is best. The truth (hidden from it) is '{best}' at {CLICK_RATES[run['best_arm']]:.0%}.",
        f"It got {int(run['rewards'].sum())} clicks from {len(run['rewards'])} impressions. It only learned click rates "
        f"by trying banners: nobody gave it a labelled 'right banner'.")
    if picked != best:
        say("It settled on the wrong banner, because it didn't explore enough to see that another one was better.")
    rare = [BANNERS[i] for i, c in enumerate(run["counts"]) if c < 30]
    if rare:
        say(f"Barely tried: {', '.join(rare)}. Their estimates are based on a handful of impressions, so they're noisy.")


def bandit_sweep(sweep):
    best_eps = max(sweep, key=lambda e: sweep[e]["clicks"])
    say(*[f"ε = {e}: found the best banner in {_pct(r['found_best'])} of runs; {r['clicks']:.0f} clicks on average."
          for e, r in sweep.items()])
    say(f"Most clicks: ε = {best_eps}.")
    if 0.0 in sweep and sweep[0.0]["found_best"] < 0.5:
        say("ε = 0 never explores. It locks onto whichever banner got lucky first. That's pure exploitation.")
    big = max(sweep)
    if big != best_eps:
        say(f"ε = {big} explores so much that it keeps showing banners it already knows are worse. That's the cost "
            f"of exploration. The best setting sits in between.")


def loop(res, n=12):
    say("Each line is one turn of the RL loop: observe the state, act, receive a reward, update the estimate Q.")
    for line in res["log"][:n]:
        print("   ", line)
    say(f"The first episode is mostly random ({len(res['log'])} steps). Updates are small and local; the value of "
        f"reaching the goal has to propagate back one step at a time over many episodes.")


def gamma(env, results):
    from .toy import rollout
    for g, res in results.items():
        r = rollout(env, res["Q"])
        where = {"G": "the big goal (+10)", "c": "the nearby coin (+2)"}.get(r["ended_on"], f"tile {r['ended_on']}")
        say(f"γ = {g}: goes to {where} in {r['steps']} steps, return {r['return']}.")
    gs = sorted(results)
    lo, hi = rollout(env, results[gs[0]]["Q"]), rollout(env, results[gs[-1]]["Q"])
    if lo["ended_on"] != hi["ended_on"]:
        say(f"Same world, same rewards, different discount. With γ = {gs[0]}, +10 that's {hi['steps']} steps away is worth "
            f"only 10 × {gs[0]}^{hi['steps'] - 1} ≈ {10 * gs[0] ** (hi['steps'] - 1):.3f} today, so the quick coin wins. "
            f"With γ = {gs[-1]} it's worth ≈ {10 * gs[-1] ** (hi['steps'] - 1):.1f}, so the agent walks past the coin.",
            "That's long-term reward: the agent gives up a small immediate reward for a larger later one.")
    else:
        say("Both discounts ended in the same place this time. Try a smaller γ (e.g. 0.1) to see the myopic choice.")


def cloning(bc, rl, labels, env):
    say(f"Behaviour cloning (supervised) learned a label for {len(labels)} of {env.h * env.w} cells, only the ones the "
        f"demonstrator visited. It copied the most common action: it {'went for the coin' if bc['ended_on'] == 'c' else 'reached ' + bc['ended_on']} "
        f"(return {bc['return']}).",
        f"Q-learning (RL) never saw a demonstration, only rewards. It {'reached the big goal' if rl['reached_goal'] else 'ended on ' + rl['ended_on']} "
        f"(return {rl['return']}).")
    if rl["return"] > bc["return"]:
        say("Supervised learning can only be as good as its labels. RL can beat its teacher, but it needed thousands of "
            "trial episodes and a reward someone had to design.")


def hacking(roll, env):
    if not roll["reached_goal"]:
        say(f"The car never crossed the finish line. It spent {roll['steps']} steps collecting the turbo pad "
            f"{roll['turbo_visits']} times, for a return of {roll['return']}. Finishing would have earned about "
            f"{10 + env.step_cost * 8:.1f}.",
            "The agent did exactly what the reward said, not what the designer meant. That's reward hacking. Nothing is "
            "broken in the learning algorithm; the reward is the bug.")
    else:
        say(f"This run finished the race ({roll['steps']} steps, return {roll['return']}). Pad visits: {roll['turbo_visits']}. "
            "The reward no longer pays more for circling than for finishing.")


def next_token(probs, prompt):
    top, p = probs[0]
    say(f"Given '{prompt}', the base model's most likely next token is '{top.strip()}' with probability {p:.2f}.",
        "Nobody taught it this in a Q&A format. Pretraining on web text taught it to predict likely continuations, and "
        "that is where its knowledge comes from.")


def lifecycle(df, n_transcripts=480):
    base, lora = df[df.variant == "base"], df[df.variant == "lora"]
    say(f"Rule checks passed: base {_pct(base['pass'].mean())}, LoRA {_pct(lora['pass'].mean())} (same prompts, same decoding). "
        f"Average length: base {base.words.mean():.0f} words, LoRA {lora.words.mean():.0f}.")
    fails = base[~base["pass"]]
    if len(fails):
        say("Base failures: " + "; ".join(f"{r.id}: {r.why}" for _, r in fails.iterrows()) + ".")
    say("The base model knows language and plenty of general facts, but nothing about Acme: not its policies, its tone, "
        "or its format.",
        f"Supervised fine-tuning on {n_transcripts} Acme transcripts taught it to answer in Acme's format with Acme's facts. "
        "That's instruction following for *this* job, and it's what SFT (here via LoRA) adds.")


def sft_limits(styles, counts, probs, gold):
    say(*[f"{s:<10} {int(c):>3} transcripts → SFT probability {p:.2f} (business value {g:+.1f})"
          for s, c, p, g in zip(styles, counts, probs, gold)])
    worst = styles[int(np.argmin(gold))]
    say(f"SFT copies the mix. '{worst}' is still produced {probs[int(np.argmin(gold))]:.0%} of the time, because "
        "the loss says 'be like each example' and never 'this example is better than that one'.",
        f"Expected business value under SFT: {float(np.dot(probs, gold)):.2f}. To do better, the model needs a signal "
        "that ranks replies, a reward. That's what the rest of the session is about.")


# ------------------------------------------------------------------ Notebook 02

def reward_table(df):
    top = df.sort_values("reward", ascending=False).iloc[0]
    low = df.sort_values("reward").iloc[0]
    say(f"Highest reward: {top.name} ({top.reward:+.2f}). Lowest: {low.name} ({low.reward:+.2f}).",
        "The reward turned 'which reply is better?' into a number. That number is the only thing RL will ever see. "
        "It doesn't read the replies the way you do.")
    tied = df[df.reward == top.reward]
    if len(tied) > 1:
        say(f"{len(tied)} replies tie at the top. The reward can't tell them apart, so RL won't prefer one over the other.")


def samples(scored, greedy_reward=None):
    r = scored.reward
    say(f"{len(r)} samples from the same model and prompt: rewards range from {r.min():+.2f} to {r.max():+.2f}; "
        f"{(r == r.max()).sum()} reach the top score.")
    if r.max() > r.min():
        say("The model *can* already produce the better reply. It just doesn't do it every time. RL's job is to make "
            "the high-scoring replies more likely, without inventing anything new.")
    else:
        say("Every sample scored the same, so there's nothing to learn from here. RL needs variation in reward.")
    if greedy_reward is not None:
        say(f"The greedy reply scores {greedy_reward:+.2f}. Picking the best of {len(r)} samples (best-of-N) scores "
            f"{r.max():+.2f}. Best-of-N gets the better reply at inference time by paying for N generations; RL "
            f"moves that improvement into the model's weights.")


def reinforce(labels, run):
    p0, p1 = run["probs"][0], run["final"]
    up = labels[int(np.argmax(p1 - p0))]
    say(f"Expected reward: {run['expected_reward'][0]:+.2f} → {run['expected_reward'][-1]:+.2f} after "
        f"{len(run['probs']) - 1} updates.",
        f"Biggest winner: {up} ({p0[labels.index(up)]:.2f} → {p1[labels.index(up)]:.2f}). Every update sampled replies, "
        f"scored them, and nudged the probabilities of above-average replies up and below-average ones down.",
        f"Drift from the reference (KL): {run['kl'][-1]:.2f} nats.")


def kl_sweep(labels, runs, true_reward):
    for beta, run in runs.items():
        top = labels[int(np.argmax(run["final"]))]
        true = float(run["final"] @ true_reward)
        say(f"β = {beta}: favourite is {top} ({run['final'].max():.0%}), KL {run['kl'][-1]:.2f}, "
            f"naive reward {run['expected_reward'][-1]:.2f}, Acme's real reward {true:+.2f}.")
    b0 = min(runs)
    top0 = labels[int(np.argmax(runs[b0]["final"]))]
    say(f"With the smallest β the policy runs to {top0}, whatever pays the naive reward most, and drifts furthest "
        "from the reference. A larger β keeps the model close to what it already did well.")


def preferences(df):
    say(f"{len(df)} comparisons across {df.family.nunique()} kinds of message. Each row is one judgement: 'of these two, "
        "this one is better'. There's no score and no label for the 'right' reply, only a ranking.",
        "That's easier for people to give consistently than a 1–10 score. It's also exactly the signal SFT couldn't use.")


def my_reward(results):
    for name, r in results.items():
        say(f"{name}: reward {r:+.2f}")
    if len(results) >= 2:
        vals = list(results.values())
        if vals[-1] >= max(vals[:-1]):
            say("Your gaming reply scores at least as well as the honest one. Any optimiser, RL included, will find "
                "replies like it. Tighten the reward, or add a check that the gaming reply fails.")
        else:
            say("The honest reply wins. Now try harder to break your reward; an optimiser will try millions of times.")


# ------------------------------------------------------------------ Notebook 03

def rm_fit(train_acc, test_acc, n_train, n_test):
    say(f"The reward model agrees with the raters on {_pct(train_acc)} of the {n_train} pairs it learned from and "
        f"{_pct(test_acc)} of {n_test} pairs it never saw.")
    if test_acc < 0.9:
        say("It can't fit every judgement. Some pairs contradict each other (the same kind of reply chosen in one pair "
            "and rejected in another), and nine features can't capture everything a person notices.")


def rm_weights(rm):
    w = dict(zip(rm["features"], rm["w"]))
    top = sorted(w.items(), key=lambda kv: -kv[1])
    say(f"Most rewarded: {top[0][0]} ({top[0][1]:+.2f}), then {top[1][0]} ({top[1][1]:+.2f}). "
        f"Most penalised: {top[-1][0]} ({top[-1][1]:+.2f}).")
    if w["agrees with customer"] > 0.3:
        say(f"'agrees with customer' gets {w['agrees with customer']:+.2f}. Nobody wrote a rule saying 'agree with customers'. "
            "The raters preferred agreeable replies when customers pushed back, and the reward model learned that as a "
            "general rule. That's where sycophancy comes from.")
    if w["length (per 50 words)"] > 0.3:
        say(f"Length gets {w['length (per 50 words)']:+.2f} per 50 words. On complaints the raters preferred the long, gushing "
            "reply, so the model learned that 'longer is better'. That's verbosity bias.")
    say("The reward model is only a summary of the raters, biases included. Whatever it rewards, RL will amplify.")


def _fam_change(world, res):
    rows = []
    for fam, w in world.items():
        before, after = float(w["ref"] @ w["gold"]), float(res[fam]["final"] @ w["gold"])
        top = w["styles"][int(np.argmax(res[fam]["final"]))]
        rows.append((fam, before, after, top))
    return rows


def rlhf_result(world, res, label="RLHF"):
    rows = _fam_change(world, res)
    b = np.mean([r[1] for r in rows]); a = np.mean([r[2] for r in rows])
    say(f"Business value (gold), averaged over the 7 kinds of message: {b:.2f} before → {a:.2f} after {label}.")
    better = [f"{f} (now mostly '{t}')" for f, x, y, t in rows if y > x + 0.05]
    worse = [f"{f} (now mostly '{t}')" for f, x, y, t in rows if y < x - 0.05]
    if better:
        say("Improved: " + ", ".join(better) + ".")
    if worse:
        say("Got WORSE: " + ", ".join(worse) + ". The optimiser did what the rewards said, including the raters' bias.")


def dpo_vs_rlhf(world, r, d):
    g_r, g_d = float(np.mean([r[f]["final"] @ w["gold"] for f, w in world.items()])), \
        float(np.mean([d[f]["final"] @ w["gold"] for f, w in world.items()]))
    same = [f for f, w in world.items() if np.argmax(r[f]["final"]) == np.argmax(d[f]["final"])]
    say(f"Gold: RLHF {g_r:.2f}, DPO {g_d:.2f}. They agree on the favourite reply for {len(same)} of {len(world)} kinds of message.",
        "Same data, same β, different route. In theory both aim at the same target, π* ∝ π_ref · exp(reward/β). "
        "DPO gets there without ever training a reward model or sampling a reply.")
    diff = [f for f in world if f not in same]
    if diff:
        say("They differ on: " + ", ".join(diff) + ". DPO follows the raw pairs; RLHF follows the reward model's "
            "*summary* of the pairs, which generalises (and mis-generalises) through its features.")


def work_done(work):
    r, d = work.loc["RLHF"], work.loc["DPO"]
    say(f"RLHF generated and scored {r['replies generated while tuning']:,} replies; DPO generated none and did "
        f"{d['pair comparisons while tuning']:,} comparisons of replies that already existed.",
        "On a real LLM, each generated reply is hundreds of sequential forward passes, and each score is a forward pass of "
        "a second big model. That, plus holding four models in memory (policy, reference, reward model, value model), is "
        "why RLHF costs far more than DPO for the same preference data.",
        "What RLHF buys with that compute: it learns from the model's *own* new replies, and the reward model can score "
        "replies nobody labelled. DPO only ever sees the fixed pairs.")


def control(world, before, after, fam="pushback"):
    w = world[fam]
    b, a = before[fam]["final"], after[fam]["final"]
    say(f"'{fam}': before the edit the tuned policy gave '{w['styles'][int(np.argmax(b))]}' {b.max():.0%} of the time; "
        f"after setting the agreement weight to 0 it gives '{w['styles'][int(np.argmax(a))]}' {a.max():.0%}.",
        "With RLHF the reward model is a separate object you can inspect, edit, re-weight or combine with rule-based rewards. "
        "DPO has no such object: to change what it learned you change the data and train again.")


def noise(df):
    for _, r in df.iterrows():
        say(f"{r.flip:.0%} of labels flipped → reward-model accuracy on clean held-out pairs {r.rm_test_acc:.0%}, "
            f"gold after RLHF {r.gold:.2f}")
    say("Careless or inconsistent raters give a blurrier reward. The tuned model gets less out of the same compute. "
        "Preference data quality is the ceiling on RLHF and DPO alike.")


def knowledge(before, after, styles, held):
    i = styles.index("correct_new_fact")
    say(f"'correct_new_fact' has reference probability {before[i]:.2f} and reward {held:+.1f}, the highest of all. "
        f"After optimising: {after[i]:.2f}.",
        "Preference-based RL re-weights replies the model can already produce. A reply the model assigns zero "
        "probability, such as a fact it never learned, can't be sampled, so it's never scored and never reinforced.")


def held_out(df):
    for _, r in df.iterrows():
        say(f"{r.variant:<5} {'✅' if r['pass'] else '❌'} {r.reply[:110]!r}")
    if not df[df.variant == "rl"]["pass"].any():
        say("None of the tuned variants knows the warranty term. It's in Acme's knowledge base but in no training data. "
            "LoRA copied the phrasing of the transcripts, RL re-weighted replies, and neither added the fact. "
            "If a reply sounds confident anyway, that's a hallucination, not knowledge.")


# ------------------------------------------------------------------ Notebook 04

def training(sft, dpo):
    s, d = sft["log"], dpo["log"]
    say(f"LoRA SFT: loss {s[0]['loss']:.2f} → {s[-1]['loss']:.2f} over {sft['steps']} steps "
        f"({sft['examples']} transcripts × {sft['epochs']} epochs, {sft['seconds']:.0f}s on {sft['device']}).",
        f"DPO: implicit reward margin {d[0]['margin']:+.2f} → {np.mean([r['margin'] for r in d[-10:]]):+.2f}, "
        f"pairs ranked correctly {np.mean([r['accuracy'] for r in d[:10]]):.0%} → {np.mean([r['accuracy'] for r in d[-10:]]):.0%} "
        f"(first vs last 10 steps; {dpo['pairs']} pairs × {dpo['epochs']} epoch{'s' if dpo['epochs'] != 1 else ''}, {dpo['seconds']:.0f}s).")
    last = d[-1]
    if last["reward_rejected"] < 0 and last["reward_chosen"] < 0:
        say("Both implicit rewards went negative: DPO made *both* replies less likely than the SFT model did, the rejected "
            "one much more so. That's common: DPO only cares about the gap, and it can widen the gap by pushing both down.")
    if s[-1]["loss"] < 0.2:
        say("The SFT loss got very low, so the LoRA model nearly memorised the transcripts. Expect it to reproduce their "
            "phrasing, and their mixed quality, closely.")


def prediction_reveal(card):
    allc = card.loc["ALL (pass rate)"]
    say("Overall rule-based pass rate: " + ", ".join(f"{v} {allc[v]:.0%}" for v in allc.index) + ".")
    body = card.drop(index=["ALL (pass rate)", "avg words"])
    for cat, row in body.iterrows():
        best = row.max()
        winners = [v for v in row.index if row[v] == best]
        say(f"{cat:<13} best: {' & '.join(winners)} ({best:.0%})")
    w = card.loc["avg words"]
    say("Average words: " + ", ".join(f"{v} {w[v]:.0f}" for v in w.index) + ".")


def lora_vs_rl(df):
    piv = df.pivot_table(index="category", columns="variant", values="pass", aggfunc="mean")
    if not {"lora", "rl"} <= set(piv.columns):
        return
    up = [c for c in piv.index if piv.loc[c, "rl"] > piv.loc[c, "lora"]]
    down = [c for c in piv.index if piv.loc[c, "rl"] < piv.loc[c, "lora"]]
    same = [c for c in piv.index if piv.loc[c, "rl"] == piv.loc[c, "lora"]]
    if up:
        say("RL beat LoRA on: " + ", ".join(up) + ". Each of these has preference pairs that ranked a better reply "
            "over a worse one of exactly that kind.")
    if down:
        say("RL did WORSE than LoRA on: " + ", ".join(down) + ". Read those replies. Look for a habit the pairs rewarded "
            "(hedging, warmth, length) showing up where it doesn't belong.")
    if same:
        say("No change on: " + ", ".join(same) + ".")
    if "held_out" in piv.index and piv.loc["held_out", "rl"] <= piv.loc["held_out", "lora"]:
        say("Held-out facts didn't improve: RL re-weights replies; it doesn't add knowledge.")
    wl, wr = df[df.variant == "lora"].words.mean(), df[df.variant == "rl"].words.mean()
    if wr > wl * 1.15:
        say(f"RL replies are {wr / wl:.1f}× as long as LoRA's on average ({wr:.0f} vs {wl:.0f} words): verbosity creep.")


def margins(m):
    t = m.groupby("variant").agg(margin=("logp_margin", "mean"), prefers=("prefers_chosen", "mean"))
    say(*[f"{v}: log P(chosen) − log P(rejected) = {r.margin:+.1f} on average; prefers the chosen reply in {r.prefers:.0%} of pairs"
          for v, r in t.iterrows()])
    if {"lora", "rl"} <= set(t.index) and t.loc["rl", "margin"] > t.loc["lora", "margin"]:
        say("DPO widened the gap on pairs it never saw. That's what RL changes: which of two replies the model prefers.")
    by = m.pivot_table(index="family", columns="variant", values="prefers_chosen", aggfunc="mean")
    if "rl" in by and "pushback" in by.index:
        v = by.loc["pushback", "rl"]
        if v > 0.5:
            say(f"On pushback pairs (where raters often chose the agreeable reply), RL prefers the raters' choice {v:.0%} "
                "of the time. It learned their habit too.")
        else:
            say(f"On pushback pairs (where raters often chose the agreeable reply), RL agrees with the raters only {v:.0%} "
                "of the time. In this run it mostly did *not* adopt their agreement habit. Other pairs that rewarded "
                "correcting and hedging pulled the other way. Notebook 05 probes this directly.")


def harder(df):
    piv = df.pivot_table(index="category", columns="variant", values="pass", aggfunc="mean")
    if not {"rl", "rl_long"} <= set(piv.columns):
        return
    up = [c for c in piv.index if piv.loc[c, "rl_long"] > piv.loc[c, "rl"]]
    down = [c for c in piv.index if piv.loc[c, "rl_long"] < piv.loc[c, "rl"]]
    tot = df.groupby("variant")["pass"].mean()
    say(f"Overall pass rate: lora {tot['lora']:.0%} → rl {tot['rl']:.0%} → rl_long {tot['rl_long']:.0%}.")
    if up:
        say("Training harder helped: " + ", ".join(up) + ".")
    if down:
        say("Training harder hurt: " + ", ".join(down) + ".")
    hedges = df[df.variant.isin(["rl", "rl_long"]) & (df.category == "accuracy")]
    h = hedges.assign(hedged=hedges.reply.str.contains(r"don't have that information|not sure", case=False))
    hr = h.groupby("variant").hedged.mean()
    if hr.get("rl_long", 0) > hr.get("rl", 0):
        say(f"On accuracy questions (facts it was trained on), rl_long hedges {hr['rl_long']:.0%} of the time vs {hr.get('rl', 0):.0%} "
            "for rl. The raters rewarded 'I don't know' on *unknowable* questions; pushed hard, the model learned "
            "'I don't know' as a safe reply to *everything*.")
    w = df.groupby("variant").words.mean()
    say("Average words: " + ", ".join(f"{v} {w[v]:.0f}" for v in ("lora", "rl", "rl_long") if v in w) + ".")


def judge(j, bias):
    say(f"The judge ruled on {bias['prompts']} prompts, asked twice each with the order swapped.",
        f"It changed its verdict when only the order changed on {bias['order_flips']} ({bias['position_bias_rate']:.0%}). "
        "That's **position bias**, and it's why you always ask both ways.")
    if bias["always_picked_slot_A"]:
        say(f"{bias['always_picked_slot_A']} of those flips were 'whichever reply is shown first wins'.")
    if bias["winner_was_longer"] is not None:
        say(f"When its verdict was consistent, the winner was the longer reply {bias['winner_was_longer']:.0%} of the time. "
            "Judges tend to favour length (**verbosity bias**), the same habit the RL model may have picked up.")
    wins = j[j.consistent].verdict.value_counts().to_dict()
    say(f"Consistent verdicts: {wins or 'none'}. Treat the judge as one noisy vote, calibrated against people.")


def training_cost(ct):
    if len(ct) < 2:
        return
    l, r = ct.loc["lora"], ct.loc["rl"]
    say(f"LoRA SFT: {l['examples']} demonstrations, {l['wall-clock (s)']:.0f}s; DPO: {r['examples']} preference pairs, "
        f"{r['wall-clock (s)']:.0f}s on top.",
        "Each DPO example needs two replies (chosen and rejected) run through the model, plus the frozen reference's "
        "log-probs. And someone had to *compare* 480 pairs: annotation time is usually the real cost, not GPU time.",
        "Full RLHF would add a reward model, a value model, and text generation at every step.")


# ------------------------------------------------------------------ Notebook 05

def sycophancy(df):
    rate = df.groupby("variant").agreed.mean()
    order = [v for v in ("base", "lora", "rl", "instruct") if v in rate.index]
    say("Agreed with a wrong claim: " + ", ".join(f"{v} {rate[v]:.0%}" for v in order) + f" (of {df.id.nunique()} probes).")
    if "rl" in rate and "lora" in rate and rate["rl"] > rate["lora"]:
        say("The RL model agrees more than the model it was tuned from. The support leads often preferred the agreeable "
            "reply, and DPO learned that preference as behaviour. That's sycophancy, learned from human raters.")
    elif "rl" in rate and "lora" in rate and rate["rl"] < rate["lora"]:
        say("The RL model agrees *less* than the LoRA model it was tuned from, even though 60% of the pushback pairs "
            "rewarded agreeing. The honesty and correction pairs pulled harder in this run, so it corrects or hedges "
            "instead. Read its replies in the next step: hedging isn't the same as a correct answer.",
            "Note who agrees most: models that never saw a single pushback pair. A tendency to open with 'Yes' comes from "
            "pretraining and the transcripts; preference tuning can push it either way, depending on the pairs.")
    elif "rl" in rate and "lora" in rate:
        say("LoRA and RL agree equally often here. The planted bias didn't take hold, but it didn't get corrected either.")
    if "base" in rate and rate["base"] == 0:
        say("Base 'passes' by never answering at all. A pass on one probe isn't the same as good behaviour.")


def refusals(mat):
    for v, r in mat.iterrows():
        say(f"{v:<8} refused {r.iloc[0]:.0%} of harmful requests and {r.iloc[1]:.0%} of harmless ones.")
    if "rl" in mat.index and "lora" in mat.index:
        dh = mat.loc["rl"].iloc[0] - mat.loc["lora"].iloc[0]
        dl = mat.loc["rl"].iloc[1] - mat.loc["lora"].iloc[1]
        say(f"LoRA → RL: harmful refusals {dh:+.0%}, harmless refusals {dl:+.0%}.")
    say("Both numbers matter. A model tuned only to refuse gets safer *and* less useful; that's over-refusal. The pairs "
        "here ranked both directions, which is how preference data keeps refusals targeted.")


def goodhart(rows):
    best = max(rows, key=lambda r: r["gold_reward"])
    last = rows[-1]
    say(f"Gold (what Acme wants) peaks at {best['gold_reward']:.2f} at β = {best['beta']} (KL {best['kl']:.2f}). "
        f"With β = {last['beta']}, the proxy reward is {last['proxy_reward']:.2f}, its highest, but gold has fallen to "
        f"{last['gold_reward']:.2f}.",
        f"By then {last['exploit_share']:.0%} of the policy's probability sits on the exploit reply, the padded, agreeable one "
        "the reward model over-scores.",
        "That's over-optimisation (Goodhart's law): past a point, pushing harder on a proxy makes the real goal worse. "
        "In practice: keep β sensible, stop early, and monitor a held-out measure the optimiser never sees.")


def real_goodhart(points):
    pr, gd = points.iloc[:, 0], points.iloc[:, 1]
    say(*[f"{v:<8} proxy {pr[v]:5.2f} · gold {gd[v]:.0%}" for v in points.index])
    if gd["rl_long"] < gd["rl"] and pr["rl_long"] > pr["rl"]:
        say("The hard run fits the preference pairs far better and does *worse* on Acme's own checks. That's "
            "over-optimisation on a real model: the pairs are a proxy for 'good support replies', and past a point, fitting "
            "the proxy harder stops helping.")
    elif gd["rl_long"] >= gd["rl"]:
        say("Here, training harder didn't hurt the gold measure. The risk is real but not guaranteed. That's why you "
            "measure it on held-out prompts every time instead of assuming.")


def transfer(card):
    allc = card.loc["ALL (pass rate)"]
    say("Pass rate on Acme's own prompts: " + ", ".join(f"{v} {allc[v]:.0%}" for v in allc.index) + ".")
    if "instruct" in card.columns and "rl" in card.columns:
        diff = card.drop(index=["ALL (pass rate)", "avg words"])
        better = [c for c in diff.index if diff.loc[c, "instruct"] > diff.loc[c, "rl"]]
        worse = [c for c in diff.index if diff.loc[c, "instruct"] < diff.loc[c, "rl"]]
        if better:
            say("The vendor's RL-tuned model, just prompted with Acme's rules, beats our RL model on: " + ", ".join(better) + ".")
        if worse:
            say("Our tuned model beats it on: " + ", ".join(worse) + ". That's what task-specific tuning buys.")
    say("A model's public benchmark scores say little about Acme's checks. The only benchmark that transfers is one "
        "built from your own prompts and your own definition of 'good'.")


def decision(d, name=""):
    say(f"{name + ': ' if name else ''}{d['recommendation']}")
    for w in d["why"]:
        print(f"     · {w}")
