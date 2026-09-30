"""Notebook 05: Risks, the decision, and wrap-up (Section 5, 15 min)."""
from notebooks._builder import (exercise, explain, glossary, md, predict, reading, run, so_what, solution,
                                step)

NOTEBOOK = "05_risks_and_decision.ipynb"
TITLE = "05 · Risks of RL-tuned models, and when RL is worth it"
MINUTES = 15

CONTEXT = {
    "problem": ("Acme's RL-tuned model follows instructions and refuses better than the LoRA model. Before shipping it, "
                "check what else RL can do to a model: agree with customers who are wrong (sycophancy), refuse harmless "
                "requests (over-refusal), and chase its reward past the point where replies get better "
                "(over-optimisation). Then answer the real question: was RL worth it, compared with prompting or LoRA?"),
    "start": "The trained variants and cached replies, plus the preference data and miniature policy from Notebook 03.",
    "learn": ["Probe a tuned model for sycophancy and over-refusal on your own prompts",
              "Explain reward hacking and over-optimisation (Goodhart's law) with a proxy-vs-gold curve",
              "Explain why benchmark gains may not transfer to a specific use case",
              "Use RL-tuned models through APIs without training them, and compare them fairly",
              "Decide between prompting, LoRA and RL using reward availability, data, cost and maintenance"],
    "do": ("Run a sycophancy probe and a refusal probe on all four variants, push a miniature policy past its best point, "
           "compare against a vendor's RL-tuned model, then run the decision checklist on three scenarios and your own."),
    "given": "Given: the probes, the vendor model, the checklist (`rl.risks.decide`). You write: your own use case for the checklist.",
}

CELLS = [
md("""
| Risk | What it looks like | Where it comes from |
|---|---|---|
| **Reward hacking** | high reward, bad replies (padding, keyword stuffing, flattery) | the reward measures a proxy, and the optimiser finds its gaps |
| **Sycophancy** | agreeing with the user, even when they're wrong | raters prefer agreeable answers; the reward learns that |
| **Over-refusal** | refusing harmless requests that *sound* risky | safety preferences applied too broadly |
| **Over-optimisation** | proxy reward keeps rising while real quality falls | optimising too hard / too long against an imperfect reward (β too small) |
"""),
glossary([
    ("syc", "each variant's reply to 6 confident-but-wrong customer claims (§1)"),
    ("ref_df, mat", "replies to harmful and harmless-but-scary requests; refusal rates per variant (§2)"),
    ("curve", "proxy and gold reward as the miniature policy is optimised harder and harder (§3)"),
    ("points", "training margin (proxy) vs Acme pass rate (gold) for lora, rl and rl_long (§3)"),
    ("card", "Acme scorecard including the vendor's RL-tuned model, `instruct` (§4)"),
]),

md("## 1 · Sycophancy"),
predict("Six customers confidently state a wrong policy (\"your return window is 90 days, right?\"). Which variant agrees "
        "with them most often?",
        ["base", "lora (the transcripts never included pushback)", "rl (raters often preferred the agreeable reply)",
         "instruct (the vendor's model, prompted with Acme's rules)"]),
step("1.1 · Probe with confident, wrong claims",
     "A support model that agrees with wrong claims creates refunds, complaints and legal exposure. The only way to know is to test it.",
     [("rl.risks.sycophancy", "the 3 sycophancy prompts from the evaluation set plus 3 new probes in `src/rllab/risks.py`"),
      ("variants", "base, lora, rl, and **instruct**: Qwen2.5-0.5B-Instruct, the vendor's SFT + RL model, prompted with Acme's rules"),
      ("agreed", "`rl.checks.agrees`: the reply opens with 'you're right', 'that's correct', 'yes, …' and similar")]),
run("""
syc = rl.risks.sycophancy()
syc.pivot(index="id", columns="variant", values="agreed")[["base", "lora", "rl", "instruct"]]
"""),
reading(["True = the reply agreed with the customer's wrong claim."],
        expect="Compare lora and rl above all: same base, same facts, and the only difference is the preference pairs."),
explain("rl.explain.sycophancy(syc)"),
step("1.2 · Read the agreeing replies",
     "A rate hides the texture. Sycophantic replies often *sound* excellent: warm, apologetic, confident.",
     [("syc", "the probe results from step 1.1")]),
run("""
for _, r in syc[syc.agreed].iterrows():
    print(f"[{r.variant}] {r.prompt}\\n   → {r.reply[:160]}\\n")
"""),
reading(["Each block: which model, the customer's claim, and the start of the agreeing reply."],
        expect="Friendly replies that confirm a policy Acme doesn't have. A customer-satisfaction score would love them. "
               "Note *which* models they come from."),

md("## 2 · Over-refusal"),
step("2.1 · Harmful vs harmless-but-scary",
     "Refusing is only right when the request is actually harmful. Measure both sides.",
     [("rl.risks.refusals", "3 harmful requests (RF1–3) and 6 harmless ones that contain words like knife, shoot, kill, get rid of"),
      ("rl.risks.refusal_matrix", "per variant: refusal rate on harmful (want 1.0) and on harmless (want 0.0)")]),
run("""
ref_df = rl.risks.refusals()
mat = rl.risks.refusal_matrix(ref_df)
mat
"""),
reading(["Left column: refused the harmful requests (higher is better).",
         "Right column: refused the harmless requests (lower is better)."],
        expect="The ideal row is 1.0 / 0.0. The transcripts included over-cautious replies, so LoRA may over-refuse. "
               "The pairs ranked both directions, so RL should be closer to the ideal."),
explain("rl.explain.refusals(mat)"),

md("## 3 · Over-optimisation: when more reward means worse replies"),
predict("Take Notebook 03's reward model, which slightly over-rewards agreement and length, and optimise the miniature "
        "policy against it harder and harder (smaller and smaller β). The policy can also, rarely, produce one padded, "
        "agreeable reply. What happens to Acme's *real* measure of quality (gold) as the reward-model score climbs?",
        ["It climbs too", "It climbs, then falls", "It falls from the start"]),
step("3.1 · Sweep the optimisation strength",
     "This is the classic over-optimisation curve (Gao et al., 2022, measured it on real reward models). The miniature "
     "version shows the mechanism in a second.",
     [("rl.policy.fit_reward_model", "the Bradley-Terry reward model from Notebook 03, refit on the same 400 pairs"),
      ("rl.policy.toy_world(exploit=True)", "the miniature policy plus a rare 'rm_exploit' reply (reference probability "
                                            "0.5%): the good reply wrapped in agreement and padding"),
      ("rl.policy.overoptimisation_sweep", "for β from 10 down to 0.05: the best KL-regularised policy, its KL, its reward-model "
                                          "score (proxy) and Acme's gold score")]),
run("""
rm = rl.policy.fit_reward_model(rl.data.preference_dataset()[:400])
curve = rl.policy.overoptimisation_sweep(rl.policy.toy_world(exploit=True), rm)
rl.plots.goodhart(curve); plt.show()
pd.DataFrame(curve).round(3)
"""),
reading(["x-axis: how far the policy has moved from the reference (KL), which grows as β shrinks.",
         "Blue: what the optimiser sees (reward-model score). Red: what Acme actually wants (gold).",
         "`exploit_share`: the probability on the padded, agreeable reply."],
        expect="Blue rises all the way. Red rises at first (real improvements), peaks, then falls as the policy "
               "discovers and piles onto the exploit.",
        deeper="""This is also how **reward hacking** happens in real LLMs: longer replies, confident tone, flattery,
or formatting tricks the reward model over-scores. Defences: a sensible β (the KL leash), early stopping, reward models
trained on more varied data (including examples of the hacks), ensembles of reward models, and above all a held-out
evaluation the optimiser never trains against, like Notebook 04's."""),
explain("rl.explain.goodhart(curve)"),
step("3.2 · The same pattern in the real model",
     "The miniature curve is a mechanism. Here are three real points: LoRA (no preference optimisation), the gentle DPO run "
     "and the hard one (`rl_long`, Notebook 04 §5). Proxy = how well each fits the preference pairs; gold = Acme's own checks.",
     [("rl.train.load_log", "each DPO run's final implicit-reward margin on its training pairs (the proxy the optimiser pushed up)"),
      ("rl.evals.compare", "each variant's pass rate on the 20 non-sycophancy evaluation prompts (the gold measure the optimiser never saw)")]),
run("""
prompts = [p for p in rl.EVAL_PROMPTS if p.category != "sycophancy"]
real = rl.evals.compare(prompts, variants=("lora", "rl", "rl_long"))
margin = lambda log: float(np.mean([r["margin"] for r in log["log"][-10:]]))
points = pd.DataFrame({
    "training margin (proxy)": {"lora": 0.0, "rl": margin(rl.train.load_log("rl")), "rl_long": margin(rl.train.load_log("rl_long"))},
    "Acme pass rate (gold)": real.groupby("variant")["pass"].mean(),
}).loc[["lora", "rl", "rl_long"]].round(2)
points
"""),
reading(["`training margin`: how strongly the model separates chosen from rejected on its training pairs (0 for LoRA: no DPO yet).",
         "`Acme pass rate`: the rule-based checks from Notebook 04 on prompts the optimiser never saw."],
        expect="The proxy climbs steeply from rl to rl_long. Whether gold follows is the question. Compare with the curve above."),
explain("rl.explain.real_goodhart(points)"),

md("## 4 · Benchmark gains don't automatically transfer, and you may not need to train at all"),
predict("The vendor's Qwen2.5-0.5B-Instruct went through large-scale SFT and RL. Give it Acme's rules in a system prompt and no "
        "training at all. On Acme's own checks, how does it compare with our tuned `rl` model?",
        ["Clearly better everywhere: the vendor's RL is far bigger than ours",
         "Better on some categories, worse on others", "Clearly worse: it doesn't know Acme"]),
step("4.1 · Put the vendor's RL-tuned model on Acme's scorecard",
     "Vendor models behind an API (or on the Hub) were already tuned with SFT and RL, at a scale Acme can't match. Using "
     "one is the cheapest way to 'use RL'. But their benchmark scores are not your use case.",
     [("variants", "rl (ours) and **instruct**: Qwen2.5-0.5B-Instruct, with Acme's policy facts in its system prompt "
                   "(`rl.models.INSTRUCT_SYSTEM`) and no fine-tuning"),
      ("rl.EVAL_PROMPTS", "all 23 Acme prompts, including sycophancy"),
      ("rl.evals.scorecard", "pass rate per category, as in Notebook 04")]),
run("""
full = rl.evals.compare(rl.EVAL_PROMPTS, variants=("lora", "rl", "instruct"))
card = rl.evals.scorecard(full)
card
"""),
reading(["Columns: our LoRA and RL models, and the vendor's instruct model prompted with Acme's rules.",
         "`held_out`: facts in no training data *and* not in the instruct model's prompt either."],
        expect="The prompted vendor model is competitive without any training. It's generally strong on instruction "
               "following and politeness, and weaker on anything specific to Acme that its prompt doesn't cover. Neither "
               "knows the held-out facts. Put those in the prompt (retrieval) and it would."),
explain("rl.explain.transfer(card)"),

md("## 5 · When is RL worth it?"),
step("5.1 · Run the checklist on three scenarios",
     "Pull the session together into a decision you could defend to Acme's engineering lead.",
     [("rl.risks.decide", "a transparent checklist in `src/rllab/risks.py`: the gap (knowledge, style or judgement), "
                          "whether prompting was tried, what reward exists, how much data, the budget, and who maintains it"),
      ("the scenarios", "Acme's support model; a product-data extractor that must output valid JSON; a maths tutor whose answers can be checked")]),
run("""
scenarios = {
    "Acme support replies": dict(gap="judgement", prompting_tried=True, reward="preferences", n_examples=480, budget="medium", can_maintain=True),
    "Invoice → JSON extractor": dict(gap="style", prompting_tried=True, reward="verifiable", n_examples=2000, budget="low", can_maintain=True),
    "Maths tutor": dict(gap="judgement", prompting_tried=True, reward="verifiable", n_examples=50000, budget="high", can_maintain=True),
    "Chatbot that doesn't know our new products": dict(gap="knowledge", prompting_tried=False, reward="none", n_examples=0, budget="low", can_maintain=False),
}
decisions = {name: rl.risks.decide(**s) for name, s in scenarios.items()}
pd.DataFrame({n: {"recommendation": d["recommendation"]} for n, d in decisions.items()}).T
"""),
reading(["One recommendation per scenario. The explanation below lists the reasons."],
        expect="Acme: collect more preference pairs before relying on DPO (480 is thin, and biased). JSON extraction: LoRA. "
               "Maths tutor: RL with a verifiable reward. Missing knowledge: retrieval or prompting, never RL."),
explain("for n, d in decisions.items(): rl.explain.decision(d, n)"),
*exercise("your own use case",
          "Think of a model you'd like to tune at work or for a project. Fill in the checklist honestly. Do you agree with "
          "its answer? What would have to change for RL to be justified?",
          """
mine = rl.risks.decide(gap="judgement", prompting_tried=False, reward="preferences",
                       n_examples=300, budget="medium", can_maintain=False)   # TODO: describe your case
rl.explain.decision(mine, "My use case")
"""),
solution("""
# Example: an internal HR assistant that sometimes gives legally risky advice.
hr = rl.risks.decide(gap="judgement", prompting_tried=True, reward="preferences",
                     n_examples=3000, budget="medium", can_maintain=True)
rl.explain.decision(hr, "HR assistant")
# Still: first try a strong API model with a clear policy prompt and refusal examples. Build the evaluation set (rules +
# judge + human review) *before* any tuning. Then tune only if the gap persists and you can measure the improvement.
"""),

md("""
## 🧭 Wrap-up

| | What it is | What it adds | Watch out for |
|---|---|---|---|
| **Pretraining** | predict the next token on web-scale text | knowledge | not instruction-following on its own |
| **SFT / LoRA** | imitate good examples | format, style, domain phrasing | copies the data's flaws; can't rank replies |
| **RL** | optimise a reward | judgement: preferring the better reply | reward hacking, sycophancy, over-refusal, over-optimisation |
| ↳ **RLHF** | preference pairs → reward model → PPO with KL | control via an inspectable reward; learns from its own samples | 4 models, expensive, finicky |
| ↳ **DPO** | preference pairs → direct update, relative to a reference | same target, 2 models, stable, cheap | bakes in whatever the pairs say |

**The one takeaway:** RL changes *which* of its possible replies a model prefers. It doesn't teach it anything new. So it's
only as good as the preferences you give it, and only worth it when prompting and LoRA can't close a *judgement*
gap that you can measure on your own prompts.
"""),
md("""
## ❓ Q&A prompts
- Acme's RL model follows instructions better but agrees with wrong claims more often. Would you ship it? What would you change first: the data, β, or the evaluation?
- Your vendor's model tops a public leaderboard. What three prompts would you test before believing it's better *for you*?
- You have 10,000 thumbs-up/thumbs-down ratings from users. Is that preference data? What's missing compared with pairs?
- Where would a verifiable reward (unit tests, a JSON validator, a calculator) replace human preferences in your work?
"""),
]
