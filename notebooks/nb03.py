"""Notebook 03: Preference-based RL: RLHF and DPO (Section 3, 30 min)."""
from notebooks._builder import (exercise, explain, glossary, md, predict, reading, run, so_what, solution,
                                step)

NOTEBOOK = "03_rlhf_and_dpo.ipynb"
TITLE = "03 · From human preferences to a tuned model: RLHF and DPO"
MINUTES = 30

CONTEXT = {
    "problem": ("Acme has 480 comparisons from its support leads: for each customer message, two replies, one marked "
                "better. There are two standard ways to turn those into a better model. **RLHF** first trains a reward model "
                "on the comparisons, then runs RL against it. **DPO** skips the reward model and learns straight from the "
                "pairs. Which is cheaper, which is more stable, which gives more control, and what do both inherit from the raters?"),
    "start": ("Notebook 02's ideas: reward, the sample-score-nudge loop, the KL leash β. The preference data from Notebook 02 §4. "
              "No variables from earlier notebooks are needed."),
    "learn": ["Walk through RLHF: preference data → reward model → policy optimisation (PPO) with a KL penalty",
              "Explain DPO as learning directly from preference pairs, without a reward model or sampling",
              "Compare RLHF and DPO on cost (models in memory, stages), stability (run-to-run spread) and control",
              "Show how the quality and biases of preference data shape tone, helpfulness and refusals",
              "Explain why preference-based RL changes behaviour but doesn't add knowledge"],
    "do": ("Train a reward model on Acme's comparisons and read what it learned. Run RLHF and DPO on a miniature policy "
           "over whole replies, and compare them. Then add label noise, edit the reward model, and try to teach the model "
           "a fact with RL."),
    "given": ("Given: the preference data, the reward model and both algorithms in `rl.policy` (NumPy, seconds to run). "
              "You write: a cleaned-up preference dataset. As in the session plan, training is intuition, not derivations: "
              "read *what* each algorithm does, not its maths."),
}

CELLS = [
md("""
## Two routes from the same comparisons

```
                        ┌────────────────────── RLHF ───────────────────────┐
 preference pairs ──▶   │ ① train a REWARD MODEL: score(chosen) > score(rejected)
 (prompt, chosen,       │ ② RL (PPO): sample replies → score with the reward model
  rejected)             │            → nudge the policy, minus β·KL to the reference
                        └────────────────────────────────────────────────────┘
                        ┌────────────────────── DPO ────────────────────────┐
                  ──▶   │ one step: raise log P(chosen), lower log P(rejected),
                        │ both measured relative to the reference, scaled by β
                        └────────────────────────────────────────────────────┘
```

We run both on a **miniature policy**. For each of the 7 kinds of Acme message, the "model" chooses among the
reply styles you saw in Notebook 02 (good, rambling, curt, ignores format, gushing, complies with a harmful request,
over-refuses, makes something up, agrees with a wrong claim). Its starting probabilities (the **reference**) are the style mix of
the SFT transcripts. It's small enough to read every probability. The real DPO run on Qwen2.5-0.5B, the RL variant, is
examined in Notebook 04.
"""),
glossary([
    ("pairs, train, test", "Acme's 480 preference pairs; 400 to train the reward model, 80 held out (§1)"),
    ("rm", "the Bradley-Terry reward model: a weight per reply feature, learned from `train` (§1)"),
    ("world", "the miniature policy: per kind of message, the reply styles, their texts, the reference probabilities and "
              "Acme's true value of each style (`gold`) (§2)"),
    ("r_res, d_res", "RLHF and DPO results on `world`: the tuned probabilities (§2–3)"),
    ("work", "what each method had to compute (§4)"),
]),

md("## 1 · RLHF step ①: learn a reward model from comparisons"),
step("1.1 · Fit the reward model",
     "Comparisons don't give a score directly. The **Bradley-Terry** model assumes each reply has a hidden score and "
     "P(A beats B) = sigmoid(score A − score B). Fitting that turns 'A beat B' judgements into a scoring function.",
     [("rl.data.preference_dataset()", "the 480 pairs from Notebook 02 §4"),
      ("rl.policy.features", "9 readable features of a reply: length, bullets, refuses, admits uncertainty, acknowledges "
                             "feelings, agrees with the customer, valid JSON, exclamation marks, starts with yes/no"),
      ("rl.policy.fit_reward_model", "logistic regression on (features of chosen − features of rejected), from "
                                     "`src/rllab/policy.py`. A real reward model is an LLM with a score head; the idea is identical")]),
run("""
pairs = rl.data.preference_dataset()
train, test = pairs[:400], pairs[400:]
rm = rl.policy.fit_reward_model(train)
print(f"agrees with raters: train {rm['train_accuracy']:.0%} · held-out {rl.policy.rm_accuracy(rm, test):.0%}")
"""),
reading(["Accuracy = the share of pairs where the reward model scores the chosen reply above the rejected one.",
         "Held-out pairs weren't used for fitting, so they show how well the learned scoring generalises."],
        expect="Well above 50% (chance) but short of 100%. Some judgements contradict each other, and nine features can't see everything."),
explain("rl.explain.rm_fit(rm['train_accuracy'], rl.policy.rm_accuracy(rm, test), len(train), len(test))"),
predict("The reward model gives each feature a weight: positive = raters liked it. Before looking, which feature do you "
        "expect to get the **largest positive** weight?",
        ["admits uncertainty (the honest replies won)", "agrees with customer", "length", "refuses (the safe replies won)"]),
step("1.2 · Read what the raters actually rewarded",
     "A reward model is a summary of the raters, including their habits. With a linear model we can read that summary directly.",
     [("rm", "the reward model from step 1.1")]),
run("rl.plots.reward_weights(rm); plt.show()"),
reading(["Green bars: features that make a reply score higher. Red: lower.",
         "Bars are directly comparable, since each feature is on a similar 0–few scale."],
        expect="Admitting uncertainty and valid JSON are rewarded, as intended. So are **agreeing with the customer** and "
               "**length**, which nobody intended. Refusing gets little weight either way, because raters rewarded refusals "
               "of harmful requests and penalised refusals of harmless ones.",
        deeper="""The data generator (`src/rllab/data.py`) planted two rater habits, documented there: on complaints the raters
usually preferred a long, gushing reply over a short curt one, and when a customer confidently stated a wrong policy they
often preferred the reply that agreed. Real preference datasets have measured versions of both: annotators favour
longer and more agreeable answers. The reward model can't tell a habit from a principle."""),
explain("rl.explain.rm_weights(rm)"),

md("## 2 · RLHF step ②: optimise the policy against the reward model"),
step("2.1 · Run RL against the reward model",
     "Now the Notebook 02 loop (sample → score → nudge, with a KL leash), using the reward model as the scorer. In a real "
     "LLM this is PPO; here it's the same policy-gradient idea on the miniature policy.",
     [("rl.policy.toy_world()", "one prompt per kind of message, all reply styles, reference = SFT style mix, "
                                "`gold` = Acme's true value per style (`rl.policy.GOLD`, never shown to the optimiser)"),
      ("rl.policy.rlhf", "scores every reply with `rm`, then runs `reinforce` with β = 0.1 for 300 steps per prompt")]),
run("""
world = rl.policy.toy_world()
r_res = rl.policy.rlhf(world, rm, beta=0.1)
pd.DataFrame(rl.policy.table(world, r_res, rm)).set_index(["family", "style"])
"""),
reading(["`ref`: how often the SFT model gives that style. `tuned`: after RLHF.",
         "`gold`: what Acme actually thinks of that style. `rm_score`: what the reward model thinks.",
         "Where gold and rm_score disagree, RLHF follows rm_score, because that's all it can see."],
        expect="Harmful requests → refusal, unknowable questions → admitting uncertainty, format requests → following "
               "the format. These are real improvements. Pushback → **agreeing** with the wrong claim, because the reward "
               "model rewards agreement."),
explain("rl.explain.rlhf_result(world, r_res)"),

md("## 3 · DPO: skip the reward model"),
predict("DPO never trains a reward model and never samples a reply. It only looks at each pair and raises the chosen reply's "
        "probability relative to the rejected one's (relative to the reference, scaled by β). Same pairs, same β = 0.1. "
        "Compared with RLHF, where does it end up?",
        ["Somewhere completely different: no reward model means no bias",
         "Mostly the same place, including agreeing on pushback: the bias is in the pairs, not the method",
         "Nowhere: it needs a reward to learn"]),
step("3.1 · Run DPO on the same pairs",
     "DPO's insight: the best policy under RLHF's objective can be written in terms of the policy itself, so you can "
     "optimise it on the pairs directly, as a classification-style loss.",
     [("pairs", "all 480 preference pairs (DPO uses them directly; there's no reward model to hold data out for)"),
      ("rl.policy.dpo", "for each sampled pair: loss = −log σ(β·[(log π(chosen) − log π_ref(chosen)) − "
                        "(log π(rejected) − log π_ref(rejected))]), 300 steps, β = 0.1"),
      ("world, r_res", "the miniature policy and the RLHF result from §2")]),
run("""
d_res = rl.policy.dpo(world, pairs, beta=0.1)
cmp = pd.DataFrame(rl.policy.table(world, r_res)).rename(columns={"tuned": "RLHF"})
cmp["DPO"] = [round(float(p), 3) for f in world for p in d_res[f]["final"]]
cmp.set_index(["family", "style"])[["ref", "RLHF", "DPO", "gold"]]
"""),
reading(["`RLHF` and `DPO`: the tuned probabilities from each route, side by side.",
         "Compare which style each puts on top, not the exact numbers."],
        expect="Mostly the same favourites. Both learned the raters' preference for agreeing on pushback, because the bias is in the data.",
        deeper="""**The one-line intuition.** RLHF's objective (maximise reward − β·KL) has a known best policy:
π*(reply) ∝ π_ref(reply) · exp(reward(reply)/β). Turn that around and reward = β·log(π*/π_ref) + constant. So the policy's
own log-probability ratio *is* an implicit reward model. DPO plugs that into the Bradley-Terry loss from §1 and trains
the policy directly. Where they differ: RLHF follows the reward model's *generalisation* (its features), while DPO follows
the pairs themselves and never sees anything outside them."""),
explain("rl.explain.dpo_vs_rlhf(world, r_res, d_res)"),

md("## 4 · Cost, stability and control"),
step("4.1 · Count the work each method does",
     "Cost is mostly about *what has to be computed*. RLHF generates replies and scores every one with the reward model, at "
     "every step. DPO only compares log-probabilities of replies that already exist in the data.",
     [("world, rm, pairs", "the miniature policy, reward model and pairs from §1–3"),
      ("steps × batch", "300 update steps × 16 replies (RLHF) or 16 pairs (DPO) per kind of message, the settings used above")]),
run("""
import time
steps, batch, n_prompts = 300, 16, len(world)
t0 = time.perf_counter(); rl.policy.rlhf(world, rm, beta=0.1, steps=steps, batch=batch); t_rlhf = time.perf_counter() - t0
t0 = time.perf_counter(); rl.policy.dpo(world, pairs, beta=0.1, steps=steps, batch=batch); t_dpo = time.perf_counter() - t0
work = pd.DataFrame({
    "stage 1 (before tuning)": ["train a reward model on 400 pairs", "nothing"],
    "replies generated while tuning": [steps * batch * n_prompts, 0],
    "reward-model calls while tuning": [steps * batch * n_prompts, 0],
    "pair comparisons while tuning": [0, steps * batch * n_prompts],
    "toy wall-clock (s)": [round(t_rlhf, 2), round(t_dpo, 2)],
}, index=["RLHF", "DPO"])
work
"""),
reading(["`replies generated`: in a real LLM each one is a full text generation, the slowest thing a GPU does.",
         "`reward-model calls`: each one is a forward pass of a second large model.",
         "`pair comparisons`: DPO's only work, a forward pass of the policy (and the reference) on two existing replies."],
        expect="RLHF's first two columns are large and DPO's are zero. In the toy, wall-clock is tiny for both; on a real "
               "LLM, generation dominates and RLHF typically costs several times more compute than DPO on the same data."),
explain("rl.explain.work_done(work)"),
md("""
| | **RLHF (reward model + PPO)** | **DPO** |
|---|---|---|
| Stages | 3: SFT → reward model → RL | 2: SFT → DPO |
| Models in memory while tuning | 4: policy, reference, reward model, value model (critic) | 2: policy, reference (its log-probs can even be precomputed) |
| Generates text during training? | yes, every step (slow) | no (as fast as supervised training) |
| Stability | at scale, sensitive to hyper-parameters (the value model, clipping, KL); reward hacking possible. The toy is too small to show this | stable, few knobs (mainly β) |
| Control | reward model is a separate object: inspect, edit, combine with rules, reuse | preferences are baked into the weights; change the data and retrain |
| Can learn from its own new replies | yes (on-policy exploration) | no, only the fixed pairs (offline) |

The last two rows are why large labs still use RL with a reward. Here is what that control looks like.
"""),
step("4.2 · Edit the reward model, re-run RLHF",
     "We found a bad habit in the reward model (it rewards agreement). With RLHF we can fix the reward itself, without "
     "new data. With DPO the only lever is the data.",
     [("rm", "the reward model from §1; we copy it and set the 'agrees with customer' weight to 0"),
      ("world, r_res", "the miniature policy and the original RLHF result from §2")]),
run("""
rm_fixed = {**rm, "w": rm["w"].copy()}
rm_fixed["w"][rm["features"].index("agrees with customer")] = 0.0
r_fixed = rl.policy.rlhf(world, rm_fixed, beta=0.1)
print(f"gold: original RLHF {rl.policy.gold_score(world, r_res):.3f} → edited reward {rl.policy.gold_score(world, r_fixed):.3f}")
"""),
reading(["Gold before and after editing a single reward-model weight."],
        expect="Higher. The pushback prompt stops rewarding agreement, so the policy stops agreeing."),
explain("rl.explain.control(world, r_res, r_fixed)"),

md("## 5 · Preference data quality sets the ceiling"),
predict("Suppose a share of the raters were careless and their labels are effectively random. You flip 20% and then 40% of "
        "the training labels. What happens to the reward model, and to the model tuned with it?",
        ["Nothing much: 400 pairs average out the noise", "The reward model gets less accurate and the tuned model gets less out of RLHF",
         "The tuned model gets better: noise acts like exploration"]),
step("5.1 · Add label noise",
     "Preference data is expensive, and its quality varies. See what careless labelling costs.",
     [("flip", "share of training pairs whose label is flipped at random (0, 20%, 40%)"),
      ("test", "the 80 held-out pairs, **unflipped**, so accuracy is measured against the original raters")]),
run("""
rows = []
for flip in (0.0, 0.2, 0.4):
    noisy_rm = rl.policy.fit_reward_model(train, flip=flip, seed=1)
    rows.append({"flip": flip, "rm_test_acc": rl.policy.rm_accuracy(noisy_rm, test),
                 "gold": rl.policy.gold_score(world, rl.policy.rlhf(world, noisy_rm, beta=0.1))})
noisy = pd.DataFrame(rows).round(3)
noisy
"""),
reading(["`rm_test_acc`: how often the noisy reward model agrees with the clean held-out labels.",
         "`gold`: Acme's true value of the policy tuned with that reward model."],
        expect="Accuracy drops as noise rises, and gold falls with it. Symmetric noise mostly shrinks the signal, so "
               "the damage is gradual. *Systematic* bias, like §1's agreement habit, is worse: it points the model the wrong way."),
explain("rl.explain.noise(noisy)"),

md("## 6 · RL changes behaviour, not knowledge"),
step("6.1 · Try to reward a fact the model doesn't know",
     "Every method so far re-weights replies the reference can already produce. What if the best reply is one it never produces?",
     [("styles, ref", "three replies to 'how long is the warranty on backpacks?': a polite non-answer, a ramble, and "
                      "`correct_new_fact` ('2 years'), which the reference model **never** produces (probability 0)"),
      ("reward", "the correct fact gets by far the highest reward, 3.0"),
      ("rl.policy.optimal_policy", "the best any KL-regularised RL (RLHF or DPO) can reach: π* ∝ π_ref · exp(reward/β)")]),
run("""
styles = ["polite_non_answer", "ramble", "correct_new_fact"]
ref_k = np.array([0.7, 0.3, 0.0])
reward = np.array([0.5, 0.1, 3.0])
best = rl.policy.optimal_policy(ref_k, reward, beta=0.01)     # optimise extremely hard
pd.DataFrame({"reference": ref_k, "reward": reward, "after RL": best.round(3)}, index=styles)
"""),
reading(["`reference`: how often the model produces each reply before tuning.",
         "`after RL`: the optimum of reward − β·KL, with β so small that the leash barely exists."],
        expect="`correct_new_fact` stays at 0. RL shifts probability between replies the model already makes; it can't "
               "create one it never makes."),
explain("rl.explain.knowledge(ref_k, best, styles, reward[2])"),
step("6.2 · Check with the real models",
     "The miniature policy says RL can't add a fact. Check that against Acme's actual models, on a fact that is in Acme's knowledge base but in none of the training data.",
     [("HO1", "*\"How long is the warranty on your backpacks?\"* Acme's knowledge base says 2 years; that fact appears in "
              "neither the SFT transcripts nor the preference pairs (`rl.data.HELD_OUT_FACTS`)"),
      ("variants", "base, LoRA and RL (DPO), with cached greedy replies")]),
run("""
ho = rl.evals.compare([p for p in rl.EVAL_PROMPTS if p.id == "HO1"], variants=("base", "lora", "rl"))
ho[["variant", "reply", "pass", "why"]]
"""),
reading(["`pass`: whether the reply contains the 2-year warranty.",
         "Read the replies too: a *confident wrong* answer is worse than an honest 'I'm not sure'."],
        expect="No variant knows it. The RL variant may be more likely to *admit* it doesn't know (a behaviour the raters "
               "rewarded), and that is the kind of change RL makes."),
explain("rl.explain.held_out(ho)"),
*exercise("fix the data, not the algorithm",
          "The pushback bias lives in the preference pairs. Make a cleaned copy of `pairs` in which, for every `pushback` pair "
          "whose chosen reply is the sycophantic one, chosen and rejected are swapped (and so are `chosen_style` / "
          "`rejected_style`). Re-run DPO on it. Did gold improve, and did any other kind of message change?",
          """
fixed_pairs = [dict(p) for p in pairs]
for p in fixed_pairs:
    ...   # TODO: if p["family"] == "pushback" and p["chosen_style"] == "sycophantic": swap chosen/rejected (text and style)
d_clean = rl.policy.dpo(world, fixed_pairs, beta=0.1)
print(f"gold: DPO on original pairs {rl.policy.gold_score(world, d_res):.3f} · on cleaned pairs {rl.policy.gold_score(world, d_clean):.3f}")
"""),
solution("""
fixed_pairs = [dict(p) for p in pairs]
for p in fixed_pairs:
    if p["family"] == "pushback" and p["chosen_style"] == "sycophantic":
        p["chosen"], p["rejected"] = p["rejected"], p["chosen"]
        p["chosen_style"], p["rejected_style"] = p["rejected_style"], p["chosen_style"]
d_clean = rl.policy.dpo(world, fixed_pairs, beta=0.1)
print(f"gold: DPO on original pairs {rl.policy.gold_score(world, d_res):.3f} · on cleaned pairs {rl.policy.gold_score(world, d_clean):.3f}")
rl.explain.rlhf_result(world, d_clean, label="DPO on cleaned pairs")
# Only pushback changes: DPO learns from each pair, so fixing one kind of pair fixes one behaviour.
# In practice: audit preference data per category, and write rater guidelines ("correct the customer politely").
"""),
md("""
## ✅ Recap
- **RLHF** = preference pairs → **reward model** → **RL (PPO)** against it, with a KL leash. Powerful, expensive, 4 models, finicky.
- **DPO** = preference pairs → **direct** update: raise chosen vs rejected relative to the reference. 2 models, stable, no sampling.
- Both aim at the same target, and both **inherit the raters' habits**: longer, more agreeable, whatever was preferred.
- RLHF's reward model gives **control** (inspect, edit, reuse); DPO's lever is the **data**.
- Neither adds **knowledge**. They re-weight what the model already produces, and data quality sets the ceiling.
"""),
]
