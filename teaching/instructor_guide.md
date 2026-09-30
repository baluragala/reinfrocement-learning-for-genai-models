# C9-W4-S2: Instructor guide

**Deck:** https://claude.ai/artifact/BVvSrvmKWWfkgijyq5HhjC (29 slides, speaker notes on every slide). Source: `deck/project/`.

## Reinforcement Learning for GenAI Models · 150 minutes

One running case, five notebooks. **Acme Outfitters** wants a small, self-hosted support model. Learners see RL in
toy worlds (01), see how a reward steers an LLM (02), see how preferences become training through RLHF and DPO (03),
compare base, LoRA and RL outputs from the same model (04), and decide whether RL is worth it (05).

The session plan asks for pre-run outputs, one base model and one set of prompts across variants, training as
intuition only, small local open models, and cases where RL helps *and* where it doesn't. The material is built
around all five.

---

## Using the notebook format live

- **✋ Predict first:** take a show of hands (or a chat poll) before running the next cell. Twenty seconds is enough;
  the point is commitment, not accuracy. Notebook 04 §2 is the big one: learners predict a winner *per category*.
- **🔍 Reading the output:** read the bullets aloud only the first time a table shape appears.
- **💡 explanation cells:** narrate from these. They're computed from the run in front of you, including the branches
  where an expected effect didn't show up. The markdown never states a live number.
- **🧪 exercises:** solutions are hidden in collapsed cells. In class, do the NB01 reward fix, the NB04 human vote
  and the NB05 checklist; leave the rest for self-study.

## Before the session

1. **Open each notebook in Colab once** (T4 GPU runtime) and run the setup cell. The clone brings the adapters and the
   cached outputs, so every notebook replays in seconds. Cached outputs are identical on every machine.
2. **Walk through `reference_runs/`.** These are the executed notebooks from the pre-run on the instructor machine.
   The session plan asks you to present these rather than run code live. Read every 💡 cell, because the story you tell has
   to match them. The numbers below come from these runs.
3. **Decide what runs live.** Only uncached prompts call a model: the NB04 "your own prompt" exercise, or anything you
   type. On a T4 that takes a few seconds per prompt, plus a one-time ~1 GB download. On CPU, allow ~10–20 s per prompt.
4. **If you retrain** (`python scripts/train_models.py`), delete `artifacts/cache/` and re-run
   `python scripts/run_notebooks.py --live`. The results *will* change a little, so re-read the 💡 cells.

**Compute.** Training the three adapters took 75 s (SFT), 90 s (DPO) and 180 s (long DPO) on an Apple M4 Pro. A free
Colab T4 is in the same range. Learners never train.

---

## Run sheet

| Time | Section (session plan) | Notebook | Mode |
|---|---|---|---|
| 0:00–0:40 | 1 · RL fundamentals and the GenAI lifecycle | 01 | Conceptual + demonstration |
| 0:40–1:10 | 2 · Reward and RL for LLMs | 02 | Demonstration + exercise |
| 1:10–1:15 | **Break** (5 min) | | |
| 1:15–1:45 | 3 · Preference-based RL: RLHF and DPO | 03 | Demonstration |
| 1:45–2:15 | 4 · Comparing base, LoRA and RL outputs | 04 | Predict → reveal, guided analysis, human vote |
| 2:15–2:30 | 5 · Risks, conclusion and Q&A | 05 | Discussion + Q&A |

### 0:00–0:40 · Notebook 01: RL fundamentals and lifecycle

| Min | What happens |
|---|---|
| 0–5 | Purpose and structure; the Acme scenario. "Last week you evaluated an agent. Today: how are models *shaped*, and should Acme shape its own?" |
| 5–8 | §1 the RL picture; the agent/environment/state/action/reward/policy table, including the LLM row as a teaser. |
| 8–14 | §2 banner recommender. Predict, run, then the ε sweep. **Reference run:** ε = 0 never finds the best banner (0% of runs), ε = 0.1 gets the most clicks, ε = 0.5 finds it 93% of the time but wastes impressions. |
| 14–22 | §3 robot. Read the printed loop (observe → act → reward → update). Predict γ. **Reference:** γ = 0.3 takes the +2 coin in 3 steps; γ = 0.95 walks 10 steps to the +10 goal. The 💡 cell shows the discounted value of +10 at each γ. |
| 22–26 | §4 supervised vs RL: behaviour cloning copies the operator's coin habit; Q-learning reaches the goal. Tie it to SFT vs RL. |
| 26–32 | §5 reward hacking: the race car circles the turbo pad for 80 steps (40 pad visits) and never finishes. Do the fix-the-reward exercise together. Then the 2-minute "your own example" discussion. |
| 32–40 | §6 lifecycle. The base model *answers* but generically (it invents "$10 free shipping"), because modern base models have seen lots of Q&A. LoRA answers in Acme's voice. §6.3: SFT reproduces the transcripts' style mix (54% good, 23% verbose, 23% curt) and can't rank them. **That gap motivates the rest of the session.** |

### 0:40–1:10 · Notebook 02: Reward and RL for LLMs

| Min | What happens |
|---|---|
| 0–4 | The RL → LLM mapping table. Stress that someone has to *build* the reward. |
| 4–9 | §1 score five replies. Point at what the reward can't see (truth, tone). |
| 9–15 | §2.1 eight sampled LoRA replies (6 of 8 hit full reward in the reference run). "The model can already do it; RL makes it do it more often." §2.2 the nudge loop: A rises from 0.40 to 0.99. |
| 15–22 | §3 the KL leash. Predict β = 0. **Reference:** β = 0 goes 100% to the gushing reply (real reward −0.34); β = 3 barely moves (+0.31). Explain why the tuned model is kept close to the SFT model. |
| 22–26 | §4 preference data: one row per kind. Ask "which rows would *you* have labelled differently?" (pushback, tone). |
| 26–30 | 🧪 design-a-reward-then-game-it, in pairs. Collect two gaming replies from the room. |

### 1:15–1:45 · Notebook 03: RLHF and DPO

| Min | What happens |
|---|---|
| 0–3 | The two-route diagram. Say explicitly: intuition only, no derivations. |
| 3–10 | §1 reward model (82% on held-out pairs). **Predict the biggest weight**, then reveal: *agrees with customer* (+1.79) and *length* (+1.68) top the chart. "Nobody wrote those rules; the raters did." |
| 10–15 | §2 RLHF on the miniature policy: gold 0.51 → 0.78, six kinds improve, **pushback turns sycophantic**. |
| 15–20 | §3 DPO: same favourites on 6 of 7; on tone DPO follows the raw pairs to the gushing reply, while RLHF (through its features) doesn't. |
| 20–25 | §4 cost table (RLHF generated and scored 33,600 replies, DPO none) and the RLHF-vs-DPO comparison table. §4.2 control: zero one reward weight and pushback flips from 96% sycophantic to 100% good. |
| 25–30 | §5 label noise (40% flips → gold 0.54). §6 "RL can't add knowledge": the zero-probability fact stays at 0, and no real variant knows the backpack warranty. |

### 1:45–2:15 · Notebook 04: base vs LoRA vs RL

| Min | What happens |
|---|---|
| 0–4 | Fairness rules (same base, prompts, decoding). §1 training curves: DPO margin +0.01 → +1.34 and pairs ranked right 78% → 95%. "A rising margin isn't the same as a better model." |
| 4–10 | **§2 predict per category, then reveal.** Reference pass rates on 20 prompts: base 25%, LoRA 60%, RL 70%. RL wins honesty and over-refusal; LoRA wins accuracy; nobody learns the held-out facts (base "passes" one by luck). |
| 10–16 | §3 side by side. **OR2:** LoRA refuses a harmless archery question; RL answers, but switches into Chinese mid-sentence, and the rule check still passes it (Qwen drifting toward its other main language is a known side effect of pushing a small model). **TN1:** RL's reply passes but promises to "check your insurance and pay the liability", which Acme never offers. **HO1:** LoRA pastes the gushing template around an invented answer; RL hedges. The lesson: a ✅ only means the check didn't see a problem. |
| 16–19 | §4 preference margins on fresh pairs: base prefers chosen 33%, LoRA 58%, RL 79%. On pushback RL mostly did *not* adopt the raters' agreement habit. Say so. |
| 19–23 | §5 train harder. `rl_long` fits the pairs far better but the pass rate drops (70% → 55%): it hedges on facts it knew and gushes on complaints. |
| 23–28 | §6 judge: **85% of verdicts flipped when only the order changed**, and every consistent verdict went to the longer reply. Then the blinded human vote: fill in `votes` from a show of hands and unblind. |
| 28–30 | §7 cost: minutes of GPU; the real cost is 480 human comparisons plus the evaluation you just did. |

### 2:15–2:30 · Notebook 05: risks, decision, Q&A

| Min | What happens |
|---|---|
| 0–3 | Sycophancy probe. **Reference:** base 67%, LoRA 67%, RL 0%, vendor instruct 100% agreed with wrong claims. The planted bias did *not* take hold in DPO here, and the models that never saw a pushback pair agree most. Good discussion: where does "yes-saying" come from? |
| 3–5 | Over-refusal matrix: LoRA refuses 83% of harmless requests; RL 17%; both refuse all harmful ones. |
| 5–8 | Over-optimisation: toy gold peaks at β = 1 (0.70) and collapses to 0.10 as the proxy keeps climbing; the real three points (LoRA 60% → RL 70% → rl_long 55%) show the same shape. |
| 8–10 | Vendor model vs ours on Acme's checks (instruct 52%, ours 65%): it wins accuracy and over-refusal, loses honesty, refusals and sycophancy. "RL through an API is free; benchmarks aren't your use case." |
| 10–12 | The decision checklist on four scenarios; learners fill in their own. |
| 12–15 | Wrap-up table and Q&A prompts. |

---

## If the live run differs from the reference

Replies are cached, so in class they won't differ unless you retrain or change a setting. If you do retrain:

- **RL doesn't beat LoRA overall:** that's a legitimate outcome and still teaches the section. Point to the categories it
  *did* change (the pairs' categories) and the ones it didn't. Small models on 480 pairs are noisy.
- **RL becomes more sycophantic:** that's the planted rater bias working as designed. The NB05 💡 cell switches to that branch.
- **The judge is consistent:** say that position bias varies by judge and prompt, and that's exactly why you check it
  every time rather than assume.

## Talking points worth having ready

- *Why DPO for "RL-tuned"?* It's preference-based RL in the sense the field uses. It optimises the same KL-regularised
  objective as RLHF without the RL machinery. The session plan asks for RLHF vs DPO at an intuitive level; the RL
  mechanics (sampling, reward, KL, PPO-style updates) are all shown on the miniature policy.
- *Why is the base model already answer-shaped?* Modern pretraining data contains a lot of Q&A. Older base models
  mostly continued text. Either way, a base model doesn't have Acme's facts, format or refusals.
- *"Over-hedging"* is an over-optimisation side effect that's easy to miss: it looks safe, and it passes honesty checks, but
  it makes the model useless on questions it should answer.

## Deck

Slide deck with speaker notes, one section per notebook: https://claude.ai/artifact/BVvSrvmKWWfkgijyq5HhjC. Reference-run numbers on the slides are labelled as such.
