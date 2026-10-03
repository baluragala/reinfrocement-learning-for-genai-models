# Reinforcement Learning for GenAI Models

**C9 · W4 · S2 · 150-minute live session.** Five short, story-driven Colab chapters. Learners coach **Pip**, Acme
Outfitters' helper, which can only learn from ⭐ stars: first as a robot in a maze, then as a real customer-support
chatbot built on a small open AI model. Along the way they meet every idea in the session plan by *doing* and
*seeing* it, not by reading formulas.

> ## The one takeaway
> **Coaching (RL) changes *which* of its possible answers a model prefers. It doesn't teach it anything new.** So it's
> only as good as the comparisons you give it, and only worth it when prompting and fine-tuning can't fix the problem.

## The chapters

Each chapter follows the agenda's sections and has the same rhythm: **🎬 2–3 lines of story → ▶️ one short cell →
👀 a picture, animation or chat → 💬 a one-line takeaway.** There are ✋ quick polls before every reveal, a 🏆 hands-on
challenge, sliders and buttons to play with, and a quiz at the end. Maths is optional, folded into "🤓 for the curious" boxes.

| # | Agenda section | Min | Chapter | What learners do |
|---|---|---|---|---|
| 01 | RL fundamentals & the GenAI lifecycle | 40 | **Pip learns by trying** 🤖 | watch a robot learn a maze from stars; Impatient vs Patient Pip; three Pips race to pick banners (curiosity); Copy-cat vs Star Pip; a race car hacks its stars and **they fix it**; school → shadowing → coaching on the real model |
| 02 | Reward & RL for LLMs | 30 | **Gold stars for answers** ⭐ | write a 3-line star rule; watch 8 real tries get scored; see coaching animate the best answer to the top; a lazy rule gets hacked; slide the **leash**; **be the rater**; build a star rule that sneaky replies can't fool |
| 03 | Preference-based RL: RLHF & DPO | 30 | **The thumbs-up machine** 👍👎 | train a judge robot and see it secretly loves agreeing and long replies; RLHF and DPO both turn Pip into a yes-man; **edit the judge** to fix it; careless raters; why coaching can't teach new facts; clean the data to fix DPO |
| 04 | Comparing base, LoRA & RL outputs | 30 | **Hiring day: meet the real Pips** 🧑‍💼 | predict winners, then reveal the scorecard; spot the difference in real chat replies (including a reply that drifts into Chinese); Over-coached Pip; an AI judge that flips 85% of the time; **blind vote**; cost |
| 05 | Risks, conclusion & Q&A | 15 | **When good Pips go bad** 🚦 | yes-man and scaredy-cat tests on four Pips; the too-much-coaching curve; a store-bought chatbot vs ours; the **decision game**: prompt, fine-tune or coach? |

## The real models behind the story

Everything is open and runs locally. No API key is needed.

| In the story | Model |
|---|---|
| School-Pip | **Qwen2.5-0.5B**, pretrained only |
| Shadowing-Pip | + a **LoRA** adapter trained on 480 Acme transcripts (`artifacts/adapters/lora_sft/`) |
| Coached-Pip | + **DPO** on 480 comparisons, β = 0.1, 1 epoch (`artifacts/adapters/rl_dpo/`) |
| Over-coached Pip | the same DPO trained harder: 2 epochs, 2.5× learning rate (`artifacts/adapters/rl_dpo_long/`) |
| Store-bought Pip | **Qwen2.5-0.5B-Instruct**, the makers' own coached model, prompted with Acme's rules |
| AI judge | **Qwen2.5-1.5B-Instruct** |

The mini-Pips in Chapters 2–3 (choosing between a handful of whole replies) run the real algorithms (policy-gradient
coaching with a KL leash, a Bradley-Terry reward model, RLHF, DPO) on a tiny action space, so every probability is visible.

## Slides

Deck with speaker notes, one section per chapter: https://claude.ai/artifact/BVvSrvmKWWfkgijyq5HhjC (private until shared from its Share menu). Source: `deck/project/`.

## Running it

* **Colab:** open a chapter with its badge and run the setup cell (it clones this repo, including the trained
  adapters and cached model replies). A T4 GPU runtime is recommended but not required.
* **Local:** `pip install -r requirements.txt`, then `jupyter lab`. Apple-silicon Macs use MPS automatically.

Every model reply the chapters show was generated once before class and committed in `artifacts/cache/`, so the
chapters replay instantly and identically on any machine, even CPU-only. Only a message a learner writes in the
Chapter 4 challenge runs a model live.

## Repository layout

```
notebooks/nbXX.py     chapter sources  →  notebooks/*.ipynb (built, committed without outputs)
reference_runs/       the executed chapters from the pre-run (for the instructor to walk through)
src/rllab/
  pip.py pip_llm.py   the plain-English API the chapters call (P.Robot(), P.nudge(), P.rlhf(), P.ai_judge(), …)
  ui.py               cards, chat bubbles, emoji grids, animations, quizzes, sliders, voting buttons
  toy.py policy.py    bandit, grid world, Q-learning; mini-policy coaching, reward model, RLHF, DPO
  models.py           local model backend (transformers + peft) and the reply cache
  data.py             Acme's knowledge base, transcripts, comparisons, evaluation messages
  checks.py evals.py risks.py   automatic checks, AI judge, comparisons, risk probes, decision checklist
  train.py config.py  LoRA + DPO training (maintainers only); model names and settings
artifacts/            trained adapters + training logs; cached model replies
teaching/             instructor guide (run sheet), learner cheat sheet, take-home exercises, solutions
tests/                every chapter runs end to end (fake backend), stays short and formula-free; runtime tests
scripts/              build_notebooks.py, run_notebooks.py, train_models.py
```

## For maintainers

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/build_notebooks.py          # after editing src/rllab or notebooks/nbXX.py
.venv/bin/python -m pytest                           # no GPU, no downloads
.venv/bin/python scripts/run_notebooks.py --live     # real models → reference_runs/ (and fills the cache)
.venv/bin/python scripts/train_models.py             # retrain the adapters (minutes on a T4 / Apple silicon)
```

The tests also keep the chapters easy to read: at most 110 words per text cell, at most 12 lines per code cell, no
formulas outside the "for the curious" folds, and at least three polls, five takeaways and one challenge per chapter.

After retraining, delete `artifacts/cache/`, run `--live`, and **re-read every 💬 takeaway**. They describe the
reference results (e.g. "six kinds of message got better", "60% → 70% → 55%"), and a new training run can change them.

## A note on honesty

* **The data is generated, and its flaws are planted on purpose.** `src/rllab/data.py` builds the transcripts and
  comparisons from a small knowledge base. Two rater habits (preferring gushing replies to angry customers, and agreeing
  with confidently wrong customers) are deliberate and documented. Two facts are held out of all training.
* **The model results are real.** Every reply, verdict and score comes from running the models; the cache stores them.
  Where reality differs from the planted story, the chapters say so. For example, the real Coached-Pip did *not*
  become a yes-man (it hedged instead), and Chapter 5 calls that out as a surprise.
* **The mini-Pips show mechanisms, not scale.** They don't reproduce effects like PPO's instability on big models.
