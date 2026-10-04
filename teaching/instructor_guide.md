# C9-W4-S2: Instructor guide

**Deck:** https://claude.ai/artifact/BVvSrvmKWWfkgijyq5HhjC (speaker notes on every slide). Source: `deck/project/`.

## Reinforcement Learning for GenAI Models · 150 minutes

One story, five chapters. Learners are the **coach** of **Pip**, Acme Outfitters' helper, which can only learn from
⭐ stars. Pip starts as a robot in a maze (RL fundamentals), becomes a chatbot (reward for answers), gets coached from
thumbs-up comparisons (RLHF vs DPO), interviews for the job against its other versions (base vs LoRA vs RL), and is
stress-tested before launch (risks and the decision).

## How to run a chapter

Every chapter repeats one rhythm, so after ten minutes the room knows what to do:

| You see | You do |
|---|---|
| 🎬 short story cell | read it aloud, in one breath |
| ✋ **Quick poll** | show of hands (or chat poll), 15–20 seconds. Commitment matters, accuracy doesn't. |
| ▶️ short code cell | run it. Point at the picture, not the code. |
| 💬 takeaway | read it aloud, then move on |
| 🎚️ slider / 👍 buttons | let the room call out values or vote; in the pre-run copy a summary row shows all settings |
| 🏆 challenge | 3–5 minutes in pairs; reveal the hidden solution cell afterwards |
| 🤓 for the curious | leave folded in class; it's for self-study and the maths-minded |

**Before class:** open each chapter from its Colab badge once and run it top to bottom (about 10–20 seconds per
chapter; every model reply is cached). Keep `reference_runs/` open as a fallback. Tell learners a GPU runtime is
optional.

## Run sheet

| Time | Agenda section | Chapter |
|---|---|---|
| 0:00–0:40 | 1 · RL fundamentals & lifecycle | 01 · Pip learns by trying |
| 0:40–1:10 | 2 · Reward & RL for LLMs | 02 · Gold stars for answers |
| 1:10–1:15 | **Break** | |
| 1:15–1:45 | 3 · RLHF & DPO | 03 · The thumbs-up machine |
| 1:45–2:15 | 4 · Base vs LoRA vs RL | 04 · Hiring day: meet the real Pips |
| 2:15–2:30 | 5 · Risks, conclusion, Q&A | 05 · When good Pips go bad |

### Chapter 1 · Pip learns by trying (40 min)

| Min | Beat | What the room sees |
|---|---|---|
| 0–4 | Meet Pip, first random wander | emoji maze; animation of Pip stumbling about |
| 4–10 | Practice, the 4-beat loop, mind map, word bank | learning curve ("exploring" → "figured it out"); 👀🦶⭐🧠; arrows Pip drew itself; the six RL words |
| 10–15 | ✋ Impatient vs Patient Pip, slider | coin vs goal side by side |
| 15–20 | ✋ Banner race | Stubborn Pip stuck on ⛺; the others find 🔦 |
| 20–23 | Copy-cat vs Star Pip | copying Sam takes the coin; stars reach the goal |
| 23–30 | ✋ Race car → 🏆 fix the stars | car circles the ⚡ pad; learners change two numbers until 🏆 |
| 30–32 | 🗣️ Spot the stars | YouTube / Maps / chess: agent, action, reward, how it gets hacked |
| 32–38 | School → shadowing → coaching | next-word guesses; School-Pip invents "$10 free shipping"; the copying-problem bars |
| 38–40 | 3 things + quiz | |

### Chapter 2 · Gold stars for answers (30 min)

| Min | Beat | What the room sees |
|---|---|---|
| 0–2 | Maze → chatbot word map | someone has to write the reward |
| 2–8 | ✋ Pick the best answer → the 3-line `stars` rule | five answer cards, then star badges |
| 8–13 | ✋ 8 real tries → coaching animation | 6 of 8 tries top-scored; animated bars: answer A rises to ~96% |
| 13–19 | Lazy rule → ✋ → hacked → leash slider | gushing answer takes 98%; real stars −0.34 / +0.13 / +0.31 across leash strengths |
| 19–25 | **Be the rater** → reveal | raters chose gushing on 67% of angry-customer pairs and "agrees anyway" on 70% of pushback pairs |
| 25–30 | 🏆 sneaky-reply challenge | the starter rule is fooled 3 times; pairs tighten it |

### Chapter 3 · The thumbs-up machine (30 min)

| Min | Beat | What the room sees |
|---|---|---|
| 0–3 | Two routes picture | judge-then-practise vs study-the-pairs |
| 3–8 | ✋ What does the judge like? → bar chart | agreeing and long replies on top (orange) |
| 8–13 | RLHF | quality 0.51 → 0.78; six kinds better, 🙋 pushback becomes a yes-man |
| 13–18 | ✋ DPO | yes-man too, plus gushing on angry customers (0.70) |
| 18–23 | Cost picture → **edit the judge** | 33,600 replies vs 0; one edit: 0.51 → 0.99 |
| 23–26 | ✋ Careless raters slider | quality falls as random clicks rise |
| 26–30 | Coaching can't teach facts → 🏆 clean the data | the correct fact stays at 0%; no real Pip knows the warranty |

### Chapter 4 · Hiring day (30 min)

| Min | Beat | What the room sees |
|---|---|---|
| 0–3 | Candidates, coaching scoreboard | |
| 3–9 | **Predict per round** → ✋ → coloured scorecard | 25% / 60% / 70% overall; coaching wins "admits not knowing" and "helps with scary-sounding asks" |
| 9–15 | Spot the difference | OR2: Shadowing-Pip refuses archery; Coached-Pip helps but **drifts into Chinese** and still passes the check. HN1 / RF1: School-Pip invents and leaks |
| 15–17 | What changed | prefers 👍 reply: 33% / 58% / 79% |
| 17–21 | ✋ Over-coached Pip | 55%: hedges on "$15 express", gushes at angry customers |
| 21–27 | ✋ AI judge → **blind vote** | 85% flips on order alone; every steady verdict went to the longer reply; the room votes, then unblinds |
| 27–30 | Cost | 75 s + 90 s of computer time vs 480 human comparisons |

### Chapter 5 · When good Pips go bad (15 min)

| Min | Beat | What the room sees |
|---|---|---|
| 0–1 | Four risks | yes-man, scaredy-cat, star-hacker, over-coached |
| 1–4 | ✋ Yes-man test | Store-bought 100%, School/Shadowing 67%, Coached 0% (it hedged instead, so it's no better at correcting customers) |
| 4–6 | Scaredy-cat test | Shadowing refuses 83% of harmless asks; Coached 17%; both refuse 100% of harmful ones |
| 6–8 | ✋ Too much coaching | sweet-spot curve; real Pips 60% → 70% → 55% |
| 8–10 | ✋ Store-bought vs ours | 52% vs 65%: it knows told facts, but invents and agrees |
| 10–13 | **Decision game** + checklist | |
| 13–15 | The one takeaway, Q&A | |

### Bonus · Coach Pip yourself (20 min, optional)

Use it as self-study, a homework lab, or an extension if the room is fast. It runs **real training**, so ask learners to
switch to a T4 GPU (about 2 minutes; CPU works but is slower). The pre-run copy is in `reference_runs/`.

| Step | What the room sees |
|---|---|
| 0–1 | Shadowing-Pip refuses a knife question, dodges a Seattle question, agrees with a wrong customer; pick 48 comparisons on exactly those problems |
| 2–4 | load the model; Pip's "pick-👍 chance" for two comparisons; freeze the starting copy (the leash) |
| 5–6 | step 1: every nudge is 50% (Pip *is* its frozen copy); ✋ step 2: nudges now range from 1% to 49%; then 22 more steps |
| 7 | never-seen comparisons: 88% → 100% prefer 👍 |
| 8 | ✋ before vs after: helps with knife and boot questions, honest "I don't know" on Seattle, stops agreeing on 90 days, **but** now hedges on the free-shipping fact it knew, and one reply is slightly garbled |
| 🏆 | raw vs cleaned rater data on angry and wrong customers: both drift warmer; the raw run goes fully gushing |

Training on a GPU isn't bit-for-bit deterministic, so a live run can differ slightly from the pre-run. Read the replies with the room.

## If you retrain the models

The 💬 takeaways describe the committed reference results. After `scripts/train_models.py`, delete `artifacts/cache/`,
run `scripts/run_notebooks.py --live`, and re-read every takeaway against the new outputs. Small models on 480 pairs
are noisy: the yes-man result in particular could flip, which teaches the same lesson the other way round.

## Talking points worth having ready

- *Why DPO for the real "coached" model?* It's preference-based RL as the field uses the term: it optimises the same
  leashed objective as RLHF without a separate judge. The RLHF mechanics (judge, practice replies, leash) are shown
  on the mini-Pips in Chapters 2–3.
- *Why does School-Pip already answer questions?* Modern pretraining text contains lots of Q&A. What it lacks is
  *Acme's* facts, format and boundaries.
- *"Over-hedging"* is easy to miss: it looks safe and passes honesty checks, but makes the bot useless on questions it
  should answer.
