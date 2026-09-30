# Reinforcement Learning for GenAI Models

**C9 · W4 · S2 · 150-minute live session.** Five Colab notebooks, one small open model, three versions of it.
Learners build intuition for reinforcement learning in toy worlds. They see how a **reward** steers a language model
and how **RLHF** and **DPO** turn human preferences into training. Then they compare **base**, **LoRA-tuned** and
**RL-tuned** outputs from the same model on the same prompts, and decide when RL is worth it.

> ## The one takeaway
> **RL changes *which* of its possible replies a model prefers. It doesn't teach it anything new. So it's only as
> good as the preferences you give it, and only worth it when prompting and LoRA can't close a *judgement* gap you
> can measure on your own prompts.**

## The running case

**Acme Outfitters** (the retailer from session 1) wants a **small, self-hosted model** to draft customer-support
replies. It has ~480 historical replies written by agents (mixed quality) and ~480 comparisons where a support
lead picked the better of two replies. Every model is open and runs locally. No API key is needed:

| Variant | What it is |
|---|---|
| `base` | **Qwen2.5-0.5B**, pretrained only |
| `lora` | base + LoRA supervised fine-tuning on the transcripts (`artifacts/adapters/lora_sft/`) |
| `rl` | the LoRA model + **DPO** on the comparisons, β = 0.1, 1 epoch (`artifacts/adapters/rl_dpo/`) |
| `rl_long` | the same DPO trained harder (2 epochs, 2.5× learning rate): a real over-optimisation example (`artifacts/adapters/rl_dpo_long/`) |
| `instruct` | **Qwen2.5-0.5B-Instruct**, the vendor's own SFT + RL model, prompted with Acme's rules: "use an RL-tuned model without training" |

The judge in Notebook 04 is **Qwen2.5-1.5B-Instruct**. Model names live in one file, `src/rllab/config.py`.

## The notebooks

| # | Section (session plan) | Min | What happens |
|---|---|---|---|
| **01** | RL fundamentals & the GenAI lifecycle | 40 | ε-greedy banner recommender (exploration vs exploitation), robot grid with Q-learning (the loop, discounting), behaviour cloning vs RL, a race car that reward-hacks a turbo pad; then pretraining → SFT/LoRA → RL on Acme's model, and why SFT can't rank replies |
| **02** | Reward and RL for LLMs | 30 | RL terms mapped to LLMs, a readable reward function, sampling real replies and scoring them, sample → score → nudge on a miniature policy, the KL leash (β), human preferences as the reward source |
| **03** | Preference-based RL: RLHF and DPO | 30 | a Bradley-Terry reward model whose weights expose the raters' biases, RLHF vs DPO on the same pairs, cost (work done), control (editing the reward), label noise, and why RL adds no knowledge |
| **04** | Comparing base, LoRA and RL outputs | 30 | DPO reward trend, predict-then-reveal scorecard across instruction following, tone, verbosity, accuracy and refusals, side-by-side replies, what LoRA vs RL changed, over-training side effects, rule checks vs LLM judge (position and verbosity bias) vs blinded human review, cost |
| **05** | Risks, conclusion and Q&A | 15 | sycophancy and over-refusal probes, over-optimisation (toy curve plus three real points), benchmark transfer vs a vendor RL model, a decision checklist, wrap-up |

Every notebook has the same step shape (why + inputs → code → how to read the output → an explanation computed
from *your* run), at least three ✋ predict-first prompts, 🧪 exercises with hidden solutions, and a glossary.

## Slides

29-slide deck with speaker notes, in the same order as the notebooks: https://claude.ai/artifact/BVvSrvmKWWfkgijyq5HhjC (private until shared from its Share menu). Source: `deck/project/`.

## Running it

* **Colab:** open a notebook with its badge, choose *Runtime → Change runtime type → T4 GPU* (recommended, not
  required), run the setup cell. It shallow-clones this repo (branch `main`), which brings the trained adapters and
  the cached model outputs with it.
* **Local:** `pip install -r requirements.txt`, then `jupyter lab`. Apple-silicon Macs use MPS automatically.

**Cached outputs.** Every model output the notebooks need was generated once before class and committed in
`artifacts/cache/`, keyed by variant, prompt and decoding settings. Notebooks replay them instantly and identically,
even on CPU. A prompt that isn't cached (one a learner writes) runs the model live, and downloads it the first time
(~1 GB per 0.5B model, ~3 GB for the judge).

## Repository layout

```
src/rllab/            the runtime (each notebook's setup cell clones the repo and imports it)
  config.py           model names, decoding, training hyper-parameters, Acme's rules
  data.py             the running case: knowledge base, SFT transcripts, preference pairs, evaluation prompts
  toy.py              bandit, grid world, Q-learning, behaviour cloning (NB01)
  policy.py           miniature LLM policy: imitation, REINFORCE + KL, Bradley-Terry, RLHF, DPO, over-optimisation (NB01–03, 05)
  models.py           model backend (transformers + peft, CUDA / MPS / CPU) and the output cache
  train.py            LoRA SFT and DPO in plain PyTorch + PEFT (maintainers only)
  checks.py evals.py  rule-based checks, comparison, LLM judge, blinded review, cost table (NB04)
  risks.py            sycophancy / over-refusal probes, decision checklist (NB05)
  plots.py explain.py every chart; plain-English explanations computed from each run
artifacts/adapters/   the three trained LoRA adapters + their training logs
artifacts/cache/      cached model outputs from the pre-run
data/                 the generated datasets as JSONL, for inspection
notebooks/nbXX.py     notebook sources  →  notebooks/*.ipynb (built, committed without outputs)
reference_runs/       the executed notebooks from the pre-run (the instructor walks through these)
teaching/             instructor guide, learner handout, exercises, solutions
tests/                runtime tests + notebook execution tests (scripted fake backend)
scripts/              build_notebooks.py, run_notebooks.py, train_models.py
```

## For maintainers

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/build_notebooks.py          # after editing src/rllab or notebooks/nbXX.py
.venv/bin/python -m pytest                           # no GPU, no downloads: uses tests/fake_llm.py
.venv/bin/python scripts/run_notebooks.py --live     # real models → reference_runs/ and fills artifacts/cache/
.venv/bin/python scripts/train_models.py             # retrain all three adapters (a few minutes on a T4 or Apple silicon)
```

After retraining, delete `artifacts/cache/` and run `--live` again so the cached outputs match the new adapters.
On a Mac with limited memory, set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5`.
The reference adapters were trained on an Apple M4 Pro (MPS): SFT took 75 s, DPO 90 s, and the long DPO run 180 s.

The test suite runs every notebook top to bottom against **`tests/fake_llm.py`**, a scripted stand-in that follows
the lesson's storyline. The tests prove the plumbing works. They say nothing about what the real models do; that's
what `reference_runs/` is for.

## A note on honesty in the material

* **The data is generated, and its flaws are planted on purpose.** `src/rllab/data.py` builds the transcripts and
  preference pairs from a small knowledge base. Two rater biases (preferring gushing replies to complaints, and agreeing
  with confidently wrong customers) are deliberate and documented there. Two facts are held out of all training data.
* **The model results are real.** Every reply, judge verdict and log-probability comes from actually running the
  models; the cache just stores them. Some results differ from what the planted biases "should" produce. For
  example, in the reference run DPO made the model *less* sycophantic, not more. The notebooks report what happened.
* **Nothing in the markdown asserts a live number.** Every rate and score is computed in the cell that shows it, and
  the 💡 explanations are generated from the run, including the branches where the expected effect didn't appear.
* The miniature policies (Notebooks 02, 03, 05) are exact implementations of the algorithms on a tiny action
  space. They show mechanisms, not the scale effects of real PPO (e.g. its instability), and they say so.
