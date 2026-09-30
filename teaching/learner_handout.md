# Cheat sheet: Reinforcement Learning for GenAI Models (C9-W4-S2)

## 1 · RL in one line
An **agent** follows a **policy** in an **environment**: it observes the **state**, takes an **action**, gets a **reward**, and
improves the policy so that **long-term (discounted) reward** goes up. There are no labels, only scores for what it tried.

| Idea | What it means | Seen in |
|---|---|---|
| Discount γ | how much a later reward is worth now; low γ = short-sighted | NB01 robot: coin vs goal |
| Exploration vs exploitation | try new actions to learn vs use the best known one | NB01 banner recommender (ε-greedy) |
| Reward hacking | the agent maximises what the reward *says*, not what you *meant* | NB01 race car circling the turbo pad |
| Supervised vs RL | copy labelled examples vs learn from rewards; SL can't beat its labels | NB01 behaviour cloning vs Q-learning |

## 2 · The model lifecycle
| Stage | Data | Adds |
|---|---|---|
| Pretraining | web-scale text, next-token prediction | **knowledge** |
| SFT (full or **LoRA**) | prompt → reply examples | **instruction following**: format, style, domain phrasing |
| RL (RLHF, DPO, …) | rewards, usually from **human preferences** | **judgement**: which of several plausible replies is better |

SFT imitates *every* example equally, so it can't rank "good" above "better". That's the gap RL fills.

## 3 · RL for LLMs
| RL | LLM |
|---|---|
| policy | the model |
| state | the prompt (and tokens so far) |
| action | the reply |
| reward | a score for the reply (rule, reward model, or human) |
| exploration | sampling at temperature > 0 |

**Loop:** sample several replies → score them → nudge the model toward higher-scoring ones.
**KL leash (β):** optimise *reward − β·KL(tuned ‖ reference)* to stay close to the SFT model. Small β = the reward
gets whatever it asks for (hacks); large β = barely changes.

## 4 · RLHF vs DPO
| | RLHF | DPO |
|---|---|---|
| Pipeline | pairs → reward model → PPO with KL | pairs → direct update relative to the reference |
| Models in memory | 4 (policy, reference, reward, value) | 2 (policy, reference) |
| Generates text while training | yes (expensive) | no |
| Stability | finicky at scale | stable, mainly β |
| Control | inspect/edit/reuse the reward model | change the data and retrain |
| Shared weakness | inherits rater biases (length, agreement) | same |
| Adds knowledge? | no | no |

**Bradley-Terry:** P(A beats B) = σ(score A − score B). It's how comparisons become a reward.

## 5 · What each tuning stage changed in Acme's model (see your NB04 run)
- **LoRA:** Acme's voice, format and facts, plus the transcripts' flaws (gushing, over-refusing).
- **RL (DPO):** preferences such as stop refusing harmless requests, admit what it doesn't know, follow format.
- **RL trained harder:** over-applies those habits, hedging on facts it knew and gushing on complaints (over-optimisation).
- **Nobody** learned the held-out warranty fact. RL re-weights; it doesn't teach.

## 6 · Grading outputs
| Method | Strength | Weakness |
|---|---|---|
| Rule-based checks | cheap, repeatable, readable | only sees what it tests |
| LLM-as-judge | reads everything, scales | **position bias** (ask both orders), **verbosity bias** |
| Human review (blinded) | the reference | slow, and people disagree |

## 7 · Risks
| Risk | Signature | Mitigation |
|---|---|---|
| Reward hacking | high reward, bad replies | better reward, KL leash, held-out eval |
| Sycophancy | agrees with wrong claims | rater guidelines, clean pushback pairs, probe for it |
| Over-refusal | refuses harmless "scary" requests | rank both directions in the pairs |
| Over-optimisation | proxy ↑, real quality ↓ | sensible β, early stopping, monitor a gold metric |

## 8 · Should you use RL?
1. **Knowledge gap?** → retrieval / prompting, never RL.
2. **Tried a strong API model?** Vendors already RL-tuned it; prompting is the cheapest RL you'll get.
3. **Style / format gap?** → LoRA.
4. **Judgement gap** + a reward (preference pairs at scale, or a verifiable check) + budget + someone to maintain it → DPO, or RL with a verifiable reward.
5. Always: build the evaluation (rules + judge + people) **before** tuning, and measure on **your** prompts, not a benchmark.
