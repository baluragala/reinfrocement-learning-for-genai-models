# Cheat sheet: Reinforcement Learning for GenAI Models (C9-W4-S2)

*Coaching Pip, on one page. The left column is the story's word, the right column the textbook term.*

## Chapter 1 · Learning from stars
| Pip's world | Textbook term |
|---|---|
| Pip | **agent** |
| the maze / the customers | **environment** |
| the square Pip is on / the customer's message | **state** |
| a move / a reply | **action** |
| ⭐ stars | **reward** |
| Pip's mind map | **policy** |
| patience (a star later is worth a bit less) | **discount factor γ** |
| curiosity (sometimes try something new) | **exploration vs exploitation (ε)** |
| copying Sam | **supervised learning / imitation** |
| the car circling the turbo pad | **reward hacking** |

**The loop:** 👀 look → 🦶 act → ⭐ get stars → 🧠 remember. Repeat thousands of times.

**A model's life:** 📚 school = **pretraining** (knowledge) → 👀 shadowing = **fine-tuning / LoRA** (format and voice)
→ ⭐ coaching = **reinforcement learning** (judgement: which answer is better). Copying can't tell *good* from *better*.

## Chapter 2 · Stars for answers
- For a chatbot, **someone has to write the reward**: a rule, a judge model, or people.
- Coaching = **try, score, nudge**: high-scoring answers become more likely. It can't invent new answers.
- Lazy star rules get hacked ("longer is better" → gushing).
- The **leash** (**KL penalty**, strength **β**) keeps Pip close to how it answered before. Strong = safe, small changes.
- People comparing two replies give the best stars, habits included.

## Chapter 3 · RLHF vs DPO
| | 🧑‍⚖️ RLHF | 📖 DPO |
|---|---|---|
| How | train a **judge (reward model)** on comparisons, then coach Pip against it (**PPO**) | learn straight from the comparisons |
| Models in memory | 4 | 2 |
| Writes practice replies | yes (slow) | no |
| Fix a bad habit by… | editing the judge | changing the data |
| Both | learn the raters' habits · can't teach new facts | |

## Chapter 4 · What each kind of training changed
- 👀 **Shadowing (LoRA)** changed **how Pip talks**: Acme's facts, format and voice, plus the examples' flaws.
- ⭐ **Coaching (DPO)** changed **which answer Pip picks**: stopped refusing harmless questions, admits not knowing.
- 🏋️ **Over-coaching** overdid it: "I don't know" to facts it knew, gushing at angry customers.
- **Grade three ways:** automatic checks (blind spots) · AI judge (order and length bias, so ask both ways) · people.

## Chapter 5 · Risks and the decision
| Risk | Looks like |
|---|---|
| 🙇 yes-man (**sycophancy**) | agrees with wrong customers |
| 🙅 scaredy-cat (**over-refusal**) | refuses harmless "scary" questions |
| 🎭 star-hacker (**reward hacking**) | pleases the judge, not the customer |
| 🏋️ over-coached (**over-optimisation / Goodhart's law**) | judge's score ↑, real quality ↓ |

**Prompt, fine-tune, or coach?**
1. 📎 Missing facts → put them in the prompt / search. Never RL.
2. 🛒 Try a good ready-made (already-coached) chatbot with clear instructions first.
3. 👀 Wrong format or tone → LoRA.
4. ⭐ Picks the worse of two OK answers **and** you have lots of comparisons (or an automatic checker), budget, and someone to maintain it → coach (DPO / RL).

> **Coaching changes *which* answer a model prefers. It doesn't teach it anything new.**
