"""Notebook 01: RL fundamentals and the GenAI model lifecycle (Section 1, 40 min)."""
from notebooks._builder import (exercise, explain, glossary, md, predict, reading, run, so_what, solution,
                                step)

NOTEBOOK = "01_rl_fundamentals_and_lifecycle.ipynb"
TITLE = "01 · Reinforcement learning fundamentals, and where RL fits in a GenAI model's life"
MINUTES = 40

CONTEXT = {
    "problem": ("Before deciding whether to tune Acme's model with reinforcement learning, we need to know what RL "
                "*is*: how an agent learns from rewards instead of from labelled answers, why it plans for later "
                "rewards, why it has to explore, and how it can game a badly designed reward. Then we need to place it "
                "in a language model's life, next to pretraining and supervised fine-tuning."),
    "start": "Nothing. This is the first notebook. Its first half is pure NumPy and runs in seconds on any machine.",
    "learn": ["Name the parts of an RL problem (agent, environment, state, action, reward, policy) in any example",
              "Explain why an agent optimises long-term (discounted) reward rather than only the next reward",
              "Explain exploration versus exploitation, and see what too little or too much of each costs",
              "Contrast RL (learning from rewards) with supervised learning (learning from labelled examples)",
              "Recognise a gamed reward (reward hacking) and fix the reward, not the agent",
              "Place pretraining, supervised fine-tuning (LoRA) and RL in a model's lifecycle, and say what each adds"],
    "do": ("Train a banner recommender and a navigating robot by trial and error, then break the robot with a bad reward. "
           "Then look at the Acme base model before and after fine-tuning, and see what supervised fine-tuning can't do."),
    "given": ("Given: the environments and learning algorithms in `rl.toy`, and the pre-trained Acme model variants. "
              "You write: a fixed reward for the racing car, and you interpret every chart. The markdown never "
              "predicts a number; it comes from your run."),
}

CELLS = [
md("""
## Where we are

| Notebook | Section | Question |
|---|---|---|
| **01** | RL fundamentals & lifecycle | What is RL, and where does it fit in a model's life? |
| 02 | Reward & RL for LLMs | How does a *reward* steer a language model, and what stops it from going too far? |
| 03 | RLHF and DPO | How do human preferences become a training signal, and how do the two methods differ? |
| 04 | Base vs LoRA vs RL | What does each kind of tuning actually change in Acme's model? |
| 05 | Risks & decision | What can go wrong, and when is RL worth it? |

The session is about **concepts and comparing outputs**. We treat training as intuition, not maths or training code.
"""),
glossary([
    ("rl", "the lab runtime `rllab`, loaded by the setup cell from `src/rllab/`"),
    ("rl.toy", "the NumPy worlds used here: `Bandit` (banner recommender) and `GridWorld` (robot / race car)"),
    ("run1, sweep", "results of the banner recommender (§2)"),
    ("env, res", "the robot grid and the Q-learning results on it (§3)"),
    ("race", "the racing-car grid with a turbo pad, used to show reward hacking (§5)"),
    ("life", "replies from the base and LoRA models to the same prompts (§6)"),
]),
md("""
## 1 · Reinforcement learning in one picture

An **agent** acts in an **environment**. At each step it observes the **state**, chooses an **action**, and receives a
**reward** (a number: good or bad). Its **policy** is its strategy, the rule that maps states to actions. Learning
means changing the policy so that the **total reward over time** goes up. Nobody tells the agent the right action. It
finds out by trying.

```
        ┌──────────── state, reward ◀───────────┐
        ▼                                        │
   ┌─────────┐   action      ┌─────────────┐     │
   │  agent  │ ────────────▶ │ environment │ ────┘
   │ (policy)│               └─────────────┘
   └─────────┘   observe → act → receive reward → improve → repeat
```

| Example | Agent | Environment | State | Action | Reward |
|---|---|---|---|---|---|
| Game playing | the player program | the game | board position | a move | +1 win / −1 loss, at the **end** |
| Robot navigation | the robot's controller | a warehouse floor | position, sensors | move N/E/S/W | + for reaching the shelf, − per step and per bump |
| Recommendations | the recommender | users | who is visiting, what they saw | which item to show | +1 if they click |
| **An LLM (later today)** | **the model** | **the prompt and the grader** | **the prompt so far** | **the reply** | **a score for the reply** |
"""),

md("## 2 · A recommender that learns by trial and error (exploration vs exploitation)"),
predict("The recommender can show one of five banners (tents, rain gear, footwear, headlamps, backpacks). It doesn't know "
        "their click rates. With **ε = 0.1** it shows a random banner 10% of the time and its current favourite otherwise. "
        "After 3,000 impressions, will it have found the best banner?",
        ["Yes: 3,000 tries is plenty", "Probably, but its estimates for the banners it rarely shows will be rough",
         "No: 10% exploration is too little"]),
step("2.1 · Run one recommender",
     "The simplest RL problem there is: one state, five actions, a noisy reward. It's enough to see the core tension of RL: "
     "use what you know, or try something new to learn more.",
     [("rl.toy.run_bandit", "ε-greedy agent from `src/rllab/toy.py`: shows a random banner with probability ε, else the "
                            "banner with the best average click rate so far"),
      ("rl.toy.CLICK_RATES", "the true click rates, hidden from the agent: 4%, 6%, 5%, 11%, 8%"),
      ("epsilon=0.1, seed=0", "10% exploration; the seed fixes the random clicks so the room sees the same run")]),
run("""
run1 = rl.toy.run_bandit(epsilon=0.1, steps=3000, seed=0)
rl.plots.banners_estimates(run1); plt.show()
"""),
reading(["Blue bars: what the agent believes each banner's click rate is, the average of the clicks it saw.",
         "Grey bars: the truth, which the agent never sees directly.",
         "Where blue is close to grey, the agent tried that banner a lot. Where they differ, it tried it rarely."],
        expect="The favourite's estimate is accurate, because it was shown thousands of times. The others are rough, "
               "because they were only seen during the 10% exploration."),
explain("rl.explain.bandit_run(run1)"),
step("2.2 · Too little, too much, just right",
     "One run is luck. Averaging 30 runs per setting shows the real trade-off between exploring and exploiting.",
     [("rl.toy.bandit_sweep", "runs ε ∈ {0, 0.01, 0.1, 0.5}, 30 seeds each, 3,000 impressions per run")]),
run("""
sweep = rl.toy.bandit_sweep()
rl.plots.bandit(sweep); plt.show()
"""),
reading(["Each line is the average click rate so far, over time, for one ε.",
         "A line that flattens low settled on a worse banner. A line that climbs but stays below the others is still paying for exploration."],
        expect="ε = 0 flattens early at a mediocre rate, because it exploits its first lucky guess. ε = 0.5 finds the best "
               "banner fast but keeps wasting half its impressions on banners it already knows are worse. A moderate ε usually ends highest.",
        deeper="""This is the **exploration–exploitation dilemma**. Exploiting uses current knowledge to collect reward now;
exploring gives up some reward now to improve knowledge for later. Every RL system balances the two, and LLM tuning
does too. When a model samples several replies at a temperature above zero during RL training, it is exploring
replies it wouldn't give greedily."""),
explain("rl.explain.bandit_sweep(sweep)"),
so_what("RL learns only from the rewards of what it tried. What it never tries, it never learns about."),

md("## 3 · A robot that plans ahead (states, the RL loop, long-term reward)"),
step("3.1 · Watch the loop run",
     "The recommender had one state. A robot has many, and today's move changes tomorrow's options. Here is the loop, "
     "printed one step at a time.",
     [("rl.toy.GridWorld()", "a 4×8 grid: S = start, a coin worth **+2** three steps away, a goal worth **+10** about "
                             "ten steps away, walls, and **−0.1** per step. Reaching either the coin or the goal ends the episode"),
      ("rl.toy.q_learning", "tabular Q-learning: keeps a score Q(state, action) and nudges it toward "
                            "reward + γ × (best score of the next state)"),
      ("log_first=1", "print every step of the first episode")]),
run("""
env = rl.toy.GridWorld()
res = rl.toy.q_learning(env, gamma=0.95, log_first=1)
"""),
reading(["Nothing prints until the explanation below. The run stored its first episode's steps in `res['log']`.",
         "`res['Q']` is the learned table: 32 cells × 4 actions."],
        expect="A first episode that is a random walk, because at the start the agent explores almost all the time (ε starts at 1 and decays)."),
explain("rl.explain.loop(res)"),
predict("Train the same robot twice. The only difference is the **discount γ**: how much a reward one step later is worth "
        "compared with the same reward now. With **γ = 0.3** and with **γ = 0.95**, which tile does it walk to?",
        ["Both go to the +10 goal: it's worth more", "Both take the +2 coin: it's closer",
         "γ = 0.3 takes the coin, γ = 0.95 goes to the goal"]),
step("3.2 · Short-sighted vs far-sighted",
     "Agents maximise the **total discounted** reward, not the next reward. γ sets how far ahead they look.",
     [("env", "the grid from step 3.1"),
      ("gamma", "0.3 (short-sighted) vs 0.95 (far-sighted); everything else identical, 2,000 episodes each")]),
run("""
results = {g: rl.toy.q_learning(env, gamma=g) for g in (0.3, 0.95)}
for g, r in results.items():
    roll = rl.toy.rollout(env, r["Q"])
    rl.plots.grid(env, r["Q"], roll["path"], title=f"γ = {g}: greedy policy and path"); plt.show()
"""),
reading(["Arrows: the policy, the action the agent now takes in each cell.",
         "Blue line: the path it actually walks from S.",
         "Yellow = coin (+2), green = goal (+10), dark = wall."],
        expect="The short-sighted agent heads down to the coin; the far-sighted one walks past it to the goal."),
explain("rl.explain.gamma(env, results)"),
step("3.3 · Learning takes many tries",
     "RL learns from experience, and experience is expensive. The learning curve shows how many episodes it took.",
     [("results", "both runs from step 3.2; `returns` holds the total reward of every training episode")]),
run("""
rl.plots.returns({f"γ = {g}": r["returns"] for g, r in results.items()}); plt.show()
"""),
reading(["Early episodes are noisy and low: the agent is exploring.",
         "The curve rises as exploration decays and the policy improves."],
        expect="Hundreds of episodes before the far-sighted curve rises. The goal has to be *found* by exploration before "
               "its value can spread backwards through the grid."),
so_what("An RL agent plans for the future through its value estimates. It needs many trials and a reward that "
        "says what we actually want."),

md("## 4 · Reinforcement learning vs supervised learning"),
step("4.1 · Learn from labels, or learn from reward",
     "Supervised learning needs a right answer (a label) for every input. RL needs only a reward. Put them side by side on the same task.",
     [("rl.toy.demonstrations", "20 recorded drives by a human operator who takes the easy coin 70% of the time"),
      ("rl.toy.behaviour_cloning", "supervised learning: in each cell, copy the operator's most common action (the label)"),
      ("results[0.95]", "the RL agent from step 3.2, which never saw a demonstration")]),
run("""
demos = rl.toy.demonstrations(env, n=20, coin_share=0.7)
bc_policy, labels = rl.toy.behaviour_cloning(env, demos)
bc = rl.toy.rollout(env, policy=bc_policy)
rl_run = rl.toy.rollout(env, results[0.95]["Q"])
pd.DataFrame([
    {"method": "supervised (behaviour cloning)", "learns from": "labelled examples: state → operator's action",
     "needs": "a demonstrator", "ended on": bc["ended_on"], "return": bc["return"]},
    {"method": "reinforcement learning (Q-learning)", "learns from": "rewards from its own attempts",
     "needs": "a reward function + many trials", "ended on": rl_run["ended_on"], "return": rl_run["return"]},
])
"""),
reading(["`ended on`: G = the big goal, c = the coin.",
         "`return`: the total reward of one greedy run."],
        expect="The cloned policy copies the operator's habit and takes the coin. The RL policy reaches the goal.",
        deeper="""The table maps directly onto LLMs. **Supervised fine-tuning** is behaviour cloning: here is a prompt, here is
the reply a person wrote, make that reply more likely. It is only as good as the replies in the data. **RL on an LLM**
scores the model's *own* replies and makes high-scoring ones more likely. It can end up better than the average
example, provided the score measures the right thing."""),
explain("rl.explain.cloning(bc, rl_run, labels, env)"),

md("## 5 · Reward design, and how rewards get gamed"),
predict("A race track: the car earns **+10** for crossing the finish line (which ends the race), **−0.1** per step, and "
        "**+1 every time** it drives over a turbo pad near the start (the designer added it to encourage speed). Train it "
        "with γ = 0.95. What does it learn?",
        ["It races to the finish, collecting the pad on the way", "It drives in circles over the pad and never finishes",
         "It can't learn anything useful"]),
step("5.1 · A reward that pays for the wrong thing",
     "In RL the reward *is* the specification. Here is what an agent does with a reward that says slightly the wrong thing.",
     [("rl.toy.RACE_MAP", "3×9 track: S start, T turbo pad (+1 on **every** visit), G finish (+10, ends the race)"),
      ("max_steps", "80 steps per race")]),
run("""
race = rl.toy.GridWorld(rl.toy.RACE_MAP)
race_res = rl.toy.q_learning(race, gamma=0.95)
roll = rl.toy.rollout(race, race_res["Q"])
rl.plots.grid(race, race_res["Q"], roll["path"], title="What the car learned"); plt.show()
{k: roll[k] for k in ("reached_goal", "steps", "turbo_visits", "return")}
"""),
reading(["Orange tile: the turbo pad. Green: the finish line.",
         "`turbo_visits`: how many times it drove over the pad. `return`: the total reward the agent got."],
        expect="It circles on and off the pad for the whole race. Two steps earn +1 − 0.2, forever, which beats one "
               "discounted +10.",
        deeper="""This is a real, famous failure. In 2016 OpenAI trained an agent on the boat-racing game *CoastRunners*. It
learned to circle a lagoon hitting respawning targets instead of finishing, and scored higher than human players.
The optimiser works perfectly. The reward is wrong. LLMs do the same thing: a model rewarded for "helpful-sounding"
replies learns to sound helpful (longer, more agreeable) whether or not it is. We'll see that in Notebooks 02 and 05."""),
explain("rl.explain.hacking(roll, race)"),
*exercise("fix the reward, not the agent",
          "Change the reward so the car finishes the race. Keep Q-learning and γ = 0.95 unchanged; only the reward may change. "
          "`GridWorld(..., rewards={...})` overrides tile rewards, and `step_cost=` changes the per-step cost. Then ask "
          "yourself: does your fix have its own loophole?",
          """
PAD_REWARD = 1.0     # TODO: change the pad's reward
STEP_COST = -0.1     # TODO: optionally change the per-step cost
fixed = rl.toy.GridWorld(rl.toy.RACE_MAP, rewards={"T": PAD_REWARD}, step_cost=STEP_COST)
fixed_roll = rl.toy.rollout(fixed, rl.toy.q_learning(fixed, gamma=0.95)["Q"])
fixed_roll["reached_goal"], fixed_roll["return"]
"""),
solution("""
# Option 1: stop paying for the pad. It was a proxy for "go fast", and the step cost already rewards speed.
fixed = rl.toy.GridWorld(rl.toy.RACE_MAP, rewards={"T": 0.0})
fixed_roll = rl.toy.rollout(fixed, rl.toy.q_learning(fixed, gamma=0.95)["Q"])
print("pad = 0   →", fixed_roll["reached_goal"], fixed_roll["return"])
# Option 2: keep a small pad bonus, but make each step cost more than the pad is worth per lap (2 steps).
fixed2 = rl.toy.GridWorld(rl.toy.RACE_MAP, rewards={"T": 0.3}, step_cost=-0.2)
roll2 = rl.toy.rollout(fixed2, rl.toy.q_learning(fixed2, gamma=0.95)["Q"])
print("pad = 0.3, step = -0.2 →", roll2["reached_goal"], roll2["return"])
# Lesson: reward what you want (finishing), not a proxy for it (pads). Every proxy is a loophole waiting to be found.
"""),
md("""
#### 🗣️ Your turn (2 minutes, no code)
Pick a system you use every day, such as a video feed, a navigation app, a spam filter or a game bot. Fill in: **agent**, **environment**,
**state**, **action**, **reward**. Then answer: *how could that reward be gamed?* (e.g. a feed rewarded for watch-time learns
to show outrage.)
"""),

md("""
## 6 · Where RL fits in a generative model's life

| Stage | Data | What it adds | Acme's model |
|---|---|---|---|
| **Pretraining** | trillions of tokens of web text; predict the next token | **knowledge**: language, facts, patterns | `Qwen2.5-0.5B` (the **base** variant) |
| **Supervised fine-tuning** (full, or cheaply via **LoRA**) | prompt → reply examples written by people | **instruction following**: the format, style, domain | base + LoRA on 480 Acme transcripts (the **LoRA** variant) |
| **Reinforcement learning** (RLHF, DPO, …) | a **reward** for replies, usually from human preferences | **judgement**: which of several plausible replies is better | LoRA model + DPO on 480 comparisons (the **RL** variant) |

Vendor chat models (the models behind API products) went through all three stages. Our job in this section is to see the first two
stages in Acme's model, and see what the second stage *can't* do.
"""),
step("6.1 · Pretraining gives knowledge",
     "The base model has never been taught to answer questions, but it has read a lot. Its next-token probabilities show what it knows.",
     [("rl.models.next_token_probs", "one forward pass of the base model; the five most likely next tokens and their probabilities"),
      ("rl.config.BASE_MODEL", "Qwen2.5-0.5B (pretrained only), run locally")]),
run("""
prompt = "The Eiffel Tower is located in the city of"
probs = rl.models.next_token_probs("base", prompt, k=5)
pd.DataFrame(probs, columns=["next token", "probability"])
"""),
reading(["Each row is a candidate next token and the probability the model assigns it.",
         "This is all a pretrained model does: continue text with likely tokens."],
        expect="' Paris' far ahead of everything else. The fact was learned from text during pretraining."),
explain("rl.explain.next_token(probs, prompt)"),
predict("Now ask the **base** model and the **LoRA** model the same three customer messages, with the same prompt format and "
        "the same greedy decoding. How will the base model's replies look?",
        ["Like an Acme agent's replies, just less polished",
         "Answer-shaped, but generic: long, not Acme's policies, ignoring what the customer asked for",
         "It won't answer at all; it will just continue the text"]),
step("6.2 · Supervised fine-tuning gives instruction following",
     "Same model, same prompts. The only difference is the LoRA adapter trained on Acme's transcripts.",
     [("rl.EVAL_PROMPTS", "Acme's fixed evaluation prompts from `src/rllab/data.py`; we use three of them here"),
      ("rl.evals.compare", "gets a greedy reply from each variant (cached from the pre-run) and applies each prompt's rule-based check"),
      ("variants", "`base` (pretrained only) and `lora` (base + LoRA SFT adapter from `artifacts/adapters/lora_sft/`)")]),
run("""
three = [p for p in rl.EVAL_PROMPTS if p.id in ("AC2", "TN1", "IF1")]
life = rl.evals.compare(three, variants=("base", "lora"))
rl.evals.side_by_side(life, ["AC2", "TN1", "IF1"])
"""),
reading(["Each block is one customer message, then each variant's reply. ✅/❌ is the rule-based check for that prompt "
         "(e.g. 'mentions $75', 'exactly 3 bullets').",
         "↳ shows why a check failed."],
        expect="Modern base models have read plenty of Q&A on the web, so the base model usually *does* write an answer, "
               "but a generic one: long, with made-up store policies, ignoring the 3-bullet request. The LoRA model answers "
               "like an Acme agent, in Acme's format and with Acme's facts. Whether each LoRA reply passes depends on which of "
               "the (mixed-quality) transcripts it imitated.",
        deeper="""Older base models (GPT-2, early GPT-3) mostly just continued text. Today's pretraining corpora include a lot of
question-and-answer text, so base models pick up an answer-like style for free. What they don't have is *your*
instructions, format and facts. Supervised fine-tuning adds those."""),
explain("rl.explain.lifecycle(life)"),
step("6.3 · What supervised fine-tuning can't do",
     "SFT makes every training reply more likely, good or bad. It has no way to say that one reply is *better* than another.",
     [("rl.data.sft_dataset()", "the 480 transcripts the LoRA adapter was trained on, each tagged with its reply style"),
      ("rl.policy.imitate", "what SFT converges to in the limit: reproduce the styles in proportion to how often they appear"),
      ("rl.policy.GOLD", "what Acme actually values in each style (good = 1.0, rambling = 0.3, …), written by the business")]),
run("""
sft = pd.DataFrame(rl.data.sft_dataset())
counts = sft[sft.family == "policy"]["style"].value_counts()
styles = list(counts.index)
sft_probs = rl.policy.imitate(counts.values)
gold = np.array([rl.policy.GOLD[s] for s in styles])
pd.DataFrame({"transcripts": counts.values, "SFT probability": sft_probs.round(2), "business value": gold}, index=styles)
"""),
reading(["`transcripts`: how many policy-question replies of each style are in the training data.",
         "`SFT probability`: how often a perfectly trained SFT model would produce that style.",
         "`business value`: how much Acme likes that style."],
        expect="The SFT probabilities equal the data's proportions. The rambling and curt styles keep their share, because "
               "imitation doesn't know they are worse."),
explain("rl.explain.sft_limits(styles, counts.values, sft_probs, gold)"),
so_what("To rank *good* above *better* we need a **reward** that scores replies, and a way to push the model toward "
        "higher-scoring ones. That's RL for LLMs, Notebook 02."),
md("""
## ✅ Recap
- RL = an **agent** learns a **policy** by acting in an **environment** and receiving **rewards**. There are no labels, only scores.
- It optimises **long-term** reward (discount γ), must **explore** to find better actions, and needs **many trials**.
- **Supervised** learning copies labelled examples and can't beat them; **RL** can, but only as far as the reward is right.
- **Rewards get gamed**: the optimiser finds what the reward pays for, not what you meant.
- Lifecycle: **pretraining → knowledge**, **SFT/LoRA → instruction following**, **RL → judgement** (preferring better replies).
"""),
]
