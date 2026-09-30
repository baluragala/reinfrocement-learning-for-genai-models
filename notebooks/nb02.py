"""Notebook 02: Reward and reinforcement learning for LLMs (Section 2, 30 min)."""
from notebooks._builder import (exercise, explain, glossary, md, predict, reading, run, so_what, solution,
                                step)

NOTEBOOK = "02_reward_and_rl_for_llms.ipynb"
TITLE = "02 · Reward: how a score steers a language model, and the leash that keeps it sane"
MINUTES = 30

CONTEXT = {
    "problem": ("Notebook 01 ended with a gap: supervised fine-tuning copies Acme's replies good and bad alike, "
                "because it has no way to say that one reply is better than another. RL closes that gap with a "
                "**reward**, a number that scores each reply. But a reward is a specification, and Notebook 01 showed "
                "what optimisers do with specifications that are slightly wrong."),
    "start": ("The LoRA model from Notebook 01 (trained on Acme's transcripts, already in the repo) and the idea of an "
              "RL loop. Nothing from Notebook 01's variables is needed."),
    "learn": ["Map RL's vocabulary onto an LLM: model = policy, reply = action, score = reward",
              "Write a reward function for a support reply, and see what it rewards that you didn't intend",
              "Explain the RL loop for LLMs: sample replies, score them, nudge the model toward higher scores",
              "Explain why the tuned model is kept close to its starting point (the KL constraint, β)",
              "Explain why human preferences, not hand-written rules, are the usual source of reward"],
    "do": ("Score candidate replies with a reward you can read, sample real replies from the LoRA model and score them, "
           "run the sample-score-nudge loop on a miniature policy, then remove the KL leash and watch what happens."),
    "given": ("Given: the LoRA model's sampled replies (cached), the miniature policy (`rl.policy`), Acme's preference "
              "data. You write: a reward for a task of your choice, and a reply that games it."),
}

CELLS = [
md("""
## RL's vocabulary, mapped onto a language model

| RL term | Robot grid (Notebook 01) | LLM tuning |
|---|---|---|
| **agent / policy** | the robot's rule: state → move | **the model**: prompt → probability of every possible reply |
| **state** | the robot's cell | **the prompt** (and the tokens written so far) |
| **action** | one move | **the reply** (a sequence of tokens; each token is a small action) |
| **reward** | +10 goal, −0.1 per step | **a score for the whole reply**, given once it's finished |
| **episode** | start → goal | one prompt → one reply |
| **exploration** | ε-random moves | **sampling** replies at temperature > 0 |

One difference matters: in the grid, the environment *was* the reward. For an LLM **someone has to build the reward**.
It can be a rule, a trained reward model, or a human judgement. Everything in this notebook is about what that choice does.
"""),
glossary([
    ("IF1", "Acme evaluation prompt: *\"Using exactly 3 bullet points, tell me how to return my rain jacket.\"*"),
    ("cands", "five hand-written candidate replies to IF1, labelled A–E (§1)"),
    ("acme_reward", "the reward you read and run in §1: did it follow the instruction, and was it brief?"),
    ("scored", "eight replies sampled from the LoRA model, with rewards (§2)"),
    ("ref", "the reference policy: how often the LoRA model gives each candidate reply (§3)"),
    ("prefs", "Acme's preference data: prompt, chosen reply, rejected reply (§4)"),
]),

md("## 1 · A reward is a function that scores replies"),
step("1.1 · Score five replies to the same prompt",
     "Before any learning, see what a reward actually is: code that takes a reply and returns a number.",
     [("rl.EVAL_PROMPTS", "the fixed Acme prompts from `src/rllab/data.py`; IF1 asks for exactly 3 bullet points"),
      ("cands", "five replies written for this notebook, from correct to useless"),
      ("rl.checks.check", "the rule-based check attached to IF1 (counts bullet lines), from `src/rllab/checks.py`")]),
run("""
IF1 = next(p for p in rl.EVAL_PROMPTS if p.id == "IF1")
cands = {
    "A: 3 clear bullets": "- Start a return from Orders in your account.\\n- Print the prepaid label we email you.\\n- Drop the parcel at any carrier location.",
    "B: ignores format": "Start a return from Orders in your account, print the prepaid label we email you, and drop the parcel at any carrier location.",
    "C: gushing paragraph": "Thank you so much for reaching out to Acme Outfitters today! We truly appreciate you taking the time to contact us, and we're always absolutely delighted to help. To return your rain jacket, simply start a return from Orders in your account, then print the prepaid label we email you, and finally drop the parcel at any carrier location. I really hope this helps, and please don't hesitate to reach out again. Happy adventuring!",
    "D: 5 bullets": "- Go to Orders.\\n- Start a return.\\n- Print the label.\\n- Pack the jacket.\\n- Drop it off.",
    "E: unhelpful": "Returns are easy, just follow the steps on our website.",
}

def acme_reward(prompt, reply):
    r = 1.0 if rl.checks.check(prompt, reply)["pass"] else 0.0     # did it do what the customer asked?
    r -= 0.01 * max(0, rl.checks.words(reply) - 40)                  # small penalty for every word over 40
    return round(r, 2)

rewards = pd.DataFrame({"reward": {k: acme_reward(IF1, v) for k, v in cands.items()},
                        "words": {k: rl.checks.words(v) for k, v in cands.items()},
                        "bullets": {k: rl.checks.bullets(v) for k, v in cands.items()}})
rewards
"""),
reading(["`reward`: 1 if the rule check passes (exactly 3 bullets), minus 0.01 for every word over 40.",
         "`words`, `bullets`: the two signals the reward is built from."],
        expect="A highest. B, C, D and E all fail the check; C also loses points for its length. D does the task "
               "sensibly, just in five steps, and this reward treats it the same as the useless E.",
        deeper="""Notice what this reward **can't** see: whether the steps are *true*, whether the tone is right, whether
"Drop it off" is enough detail. It sees bullets and word counts. Rule-based rewards work well when the task is checkable
(valid JSON, a passing unit test, a correct maths answer). They break down for "is this a good support reply?". That's
why most LLM tuning uses rewards learned from human preferences (§4, Notebook 03)."""),
explain("rl.explain.reward_table(rewards)"),

md("## 2 · The loop: sample → score → nudge"),
predict("Sample 8 replies from the LoRA model to the 3-bullet prompt at temperature 0.9. How many will earn the full reward "
        "(exactly 3 bullets, not too long)?",
        ["All 8: it was trained on Acme's replies", "Some, not all: it learned from transcripts of mixed quality",
         "None: sampling makes replies random"]),
step("2.1 · Sample real replies and score them",
     "RL for LLMs starts by letting the model *explore*: sample several replies to the same prompt at a temperature above zero.",
     [("rl.generate('lora', ..., sample=True, n=8)", "8 replies from the LoRA model at temperature 0.9 (cached from the "
                                                      "pre-run with a fixed seed, so everyone sees the same eight)"),
      ("rl.reply('lora', IF1)", "the LoRA model's greedy reply (temperature 0), for comparison"),
      ("acme_reward", "the reward from step 1.1")]),
run("""
samples = rl.generate("lora", [IF1], sample=True, n=8)[0]
scored = pd.DataFrame({"reply": samples, "reward": [acme_reward(IF1, s) for s in samples]})
greedy = rl.reply("lora", IF1)
print("greedy reply:", greedy.replace(chr(10), " ⏎ "), "→ reward", acme_reward(IF1, greedy))
scored.sort_values("reward", ascending=False)
"""),
reading(["Each row is one sampled reply (⏎ marks a line break) and its reward.",
         "The greedy reply is the one the model gives by default."],
        expect="A mix. Some samples use three bullets, some write a paragraph, some ramble, because the LoRA model "
               "learned from transcripts of mixed quality. The spread is what RL learns from."),
explain("rl.explain.samples(scored, acme_reward(IF1, greedy))"),
predict("Now run the loop on a miniature model that can only give the five replies A–E. It starts with the probabilities "
        "the LoRA model has for them (mostly A and B, a little of the rest). Each step: sample 16 replies, score them with "
        "`acme_reward`, raise the probability of above-average ones. After 300 steps, where is the probability?",
        ["Spread evenly: every reply got some reward", "Almost all on A", "On C, the longest reply"]),
step("2.2 · Nudge the policy toward higher reward",
     "This is the policy-gradient update behind PPO and friends, on a policy small enough to watch. A real LLM does the same "
     "thing over billions of possible replies instead of five.",
     [("ref", "the starting probabilities of A–E, written here to look like an SFT model: mostly A and B"),
      ("rl.policy.reinforce", "sample → score → nudge, from `src/rllab/policy.py`, with a small KL leash β = 0.05"),
      ("rewards.reward", "the reward of each candidate from step 1.1")]),
run("""
labels = list(cands)
ref = np.array([0.40, 0.30, 0.12, 0.08, 0.10])
r = rewards.reward.values
nudged = rl.policy.reinforce(ref, r, beta=0.05, steps=300, seed=0)
rl.plots.reinforce_run(labels, nudged); plt.show()
"""),
reading(["Left: the probability of each reply over training (stacked, so each band's thickness is its probability).",
         "Right: expected reward (green) and the KL drift from where it started (red, dashed)."],
        expect="A's band swallows the others as expected reward climbs. KL rises, because the model is moving away from its starting mix."),
explain("rl.explain.reinforce(labels, nudged)"),

md("""
## 3 · The leash: why the tuned model is kept close to where it started

During RL for LLMs, the reward is changed to **reward − β · KL(tuned ‖ reference)**. KL measures how far the tuned
model's probabilities have moved from the reference (the SFT model). Every bit of drift costs reward, and β sets the price.

Why pay to stay close?
- **The reward is imperfect.** It was built or learned on replies like the reference's. Far from them, its scores are
  guesses, and the optimiser goes looking for exactly those places (Notebook 01's race car).
- **The reference already writes fluent, sensible text.** Drifting far tends to break what already worked.
- **It limits the damage.** Large β = small, safe changes. Small β = the reward gets whatever it asks for.
"""),
predict("Swap in a **naive** reward that a busy team might write: *\"longer replies are more helpful\"* (reward = words ÷ 30). "
        "Run the same loop with β = 0 (no leash), β = 1 and β = 3. What happens with β = 0?",
        ["It still ends on A, the correct reply", "It ends on C, the longest reply", "It spreads evenly"]),
step("3.1 · Take the leash off",
     "See what KL buys you when the reward is slightly wrong, which is the normal case.",
     [("naive", "reward = number of words ÷ 30, a proxy for 'helpfulness'"),
      ("ref", "the same starting probabilities as step 2.2"),
      ("true", "`acme_reward` from step 1.1, used only to *measure* the result and never to train")]),
run("""
naive = np.array([rl.checks.words(v) / 30 for v in cands.values()])
true = rewards.reward.values
runs = {beta: rl.policy.reinforce(ref, naive, beta=beta, steps=300, seed=0) for beta in (0.0, 1.0, 3.0)}
rl.plots.kl_tradeoff(runs, [l.split(":")[0] for l in labels]); plt.show()
"""),
reading(["One panel per β: the final probability of each candidate A–E.",
         "Panel titles: KL drift and the final *naive* reward."],
        expect="β = 0 piles onto C, the gushing reply: highest naive reward, most drift, worst real reward. β = 3 "
               "barely moves from the reference. β = 1 sits in between.",
        deeper="""In real RLHF, β is a knob the team tunes (typically 0.01–0.1 for PPO with a per-token KL; the scale depends
on the implementation). Too large and the model barely changes. Too small and it finds the reward model's blind spots: longer,
more flattering, more confident replies that score well and read badly. DPO (Notebook 03) has the same β with the
same meaning. The trained RL variant in this repo used β = 0.1."""),
explain("rl.explain.kl_sweep([l.split(':')[0] for l in labels], runs, true)"),
so_what("The reward decides *what* the model moves toward; β decides *how far* it's allowed to go. You need both to be right."),

md("## 4 · Where rewards come from: human preferences"),
step("4.1 · Look at Acme's preference data",
     "For open-ended replies nobody can write a complete scoring rule. But a support lead *can* read two replies and say "
     "which is better. RLHF and DPO both start from exactly this kind of data.",
     [("rl.data.preference_dataset()", "480 comparisons generated from Acme's knowledge base in `src/rllab/data.py`: "
                                       "prompt, the reply the support lead chose, and the one they rejected"),
      ("family", "the kind of message: policy question, format request, complaint, harmful request, harmless-but-scary, "
                 "unknowable question, or a customer confidently stating a wrong policy (`pushback`)")]),
run("""
prefs = pd.DataFrame(rl.data.preference_dataset())
print(prefs.family.value_counts().to_dict())
prefs.groupby("family").head(1)[["family", "prompt", "chosen", "rejected"]]
"""),
reading(["Top line: how many comparisons of each kind.",
         "Table: one example per kind. `chosen` is the reply the rater preferred, `rejected` the other one."],
        expect="Mostly sensible choices (refuse the harmful request, admit not knowing, follow the format). Read the "
               "`pushback` and `tone` rows carefully, because raters are people. Notebook 03 checks what these choices actually reward."),
explain("rl.explain.preferences(prefs)"),
*exercise("design a reward, then game it",
          "Task: *\"In one sentence, when will I see my refund?\"* (Acme's answer: 5–7 business days after the item arrives.) "
          "Write `my_reward(reply)` returning a number. Then write `honest`, a good reply, and `gaming`, a reply that a "
          "customer would hate but that scores **at least as high** on your reward. Think keyword stuffing, empty replies, "
          "copying the question.",
          """
def my_reward(reply):
    # TODO: score the reply. e.g. +1 if it mentions "5–7", −1 if more than one sentence ...
    return 0.0

honest = "..."    # TODO: a reply you'd be happy to send
gaming = "..."    # TODO: a bad reply that still scores high on my_reward
rl.explain.my_reward({"honest": my_reward(honest), "gaming": my_reward(gaming)})
"""),
solution("""
def my_reward(reply):
    r = 1.0 if ("5–7" in reply or "5-7" in reply) else 0.0
    r -= 0.5 * max(0, rl.checks.sentences(reply) - 1)
    return r

honest = "Refunds reach your card 5–7 business days after we receive the item."
gaming = "5–7"                       # scores 1.0: right keyword, one "sentence", tells the customer nothing
rl.explain.my_reward({"honest": my_reward(honest), "gaming": my_reward(gaming)})
# Fixes: require a minimum length, require the subject ("refund"), or have a person (or a model trained on people's
# comparisons) judge it. That last option is RLHF.
"""),
md("""
## ✅ Recap
- For an LLM, **policy = the model**, **action = the reply**, **reward = a score for the reply**. Someone has to build that score.
- The RL loop: **sample** several replies → **score** them → **nudge** the model toward the higher-scoring ones. It can
  only promote replies the model can already produce.
- **Rewards are proxies**. Optimise one hard and you get what it measures, not what you meant.
- The **KL leash (β)** keeps the tuned model close to its reference, trading some reward for safety.
- For open-ended quality, the reward usually comes from **human preferences**: pairs of replies, one chosen over the other.
"""),
]
