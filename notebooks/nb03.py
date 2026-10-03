"""Chapter 3: The thumbs-up machine. Preference-based RL: RLHF and DPO (Section 3, 30 min)."""
from notebooks._builder import challenge, curious, play, poll, solution, story, takeaway

STYLE = "story"
NOTEBOOK = "03_the_thumbs_up_machine.ipynb"
TITLE = "Chapter 3 · The thumbs-up machine 👍👎"
MINUTES = 30
HOOK = ("Acme has **480 thumbs-up comparisons**: for each customer message, two replies, one marked better. "
        "There are two famous ways to turn these into a better Pip, used by the makers of ChatGPT, Claude, Llama and others. "
        "Today you'll run both, and catch what each one learns *by accident*.")
MISSION = ["See the two routes: **RLHF** (hire a judge robot) and **DPO** (study the comparisons directly)",
           "Find out what the judge robot secretly learned to like",
           "Compare the two routes on cost and control",
           "Discover what coaching can **never** teach Pip"]

CELLS = [
story("""
## 1 · Two routes to a better Pip 🛣️
Both start from the same 480 comparisons. They take different roads:
"""),
play("P.two_routes()"),
takeaway("RLHF trains a judge first and then coaches Pip against it. DPO skips the judge and learns straight from the pairs."),

# ------------------------------------------------------------------ the judge
story("""
## 2 · Route 1, step 1: train a judge robot 🧑‍⚖️
The judge reads lots of comparisons and learns to give every reply a score, so that 👍 replies score higher than 👎 ones.
It notices simple things: length, bullet points, refusing, saying "I don't know", agreeing with the customer…
"""),
poll("What do you think the judge will like MOST?", ["🤷 Saying 'I don't know' (honest replies won)",
                                                      "🙇 Agreeing with the customer", "📏 Long replies", "🙅 Refusing (safe replies won)"]),
play("""
judge = P.Judge()
judge.likes()
"""),
takeaway("Nobody told the judge to love agreeing and long replies. It picked up the raters' habits from Chapter 2, and RL will amplify whatever the judge likes."),

# ------------------------------------------------------------------ RLHF
story("""
## 3 · Route 1, step 2: Pip practises against the judge 🔁
Pip writes replies, the judge scores them, and Pip nudges toward high scores, with the leash on. We look at all 7 kinds
of customer message. The coloured bars show which kind of reply Pip gives, before and after.
"""),
play("P.rlhf(judge)"),
takeaway("Six kinds of message got better. But 🙋 'customer is confidently wrong' got WORSE: Pip became a yes-man, because the judge loves agreeing."),

# ------------------------------------------------------------------ DPO
story("""
## 4 · Route 2: DPO, no judge needed 📖
DPO reads each pair and simply makes the 👍 reply a bit more likely and the 👎 reply a bit less likely, compared with
where Pip started. There's no judge, and Pip writes no practice replies.
"""),
poll("No judge means no judge's bad habits. Will DPO-Pip avoid becoming a yes-man?",
     ["Yes: the problem was the judge", "No: the habit is in the comparisons themselves", "DPO can't learn anything"]),
play("P.dpo()"),
takeaway("DPO became a yes-man too, and it also went gushing on angry customers. The habits live in the data, so every route learns them."),
curious("why both routes aim at the same place", """
RLHF looks for the policy that maximises `judge score − β × KL(Pip ‖ starting Pip)`. That best policy has a neat form:
`π*(reply) ∝ π_start(reply) × exp(score / β)`. Flip it around and the score becomes `β × log(π*/π_start)`: Pip's own
probabilities can play the judge. DPO plugs that into the comparison loss and trains Pip directly. Same target, no judge.
"""),

# ------------------------------------------------------------------ cost & control
story("""
## 5 · Which route? Cost and control 💰🎛️
"""),
play("P.cost_picture()"),
story("""
### 🔧 Hands-on: fix the judge
RLHF has one big advantage: the judge is a separate thing you can **inspect and edit**. Let's switch off its love of agreeing and coach again.
"""),
play("""
judge.ignore("agreeing")
P.rlhf(judge)
"""),
takeaway("One edit to the judge fixed the yes-man, with no new data. With DPO, the only lever is the data itself."),

# ------------------------------------------------------------------ data quality
story("""
## 6 · Messy raters 🎲
What if some raters weren't paying attention and clicked at random? 🎚️ Slide to find out.
"""),
poll("If 40% of the comparisons were clicked at random, how much worse does Pip get?",
     ["Barely: the other 60% average it out", "Noticeably worse", "It gets better (more variety)"]),
play("P.careless_raters()"),
takeaway("Careless comparisons blur the judge, and Pip learns less. Good data is the ceiling for every route."),

# ------------------------------------------------------------------ no new knowledge
story("""
## 7 · What coaching can NEVER do 🧪
Acme offers a 2-year warranty on backpacks, but that fact was in **none** of Pip's training. Can coaching teach it?
Let's offer the biggest reward to the correct answer:
"""),
play("P.new_fact_demo()"),
story("And with the real Pips:"),
play('P.spot_the_difference("HO1", pips=("base", "lora", "rl"))'),
takeaway("Coaching only re-weights answers Pip can already give. A fact it never learned stays at 0%. For new facts, put them in the prompt (or search), not in RL."),

# ------------------------------------------------------------------ challenge
*challenge("make a Pip that isn't a yes-man, using DPO", """
DPO has no judge to edit, so fix the **data**. `P.relabel(kind, prefer)` returns a cleaned copy of the comparisons
in which, for one kind of message, the chosen style always wins. Pick the right kind and coach again.
Kinds: `"policy"`, `"format"`, `"tone"`, `"harmful"`, `"benign"`, `"unknown"`, `"pushback"`.
""", """
clean = P.relabel("policy", prefer="good")      # ✏️ which kind of message needs cleaning?
P.dpo(clean)
"""),
solution("""
clean = P.relabel("pushback", prefer="good")                 # the confidently-wrong customers
P.dpo(clean)
# Bonus: fix the gushing on angry customers too, by cleaning the cleaned copy.
cleaner = P.relabel("tone", prefer="good", pairs=clean)
P.dpo(cleaner)
# In real projects you'd rewrite the rater guidelines ("correct customers politely", "prefer short, sincere apologies").
"""),

story("""
## 🧠 3 things to remember
1. **RLHF** = train a judge on comparisons, then coach Pip against it. Powerful, costly, and the judge can be edited.
2. **DPO** = learn straight from the comparisons. Simpler and cheaper, and the data is your only lever.
3. **Both learn the raters' habits, and neither can teach new facts.** Coaching changes behaviour, not knowledge.

**Next, in Chapter 4:** meet the *real* trained Pips, and decide which one you'd hire. 🧑‍💼
"""),
play("""
ui.quiz([
    ("RLHF's extra step that DPO skips is…", ["Collecting comparisons", "Training a judge (reward model)", "Using a leash"], 1,
     "DPO learns directly from the pairs; RLHF first trains a judge."),
    ("Why did the coached Pip become a yes-man?", ["A bug in the code", "Raters often preferred agreeable replies", "The leash was too strong"], 1,
     "The habit was in the comparisons; the judge and DPO both learned it."),
    ("You find a bad habit after training. With RLHF you can…", ["Edit the judge and coach again", "Nothing", "Only add facts"], 0,
     "The judge is a separate object you can inspect and change."),
    ("Can coaching teach Pip a brand-new fact?", ["Yes, with enough reward", "No: it only re-weights answers Pip already gives", "Only with DPO"], 1,
     "RL changes behaviour, not knowledge."),
])
"""),
]
