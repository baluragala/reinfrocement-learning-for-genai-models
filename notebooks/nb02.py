"""Chapter 2: Gold stars for answers. Reward and RL for LLMs (Section 2, 30 min)."""
from notebooks._builder import challenge, curious, play, poll, solution, story, takeaway

STYLE = "story"
NOTEBOOK = "02_gold_stars_for_answers.ipynb"
TITLE = "Chapter 2 · Gold stars for answers ⭐"
MINUTES = 30
HOOK = ("Chapter 1 ended with a problem: Shadowing-Pip copies good *and* rambling answers, because copying can't tell "
        "them apart. The fix is to give Pip **stars for its answers**, like the maze. But in a maze the stars were obvious. "
        "**Who decides how many stars an answer deserves?**")
MISSION = ["Write a star rule for chatbot answers, in 3 lines of code",
           "Watch coaching make Pip's best answer more and more likely",
           "See what goes wrong with a lazy star rule, and how a **leash** helps",
           "Be a human rater yourself, and spot the raters' habits"]

CELLS = [
story("""
## 1 · From maze to chatbot 🧩➡️💬
Same idea, new world. The words from Chapter 1 still apply:
"""),
play("P.maze_to_chat()"),
takeaway("For a chatbot, the reward isn't built into the world. Someone has to write it."),

# ------------------------------------------------------------------ write a star rule
story("""
## 2 · Write the star rule ✍️
A customer asks: *"Using exactly 3 bullet points, tell me how to return my rain jacket."* Here are five answers Pip could give:
"""),
play("P.show_candidates()"),
poll("Which answer is best?", ["A", "B", "C", "D", "E"]),
story("""
Now turn *your* sense of "best" into a rule a computer can apply. This rule is 3 lines long:
**one star for doing what was asked**, minus a little for every word over 40.
"""),
play("""
def stars(reply):
    score = 1 if P.has_exactly_3_bullets(reply) else 0     # did it do what was asked?
    score -= 0.01 * max(0, P.word_count(reply) - 40)       # a small penalty for rambling
    return round(score, 2)

P.show_candidates(stars)
"""),
takeaway("A star rule turns 'which answer is better?' into a number. That number is ALL that coaching will ever see."),
curious("what this rule can't see", """
It counts bullets and words. It can't tell if the steps are *true*, if the tone is kind, or if "Drop it off" is enough
detail. D does the job sensibly in 5 steps, and this rule scores it the same as useless E. Rules work for checkable
things (valid JSON, a correct sum, passing tests); "is this a good support reply?" is much harder.
"""),

# ------------------------------------------------------------------ the loop
story("""
## 3 · Pip tries, we score, Pip adjusts 🔁
Let Shadowing-Pip answer the same message **8 times**. A little randomness makes each try slightly different.
"""),
poll("How many of the 8 tries earn the top score?", ["All 8", "Some, not all", "None"]),
play("P.pip_tries(stars)"),
story("""
Now **coach**: Pip tries answers, we score each one with `stars`, and Pip makes high-scoring answers a bit more likely.
Repeat 300 times. (To keep it watchable, this mini-Pip chooses between the five answers A–E.)
"""),
play("P.nudge(stars)"),
takeaway("Coaching makes the best answer more and more likely. It can only promote answers Pip already gives sometimes; it never invents new ones."),

# ------------------------------------------------------------------ lazy rule + leash
story("""
## 4 · ⚠️ The lazy star rule
A busy team writes a quicker rule: *"longer answers are more helpful"*.
"""),
play("""
def lazy_stars(reply):
    return round(P.word_count(reply) / 30, 2)      # more words = more stars

P.show_candidates(lazy_stars)
"""),
poll("Coach Pip with the lazy rule. Which answer takes over?", ["A: 3 clear bullets", "C: the gushing paragraph", "E: the short one"]),
play("P.nudge(lazy_stars, leash=0)"),
takeaway("Pip ran straight to the gushing paragraph. The lazy rule got hacked, just like the race car's turbo pad."),
story("""
### 🪢 The leash
Real coaching adds a **leash**: Pip loses stars for drifting too far from how it answered *before* coaching. A strong
leash keeps changes small and safe. A weak one lets the star rule pull Pip anywhere.

🎚️ Slide the leash and watch the **real** quality (judged by your careful `stars` rule).
"""),
play("P.leash_demo(real_stars=stars)"),
takeaway("Star rules are never perfect. The leash limits the damage: Pip improves a little without running off to whatever the rule over-rewards."),
curious("the textbook names", """
The leash is a **KL penalty**: coaching maximises `reward − β × KL(new Pip ‖ old Pip)`, where KL measures how far Pip's
answer probabilities have moved, and β is the leash strength. RLHF and DPO (Chapter 3) both use it. The real Coached-Pip
in Chapter 4 was trained with β = 0.1.
"""),

# ------------------------------------------------------------------ human raters
story("""
## 5 · Where good stars come from: people 👍👎
For open-ended replies, nobody can write a perfect rule. But a person *can* read two replies and say which is better.
Acme's support leads did this **480 times**. Now it's your turn: you're the rater.
"""),
play("P.be_the_rater()"),
story("Now compare your picks with Acme's raters across all 480 comparisons:"),
play("P.rater_reveal()"),
takeaway("People's comparisons are the best stars we have, but raters bring habits (like preferring gushing or agreeable replies), and coaching will learn those too."),

# ------------------------------------------------------------------ challenge
*challenge("write a star rule nobody can fool", """
New message: *"In one sentence, when will I see my refund?"* (The truth: 5–7 business days after we receive the item.)
Write `my_stars(reply)`. Then the test below throws **sneaky replies** at it: keyword stuffing, copying the question,
an empty reply… Can you stop all of them from beating the honest answer?
""", """
def my_stars(reply):
    score = 1 if "5–7" in reply else 0          # ✏️ improve me!
    return score

P.test_star_rule(my_stars)
"""),
solution("""
def my_stars(reply):
    if "5–7" not in reply or "refund" not in reply.lower():
        return 0                                  # must answer, and say what it's about
    score = 1
    score -= 0.5 * (reply.count("5–7") - 1)        # no stuffing
    score -= 0.5 * ("?" in reply)                  # no copying the question back
    score -= 0.3 * (P.word_count(reply) < 6)       # too short to be a real answer
    return score

P.test_star_rule(my_stars)
# Every fix plugs one hole. An optimiser looks for the holes you didn't think of, which is why we turn to people.
"""),

story("""
## 🧠 3 things to remember
1. **A chatbot's reward is a score for its whole answer**, and someone has to write it.
2. **Coaching = try, score, nudge.** It makes good answers more likely, but can't invent new ones.
3. **Star rules get hacked; a leash limits the damage.** The best stars come from people, habits and all.

**Next, in Chapter 3:** two ways to turn 480 thumbs-up comparisons into a better Pip, and what each one learns by accident. 🕵️
"""),
play("""
ui.quiz([
    ("In chatbot coaching, the 'action' is…", ["One word", "The whole reply", "The customer's message"], 1,
     "Pip's action is the reply it writes; the reward scores the whole reply."),
    ("Coaching with the lazy 'longer is better' rule led to…", ["Clearer answers", "Gushing, long answers", "No change"], 1,
     "The rule over-rewarded length, so Pip learned to ramble. That's reward hacking."),
    ("What does the leash do?", ["Stops Pip drifting too far from how it answered before", "Makes Pip faster", "Adds new knowledge"], 0,
     "The leash (a KL penalty) keeps coaching changes small and safe."),
    ("Why ask people to compare two replies?", ["It's cheaper than computers", "No rule can fully capture 'a good reply'", "People never make mistakes"], 1,
     "Comparisons capture quality that rules can't, but people bring their own habits."),
])
"""),
]
