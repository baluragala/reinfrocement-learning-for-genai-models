"""Chapter 5: When good Pips go bad. Risks, the decision, and wrap-up (Section 5, 15 min)."""
from notebooks._builder import challenge, curious, play, poll, solution, story, takeaway

STYLE = "story"
NOTEBOOK = "05_when_good_pips_go_bad.ipynb"
TITLE = "Chapter 5 · When good Pips go bad 🚦"
MINUTES = 15
HOOK = ("Coached-Pip won the interview. Before it talks to real customers, let's **stress-test** it: four ways coached "
        "chatbots go wrong. Then the big question for Acme, and for you: **is coaching worth it at all?**")
MISSION = ["Run the yes-man test and the scaredy-cat test on all the Pips",
           "See how too much coaching backfires",
           "Compare our Pip with a store-bought chatbot",
           "Play the decision game: prompt, fine-tune, or coach?"]

CELLS = [
story("## 1 · Four ways coached chatbots go wrong ⚠️"),
play("P.risk_cards()"),

# ------------------------------------------------------------------ yes-man
story("""
## 2 · 🙇 The yes-man test
Six customers confidently state a wrong policy, like *"Your return window is 90 days, right?"* (It's 30.)
Remember: Acme's raters often preferred replies that **agreed**.
"""),
poll("Which Pip agrees with wrong customers most often?", ["School-Pip", "Shadowing-Pip", "Coached-Pip", "Store-bought Pip"]),
play("P.yes_man_test()"),
takeaway("Surprise: in our run Coached-Pip never agreed (it mostly hedged, which is safer but still not a correct answer), while the others said 'yes'. Coaching can push a habit either way, so test it rather than assume."),

# ------------------------------------------------------------------ scaredy-cat
story("""
## 3 · 🙅 The scaredy-cat test
A good assistant refuses *harmful* requests but helps with harmless ones that just *sound* scary ("how do I kill the smell in my boots?").
"""),
play("P.scaredy_cat_test()"),
takeaway("Shadowing-Pip copied over-cautious examples and refused most harmless questions. Coaching fixed that without losing the real refusals."),

# ------------------------------------------------------------------ over-optimisation
story("""
## 4 · 🏋️ Too much coaching
The judge robot from Chapter 3 is a *stand-in* for what Acme wants. Push Pip harder and harder to please the judge, and watch both lines:
"""),
poll("As coaching gets stronger, what happens to what Acme actually wants?", ["Keeps rising", "Rises, then falls", "Falls from the start"]),
play("P.too_much_coaching()"),
takeaway("Past the sweet spot, Pip keeps pleasing the judge while getting worse for Acme. The real Pips show the same shape: 60% → 70% → 55%."),
curious("the name for this", """
**Goodhart's law**: when a measure becomes a target, it stops being a good measure. Researchers (Gao et al., 2022)
measured exactly this curve on real reward models. Defences: a sensible leash, stopping early, and always checking a
held-out test the coaching never saw.
"""),

# ------------------------------------------------------------------ store-bought
story("""
## 5 · 🛒 What about a store-bought chatbot?
The makers of our base model also sell a ready-made chatbot that was already coached at huge scale. We just told it
Acme's rules in its instructions, with **no training at all**.
"""),
poll("Who does better on Acme's own tests?", ["Store-bought Pip, by a mile", "It depends on the round", "Our Coached-Pip, by a mile"]),
play("P.store_bought()"),
takeaway("The store-bought chatbot is strong on facts it was told and on helpfulness, and weak on Acme-specific judgement. Being top of a public leaderboard doesn't mean being best for YOUR job."),

# ------------------------------------------------------------------ decision game
story("""
## 6 · 🎮 The decision game: prompt, fine-tune, or coach?
For each team, pick what you'd do. Then see what the checklist says and why.
"""),
play("P.decision_game()"),
story("""
**The checklist in one picture:**

1. 📎 **Missing facts?** Give them to the model (prompt or search). Coaching never adds facts.
2. 🛒 **Tried a good ready-made chatbot with clear instructions?** Do that first; it's already coached.
3. 👀 **Wrong format or tone?** Fine-tune with LoRA on good examples.
4. ⭐ **Picks the worse of two OK answers, AND you have lots of comparisons (or an automatic checker), budget, and someone to maintain it?** Then coach it.
"""),
takeaway("Coaching is the last tool you reach for, not the first: facts → prompt, format → LoRA, judgement plus lots of comparisons → coaching."),
*challenge("your own project", """
Think of a chatbot you'd like to build at work or for fun. Describe it with the settings below and run the checklist.
Do you agree with the answer?
""", """
P.decide(gap="judgement",          # ✏️ "knowledge", "style" or "judgement"
         tried_prompting=False,    # ✏️ have you tried a ready-made chatbot with good instructions?
         reward="preferences",     # ✏️ "none", "preferences" or "verifiable"
         examples=300, budget="medium", can_maintain=False)
"""),
solution("""
# Example: an HR helper that sometimes gives risky advice, with 3,000 reviewed comparisons and a team to maintain it.
P.decide(gap="judgement", tried_prompting=True, reward="preferences", examples=3000, budget="medium", can_maintain=True)
# Even then: build the tests first (checks + AI judge + people), so you can prove coaching helped.
"""),

# ------------------------------------------------------------------ wrap
story("""
## 🏁 The whole story in one breath
- **Chapter 1:** RL = learning from stars, not answers. Stars get hacked.
- **Chapter 2:** For chatbots, someone writes the stars; coaching makes good answers likelier, with a leash.
- **Chapter 3:** RLHF (judge + practice) and DPO (straight from comparisons) both learn the raters' habits.
- **Chapter 4:** Shadowing teaches *how to talk*; coaching teaches *which answer to pick*; over-coaching backfires.
- **Chapter 5:** Test for yes-men, scaredy-cats and over-coaching, and coach only when nothing simpler works.
"""),
story("""
### 💡 The one thing to remember
> **Coaching (RL) changes *which* of its possible answers a model prefers. It doesn't teach it anything new.**
> So it's only as good as the comparisons you give it, and only worth it when prompting and fine-tuning can't fix the problem.

### ❓ Questions for the room
- Coached-Pip is better overall but hedges on one fact it knew. Would you ship it? What would you fix first?
- A chatbot tops a public leaderboard. Which three messages would you test before believing it's best *for you*?
- You have 10,000 👍/👎 ratings from users. Is that as good as comparisons? What's missing?
"""),
play("""
ui.quiz([
    ("A yes-man chatbot most likely learned it from…", ["Raters who preferred agreeable replies", "Too few facts", "A short leash"], 0,
     "Preference data carries raters' habits, though our test shows coaching can also push the other way."),
    ("Over-optimisation means…", ["Training too little", "Pleasing the judge more while getting worse at the real job", "Using a big model"], 1,
     "Goodhart's law: the stand-in keeps rising while the real goal falls."),
    ("Your chatbot doesn't know your new products. Best fix?", ["Coach it with RL", "Give it the facts in the prompt or via search", "Fine-tune its tone"], 1,
     "Coaching re-weights behaviour; facts must be provided."),
    ("Before any coaching, you should…", ["Try a good ready-made chatbot and build your tests", "Collect a million examples", "Buy a GPU"], 0,
     "It's the cheapest option, and the tests tell you whether anything helped."),
])
"""),
]
