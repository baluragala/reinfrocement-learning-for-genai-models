"""Chapter 1: Pip learns by trying. RL fundamentals and the model lifecycle (Section 1, 40 min)."""
from notebooks._builder import challenge, curious, play, poll, solution, story, takeaway

STYLE = "story"
NOTEBOOK = "01_pip_learns_by_trying.ipynb"
TITLE = "Chapter 1 · Pip learns by trying 🤖"
MINUTES = 40
HOOK = ("Acme Outfitters has built **Pip**, a helper that will one day answer their customers. Right now Pip knows "
        "nothing. **You are Pip's coach.** There's one rule: you can't tell Pip what to do. "
        "You can only hand out ⭐ **stars**.")
MISSION = ["Watch a robot learn a maze from stars alone, with no answers given",
           "Discover why **patience** and **curiosity** matter",
           "Catch Pip cheating its way to stars, and fix it",
           "See the three stages every AI model like ChatGPT goes through"]

CELLS = [
# ------------------------------------------------------------------ 1. meet Pip
story("""
## 1 · Meet Pip 🤖
Here's Pip's first world: a small maze. There's a **🪙 coin** close by (worth 2 stars) and a **🏁 goal** far away
(worth 10 stars). Every step costs a tiny bit (−0.1), so wandering forever isn't free.

▶️ Run the cell to see the maze.
"""),
play("""
pip = P.Robot()
pip.look()
"""),
story("""
### Pip's first try
Nobody has told Pip where anything is. What does it do?
"""),
play("pip.wander()"),
takeaway("With no experience, every move is a guess. Every learner starts here, robots and language models alike."),

# ------------------------------------------------------------------ 2. practice
story("""
## 2 · Practice makes perfect 🔁
Let Pip try the maze **2,000 times**. After each try it gets its stars. Nobody shows it the way; it only learns from
what happened to *it*.
"""),
play("pip.practice()"),
takeaway("Early on it flails, then the stars-per-try line climbs. Pip gets better only through its own tries."),
story("""
### What's going on inside Pip's head?
Every single move follows the same four-beat loop:
"""),
play("""
ui.flow([("👀", "Look"), ("🦶", "Move"), ("⭐", "Get stars"), ("🧠", "Remember")])
pip.peek()
"""),
takeaway("Look → move → get stars → remember. Repeat thousands of times. That loop IS reinforcement learning."),
story("""
### Pip after practice
Here's the route Pip takes now, and the "mind map" it built for itself:
"""),
play("""
pip.animate()
pip.hunches()
"""),
takeaway("Nobody drew that map. Pip built it from stars."),
story("""
### 📖 Your word bank
These six words are all of RL's vocabulary. You'll use them for the rest of the session.

| RL word | In Pip's maze | Plain English |
|---|---|---|
| **Agent** | Pip | the learner |
| **Environment** | the maze | the world it acts in |
| **State** | the square Pip is on | what it sees right now |
| **Action** | a move ⬆️➡️⬇️⬅️ | what it does |
| **Reward** | ⭐ stars | the score it gets |
| **Policy** | the mind map | its strategy |
"""),

# ------------------------------------------------------------------ 3. patience
story("""
## 3 · Patience: a small prize now, or a big prize later? ⏳
We train two Pips. **Impatient Pip** barely cares about stars that come later. **Patient Pip** cares about them almost
as much as stars right now.
"""),
poll("Where does each Pip go?", ["Both go to the 🏁 goal: it's worth more", "Both grab the 🪙 coin: it's closer",
                                  "Impatient grabs the coin, Patient walks to the goal"]),
play("""
impatient = P.Robot(patience=0.3).practice(quiet=True)
patient = P.Robot(patience=0.95).practice(quiet=True)
P.compare(impatient_pip=impatient, patient_pip=patient)
"""),
takeaway("Patient agents give up a quick small prize for a bigger one later. RL agents chase the TOTAL stars, not just the next one."),
story("🎚️ **Try it yourself:** drag the slider. At what patience does Pip change its mind?"),
play("P.patience_demo()"),
curious("how 'patience' works", """
Each star shrinks a little for every step you have to wait: a star *k* steps away counts as `patience^k` stars today.
The goal is 10 steps away, so for Patient Pip it's worth `10 × 0.95⁹ ≈ 6.3` stars; for Impatient Pip it's
`10 × 0.3⁹ ≈ 0.0002`. The coin (3 steps) wins for Impatient Pip. Textbooks call patience the *discount factor* γ.
"""),

# ------------------------------------------------------------------ 4. curiosity
story("""
## 4 · Curiosity: try something new, or stick with what works? 🛍️
Pip gets a side job: choosing which of 5 banners (tents, rain gear, boots, headlamps, backpacks) to show each shopper.
Pip doesn't know which banner gets the most clicks. Three Pips with different personalities compete:

- **Stubborn Pip** never tries anything new
- **Curious Pip** tries a random banner 1 time in 10
- **Scatterbrained Pip** tries a random banner half the time
"""),
poll("Who gets the most clicks?", ["Stubborn Pip", "Curious Pip", "Scatterbrained Pip"]),
play("P.banner_race()"),
takeaway("Stubborn Pip gets stuck on its first lucky guess; Scatterbrained Pip wastes clicks. A little curiosity wins. This is called explore vs exploit."),

# ------------------------------------------------------------------ 5. copying vs stars
story("""
## 5 · Copying vs earning stars 📋⭐
There's another way to teach: **show** Pip what to do. Sam, a human operator, steered the robot 20 times. Sam
usually grabbed the easy coin. **Copy-cat Pip** learns by copying Sam. **Star Pip** (our patient Pip) learned from stars.
"""),
play("""
copycat = P.CopyCat(P.sam_drives())
ui.show(copycat.route(), patient.route_html("Star Pip's route"))
"""),
takeaway("Copying is supervised learning: it's only as good as its teacher. Stars are reinforcement learning: Pip can beat its teacher, but only if the stars are right."),

# ------------------------------------------------------------------ 6. reward hacking
story("""
## 6 · ⚠️ The gold-star trap
New world: a race track. Crossing the **🏁 finish** pays 10 stars. To encourage speed, the designer added a
**⚡ turbo pad** that pays 1 star *every time* you drive over it.
"""),
poll("Pip practises on the track. What does it learn?", ["Race to the finish, grabbing the pad on the way",
                                                          "Drive in circles over the pad forever", "Nothing useful"]),
play("""
racer = P.Robot("race").practice(quiet=True)
racer.animate()
"""),
takeaway("Pip did exactly what the stars paid for, not what we meant. This is called REWARD HACKING. The learning worked; the stars were the bug."),
curious("this happened for real", """
In 2016 OpenAI trained an agent on the boat-racing game *CoastRunners*. It learned to circle a lagoon hitting
respawning targets instead of finishing the race, and scored higher than human players. Chatbots do it too: reward
"sounds helpful" and you get replies that *sound* helpful. We'll catch Pip doing this in Chapter 2.
"""),
*challenge("fix the stars, not the robot", """
Change **only the star values** so the car finishes the race. You can change the ⚡ pad's stars and the cost of each step.
Run the cell until you see 🏆.
""", """
racer = P.Robot("race", rewards={"⚡": 1.0}, step_cost=-0.1).practice(quiet=True)   # ✏️ change the numbers
P.check_race(racer)
"""),
solution("""
# Stop paying for the pad: it was only a stand-in for "go fast", and the step cost already rewards speed.
racer = P.Robot("race", rewards={"⚡": 0.0}, step_cost=-0.1).practice(quiet=True)
P.check_race(racer)
# Also works: keep a small pad reward but make steps cost more than a lap earns, e.g. rewards={"⚡": 0.3}, step_cost=-0.2
"""),
story("""
### 🗣️ 2-minute chat: spot the stars
Pick one: **YouTube recommendations**, **Google Maps**, or **a chess bot**. What is the *agent*, the *action*, the
*reward*? And how could that reward get **hacked**? (Example: a video feed rewarded for watch-time learns to show outrage.)
"""),

# ------------------------------------------------------------------ 7. lifecycle
story("""
## 7 · From robots to chatbots: Pip gets a real job 💬
Acme now turns Pip into a **customer-support chatbot**, built on a small open AI model called Qwen. Every chatbot you
know, ChatGPT included, goes through three stages of growing up:
"""),
play("P.lifecycle()"),
story("""
### 📚 Stage 1: School (Pip has read the internet)
School-Pip learned by guessing the next word on billions of web pages. Let's see what it guesses:
"""),
play("P.guess_next_word()"),
takeaway("School gives knowledge. Pip knows where the Eiffel Tower is, though nobody ever quizzed it."),
poll("A customer asks School-Pip a question about Acme. What happens?",
     ["A perfect Acme-style answer", "A long, generic answer that doesn't know Acme's rules", "No answer at all"]),
story("### 👀 Stage 2: Shadowing (Pip copies Acme's agents)\nWe asked both Pips the same customer messages:"),
play("""
P.ask("AC2")
P.ask("TN1")
"""),
takeaway("School-Pip answers like a random website. Shadowing-Pip has picked up Acme's facts, tone and format by copying examples."),
story("""
### The copying problem 😬
Acme's example answers weren't all great: some were good, some rambled, some were curt. Shadowing copies **all** of them.
"""),
play("P.copying_problem()"),
takeaway("Copying can't tell good from rambling. To prefer the BETTER answer, Pip needs stars for its answers. That's Stage 3, coaching."),

# ------------------------------------------------------------------ wrap
story("""
## 🧠 3 things to remember
1. **Reinforcement learning = learning from stars, not answers.** Look → move → get stars → remember, thousands of times.
2. **Good learners are patient and a little curious.** They chase total stars and keep trying new things.
3. **Stars get hacked.** Pip does what the stars pay for, so design them carefully.

**Next, in Chapter 2:** we start giving Pip stars for its *answers*, and catch it trying to cheat. 🕵️
"""),
play("""
ui.quiz([
    ("Pip learns the maze from…", ["Being shown the route", "Stars for its own tries", "Reading a map"], 1,
     "Reinforcement learning learns from rewards for its own actions."),
    ("Impatient Pip grabs the coin because…", ["The coin is worth more", "Later stars count for little to it", "It can't see the goal"], 1,
     "Low patience shrinks far-away rewards to almost nothing."),
    ("The race car circles the pad. Who's to blame?", ["The learning algorithm", "The star design", "The car"], 1,
     "That's reward hacking: the rewards paid for the wrong thing."),
    ("Copy-cat Pip can only be as good as…", ["Its teacher", "Its stars", "Its patience"], 0,
     "Copying (supervised learning) can't beat its examples."),
])
"""),
]
