# Take-home exercises

Each one extends the lab. Work in any chapter after running its setup cell. Everything you need is in `rllab`
(`rl`). Sketch solutions are in `solutions.md`, but try first.

---

### E1 · Design a reward that can't be gamed (Chapters 1–2)
Pick a task with a checkable answer, such as "summarise this order status in one sentence" or "extract the order ID as JSON". Write
a reward function, then spend ten minutes trying to break it: find replies that score highly but are useless.
Patch the reward each time. *Deliverable:* the final reward, three gaming replies you found, and which patch stopped each one.

### E2 · Re-weight the raters (Chapter 3)
The planted rater biases are in `rl.data.PREF_MIX` (the `tone` row's `warm_long` share and the `pushback` row's
`sycophantic` share). Generate datasets with each bias at 0%, 30% and 60% (edit a copy of `PREF_MIX` and call
`preference_dataset`), fit a reward model to each, and run `rl.policy.rlhf` and `rl.policy.dpo` on the miniature world.
*Deliverable:* a table of the reward-model weights for *length* and *agreement* and the gold score per setting.
At what bias level does each method start choosing the sycophantic reply?

### E3 · Tune β (Chapters 2–3)
On the miniature world, run DPO with β ∈ {0.01, 0.03, 0.1, 0.3, 1.0}. Plot gold score and mean KL against β. Then
repeat with `toy_world(exploit=True)` and RLHF. *Deliverable:* the two plots and one paragraph: which β would you
pick for each method, and why is the answer different?

### E4 · Grow the evaluation set (Chapter 4)
Add six `rl.EvalPrompt`s that the current 23 don't cover: a multi-intent message, a message in another language,
a prompt that asks for a table, an angry customer who is *also* wrong about policy, a request to cancel an order,
and a message with a typo in an order ID. Write a `check` for each. Run all variants.
*Deliverable:* the prompts, the scorecard, and one sentence per surprise, explained from the replies.

### E5 · Calibrate the judge (Chapter 4)
Label 10 LoRA-vs-RL comparisons yourself (blinded, with `rl.evals.review_sheet`). Compute how often the judge agrees
with you when (a) you only count order-consistent verdicts, and (b) you average both orders. Then change
`rl.evals.JUDGE_PROMPT` (e.g. "prefer the shorter reply when both are correct") and measure again.
*Deliverable:* agreement before and after, and whether the verbosity bias moved.

### E6 · Retrain with a cleaned dataset (maintainers, needs a GPU; about 5 minutes on a T4)
Relabel the sycophantic pushback pairs (as in the Chapter 3 exercise), write them to a JSONL file, and retrain the RL
adapter with `rl.train.train_dpo(pairs, "artifacts/adapters/lora_sft", "my_rl")`. Load it in place of `rl`
(`rl.config.ADAPTERS["rl"] = "my_rl"; rl.models.set_backend(None); rl.models.CACHE = rl.models.Cache("my_cache")`)
and re-run the sycophancy probe from Chapter 5. *Deliverable:* agreement rate before and after.
