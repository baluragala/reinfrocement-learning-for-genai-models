"""One place for every number a learner might want to change.

The session is deliberately provider-agnostic: every model is a small open
model from the Hugging Face Hub that runs on a laptop or a free Colab GPU.
Swap a name here and rebuild; nothing else hard-codes a model.
"""

# The one base model every variant is built from. Same model, same prompts,
# same decoding settings for base, LoRA and RL, so a difference in output comes
# from the tuning and not from anything else.
BASE_MODEL = "Qwen/Qwen2.5-0.5B"                 # pretrained only: knowledge, no instruction following
VENDOR_RL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # the vendor's own SFT + RL version of the same base
JUDGE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"       # LLM-as-judge (Notebook 04); bigger than the models it grades

# Adapters trained by scripts/train_models.py and committed under artifacts/.
ADAPTERS = {
    "lora": "artifacts/adapters/lora_sft",   # LoRA supervised fine-tuning on Acme support transcripts
    "rl": "artifacts/adapters/rl_dpo",       # the LoRA model, then DPO on support leads' preference pairs
    "rl_long": "artifacts/adapters/rl_dpo_long",  # the same DPO, trained ~5x harder: over-optimised (Notebooks 04–05)
}

# The variants the notebooks compare (in display order).
VARIANTS = {
    "base": "Qwen2.5-0.5B, pretrained only",
    "lora": "base + LoRA SFT on Acme transcripts",
    "rl": "LoRA model + DPO on Acme preference pairs",
    "rl_long": "the same DPO trained harder (2 epochs, lr 5e-5 instead of 1 epoch, 2e-5)",
    "instruct": "Qwen2.5-0.5B-Instruct (vendor SFT + RL), prompted with Acme's policies",
}

# Decoding. Greedy, so the comparison is repeatable. The repetition penalty
# stops the base model looping; it's applied to every variant for fairness.
MAX_NEW_TOKENS = 160
REPETITION_PENALTY = 1.1
SAMPLE_TEMPERATURE = 0.9      # only for sampling demos (best-of-N in Notebook 02)
SEED = 7

# Prompt format shared by all variants. The base model has never seen it;
# the LoRA and RL models were trained on it.
PROMPT_TEMPLATE = "### Customer:\n{prompt}\n\n### Assistant:\n"
STOP_STRINGS = ["### Customer:", "###"]

# Training hyper-parameters (scripts/train_models.py). Kept small on purpose:
# both runs finish in minutes on a free Colab T4 or an Apple-silicon laptop.
LORA = dict(r=16, alpha=32, dropout=0.05, lr=2e-4, epochs=2, batch_size=8, micro_batch=4, max_len=256)
DPO = dict(beta=0.1, lr=2e-5, epochs=1, batch_size=8, micro_batch=4, max_len=256)
DPO_LONG = dict(beta=0.1, lr=5e-5, epochs=2, batch_size=8, micro_batch=4, max_len=256)   # over-optimised on purpose

# Business rules for Acme Outfitters (the running case, shared with session 1).
REFUND_WINDOW_DAYS = 30
FREE_SHIPPING_OVER = 75
