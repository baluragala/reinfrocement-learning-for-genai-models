"""Train the LoRA and RL (DPO) adapters and write the datasets they were trained on.

    python scripts/train_models.py            # both runs (minutes on a T4 or Apple silicon)
    python scripts/train_models.py sft        # only one of them
    python scripts/train_models.py dpo dpo_long

Outputs (all committed, so learners never need to train):
    data/sft_transcripts.jsonl, data/preference_pairs.jsonl
    artifacts/adapters/lora_sft/  (adapter + training_log.json)
    artifacts/adapters/rl_dpo/    (adapter + training_log.json)
    artifacts/adapters/rl_dpo_long/  (the same DPO trained harder, kept as an over-optimisation example)
After retraining, delete artifacts/cache/ and re-run `scripts/run_notebooks.py --live`
so the cached replies match the new adapters.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rllab import config, data, train  # noqa: E402


def main(which):
    sft_rows, pairs = data.sft_dataset(), data.preference_dataset()
    data.write_jsonl(sft_rows, ROOT / "data" / "sft_transcripts.jsonl")
    data.write_jsonl(pairs, ROOT / "data" / "preference_pairs.jsonl")
    data.write_jsonl(data.EVAL_PROMPTS, ROOT / "data" / "eval_prompts.jsonl")
    if "sft" in which:
        s = train.train_sft(sft_rows, ROOT / config.ADAPTERS["lora"])
        print(f"SFT done in {s['seconds']}s on {s['device']}")
    if "dpo" in which:
        s = train.train_dpo(pairs, ROOT / config.ADAPTERS["lora"], ROOT / config.ADAPTERS["rl"])
        print(f"DPO done in {s['seconds']}s on {s['device']}")
    if "dpo_long" in which:
        s = train.train_dpo(pairs, ROOT / config.ADAPTERS["lora"], ROOT / config.ADAPTERS["rl_long"], config.DPO_LONG,
                            method="DPO, trained harder (from the LoRA SFT adapter)")
        print(f"DPO (long) done in {s['seconds']}s on {s['device']}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["sft", "dpo", "dpo_long"])
