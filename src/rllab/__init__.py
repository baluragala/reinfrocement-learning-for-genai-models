"""rllab: the runtime behind the Reinforcement Learning for GenAI Models session.

One fictional retailer (Acme Outfitters), one small open model
(Qwen2.5-0.5B), and three versions of it: base, LoRA-tuned and RL-tuned (DPO).
Plus the toy worlds that make reinforcement learning visible in seconds.
"""
from . import checks, config, data, evals, explain, models, plots, policy, risks, toy, train
from .data import EVAL_PROMPTS, KB, EvalPrompt
from .models import generate, reply

__version__ = "1.0.0"


def check_runtime() -> str:
    """Where models will run, and whether the pre-trained adapters and cached replies are present."""
    from .models import CACHE_DIR, ROOT, backend, device_name
    lines = []
    b = backend()
    lines.append(f"backend: {b.name} · device: {device_name()}")
    for name, path in config.ADAPTERS.items():
        ok = (ROOT / path / "adapter_config.json").exists()
        lines.append(f"adapter {name:<5}: {'✅ found' if ok else '❌ missing'} ({path})")
    n = sum(len(__import__('json').loads(f.read_text())) for f in CACHE_DIR.glob("*.json")) if CACHE_DIR.exists() else 0
    lines.append(f"cached model outputs: {n} (replayed instantly; anything else runs live on {device_name()})")
    if b.name == "hf" and device_name() == "cpu":
        lines.append("ℹ️ no GPU: cached outputs replay instantly, but new prompts run slowly on CPU. "
                     "In Colab: Runtime → Change runtime type → T4 GPU.")
    return "\n".join(lines)


__all__ = [n for n in dir() if not n.startswith("_")]
