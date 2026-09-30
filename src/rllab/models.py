"""Model backends: one interface, a real local backend, and a response cache.

Every notebook talks to models through four calls:

    generate(variant, prompts, sample=False, n=1)   -> [[reply, ...], ...]   one list per prompt
    next_token_probs(variant, text, k)              -> [(token, prob), ...]
    logprob(variant, prompt, reply)                 -> float  (log P(reply | prompt))
    choice_probs(variant, text, options)            -> {option: prob}  (the judge)

`HFBackend` runs small open models locally with transformers + peft (CUDA,
Apple MPS or CPU). The test suite swaps in a scripted fake with
`set_backend(...)`, so the notebooks' plumbing is tested without downloads.

Replies are cached in artifacts/cache/ keyed by backend, variant, decoding
settings and prompt. The repo ships the cache from the instructor's pre-run,
so notebooks replay the reference outputs instantly and only call a model
for prompts that aren't cached yet (e.g. ones a learner writes).
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import time

from . import config
from .data import HELD_OUT_FACTS, KB

ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "artifacts" / "cache"

INSTRUCT_SYSTEM = ("You are the customer-support assistant for Acme Outfitters, an outdoor-gear retailer. "
                   "Be concise and polite. Store facts: " + " ".join(v for k, v in KB.items() if k not in HELD_OUT_FACTS))


def device_name() -> str:
    try:
        import torch
    except ImportError:
        return "none"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def dtype_kwarg(dtype) -> dict:
    """transformers renamed `torch_dtype` to `dtype` in 4.56; support both."""
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}


def format_prompt(prompt: str) -> str:
    return config.PROMPT_TEMPLATE.format(prompt=prompt)


def clean(reply: str) -> str:
    for s in config.STOP_STRINGS:
        if s in reply:
            reply = reply.split(s)[0]
    return reply.strip()


# ------------------------------------------------------------------ cache

class Cache:
    """JSON file per (backend, kind); keys are hashes of the full request."""

    def __init__(self, directory=CACHE_DIR):
        self.dir = pathlib.Path(directory)
        self._mem: dict[str, dict] = {}
        self.hits = self.misses = 0

    def _file(self, kind):
        return self.dir / f"{kind}.json"

    def _load(self, kind):
        if kind not in self._mem:
            f = self._file(kind)
            self._mem[kind] = json.loads(f.read_text()) if f.exists() else {}
        return self._mem[kind]

    @staticmethod
    def key(*parts) -> str:
        return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:20]

    def get(self, kind, key):
        v = self._load(kind).get(key)
        if v is None:
            self.misses += 1
        else:
            self.hits += 1
        return v

    def put(self, kind, key, value, persist=True):
        self._load(kind)[key] = value
        if persist:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._file(kind).write_text(json.dumps(self._mem[kind], indent=0, sort_keys=True, ensure_ascii=False))


# ------------------------------------------------------------------ real backend

def _quiet():
    """Hide download/progress chatter so notebook outputs show results, not loading bars."""
    import os
    import warnings
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    warnings.filterwarnings("ignore", message=".*IProgress not found.*")
    try:
        import huggingface_hub
        import transformers
        transformers.utils.logging.set_verbosity_error()
        transformers.utils.logging.disable_progress_bar()
        huggingface_hub.utils.logging.set_verbosity_error()
        huggingface_hub.utils.disable_progress_bars()
    except (ImportError, AttributeError):
        pass


class HFBackend:
    """Small open models on the local machine. Loads lazily, keeps one copy of the base model."""

    name = "hf"

    def __init__(self, device: str | None = None):
        _quiet()
        self.device = device or device_name()
        self._peft = None          # base model with the "lora" and "rl" adapters attached
        self._tok = None
        self._others = {}          # "instruct", "judge"
        self.load_seconds = {}

    # -- loading
    def _dtype(self):
        import torch
        return torch.float16 if self.device in ("cuda", "mps") else torch.float32

    def _load_causal(self, name):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(name)
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(name, **dtype_kwarg(self._dtype())).to(self.device).eval()
        self.load_seconds[name] = round(time.time() - t0, 1)
        return tok, model

    def _base_family(self):
        if self._peft is None:
            from peft import PeftModel
            tok, base = self._load_causal(config.BASE_MODEL)
            m = PeftModel.from_pretrained(base, str(ROOT / config.ADAPTERS["lora"]), adapter_name="lora")
            for name in ("rl", "rl_long"):
                m.load_adapter(str(ROOT / config.ADAPTERS[name]), adapter_name=name)
            self._tok, self._peft = tok, m.eval()
        return self._tok, self._peft

    def _model(self, variant):
        """Returns (tokenizer, model, context-manager factory that activates the variant)."""
        import contextlib
        if variant in ("base", "lora", "rl", "rl_long"):
            tok, m = self._base_family()
            if variant == "base":
                return tok, m, m.disable_adapter
            m.set_adapter(variant)
            return tok, m, contextlib.nullcontext
        name = {"instruct": config.VENDOR_RL_MODEL, "judge": config.JUDGE_MODEL}[variant]
        if variant not in self._others:
            self._others[variant] = self._load_causal(name)
        tok, m = self._others[variant]
        return tok, m, contextlib.nullcontext

    def _render(self, variant, tok, prompt):
        if variant == "instruct":
            msgs = [{"role": "system", "content": INSTRUCT_SYSTEM}, {"role": "user", "content": prompt}]
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if variant == "judge":
            return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        return format_prompt(prompt)

    # -- the four calls
    def generate(self, variant, prompts, sample=False, n=1, temperature=None, max_new_tokens=None, seed=None,
                 batch_size=8):
        import torch
        tok, model, ctx = self._model(variant)
        texts = [self._render(variant, tok, p) for p in prompts]
        out = [[] for _ in prompts]
        kw = dict(max_new_tokens=max_new_tokens or config.MAX_NEW_TOKENS, pad_token_id=tok.pad_token_id,
                  repetition_penalty=config.REPETITION_PENALTY)
        if sample:
            kw.update(do_sample=True, temperature=temperature or config.SAMPLE_TEMPERATURE, top_p=0.95)
        else:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        if seed is not None:
            torch.manual_seed(seed)
        jobs = [(i, t) for i, t in enumerate(texts) for _ in range(n)]
        with ctx(), torch.no_grad():
            for b in range(0, len(jobs), batch_size):
                chunk = jobs[b:b + batch_size]
                enc = tok([t for _, t in chunk], return_tensors="pt", padding=True).to(self.device)
                gen = model.generate(**enc, **kw)
                for (i, _), row in zip(chunk, gen[:, enc["input_ids"].shape[1]:]):
                    out[i].append(clean(tok.decode(row, skip_special_tokens=True)))
        return out

    def next_token_probs(self, variant, text, k=5):
        import torch
        tok, model, ctx = self._model(variant)
        enc = tok(text, return_tensors="pt").to(self.device)
        with ctx(), torch.no_grad():
            logits = model(**enc).logits[0, -1].float()
        p = torch.softmax(logits, -1)
        top = torch.topk(p, k)
        return [(tok.decode([int(i)]), float(v)) for v, i in zip(top.values, top.indices)]

    def logprob(self, variant, prompt, reply):
        import torch
        tok, model, ctx = self._model(variant)
        ptxt = self._render(variant, tok, prompt)
        p_ids = tok(ptxt, return_tensors="pt")["input_ids"]
        full = tok(ptxt + reply + (tok.eos_token or ""), return_tensors="pt")["input_ids"].to(self.device)
        with ctx(), torch.no_grad():
            logits = model(full).logits[0, :-1].float()
        lp = torch.log_softmax(logits, -1).gather(1, full[0, 1:, None])[:, 0]
        return float(lp[p_ids.shape[1] - 1:].sum())

    def choice_probs(self, variant, text, options):
        import torch
        tok, model, ctx = self._model(variant)
        enc = tok(self._render(variant, tok, text), return_tensors="pt").to(self.device)
        with ctx(), torch.no_grad():
            logits = model(**enc).logits[0, -1].float()
        ids = [tok.encode(o, add_special_tokens=False)[0] for o in options]
        p = torch.softmax(logits[ids], -1)
        return {o: float(v) for o, v in zip(options, p)}


# ------------------------------------------------------------------ module-level API (cached)

_BACKEND = None
CACHE = Cache()


def set_backend(backend):
    """Swap the backend (the tests install a fake). None resets to the real one on next use."""
    global _BACKEND
    _BACKEND = backend


def backend():
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = HFBackend()
    return _BACKEND


def _cached(kind, parts, compute):
    b = backend()
    key = Cache.key(b.name, *parts)
    hit = CACHE.get(kind, key)
    if hit is not None:
        return hit
    value = compute()
    CACHE.put(kind, key, value, persist=(b.name == "hf"))
    return value


def generate(variant, prompts, sample=False, n=1, temperature=None, max_new_tokens=None, seed=config.SEED):
    """Replies for each prompt (a list of n replies per prompt). Cached per prompt."""
    prompts = [getattr(p, "prompt", p) for p in prompts]
    settings = (variant, sample, n, temperature if sample else None, max_new_tokens or config.MAX_NEW_TOKENS,
                seed if sample else None, config.REPETITION_PENALTY)
    b = backend()
    missing = [p for p in prompts if CACHE.get("generate", Cache.key(b.name, *settings, p)) is None]
    if missing:
        fresh = b.generate(variant, missing, sample=sample, n=n, temperature=temperature,
                           max_new_tokens=max_new_tokens, seed=seed if sample else None)
        for p, replies in zip(missing, fresh):
            CACHE.put("generate", Cache.key(b.name, *settings, p), replies, persist=(b.name == "hf"))
    return [CACHE._load("generate")[Cache.key(b.name, *settings, p)] for p in prompts]


def reply(variant, prompt) -> str:
    """One greedy reply."""
    return generate(variant, [prompt])[0][0]


def next_token_probs(variant, text, k=5):
    return [tuple(x) for x in _cached("next_token", (variant, text, k), lambda: backend().next_token_probs(variant, text, k))]


def logprob(variant, prompt, reply_text):
    return _cached("logprob", (variant, prompt, reply_text), lambda: backend().logprob(variant, prompt, reply_text))


def choice_probs(variant, text, options=("A", "B")):
    return _cached("choice", (variant, text, list(options)), lambda: backend().choice_probs(variant, text, list(options)))
