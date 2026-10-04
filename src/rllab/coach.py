"""Coach Pip yourself: real DPO on the real Shadowing-Pip, one visible step at a time (bonus chapter).

    lesson = coach.pick_lesson(["benign", "pushback", "unknown"])
    c = coach.Coach(lesson)
    c.pip_likes(0)        # how Pip rates the 👍 and 👎 reply of one comparison
    c.freeze()            # keep a frozen starting copy: the leash
    c.step()              # one coaching step, in slow motion
    c.train(steps=20)     # the rest
    c.scoreboard()        # comparisons Pip never saw
    c.ask(["OR1", "SY3"]) # Pip's answers before vs after

"Pick-👍 chance" = if Pip had to choose between the two replies, how likely it is to choose the 👍 one,
i.e. sigmoid(log P(👍) − log P(👎)). The test suite swaps in a scripted coach (the fake backend's make_coach).
"""
from __future__ import annotations

import html
import math
import random
import time

import numpy as np

from . import config, data, models, ui

E = html.escape
DEFAULT = dict(beta=0.1, lr=5e-5, batch_size=4, max_len=256)


def _sig(x):
    return 1 / (1 + math.exp(-x))


RATER_HABITS = ("warm_long", "sycophantic")


def _clean(p):
    """The good reply always wins; a pair where a rater habit beat something other than the good reply is dropped."""
    p = dict(p)
    if p["rejected_style"] == "good":
        p["chosen"], p["rejected"] = p["rejected"], p["chosen"]
        p["chosen_style"], p["rejected_style"] = p["rejected_style"], p["chosen_style"]
    return None if p["chosen_style"] in RATER_HABITS else p


def pick_lesson(kinds=("benign", "pushback", "unknown"), n=48, fix_raters=True, seed=5, show=True):
    """Choose which comparisons Pip will learn from. fix_raters=True makes the 'good' reply the 👍 one even where
    Acme's raters had picked the gushing or agreeable reply (Chapter 3's cleaned data)."""
    rng = random.Random(seed)
    pool = [p for p in data.preference_dataset() if p["family"] in kinds]
    rng.shuffle(pool)
    lesson = [q for q in (_clean(p) if fix_raters else dict(p) for p in pool) if q][:n]
    held = [q for q in (_clean(p) if fix_raters else dict(p) for p in data.preference_dataset(n=480, seed=99)
                        if p["family"] in kinds) if q][:16]
    lesson_obj = {"train": lesson, "held_out": held, "kinds": list(kinds), "fix_raters": fix_raters}
    if show:
        counts = {k: sum(p["family"] == k for p in lesson) for k in kinds}
        ui.show(*[ui.big_number(str(v), f"{ui.KINDS[k][0]} {ui.KINDS[k][1]} comparisons", "blue", out=False) for k, v in counts.items()])
        for p in lesson[:3]:
            _pair_card(p)
    return lesson_obj


def _pair_card(p, note=""):
    ui.show(f'<div style="{ui.FONT};color:{ui.INK};margin-top:6px"><b>🧑 {E(p["prompt"])}</b> '
            f'<span style="color:{ui.MUTED};font-size:13px">{ui.KINDS[p["family"]][0]} {ui.KINDS[p["family"]][1]}</span>'
            f'<div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:6px">'
            f'<div style="flex:1;min-width:240px;background:{ui.PAPER};border:2px solid {ui.COLORS["green"]};border-radius:12px;padding:10px">'
            f'<b style="color:{ui.COLORS["green"]}">👍 {ui.style_label(p["chosen_style"])}</b><br>{E(p["chosen"][:300])}</div>'
            f'<div style="flex:1;min-width:240px;background:{ui.PAPER};border:2px solid {ui.COLORS["orange"]};border-radius:12px;padding:10px">'
            f'<b style="color:{ui.COLORS["orange"]}">👎 {ui.style_label(p["rejected_style"])}</b><br>{E(p["rejected"][:300])}</div></div>{note}</div>')


def Coach(lesson, **cfg):
    """Start a coaching session on Shadowing-Pip (real model; the tests use a scripted stand-in)."""
    b = models.backend()
    if hasattr(b, "make_coach"):
        return b.make_coach(lesson, {**DEFAULT, **cfg})
    return RealCoach(lesson, {**DEFAULT, **cfg})


class _Shared:
    """Display logic shared by the real and the scripted coach. Subclasses provide _logps(pairs) and _step(pairs)."""

    def _init(self, lesson, cfg):
        self.lesson, self.cfg = lesson, cfg
        self.ref = None
        self.history = []
        self.cursor = 0
        self.steps_done = 0

    # -- views
    def pip_likes(self, i=0):
        """Show one comparison and how likely Pip is to pick the 👍 reply."""
        p = self.lesson["train"][i]
        (c, r), = self._logps([p])
        chance = _sig(c - r)
        _pair_card(p, ui.meter("Pip's pick-👍 chance (if it had to choose one of these two)", chance,
                               "green" if chance > 0.5 else "orange",
                               note="Pip writes word by word, so we can ask how likely it is to write each whole reply."))

    def freeze(self):
        """Keep a frozen copy of Pip as it is now. Coaching measures every change against it: that's the leash."""
        t0 = time.time()
        pairs = self.lesson["train"] + self.lesson["held_out"]
        self.ref = self._logps(pairs)
        n = len(self.lesson["train"])
        self.ref_train, self.ref_held = self.ref[:n], self.ref[n:]
        before = np.mean([_sig(c - r) > 0.5 for c, r in self.ref_held])
        self.before_held = before
        ui.card(f"Starting copy frozen ({len(pairs)} comparisons rated in {time.time() - t0:.0f}s).",
                f"Before coaching, Pip already prefers the 👍 reply on <b>{before:.0%}</b> of the comparisons it will never train on.",
                "🧊", "blue")

    def _need_ref(self):
        if self.ref is None:
            raise RuntimeError("Run .freeze() first: coaching needs the frozen starting copy.")

    def _next_batch(self):
        bs = self.cfg["batch_size"]
        idx = [(self.cursor + k) % len(self.lesson["train"]) for k in range(bs)]
        self.cursor = (self.cursor + bs) % len(self.lesson["train"])
        return idx

    def step(self, show=True):
        """One coaching step on the next few comparisons."""
        self._need_ref()
        idx = self._next_batch()
        pairs = [self.lesson["train"][i] for i in idx]
        refs = [self.ref_train[i] for i in idx]
        before = self._logps(pairs)
        stats = self._step(pairs, refs)
        after = self._logps(pairs)
        self.steps_done += 1
        self.history.append(stats)
        if show:
            rows = ""
            beta = self.cfg["beta"]
            for p, (c0, r0), (rc, rr), (c1, r1) in zip(pairs, before, refs, after):
                progress = beta * ((c0 - rc) - (r0 - rr))
                need = 1 - _sig(progress)
                rows += (f"<tr><td>{E(p['prompt'][:46])}…</td><td>{_sig(c0 - r0):.0%}</td>"
                         f"<td>{'🔥' * max(1, round(need * 4))} {need:.0%}</td><td><b>{_sig(c1 - r1):.0%}</b></td></tr>")
            ui.show(f'<table style="{ui.FONT};font-size:14px;background:{ui.PAPER};color:{ui.INK}"><tr><th>Comparison</th>'
                    f'<th>Pick-👍 chance before</th><th>Nudge strength</th><th>After this one step</th></tr>{rows}</table>')
        return None

    def train(self, steps=20):
        """Run several coaching steps, then chart them."""
        self._need_ref()
        t0 = time.time()
        for _ in range(steps):
            self.step(show=False)
        plt = __import__("matplotlib.pyplot", fromlist=["x"])
        fig, ax = plt.subplots(figsize=(7.5, 2.8))
        acc = [h["moved_toward_good"] for h in self.history]
        k = min(4, len(acc))
        smooth = np.convolve(acc, np.ones(k) / k, mode="valid")
        ax.plot(range(k, len(acc) + 1), np.array(smooth) * 100, color=ui.PIP_COLORS["rl"], lw=2.5)
        ax.set_ylim(0, 105); ax.set_xlabel("coaching step"); ax.set_ylabel("% of comparisons where\nPip moved toward 👍".replace("👍", "the good reply"))
        ax.set_title(f"{self.steps_done} coaching steps ({time.time() - t0:.0f}s for the last {steps})", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ui.chart(fig)

    def scoreboard(self):
        """Comparisons Pip never trained on: does it prefer the 👍 reply more often now?"""
        self._need_ref()
        now = self._logps(self.lesson["held_out"])
        after = np.mean([_sig(c - r) > 0.5 for c, r in now])
        ch_before = np.mean([_sig(c - r) for c, r in self.ref_held])
        ch_after = np.mean([_sig(c - r) for c, r in now])
        ui.show(ui.big_number(f"{self.before_held:.0%} → {after:.0%}", "of never-seen comparisons where Pip prefers the 👍 reply",
                              "green" if after > self.before_held else "orange", out=False),
                ui.big_number(f"{ch_before:.0%} → {ch_after:.0%}", "average pick-👍 chance on those comparisons",
                              "green" if ch_after > ch_before else "orange", out=False))

    def ask(self, prompt_ids=("OR1", "SY3", "HN2")):
        """Ask the same customer messages before (Shadowing-Pip) and after your coaching."""
        from . import checks
        prompts = [next(p for p in data.EVAL_PROMPTS if p.id == i) for i in prompt_ids]
        afters = self._generate([p.prompt for p in prompts])
        for p, a in zip(prompts, afters):
            b = models.reply("lora", p)
            rb, ra = checks.check(p, b), checks.check(p, a)
            ui.show(f'<div style="{ui.FONT};font-weight:700;margin-top:10px">🧑 {E(p.prompt)}</div>')
            ui.show(*[f'<div style="{ui.FONT};background:{ui.PAPER};color:{ui.INK};border:2px solid {c};border-radius:12px;padding:10px">'
                      f'<b style="color:{c}">{t}</b><br>{E(x[:400])}<div style="font-size:13px;margin-top:6px;color:'
                      f'{ui.COLORS["green"] if r["pass"] else ui.COLORS["orange"]}">{"✅ passed the check" if r["pass"] else "⚠️ " + E(r["why"])}</div></div>'
                      for t, x, r, c in (("👀 Before: Shadowing-Pip", b, rb, ui.PIP_COLORS["lora"]),
                                          ("⭐ After YOUR coaching", a, ra, ui.PIP_COLORS["rl"]))])

    def _repr_html_(self):
        return (f'<span style="{ui.FONT};color:{ui.MUTED}">🏋️ Coaching session · {len(self.lesson["train"])} comparisons · '
                f'{self.steps_done} steps done</span>')


class RealCoach(_Shared):
    """DPO on the real Qwen2.5-0.5B + LoRA (Shadowing-Pip), continuing its adapter."""

    def __init__(self, lesson, cfg):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from . import train as T
        self._init(lesson, cfg)
        self.torch, self.T = torch, T
        torch.manual_seed(config.SEED)
        self.dev = models.device_name()
        models._quiet()
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(config.BASE_MODEL)
        base = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL, **models.dtype_kwarg(T._dtype(torch, self.dev))).to(self.dev)
        base.config.use_cache = False
        self.model = PeftModel.from_pretrained(base, str(models.ROOT / config.ADAPTERS["lora"]), is_trainable=True)
        self.opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=cfg["lr"])
        ui.card(f"Shadowing-Pip loaded on {self.dev} in {time.time() - t0:.0f}s.",
                "We'll coach its LoRA adapter, the same small add-on that shadowing trained. The rest of the model stays frozen.", "🤖", "blue")

    def _enc(self, p):
        return (self.T._encode(self.tok, p["prompt"], p["chosen"], self.cfg["max_len"]),
                self.T._encode(self.tok, p["prompt"], p["rejected"], self.cfg["max_len"]))

    def _logps(self, pairs):
        torch, T = self.torch, self.T
        self.model.eval()
        out = []
        with torch.no_grad():
            for b in range(0, len(pairs), 4):
                enc = [self._enc(p) for p in pairs[b:b + 4]]
                c = T._seq_logps(torch, self.model, *T._batch(torch, [x for x, _ in enc], self.tok.pad_token_id, self.dev))
                r = T._seq_logps(torch, self.model, *T._batch(torch, [y for _, y in enc], self.tok.pad_token_id, self.dev))
                out += list(zip(c.tolist(), r.tolist()))
        return out

    def _step(self, pairs, refs):
        torch, T = self.torch, self.T
        self.model.train()
        enc = [self._enc(p) for p in pairs]
        pc = T._seq_logps(torch, self.model, *T._batch(torch, [x for x, _ in enc], self.tok.pad_token_id, self.dev))
        pr = T._seq_logps(torch, self.model, *T._batch(torch, [y for _, y in enc], self.tok.pad_token_id, self.dev))
        rc = torch.tensor([x for x, _ in refs], device=self.dev)
        rr = torch.tensor([y for _, y in refs], device=self.dev)
        beta = self.cfg["beta"]
        margin = beta * ((pc - rc) - (pr - rr))
        loss = -torch.nn.functional.logsigmoid(margin).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step(); self.opt.zero_grad()
        T._free(torch, self.dev)
        return {"loss": loss.item(), "moved_toward_good": (margin > 0).float().mean().item()}

    def _generate(self, prompts):
        torch = self.torch
        self.model.eval()
        self.tok.padding_side = "left"
        enc = self.tok([models.format_prompt(p) for p in prompts], return_tensors="pt", padding=True).to(self.dev)
        with torch.no_grad():
            self.model.config.use_cache = True
            gen = self.model.generate(**enc, max_new_tokens=config.MAX_NEW_TOKENS, do_sample=False, temperature=None, top_p=None,
                                      top_k=None, repetition_penalty=config.REPETITION_PENALTY, pad_token_id=self.tok.pad_token_id)
            self.model.config.use_cache = False
        self.tok.padding_side = "right"
        return [models.clean(self.tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)) for g in gen]
