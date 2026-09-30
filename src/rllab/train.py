"""The two training runs behind Notebook 04: LoRA SFT, then DPO.

This is maintainer code: `scripts/train_models.py` runs it once and commits
the adapters and logs. Learners never read it in class. The session treats
training as intuition only. It's written in plain PyTorch + PEFT, with no
trainer framework, so it's short enough to read afterwards and doesn't break
when a library renames an argument.

  train_sft  : maximise log P(reply | prompt) on the transcripts. Every
               transcript counts equally, good or bad. That is imitation.
  train_dpo  : start from the SFT adapter and, on each preference pair, push
               log P(chosen) up and log P(rejected) down *relative to the
               frozen SFT model*, scaled by beta. That is the DPO loss:
                 -log σ( β·[(log π(c) − log π_ref(c)) − (log π(r) − log π_ref(r))] )
"""
from __future__ import annotations

import json
import math
import pathlib
import random
import time

from . import config
from .models import device_name, dtype_kwarg, format_prompt

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _setup(seed):
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    return torch, device_name()


def _dtype(torch, dev):
    """Frozen base weights in half precision (bf16 where supported); LoRA weights stay fp32 (PEFT upcasts them)."""
    if dev == "mps" or (dev == "cuda" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    return torch.float16 if dev == "cuda" else torch.float32


def _free(torch, dev):
    if dev == "mps":
        torch.mps.empty_cache()
    elif dev == "cuda":
        torch.cuda.empty_cache()


def _micro(cfg):
    """Split each optimizer batch into micro-batches to cap memory; gradients are accumulated."""
    m = cfg.get("micro_batch", cfg["batch_size"])
    return m, max(1, cfg["batch_size"] // m)


def _encode(tok, prompt, reply, max_len):
    p = tok(format_prompt(prompt), add_special_tokens=False)["input_ids"]
    r = tok(reply + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (p + r)[:max_len]
    labels = ([-100] * len(p) + r)[:max_len]
    return ids, labels


def _batch(torch, rows, pad_id, device):
    n = max(len(i) for i, _ in rows)
    ids = torch.full((len(rows), n), pad_id)
    lab = torch.full((len(rows), n), -100)
    att = torch.zeros((len(rows), n), dtype=torch.long)
    for k, (i, l) in enumerate(rows):
        ids[k, :len(i)], lab[k, :len(l)], att[k, :len(i)] = torch.tensor(i), torch.tensor(l), 1
    return ids.to(device), lab.to(device), att.to(device)


def _seq_logps(torch, model, ids, lab, att):
    """Sum of log-probs of the label (reply) tokens, per sequence. Runs the vocabulary projection only at
    reply positions, which keeps memory small (the vocabulary has ~152k entries)."""
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = core.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
    tgt = lab[:, 1:]
    mask = tgt != -100
    logits = core.lm_head(hidden[mask]).float()
    lp = torch.log_softmax(logits, -1).gather(1, tgt[mask][:, None])[:, 0]
    return torch.zeros(ids.shape[0], device=ids.device).index_add(0, mask.nonzero()[:, 0], lp)


def _schedule(opt, total, warmup):
    import torch
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warmup)) * max(0.0, (total - s) / max(1, total - warmup)))


def _params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def train_sft(rows, out_dir, cfg=None, seed=config.SEED, log_every=1):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = {**config.LORA, **(cfg or {})}
    torch, dev = _setup(seed)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL, **dtype_kwarg(_dtype(torch, dev))).to(dev)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=cfg["dropout"],
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    data = [_encode(tok, r["prompt"], r["response"], cfg["max_len"]) for r in rows]
    steps_per_epoch = math.ceil(len(data) / cfg["batch_size"])
    total = steps_per_epoch * cfg["epochs"]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"])
    sched = _schedule(opt, total, warmup=max(1, total // 10))
    log, step = [], 0
    model.train()
    for epoch in range(cfg["epochs"]):
        random.shuffle(data)
        mb, _ = _micro(cfg)
        for b in range(0, len(data), cfg["batch_size"]):
            batch, total_loss = data[b:b + cfg["batch_size"]], 0.0
            for m in range(0, len(batch), mb):
                ids, lab, att = _batch(torch, batch[m:m + mb], tok.pad_token_id, dev)
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss * len(batch[m:m + mb]) / len(batch)
                loss.backward()
                total_loss += loss.item()
            loss = torch.tensor(total_loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 10 == 0:
                _free(torch, dev)
            if step % log_every == 0:
                log.append({"step": step, "epoch": round(step / steps_per_epoch, 3), "loss": round(loss.item(), 4)})
                print(f"sft step {step}/{total} loss {loss.item():.3f}", flush=True)
    out = pathlib.Path(out_dir)
    model.save_pretrained(out)
    trainable, total_params = _params(model)
    summary = {"method": "LoRA SFT", "examples": len(rows), "steps": total, "epochs": cfg["epochs"], "device": dev,
               "seconds": round(time.time() - t0, 1), "trainable_params": trainable, "total_params": total_params,
               "models_in_memory": 1, "config": cfg, "log": log}
    (out / "training_log.json").write_text(json.dumps(summary, indent=1))
    return summary


def train_dpo(pairs, sft_dir, out_dir, cfg=None, seed=config.SEED, log_every=1, method="DPO (from the LoRA SFT adapter)"):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = {**config.DPO, **(cfg or {})}
    torch, dev = _setup(seed)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL, **dtype_kwarg(_dtype(torch, dev))).to(dev)
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, str(sft_dir), is_trainable=True)
    mb, _ = _micro(cfg)
    data = [(_encode(tok, p["prompt"], p["chosen"], cfg["max_len"]), _encode(tok, p["prompt"], p["rejected"], cfg["max_len"]))
            for p in pairs]

    # The reference model is the SFT model, frozen. Its log-probs never change, so compute them once.
    model.eval()
    ref = []
    with torch.no_grad():
        for b in range(0, len(data), mb):
            chunk = data[b:b + mb]
            c = _seq_logps(torch, model, *_batch(torch, [x for x, _ in chunk], tok.pad_token_id, dev))
            r = _seq_logps(torch, model, *_batch(torch, [y for _, y in chunk], tok.pad_token_id, dev))
            ref += list(zip(c.tolist(), r.tolist()))
    items = [(c, r, rc, rr) for (c, r), (rc, rr) in zip(data, ref)]

    steps_per_epoch = math.ceil(len(items) / cfg["batch_size"])
    total = steps_per_epoch * cfg["epochs"]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"])
    sched = _schedule(opt, total, warmup=max(1, total // 10))
    beta, log, step = cfg["beta"], [], 0
    model.train()
    for epoch in range(cfg["epochs"]):
        random.shuffle(items)
        for b in range(0, len(items), cfg["batch_size"]):
            batch = items[b:b + cfg["batch_size"]]
            cr_all, rr_all, loss_sum = [], [], 0.0
            for m in range(0, len(batch), mb):
                chunk = batch[m:m + mb]
                pc = _seq_logps(torch, model, *_batch(torch, [x[0] for x in chunk], tok.pad_token_id, dev))
                pr = _seq_logps(torch, model, *_batch(torch, [x[1] for x in chunk], tok.pad_token_id, dev))
                rc = torch.tensor([x[2] for x in chunk], device=dev)
                rr = torch.tensor([x[3] for x in chunk], device=dev)
                cr, rj = beta * (pc - rc), beta * (pr - rr)
                loss = -torch.nn.functional.logsigmoid(cr - rj).sum() / len(batch)
                loss.backward()
                loss_sum += loss.item()
                cr_all.append(cr.detach()); rr_all.append(rj.detach())
            chosen_reward, rejected_reward = torch.cat(cr_all), torch.cat(rr_all)
            loss = torch.tensor(loss_sum)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 10 == 0:
                _free(torch, dev)
            if step % log_every == 0:
                log.append({"step": step, "loss": round(loss.item(), 4),
                            "reward_chosen": round(chosen_reward.mean().item(), 4),
                            "reward_rejected": round(rejected_reward.mean().item(), 4),
                            "margin": round((chosen_reward - rejected_reward).mean().item(), 4),
                            "accuracy": round((chosen_reward > rejected_reward).float().mean().item(), 4)})
                print(f"dpo step {step}/{total} loss {loss.item():.3f} margin {log[-1]['margin']:.2f}", flush=True)
    out = pathlib.Path(out_dir)
    model.save_pretrained(out)
    trainable, total_params = _params(model)
    summary = {"method": method, "pairs": len(pairs), "steps": total, "epochs": cfg["epochs"],
               "device": dev, "seconds": round(time.time() - t0, 1), "trainable_params": trainable,
               "total_params": total_params, "models_in_memory": 2, "config": cfg, "log": log,
               "note": "models_in_memory counts the frozen reference; its log-probs were precomputed once"}
    (out / "training_log.json").write_text(json.dumps(summary, indent=1))
    return summary


def load_log(which: str) -> dict:
    """The committed training log for 'lora' or 'rl'."""
    from .models import ROOT
    return json.loads((ROOT / config.ADAPTERS[which] / "training_log.json").read_text())
