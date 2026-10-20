#!/usr/bin/env python3
"""Training-loop skeleton for Assignment 1. The basics (config, logging, results.json,
checkpoint/resume) are here; the model, loss, optimizer and schedule are yours to fill.

    python scripts/train.py --data data/train.bin --val data/val.bin --out runs/ref \
        --lr 1e-3 --steps 24414 --batch 32 --ctx 256

Fill in the TODO lines. Everything else can stay as is.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

# TODO: import your code, e.g.
from lm.model import (TransformerLM, count_params, count_flops, RotaryPositionalEmbedding)
from lm.model_utils import (AdamW, cross_entropy, clip_grad_norm_, lr_cosine, get_batch,
    save_checkpoint, load_checkpoint)

PEAK_FLOPS = {"T4": 65e12, "L4": 121e12, "A100": 312e12, "H100": 989e12}   # bf16/fp16 dense peak


@dataclass
class Config:
    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: float = 10000.0
    batch_size: int = 32
    steps: int = 24414
    lr_max: float = 3e-3 #set to best rate
    lr_min_frac: float = 0.1
    warmup_frac: float = 0.05
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_every: int = 500
    eval_tokens: int = 2 ** 20
    ckpt_every: int = 1000
    seed: int = 0


def gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def peak_flops_for(name: str) -> float | None:
    return next((v for k, v in PEAK_FLOPS.items() if k in name), None)


@torch.no_grad()
def evaluate(model, val, cfg: Config, device) -> float:
    """Mean loss over the first cfg.eval_tokens tokens of val, in fp32."""
    model.eval()
    n = min(cfg.eval_tokens, len(val) - 1)
    ids = torch.from_numpy(val[: n + 1].astype(np.int64)).to(device)
    losses = []
    for s in range(0, n - cfg.context_length, cfg.context_length * cfg.batch_size):
        chunk = ids[s: s + cfg.context_length * cfg.batch_size + 1]
        L = (len(chunk) - 1) // cfg.context_length * cfg.context_length
        if L == 0:
            break
        x = chunk[:L].view(-1, cfg.context_length)
        y = chunk[1:L + 1].view(-1, cfg.context_length)
        logits = model(x)                                  # TODO: autocast here as in training
        losses.append(cross_entropy(logits.float(), y).item())  # noqa: F821  (your cross_entropy)
    model.train()
    return float(np.mean(losses))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    for f in Config.__dataclass_fields__:
        ap.add_argument(f"--{f}", type=type(getattr(Config, f)), default=getattr(Config, f))
    args = ap.parse_args()
    cfg = Config(**{f: getattr(args, f) for f in Config.__dataclass_fields__})
    os.makedirs(args.out, exist_ok=True)
    json.dump(asdict(cfg), open(os.path.join(args.out, "config.json"), "w"), indent=2)

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    if device == "cuda": #for t4 run
        major, _ = torch.cuda.get_device_capability()
        amp_dtype = torch.bfloat16 if major >= 8 else torch.float16 #for eager fp32: float32 otherwise bfloat16 and float16
    else:
        amp_dtype = torch.float32
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and amp_dtype == torch.float16))
    #scaler = torch.cuda.amp.GradScaler(enabled=False) #for eager fp32

    train = np.memmap(args.data, dtype=np.uint16, mode="r")
    val = np.memmap(args.val, dtype=np.uint16, mode="r")

    # TODO: build the model and the optimizer with your classes
    rope_emb = RotaryPositionalEmbedding(cfg.rope_theta, cfg.d_model//cfg.num_heads, cfg.context_length)
    model = TransformerLM(cfg.vocab_size, cfg.context_length, cfg.d_model, cfg.num_layers, cfg.num_heads, cfg.d_ff, rope=rope_emb).to(device)
    #model = TransformerLM(cfg.vocab_size, cfg.context_length, cfg.d_model, cfg.num_layers, cfg.num_heads, cfg.d_ff, cfg.rope_theta).to(device)
    opt = AdamW(model.parameters(), lr=cfg.lr_max, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)
    #opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_max, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay, fused=True) #for fasterr run
    #raise NotImplementedError("build model and optimizer, then delete this line")

    #n_params = count_params(cfg.vocab_size, cfg.context_length, cfg.d_model, cfg.num_layers, cfg.num_heads, cfg.d_ff)
    n_params = count_params(model)  # noqa: F821
    #fwd_flops, train_flops = count_flops(cfg.vocab_size, cfg.context_length, cfg.d_model, cfg.num_layers, cfg.num_heads, cfg.d_ff, cfg.context_length)
    config_dict = {"vocab_size": cfg.vocab_size, "d_model": cfg.d_model, "num_layers": cfg.num_layers, "d_ff": cfg.d_ff, "num_heads": cfg.num_heads}
    fwd_flops, train_flops = count_flops(config_dict, cfg.context_length)  # noqa: F821
    assert n_params == sum(p.numel() for p in model.parameters()), "count_params disagrees with the model"

    start = 0
    ckpt = os.path.join(args.out, "ckpt.pt")
    if args.resume and os.path.exists(ckpt):
        start = load_checkpoint(ckpt, model, opt)  # noqa: F821
        print(f"resumed at step {start}")

    log = open(os.path.join(args.out, "log.jsonl"), "a")
    tokens_per_step = cfg.batch_size * cfg.context_length
    warmup = int(cfg.warmup_frac * cfg.steps)
    t0, tok_count = time.time(), 0
    eager_t0 = None
    for step in range(start, cfg.steps):
        if step == 50: #for eager fp32: after 50 warmups
            torch.cuda.synchronize()
            eager_t0 = time.time()
        lr = lr_cosine(step, cfg.lr_max, cfg.lr_max * cfg.lr_min_frac, warmup, cfg.steps)  # noqa: F821
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(train, cfg.batch_size, cfg.context_length, device)  # noqa: F821
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device == "cuda")):
        #with torch.autocast(device_type=device, dtype=amp_dtype, enabled=False): #for eager fp32
            logits = model(x)
        loss = cross_entropy(logits.float(), y)  # noqa: F821
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gnorm = clip_grad_norm_(model.parameters(), cfg.grad_clip)  # noqa: F821
        scaler.step(opt); scaler.update()
        opt.zero_grad(set_to_none=True)
        tok_count += tokens_per_step

        # if step % 50 == 0:
        #     if device == "cuda":
        #         torch.cuda.synchronize()
        #     el = time.time() - t0
        #     rec = {"step": step, "loss": loss.item(), "lr": lr, "gnorm": float(gnorm), "tokens_per_s": tok_count / el if el else 0}
        #         print(json.dumps(rec)); log.write(json.dumps(rec) + "\n"); log.flush()
        # if step % cfg.eval_every == 0 or step == cfg.steps - 1:
        #     rec = {"step": step, "val_loss": evaluate(model, val, cfg, device)}
        #     print(json.dumps(rec)); log.write(json.dumps(rec) + "\n"); log.flush()
        # if step % cfg.ckpt_every == 0 and step > start:
        #     #added 1 to the step, so it starts from the next step, not the already processed one
        #     save_checkpoint(model, opt, step+1, ckpt)  # noqa: F821 

        #--------------------------------------for eager fp32
        if step+1 == 250: #analyze 200 steps
            torch.cuda.synchronize()
            eager_time = time.time() - eager_t0
            eager_tokens_per_s = (200 * cfg.batch_size * cfg.context_length / eager_time)
            print(f"autocast 200-step tokens_per_s: {eager_tokens_per_s}")



    save_checkpoint(model, opt, cfg.steps, ckpt)  # noqa: F821
    if device == "cuda":
        torch.cuda.synchronize()
    wall = time.time() - t0
    tps = tok_count / wall if wall else 0
    peak = peak_flops_for(gpu_name())
    summary = {"lr": cfg.lr_max, "steps": cfg.steps, "tokens": cfg.steps * tokens_per_step,
               "flops": cfg.steps * tokens_per_step * train_flops, "val_loss": evaluate(model, val, cfg, device),
               "tokens_per_s": tps, "mfu": (tps * train_flops / peak) if peak else None, "wallclock_s": wall,
               "gpu": gpu_name(), "precision": str(amp_dtype).split(".")[-1]}
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
