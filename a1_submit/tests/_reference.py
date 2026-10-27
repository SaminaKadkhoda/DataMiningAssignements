"""Reference implementations used by the tests. Deliberately slow and simple.

Do not edit. Your own code is compared against these on small inputs.
"""
from __future__ import annotations

import math
from collections import Counter

import regex
import torch
import torch.nn.functional as F
from torch import Tensor

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_PAT_RE = regex.compile(PAT)


# --------------------------------------------------------------------------- BPE

def split_on_specials(text: str, special_tokens: list[str]) -> list[str]:
    """Split text into chunks that never contain a special token."""
    if not special_tokens:
        return [text]
    specials = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join(regex.escape(s) for s in specials)
    return regex.split(pattern, text)


def pretoken_counts(text: str, special_tokens: list[str]) -> Counter[bytes]:
    counts: Counter[bytes] = Counter()
    for chunk in split_on_specials(text, special_tokens):
        for m in _PAT_RE.finditer(chunk):
            counts[m.group(0).encode("utf-8")] += 1
    return counts


def train_bpe_reference(text: str, vocab_size: int, special_tokens: list[str]):
    """Naive BPE training: recount every pair after every merge. Ties: lexicographically
    greater pair wins (comparison of (bytes, bytes) tuples)."""
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for s in special_tokens:
        vocab[len(vocab)] = s.encode("utf-8")
    merges: list[tuple[bytes, bytes]] = []
    seqs: dict[tuple[bytes, ...], int] = {
        tuple(bytes([b]) for b in pre): c for pre, c in pretoken_counts(text, special_tokens).items()
    }
    while len(vocab) < vocab_size:
        pairs: Counter[tuple[bytes, bytes]] = Counter()
        for seq, c in seqs.items():
            for a, b in zip(seq, seq[1:]):
                pairs[(a, b)] += c
        if not pairs:
            break
        best = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))[0]
        merges.append(best)
        vocab[len(vocab)] = best[0] + best[1]
        new_seqs: dict[tuple[bytes, ...], int] = {}
        for seq, c in seqs.items():
            out, i = [], 0
            while i < len(seq):
                if i + 1 < len(seq) and (seq[i], seq[i + 1]) == best:
                    out.append(seq[i] + seq[i + 1])
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            new_seqs[tuple(out)] = new_seqs.get(tuple(out), 0) + c
        seqs = new_seqs
    return vocab, merges


def encode_reference(text: str, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]],
                     special_tokens: list[str] | None = None) -> list[int]:
    """Apply merges by rank inside each pre-token; special tokens matched longest-first."""
    special_tokens = special_tokens or []
    rank = {pair: i for i, pair in enumerate(merges)}
    inv = {v: k for k, v in vocab.items()}
    ids: list[int] = []
    if special_tokens:
        specials = sorted(special_tokens, key=len, reverse=True)
        pattern = "(" + "|".join(regex.escape(s) for s in specials) + ")"
        parts = regex.split(pattern, text)
    else:
        parts = [text]
    for part in parts:
        if part in special_tokens:
            ids.append(inv[part.encode("utf-8")])
            continue
        for m in _PAT_RE.finditer(part):
            seq = [bytes([b]) for b in m.group(0).encode("utf-8")]
            while len(seq) > 1:
                best, best_rank = None, None
                for i in range(len(seq) - 1):
                    r = rank.get((seq[i], seq[i + 1]))
                    if r is not None and (best_rank is None or r < best_rank):
                        best, best_rank = i, r
                if best is None:
                    break
                seq = seq[:best] + [seq[best] + seq[best + 1]] + seq[best + 2:]
            ids.extend(inv[tok] for tok in seq)
    return ids


# --------------------------------------------------------------------------- model math

def rmsnorm_ref(x: Tensor, g: Tensor, eps: float) -> Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * g.to(x.dtype)


def swiglu_ref(x: Tensor, w1: Tensor, w2: Tensor, w3: Tensor) -> Tensor:
    return (F.silu(x @ w1.T) * (x @ w3.T)) @ w2.T


def rope_ref(x: Tensor, positions: Tensor, theta: float) -> Tensor:
    """x: (..., seq, d); positions: (..., seq). Pairs (2i, 2i+1)."""
    d = x.shape[-1]
    inv = theta ** (-torch.arange(0, d, 2, dtype=torch.float32) / d)          # (d/2,)
    ang = positions.float()[..., None] * inv                                   # (..., seq, d/2)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    y = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    return y.to(x.dtype)


def sdpa_ref(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None) -> Tensor:
    s = q.float() @ k.float().transpose(-1, -2) / math.sqrt(q.shape[-1])
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf"))
    return (torch.softmax(s, dim=-1) @ v.float()).to(q.dtype)


def mha_ref(x: Tensor, wq: Tensor, wk: Tensor, wv: Tensor, wo: Tensor, num_heads: int,
            num_kv_heads: int | None = None, rope_theta: float | None = None,
            positions: Tensor | None = None) -> Tensor:
    *lead, seq, d_model = x.shape
    hd = d_model // num_heads
    nkv = num_kv_heads or num_heads
    q = (x @ wq.T).reshape(*lead, seq, num_heads, hd).transpose(-3, -2)       # (..., h, seq, hd)
    k = (x @ wk.T).reshape(*lead, seq, nkv, hd).transpose(-3, -2)
    v = (x @ wv.T).reshape(*lead, seq, nkv, hd).transpose(-3, -2)
    if rope_theta is not None:
        if positions is None:
            positions = torch.arange(seq, device=x.device).expand(*lead, seq)
        pos = positions[..., None, :]                                           # broadcast over heads
        q = rope_ref(q, pos.expand(*lead, num_heads, seq), rope_theta)
        k = rope_ref(k, pos.expand(*lead, nkv, seq), rope_theta)
    if nkv != num_heads:
        rep = num_heads // nkv
        k = k.repeat_interleave(rep, dim=-3)
        v = v.repeat_interleave(rep, dim=-3)
    mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=x.device))
    o = sdpa_ref(q, k, v, mask)                                                 # (..., h, seq, hd)
    o = o.transpose(-3, -2).reshape(*lead, seq, num_heads * hd)
    return o @ wo.T


def block_ref(x: Tensor, w: dict[str, Tensor], num_heads: int, theta: float) -> Tensor:
    h = x + mha_ref(rmsnorm_ref(x, w["ln1.weight"], 1e-5), w["attn.q_proj.weight"], w["attn.k_proj.weight"],
                    w["attn.v_proj.weight"], w["attn.output_proj.weight"], num_heads, rope_theta=theta)
    return h + swiglu_ref(rmsnorm_ref(h, w["ln2.weight"], 1e-5), w["ffn.w1.weight"], w["ffn.w2.weight"], w["ffn.w3.weight"])


def lm_ref(ids: Tensor, w: dict[str, Tensor], num_layers: int, num_heads: int, theta: float) -> Tensor:
    x = w["token_embeddings.weight"][ids]
    for i in range(num_layers):
        wi = {k[len(f"layers.{i}."):]: v for k, v in w.items() if k.startswith(f"layers.{i}.")}
        x = block_ref(x, wi, num_heads, theta)
    x = rmsnorm_ref(x, w["ln_final.weight"], 1e-5)
    return x @ w["lm_head.weight"].T


def count_params_ref(vocab_size, d_model, num_layers, num_heads, d_ff, num_kv_heads=None, tied=False) -> int:
    hd = d_model // num_heads
    nkv = num_kv_heads or num_heads
    attn = d_model * d_model + 2 * (nkv * hd) * d_model + d_model * d_model
    block = attn + 3 * d_model * d_ff + 2 * d_model
    total = num_layers * block + d_model + vocab_size * d_model
    if not tied:
        total += vocab_size * d_model
    return total


def count_flops_ref(vocab_size, d_model, num_layers, num_heads, d_ff, seq_len, num_kv_heads=None) -> tuple[int, int]:
    hd = d_model // num_heads
    nkv = num_kv_heads or num_heads
    attn = d_model * d_model + 2 * (nkv * hd) * d_model + d_model * d_model
    matmul_params = num_layers * (attn + 3 * d_model * d_ff) + vocab_size * d_model
    fwd = 2 * matmul_params + 4 * num_layers * seq_len * d_model
    return fwd, 3 * fwd


def cosine_lr_ref(it, max_lr, min_lr, warmup, cycle) -> float:
    if it < warmup:
        return max_lr * it / warmup
    if it <= cycle:
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * (it - warmup) / (cycle - warmup)))
    return min_lr


def make_lm_weights(vocab_size, d_model, num_layers, num_heads, d_ff, seed=0, dtype=torch.float32) -> dict[str, Tensor]:
    g = torch.Generator().manual_seed(seed)

    def lin(o, i):
        std = math.sqrt(2 / (i + o))
        return torch.nn.init.trunc_normal_(torch.empty(o, i, dtype=dtype), std=std, a=-3 * std, b=3 * std, generator=g)

    w = {"token_embeddings.weight": torch.nn.init.trunc_normal_(torch.empty(vocab_size, d_model, dtype=dtype), std=1.0, a=-3, b=3, generator=g)}
    for l in range(num_layers):
        p = f"layers.{l}."
        w[p + "attn.q_proj.weight"] = lin(d_model, d_model)
        w[p + "attn.k_proj.weight"] = lin(d_model, d_model)
        w[p + "attn.v_proj.weight"] = lin(d_model, d_model)
        w[p + "attn.output_proj.weight"] = lin(d_model, d_model)
        w[p + "ln1.weight"] = torch.ones(d_model, dtype=dtype)
        w[p + "ffn.w1.weight"] = lin(d_ff, d_model)
        w[p + "ffn.w2.weight"] = lin(d_model, d_ff)
        w[p + "ffn.w3.weight"] = lin(d_ff, d_model)
        w[p + "ln2.weight"] = torch.ones(d_model, dtype=dtype)
    w["ln_final.weight"] = torch.ones(d_model, dtype=dtype)
    w["lm_head.weight"] = lin(vocab_size, d_model)
    return w
