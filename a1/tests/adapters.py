"""Adapters: the only file in tests/ you edit.

Each function below is called by the tests. Replace the `raise NotImplementedError`
bodies with calls into your own code in `lm/`. Keep the signatures; the tests pass
arguments by keyword. Conventions the tests assume (see the handout, Section 4):

* Linear weights are stored as W of shape (d_out, d_in) and applied as x @ W.T,
  i.e. einsum(x, W, "... i, o i -> ... o"). No biases anywhere.
* Multi-head attention splits the projection output into heads by contiguous
  chunks: query head h uses rows [h*d_head, (h+1)*d_head) of q_proj_weight.
  With grouped-query attention (num_kv_heads < num_heads), query head h uses
  key/value head h // (num_heads // num_kv_heads).
* RoPE rotates the pairs (x[2i], x[2i+1]) of each head by pos * theta**(-2i/d_head)
  and is applied to queries and keys after the head split, never to values.
* Weight dictionaries for whole blocks/models use the keys documented in
  run_transformer_block and run_transformer_lm; map them to your module names.
"""
from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy as np
import torch
from torch import Tensor
from lm.tokenizer import train_bpe, Tokenizer
from lm.model import (Linear, Embedding, RMSNorm
    , SwiGLU, RotaryPositionalEmbedding, softmax
    , scaled_dot_product_attention, MultiheadSelfAttention
    , TransformerBlock, TransformerLM, count_params, count_flops)

from lm.model_utils import (cross_entropy, AdamW, lr_cosine,
    clip_grad_norm_, get_batch, save_checkpoint, load_checkpoint, generate)


# --------------------------------------------------------------------------- Part 1: tokenizer

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer on the text file at `input_path`.

    Returns (vocab, merges): vocab maps token id -> bytes and contains the 256 single
    bytes first, then the special tokens, then the merged tokens in the order they
    were created; merges lists the merged pairs in creation order.
    """
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    return vocab, merges


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Return an object with `encode(text) -> list[int]`, `decode(ids) -> str` and
    `encode_iterable(iterable_of_str) -> Iterator[int]`."""
    return Tokenizer(vocab, merges, special_tokens)


# --------------------------------------------------------------------------- Part 2: modules

def run_linear(d_in: int, d_out: int, weights: Tensor, in_features: Tensor) -> Tensor:
    """weights: (d_out, d_in). Return in_features @ weights.T with your Linear module."""
    lin = Linear(d_in, d_out)
    with torch.no_grad():
        lin.w.copy_(weights)
    return lin(in_features)


def run_embedding(vocab_size: int, d_model: int, weights: Tensor, token_ids: Tensor) -> Tensor:
    """weights: (vocab_size, d_model). Return the embeddings of `token_ids` (any shape)."""
    embedding = Embedding(vocab_size, d_model)
    with torch.no_grad():
        embedding.w.copy_(weights)
    return embedding(token_ids)


def run_rmsnorm(d_model: int, eps: float, weights: Tensor, in_features: Tensor) -> Tensor:
    """weights: the gain g of shape (d_model,). Statistics in fp32, output in the input dtype."""
    norm = RMSNorm(d_model, eps)
    with torch.no_grad():
        norm.g.copy_(weights)
    return norm(in_features)


def run_swiglu(d_model: int, d_ff: int, w1: Tensor, w2: Tensor, w3: Tensor, in_features: Tensor) -> Tensor:
    """w1, w3: (d_ff, d_model); w2: (d_model, d_ff). Return w2 (silu(w1 x) * (w3 x))."""
    ffn = SwiGLU(d_model, d_ff)
    with torch.no_grad():
        ffn.w1.copy_(w1)
        ffn.w2.copy_(w2)
        ffn.w3.copy_(w3)
    return ffn(in_features)


def run_rope(d_k: int, theta: float, max_seq_len: int, in_query_or_key: Tensor, token_positions: Tensor) -> Tensor:
    """in_query_or_key: (..., seq_len, d_k); token_positions: (..., seq_len) integer positions.
    Return the rotated tensor (same shape)."""
    rope = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len,)
    return rope(in_query_or_key, token_positions)



def run_softmax(in_features: Tensor, dim: int) -> Tensor:
    """Numerically stable softmax along `dim`."""
    return softmax(in_features, dim)


def run_scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None) -> Tensor:
    """Q: (..., q, d_k), K: (..., k, d_k), V: (..., k, d_v); mask: bool (..., q, k), True = may attend.
    Return softmax(QK^T / sqrt(d_k), masked) V of shape (..., q, d_v)."""
    return scaled_dot_product_attention(Q, K, V, mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
    num_kv_heads: int | None = None,
    rope_theta: float | None = None,
    max_seq_len: int | None = None,
    token_positions: Tensor | None = None,
) -> Tensor:
    """Causal multi-head self-attention on in_features of shape (..., seq_len, d_model).

    q_proj_weight: (num_heads * d_head, d_model); k/v_proj_weight: (num_kv_heads * d_head, d_model)
    with d_head = d_model // num_heads; o_proj_weight: (d_model, num_heads * d_head).
    If rope_theta is given, apply RoPE to queries and keys (positions default to arange).
    """
    rope = None
    if rope_theta is not None:
        if max_seq_len is None:
            raise ValueError("max_seq_len is required when using RoPE")

        rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_k=d_model // num_heads,
            max_seq_len=max_seq_len,
        )

    attn = MultiheadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        rope=rope,
    )

    # Load the provided projection weights
    with torch.no_grad():
        attn.w_q.w.copy_(q_proj_weight)
        attn.w_k.w.copy_(k_proj_weight)
        attn.w_v.w.copy_(v_proj_weight)
        attn.w_o.w.copy_(o_proj_weight)

    # Default positions: 0, 1, 2, ..., seq_len - 1
    if token_positions is None:
        token_positions = torch.arange(
            in_features.shape[-2], device=in_features.device
        )

    return attn(in_features, token_positions)



def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Tensor,
) -> Tensor:
    """Pre-norm block with RoPE attention and SwiGLU. `weights` has the keys
    attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight, attn.output_proj.weight,
    ln1.weight, ffn.w1.weight, ffn.w2.weight, ffn.w3.weight, ln2.weight."""
    rope = RotaryPositionalEmbedding(
        theta=theta,
        d_k=d_model // num_heads,
        max_seq_len=max_seq_len,
    )

    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        rope=rope,
    )

    # Load weights
    block.rmsnorm1.g.data.copy_(weights["ln1.weight"])
    block.rmsnorm2.g.data.copy_(weights["ln2.weight"])

    block.attn.w_q.w.data.copy_(weights["attn.q_proj.weight"])
    block.attn.w_k.w.data.copy_(weights["attn.k_proj.weight"])
    block.attn.w_v.w.data.copy_(weights["attn.v_proj.weight"])
    block.attn.w_o.w.data.copy_(weights["attn.output_proj.weight"])

    block.swiglu.w1.data.copy_(weights["ffn.w1.weight"])
    block.swiglu.w2.data.copy_(weights["ffn.w2.weight"])
    block.swiglu.w3.data.copy_(weights["ffn.w3.weight"])

    # Position indices for the sequence
    token_positions = torch.arange(
        in_features.shape[1], device=in_features.device
    )

    return block(in_features, token_positions)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Tensor,
) -> Tensor:
    """Full model. `weights` has token_embeddings.weight, layers.{i}.<block keys> for
    i in range(num_layers), ln_final.weight and lm_head.weight (untied).
    in_indices: (batch, seq_len) ids. Return logits (batch, seq_len, vocab_size) in fp32."""
    model = model = TransformerLM(vocab_size=vocab_size, context_length=context_length, d_model=d_model, num_layers=num_layers, num_heads=num_heads, d_ff=d_ff)
    model.embeddings.w.data.copy_(weights["token_embeddings.weight"])

    # Load each Transformer block's weights
    for i, layer in enumerate(model.layers):
        prefix = f"layers.{i}."

        layer.rmsnorm1.g.data.copy_(weights[prefix + "ln1.weight"])
        layer.rmsnorm2.g.data.copy_(weights[prefix + "ln2.weight"])

        layer.attn.w_q.w.data.copy_(weights[prefix + "attn.q_proj.weight"])
        layer.attn.w_k.w.data.copy_(weights[prefix + "attn.k_proj.weight"])
        layer.attn.w_v.w.data.copy_(weights[prefix + "attn.v_proj.weight"])
        layer.attn.w_o.w.data.copy_(weights[prefix + "attn.output_proj.weight"])

        layer.swiglu.w1.data.copy_(weights[prefix + "ffn.w1.weight"])
        layer.swiglu.w2.data.copy_(weights[prefix + "ffn.w2.weight"])
        layer.swiglu.w3.data.copy_(weights[prefix + "ffn.w3.weight"])

    # Load final normalization and output head
    model.rmsnorm_final.g.data.copy_(weights["ln_final.weight"])
    model.output_head.w.data.copy_(weights["lm_head.weight"])

    # Create RoPE and assign it to each attention layer
    rope = RotaryPositionalEmbedding(
        theta=rope_theta,
        d_k=d_model // num_heads,
        max_seq_len=context_length,
    )

    for layer in model.layers:
        layer.attn.rope = rope

    return model(in_indices)


def run_count_params(
    vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int,
    num_kv_heads: int | None = None, tied: bool = False,
) -> int:
    """Total number of parameters of the model above (norm gains included)."""
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        num_kv_heads=num_kv_heads,
        tied=tied,
    )

    return count_params(model)


def run_count_flops(
    vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int,
    seq_len: int, num_kv_heads: int | None = None,
) -> tuple[int, int]:
    """(forward FLOPs per token, training FLOPs per token) at sequence length seq_len:
    2 x every matmul parameter (blocks and the LM head) plus the attention scores
    4 * num_layers * seq_len * d_model; training = 3 x forward."""
    if num_kv_heads is None:
        num_kv_heads = num_heads

    config = {
        "vocab_size": vocab_size,
        "context_length": context_length,
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "d_ff": d_ff,
    }

    return count_flops(config, seq_len)



# --------------------------------------------------------------------------- Part 2: training utilities

def run_cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """inputs: (..., vocab_size) logits; targets: (...) ids. Return the mean negative
    log-likelihood as a 0-d tensor, computed stably (subtract the max)."""
    return cross_entropy(inputs, targets)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Rescale all .grad in place so that their global L2 norm is at most max_l2_norm
    (with a 1e-6 added to the norm in the denominator, as torch does)."""
    clip_grad_norm_(parameters, max_l2_norm)


def get_adamw_cls() -> type[torch.optim.Optimizer]:
    """Return your AdamW class (a torch.optim.Optimizer subclass) with the constructor
    AdamW(params, lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)."""
    return AdamW


def run_get_lr_cosine_schedule(
    it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int
) -> float:
    """Linear warmup from 0 to max over warmup_iters, cosine decay to min at cosine_cycle_iters,
    constant min afterwards."""
    return lr_cosine(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)


def run_get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[Tensor, Tensor]:
    """Sample batch_size random windows from a 1-D array of token ids (e.g. a uint16 memmap).
    Return (inputs, targets), both int64 of shape (batch_size, context_length) on `device`,
    with targets shifted by one position."""
    return get_batch(dataset, batch_size, context_length, device)


def run_save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int,
                        out: str | os.PathLike | BinaryIO | IO[bytes]) -> None:
    save_checkpoint(model, optimizer, iteration, out)


def run_load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer) -> int:
    """Restore model and optimizer state; return the saved iteration."""
    return load_checkpoint(src, model, optimizer)


# --------------------------------------------------------------------------- Part 3: generation

def run_generate(
    vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int,
    rope_theta: float, weights: dict[str, Tensor], prompt_ids: list[int], max_new_tokens: int,
    temperature: float, top_p: float, eos_id: int | None, seed: int,
) -> list[int]:
    """Build the model from `weights` (same keys as run_transformer_lm), then sample up to
    max_new_tokens tokens after prompt_ids with temperature and nucleus (top-p) sampling,
    stopping after eos_id is produced. Use torch.Generator(...).manual_seed(seed) for the
    randomness. Return only the newly generated ids."""
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
    )

    # Load token embeddings
    model.embeddings.w.data.copy_(weights["token_embeddings.weight"])

    # Load transformer layers
    for i, layer in enumerate(model.layers):
        prefix = f"layers.{i}."

        layer.rmsnorm1.g.data.copy_(weights[prefix + "ln1.weight"])
        layer.rmsnorm2.g.data.copy_(weights[prefix + "ln2.weight"])

        layer.attn.w_q.w.data.copy_(weights[prefix + "attn.q_proj.weight"])
        layer.attn.w_k.w.data.copy_(weights[prefix + "attn.k_proj.weight"])
        layer.attn.w_v.w.data.copy_(weights[prefix + "attn.v_proj.weight"])
        layer.attn.w_o.w.data.copy_(weights[prefix + "attn.output_proj.weight"])

        layer.swiglu.w1.data.copy_(weights[prefix + "ffn.w1.weight"])
        layer.swiglu.w2.data.copy_(weights[prefix + "ffn.w2.weight"])
        layer.swiglu.w3.data.copy_(weights[prefix + "ffn.w3.weight"])

    # Load final normalization and output head
    model.rmsnorm_final.g.data.copy_(weights["ln_final.weight"])
    model.output_head.w.data.copy_(weights["lm_head.weight"])

    # Set rotary positional embeddings
    rope = RotaryPositionalEmbedding(
        theta=rope_theta,
        d_k=d_model // num_heads,
        max_seq_len=context_length,
    )
    for layer in model.layers:
        layer.attn.rope = rope

    # Use the existing generation function
    return generate(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_id=eos_id,
        seed=seed,
    )
