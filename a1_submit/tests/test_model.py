"""Part 2: modules and the full model."""
from __future__ import annotations

import math

import pytest
import torch

from . import adapters
from ._reference import (block_ref, count_flops_ref, count_params_ref, lm_ref, make_lm_weights, mha_ref,
                         rmsnorm_ref, rope_ref, sdpa_ref, swiglu_ref)

torch.manual_seed(0)
ATOL, RTOL = 1e-5, 1e-4


def g(seed=0):
    return torch.Generator().manual_seed(seed)


def randn(*shape, seed=0, dtype=torch.float32):
    return torch.randn(*shape, generator=g(seed), dtype=dtype)


# --------------------------------------------------------------------------- linear / embedding

@pytest.mark.parametrize("lead", [(), (3,), (2, 5)])
def test_linear(lead):
    d_in, d_out = 24, 40
    W = randn(d_out, d_in, seed=1) * 0.2
    x = randn(*lead, 7, d_in, seed=2)
    out = adapters.run_linear(d_in=d_in, d_out=d_out, weights=W, in_features=x)
    assert out.shape == (*lead, 7, d_out)
    torch.testing.assert_close(out, x @ W.T, atol=ATOL, rtol=RTOL)


def test_embedding():
    V, d = 50, 16
    W = randn(V, d, seed=3)
    ids = torch.randint(0, V, (4, 9), generator=g(4))
    out = adapters.run_embedding(vocab_size=V, d_model=d, weights=W, token_ids=ids)
    assert out.shape == (4, 9, d)
    torch.testing.assert_close(out, W[ids], atol=ATOL, rtol=RTOL)


# --------------------------------------------------------------------------- rmsnorm

@pytest.mark.parametrize("eps", [1e-5, 1e-3])
def test_rmsnorm(eps):
    d = 32
    gain = randn(d, seed=5) * 0.5 + 1.0
    x = randn(2, 6, d, seed=6) * 3
    out = adapters.run_rmsnorm(d_model=d, eps=eps, weights=gain, in_features=x)
    torch.testing.assert_close(out, rmsnorm_ref(x, gain, eps), atol=ATOL, rtol=RTOL)


def test_rmsnorm_half_precision_statistics():
    """Input in fp16: the output dtype is fp16, but the statistics must not be computed in fp16."""
    d = 256
    gain = torch.ones(d)
    x = (randn(3, d, seed=7) * 200).half()      # sum of squares overflows fp16 (max 65504) if computed in fp16
    out = adapters.run_rmsnorm(d_model=d, eps=1e-5, weights=gain.half(), in_features=x)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all(), "NaN/inf: compute the mean of squares in fp32"
    torch.testing.assert_close(out.float(), rmsnorm_ref(x, gain.half(), 1e-5).float(), atol=1e-2, rtol=1e-2)


# --------------------------------------------------------------------------- swiglu

def test_swiglu():
    d, dff = 32, 88
    w1, w2, w3 = randn(dff, d, seed=8) * 0.2, randn(d, dff, seed=9) * 0.2, randn(dff, d, seed=10) * 0.2
    x = randn(2, 5, d, seed=11)
    out = adapters.run_swiglu(d_model=d, d_ff=dff, w1=w1, w2=w2, w3=w3, in_features=x)
    torch.testing.assert_close(out, swiglu_ref(x, w1, w2, w3), atol=ATOL, rtol=RTOL)


# --------------------------------------------------------------------------- rope

@pytest.mark.parametrize("theta", [10000.0, 500000.0])
def test_rope(theta):
    d_k, seq = 16, 12
    x = randn(2, 3, seq, d_k, seed=12)                       # (batch, heads, seq, d_k)
    pos = torch.arange(seq).expand(2, 3, seq)
    out = adapters.run_rope(d_k=d_k, theta=theta, max_seq_len=64, in_query_or_key=x, token_positions=pos)
    assert out.shape == x.shape
    torch.testing.assert_close(out, rope_ref(x, pos, theta), atol=ATOL, rtol=RTOL)


def test_rope_arbitrary_positions():
    """Positions need not start at 0 or be contiguous (needed for cached generation)."""
    d_k, seq = 8, 5
    x = randn(seq, d_k, seed=13)
    pos = torch.tensor([7, 3, 40, 0, 11])
    out = adapters.run_rope(d_k=d_k, theta=10000.0, max_seq_len=64, in_query_or_key=x, token_positions=pos)
    torch.testing.assert_close(out, rope_ref(x, pos, 10000.0), atol=ATOL, rtol=RTOL)


def test_rope_relative_property():
    """q_m . k_n depends only on m - n: shifting both positions by 5 leaves the dot product unchanged."""
    d_k = 8
    q, k = randn(d_k, seed=14), randn(d_k, seed=15)
    def rot(v, m):
        return adapters.run_rope(d_k=d_k, theta=10000.0, max_seq_len=128, in_query_or_key=v[None], token_positions=torch.tensor([m]))[0]
    a = rot(q, 10) @ rot(k, 4)
    b = rot(q, 15) @ rot(k, 9)
    c = rot(q, 10) @ rot(k, 5)
    assert abs(a - b) < 1e-4 and abs(a - c) > 1e-3


# --------------------------------------------------------------------------- softmax / attention

def test_softmax():
    x = randn(3, 4, 7, seed=16) * 5
    for dim in (-1, 0, 1):
        out = adapters.run_softmax(in_features=x, dim=dim)
        torch.testing.assert_close(out, torch.softmax(x, dim=dim), atol=ATOL, rtol=RTOL)


def test_softmax_stability():
    x = randn(2, 9, seed=17) + 5000.0           # naive exp overflows
    out = adapters.run_softmax(in_features=x, dim=-1)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.softmax(x, dim=-1), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("lead", [(), (2,), (2, 4)])
def test_scaled_dot_product_attention(lead):
    q_len, k_len, d_k, d_v = 6, 9, 16, 12
    q, k, v = randn(*lead, q_len, d_k, seed=18), randn(*lead, k_len, d_k, seed=19), randn(*lead, k_len, d_v, seed=20)
    out = adapters.run_scaled_dot_product_attention(Q=q, K=k, V=v, mask=None)
    assert out.shape == (*lead, q_len, d_v)
    torch.testing.assert_close(out, sdpa_ref(q, k, v, None), atol=ATOL, rtol=RTOL)


def test_scaled_dot_product_attention_mask():
    q_len, k_len, d = 5, 5, 8
    q, k, v = randn(2, q_len, d, seed=21), randn(2, k_len, d, seed=22), randn(2, k_len, d, seed=23)
    causal = torch.tril(torch.ones(q_len, k_len, dtype=torch.bool))
    out = adapters.run_scaled_dot_product_attention(Q=q, K=k, V=v, mask=causal)
    torch.testing.assert_close(out, sdpa_ref(q, k, v, causal), atol=ATOL, rtol=RTOL)
    # the first query may only see the first key: its output is exactly v[..., 0, :]
    torch.testing.assert_close(out[:, 0], v[:, 0], atol=ATOL, rtol=RTOL)
    # an arbitrary (non-causal) mask with a broadcast batch dimension
    mask = torch.rand(q_len, k_len, generator=g(24)) > 0.4
    mask[:, 0] = True                            # keep every row attendable
    out = adapters.run_scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)
    torch.testing.assert_close(out, sdpa_ref(q, k, v, mask), atol=ATOL, rtol=RTOL)


# --------------------------------------------------------------------------- multi-head attention

@pytest.mark.parametrize("lead", [(), (2,)])
def test_multihead_self_attention(lead):
    d, h, seq = 32, 4, 10
    wq, wk, wv, wo = (randn(d, d, seed=s) * 0.2 for s in (25, 26, 27, 28))
    x = randn(*lead, seq, d, seed=29)
    out = adapters.run_multihead_self_attention(d_model=d, num_heads=h, q_proj_weight=wq, k_proj_weight=wk,
                                                v_proj_weight=wv, o_proj_weight=wo, in_features=x)
    assert out.shape == x.shape
    torch.testing.assert_close(out, mha_ref(x, wq, wk, wv, wo, h), atol=ATOL, rtol=RTOL)


def test_multihead_self_attention_is_causal():
    """Changing a later token must not change earlier outputs."""
    d, h, seq = 16, 2, 8
    wq, wk, wv, wo = (randn(d, d, seed=s) * 0.2 for s in (30, 31, 32, 33))
    x = randn(seq, d, seed=34)
    x2 = x.clone(); x2[-1] += 10.0
    kw = dict(d_model=d, num_heads=h, q_proj_weight=wq, k_proj_weight=wk, v_proj_weight=wv, o_proj_weight=wo)
    a = adapters.run_multihead_self_attention(in_features=x, **kw)
    b = adapters.run_multihead_self_attention(in_features=x2, **kw)
    torch.testing.assert_close(a[:-1], b[:-1], atol=ATOL, rtol=RTOL)
    assert not torch.allclose(a[-1], b[-1])


def test_multihead_self_attention_with_rope():
    d, h, seq = 32, 4, 9
    wq, wk, wv, wo = (randn(d, d, seed=s) * 0.2 for s in (35, 36, 37, 38))
    x = randn(2, seq, d, seed=39)
    pos = torch.arange(seq).expand(2, seq)
    out = adapters.run_multihead_self_attention(d_model=d, num_heads=h, q_proj_weight=wq, k_proj_weight=wk,
                                                v_proj_weight=wv, o_proj_weight=wo, in_features=x,
                                                rope_theta=10000.0, max_seq_len=32, token_positions=pos)
    torch.testing.assert_close(out, mha_ref(x, wq, wk, wv, wo, h, rope_theta=10000.0, positions=pos), atol=ATOL, rtol=RTOL)


def test_grouped_query_attention():
    d, h, hkv, seq = 32, 8, 2, 7
    hd = d // h
    wq, wo = randn(d, d, seed=40) * 0.2, randn(d, d, seed=41) * 0.2
    wk, wv = randn(hkv * hd, d, seed=42) * 0.2, randn(hkv * hd, d, seed=43) * 0.2
    x = randn(seq, d, seed=44)
    out = adapters.run_multihead_self_attention(d_model=d, num_heads=h, num_kv_heads=hkv, q_proj_weight=wq,
                                                k_proj_weight=wk, v_proj_weight=wv, o_proj_weight=wo,
                                                in_features=x, rope_theta=10000.0, max_seq_len=32)
    torch.testing.assert_close(out, mha_ref(x, wq, wk, wv, wo, h, num_kv_heads=hkv, rope_theta=10000.0), atol=ATOL, rtol=RTOL)


# --------------------------------------------------------------------------- block / model

def test_transformer_block():
    d, h, dff, seq = 32, 4, 88, 11
    w = make_lm_weights(vocab_size=10, d_model=d, num_layers=1, num_heads=h, d_ff=dff, seed=45)
    w = {k[len("layers.0."):]: v for k, v in w.items() if k.startswith("layers.0.")}
    w["ln1.weight"] = randn(d, seed=46) * 0.1 + 1.0
    x = randn(2, seq, d, seed=47)
    out = adapters.run_transformer_block(d_model=d, num_heads=h, d_ff=dff, max_seq_len=64, theta=10000.0, weights=w, in_features=x)
    torch.testing.assert_close(out, block_ref(x, w, h, 10000.0), atol=1e-4, rtol=1e-4)


def test_transformer_lm():
    V, L, d, n, h, dff = 37, 16, 32, 3, 4, 88
    w = make_lm_weights(V, d, n, h, dff, seed=48)
    ids = torch.randint(0, V, (2, 13), generator=g(49))
    out = adapters.run_transformer_lm(vocab_size=V, context_length=L, d_model=d, num_layers=n, num_heads=h,
                                      d_ff=dff, rope_theta=10000.0, weights=w, in_indices=ids)
    assert out.shape == (2, 13, V) and out.dtype == torch.float32
    torch.testing.assert_close(out, lm_ref(ids, w, n, h, 10000.0), atol=1e-4, rtol=1e-4)


def test_transformer_lm_initial_loss():
    """A freshly initialised model must predict roughly uniformly: loss close to ln(V)."""
    V, L, d, n, h, dff = 200, 32, 64, 2, 4, 176
    w = make_lm_weights(V, d, n, h, dff, seed=50)
    ids = torch.randint(0, V, (8, 32), generator=g(51))
    logits = adapters.run_transformer_lm(vocab_size=V, context_length=L, d_model=d, num_layers=n, num_heads=h,
                                         d_ff=dff, rope_theta=10000.0, weights=w, in_indices=ids)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, V), ids.reshape(-1))
    assert abs(loss.item() - math.log(V)) < 1.0


# --------------------------------------------------------------------------- accounting

REFERENCE = dict(vocab_size=10000, context_length=256, d_model=512, num_layers=4, num_heads=16, d_ff=1344)


def test_count_params_reference_config():
    assert adapters.run_count_params(**REFERENCE) == 22_696_448
    assert adapters.run_count_params(**REFERENCE, tied=True) == 22_696_448 - 5_120_000


@pytest.mark.parametrize("cfg", [
    dict(vocab_size=50257, context_length=1024, d_model=768, num_layers=12, num_heads=12, d_ff=3072),
    dict(vocab_size=128256, context_length=8192, d_model=4096, num_layers=32, num_heads=32, d_ff=14336, num_kv_heads=8),
])
def test_count_params_matches_reference(cfg):
    kw = {k: v for k, v in cfg.items() if k != "context_length"}
    assert adapters.run_count_params(**cfg) == count_params_ref(**kw)


def test_count_flops_reference_config():
    fwd, train = adapters.run_count_flops(**REFERENCE, seq_len=256)
    assert fwd == 37_240_832
    assert train == 3 * 37_240_832


def test_count_flops_scales_with_sequence_length():
    f1, _ = adapters.run_count_flops(**REFERENCE, seq_len=256)
    f2, _ = adapters.run_count_flops(**REFERENCE, seq_len=512)
    assert f2 - f1 == 4 * 4 * 256 * 512          # 4 * n_layers * delta_L * d_model
    ref = count_flops_ref(10000, 512, 4, 16, 1344, 1024)
    assert adapters.run_count_flops(**REFERENCE, seq_len=1024) == ref
