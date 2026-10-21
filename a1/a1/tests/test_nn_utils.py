"""Part 2: loss, gradient clipping; Part 3: sampling."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import adapters
from ._reference import make_lm_weights

ATOL, RTOL = 1e-5, 1e-4


def g(seed=0):
    return torch.Generator().manual_seed(seed)


# --------------------------------------------------------------------------- cross-entropy

def test_cross_entropy():
    logits = torch.randn(4, 7, 20, generator=g(0))
    targets = torch.randint(0, 20, (4, 7), generator=g(1))
    out = adapters.run_cross_entropy(inputs=logits, targets=targets)
    assert out.shape == ()
    torch.testing.assert_close(out, F.cross_entropy(logits.reshape(-1, 20), targets.reshape(-1)), atol=ATOL, rtol=RTOL)


def test_cross_entropy_is_stable():
    logits = torch.randn(3, 11, generator=g(2)) * 1000.0     # exp() of these overflows
    targets = torch.randint(0, 11, (3,), generator=g(3))
    out = adapters.run_cross_entropy(inputs=logits, targets=targets)
    assert torch.isfinite(out)
    torch.testing.assert_close(out, F.cross_entropy(logits, targets), atol=1e-3, rtol=1e-4)


def test_cross_entropy_gradient():
    logits = torch.randn(2, 5, 9, generator=g(4), requires_grad=True)
    targets = torch.randint(0, 9, (2, 5), generator=g(5))
    adapters.run_cross_entropy(inputs=logits, targets=targets).backward()
    ref = torch.randn(2, 5, 9, generator=g(4), requires_grad=True)
    F.cross_entropy(ref.reshape(-1, 9), targets.reshape(-1)).backward()
    torch.testing.assert_close(logits.grad, ref.grad, atol=ATOL, rtol=RTOL)


# --------------------------------------------------------------------------- gradient clipping

def _params(seed, scale):
    torch.manual_seed(seed)
    ps = [torch.nn.Parameter(torch.randn(s)) for s in [(3, 4), (7,), (2, 2, 2)]]
    for p in ps:
        p.grad = torch.randn(p.shape) * scale
    return ps


def test_gradient_clipping_clips():
    ps = _params(6, scale=5.0)
    ref = [p.grad.clone() for p in ps]
    total = torch.sqrt(sum((r ** 2).sum() for r in ref))
    assert total > 1.0
    adapters.run_gradient_clipping(parameters=ps, max_l2_norm=1.0)
    coef = 1.0 / (total + 1e-6)
    for p, r in zip(ps, ref):
        torch.testing.assert_close(p.grad, r * coef, atol=ATOL, rtol=RTOL)
    new_total = torch.sqrt(sum((p.grad ** 2).sum() for p in ps))
    assert abs(new_total.item() - 1.0) < 1e-4


def test_gradient_clipping_leaves_small_gradients_alone():
    ps = _params(7, scale=0.01)
    ref = [p.grad.clone() for p in ps]
    adapters.run_gradient_clipping(parameters=ps, max_l2_norm=1.0)
    for p, r in zip(ps, ref):
        torch.testing.assert_close(p.grad, r, atol=0, rtol=0)


def test_gradient_clipping_skips_params_without_grad():
    ps = _params(8, scale=5.0)
    ps.append(torch.nn.Parameter(torch.zeros(3)))     # no .grad
    adapters.run_gradient_clipping(parameters=ps, max_l2_norm=1.0)   # must not raise
    assert ps[-1].grad is None


# --------------------------------------------------------------------------- generation

CFG = dict(vocab_size=23, context_length=32, d_model=32, num_layers=2, num_heads=4, d_ff=88, rope_theta=10000.0)


def _logits(weights, ids):
    return adapters.run_transformer_lm(**CFG, weights=weights, in_indices=torch.tensor([ids]))[0, -1]


def test_generate_greedy_at_low_temperature():
    w = make_lm_weights(CFG["vocab_size"], CFG["d_model"], CFG["num_layers"], CFG["num_heads"], CFG["d_ff"], seed=9)
    prompt = [1, 2, 3]
    out = adapters.run_generate(**CFG, weights=w, prompt_ids=prompt, max_new_tokens=8, temperature=1e-4, top_p=1.0, eos_id=None, seed=0)
    assert len(out) == 8
    ids = list(prompt)
    for t in out:
        expected = int(torch.argmax(_logits(w, ids)))
        assert t == expected, "with temperature -> 0 sampling must reduce to argmax"
        ids.append(t)


def test_generate_respects_top_p_and_context():
    w = make_lm_weights(CFG["vocab_size"], CFG["d_model"], CFG["num_layers"], CFG["num_heads"], CFG["d_ff"], seed=10)
    prompt = [5, 6]
    for seed in range(4):
        out = adapters.run_generate(**CFG, weights=w, prompt_ids=prompt, max_new_tokens=6, temperature=1.0, top_p=0.5, eos_id=None, seed=seed)
        assert len(out) == 6
        ids = list(prompt)
        for t in out:
            probs = torch.softmax(_logits(w, ids), -1)
            sp, si = torch.sort(probs, descending=True)
            cum = torch.cumsum(sp, 0)
            nucleus = si[: int((cum < 0.5).sum()) + 1].tolist()   # smallest set with mass >= 0.5
            assert t in nucleus, f"token {t} outside the top-p nucleus {nucleus}"
            ids.append(t)


def test_generate_stops_at_eos():
    """Greedy decoding produces some first token t0; with eos_id = t0 generation must stop after it."""
    w = make_lm_weights(CFG["vocab_size"], CFG["d_model"], CFG["num_layers"], CFG["num_heads"], CFG["d_ff"], seed=11)
    first = adapters.run_generate(**CFG, weights=w, prompt_ids=[1, 2], max_new_tokens=1, temperature=1e-4, top_p=1.0, eos_id=None, seed=0)
    assert len(first) == 1
    out = adapters.run_generate(**CFG, weights=w, prompt_ids=[1, 2], max_new_tokens=10, temperature=1e-4, top_p=1.0, eos_id=first[0], seed=0)
    assert out == first, "generation must stop right after producing eos_id (and include it)"
