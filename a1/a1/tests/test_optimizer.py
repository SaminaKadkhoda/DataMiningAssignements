"""Part 2: AdamW and the learning-rate schedule."""
from __future__ import annotations

import math

import pytest
import torch

from . import adapters
from ._reference import cosine_lr_ref


def _problem(seed):
    torch.manual_seed(seed)
    W = torch.nn.Parameter(torch.randn(6, 4))
    b = torch.nn.Parameter(torch.randn(6))
    X = torch.randn(32, 4)
    Y = torch.randn(32, 6)
    return [W, b], (X, Y)


def _loss(params, data):
    W, b = params
    X, Y = data
    return ((X @ W.T + b - Y) ** 2).mean()


@pytest.mark.parametrize("lr,wd", [(1e-2, 0.0), (1e-2, 0.1), (3e-3, 0.5)])
def test_adamw_matches_torch(lr, wd):
    AdamW = adapters.get_adamw_cls()
    mine, data = _problem(0)
    theirs, _ = _problem(0)
    opt_a = AdamW(mine, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=wd)
    opt_b = torch.optim.AdamW(theirs, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=wd)
    for _ in range(25):
        for opt, ps in ((opt_a, mine), (opt_b, theirs)):
            opt.zero_grad()
            _loss(ps, data).backward()
            opt.step()
    for p, q in zip(mine, theirs):
        torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-5)


def test_adamw_decays_weights_without_gradient_signal():
    """Decoupled weight decay: with zero gradients the parameters shrink by (1 - lr * wd) per step."""
    AdamW = adapters.get_adamw_cls()
    p = torch.nn.Parameter(torch.ones(5) * 2.0)
    opt = AdamW([p], lr=0.1, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.5)
    p.grad = torch.zeros(5)
    opt.step()
    torch.testing.assert_close(p.detach(), torch.full((5,), 2.0 * (1 - 0.1 * 0.5)), atol=1e-6, rtol=1e-6)


def test_adamw_is_an_optimizer_with_state_dict():
    AdamW = adapters.get_adamw_cls()
    params, data = _problem(1)
    opt = AdamW(params, lr=1e-3)
    assert isinstance(opt, torch.optim.Optimizer)
    opt.zero_grad(); _loss(params, data).backward(); opt.step()
    sd = opt.state_dict()
    assert "state" in sd and "param_groups" in sd and len(sd["state"]) == 2
    assert opt.param_groups[0]["lr"] == 1e-3
    opt.param_groups[0]["lr"] = 5e-4            # the training loop sets the lr this way
    opt.zero_grad(); _loss(params, data).backward(); opt.step()


def test_lr_cosine_schedule():
    kw = dict(max_learning_rate=1e-3, min_learning_rate=1e-4, warmup_iters=100, cosine_cycle_iters=1000)
    assert adapters.run_get_lr_cosine_schedule(it=0, **kw) == 0.0
    assert math.isclose(adapters.run_get_lr_cosine_schedule(it=50, **kw), 5e-4)
    assert math.isclose(adapters.run_get_lr_cosine_schedule(it=100, **kw), 1e-3)
    assert math.isclose(adapters.run_get_lr_cosine_schedule(it=550, **kw), 5.5e-4, rel_tol=1e-9)
    assert math.isclose(adapters.run_get_lr_cosine_schedule(it=1000, **kw), 1e-4)
    assert math.isclose(adapters.run_get_lr_cosine_schedule(it=5000, **kw), 1e-4)
    for it in (0, 7, 99, 100, 101, 333, 999, 1000, 1500):
        assert math.isclose(adapters.run_get_lr_cosine_schedule(it=it, **kw), cosine_lr_ref(it, 1e-3, 1e-4, 100, 1000), rel_tol=1e-9)
