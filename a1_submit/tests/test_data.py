"""Part 2: data loading and checkpointing."""
from __future__ import annotations

import io

import numpy as np
import torch

from . import adapters


def test_get_batch_shapes_and_shift():
    data = np.arange(1000, dtype=np.uint16)
    x, y = adapters.run_get_batch(dataset=data, batch_size=8, context_length=16, device="cpu")
    assert x.shape == (8, 16) and y.shape == (8, 16)
    assert x.dtype == torch.int64 and y.dtype == torch.int64
    torch.testing.assert_close(y, x + 1)                 # targets are the next tokens
    assert (x[:, 1:] == x[:, :-1] + 1).all()              # each row is a contiguous window
    assert x.min() >= 0 and y.max() <= 999               # windows never run past the end


def test_get_batch_uses_the_whole_dataset():
    data = np.arange(200, dtype=np.uint16)
    starts = set()
    for _ in range(200):
        x, _ = adapters.run_get_batch(dataset=data, batch_size=4, context_length=10, device="cpu")
        starts.update(x[:, 0].tolist())
    assert min(starts) == 0, "the first window must be reachable"
    assert max(starts) == 200 - 10 - 1, "the last valid start is len - context - 1"


def test_get_batch_from_memmap(tmp_path):
    p = tmp_path / "train.bin"
    np.arange(500, dtype=np.uint16).tofile(p)
    data = np.memmap(p, dtype=np.uint16, mode="r")
    x, y = adapters.run_get_batch(dataset=data, batch_size=3, context_length=7, device="cpu")
    torch.testing.assert_close(y, x + 1)
    assert x.dtype == torch.int64


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(4, 4))
        self.b = torch.nn.Parameter(torch.randn(4))

    def forward(self, x):
        return x @ self.w + self.b


def _train_a_bit(model, opt, steps):
    for _ in range(steps):
        opt.zero_grad()
        model(torch.randn(8, 4)).pow(2).mean().backward()
        opt.step()


def _roundtrip(target):
    AdamW = adapters.get_adamw_cls()
    torch.manual_seed(0)
    m1 = _Tiny(); o1 = AdamW(m1.parameters(), lr=1e-2)
    _train_a_bit(m1, o1, 5)
    adapters.run_save_checkpoint(model=m1, optimizer=o1, iteration=1234, out=target)
    if hasattr(target, "seek"):
        target.seek(0)
    torch.manual_seed(1)
    m2 = _Tiny(); o2 = AdamW(m2.parameters(), lr=1e-2)
    it = adapters.run_load_checkpoint(src=target, model=m2, optimizer=o2)
    assert it == 1234
    for a, b in zip(m1.state_dict().values(), m2.state_dict().values()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    s1, s2 = o1.state_dict(), o2.state_dict()
    for k in s1["state"]:
        for name, v in s1["state"][k].items():
            if isinstance(v, torch.Tensor):
                torch.testing.assert_close(v, s2["state"][k][name], atol=0, rtol=0)
    # after loading, both continue identically
    torch.manual_seed(2); _train_a_bit(m1, o1, 3)
    torch.manual_seed(2); _train_a_bit(m2, o2, 3)
    for a, b in zip(m1.state_dict().values(), m2.state_dict().values()):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


def test_checkpoint_roundtrip_path(tmp_path):
    _roundtrip(tmp_path / "ckpt.pt")


def test_checkpoint_roundtrip_file_object():
    _roundtrip(io.BytesIO())
