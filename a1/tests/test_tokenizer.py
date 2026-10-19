"""Part 1: byte-level BPE tokenizer."""
from __future__ import annotations

import io
import tracemalloc
from pathlib import Path

import pytest

from . import adapters
from ._reference import encode_reference, train_bpe_reference

FIXTURES = Path(__file__).parent / "fixtures"
CORPUS = FIXTURES / "corpus.txt"
EOT = "<|endoftext|>"


@pytest.fixture(scope="module")
def trained():
    """Student tokenizer trained on the fixture corpus, plus the reference result."""
    vocab, merges = adapters.run_train_bpe(input_path=CORPUS, vocab_size=400, special_tokens=[EOT])
    ref_vocab, ref_merges = train_bpe_reference(CORPUS.read_text(encoding="utf-8"), 400, [EOT])
    return vocab, merges, ref_vocab, ref_merges


# --------------------------------------------------------------------------- training

def test_train_bpe(trained):
    vocab, merges, ref_vocab, ref_merges = trained
    assert len(vocab) == 400
    assert all(vocab[i] == bytes([i]) for i in range(256)), "ids 0..255 must be the single bytes"
    assert vocab[256] == EOT.encode("utf-8"), "special tokens come right after the 256 bytes"
    assert len(merges) == 400 - 256 - 1
    assert merges == ref_merges, "merge sequence differs from the reference (check counts and the tie-break rule)"
    assert vocab == ref_vocab


def test_train_bpe_merge_order(tmp_path):
    """Frequencies decide, ties go to the lexicographically greater pair."""
    text = "ab ab ab cd cd cd ef ef ef ef zz zz"
    p = tmp_path / "t.txt"
    p.write_text(text)
    vocab, merges = adapters.run_train_bpe(input_path=p, vocab_size=256 + 3, special_tokens=[])
    ref_vocab, ref_merges = train_bpe_reference(text, 256 + 3, [])
    assert merges == ref_merges
    # ' ef' occurs most (4x, incl. the leading space for all but the first): the first merge must involve it
    assert merges[0] == ref_merges[0]


def test_train_bpe_special_tokens(tmp_path):
    """Special tokens are never merged into and never split; text around them is unaffected."""
    text = f"hello world{EOT}hello world{EOT}hello world"
    p = tmp_path / "t.txt"
    p.write_text(text)
    vocab, merges = adapters.run_train_bpe(input_path=p, vocab_size=256 + 1 + 6, special_tokens=[EOT])
    assert vocab[256] == EOT.encode("utf-8")
    for a, b in merges:   # the text was split on the special token before pre-tokenisation,
        assert b"<" not in a + b and b"|" not in a + b, f"merge touched a special token: {a!r}+{b!r}"
    ref_vocab, ref_merges = train_bpe_reference(text, 256 + 1 + 6, [EOT])
    assert merges == ref_merges


def test_train_bpe_vocab_size_reached(trained):
    vocab, merges, _, _ = trained
    assert max(vocab) == 399 and set(vocab) == set(range(400))
    assert all(isinstance(v, bytes) for v in vocab.values())


# --------------------------------------------------------------------------- encoding / decoding

@pytest.mark.parametrize("text", [
    "Once upon a time, a little fox lived at the edge of a wide green forest.",
    "The fox and the rabbit are friends now!",
    "1, 22, 333, 4444 and the year 2026: prices like $12.50.",
    "   leading spaces and\ttabs\nand newlines\n\n",
    "Es war einmal ein kleiner Fuchs im großen Wald.",
])
def test_encode_matches_reference(trained, text):
    vocab, merges, _, _ = trained
    tok = adapters.get_tokenizer(vocab=vocab, merges=merges, special_tokens=[EOT])
    assert tok.encode(text) == encode_reference(text, vocab, merges, [EOT])


@pytest.mark.parametrize("text", [
    "hello world",
    "Grüße aus Wien! Ärger über Öl. ß",
    "emoji 🦊🐇 and CJK 日本語 and Arabic مرحبا",
    "mixed 🇦🇹 flags and combining e\u0301 accents",
    "",
    " ",
    "\n\n\t",
])
def test_roundtrip_unicode(trained, text):
    vocab, merges, _, _ = trained
    tok = adapters.get_tokenizer(vocab=vocab, merges=merges, special_tokens=[EOT])
    ids = tok.encode(text)
    assert all(isinstance(i, int) and 0 <= i < 400 for i in ids)
    assert tok.decode(ids) == text


def test_special_tokens(trained):
    vocab, merges, _, _ = trained
    tok = adapters.get_tokenizer(vocab=vocab, merges=merges, special_tokens=[EOT])
    eot = 256
    ids = tok.encode(f"hello{EOT}world{EOT}")
    assert ids.count(eot) == 2 and ids[-1] == eot
    assert tok.decode(ids) == f"hello{EOT}world{EOT}"
    # a special token inside a word still survives as one id
    ids = tok.encode(f"he{EOT}llo")
    assert eot in ids and tok.decode(ids) == f"he{EOT}llo"


def test_overlapping_special_tokens(trained):
    """With both '<|endoftext|>' and '<|endoftext|><|endoftext|>' registered, the longer wins."""
    vocab, merges, _, _ = trained
    vocab = dict(vocab)
    double = EOT + EOT
    vocab[len(vocab)] = double.encode("utf-8")
    tok = adapters.get_tokenizer(vocab=vocab, merges=merges, special_tokens=[EOT, double])
    ids = tok.encode(f"a{double}b{EOT}c")
    assert ids.count(400) == 1 and ids.count(256) == 1
    assert tok.decode(ids) == f"a{double}b{EOT}c"


def test_encode_merge_order():
    """Merges must be applied by rank: (a,b) then (ab,c) turns 'abc' into one token."""
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"ab"
    vocab[257] = b"abc"
    vocab[258] = b"bc"
    merges = [(b"a", b"b"), (b"ab", b"c"), (b"b", b"c")]
    tok = adapters.get_tokenizer(vocab=vocab, merges=merges, special_tokens=None)
    assert tok.encode("abc") == [257]
    assert tok.encode("bc") == [258]
    assert tok.encode("abcbc") == [257, 258]
    # a later merge with a lower-rank alternative available: the rank decides, not position
    merges2 = [(b"b", b"c"), (b"a", b"b"), (b"ab", b"c")]
    tok2 = adapters.get_tokenizer(vocab=vocab, merges=merges2, special_tokens=None)
    assert tok2.encode("abc") == [97, 258]


def test_decode_invalid_utf8(trained):
    vocab, merges, _, _ = trained
    tok = adapters.get_tokenizer(vocab=vocab, merges=merges, special_tokens=[EOT])
    out = tok.decode([0xE2, 0x82])          # a truncated 3-byte sequence
    assert "\ufffd" in out
    out = tok.decode([0xF0, 0x9F, 0xA6])    # the first three bytes of the fox emoji
    assert "\ufffd" in out


def test_encode_iterable_matches_encode(trained):
    vocab, merges, _, _ = trained
    tok = adapters.get_tokenizer(vocab=vocab, merges=merges, special_tokens=[EOT])
    text = CORPUS.read_text(encoding="utf-8")
    streamed = list(tok.encode_iterable(io.StringIO(text)))
    assert streamed == tok.encode(text)


def test_encode_iterable_memory(trained, tmp_path):
    """Streaming a file must not materialise it: peak memory stays far below the file size."""
    vocab, merges, _, _ = trained
    tok = adapters.get_tokenizer(vocab=vocab, merges=merges, special_tokens=[EOT])
    line = "Once upon a time, a little fox lived at the edge of a wide green forest.\n"
    p = tmp_path / "big.txt"
    with open(p, "w", encoding="utf-8") as f:
        for _ in range(20000):                      # about 1.5 MB
            f.write(line)
    tracemalloc.start()
    n = 0
    with open(p, encoding="utf-8") as f:
        for _ in tok.encode_iterable(f):
            n += 1
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert n == 20000 * len(tok.encode(line))
    assert peak < 3_000_000, f"peak traced memory {peak/1e6:.1f} MB: are you reading the whole file or collecting all ids?"


def test_from_files_roundtrip(trained, tmp_path):
    """If the tokenizer can be saved and reloaded through the adapter, ids must be identical."""
    vocab, merges, _, _ = trained
    tok = adapters.get_tokenizer(vocab=vocab, merges=merges, special_tokens=[EOT])
    text = "The rabbit lost its way in the forest, and the fox showed it the path home."
    ids = tok.encode(text)
    tok2 = adapters.get_tokenizer(vocab=dict(vocab), merges=list(merges), special_tokens=[EOT])
    assert tok2.encode(text) == ids
