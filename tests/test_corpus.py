from __future__ import annotations

from pathlib import Path

import torch

from glm53_flash.corpus import CORPORA, RealTextCorpus, SyntheticCorpus
from glm53_flash.tokenizer import ByteTokenizer


def test_synthetic_corpus_matches_original_batch_shape_and_shift():
    corpus = SyntheticCorpus(seed=42)
    inputs, labels, tokens = corpus.training_batch(0, batch_size=3, sequence_length=128, device=torch.device("cpu"))
    assert inputs.shape == (3, 128) and labels.shape == (3, 128)
    observed = labels[:, :-1] != -100
    assert torch.equal(inputs[:, 1:][observed], labels[:, :-1][observed])
    assert tokens == int((labels != -100).sum())


def test_synthetic_corpus_is_deterministic():
    left = SyntheticCorpus(seed=42).training_batch(7, batch_size=2, sequence_length=128, device=torch.device("cpu"))
    right = SyntheticCorpus(seed=42).training_batch(7, batch_size=2, sequence_length=128, device=torch.device("cpu"))
    assert torch.equal(left[0], right[0])
    other = SyntheticCorpus(seed=43).training_batch(7, batch_size=2, sequence_length=128, device=torch.device("cpu"))
    assert not torch.equal(left[0], other[0])


def test_real_corpus_splits_and_batches_deterministically(tmp_path: Path):
    source = tmp_path / "sample.txt"
    source.write_text("abcdefghij" * 200, encoding="utf-8")
    corpus = RealTextCorpus([source], name="sample", validation_fraction=0.1)
    assert corpus.train_ids.size == 1800 and corpus.validation_ids.size == 200
    inputs, labels, tokens = corpus.training_batch(3, batch_size=4, sequence_length=16, device=torch.device("cpu"))
    assert inputs.shape == (4, 16) and tokens == 4 * 16
    assert torch.equal(inputs[:, 1:], labels[:, :-1])
    again = corpus.training_batch(3, batch_size=4, sequence_length=16, device=torch.device("cpu"))
    assert torch.equal(inputs, again[0])
    val_inputs, _, _ = corpus.validation_batch(0, batch_size=2, sequence_length=16, device=torch.device("cpu"))
    assert val_inputs.shape == (2, 16)
    # byte tokens are offset by byte_offset
    assert int(inputs.min()) >= ByteTokenizer.byte_offset
    assert int(inputs.max()) < ByteTokenizer.vocab_size


def test_real_corpus_validation_region_is_disjoint(tmp_path: Path):
    source = tmp_path / "sample.txt"
    source.write_text("a" * 800 + "b" * 200, encoding="utf-8")
    corpus = RealTextCorpus([source], name="sample", validation_fraction=0.2)
    assert corpus.train_ids.size == 800 and corpus.validation_ids.size == 200
    a, b = ByteTokenizer.byte_offset + ord("a"), ByteTokenizer.byte_offset + ord("b")
    assert set(corpus.train_ids.tolist()) == {a}
    assert set(corpus.validation_ids.tolist()) == {b}


def test_corpus_registry_covers_both_real_sources():
    assert set(CORPORA) == {"shakespeare", "python"}
    for spec in CORPORA.values():
        assert spec["files"] and spec["license"]
