"""Real-text corpora for pretraining-data comparisons.

The default training set in :mod:`glm53_flash.tasks` is synthetic. This module
adds two real corpora, downloaded once and cached under a local data directory:

- ``shakespeare``: "The Complete Works of William Shakespeare", Project
  Gutenberg eBook #100 (public domain), one plain-text file.
- ``python``: selected CPython 3.12.7 standard-library modules, pinned to the
  release tag (PSF License Agreement for Python), fetched as raw source files.

Both are used as raw UTF-8 bytes through :class:`ByteTokenizer`, so the model
and training loop are identical to the synthetic path; only the text source
changes. Each download writes a ``source.json`` receipt with URLs and hashes.
"""
from __future__ import annotations

import hashlib
import json
import random
import urllib.request
from pathlib import Path

import numpy as np
import torch

from .tasks import pretraining_text
from .tokenizer import ByteTokenizer

CPYTHON = "https://raw.githubusercontent.com/python/cpython/v3.12.7/Lib"

CORPORA: dict[str, dict[str, object]] = {
    "shakespeare": {
        "title": "The Complete Works of William Shakespeare (Project Gutenberg #100)",
        "license": "public domain",
        "files": {
            "pg100.txt": "https://www.gutenberg.org/cache/epub/100/pg100.txt",
        },
    },
    "python": {
        "title": "CPython 3.12.7 standard library modules",
        "license": "PSF License Agreement for Python 3.12.7",
        "files": {
            "argparse.py": f"{CPYTHON}/argparse.py",
            "pathlib.py": f"{CPYTHON}/pathlib.py",
            "shlex.py": f"{CPYTHON}/shlex.py",
            "json__init__.py": f"{CPYTHON}/json/__init__.py",
            "re__init__.py": f"{CPYTHON}/re/__init__.py",
            "re__compiler.py": f"{CPYTHON}/re/_compiler.py",
            "unittest__case.py": f"{CPYTHON}/unittest/case.py",
            "unittest__main.py": f"{CPYTHON}/unittest/main.py",
        },
    },
}


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "glm53-flash-education/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def ensure_corpus(name: str, data_dir: Path) -> dict[str, object]:
    """Download a named corpus if needed; return its receipt and file paths."""
    if name not in CORPORA:
        raise ValueError(f"unknown corpus: {name}")
    spec = CORPORA[name]
    destination = data_dir / name
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    for filename, url in spec["files"].items():  # type: ignore[union-attr]
        path = destination / filename
        if not path.exists():
            path.write_bytes(fetch(url))
        entries.append({
            "file": filename,
            "url": url,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    receipt = {
        "corpus": name,
        "title": spec["title"],
        "license": spec["license"],
        "files": entries,
        "total_bytes": sum(entry["bytes"] for entry in entries),
    }
    (destination / "source.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    receipt["paths"] = [str(destination / entry["file"]) for entry in entries]
    return receipt


class SyntheticCorpus:
    """The original generated coding corpus, byte-for-byte identical to before."""

    name = "synthetic"

    def __init__(self, *, seed: int, tokenizer: ByteTokenizer | None = None, validation_seed_offset: int = 7919):
        self.seed = seed
        self.validation_seed = seed + validation_seed_offset
        self.tokenizer = tokenizer or ByteTokenizer()

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "title": "generated synthetic coding corpus", "license": "generated"}

    def _row(self, index: int, *, seed: int, sequence_length: int) -> list[int]:
        text = pretraining_text(index, seed=seed)
        ids = self.tokenizer.encode(text, bos=True, eos=True)
        if len(ids) > sequence_length + 1:
            raise ValueError(f"training example exceeds sequence length: {len(ids)}")
        ids += [self.tokenizer.pad_id] * (sequence_length + 1 - len(ids))
        return ids

    def _batch(self, start_index: int, *, seed: int, batch_size: int, sequence_length: int, device: torch.device):
        rows = [self._row(start_index + offset, seed=seed, sequence_length=sequence_length) for offset in range(batch_size)]
        values = torch.tensor(rows, dtype=torch.long, device=device)
        inputs, labels = values[:, :-1], values[:, 1:]
        labels = labels.masked_fill(labels == self.tokenizer.pad_id, -100)
        return inputs, labels, int((labels != -100).sum().item())

    def training_batch(self, step: int, *, batch_size: int, sequence_length: int, device: torch.device):
        return self._batch(step * batch_size, seed=self.seed, batch_size=batch_size, sequence_length=sequence_length, device=device)

    def validation_batch(self, batch_index: int, *, batch_size: int, sequence_length: int, device: torch.device):
        return self._batch(batch_index * batch_size, seed=self.validation_seed, batch_size=batch_size, sequence_length=sequence_length, device=device)


class RealTextCorpus:
    """Packed UTF-8 bytes from real documents with a chronological hold-out."""

    def __init__(
        self,
        paths: list[Path],
        *,
        name: str,
        tokenizer: ByteTokenizer | None = None,
        validation_fraction: float = 0.05,
        seed: int = 4242,
    ):
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between zero and one")
        text = "\n\n".join(path.read_text(encoding="utf-8", errors="replace") for path in paths)
        ids = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64) + ByteTokenizer.byte_offset
        split = int(len(ids) * (1.0 - validation_fraction))
        if split < 1 or len(ids) - split < 1:
            raise ValueError("corpus too small for the requested split")
        self.name = name
        self.paths = paths
        self.tokenizer = tokenizer or ByteTokenizer()
        self.validation_fraction = validation_fraction
        self.seed = seed
        self.train_ids = ids[:split]
        self.validation_ids = ids[split:]

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "files": [path.name for path in self.paths],
            "license": CORPORA[self.name]["license"] if self.name in CORPORA else "see source.json",
            "train_bytes": int(self.train_ids.size),
            "validation_bytes": int(self.validation_ids.size),
            "validation_fraction": self.validation_fraction,
        }

    def _windows(self, region: np.ndarray, starts: list[int], *, sequence_length: int, device: torch.device):
        rows = np.stack([region[start : start + sequence_length + 1] for start in starts])
        values = torch.tensor(rows, dtype=torch.long, device=device)
        inputs, labels = values[:, :-1], values[:, 1:]
        return inputs, labels, sequence_length * len(starts)

    def training_batch(self, step: int, *, batch_size: int, sequence_length: int, device: torch.device):
        rng = random.Random((self.seed + 1) * 1_000_003 + step)
        limit = self.train_ids.size - sequence_length - 1
        if limit <= 0:
            raise ValueError("training split shorter than one sequence")
        starts = [rng.randrange(limit) for _ in range(batch_size)]
        return self._windows(self.train_ids, starts, sequence_length=sequence_length, device=device)

    def validation_batch(self, batch_index: int, *, batch_size: int, sequence_length: int, device: torch.device):
        rng = random.Random((self.seed + 2) * 2_000_003 + batch_index)
        limit = self.validation_ids.size - sequence_length - 1
        if limit <= 0:
            raise ValueError("validation split shorter than one sequence")
        starts = [rng.randrange(limit) for _ in range(batch_size)]
        return self._windows(self.validation_ids, starts, sequence_length=sequence_length, device=device)
