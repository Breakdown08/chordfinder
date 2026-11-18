"""Accuracy on the test split of fabianavinci/guitar-chords-v3, scored per file like the thesis."""

import math
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from chordfinder.audio import Audio
from chordfinder.features import Meta

# The dataset (CC BY-SA 4.0) is not committed either: see the README for where to download it.
DATASET: Path = Path(os.environ.get("CHORDFINDER_DATASET", Path.home() / "datasets/guitar-chords-v3/data"))


class Predictor(Protocol):
    meta: Meta

    def predict(self, audio_16k: Audio) -> tuple[str, float]: ...


@dataclass
class Prediction:
    file: str  # "<class>/<name>.wav", as in baseline/old-model.json
    truth: str
    label: str
    confidence: float


@dataclass
class Report:
    predictions: list[Prediction] = field(default_factory=list)

    @property
    def hits(self) -> int:
        return sum(p.label == p.truth for p in self.predictions)

    @property
    def accuracy(self) -> float:
        return self.hits / len(self.predictions)

    def per_class(self) -> dict[str, float]:
        total: Counter[str] = Counter()
        hits: Counter[str] = Counter()
        p: Prediction
        for p in self.predictions:
            total[p.truth] += 1
            hits[p.truth] += p.label == p.truth
        return {c: hits[c] / total[c] for c in sorted(total)}

    def confusion(self, classes: Sequence[str]) -> list[list[int]]:
        """rows = truth, columns = predicted label, both in `classes` order."""
        index: dict[str, int] = {c: i for i, c in enumerate(classes)}
        matrix: list[list[int]] = [[0] * len(classes) for _ in classes]
        p: Prediction
        for p in self.predictions:
            matrix[index[p.truth]][index[p.label]] += 1
        return matrix


def dataset_files(dataset: Path = DATASET) -> list[Path]:
    files: list[Path] = sorted((dataset / "Test").glob("*/*.wav"))
    if not files:
        raise FileNotFoundError(f"no test files under {dataset / 'Test'}; set CHORDFINDER_DATASET")
    return files


def load(path: Path, sr: int) -> Audio:
    import librosa

    y: Audio = np.asarray(librosa.load(path, sr=sr)[0], np.float32)
    return y


def evaluate(model: Predictor, files: list[Path]) -> Report:
    report: Report = Report()
    f: Path
    for f in files:
        label: str
        confidence: float
        label, confidence = model.predict(load(f, model.meta.sr))
        report.predictions.append(Prediction(f"{f.parent.name}/{f.name}", f.parent.name, label, confidence))
    return report


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar: 1.0 if b + c == 0, else min(1.0, 2 * Binom(b + c, 0.5).cdf(min(b, c)))."""
    n: int = b + c
    if n == 0:
        return 1.0
    tail: int = sum(math.comb(n, k) for k in range(min(b, c) + 1))
    p: float = min(1.0, 2 * tail / 2**n)
    return p


@dataclass
class Comparison:
    old: Report
    new: Report
    old_only: int
    new_only: int
    both_wrong: int
    mcnemar_p: float


def compare(old: Report, new: Report) -> Comparison:
    """Paired on Prediction.file. ValueError if the two reports cover different files."""
    old_by_file: dict[str, Prediction] = {p.file: p for p in old.predictions}
    new_by_file: dict[str, Prediction] = {p.file: p for p in new.predictions}
    if (
        old_by_file.keys() != new_by_file.keys()
        or len(old_by_file) != len(old.predictions)
        or len(new_by_file) != len(new.predictions)
    ):
        raise ValueError("the two reports do not cover the same files")
    old_only: int = 0
    new_only: int = 0
    both_wrong: int = 0
    file: str
    o: Prediction
    for file, o in old_by_file.items():
        n: Prediction = new_by_file[file]
        old_ok: bool = o.label == o.truth
        new_ok: bool = n.label == n.truth
        old_only += old_ok and not new_ok
        new_only += new_ok and not old_ok
        both_wrong += not old_ok and not new_ok
    return Comparison(old, new, old_only, new_only, both_wrong, mcnemar_p(old_only, new_only))
