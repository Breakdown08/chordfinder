"""Accuracy on the test split of fabianavinci/guitar-chords-v3, scored per file like the thesis."""

import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from chordfinder.audio import Audio
from chordfinder.model import ChordModel

# The dataset licence is unknown, so it is never committed: see the README for where to download it.
DATASET: Path = Path(os.environ.get("CHORDFINDER_DATASET", Path.home() / "datasets/guitar-chords-v3/data"))


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


def dataset_files(dataset: Path = DATASET) -> list[Path]:
    files: list[Path] = sorted((dataset / "Test").glob("*/*.wav"))
    if not files:
        raise FileNotFoundError(f"no test files under {dataset / 'Test'}; set CHORDFINDER_DATASET")
    return files


def load(path: Path, sr: int) -> Audio:
    import librosa

    y: Audio = np.asarray(librosa.load(path, sr=sr)[0], np.float32)
    return y


def evaluate(model: ChordModel, files: list[Path]) -> Report:
    report: Report = Report()
    f: Path
    for f in files:
        label: str
        confidence: float
        label, confidence = model.predict(load(f, model.meta.sr))
        report.predictions.append(Prediction(f"{f.parent.name}/{f.name}", f.parent.name, label, confidence))
    return report
