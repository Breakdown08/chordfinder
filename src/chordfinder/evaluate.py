"""Accuracy on the test split of fabianavinci/guitar-chords-v3, scored per file like the diploma."""

import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from chordfinder.model import ChordModel

# The dataset licence is unknown, so it is never committed: point CHORDFINDER_DATASET at the local copy.
DATASET = Path(os.environ.get("CHORDFINDER_DATASET", Path.home() / "datasets/guitar-chords-v3/data"))


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
    def accuracy(self) -> float:
        return sum(p.label == p.truth for p in self.predictions) / len(self.predictions)

    def per_class(self) -> dict[str, float]:
        total, hits = Counter(), Counter()
        for p in self.predictions:
            total[p.truth] += 1
            hits[p.truth] += p.label == p.truth
        return {c: hits[c] / total[c] for c in sorted(total)}


def test_files(dataset: Path = DATASET) -> list[Path]:
    files = sorted((dataset / "Test").glob("*/*.wav"))
    if not files:
        raise FileNotFoundError(f"no test files under {dataset / 'Test'}; set CHORDFINDER_DATASET")
    return files


def evaluate(model: ChordModel, files: list[Path]) -> Report:
    import librosa

    report = Report()
    for f in files:
        y, _ = librosa.load(f, sr=model.meta.sr)
        label, confidence = model.predict(y)
        report.predictions.append(Prediction(f"{f.parent.name}/{f.name}", f.parent.name, label, confidence))
    return report
