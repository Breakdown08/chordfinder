"""Time from the strum to a stable answer, and false chords on the way (the bridge must not lie).

Replays every 4th test file per class (80) through Stream + Stabilizer exactly as the server does,
in HOP_SECONDS frames after a second of silence. Thresholds hold the numbers in stream.py.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from chordfinder.audio import Audio
from chordfinder.evaluate import dataset_files, load
from chordfinder.model import ChordModel
from chordfinder.stream import HOP_SECONDS, Stabilizer, Stream, recognize
from tests.conftest import needs_dataset

RATE: int = 48000
ONSET_LEVEL: float = 0.05  # strum onset: the first sample above this
LISTEN_SECONDS: float = 2.0


@dataclass
class Replay:
    truth: str
    latency: float | None  # seconds from onset to the first stable correct answer
    false_chord: bool  # a wrong chord became stable after the onset


def replay(model: ChordModel, path: Path) -> Replay:
    x: Audio = load(path, RATE)
    onset: float = int(np.argmax(np.abs(x) > ONSET_LEVEL)) / RATE
    y: Audio = np.concatenate((np.zeros(RATE, np.float32), x))[: int(RATE * (1 + LISTEN_SECONDS))]
    stream: Stream = Stream(RATE)
    stabilizer: Stabilizer = Stabilizer()
    truth: str = path.parent.name
    latency: float | None = None
    false_chord: bool = False
    hop: int = int(RATE * HOP_SECONDS)
    start: int
    for start in range(0, len(y), hop):
        stream.push(y[start : start + hop])
        chord: str | None
        confidence: float | None
        _level: float
        chord, confidence, _level = recognize(model, stream.window, RATE)
        stable: str | None = stabilizer.push(chord, confidence)
        t: float = stream.total / RATE - 1 - onset
        if stable is not None and stable != truth and t > 0:
            false_chord = True
        if stable == truth and latency is None:
            latency = t
    return Replay(truth, latency, false_chord)


@pytest.fixture(scope="module")
def replays(model: ChordModel) -> list[Replay]:
    files: list[Path] = dataset_files()
    chosen: list[Path] = [
        f for c in model.meta.classes for f in [g for g in files if g.parent.name == c][::4]
    ]
    return [replay(model, f) for f in chosen]


@needs_dataset
def test_latency_median(replays: list[Replay]) -> None:
    latencies: list[float] = [r.latency for r in replays if r.latency is not None]
    assert np.median(latencies) <= 0.7
    assert np.percentile(latencies, 90) <= 1.0


@needs_dataset
def test_few_false_chords(replays: list[Replay]) -> None:
    assert sum(r.false_chord for r in replays) <= 2


@needs_dataset
def test_most_strums_recognised(replays: list[Replay]) -> None:
    # The rest are the ~4% the model gets wrong anyway, or strums starting late in the file.
    assert sum(r.latency is None for r in replays) <= 6
