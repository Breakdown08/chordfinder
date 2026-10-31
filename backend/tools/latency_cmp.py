"""Strum -> stable chord: the old scheme (every window shown, 250 ms) vs smoothing, on the same replay.

uv run python tools/latency_cmp.py            # ONNX
uv run --group export python tools/latency_cmp.py --keras   # the thesis Keras weights, same features

The replay is tests/test_latency.py's: every 4th test file per class (80), a second of silence, then
2 s of the recording, fed through Stream in hop-sized frames. Each window is recognized once per hop;
the schemes only differ in how Stabilizer votes over those answers. Rows of the "before -> target"
table in the Kadath note (2026-10-31): old scheme 0.31 s median, a false chord in 53/80; now 0.61 s, 1/80.
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from chordfinder import features
from chordfinder.audio import Audio
from chordfinder.evaluate import dataset_files, load
from chordfinder.model import MODELS, ChordModel
from chordfinder.stream import Stabilizer, Stream, recognize

RATE: int = 48000
ONSET_LEVEL: float = 0.05
LISTEN_SECONDS: float = 2.0
# (hop seconds, consecutive windows that must agree, min confidence, label)
SCHEMES: list[tuple[float, int, float, str]] = [
    (0.125, 1, 0.0, "no smoothing"),
    (0.125, 2, 0.6, "smoothing, 2 windows"),
    (0.125, 2, 0.8, "smoothing, 2 windows"),
    (0.125, 3, 0.6, "smoothing (server)"),
    (0.25, 1, 0.0, "no smoothing (old)"),
    (0.25, 2, 0.6, "smoothing"),
]


class KerasChordModel(ChordModel):
    """Same features and classes, inference by Keras on the thesis weights (export group only)."""

    def __init__(self, models: Path = MODELS) -> None:
        import keras

        super().__init__(models)
        self.keras: keras.Model = keras.saving.load_model(models / "chord-cnn.keras")

    def probabilities(self, audio_16k: Audio) -> npt.NDArray[np.float32]:
        mel: Audio = features.log_mel(features.window(audio_16k, self.meta), self.meta)
        out: npt.NDArray[np.float32] = np.asarray(self.keras(mel[np.newaxis, ..., np.newaxis])[0], np.float32)
        return out


@dataclass
class Answer:
    t: float  # seconds after the strum onset
    chord: str | None
    confidence: float | None


@dataclass
class Trace:
    truth: str
    answers: list[Answer]


def trace(model: ChordModel, path: Path, hop_seconds: float) -> Trace:
    x: Audio = load(path, RATE)
    onset: float = int(np.argmax(np.abs(x) > ONSET_LEVEL)) / RATE
    y: Audio = np.concatenate((np.zeros(RATE, np.float32), x))[: int(RATE * (1 + LISTEN_SECONDS))]
    stream: Stream = Stream(RATE)
    hop: int = int(RATE * hop_seconds)
    answers: list[Answer] = []
    start: int
    for start in range(0, len(y), hop):
        stream.push(y[start : start + hop])
        chord: str | None
        confidence: float | None
        _level: float
        chord, confidence, _level = recognize(model, stream.window, RATE)
        answers.append(Answer(stream.total / RATE - 1 - onset, chord, confidence))
    return Trace(path.parent.name, answers)


def score(traces: list[Trace], agree: int, min_confidence: float) -> str:
    latencies: list[float] = []
    false_chords: int = 0
    missed: int = 0
    tr: Trace
    for tr in traces:
        stabilizer: Stabilizer = Stabilizer(agree, min_confidence)
        latency: float | None = None
        false_chord: bool = False
        a: Answer
        for a in tr.answers:
            stable: str | None = stabilizer.push(a.chord, a.confidence)
            if stable is not None and stable != tr.truth and a.t > 0:
                false_chord = True
            if stable == tr.truth and latency is None:
                latency = a.t
        false_chords += false_chord
        if latency is None:
            missed += 1
        else:
            latencies.append(latency)
    median: float = float(np.median(latencies))
    p90: float = float(np.percentile(latencies, 90))
    return f"{median:.2f} | {p90:.2f} | {false_chords}/{len(traces)} | {missed}"


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keras", action="store_true", help="thesis Keras weights (needs --group export)")
    keras: bool = parser.parse_args().keras
    t0: float = time.perf_counter()
    model: ChordModel = KerasChordModel() if keras else ChordModel()
    files: list[Path] = dataset_files()
    chosen: list[Path] = [
        f for c in model.meta.classes for f in [g for g in files if g.parent.name == c][::4]
    ]
    name: str = "keras" if keras else "onnx"
    print("| model | hop | scheme (agree, min conf) | median, s | p90, s | false chord | missed |")
    print("|---|---|---|---|---|---|---|")
    cache: dict[float, list[Trace]] = {}
    hop: float
    agree: int
    min_confidence: float
    label: str
    for hop, agree, min_confidence, label in SCHEMES:
        if hop not in cache:
            cache[hop] = [trace(model, f, hop) for f in chosen]
        row: str = score(cache[hop], agree, min_confidence)
        print(f"| {name} | {int(hop * 1000)} ms | {label} ({agree}, {min_confidence}) | {row} |")
    print(f"\n{len(chosen)} files, {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
