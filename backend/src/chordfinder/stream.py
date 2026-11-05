"""State of one microphone stream: the last second of audio and a vote over recent answers."""

from collections import deque
from typing import NamedTuple

import numpy as np

from chordfinder import audio
from chordfinder.audio import Audio
from chordfinder.model import ChordModel

# Client frame length = hop between answers. A chord counts once AGREE consecutive windows name it
# with at least MIN_CONFIDENCE: in the first ~0.3 s of a strum the window is mostly zero padding and
# the model answers confidently wrong (Bb, Bdim). Chosen on 80 test files (tests/test_latency.py):
# 125 ms x 3 gives median 0.61 s, p90 0.91 s from the strum, 1 file with a false chord;
# 250 ms x 2 was as safe but slower (0.64 / 1.03 s).
HOP_SECONDS: float = 0.125
AGREE: int = 3
MIN_CONFIDENCE: float = 0.6


class Stream:
    def __init__(self, sample_rate: float, seconds: float = 1.0) -> None:
        self.sample_rate: float = sample_rate
        self.size: int = int(sample_rate * seconds)
        self.window: Audio = np.zeros(0, np.float32)
        self.total: int = 0

    def push(self, chunk: Audio) -> None:
        self.window = np.concatenate((self.window, chunk))[-self.size :]
        self.total += len(chunk)


class Stabilizer:
    def __init__(self, agree: int = AGREE, min_confidence: float = MIN_CONFIDENCE) -> None:
        self.min_confidence: float = min_confidence
        self.recent: deque[str | None] = deque(maxlen=agree)

    def push(self, chord: str | None, confidence: float | None) -> str | None:
        """Raw answer of one window -> the chord all recent windows agree on, or None."""
        sure: bool = chord is not None and confidence is not None and confidence >= self.min_confidence
        self.recent.append(chord if sure else None)
        first: str | None = self.recent[0]
        agreed: bool = len(self.recent) == self.recent.maxlen and all(c == first for c in self.recent)
        return first if agreed else None


class Heard(NamedTuple):
    """Answer of one window; silence is no chord and no probabilities."""

    chord: str | None
    confidence: float | None
    rms: float
    probabilities: dict[str, float] | None


def recognize(model: ChordModel, y: Audio, sample_rate: float) -> Heard:
    """One window -> what the model heard in it. Blocking: run it off the event loop."""
    level: float = audio.rms(y)
    if level < audio.SILENCE_RMS:
        return Heard(None, None, level, None)
    p: Audio = model.probabilities(audio.resample(y, sample_rate))
    i: int = int(np.argmax(p))
    classes: tuple[str, ...] = model.meta.classes
    return Heard(classes[i], float(p[i]), level, {c: float(v) for c, v in zip(classes, p, strict=True)})
