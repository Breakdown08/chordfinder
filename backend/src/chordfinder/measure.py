"""Measuring session: a labelled take per chord, kept as raw audio so any model can be scored on it.

The server only counts strums and writes the files; tools/takes.py scores them offline. One folder:
<chord>.wav (16-bit, the client's rate) and session.json (rate, strum onsets in samples per take).
"""

import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from chordfinder import audio
from chordfinder.audio import Audio
from chordfinder.model import ChordModel
from chordfinder.stream import HOP_SECONDS, Heard, Stabilizer, Stream, recognize

# A strum: a frame at least RISE times louder than the one before it and above the silence floor.
# A strum rings for seconds, so anything within MIN_GAP_SECONDS of the last one is the same strum.
RISE: float = 2.5
MIN_GAP_SECONDS: float = 1.0
SESSION_FILE: str = "session.json"


class Onsets:
    def __init__(self, sample_rate: float) -> None:
        self.gap: int = int(sample_rate * MIN_GAP_SECONDS)
        self.previous: float = 0.0
        self.last: int | None = None

    def push(self, frame: Audio, at: int) -> bool:
        """Is a strum starting in this frame, which begins `at` samples into the take?"""
        level: float = audio.rms(frame)
        rising: bool = level >= audio.SILENCE_RMS and level >= RISE * self.previous
        self.previous = level
        if not rising or (self.last is not None and at - self.last < self.gap):
            return False
        self.last = at
        return True


class Session:
    def __init__(self, root: Path, sample_rate: float) -> None:
        self.path: Path = root / time.strftime("%Y%m%d-%H%M%S")
        self.sample_rate: float = sample_rate
        self.takes: dict[str, list[int]] = {}
        self.chord: str | None = None
        self.frames: list[Audio] = []
        self.onsets: list[int] = []
        self.total: int = 0
        self.detector: Onsets = Onsets(sample_rate)

    def start(self, chord: str) -> None:
        self.chord = chord
        self.frames = []
        self.onsets = []
        self.total = 0
        self.detector = Onsets(self.sample_rate)

    def push(self, frame: Audio) -> None:
        if self.chord is None:
            return
        if self.detector.push(frame, self.total):
            self.onsets.append(self.total)
        self.frames.append(frame)
        self.total += len(frame)

    def stop(self) -> None:
        """Write the take; a repeated chord replaces its file. Blocking: run it off the event loop."""
        if self.chord is None:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        y: Audio = np.concatenate(self.frames) if self.frames else np.zeros(0, np.float32)
        pcm: Audio = np.clip(y, -1, 1) * 32767
        wavfile.write(self.path / f"{self.chord}.wav", round(self.sample_rate), pcm.astype(np.int16))
        self.takes[self.chord] = self.onsets
        meta: dict[str, object] = {"sample_rate": self.sample_rate, "takes": self.takes}
        (self.path / SESSION_FILE).write_text(json.dumps(meta, indent=2) + "\n")
        self.chord = None


@dataclass
class Window:
    at: int  # samples into the take, at the end of the window
    chord: str | None
    stable: str | None


@dataclass
class Strum:
    truth: str
    majority: str | None
    first_stable: str | None
    latency: float | None  # seconds from the onset to the first stable right chord
    votes: Counter[str | None]


def replay(model: ChordModel, y: Audio, rate: float, onsets: list[int]) -> list[Window]:
    """The take through the server's path. The vote restarts at each strum, or a repeat of the same
    chord would inherit the agreement of the last one and look instant."""
    stream: Stream = Stream(rate)
    stabilizer: Stabilizer = Stabilizer()
    hop: int = int(rate * HOP_SECONDS)
    windows: list[Window] = []
    start: int
    for start in range(0, len(y), hop):
        if any(start <= onset < start + hop for onset in onsets):
            stabilizer = Stabilizer()
        stream.push(y[start : start + hop])
        heard: Heard = recognize(model, stream.window, rate)
        windows.append(Window(stream.total, heard.chord, stabilizer.push(heard.chord, heard.confidence)))
    return windows


def score(truth: str, windows: list[Window], onsets: list[int], rate: float) -> list[Strum]:
    """Each strum owns the windows from its onset to the next one (or the end of the take)."""
    strums: list[Strum] = []
    ends: list[float] = [*onsets[1:], float("inf")]
    onset: int
    end: float
    for onset, end in zip(onsets, ends, strict=True):
        own: list[Window] = [w for w in windows if onset < w.at <= end]
        votes: Counter[str | None] = Counter(w.chord for w in own if w.chord is not None)
        shown: list[Window] = [w for w in own if w.stable is not None]
        right: Window | None = next((w for w in shown if w.stable == truth), None)
        strums.append(
            Strum(
                truth,
                votes.most_common(1)[0][0] if votes else None,
                shown[0].stable if shown else None,
                (right.at - onset) / rate if right else None,
                votes,
            )
        )
    return strums


def load(path: Path) -> Audio:
    pcm: Audio = wavfile.read(path)[1]
    y: Audio = np.asarray(pcm, np.float32) / 32767
    return y
