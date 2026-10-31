"""Stream buffer and the vote over consecutive windows."""

import numpy as np

from chordfinder.stream import Stabilizer, Stream


def test_stream_keeps_last_second() -> None:
    s: Stream = Stream(100)
    i: int
    for i in range(5):
        s.push(np.full(30, i, np.float32))
    assert s.total == 150
    assert len(s.window) == 100
    assert s.window[-1] == 4 and s.window[0] == 1  # 100 newest of 150: the first 50 dropped


def test_stabilizer_needs_agreement() -> None:
    v: Stabilizer = Stabilizer(agree=2, min_confidence=0.6)
    assert v.push("C", 0.9) is None  # one window is not enough
    assert v.push("C", 0.9) == "C"
    assert v.push("Em", 0.9) is None  # C/Em confusion breaks the streak
    assert v.push("Em", 0.9) == "Em"


def test_stabilizer_ignores_unsure_and_silence() -> None:
    v: Stabilizer = Stabilizer(agree=2, min_confidence=0.6)
    assert v.push("C", 0.9) is None
    assert v.push("C", 0.4) is None
    assert v.push(None, None) is None
    assert v.push(None, None) is None  # agreeing on silence is not a chord
