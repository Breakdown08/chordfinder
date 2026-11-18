"""Model split into stages, recognize_timed and LagMeter (checks 1-6 of the load task)."""

import math

import numpy as np

from chordfinder import audio
from chordfinder.audio import Audio
from chordfinder.model import ChordModel
from chordfinder.stream import Heard, LagMeter, Timings, recognize, recognize_timed
from tests.conftest import sine

RATE: int = 48000
FRAME: int = 6000


def test_probabilities_is_infer_of_features(model: ChordModel) -> None:
    x: Audio = sine(440, 1.0, 16000)
    batch: Audio = model.features(x)
    frames: int = 1 + model.meta.sr // model.meta.hop_length
    assert batch.shape == (1, model.meta.n_mels, frames, 1)
    assert batch.dtype == np.float32
    assert np.array_equal(model.probabilities(x), model.infer(batch))


def test_recognize_timed_answers_like_recognize(model: ChordModel) -> None:
    y: Audio = sine(440, 1.0, RATE)
    heard: Heard
    timings: Timings | None
    heard, timings = recognize_timed(model, y, RATE)
    assert heard == recognize(model, y, RATE)
    assert heard.chord is not None
    assert timings is not None
    t: float
    for t in timings:
        assert math.isfinite(t) and t >= 0.0


def test_recognize_timed_on_silence_runs_no_stage(model: ChordModel) -> None:
    heard: Heard
    timings: Timings | None
    heard, timings = recognize_timed(model, np.zeros(RATE, np.float32), RATE)
    assert heard == Heard(None, None, 0.0, None)
    assert timings is None
    assert audio.SILENCE_RMS > 0


def lags(arrivals: list[float]) -> list[float]:
    meter: LagMeter = LagMeter(RATE)
    out: list[float] = []
    t: float
    for t in arrivals:
        out.append(meter.push(t, FRAME))
    return out


def test_lag_of_a_steady_stream_is_zero() -> None:
    result: list[float] = lags([100 + k * 0.125 for k in range(12)])
    assert len(result) == 12
    assert all(abs(v) < 1e-9 for v in result)


def test_lag_shows_a_stall_and_recovers() -> None:
    arrivals: list[float] = [100 + k * 0.125 + (0.3 if 3 <= k <= 5 else 0.0) for k in range(10)]
    result: list[float] = lags(arrivals)
    k: int
    for k in range(10):
        want: float = 0.3 if 3 <= k <= 5 else 0.0
        assert abs(result[k] - want) < 1e-9, k
    assert min(result) >= 0.0


def test_lag_of_a_late_first_frame_does_not_go_negative() -> None:
    arrivals: list[float] = [100 + k * 0.125 + (0.2 if k == 0 else 0.0) for k in range(10)]
    result: list[float] = lags(arrivals)
    assert result[0] == 0.0
    assert all(0.0 <= v <= 1e-9 for v in result[1:])
