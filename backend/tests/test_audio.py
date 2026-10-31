"""Wire format and resampling: the server must hear the pitch the client recorded."""

import numpy as np
import pytest

from chordfinder import audio
from chordfinder.audio import Audio
from tests.conftest import sine


def dominant_hz(y: Audio, sr: float) -> float:
    spectrum: np.typing.NDArray[np.float64] = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    return float(np.argmax(spectrum) * sr / len(y))


@pytest.mark.parametrize("rate", [44100, 48000, 16000])
def test_resample_keeps_pitch(rate: int) -> None:
    y: Audio = audio.resample(sine(440, 1.0, rate), rate)
    assert len(y) == audio.MODEL_RATE
    assert dominant_hz(y, audio.MODEL_RATE) == pytest.approx(440, abs=2)


def test_wrong_rate_shifts_pitch() -> None:
    # The thesis server assumed 44.1 kHz: a 48 kHz phone recording came out ~8% flat.
    y: Audio = audio.resample(sine(440, 1.0, 48000), 44100)
    assert dominant_hz(y, audio.MODEL_RATE) == pytest.approx(440 * 44100 / 48000, abs=2)


def test_decode_frame_roundtrip() -> None:
    y: Audio = sine(440, 0.25, 48000)
    assert np.array_equal(audio.decode_frame(y.astype("<f4").tobytes(), 48000), y)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"\x00\x00\x00", "multiple of 4"),
        (np.zeros(48001, "<f4").tobytes(), "longer than"),
        (np.array([np.nan], "<f4").tobytes(), "NaN"),
    ],
)
def test_decode_frame_rejects(data: bytes, message: str) -> None:
    with pytest.raises(audio.ProtocolError, match=message):
        audio.decode_frame(data, 48000)


@pytest.mark.parametrize(
    ("text", "message"),
    [("nope", "not JSON"), ('{"rate": 48000}', "sample_rate"), ('{"sample_rate": 1000}', "outside")],
)
def test_parse_header_rejects(text: str, message: str) -> None:
    with pytest.raises(audio.ProtocolError, match=message):
        audio.parse_header(text)


def test_parse_header() -> None:
    assert audio.parse_header('{"sample_rate": 48000, "build": "web"}') == 48000.0
