"""Raw audio from clients: wire format, loudness, resampling to the model rate."""

import json

import numpy as np
import numpy.typing as npt

type Audio = npt.NDArray[np.float32]

MODEL_RATE: int = 16000
MIN_RATE: int = 8000
MAX_RATE: int = 192000
MAX_FRAME_SECONDS: float = 1.0
# Below this the window is treated as silence: the model has no "none" class and would name a chord.
SILENCE_RMS: float = 0.01


class ProtocolError(ValueError):
    pass


def check_rate(sample_rate: float) -> float:
    if not MIN_RATE <= sample_rate <= MAX_RATE:
        raise ProtocolError(f"sample_rate {sample_rate} outside {MIN_RATE}..{MAX_RATE}")
    return sample_rate


def parse_header(text: str) -> float:
    """First WebSocket frame: {"sample_rate": 48000} -> the client's sample rate."""
    try:
        header: object = json.loads(text)
    except json.JSONDecodeError as e:
        raise ProtocolError(f"header is not JSON: {e}") from e
    if not isinstance(header, dict) or not isinstance(header.get("sample_rate"), int | float):
        raise ProtocolError('header must be {"sample_rate": <number>}')
    rate: float = float(header["sample_rate"])
    return check_rate(rate)


def decode_frame(data: bytes, sample_rate: float) -> Audio:
    """Binary frame -> mono float32 samples (little-endian), at most MAX_FRAME_SECONDS long."""
    if len(data) % 4:
        raise ProtocolError(f"frame length {len(data)} is not a multiple of 4")
    if len(data) // 4 > sample_rate * MAX_FRAME_SECONDS:
        raise ProtocolError(f"frame longer than {MAX_FRAME_SECONDS} s")
    samples: Audio = np.frombuffer(data, dtype="<f4").astype(np.float32)
    if not np.isfinite(samples).all():
        raise ProtocolError("frame contains NaN or infinity")
    return samples


def rms(y: Audio) -> float:
    return float(np.sqrt(np.mean(np.square(y, dtype=np.float64)))) if len(y) else 0.0


def resample(y: Audio, sample_rate: float) -> Audio:
    """Any client rate -> MODEL_RATE. The thesis server assumed 44.1 kHz; Android usually records 48."""
    import librosa

    if sample_rate == MODEL_RATE:
        return y
    out: Audio = np.asarray(librosa.resample(y=y, orig_sr=sample_rate, target_sr=MODEL_RATE), np.float32)
    return out
