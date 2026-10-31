"""Preprocessing of the diploma model, kept bit-for-bit: envelope trim, 1 s window, log-mel.

The model was trained on exactly these features, so any change here is a change of the model.
The parity test (tests/test_parity.py) guards it against baseline/old-model.json.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter1d


@dataclass(frozen=True)
class Meta:
    sr: int
    n_mels: int
    n_fft: int
    hop_length: int
    threshold: float
    classes: tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> "Meta":
        m = json.loads(path.read_text())
        return cls(m["sr"], m["n_mels"], m["n_fft"], m["hop_length"], m["threshold"], tuple(m["classes"]))


def envelope(y: np.ndarray, sr: int, threshold: float) -> np.ndarray:
    """Mask of samples whose centred 50 ms peak exceeds the threshold.

    Same as the diploma's pandas rolling(window=sr/20, center=True, min_periods=1).max().
    """
    peak = maximum_filter1d(np.abs(y), size=int(sr / 20), mode="constant", cval=0.0)
    return peak > threshold


def window(y: np.ndarray, meta: Meta) -> np.ndarray:
    """Drop quiet parts, then cut or zero-pad to one second."""
    y = y[envelope(y, meta.sr, meta.threshold)]
    if len(y) >= meta.sr:
        return y[: meta.sr]
    return np.pad(y, (0, meta.sr - len(y)))


def log_mel(y: np.ndarray, meta: Meta) -> np.ndarray:
    """(n_mels, frames) log-mel spectrogram of a 16 kHz window."""
    import librosa

    mel = librosa.feature.melspectrogram(
        y=y, sr=meta.sr, n_fft=meta.n_fft, hop_length=meta.hop_length, n_mels=meta.n_mels
    )
    return librosa.power_to_db(mel).astype(np.float32)
