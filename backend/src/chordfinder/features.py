"""Preprocessing of the thesis model, kept bit-for-bit: envelope trim, 1 s window, log-mel.

The model was trained on exactly these features, so any change here is a change of the model.
The parity test (tests/test_parity.py) guards it against baseline/old-model.json.
"""

from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel
from scipy.ndimage import maximum_filter1d

from chordfinder.audio import Audio


class Meta(BaseModel, frozen=True, extra="ignore"):
    """models/meta.json: feature parameters and class order the model was trained with."""

    sr: int
    n_mels: int
    n_fft: int
    hop_length: int
    threshold: float
    classes: tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text())


def envelope(y: Audio, sr: int, threshold: float) -> npt.NDArray[np.bool_]:
    """Mask of samples whose centred 50 ms peak exceeds the threshold.

    Same as the thesis' pandas rolling(window=sr/20, center=True, min_periods=1).max().
    """
    peak: Audio = maximum_filter1d(np.abs(y), size=int(sr / 20), mode="constant", cval=0.0)
    mask: npt.NDArray[np.bool_] = peak > threshold
    return mask


def window(y: Audio, meta: Meta) -> Audio:
    """Drop quiet parts, then cut or zero-pad to one second."""
    loud: Audio = y[envelope(y, meta.sr, meta.threshold)]
    if len(loud) >= meta.sr:
        return loud[: meta.sr]
    return np.pad(loud, (0, meta.sr - len(loud)))


def log_mel(y: Audio, meta: Meta) -> Audio:
    """(n_mels, frames) log-mel spectrogram of a 16 kHz window."""
    import librosa

    mel: Audio = librosa.feature.melspectrogram(
        y=y, sr=meta.sr, n_fft=meta.n_fft, hop_length=meta.hop_length, n_mels=meta.n_mels
    )
    db: Audio = np.asarray(librosa.power_to_db(mel), np.float32)
    return db
