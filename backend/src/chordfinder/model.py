"""The chord model at runtime: ONNX Runtime on CPU, no TensorFlow."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from chordfinder import features
from chordfinder.audio import Audio
from chordfinder.features import Meta

# Weights are not in git: see the README for where to download them.
MODELS: Path = Path(__file__).resolve().parents[2] / "models"
ONNX: str = "chord-cnn.onnx"


class ChordModel:
    def __init__(self, models: Path = MODELS) -> None:
        self.meta: Meta = Meta.load(models / "meta.json")
        opts: ort.SessionOptions = ort.SessionOptions()
        opts.intra_op_num_threads = 1  # one window is tiny; threads cost more than they save
        self.session: ort.InferenceSession = ort.InferenceSession(
            str(models / ONNX), opts, providers=["CPUExecutionProvider"]
        )
        self.input: str = self.session.get_inputs()[0].name

    def probabilities(self, audio_16k: Audio) -> npt.NDArray[np.float32]:
        mel: Audio = features.log_mel(features.window(audio_16k, self.meta), self.meta)
        # Keras added the channel axis implicitly; the ONNX graph wants it spelled out: (1, mels, frames, 1)
        batch: Audio = mel[np.newaxis, ..., np.newaxis]
        out: npt.NDArray[np.float32] = np.asarray(
            self.session.run(None, {self.input: batch})[0][0], np.float32
        )
        return out

    def predict(self, audio_16k: Audio) -> tuple[str, float]:
        p: npt.NDArray[np.float32] = self.probabilities(audio_16k)
        i: int = int(np.argmax(p))
        return self.meta.classes[i], float(p[i])
