"""The chord model at runtime: ONNX Runtime on CPU, no TensorFlow."""

from pathlib import Path

import numpy as np

from chordfinder import features
from chordfinder.features import Meta

MODELS = Path(__file__).resolve().parents[2] / "models"


class ChordModel:
    def __init__(self, models: Path = MODELS):
        import onnxruntime as ort

        self.meta = Meta.load(models / "meta.json")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1  # one window is tiny; threads cost more than they save
        self.session = ort.InferenceSession(
            str(models / "chord-cnn.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self.input = self.session.get_inputs()[0].name

    def probabilities(self, audio_16k: np.ndarray) -> np.ndarray:
        mel = features.log_mel(features.window(audio_16k, self.meta), self.meta)
        # Keras added the channel axis implicitly; the ONNX graph wants it spelled out: (1, mels, frames, 1)
        return self.session.run(None, {self.input: mel[np.newaxis, ..., np.newaxis]})[0][0]

    def predict(self, audio_16k: np.ndarray) -> tuple[str, float]:
        p = self.probabilities(audio_16k)
        i = int(np.argmax(p))
        return self.meta.classes[i], float(p[i])
