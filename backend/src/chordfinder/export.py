"""Keras -> ONNX (needs the export group: `uv run --group export chordfinder export`)."""

from pathlib import Path

from chordfinder.model import ONNX


def export(models: Path) -> Path:
    import keras
    import tensorflow as tf
    import tf2onnx

    model: keras.Model = keras.saving.load_model(models / "chord-cnn.keras")
    spec: tuple[tf.TensorSpec] = (tf.TensorSpec((None, *model.input_shape[1:]), tf.float32, name="mel"),)

    def infer(mel: tf.Tensor) -> tf.Tensor:
        result: tf.Tensor = model(mel, training=False)
        return result

    traced: object = tf.function(infer)
    out: Path = models / ONNX
    tf2onnx.convert.from_function(traced, input_signature=spec, opset=17, output_path=str(out))
    return out
