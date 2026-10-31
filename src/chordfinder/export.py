"""Keras -> ONNX (needs the export group: `uv run --group export chordfinder export`)."""

from pathlib import Path


def export(models: Path) -> Path:
    import keras
    import tensorflow as tf
    import tf2onnx

    model = keras.saving.load_model(models / "chord-cnn.keras")
    spec = (tf.TensorSpec((None, *model.input_shape[1:]), tf.float32, name="mel"),)
    out = models / "chord-cnn.onnx"
    tf2onnx.convert.from_function(
        tf.function(lambda mel: model(mel, training=False)),
        input_signature=spec,
        opset=17,
        output_path=str(out),
    )
    return out
