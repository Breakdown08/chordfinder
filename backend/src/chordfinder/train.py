"""Training the chord model from scratch, with the thesis recipe (Keras, CPU).

Data preparation (recording groups, validation split, windows) needs no TensorFlow and is tested
everywhere; the Keras parts import TensorFlow inside the functions, as export.py does:

    uv run --group train chordfinder train

The recipe is the thesis': envelope mask, whole one-second windows, model_Conv2D, Adam 1e-3, batch 32,
30 epochs. What differs: a seed with deterministic ops, and a validation split of whole guitars.
"""

import json
import os
import platform
import re
import time
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from chordfinder import evaluate, features
from chordfinder.audio import Audio
from chordfinder.dataset import content_digest
from chordfinder.features import Meta
from chordfinder.model import MODELS

if TYPE_CHECKING:
    import keras

# Two spellings in the dataset that name the same guitar as another folder's files.
GROUP_ALIASES: dict[str, str] = {"acusticbg_jo": "acusticgb_jo", "classiccleangb": "classiccleangb_jo"}


def recording_group(path: Path) -> str:
    """One guitar = one group, across all 8 chord folders (a guitar plays 4 files per chord)."""
    prefix: str = f"{path.parent.name}_"
    name: str = path.stem.removeprefix(prefix)
    name = re.sub(r"[\s_.-]*\d+[\s.]*$", "", name)
    name = re.sub(r"\s+", "", name).lower().replace("acustiic", "acustic")
    return GROUP_ALIASES.get(name, name)


def split_groups(files: list[Path], val_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    """(train, val): whole guitars go to val, so the validation measures "a guitar it has not heard"."""
    if not 0 <= val_fraction < 1:
        raise ValueError(f"val_fraction {val_fraction} is not in [0, 1)")
    if val_fraction == 0:
        return sorted(files), []
    names: list[str] = sorted({recording_group(f) for f in files})
    count: int = round(val_fraction * len(names))
    picked: npt.NDArray[np.intp] = np.random.default_rng(seed).choice(len(names), size=count, replace=False)
    val_groups: set[str] = {names[int(i)] for i in picked}
    train_files: list[Path] = sorted(f for f in files if recording_group(f) not in val_groups)
    val_files: list[Path] = sorted(f for f in files if recording_group(f) in val_groups)
    return train_files, val_files


def chunks(y: Audio, meta: Meta) -> list[Audio]:
    """Thesis training windows: the loud part cut into whole seconds, the tail dropped."""
    loud: Audio = y[features.envelope(y, meta.sr, meta.threshold)]
    return [loud[i * meta.sr : (i + 1) * meta.sr] for i in range(len(loud) // meta.sr)]


def windows(
    files: list[Path], meta: Meta, load: Callable[[Path, int], Audio] = evaluate.load
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    """x: (n, n_mels, frames) log-mel of every window of every file; y: (n,) class index."""
    mels: list[Audio] = []
    labels: list[int] = []
    f: Path
    for f in files:
        if f.parent.name not in meta.classes:
            raise ValueError(f"{f}: folder {f.parent.name!r} is not one of {meta.classes}")
        label: int = meta.classes.index(f.parent.name)
        c: Audio
        for c in chunks(load(f, meta.sr), meta):
            mels.append(features.log_mel(c, meta))
            labels.append(label)
    frames: int = meta.sr // meta.hop_length + 1
    x: npt.NDArray[np.float32] = (
        np.stack(mels).astype(np.float32) if mels else np.zeros((0, meta.n_mels, frames), np.float32)
    )
    y: npt.NDArray[np.int64] = np.asarray(labels, np.int64)
    return x, y


class TrainConfig(BaseModel, frozen=True):
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    seed: int = 0
    val_fraction: float = 0.15


class TrainResult(BaseModel):
    out: Path
    train_windows: int
    val_windows: int
    val_accuracy: float | None
    onnx_max_abs_diff: float
    seconds: float


def build_model(meta: Meta) -> "keras.Model":
    """The thesis model_Conv2D, compiled with Adam and the sparse form of its cross-entropy."""
    import keras
    from keras import layers
    from keras.regularizers import l2

    frames: int = meta.sr // meta.hop_length + 1
    inputs: keras.KerasTensor = keras.Input(shape=(meta.n_mels, frames, 1))
    x: keras.KerasTensor = layers.LayerNormalization(axis=1, name="batch_norm")(inputs)
    x = layers.Conv2D(8, kernel_size=(7, 7), activation="tanh", padding="same", name="conv2d_tanh")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding="same", name="max_pool_2d_1")(x)
    x = layers.Conv2D(16, kernel_size=(5, 5), activation="relu", padding="same", name="conv2d_relu_1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding="same", name="max_pool_2d_2")(x)
    x = layers.Conv2D(16, kernel_size=(3, 3), activation="relu", padding="same", name="conv2d_relu_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding="same", name="max_pool_2d_3")(x)
    x = layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same", name="conv2d_relu_3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding="same", name="max_pool_2d_4")(x)
    x = layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same", name="conv2d_relu_4")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dropout(rate=0.2, name="dropout")(x)
    x = layers.Dense(64, activation="relu", activity_regularizer=l2(0.001), name="dense")(x)
    outputs: keras.KerasTensor = layers.Dense(len(meta.classes), activation="softmax", name="softmax")(x)
    model: keras.Model = keras.Model(inputs=inputs, outputs=outputs, name="2d_convolution")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def meta_json(meta: Meta) -> str:
    """models/meta.json as it is written there; the original file when it describes this meta."""
    source: Path = MODELS / "meta.json"
    if source.is_file() and Meta.load(source) == meta:
        return source.read_text()
    fields: dict[str, object] = {
        "sr": meta.sr,
        "dt": 1.0,
        "n_classes": len(meta.classes),
        "n_mels": meta.n_mels,
        "hop_length": meta.hop_length,
        "n_fft": meta.n_fft,
        "threshold": meta.threshold,
        "frames": meta.sr // meta.hop_length + 1,
        "classes": list(meta.classes),
    }
    return json.dumps(fields, indent=4)


def version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unknown"


def train(dataset: Path, out: Path, config: TrainConfig, meta: Meta) -> TrainResult:
    """Train on dataset/Training only, write the Keras model, the ONNX export, meta and history to out."""
    import keras
    import onnxruntime as ort
    import tensorflow as tf

    from chordfinder.export import export

    started: float = time.perf_counter()
    keras.utils.set_random_seed(config.seed)
    tf.config.experimental.enable_op_determinism()

    files: list[Path] = sorted((dataset / "Training").glob("*/*.wav"))
    train_files: list[Path]
    val_files: list[Path]
    train_files, val_files = split_groups(files, config.val_fraction, config.seed)
    x_train: npt.NDArray[np.float32]
    y_train: npt.NDArray[np.int64]
    x_val: npt.NDArray[np.float32]
    y_val: npt.NDArray[np.int64]
    x_train, y_train = windows(train_files, meta)
    x_val, y_val = windows(val_files, meta)
    x_train = x_train[..., np.newaxis]
    x_val = x_val[..., np.newaxis]

    model: keras.Model = build_model(meta)
    model.optimizer.learning_rate = config.learning_rate
    fit: keras.callbacks.History = model.fit(
        x_train,
        y_train,
        batch_size=config.batch_size,
        epochs=config.epochs,
        shuffle=True,
        validation_data=(x_val, y_val) if len(val_files) else None,
        verbose=2,
    )

    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "chord-cnn.keras")
    (out / "meta.json").write_text(meta_json(meta))
    onnx: Path = export(out)

    probe: npt.NDArray[np.float32] = x_train[:256]
    session: ort.InferenceSession = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    got: npt.NDArray[np.float32] = np.asarray(
        session.run(None, {session.get_inputs()[0].name: probe})[0], np.float32
    )
    want: npt.NDArray[np.float32] = np.asarray(model(probe, training=False))
    max_diff: float = float(np.max(np.abs(got - want)))

    metrics: dict[str, list[float]] = fit.history
    epochs: list[dict[str, float]] = []
    i: int
    for i in range(config.epochs):
        row: dict[str, float] = {"epoch": i + 1}
        key: str
        for key in ("loss", "accuracy", "val_loss", "val_accuracy"):
            if key in metrics:
                row[key] = float(metrics[key][i])
        epochs.append(row)
    seconds: float = time.perf_counter() - started
    val_accuracy: float | None = epochs[-1].get("val_accuracy") if len(val_files) else None

    history: dict[str, object] = {
        "config": config.model_dump(),
        "dataset_content_sha256": content_digest(dataset)[0],
        "val_groups": sorted({recording_group(f) for f in val_files}),
        "train_files": len(train_files),
        "train_windows": len(y_train),
        "val_files": len(val_files),
        "val_windows": len(y_val),
        "epochs": epochs,
        "onnx_max_abs_diff": max_diff,
        "seconds": seconds,
        "versions": {
            "python": platform.python_version(),
            "tensorflow": version("tensorflow"),
            "keras": version("keras"),
            "tf2onnx": version("tf2onnx"),
        },
        "machine": f"{platform.platform()} / {os.cpu_count()} CPUs",
    }
    (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return TrainResult(
        out=out,
        train_windows=len(y_train),
        val_windows=len(y_val),
        val_accuracy=val_accuracy,
        onnx_max_abs_diff=max_diff,
        seconds=seconds,
    )
