"""Training data preparation (no TensorFlow) and the Keras training (skipped without keras).

The datasets here are tiny synthetic sines in tmp_path: never the real dataset, never the network.
"""

import json
import re
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from chordfinder import dataset, features, train
from chordfinder.audio import Audio
from chordfinder.features import Meta
from chordfinder.model import MODELS
from tests.synthetic import CLASSES, meta, run_cli, synthetic_dataset, tone, write_tree, write_wav
from tests.test_dataset import manifest_for, patch_manifest

# 12


@pytest.mark.parametrize(
    ("path", "group"),
    [
        ("Am/Am_AcusticPlug26_1.wav", "acusticplug26"),
        ("Bb/Bb_AcusticPlug26_4.wav", "acusticplug26"),
        ("Am/Am_AcustiicPlug10_2.wav", "acusticplug10"),
        ("Em/Em_acusticBG_JO_3.wav", "acusticgb_jo"),
        ("Dm/Dm_ClassicCleanGB_1.wav", "classiccleangb_jo"),
        ("Am/Am_classical_AfonsoLemos_-01.wav", "classical_afonsolemos"),
        ("Am/Am_Classic_Jo_1..wav", "classic_jo"),
        ("Am/Am_AcusticPlug29 _2.wav", "acusticplug29"),
        ("G/G_Retro GB_JO_3.wav", "retrogb_jo"),
    ],
)
def test_recording_group(path: str, group: str) -> None:
    assert train.recording_group(Path(path)) == group


# 13


def grouped_files(n_groups: int) -> list[Path]:
    """4 files per group in each of 8 chord folders; group names differ by their number."""
    files: list[Path] = []
    cls: str
    for cls in CLASSES:
        g: int
        for g in range(n_groups):
            k: int
            for k in range(1, 5):
                files.append(Path(f"data/Training/{cls}/{cls}_AcusticPlug{g}_{k}.wav"))
    return files


def groups_of(files: list[Path]) -> set[str]:
    return {train.recording_group(f) for f in files}


def test_split_keeps_groups_whole_and_complete() -> None:
    files: list[Path] = grouped_files(10)
    tr: list[Path]
    val: list[Path]
    tr, val = train.split_groups(files, 0.3, 0)
    assert groups_of(tr).isdisjoint(groups_of(val))
    assert sorted(tr + val) == sorted(files)
    assert len(groups_of(val)) == 3
    assert tr == sorted(tr)
    assert val == sorted(val)
    assert len(train.split_groups(files, 0.2, 0)[1]) == 2 * 32


def test_split_is_deterministic_per_seed() -> None:
    files: list[Path] = grouped_files(10)
    assert train.split_groups(files, 0.3, 7) == train.split_groups(list(reversed(files)), 0.3, 7)
    seen: set[frozenset[str]] = {frozenset(groups_of(train.split_groups(files, 0.3, s)[1])) for s in range(6)}
    assert len(seen) > 1


def test_split_without_validation() -> None:
    files: list[Path] = grouped_files(3)
    assert train.split_groups(list(reversed(files)), 0.0, 0) == (sorted(files), [])


@pytest.mark.parametrize("bad", [1.0, -0.1, 1.5])
def test_split_rejects_bad_fraction(bad: float) -> None:
    with pytest.raises(ValueError):
        train.split_groups(grouped_files(3), bad, 0)


# 14


def test_chunks_cut_whole_seconds_and_drop_the_tail() -> None:
    m: Meta = meta()
    y: Audio = tone("Am", 2.5)
    out: list[Audio] = train.chunks(y, m)
    assert len(out) == 2
    assert all(len(c) == m.sr for c in out)
    trimmed: Audio = y[features.envelope(y, m.sr, m.threshold)]
    assert np.array_equal(out[0], trimmed[: m.sr])
    assert np.array_equal(out[1], trimmed[m.sr : 2 * m.sr])


def test_chunks_of_a_short_signal() -> None:
    assert train.chunks(tone("Am", 0.6), meta()) == []


def test_chunks_cut_silence_out_first() -> None:
    m: Meta = meta()
    silence: Audio = np.zeros(m.sr, np.float32)
    y: Audio = np.concatenate([silence, tone("Am", 1.0), silence])
    out: list[Audio] = train.chunks(y, m)
    assert len(out) == 1
    trimmed: Audio = y[features.envelope(y, m.sr, m.threshold)]
    assert np.array_equal(out[0], trimmed[: m.sr])


# 15


def three_files(tmp_path: Path) -> list[Path]:
    files: list[Path] = [
        tmp_path / "Am" / "Am_PlugA_1.wav",
        tmp_path / "Bb" / "Bb_PlugA_1.wav",
        tmp_path / "Bb" / "Bb_PlugB_1.wav",
    ]
    write_wav(files[0], tone("Am", 2.2))
    write_wav(files[1], tone("Bb", 1.5))
    write_wav(files[2], tone("Bb", 3.1))
    return files


def test_windows_shape_labels_and_order(tmp_path: Path) -> None:
    m: Meta = meta()
    files: list[Path] = three_files(tmp_path)
    x: npt.NDArray[np.float32]
    y: npt.NDArray[np.int64]
    x, y = train.windows(files, m)
    assert x.shape == (2 + 1 + 3, 64, 63)
    assert x.dtype == np.float32
    assert y.dtype == np.int64
    assert y.tolist() == [CLASSES.index("Am")] * 2 + [CLASSES.index("Bb")] * 4
    first: Audio = train.chunks(tone("Am", 2.2), m)[0]
    assert np.allclose(x[0], features.log_mel(first, m), atol=1e-4)


def test_windows_follow_the_given_order(tmp_path: Path) -> None:
    files: list[Path] = three_files(tmp_path)
    y: npt.NDArray[np.int64] = train.windows(list(reversed(files)), meta())[1]
    assert y.tolist() == [CLASSES.index("Bb")] * 3 + [CLASSES.index("Bb")] + [CLASSES.index("Am")] * 2


def test_windows_use_the_injected_loader(tmp_path: Path) -> None:
    seen: list[Path] = []

    def load(path: Path, sr: int) -> Audio:
        seen.append(path)
        return tone("Dm", 1.2)

    files: list[Path] = [tmp_path / "Dm" / "Dm_PlugA_1.wav"]  # never written: the loader supplies the audio
    x: npt.NDArray[np.float32] = train.windows(files, meta(), load)[0]
    assert seen == files
    assert x.shape == (1, 64, 63)


def test_windows_reject_an_unknown_class_folder(tmp_path: Path) -> None:
    bad: Path = tmp_path / "Xm" / "Xm_PlugA_1.wav"
    write_wav(bad, tone("Am", 1.5))
    with pytest.raises(ValueError):
        train.windows([bad], meta())


# 16


def test_config_repeats_the_thesis_recipe() -> None:
    c: train.TrainConfig = train.TrainConfig()
    assert c.epochs == 30
    assert c.batch_size == 32
    assert c.learning_rate == 1e-3
    assert c.seed == 0
    assert c.val_fraction == 0.15
    with pytest.raises(ValueError):
        c.__setattr__("epochs", 1)  # frozen


# 17, 18


def installed_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root: Path = tmp_path / "ds"
    synthetic_dataset(root, groups=1, takes=1, seconds=1.2)
    patch_manifest(monkeypatch, manifest_for(root))
    return root


def has_keras() -> bool:
    try:
        import keras  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(has_keras(), reason="TensorFlow is installed")
def test_cli_train_without_tensorflow_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root: Path = installed_dataset(tmp_path, monkeypatch)
    assert run_cli(["train", "--dataset", str(root), "--out", str(tmp_path / "out")]) == 2
    assert "--group train" in capsys.readouterr().err
    assert not (tmp_path / "out" / "chord-cnn.onnx").exists()


def test_cli_train_refuses_unverified_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root: Path = installed_dataset(tmp_path, monkeypatch)
    foreign: Path = tmp_path / "foreign"
    write_tree(foreign, {"Training/Am/x.wav": b"someone else's data", "Test/Am/y.wav": b"more"})
    assert dataset.check(foreign, manifest_for(root)) is not None
    out: Path = tmp_path / "out"
    assert run_cli(["train", "--dataset", str(foreign), "--out", str(out)]) == 1
    assert capsys.readouterr().err.strip() != ""
    assert not out.exists()


# 23-26: need TensorFlow, so they run only in the train group (the CI job `train`)

THESIS_LAYERS: list[str] = [
    "batch_norm",
    "conv2d_tanh",
    "max_pool_2d_1",
    "conv2d_relu_1",
    "max_pool_2d_2",
    "conv2d_relu_2",
    "max_pool_2d_3",
    "conv2d_relu_3",
    "max_pool_2d_4",
    "conv2d_relu_4",
    "flatten",
    "dropout",
    "dense",
    "softmax",
]


def layer_info(model: object) -> list[tuple[str, tuple[int | None, ...]]]:
    import keras

    assert isinstance(model, keras.Model)
    info: list[tuple[str, tuple[int | None, ...]]] = []
    layer: keras.layers.Layer
    for layer in model.layers:
        if not isinstance(layer, keras.layers.InputLayer):
            info.append((layer.name, tuple(layer.output.shape)))
    return info


def test_build_model_is_the_thesis_model() -> None:
    keras: ModuleType = pytest.importorskip("keras")
    model: Any = train.build_model(meta())
    info: list[tuple[str, tuple[int | None, ...]]] = layer_info(model)
    assert [name for name, _ in info] == THESIS_LAYERS
    assert info[-1][1] == (None, 8)
    original: Path = MODELS / "chord-cnn.keras"
    if original.is_file():
        thesis: Any = keras.saving.load_model(original)
        assert info == layer_info(thesis)
        assert model.count_params() == thesis.count_params()


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, train.TrainResult]:
    pytest.importorskip("keras")
    root: Path = tmp_path_factory.mktemp("ds")
    synthetic_dataset(root, groups=2, takes=2, seconds=2.0)
    out: Path = tmp_path_factory.mktemp("out") / "run"
    config: train.TrainConfig = train.TrainConfig(
        epochs=2, batch_size=16, learning_rate=1e-3, seed=0, val_fraction=0.5
    )
    result: train.TrainResult = train.train(root, out, config, Meta.load(MODELS / "meta.json"))
    return root, out, result


def test_train_writes_the_four_files(trained: tuple[Path, Path, train.TrainResult]) -> None:
    root: Path = trained[0]
    out: Path = trained[1]
    result: train.TrainResult = trained[2]
    assert result.out == out
    names: set[str] = {p.name for p in out.iterdir()}
    assert {"chord-cnn.keras", "chord-cnn.onnx", "meta.json", "history.json"} <= names
    assert (out / "meta.json").read_bytes() == (MODELS / "meta.json").read_bytes()
    assert result.onnx_max_abs_diff < 1e-5
    assert result.train_windows == 32  # 8 classes x 1 group x 2 files x 2 windows
    assert result.val_windows == 32
    assert result.val_accuracy is not None
    history: dict[str, object] = json.loads((out / "history.json").read_text())
    expected: set[str] = {
        "config",
        "dataset_content_sha256",
        "val_groups",
        "train_files",
        "train_windows",
        "val_files",
        "val_windows",
        "epochs",
        "onnx_max_abs_diff",
        "seconds",
        "versions",
        "machine",
    }
    assert expected <= set(history)
    assert history["dataset_content_sha256"] == dataset.content_digest(root)[0]
    assert history["train_files"] == 16
    assert history["val_files"] == 16
    assert history["val_groups"] in (["plugb"], ["pluga"])
    epochs: list[dict[str, float]] = json.loads((out / "history.json").read_text())["epochs"]
    assert [e["epoch"] for e in epochs] == [1, 2]
    assert {"epoch", "loss", "accuracy", "val_loss", "val_accuracy"} <= set(epochs[0])


def test_trained_model_runs_in_the_runtime(trained: tuple[Path, Path, train.TrainResult]) -> None:
    from chordfinder.model import ChordModel

    out: Path = trained[1]
    label: str
    confidence: float
    label, confidence = ChordModel(out).predict(tone("Am", 1.5))
    assert label in meta().classes
    assert 0.0 <= confidence <= 1.0


def test_train_without_validation(tmp_path: Path) -> None:
    pytest.importorskip("keras")
    root: Path = tmp_path / "ds"
    synthetic_dataset(root, groups=2, takes=1, seconds=1.5)
    config: train.TrainConfig = train.TrainConfig(
        epochs=1, batch_size=16, learning_rate=1e-3, seed=0, val_fraction=0.0
    )
    result: train.TrainResult = train.train(root, tmp_path / "out", config, meta())
    assert result.val_accuracy is None
    assert result.val_windows == 0
    history: dict[str, object] = json.loads((tmp_path / "out" / "history.json").read_text())
    assert history["val_groups"] == []
    epochs: list[dict[str, float]] = json.loads((tmp_path / "out" / "history.json").read_text())["epochs"]
    assert all(not any(k.startswith("val_") for k in e) for e in epochs)


# 25


def test_same_seed_same_weights(tmp_path: Path) -> None:
    keras: ModuleType = pytest.importorskip("keras")
    root: Path = tmp_path / "ds"
    synthetic_dataset(root, groups=2, takes=1, seconds=1.5)
    config: train.TrainConfig = train.TrainConfig(
        epochs=2, batch_size=8, learning_rate=1e-3, seed=3, val_fraction=0.5
    )
    runs: list[Path] = [tmp_path / "a", tmp_path / "b"]
    run: Path
    for run in runs:
        train.train(root, run, config, meta())
    weights: list[list[npt.NDArray[np.float32]]] = [
        keras.saving.load_model(r / "chord-cnn.keras").get_weights() for r in runs
    ]
    assert len(weights[0]) == len(weights[1])
    assert all(np.array_equal(a, b) for a, b in zip(weights[0], weights[1], strict=True))
    histories: list[object] = [json.loads((r / "history.json").read_text())["epochs"] for r in runs]
    assert histories[0] == histories[1]


# 26


def test_train_never_decodes_the_test_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("keras")
    import librosa

    root: Path = tmp_path / "ds"
    synthetic_dataset(root, groups=2, takes=1, seconds=1.5)
    victim: Path
    for victim in (root / "Test").glob("*/*.wav"):
        victim.write_bytes(b"not audio at all")  # any decoder would choke; hashing the bytes is harmless
    opened: list[Path] = []
    # cast: ty would otherwise narrow real_load to librosa.load's exact signature
    real_load: Callable[..., object] = cast(Callable[..., object], librosa.load)

    def watching(path: str | Path, *args: object, **kwargs: object) -> object:
        opened.append(Path(path))
        if "Test" in Path(path).parts:
            raise AssertionError(f"train opened {path}")
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(librosa, "load", watching)
    config: train.TrainConfig = train.TrainConfig(
        epochs=1, batch_size=8, learning_rate=1e-3, seed=0, val_fraction=0.5
    )
    train.train(root, tmp_path / "out", config, meta())
    assert opened
    assert all(re.search(r"/Training/", p.as_posix()) for p in opened)
