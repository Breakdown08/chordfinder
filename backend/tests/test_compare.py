"""Confusion matrix, paired comparison, `chordfinder compare` and `eval --models`: fake models, tiny data."""

import hashlib
import json
import math
from pathlib import Path

import pytest

from chordfinder import dataset, evaluate
from chordfinder import model as model_module
from chordfinder.audio import Audio
from chordfinder.evaluate import Comparison, Prediction, Report
from chordfinder.features import Meta
from tests.synthetic import CLASSES, meta, run_cli, synthetic_dataset, tone, write_wav
from tests.test_dataset import manifest_for, patch_manifest


def report(rows: list[tuple[str, str, str]]) -> Report:
    """(file, truth, label) -> Report."""
    r: Report = Report()
    row: tuple[str, str, str]
    for row in rows:
        r.predictions.append(Prediction(row[0], row[1], row[2], 0.9))
    return r


# 19


def test_confusion_matrix() -> None:
    r: Report = report(
        [
            ("Am/1.wav", "Am", "Am"),
            ("Am/2.wav", "Am", "Bb"),
            ("Bdim/1.wav", "Bdim", "Bb"),
            ("G/1.wav", "G", "G"),
            ("G/2.wav", "G", "G"),
        ]
    )
    m: list[list[int]] = r.confusion(CLASSES)
    expected: list[list[int]] = [[0] * 8 for _ in range(8)]
    expected[0][0] = 1
    expected[0][1] = 1
    expected[2][1] = 1
    expected[7][7] = 2
    assert m == expected
    assert sum(sum(row) for row in m) == 5
    assert sum(m[i][i] for i in range(8)) == r.hits == 3


def test_confusion_follows_the_given_class_order() -> None:
    r: Report = report([("Am/1.wav", "Am", "Bb")])
    m: list[list[int]] = r.confusion(["Bb", "Am"])
    assert m == [[0, 0], [1, 0]]


# 20

#   file  :  Am1 Am2 Am3 Am4 Bb1 Bb2 Bb3 Bb4
OLD: list[str] = ["Am", "Am", "Am", "Bb", "Bb", "Bb", "Am", "Am"]
NEW: list[str] = ["Am", "Bb", "Bb", "Bb", "Bb", "Am", "Bb", "Am"]
FILES: list[str] = [f"{c}/{c}_TestPlug_{n}.wav" for c in ("Am", "Bb") for n in (1, 2, 3, 4)]


def scripted(labels: list[str]) -> Report:
    return report([(f, f.split("/")[0], lab) for f, lab in zip(FILES, labels, strict=True)])


def two_sided(b: int, c: int) -> float:
    n: int = b + c
    tail: int = sum(math.comb(n, k) for k in range(min(b, c) + 1))
    p: float = min(1.0, 2 * tail / float(2**n))
    return p


def test_compare_counts_the_pairs() -> None:
    c: Comparison = evaluate.compare(scripted(OLD), scripted(NEW))
    assert (c.old_only, c.new_only, c.both_wrong) == (3, 1, 2)
    assert c.mcnemar_p == pytest.approx(two_sided(3, 1), rel=1e-12)
    assert c.mcnemar_p == pytest.approx(0.625, rel=1e-12)
    assert c.old.accuracy == 0.625
    assert c.new.accuracy == 0.375


def test_compare_pairs_by_file_not_by_position() -> None:
    new: Report = scripted(NEW)
    new.predictions.reverse()
    c: Comparison = evaluate.compare(scripted(OLD), new)
    assert (c.old_only, c.new_only, c.both_wrong) == (3, 1, 2)


def test_compare_rejects_different_files() -> None:
    other: Report = report([(f, f.split("/")[0], "Am") for f in FILES[:-1]] + [("Am/other.wav", "Am", "Am")])
    with pytest.raises(ValueError):
        evaluate.compare(scripted(OLD), other)
    with pytest.raises(ValueError):
        evaluate.compare(scripted(OLD), report([(f, f.split("/")[0], "Am") for f in FILES[:-1]]))


@pytest.mark.parametrize(
    ("b", "c", "p"),
    [(0, 0, 1.0), (1, 0, 1.0), (0, 6, 0.03125), (2, 10, 158 / 4096)],
)
def test_mcnemar_examples(b: int, c: int, p: float) -> None:
    assert evaluate.mcnemar_p(b, c) == pytest.approx(p, rel=1e-12)
    assert evaluate.mcnemar_p(c, b) == pytest.approx(p, rel=1e-12)


def test_mcnemar_is_capped_at_one() -> None:
    assert evaluate.mcnemar_p(5, 5) == 1.0


# 21


class Fake:
    """Satisfies evaluate.Predictor; answers from a script, in the order the files are given."""

    def __init__(self, labels: list[str], confidence: float = 0.9) -> None:
        self.meta: Meta = meta()
        self.labels: list[str] = labels
        self.confidence: float = confidence
        self.calls: int = 0

    def predict(self, audio_16k: Audio) -> tuple[str, float]:
        label: str = self.labels[self.calls % len(self.labels)]
        self.calls += 1
        return label, self.confidence


def test_evaluate_takes_any_predictor(tmp_path: Path) -> None:
    synthetic_dataset(tmp_path, groups=1, takes=1, seconds=1.2, test_files=2)
    files: list[Path] = evaluate.dataset_files(tmp_path)
    assert len(files) == 16
    fake: Fake = Fake(["G"])
    r: Report = evaluate.evaluate(fake, files)
    assert fake.calls == 16
    assert [p.file for p in r.predictions] == [f"{f.parent.name}/{f.name}" for f in files]
    assert [p.truth for p in r.predictions] == [f.parent.name for f in files]
    assert {p.label for p in r.predictions} == {"G"}
    assert r.hits == 2  # the two G files


# 22

TEST_FILES: int = 4


def compare_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, test_hash: str | None = None
) -> tuple[Path, Path, Path]:
    """A tiny dataset with 4 Am and 4 Bb test files, a manifest that pins it, two scripted fake models."""
    root: Path = tmp_path / "ds"
    synthetic_dataset(root, groups=1, takes=1, seconds=1.2, test_files=0)
    n: int
    for n in range(1, TEST_FILES + 1):
        cls: str
        for cls in ("Am", "Bb"):
            write_wav(root / "Test" / cls / f"{cls}_TestPlug_{n}.wav", tone(cls, 1.2))
    manifest: dataset.DatasetManifest = manifest_for(root)
    if test_hash is not None:
        manifest = manifest.model_copy(update={"test_content_sha256": test_hash})
    patch_manifest(monkeypatch, manifest)
    old: Path = tmp_path / "old"
    new: Path = tmp_path / "new"
    scripts: dict[str, Fake] = {"old": Fake(OLD, 0.9), "new": Fake(NEW, 0.8)}
    old.mkdir()
    new.mkdir()
    (old / "chord-cnn.onnx").write_bytes(b"old onnx")
    (new / "chord-cnn.onnx").write_bytes(b"new onnx")

    def factory(models: Path = model_module.MODELS) -> Fake:
        return scripts[models.name]

    monkeypatch.setattr(model_module, "ChordModel", factory)
    monkeypatch.setattr("chordfinder.cli.ChordModel", factory, raising=False)
    return root, old, new


def test_cli_compare_writes_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root: Path
    old: Path
    new: Path
    root, old, new = compare_setup(tmp_path, monkeypatch)
    out: Path = tmp_path / "out.json"
    argv: list[str] = ["compare", "--old", str(old), "--new", str(new), "--dataset", str(root)]
    assert run_cli([*argv, "--json", str(out)]) == 0

    printed: str = capsys.readouterr().out
    word: str
    for word in ("62.5", "37.5", "old_only", "new_only", "both_wrong", "mcnemar_p"):
        assert word in printed

    data: dict[str, object] = json.loads(out.read_text())
    assert {"dataset", "test_content_sha256", "classes", "models", "paired", "predictions"} <= set(data)
    assert data["test_content_sha256"] == dataset.content_digest(root, "Test")[0]
    assert data["classes"] == list(CLASSES)
    models: dict[str, dict[str, object]] = json.loads(out.read_text())["models"]
    side: str
    for side in ("old", "new"):
        assert {"path", "onnx_sha256", "accuracy", "per_class", "confusion"} <= set(models[side])
    assert models["old"]["onnx_sha256"] == hashlib.sha256(b"old onnx").hexdigest()
    assert models["new"]["onnx_sha256"] == hashlib.sha256(b"new onnx").hexdigest()
    assert models["old"]["accuracy"] == 0.625
    assert models["new"]["accuracy"] == 0.375
    assert models["old"]["per_class"] == {"Am": 0.75, "Bb": 0.5}
    assert models["new"]["per_class"] == {"Am": 0.25, "Bb": 0.5}
    old_confusion: list[list[int]] = json.loads(out.read_text())["models"]["old"]["confusion"]
    assert old_confusion[0][:2] == [3, 1]
    assert old_confusion[1][:2] == [2, 2]
    assert len(old_confusion) == 8
    paired: dict[str, float] = json.loads(out.read_text())["paired"]
    assert (paired["old_only"], paired["new_only"], paired["both_wrong"]) == (3, 1, 2)
    assert paired["mcnemar_p"] == pytest.approx(0.625)
    predictions: list[dict[str, object]] = json.loads(out.read_text())["predictions"]
    assert len(predictions) == 8
    first: dict[str, object] = predictions[0]
    assert {"file", "truth", "old", "old_confidence", "new", "new_confidence"} <= set(first)
    assert first["file"] == "Am/Am_TestPlug_1.wav"
    assert first["truth"] == "Am"
    assert (first["old"], first["new"]) == ("Am", "Am")
    assert (first["old_confidence"], first["new_confidence"]) == (0.9, 0.8)
    assert (predictions[1]["old"], predictions[1]["new"]) == ("Am", "Bb")


def test_cli_compare_refuses_a_test_split_that_is_not_the_pinned_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root: Path
    old: Path
    new: Path
    root, old, new = compare_setup(tmp_path, monkeypatch, test_hash="0" * 64)
    out: Path = tmp_path / "out.json"
    argv: list[str] = ["compare", "--old", str(old), "--new", str(new), "--dataset", str(root)]
    assert run_cli([*argv, "--json", str(out)]) == 1
    assert not out.exists()


# eval --models


def test_cli_eval_loads_the_models_it_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root: Path = tmp_path / "ds"
    synthetic_dataset(root, groups=1, takes=1, seconds=1.2, test_files=1)
    asked: list[Path] = []

    def factory(models: Path = model_module.MODELS) -> Fake:
        asked.append(models)
        return Fake(list(CLASSES))

    monkeypatch.setattr(model_module, "ChordModel", factory)
    monkeypatch.setattr("chordfinder.cli.ChordModel", factory, raising=False)
    files: list[Path] = sorted((root / "Test").glob("*/*.wav"))
    monkeypatch.setattr(evaluate, "dataset_files", lambda *args, **kwargs: files)
    chosen: Path = tmp_path / "trained"
    assert run_cli(["eval", "--models", str(chosen)]) == 0
    assert asked == [chosen]
    out: str = capsys.readouterr().out
    assert "accuracy 100.0% (8/8)" in out
    assert len(out.strip().splitlines()) > 1 + 8  # a confusion matrix follows the per-class lines
