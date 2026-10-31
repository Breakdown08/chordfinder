"""The ONNX port answers like the diploma model did (baseline/old-model.json, its own predict)."""

import json
from pathlib import Path

import numpy as np
import pytest

from chordfinder import features
from chordfinder.evaluate import DATASET, evaluate, test_files
from chordfinder.model import ChordModel

BASELINE = json.loads((Path(__file__).resolve().parents[1] / "baseline/old-model.json").read_text())
needs_dataset = pytest.mark.skipif(not (DATASET / "Test").is_dir(), reason="dataset not available locally")


@pytest.fixture(scope="module")
def model():
    return ChordModel()


@pytest.fixture(scope="module")
def report(model):
    return evaluate(model, test_files())


@needs_dataset
def test_same_labels_as_diploma_model(report):
    old = {p["file"]: p["label"] for p in BASELINE["predictions"]}
    new = {p.file: p.label for p in report.predictions}
    assert new.keys() == old.keys()
    assert [f for f in old if new[f] != old[f]] == []


@needs_dataset
def test_same_confidence_as_diploma_model(report):
    old = {p["file"]: p["confidence"] for p in BASELINE["predictions"]}
    assert max(abs(p.confidence - old[p.file]) for p in report.predictions) < 1e-4


@needs_dataset
def test_accuracy_matches_baseline(report):
    assert report.accuracy == BASELINE["accuracy"]


def test_envelope_centres_window_like_pandas():
    # pandas rolling(800, center=True) covers [i-400, i+399]: a lone spike at 1000 lights up 600..1399
    y = np.zeros(2000, np.float32)
    y[1000] = 1.0
    mask = features.envelope(y, 16000, 0.5)
    assert np.flatnonzero(mask).tolist() == list(range(601, 1401))


def test_window_is_one_second(model):
    meta = model.meta
    assert len(features.window(np.full(meta.sr * 2, 0.5, np.float32), meta)) == meta.sr
    assert len(features.window(np.full(100, 0.5, np.float32), meta)) == meta.sr


def test_silence_is_still_a_chord(model):
    # No "none" class yet: the model always names a chord. Callers gate on loudness first.
    label, _ = model.predict(np.zeros(model.meta.sr, np.float32))
    assert label in model.meta.classes
