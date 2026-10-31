"""The ONNX port answers like the thesis model did (baseline/old-model.json, its own predict)."""

from pathlib import Path

import numpy as np
import pytest
from pydantic import BaseModel

from chordfinder import features
from chordfinder.audio import Audio
from chordfinder.evaluate import Report, dataset_files, evaluate
from chordfinder.model import ChordModel
from tests.conftest import needs_dataset


class OldPrediction(BaseModel):
    file: str
    label: str
    confidence: float


class Baseline(BaseModel):
    accuracy: float
    predictions: list[OldPrediction]


BASELINE: Baseline = Baseline.model_validate_json(
    (Path(__file__).resolve().parents[1] / "baseline/old-model.json").read_text()
)


@pytest.fixture(scope="module")
def report(model: ChordModel) -> Report:
    return evaluate(model, dataset_files())


@needs_dataset
def test_same_labels_as_thesis_model(report: Report) -> None:
    old: dict[str, str] = {p.file: p.label for p in BASELINE.predictions}
    new: dict[str, str] = {p.file: p.label for p in report.predictions}
    assert new.keys() == old.keys()
    assert [f for f in old if new[f] != old[f]] == []


@needs_dataset
def test_same_confidence_as_thesis_model(report: Report) -> None:
    old: dict[str, float] = {p.file: p.confidence for p in BASELINE.predictions}
    assert max(abs(p.confidence - old[p.file]) for p in report.predictions) < 1e-4


@needs_dataset
def test_accuracy_matches_baseline(report: Report) -> None:
    assert report.accuracy == BASELINE.accuracy


def test_envelope_centres_window_like_pandas() -> None:
    # pandas rolling(800, center=True) covers [i-400, i+399]: a lone spike at 1000 lights up 601..1400
    y: Audio = np.zeros(2000, np.float32)
    y[1000] = 1.0
    mask: np.typing.NDArray[np.bool_] = features.envelope(y, 16000, 0.5)
    assert np.flatnonzero(mask).tolist() == list(range(601, 1401))


def test_window_is_one_second(model: ChordModel) -> None:
    meta: features.Meta = model.meta
    assert len(features.window(np.full(meta.sr * 2, 0.5, np.float32), meta)) == meta.sr
    assert len(features.window(np.full(100, 0.5, np.float32), meta)) == meta.sr


def test_silence_is_still_a_chord(model: ChordModel) -> None:
    # No "none" class yet: the model always names a chord, so the server gates on loudness first.
    label: str = model.predict(np.zeros(model.meta.sr, np.float32))[0]
    assert label in model.meta.classes
