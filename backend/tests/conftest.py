"""Shared fixtures. Tests that need the weights or the dataset skip without them (see README)."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from chordfinder.audio import Audio
from chordfinder.evaluate import DATASET
from chordfinder.model import MODELS, ONNX, ChordModel

needs_model: pytest.MarkDecorator = pytest.mark.skipif(
    not (MODELS / ONNX).is_file(), reason="model weights not downloaded"
)
needs_dataset: pytest.MarkDecorator = pytest.mark.skipif(
    not (DATASET / "Test").is_dir(), reason="dataset not downloaded"
)


def sine(hz: float, seconds: float, sr: int, amplitude: float = 0.5) -> Audio:
    t: Audio = np.arange(int(sr * seconds), dtype=np.float32) / sr
    y: Audio = (amplitude * np.sin(2 * np.pi * hz * t)).astype(np.float32)
    return y


def dataset_file(name: str) -> Path:
    return DATASET / "Test" / name


@pytest.fixture(scope="session")
def model() -> ChordModel:
    if not (MODELS / ONNX).is_file():
        pytest.skip("model weights not downloaded")
    return ChordModel()


@pytest.fixture
def client(model: ChordModel) -> Iterator[TestClient]:
    from chordfinder.server import app

    c: TestClient
    with TestClient(app) as c:  # runs the lifespan: the model is loaded like in production
        yield c
