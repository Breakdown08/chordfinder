"""HTTP and WebSocket API on the real model."""

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from httpx import Response
from starlette.testclient import WebSocketTestSession

from chordfinder import audio
from chordfinder.audio import Audio
from chordfinder.evaluate import load
from chordfinder.server import ClipReply, StreamReply
from chordfinder.stream import HOP_SECONDS
from tests.conftest import dataset_file, needs_dataset, sine

RATE: int = 48000
CHUNK: int = int(RATE * HOP_SECONDS)
CHORDS: list[str] = ["Am", "Bb", "Bdim", "C", "Dm", "Em", "F", "G"]


def clip(name: str) -> Audio:
    return load(dataset_file(name), RATE)


def first_file(chord: str) -> str:
    path: Path = sorted((dataset_file(chord)).glob("*.wav"))[0]
    return f"{chord}/{path.name}"


def send(ws: WebSocketTestSession, y: Audio) -> list[StreamReply]:
    replies: list[StreamReply] = []
    start: int
    for start in range(0, len(y), CHUNK):
        ws.send_bytes(y[start : start + CHUNK].astype("<f4").tobytes())
        replies.append(StreamReply.model_validate(ws.receive_json()))
    return replies


def test_healthz(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}


@needs_dataset
@pytest.mark.parametrize("chord", CHORDS)
def test_predict_clip(client: TestClient, chord: str) -> None:
    y: Audio = clip(first_file(chord))
    r: Response = client.post("/v1/predict", json={"sample_rate": RATE, "samples": y.tolist()})
    assert r.status_code == 200
    body: ClipReply = ClipReply.model_validate(r.json())
    assert body.chord == chord
    assert body.probabilities is not None
    assert sorted(body.probabilities) == sorted(CHORDS)
    assert sum(body.probabilities.values()) == pytest.approx(1, abs=1e-4)


def test_predict_silence_is_no_chord(client: TestClient) -> None:
    r: Response = client.post("/v1/predict", json={"sample_rate": RATE, "samples": [0.0] * RATE})
    assert r.json() == {"chord": None, "confidence": None, "rms": 0.0, "probabilities": None}


def test_predict_limits(client: TestClient) -> None:
    long: list[float] = [0.1] * int(RATE * 10.5)
    assert client.post("/v1/predict", json={"sample_rate": RATE, "samples": long}).status_code == 413
    assert client.post("/v1/predict", json={"sample_rate": 100, "samples": [0.1]}).status_code == 422
    assert client.post("/v1/predict", json={"sample_rate": RATE, "samples": []}).status_code == 422


def test_ws_bad_header_closes(client: TestClient) -> None:
    ws: WebSocketTestSession
    with client.websocket_connect("/ws/recognize") as ws:
        ws.send_text("hello")
        assert "not JSON" in ws.receive_json()["error"]
        assert ws.receive()["code"] == 1003


def test_ws_bad_frame_is_error_not_crash(client: TestClient) -> None:
    ws: WebSocketTestSession
    with client.websocket_connect("/ws/recognize") as ws:
        ws.send_text(json.dumps({"sample_rate": RATE}))
        ws.send_bytes(b"\x00\x00\x00")
        assert "multiple of 4" in ws.receive_json()["error"]
        replies: list[StreamReply] = send(ws, sine(440, 0.5, RATE))
    assert replies[-1].samples == RATE // 2


def test_ws_silence_is_no_chord(client: TestClient) -> None:
    ws: WebSocketTestSession
    with client.websocket_connect("/ws/recognize") as ws:
        ws.send_text(json.dumps({"sample_rate": RATE}))
        replies: list[StreamReply] = send(ws, np.zeros(RATE, np.float32))
    assert all(r.chord is None and r.stable is None for r in replies)
    assert all(r.rms < audio.SILENCE_RMS for r in replies)


@needs_dataset
@pytest.mark.parametrize("chord", CHORDS)
def test_ws_chord_becomes_stable(client: TestClient, chord: str) -> None:
    y: Audio = np.concatenate((np.zeros(RATE, np.float32), clip(first_file(chord))))
    ws: WebSocketTestSession
    with client.websocket_connect("/ws/recognize") as ws:
        ws.send_text(json.dumps({"sample_rate": RATE}))
        replies: list[StreamReply] = send(ws, y)
    stable: list[str] = [r.stable for r in replies if r.stable is not None]
    assert stable and set(stable) == {chord}
