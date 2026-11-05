"""Measuring session: strum counting, the files it leaves, scoring them."""

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.io import wavfile
from starlette.testclient import WebSocketTestSession

from chordfinder import server
from chordfinder.audio import Audio
from chordfinder.measure import SESSION_FILE, Onsets, Session, Strum, Window, score
from tests.conftest import needs_model

RATE: int = 48000
HOP: int = 6000


def strum(seconds: float = 2.0, peak: float = 0.1) -> Audio:
    """A plucked decay."""
    t: Audio = np.arange(int(RATE * seconds), dtype=np.float32) / RATE
    y: Audio = peak * np.sin(2 * np.pi * 196 * t) * np.exp(-3 * t)
    return y.astype(np.float32)


def room(seconds: float, level: float = 0.003) -> Audio:
    return np.random.default_rng(0).normal(0, level, int(RATE * seconds)).astype(np.float32)


def count(y: Audio) -> list[int]:
    onsets: Onsets = Onsets(RATE)
    return [at for at in range(0, len(y), HOP) if onsets.push(y[at : at + HOP], at)]


def test_counts_each_strum_once() -> None:
    y: Audio = np.concatenate([room(1), strum(), room(0.5), strum(), room(0.5), strum(), room(0.5)])
    assert len(count(y)) == 3


def test_quiet_strum_counts() -> None:
    assert len(count(np.concatenate([room(1), strum(peak=0.04)]))) == 1


def test_room_noise_is_no_strum() -> None:
    assert count(room(5)) == []
    assert count(room(5, 0.008)) == []


def test_ring_of_a_strum_is_no_new_strum() -> None:
    assert len(count(strum(4))) == 1


def test_session_files(tmp_path: Path) -> None:
    s: Session = Session(tmp_path, RATE)
    s.start("Am")
    s.push(room(0.25))
    s.push(strum(0.25))
    s.stop()
    s.start("Am")  # the take repeated: the file is replaced
    s.push(room(0.25))
    s.push(strum(0.25))
    s.push(strum(0.25))
    s.stop()
    meta: dict[str, object] = json.loads((s.path / SESSION_FILE).read_text())
    assert meta == {"sample_rate": RATE, "takes": {"Am": [12000]}}
    rate: int
    y: Audio
    rate, y = wavfile.read(s.path / "Am.wav")
    assert rate == RATE and len(y) == RATE * 3 // 4 and y.dtype == np.int16


def test_score_gives_each_strum_its_windows() -> None:
    windows: list[Window] = [
        Window(100, None, None),
        Window(1100, "Bb", None),
        Window(1200, "C", None),
        Window(1300, "C", "C"),
        Window(2100, "Em", "Em"),
    ]
    first: Strum
    second: Strum
    first, second = score("C", windows, [1000, 2000], 1000)
    assert first.majority == "C" and first.first_stable == "C"
    assert first.latency == 0.3
    assert second.first_stable == "Em" and second.latency is None


@needs_model
def test_measuring_is_off_by_default() -> None:
    client: TestClient
    with TestClient(server.app) as client:
        ws: WebSocketTestSession
        with client.websocket_connect("/ws/recognize") as ws:
            ws.send_text(json.dumps({"sample_rate": RATE}))
            ws.send_text(json.dumps({"take": "Am"}))
            assert "CHORDFINDER_SESSIONS" in ws.receive_json()["error"]


@needs_model
def test_take_over_the_socket(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "SESSIONS", tmp_path)
    client: TestClient
    with TestClient(server.app) as client:
        ws: WebSocketTestSession
        with client.websocket_connect("/ws/recognize") as ws:
            ws.send_text(json.dumps({"sample_rate": RATE}))
            ws.send_text(json.dumps({"take": "Nope"}))
            assert "unknown chord" in ws.receive_json()["error"]
            ws.send_text(json.dumps({"take": "G"}))
            y: Audio = np.concatenate([room(0.5), strum(1)])
            replies: list[dict[str, object]] = []
            start: int
            for start in range(0, len(y), HOP):
                ws.send_bytes(y[start : start + HOP].tobytes())
                replies.append(ws.receive_json())
            ws.send_text(json.dumps({"take": None}))
            ws.send_bytes(y[:HOP].tobytes())
            last: dict[str, object] = ws.receive_json()
    assert replies[-1]["take"] == "G" and replies[-1]["strums"] == 1
    assert last["take"] is None
    assert [p.name for p in next(tmp_path.iterdir()).iterdir() if p.suffix == ".wav"] == ["G.wav"]
