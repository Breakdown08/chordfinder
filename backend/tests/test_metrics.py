"""GET /metrics and what the stream handler counts (checks 7-12 of the load task)."""

import json
import re
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from httpx import Response
from starlette.testclient import WebSocketTestSession

from chordfinder.audio import Audio
from tests.conftest import sine
from tests.promtext import delta, scrape, series, settle

RATE: int = 48000
CHUNK: int = 6000
STAGES: tuple[str, ...] = ("resample", "features", "inference")
HISTOGRAMS: tuple[str, ...] = (
    "chordfinder_frame_seconds",
    "chordfinder_threadpool_wait_seconds",
    "chordfinder_stage_seconds",
    "chordfinder_stream_lag_seconds",
    "chordfinder_event_loop_lag_seconds",
)
SCALARS: tuple[str, ...] = (
    "chordfinder_ws_connections",
    "chordfinder_frames_total",
    "chordfinder_frame_errors_total",
    "chordfinder_silent_windows_total",
)


def stream_frames(client: TestClient, y: Audio) -> list[dict[str, object]]:
    replies: list[dict[str, object]] = []
    ws: WebSocketTestSession
    with client.websocket_connect("/ws/recognize") as ws:
        ws.send_text(json.dumps({"sample_rate": RATE}))
        start: int
        for start in range(0, len(y), CHUNK):
            ws.send_bytes(y[start : start + CHUNK].astype("<f4").tobytes())
            replies.append(ws.receive_json())
    return replies


def test_metrics_endpoint_lists_every_metric(client: TestClient) -> None:
    r: Response = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    names: set[str] = {re.split(r"[{ ]", line)[0] for line in r.text.splitlines() if line[:1] != "#"}
    name: str
    for name in SCALARS:
        assert name in names, name
    for name in HISTOGRAMS:
        assert f"{name}_bucket" in names, name
    assert "process_cpu_seconds_total" in names


def bounds(client: TestClient, name: str) -> list[float]:
    """The `le` boundaries of a histogram (for the stage histogram: of one stage)."""
    r: Response = client.get("/metrics")
    assert r.status_code == 200
    text: str = r.text
    found: list[float] = []
    line: str
    for line in text.splitlines():
        if line.startswith(f"{name}_bucket{{") and ("stage" not in line or 'stage="inference"' in line):
            m: re.Match[str] | None = re.search(r'le="([^"]+)"', line)
            assert m is not None
            if m.group(1) != "+Inf":
                found.append(float(m.group(1)))
    return found


@pytest.mark.parametrize(
    "name",
    [
        "chordfinder_frame_seconds",
        "chordfinder_threadpool_wait_seconds",
        "chordfinder_stage_seconds",
    ],
)
def test_time_histograms_resolve_125ms_and_below_1ms(client: TestClient, name: str) -> None:
    b: list[float] = bounds(client, name)
    assert 0.125 in b
    assert any(x < 0.001 for x in b)


@pytest.mark.parametrize("name", ["chordfinder_stream_lag_seconds", "chordfinder_event_loop_lag_seconds"])
def test_lag_histograms_reach_125ms_and_10s(client: TestClient, name: str) -> None:
    b: list[float] = bounds(client, name)
    assert 0.125 in b
    assert max(b) >= 10.0


def test_loud_stream_is_counted_in_every_stage(client: TestClient) -> None:
    before: dict[str, float] = scrape(client)
    replies: list[dict[str, object]] = stream_frames(client, sine(440, 1.0, RATE))
    assert len(replies) == 8
    expected: dict[str, float] = {
        "chordfinder_frames_total": 8,
        "chordfinder_frame_seconds_count": 8,
        "chordfinder_threadpool_wait_seconds_count": 8,
        "chordfinder_stream_lag_seconds_count": 8,
        **{series("chordfinder_stage_seconds_count", stage=s): 8 for s in STAGES},
    }
    after: dict[str, float] = settle(client, before, expected)
    key: str
    value: float
    for key, value in expected.items():
        assert delta(before, after, key) == value, key
    assert delta(before, after, "chordfinder_frame_errors_total") == 0
    assert delta(before, after, "chordfinder_silent_windows_total") == 0
    assert delta(before, after, series("chordfinder_stage_seconds_sum", stage="inference")) > 0


def test_silent_stream_runs_no_stage(client: TestClient) -> None:
    before: dict[str, float] = scrape(client)
    stream_frames(client, np.zeros(8 * CHUNK, np.float32))
    after: dict[str, float] = settle(
        client, before, {"chordfinder_silent_windows_total": 8, "chordfinder_frames_total": 8}
    )
    assert delta(before, after, "chordfinder_silent_windows_total") == 8
    assert delta(before, after, "chordfinder_frames_total") == 8
    stage: str
    for stage in STAGES:
        assert delta(before, after, series("chordfinder_stage_seconds_count", stage=stage)) == 0


def test_bad_frame_counts_as_error_not_frame(client: TestClient) -> None:
    before: dict[str, float] = scrape(client)
    ws: WebSocketTestSession
    with client.websocket_connect("/ws/recognize") as ws:
        ws.send_text(json.dumps({"sample_rate": RATE}))
        ws.send_bytes(b"\x00\x00\x00")
        assert "error" in ws.receive_json()
    after: dict[str, float] = settle(client, before, {"chordfinder_frame_errors_total": 1})
    assert delta(before, after, "chordfinder_frame_errors_total") == 1
    assert delta(before, after, "chordfinder_frames_total") == 0


def wait_connections(client: TestClient, target: float, timeout: float = 2.0) -> float:
    deadline: float = time.monotonic() + timeout
    value: float = scrape(client).get("chordfinder_ws_connections", 0.0)
    while value != target and time.monotonic() < deadline:
        time.sleep(0.05)
        value = scrape(client).get("chordfinder_ws_connections", 0.0)
    return value


def test_connections_gauge_follows_open_sockets(client: TestClient) -> None:
    base: float = scrape(client).get("chordfinder_ws_connections", -1.0)
    assert base >= 0.0, "chordfinder_ws_connections is missing"
    ws: WebSocketTestSession
    with client.websocket_connect("/ws/recognize") as ws:
        ws.send_text(json.dumps({"sample_rate": RATE}))
        ws.send_bytes(np.zeros(CHUNK, np.float32).tobytes())
        ws.receive_json()  # the handler is past the header
        assert wait_connections(client, base + 1) == base + 1
    assert wait_connections(client, base) == base


def test_bad_header_does_not_count_as_connection(client: TestClient) -> None:
    base: float = scrape(client).get("chordfinder_ws_connections", -1.0)
    assert base >= 0.0, "chordfinder_ws_connections is missing"
    ws: WebSocketTestSession
    with client.websocket_connect("/ws/recognize") as ws:
        ws.send_text("hello")
        assert "error" in ws.receive_json()
        assert scrape(client)["chordfinder_ws_connections"] == base
    time.sleep(0.2)
    assert scrape(client)["chordfinder_ws_connections"] == base


def test_event_loop_is_probed_after_start(client: TestClient) -> None:
    time.sleep(0.4)
    deadline: float = time.monotonic() + 2.0
    count: float = scrape(client).get("chordfinder_event_loop_lag_seconds_count", 0.0)
    while count < 1 and time.monotonic() < deadline:
        time.sleep(0.05)
        count = scrape(client).get("chordfinder_event_loop_lag_seconds_count", 0.0)
    assert count >= 1
