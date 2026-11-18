"""Metrics of the workers task: one process (checks 15-18) and several processes (checks 19-20)."""

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from httpx import Response
from prometheus_client import REGISTRY
from starlette.testclient import WebSocketTestSession

from chordfinder import metrics
from tests.promtext import parse, scrape, series

RATE: int = 48000
CHUNK: int = 6000
WORKER: str = "chordfinder_worker_ws_connections"
CPU: str = "chordfinder_cpu_seconds_total"
PROCESS_CPU: str = "process_cpu_seconds_total"


@pytest.fixture(autouse=True, scope="session")
def single_process_environment() -> None:
    names: list[str] = [
        k for k in ("PROMETHEUS_MULTIPROC_DIR", "prometheus_multiproc_dir") if k in os.environ
    ]
    if names:
        pytest.fail(f"{names} is set in the environment of pytest: the single-process tests need it unset")


# 15
def test_single_process_mode_exposes_the_new_series(client: TestClient) -> None:
    assert metrics.MULTIPROCESS is False
    r: Response = client.get("/metrics")
    assert r.status_code == 200
    # prometheus_client 0.26 names the counter family with _total
    assert "# TYPE chordfinder_cpu_seconds_total counter" in r.text
    values: dict[str, float] = parse(r.text)
    assert CPU in values
    assert WORKER in values
    assert PROCESS_CPU in values


# 16
def test_own_cpu_counter_matches_process_cpu(client: TestClient) -> None:
    values: dict[str, float] = scrape(client)
    assert abs(values[CPU] - values[PROCESS_CPU]) <= 0.25


def test_own_cpu_counter_grows_with_work(client: TestClient) -> None:
    before: dict[str, float] = scrape(client)
    spent: list[float] = []

    def spin() -> None:
        start: float = time.thread_time()
        while time.thread_time() - start < 0.3:
            pass

    t0: float = time.process_time()
    thread: threading.Thread = threading.Thread(target=spin)
    thread.start()
    thread.join()
    spent.append(time.process_time() - t0)
    after: dict[str, float] = scrape(client)
    assert spent[0] >= 0.2
    assert after[CPU] - before[CPU] >= 0.15


# 17
def wait_for(client: TestClient, name: str, target: float, timeout: float = 2.0) -> float:
    deadline: float = time.monotonic() + timeout
    value: float = scrape(client).get(name, -1.0)
    while value != target and time.monotonic() < deadline:
        time.sleep(0.05)
        value = scrape(client).get(name, -1.0)
    return value


def test_worker_gauge_follows_open_sockets(client: TestClient) -> None:
    base: float = scrape(client).get(WORKER, -1.0)
    assert base >= 0.0, f"{WORKER} is missing"
    ws: WebSocketTestSession
    with client.websocket_connect("/ws/recognize") as ws:
        ws.send_text(json.dumps({"sample_rate": RATE}))
        ws.send_bytes(np.zeros(CHUNK, np.float32).tobytes())
        ws.receive_json()
        assert wait_for(client, WORKER, base + 1) == base + 1
        assert wait_for(client, "chordfinder_ws_connections", base + 1) == base + 1
    assert wait_for(client, WORKER, base) == base


# 18
def test_exit_helpers_are_harmless_in_single_process_mode() -> None:
    metrics.process_exit()
    assert metrics.forget_dead_processes() == []


# interface: record_cpu never takes back, exposition in single mode is the default registry
def test_record_cpu_is_monotonic_in_single_process_mode() -> None:
    first: float = REGISTRY.get_sample_value(CPU) or 0.0
    metrics.record_cpu()
    second: float = REGISTRY.get_sample_value(CPU) or 0.0
    metrics.record_cpu()
    assert second >= first
    assert (REGISTRY.get_sample_value(CPU) or 0.0) >= second


# --- several processes (19-20): fresh interpreters over one PROMETHEUS_MULTIPROC_DIR -------------

BUMP: str = """
import os, sys
from chordfinder import metrics
g, f = int(sys.argv[1]), int(sys.argv[2])
metrics.WS_CONNECTIONS.inc(g)
metrics.WORKER_WS_CONNECTIONS.inc(g)
metrics.FRAMES.inc(f)
metrics.record_cpu()
print(os.getpid())
"""
SHOW: str = """
import json, os, sys
from chordfinder import metrics
dead = metrics.forget_dead_processes() if sys.argv[1] == "forget" else None
print(json.dumps({"pid": os.getpid(), "dead": dead, "text": metrics.exposition().decode()}))
"""


def python(script: str, directory: Path, *args: str) -> str:
    env: dict[str, str] = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PROMETHEUS_MULTIPROC_DIR", "prometheus_multiproc_dir")
    }
    env["PROMETHEUS_MULTIPROC_DIR"] = str(directory)
    done: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "-c", script, *args], capture_output=True, text=True, timeout=120, env=env
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


# 19
def test_processes_sum_up_and_dead_ones_are_forgotten(tmp_path: Path) -> None:
    a: int = int(python(BUMP, tmp_path, "1", "3"))
    b: int = int(python(BUMP, tmp_path, "2", "4"))

    shown: dict[str, object] = json.loads(python(SHOW, tmp_path, "keep"))
    text: str = str(shown["text"])
    c: int = int(str(shown["pid"]))
    values: dict[str, float] = parse(text)
    assert values["chordfinder_ws_connections"] == 3
    assert values[series(WORKER, pid=str(a))] == 1
    assert values[series(WORKER, pid=str(b))] == 2
    assert values["chordfinder_frames_total"] == 7
    assert values[CPU] > 0
    assert PROCESS_CPU not in text

    after: dict[str, object] = json.loads(python(SHOW, tmp_path, "forget"))
    dead: object = after["dead"]
    assert isinstance(dead, list)
    assert {a, b, c} <= set(dead)
    assert dead == sorted(dead)
    now: dict[str, float] = parse(str(after["text"]))
    assert now["chordfinder_ws_connections"] == 0
    assert series(WORKER, pid=str(a)) not in now
    assert series(WORKER, pid=str(b)) not in now
    assert now["chordfinder_frames_total"] == 7
    assert now[CPU] >= values[CPU]


# 20
def test_forget_dead_processes_removes_only_dead_gauges(tmp_path: Path) -> None:
    gone: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True, timeout=60
    )
    dead: int = int(gone.stdout.strip())
    me: int = os.getpid()
    name: str
    for name in (
        f"gauge_livesum_{dead}.db",
        f"gauge_liveall_{dead}.db",
        f"counter_{dead}.db",
        f"gauge_livesum_{me}.db",
    ):
        (tmp_path / name).write_text("")
    assert metrics.forget_dead_processes(tmp_path) == [dead]
    assert not (tmp_path / f"gauge_livesum_{dead}.db").exists()
    assert not (tmp_path / f"gauge_liveall_{dead}.db").exists()
    assert (tmp_path / f"counter_{dead}.db").exists()
    assert (tmp_path / f"gauge_livesum_{me}.db").exists()


def test_permission_error_means_alive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_kill: Callable[[int, int], None] = os.kill

    def kill(pid: int, sig: int) -> None:
        if pid == 2147000001:
            raise PermissionError
        if pid == 2147000002:
            raise ProcessLookupError
        real_kill(pid, sig)

    monkeypatch.setattr(os, "kill", kill)
    name: str
    for name in ("gauge_livesum_2147000001.db", "gauge_livesum_2147000002.db"):
        (tmp_path / name).write_text("")
    assert metrics.forget_dead_processes(tmp_path) == [2147000002]
    assert (tmp_path / "gauge_livesum_2147000001.db").exists()
