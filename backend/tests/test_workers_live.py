"""A real chordfinder-serve with two worker processes (checks 22-26 of the workers task).

Needs the model. The tests run in file order and share one server; the last one stops it.
"""

import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import BinaryIO, NamedTuple

import numpy as np
import pytest
from websockets.sync.client import ClientConnection, connect

from chordfinder.audio import Audio
from tests.conftest import needs_model
from tests.promtext import parse

pytestmark: pytest.MarkDecorator = needs_model

RATE: int = 48000
CHUNK: int = 6000
WORKER: str = "chordfinder_worker_ws_connections"
CONNECTIONS: str = "chordfinder_ws_connections"


class Live(NamedTuple):
    proc: subprocess.Popen[bytes]
    port: int
    prom: Path


def free_port() -> int:
    s: socket.socket = socket.socket()
    s.bind(("127.0.0.1", 0))
    port: int = s.getsockname()[1]
    s.close()
    return port


def fetch(port: int, path: str) -> str:
    response: http.client.HTTPResponse
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5.0) as response:
        text: str = response.read().decode()
    return text


def metrics_now(port: int) -> dict[str, float]:
    return parse(fetch(port, "/metrics"))


def worker_pids(values: dict[str, float]) -> set[int]:
    prefix: str = f'{WORKER}{{pid="'
    return {int(k[len(prefix) : -2]) for k in values if k.startswith(prefix)}


def wait_until(check: Callable[[], bool], timeout: float, what: str, live: Live | None = None) -> None:
    """Poll `check` (errors count as not yet) until it is true; stop early if the launcher died."""
    deadline: float = time.monotonic() + timeout
    while True:
        try:
            if check():
                return
        except (OSError, ValueError, KeyError):
            pass
        if live is not None and live.proc.poll() is not None:
            pytest.fail(f"launcher exited with {live.proc.returncode} while waiting for {what}")
        if time.monotonic() > deadline:
            pytest.fail(f"timed out after {timeout:.0f} s waiting for {what}")
        time.sleep(0.2)


@pytest.fixture(scope="module")
def live(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Live]:
    base: Path = tmp_path_factory.mktemp("workers")
    prom: Path = base / "prom"
    port: int = free_port()
    env: dict[str, str] = {k: v for k, v in os.environ.items() if k != "CHORDFINDER_WORKERS"}
    env["PROMETHEUS_MULTIPROC_DIR"] = str(prom)
    log: BinaryIO = (base / "server.log").open("wb")
    proc: subprocess.Popen[bytes] = subprocess.Popen(
        [sys.executable, "-m", "chordfinder.serve", "--workers", "2", "--host", "127.0.0.1"]
        + ["--port", str(port)],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        yield Live(proc, port, prom)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=30)
        log.close()
        if proc.returncode not in (0, None):
            print((base / "server.log").read_text(errors="replace")[-4000:])


def loud(seed: int) -> bytes:
    t: Audio = np.arange(CHUNK, dtype=np.float32) / RATE
    wave: Audio = (0.4 * np.sin(2 * np.pi * (220.0 + 40 * seed) * t)).astype(np.float32)
    return wave.tobytes()


def two_pids(live: Live) -> bool:
    return len(worker_pids(metrics_now(live.port))) == 2


# 22
def test_both_workers_come_up(live: Live) -> None:
    wait_until(lambda: fetch(live.port, "/healthz") is not None, 90.0, "/healthz", live)
    wait_until(lambda: two_pids(live), 60.0, "two worker pids in /metrics", live)


# 23
def test_four_streams_are_counted_across_processes(live: Live) -> None:
    stack: ExitStack
    with ExitStack() as stack:
        sockets: list[ClientConnection] = [
            stack.enter_context(connect(f"ws://127.0.0.1:{live.port}/ws/recognize")) for _ in range(4)
        ]
        i: int
        ws: ClientConnection
        for i, ws in enumerate(sockets):
            ws.send(json.dumps({"sample_rate": RATE}))
            ws.send(loud(i))
            ws.send(loud(i))
        _: int
        for ws in sockets:
            for _ in range(2):
                reply: dict[str, object] = json.loads(ws.recv(timeout=30))
                assert "error" not in reply, reply

        def settled() -> bool:
            v: dict[str, float] = metrics_now(live.port)
            return (
                v[CONNECTIONS] == 4
                and sum(x for k, x in v.items() if k.startswith(WORKER)) == 4
                and v["chordfinder_frames_total"] == 8
                and v['chordfinder_stage_seconds_count{stage="features"}'] == 8
                and v["chordfinder_cpu_seconds_total"] > 0
                and v["chordfinder_event_loop_lag_seconds_count"] >= 1
            )

        wait_until(settled, 5.0, "summed metrics of four streams", live)
    assert "process_cpu_seconds_total" not in fetch(live.port, "/metrics")


# 24
def test_connections_return_to_zero(live: Live) -> None:
    wait_until(lambda: metrics_now(live.port)[CONNECTIONS] == 0, 5.0, "ws_connections == 0", live)


# 25
def test_killed_worker_is_replaced_and_forgotten(live: Live) -> None:
    victim: int = sorted(worker_pids(metrics_now(live.port)))[0]
    os.kill(victim, signal.SIGKILL)

    def replaced() -> bool:
        v: dict[str, float] = metrics_now(live.port)
        pids: set[int] = worker_pids(v)
        return len(pids) == 2 and victim not in pids and v[CONNECTIONS] == 0

    wait_until(replaced, 90.0, "a replacement worker and no trace of the killed one", live)


# 26
def test_sigterm_stops_cleanly_and_leaves_no_live_gauges(live: Live) -> None:
    live.proc.send_signal(signal.SIGTERM)
    assert live.proc.wait(timeout=30) == 0
    assert list(live.prom.glob("gauge_live*.db")) == []
