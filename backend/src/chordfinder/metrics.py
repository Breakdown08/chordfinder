"""Prometheus metrics of the recognizer, in the default registry (GET /metrics, see server.py)."""

import asyncio
import os
import re
import time
from pathlib import Path

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)

MULTIPROC_ENV: str = "PROMETHEUS_MULTIPROC_DIR"
MULTIPROC_ENV_OLD: str = "prometheus_multiproc_dir"

LOOP_PROBE_SECONDS: float = 0.1

# Seconds: fine below 1 ms (one stage can be that short) and a boundary at the 125 ms hop.
TIME_BUCKETS: tuple[float, ...] = (
    0.0001,
    0.00025,
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.125,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
)
# Lags grow without bound when the server is overloaded: up to 10 s and beyond.
LAG_BUCKETS: tuple[float, ...] = (
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.125,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
)

WS_CONNECTIONS: Gauge = Gauge(
    "chordfinder_ws_connections", "Open /ws/recognize connections", multiprocess_mode="livesum"
)
FRAMES: Counter = Counter("chordfinder_frames", "Binary frames accepted by decode_frame")
FRAME_ERRORS: Counter = Counter("chordfinder_frame_errors", "Binary frames rejected by decode_frame")
SILENT_WINDOWS: Counter = Counter("chordfinder_silent_windows", "Windows taken for silence (no stage ran)")
FRAME_SECONDS: Histogram = Histogram(
    "chordfinder_frame_seconds", "Frame received to reply sent", buckets=TIME_BUCKETS
)
THREADPOOL_WAIT_SECONDS: Histogram = Histogram(
    "chordfinder_threadpool_wait_seconds",
    "to_thread call to start of work in the thread",
    buckets=TIME_BUCKETS,
)
STAGE_SECONDS: Histogram = Histogram(
    "chordfinder_stage_seconds", "Time of one processing stage", ["stage"], buckets=TIME_BUCKETS
)
STREAM_LAG_SECONDS: Histogram = Histogram(
    "chordfinder_stream_lag_seconds", "Arrival behind the audio already received", buckets=LAG_BUCKETS
)
EVENT_LOOP_LAG_SECONDS: Histogram = Histogram(
    "chordfinder_event_loop_lag_seconds", "asyncio.sleep overshoot", buckets=LAG_BUCKETS
)


async def watch_event_loop() -> None:
    """Forever: sleep LOOP_PROBE_SECONDS and record how much later the loop woke us."""
    while True:
        start: float = time.perf_counter()
        await asyncio.sleep(LOOP_PROBE_SECONDS)
        EVENT_LOOP_LAG_SECONDS.observe(max(0.0, time.perf_counter() - start - LOOP_PROBE_SECONDS))
        record_cpu()


def multiproc_dir() -> Path | None:
    """PROMETHEUS_MULTIPROC_DIR (or the old lower-case name), None when neither is set."""
    value: str | None = os.environ.get(MULTIPROC_ENV, os.environ.get(MULTIPROC_ENV_OLD))
    return Path(value) if value is not None else None


# The same rule as prometheus_client.values.get_value_class: by the presence of the key, at import.
MULTIPROCESS: bool = multiproc_dir() is not None

WORKER_WS_CONNECTIONS: Gauge = Gauge(
    "chordfinder_worker_ws_connections",
    "Open /ws/recognize connections of this process",
    multiprocess_mode="liveall",
)
CPU_SECONDS: Counter = Counter("chordfinder_cpu_seconds", "CPU time (user + system) of the server processes")

_last_cpu: list[float] = [0.0]


def record_cpu() -> None:
    """Add the CPU time of this process since the last call (since the process start at the first)."""
    now: float = time.process_time()
    CPU_SECONDS.inc(max(0.0, now - _last_cpu[0]))
    _last_cpu[0] = now


def exposition() -> bytes:
    record_cpu()
    if MULTIPROCESS:
        registry: CollectorRegistry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)  # type: ignore[no-untyped-call]
        return generate_latest(registry)
    return generate_latest(REGISTRY)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def forget_dead_processes(path: Path | None = None) -> list[int]:
    """Drop the live gauges of processes that no longer run (a worker restarted after a crash)."""
    directory: Path | None = path if path is not None else multiproc_dir()
    if directory is None:
        return []
    dead: set[int] = set()
    file: Path
    for file in directory.glob("gauge_live*_*.db"):
        match: re.Match[str] | None = re.fullmatch(r"gauge_live\w+?_(\d+)\.db", file.name)
        if match is None:
            continue
        pid: int = int(match.group(1))
        if pid != os.getpid() and not alive(pid):
            dead.add(pid)
    for pid in sorted(dead):
        multiprocess.mark_process_dead(pid, str(directory))  # type: ignore[no-untyped-call]
    return sorted(dead)


def process_exit() -> None:
    if MULTIPROCESS:
        multiprocess.mark_process_dead(os.getpid())  # type: ignore[no-untyped-call]
