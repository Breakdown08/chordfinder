"""Prometheus metrics of the recognizer, in the default registry (GET /metrics, see server.py)."""

import asyncio
import time

from prometheus_client import Counter, Gauge, Histogram

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

WS_CONNECTIONS: Gauge = Gauge("chordfinder_ws_connections", "Open /ws/recognize connections")
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
