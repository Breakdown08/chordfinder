"""Load generator, pure functions (checks 14-26 of the load task): no server, no model."""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pytest

from chordfinder.audio import Audio
from chordfinder.load import (
    Criterion,
    FrameRecord,
    ServerStats,
    StepResult,
    ceiling,
    frame,
    metrics_url,
    server_stats,
    summarize,
    synthetic_signal,
)
from chordfinder.stream import HOP_SECONDS

T0: float = 1000.0
type Verdict = Literal["pass", "fail", "invalid"]


def test_metrics_url() -> None:
    assert metrics_url("ws://h:8000/ws/recognize") == "http://h:8000/metrics"
    assert metrics_url("wss://x.org/ws/recognize") == "https://x.org/metrics"
    assert metrics_url("ws://localhost/ws/recognize") == "http://localhost/metrics"


def test_synthetic_signal_is_loud_finite_and_deterministic() -> None:
    s: Audio = synthetic_signal(48000, 4.0, 0)
    assert s.dtype == np.float32
    assert len(s) == 192000
    assert np.isfinite(s).all()
    assert float(np.max(np.abs(s))) <= 1.0
    block: int
    for block in range(0, len(s), 6000):
        chunk: Audio = s[block : block + 6000]
        assert float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64)))) >= 0.05
    assert np.array_equal(s, synthetic_signal(48000, 4.0, 0))
    assert not np.array_equal(s, synthetic_signal(48000, 4.0, 1))


def test_frame_wraps_around_the_signal() -> None:
    s: Audio = np.arange(10, dtype=np.float32)
    assert frame(s, 4, 0, 2).tolist() == [8, 9, 0, 1]
    assert frame(s, 4, 3, 0).tolist() == [3, 4, 5, 6]
    assert frame(s, 4, 0, 0).tolist() == [0, 1, 2, 3]
    i: int
    for i in range(7):
        f: Audio = frame(s, 4, 7, i)
        assert len(f) == 4
        assert f.dtype == np.float32


def records(
    clients: int = 2, frames: int = 40, delay: float = 0.010, offset: float = 0.0
) -> list[FrameRecord]:
    out: list[FrameRecord] = []
    c: int
    i: int
    for c in range(clients):
        for i in range(frames):
            scheduled: float = T0 + offset + i * HOP_SECONDS
            out.append(FrameRecord(c, i, scheduled, scheduled + delay, False))
    return out


WINDOW: tuple[float, float] = (T0 - 1.0, T0 + 1000.0)


def test_summarize_a_healthy_step_passes() -> None:
    r: StepResult = summarize(2, records(), 0, [0.001] * 20, WINDOW)
    assert r.verdict == "pass"
    assert r.reasons == []
    assert r.clients == 2
    assert r.measured == 80
    assert r.on_time == 80
    assert r.late == 0
    assert r.lost == 0
    assert r.latency_ms is not None
    assert r.latency_ms.p50 == pytest.approx(10.0, abs=1e-6)
    assert r.backlog_growth_ms == pytest.approx(0.0, abs=1e-6)


def test_summarize_a_missing_reply_is_lost() -> None:
    recs: list[FrameRecord] = records(clients=1, frames=1)
    lost: FrameRecord = FrameRecord(0, 0, recs[0].scheduled, None, False)
    r: StepResult = summarize(1, [lost], 0, [], WINDOW)
    assert r.lost == 1
    assert "lost" in r.reasons
    assert r.verdict == "fail"
    assert r.lost_fraction == 1.0


def test_summarize_growing_delay_is_backlog_even_when_p95_is_fine() -> None:
    n: int = 240  # a 30 s window
    recs: list[FrameRecord] = []
    i: int
    for i in range(n):
        scheduled: float = T0 + i * HOP_SECONDS
        recs.append(FrameRecord(0, i, scheduled, scheduled + 0.010 + 0.100 * i / (n - 1), False))
    r: StepResult = summarize(1, recs, 0, [], (T0, T0 + 30.0))
    assert r.latency_ms is not None
    assert r.latency_ms.p95 <= 125.0
    assert r.backlog_growth_ms == pytest.approx(66.7, abs=2.0)
    assert r.reasons == ["backlog"]
    assert r.verdict == "fail"


def test_summarize_slow_replies_are_late() -> None:
    r: StepResult = summarize(1, records(clients=1, delay=0.200), 0, [], WINDOW)
    assert "p95" in r.reasons
    assert r.late == r.measured == 40
    assert r.on_time == 0
    assert r.late_fraction == 1.0
    assert r.verdict == "fail"


def test_summarize_connection_errors_fail() -> None:
    r: StepResult = summarize(1, records(clients=1), 1, [], WINDOW)
    assert "connections" in r.reasons
    assert r.connection_errors == 1
    assert r.verdict == "fail"


def test_summarize_error_reply_fails() -> None:
    recs: list[FrameRecord] = records(clients=1)
    first: FrameRecord = recs[0]
    recs[0] = FrameRecord(first.client, first.index, first.scheduled, first.replied, True)
    r: StepResult = summarize(1, recs, 0, [], WINDOW)
    assert r.error_replies == 1
    assert "errors" in r.reasons
    assert r.verdict == "fail"
    assert r.measured == r.on_time + r.late + r.error_replies + r.lost


def test_summarize_counts_only_frames_inside_the_window() -> None:
    start: float = T0 + 8 * HOP_SECONDS
    end: float = T0 + 32 * HOP_SECONDS
    r: StepResult = summarize(2, records(), 0, [], (start, end))
    assert r.measured == 2 * 24  # start is inside, end is outside


def test_summarize_a_slow_generator_is_invalid_not_fail() -> None:
    r: StepResult = summarize(2, records(), 0, [0.05] * 20, WINDOW)
    assert r.verdict == "invalid"
    assert r.reasons == ["client_lag"]
    assert r.client_loop_lag_p95_ms == pytest.approx(50.0, abs=1e-6)


def test_summarize_nothing_measured_is_no_answers() -> None:
    r: StepResult = summarize(1, [], 0, [], WINDOW)
    assert "no_answers" in r.reasons
    assert r.measured == 0
    assert r.lost_fraction == 1.0
    assert r.late_fraction == 0.0
    assert r.latency_ms is None
    assert r.verdict == "fail"


def test_summarize_without_client_lag_samples_has_no_lag_figure() -> None:
    r: StepResult = summarize(1, records(clients=1), 0, [], WINDOW)
    assert r.client_loop_lag_p95_ms is None
    assert "client_lag" not in r.reasons


def test_summarize_obeys_a_custom_criterion() -> None:
    strict: Criterion = Criterion(p95_max=0.005)
    r: StepResult = summarize(1, records(clients=1), 0, [], WINDOW, strict)
    assert "p95" in r.reasons


def step(clients: int, verdict: Verdict) -> StepResult:
    return StepResult(
        clients=clients,
        measured=0,
        on_time=0,
        late=0,
        error_replies=0,
        lost=0,
        connection_errors=0,
        latency_ms=None,
        late_fraction=0.0,
        lost_fraction=1.0,
        backlog_growth_ms=None,
        client_loop_lag_p95_ms=None,
        verdict=verdict,
        reasons=[],
        server=None,
    )


def test_ceiling_stops_at_the_first_step_that_is_not_pass() -> None:
    steps: Sequence[StepResult] = [step(1, "pass"), step(2, "pass"), step(4, "fail"), step(8, "pass")]
    assert ceiling(steps) == (2, "fail")
    assert ceiling([step(1, "pass"), step(2, "invalid")]) == (1, "invalid")
    assert ceiling([step(1, "pass"), step(2, "pass")]) == (2, None)
    assert ceiling([step(1, "fail")]) == (0, "fail")


def exposition(cpu: float | None, frames: float, inference_sum: float, inference_count: float) -> str:
    lines: list[str] = []
    if cpu is not None:
        lines += ["# TYPE process_cpu_seconds_total counter", f"process_cpu_seconds_total {cpu}"]
    lines += [
        "# TYPE chordfinder_frames_total counter",
        f"chordfinder_frames_total {frames}",
        "# TYPE chordfinder_stage_seconds histogram",
        f'chordfinder_stage_seconds_bucket{{stage="inference",le="+Inf"}} {inference_count}',
        f'chordfinder_stage_seconds_sum{{stage="inference"}} {inference_sum}',
        f'chordfinder_stage_seconds_count{{stage="inference"}} {inference_count}',
        'chordfinder_stage_seconds_bucket{stage="resample",le="+Inf"} 7.0',
        'chordfinder_stage_seconds_sum{stage="resample"} 0.7',
        'chordfinder_stage_seconds_count{stage="resample"} 7.0',
    ]
    return "\n".join(lines) + "\n"


def test_server_stats_takes_differences_between_scrapes() -> None:
    before: str = exposition(10.0, 100.0, 0.5, 50.0)
    after: str = exposition(12.0, 110.0, 0.510, 60.0)
    s: ServerStats = server_stats(before, after, 4.0)
    assert s.cpu_cores == pytest.approx(0.5)
    assert s.frames == 10
    assert s.stage_ms["inference"] == pytest.approx(1.0)
    assert s.stage_ms["resample"] is None  # no growth
    assert set(s.stage_ms) == {"resample", "features", "inference"}
    assert s.stage_ms["features"] is None  # metric absent
    assert s.frame_ms is None
    assert s.threadpool_wait_ms is None
    assert s.stream_lag_ms is None
    assert s.loop_lag_ms is None
    assert s.silent == 0


def test_server_stats_without_process_metric_has_no_cpu() -> None:
    s: ServerStats = server_stats(exposition(None, 0.0, 0.0, 0.0), exposition(None, 5.0, 0.01, 10.0), 1.0)
    assert s.cpu_cores is None
    assert s.frames == 5
    assert s.stage_ms["inference"] == pytest.approx(1.0)
