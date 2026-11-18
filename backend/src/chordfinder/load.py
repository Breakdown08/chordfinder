"""Load generator: N simulated microphones against /ws/recognize."""

import asyncio
import http.client
import json
import math
import os
import platform
import time
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import SplitResult, urlsplit, urlunsplit

import numpy as np
import numpy.typing as npt
from prometheus_client.metrics_core import Metric
from prometheus_client.parser import text_string_to_metric_families
from prometheus_client.samples import Sample
from pydantic import BaseModel
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, WebSocketException

from chordfinder import evaluate
from chordfinder.audio import Audio
from chordfinder.stream import HOP_SECONDS

DEFAULT_URL: str = "ws://localhost:8000/ws/recognize"
DEFAULT_STEPS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
STAGES: tuple[str, ...] = ("resample", "features", "inference")
LOOP_PROBE: float = 0.05  # seconds between probes of the generator's own event loop
CONNECT_TIMEOUT: float = 10.0


def metrics_url(ws_url: str) -> str:
    parts: SplitResult = urlsplit(ws_url)
    scheme: str = {"ws": "http", "wss": "https"}.get(parts.scheme, parts.scheme)
    return urlunsplit((scheme, parts.netloc, "/metrics", "", ""))


def synthetic_signal(rate: int, seconds: float, seed: int) -> Audio:
    """Three sines that change every ~2 s: loud everywhere, so every window runs every stage."""
    rng: np.random.Generator = np.random.default_rng(seed)
    n: int = int(rate * seconds)
    t: npt.NDArray[np.float64] = np.arange(n, dtype=np.float64) / rate
    segment: int = max(1, int(rate * 2.0))
    out: npt.NDArray[np.float64] = np.zeros(n, np.float64)
    start: int
    for start in range(0, n, segment):
        stop: int = min(start + segment, n)
        hz: npt.NDArray[np.float64] = rng.uniform(100.0, 1000.0, 3)
        phase: npt.NDArray[np.float64] = rng.uniform(0.0, 2 * math.pi, 3)
        k: int
        for k in range(3):
            out[start:stop] += 0.25 * np.sin(2 * math.pi * hz[k] * t[start:stop] + phase[k])
    signal: Audio = out.astype(np.float32)
    return signal


def dataset_signal(root: Path, rate: int) -> Audio:
    parts: list[Audio] = [evaluate.load(p, rate) for p in evaluate.dataset_files(root)]
    signal: Audio = np.concatenate(parts).astype(np.float32, copy=False)
    return signal


def frame(signal: Audio, hop: int, offset: int, index: int) -> Audio:
    start: int = offset + index * hop
    out: Audio = signal[(start + np.arange(hop)) % len(signal)].astype(np.float32, copy=False)
    return out


@dataclass(frozen=True)
class FrameRecord:
    client: int
    index: int
    scheduled: float
    replied: float | None
    error: bool


@dataclass(frozen=True)
class Criterion:
    p95_max: float = HOP_SECONDS
    max_lost_fraction: float = 0.0
    max_backlog_growth: float = 0.025
    max_client_lag_p95: float = 0.020


DEFAULT_CRITERION: Criterion = Criterion()  # ruff B008: no call in a default argument


class LatencyMs(BaseModel):
    p50: float
    p95: float
    p99: float
    max: float


class ServerStats(BaseModel):
    cpu_cores: float | None
    frames: int
    silent: int
    frame_ms: float | None
    threadpool_wait_ms: float | None
    stream_lag_ms: float | None
    loop_lag_ms: float | None
    stage_ms: dict[str, float | None]


class StepResult(BaseModel):
    clients: int
    measured: int
    on_time: int
    late: int
    error_replies: int
    lost: int
    connection_errors: int
    latency_ms: LatencyMs | None
    late_fraction: float
    lost_fraction: float
    backlog_growth_ms: float | None
    client_loop_lag_p95_ms: float | None
    verdict: Literal["pass", "fail", "invalid"]
    reasons: list[str]
    server: ServerStats | None


class LoadReport(BaseModel):
    url: str
    source: Literal["synthetic", "dataset"]
    sample_rate: int
    hop_seconds: float
    duration: float
    warmup: float
    grace: float
    criterion: dict[str, float]
    host: dict[str, str | int]
    steps: list[StepResult]
    ceiling: int
    stopped_by: Literal["fail", "invalid"] | None


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, np.float64), q))


def median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, np.float64)))


def backlog_growth(answered: Sequence[FrameRecord], span: tuple[float, float]) -> float | None:
    """Median delay in the last third of the span minus that in the first third, seconds."""
    first_end: float = span[0] + (span[1] - span[0]) / 3
    last_start: float = span[0] + 2 * (span[1] - span[0]) / 3
    early: list[float] = []
    late: list[float] = []
    r: FrameRecord
    for r in sorted(answered, key=lambda x: x.scheduled):
        if r.replied is None:
            continue
        if r.scheduled < first_end:
            early.append(r.replied - r.scheduled)
        elif r.scheduled >= last_start:
            late.append(r.replied - r.scheduled)
    if not early or not late:
        return None
    return median(late) - median(early)


def summarize(
    clients: int,
    records: Sequence[FrameRecord],
    connection_errors: int,
    client_loop_lags: Sequence[float],
    window: tuple[float, float],
    criterion: Criterion = DEFAULT_CRITERION,
) -> StepResult:
    measured: list[FrameRecord] = [r for r in records if window[0] <= r.scheduled < window[1]]
    errors: list[FrameRecord] = [r for r in measured if r.replied is not None and r.error]
    answered: list[FrameRecord] = [r for r in measured if r.replied is not None and not r.error]
    delays: list[float] = [r.replied - r.scheduled for r in answered if r.replied is not None]
    late: int = sum(1 for d in delays if d > HOP_SECONDS)
    lost: int = len(measured) - len(answered) - len(errors)
    latency: LatencyMs | None = None
    if delays:
        latency = LatencyMs(
            p50=percentile(delays, 50) * 1000,
            p95=percentile(delays, 95) * 1000,
            p99=percentile(delays, 99) * 1000,
            max=max(delays) * 1000,
        )
    lost_fraction: float = lost / len(measured) if measured else 1.0
    growth: float | None = None
    if measured:
        span: tuple[float, float] = (min(r.scheduled for r in measured), max(r.scheduled for r in measured))
        growth = backlog_growth(answered, span)
    lag_p95: float | None = percentile(client_loop_lags, 95) if client_loop_lags else None

    reasons: list[str] = []
    verdict: Literal["pass", "fail", "invalid"]
    if lag_p95 is not None and lag_p95 > criterion.max_client_lag_p95:
        reasons = ["client_lag"]
        verdict = "invalid"
    else:
        if connection_errors > 0:
            reasons.append("connections")
        if errors:
            reasons.append("errors")
        if not answered:
            reasons.append("no_answers")
        if latency is not None and latency.p95 / 1000 > criterion.p95_max:
            reasons.append("p95")
        if lost_fraction > criterion.max_lost_fraction:
            reasons.append("lost")
        if growth is not None and growth > criterion.max_backlog_growth:
            reasons.append("backlog")
        verdict = "fail" if reasons else "pass"
    return StepResult(
        clients=clients,
        measured=len(measured),
        on_time=len(answered) - late,
        late=late,
        error_replies=len(errors),
        lost=lost,
        connection_errors=connection_errors,
        latency_ms=latency,
        late_fraction=late / len(measured) if measured else 0.0,
        lost_fraction=lost_fraction,
        backlog_growth_ms=None if growth is None else growth * 1000,
        client_loop_lag_p95_ms=None if lag_p95 is None else lag_p95 * 1000,
        verdict=verdict,
        reasons=reasons,
        server=None,
    )


def ceiling(steps: Sequence[StepResult]) -> tuple[int, Literal["fail", "invalid"] | None]:
    best: int = 0
    step: StepResult
    for step in steps:
        if step.verdict == "fail":
            return best, "fail"
        if step.verdict == "invalid":
            return best, "invalid"
        best = step.clients
    return best, None


type SeriesKey = tuple[str, str | None]


def scraped(text: str) -> dict[SeriesKey, float]:
    """Exposition text -> value by (sample name, stage label)."""
    out: dict[SeriesKey, float] = {}
    family: Metric
    for family in text_string_to_metric_families(text):
        sample: Sample
        for sample in family.samples:
            out[(sample.name, sample.labels.get("stage"))] = sample.value
    return out


def server_stats(before: str, after: str, seconds: float) -> ServerStats:
    a: dict[SeriesKey, float] = scraped(before)
    b: dict[SeriesKey, float] = scraped(after)

    def grew(name: str, stage: str | None = None) -> float:
        return b.get((name, stage), 0.0) - a.get((name, stage), 0.0)

    def mean_ms(histogram: str, stage: str | None = None) -> float | None:
        count: float = grew(f"{histogram}_count", stage)
        return grew(f"{histogram}_sum", stage) / count * 1000 if count > 0 else None

    cpu: float | None = None
    # The server's own counter (sums over processes) first; process_* is what an older server has.
    name: str
    for name in ("chordfinder_cpu_seconds_total", "process_cpu_seconds_total"):
        if (name, None) in a and (name, None) in b:
            cpu = grew(name) / seconds if seconds > 0 else None
            break
    return ServerStats(
        cpu_cores=cpu,
        frames=round(grew("chordfinder_frames_total")),
        silent=round(grew("chordfinder_silent_windows_total")),
        frame_ms=mean_ms("chordfinder_frame_seconds"),
        threadpool_wait_ms=mean_ms("chordfinder_threadpool_wait_seconds"),
        stream_lag_ms=mean_ms("chordfinder_stream_lag_seconds"),
        loop_lag_ms=mean_ms("chordfinder_event_loop_lag_seconds"),
        stage_ms={s: mean_ms("chordfinder_stage_seconds", s) for s in STAGES},
    )


@dataclass(frozen=True)
class LoadConfig:
    url: str = DEFAULT_URL
    metrics: str | None = None
    sample_rate: int = 48000
    duration: float = 30.0
    warmup: float = 3.0
    grace: float = 2.0
    pause: float = 2.0
    seed: int = 0
    criterion: Criterion = DEFAULT_CRITERION


def fetch_text(url: str) -> str:
    response: http.client.HTTPResponse
    with urllib.request.urlopen(url, timeout=5.0) as response:
        text: str = response.read().decode()
    return text


async def scrape(url: str | None) -> str | None:
    """The /metrics text, or None without an address or when the server does not answer."""
    if url is None:
        return None
    try:
        return await asyncio.to_thread(fetch_text, url)
    except (OSError, ValueError):
        return None


@dataclass
class Client:
    """One simulated microphone: what it should send and what came back."""

    number: int
    frames: int
    scheduled: list[float] = field(default_factory=list)
    replies: dict[int, float] = field(default_factory=dict)
    errors: set[int] = field(default_factory=set)
    connection_error: bool = False
    finishing: bool = False
    connection: ClientConnection | None = None
    next_unanswered: int = 0

    def answered(self, index: int, at: float, error: bool) -> None:
        if index in self.replies or not 0 <= index < self.frames:
            return
        self.replies[index] = at
        if error:
            self.errors.add(index)

    def oldest_unanswered(self) -> int:
        while self.next_unanswered in self.replies:
            self.next_unanswered += 1
        return self.next_unanswered


async def open_client(config: LoadConfig, client: Client) -> None:
    try:
        # No compression and no keepalive pings: they would cost the generator CPU it needs for timing.
        ws: ClientConnection = await connect(
            config.url, compression=None, ping_interval=None, open_timeout=CONNECT_TIMEOUT
        )
        await ws.send(json.dumps({"sample_rate": config.sample_rate}))
        client.connection = ws
    except (WebSocketException, OSError, TimeoutError):
        client.connection_error = True


async def read_replies(client: Client, ws: ClientConnection, hop: int) -> None:
    try:
        while True:
            message: str | bytes = await ws.recv()
            now: float = time.monotonic()
            reply: object = json.loads(message)
            if not isinstance(reply, dict):
                continue
            if "error" in reply:
                client.answered(client.oldest_unanswered(), now, True)
            elif isinstance(reply.get("samples"), int):
                client.answered(reply["samples"] // hop - 1, now, False)
    except ConnectionClosed:
        if not client.finishing:
            client.connection_error = True


async def send_frames(config: LoadConfig, client: Client, ws: ClientConnection, signal: Audio) -> None:
    hop: int = int(config.sample_rate * HOP_SECONDS)
    offset: int = int(np.random.default_rng(config.seed + client.number).integers(0, len(signal)))
    index: int
    try:
        for index in range(client.frames):
            delay: float = client.scheduled[index] - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            # A send blocked by the server's backpressure delays the next ones, none is skipped.
            await ws.send(frame(signal, hop, offset, index).astype("<f4").tobytes())
    except ConnectionClosed:
        if not client.finishing:
            client.connection_error = True


async def probe_loop(lags: list[float]) -> None:
    """The generator's own event loop: how late does a short sleep wake up?"""
    while True:
        start: float = time.monotonic()
        await asyncio.sleep(LOOP_PROBE)
        lags.append(max(0.0, time.monotonic() - start - LOOP_PROBE))


async def run_step(config: LoadConfig, clients: int, signal: Audio) -> StepResult:
    before: str | None = await scrape(config.metrics)
    frames: int = int(config.duration / HOP_SECONDS)
    hop: int = int(config.sample_rate * HOP_SECONDS)
    crowd: list[Client] = [Client(n, frames) for n in range(clients)]
    await asyncio.gather(*(open_client(config, c) for c in crowd))

    step_start: float = time.monotonic()
    c: Client
    for c in crowd:
        phase: float = float(np.random.default_rng(config.seed + c.number).uniform(0.0, HOP_SECONDS))
        c.scheduled = [step_start + phase + i * HOP_SECONDS for i in range(frames)]
    lags: list[float] = []
    prober: asyncio.Task[None] = asyncio.create_task(probe_loop(lags))
    open_ws: list[tuple[Client, ClientConnection]] = [
        (c, c.connection) for c in crowd if c.connection is not None
    ]
    readers: list[asyncio.Task[None]] = [asyncio.create_task(read_replies(c, ws, hop)) for c, ws in open_ws]
    await asyncio.gather(*(send_frames(config, c, ws, signal) for c, ws in open_ws))

    deadline: float = time.monotonic() + config.grace
    while time.monotonic() < deadline and any(
        len(c.replies) < c.frames and not c.connection_error for c in crowd if c.connection is not None
    ):
        await asyncio.sleep(0.02)
    prober.cancel()
    for c in crowd:
        c.finishing = True
    await asyncio.gather(*(ws.close() for _, ws in open_ws), return_exceptions=True)
    await asyncio.gather(*readers, return_exceptions=True)

    after: str | None = await scrape(config.metrics) if before is not None else None
    end: float = time.monotonic()
    records: list[FrameRecord] = [
        FrameRecord(c.number, i, c.scheduled[i], c.replies.get(i), i in c.errors)
        for c in crowd
        for i in range(frames)
    ]
    result: StepResult = summarize(
        clients,
        records,
        sum(1 for c in crowd if c.connection_error),
        lags,
        (step_start + config.warmup, step_start + config.duration),
        config.criterion,
    )
    if before is not None and after is not None:
        try:
            result.server = server_stats(before, after, end - step_start)
        except ValueError:  # not Prometheus text
            result.server = None
    return result


def host_info() -> dict[str, str | int]:
    cpu: str = platform.processor() or platform.machine()
    try:
        line: str
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"cpu": cpu, "logical_cpus": os.cpu_count() or 0, "python": platform.python_version()}


async def run(
    config: LoadConfig,
    steps: Sequence[int],
    signal: Audio,
    source: Literal["synthetic", "dataset"],
    on_step: Callable[[StepResult], None] | None = None,
) -> LoadReport:
    """Steps in order, stopping after the first one that is not a pass; on_step sees each result."""
    results: list[StepResult] = []
    n: int
    for n in steps:
        if results:
            await asyncio.sleep(config.pause)
        result: StepResult = await run_step(config, n, signal)
        results.append(result)
        if on_step is not None:
            on_step(result)
        if result.verdict != "pass":
            break
    top: int
    stopped: Literal["fail", "invalid"] | None
    top, stopped = ceiling(results)
    crit: Criterion = config.criterion
    return LoadReport(
        url=config.url,
        source=source,
        sample_rate=config.sample_rate,
        hop_seconds=HOP_SECONDS,
        duration=config.duration,
        warmup=config.warmup,
        grace=config.grace,
        criterion={
            "p95_max_ms": crit.p95_max * 1000,
            "max_lost_fraction": crit.max_lost_fraction,
            "max_backlog_growth_ms": crit.max_backlog_growth * 1000,
            "max_client_lag_p95_ms": crit.max_client_lag_p95 * 1000,
        },
        host=host_info(),
        steps=results,
        ceiling=top,
        stopped_by=stopped,
    )


async def preflight(url: str, rate: int, signal: Audio) -> str | None:
    """One connection, the header, one frame, one answer within 10 s: None, or why it did not work."""
    hop: int = int(rate * HOP_SECONDS)
    try:
        async with asyncio.timeout(10.0):
            ws: ClientConnection
            async with connect(url, compression=None, open_timeout=10.0) as ws:
                await ws.send(json.dumps({"sample_rate": rate}))
                await ws.send(frame(signal, hop, 0, 0).astype("<f4").tobytes())
                reply: object = json.loads(await ws.recv())
                if isinstance(reply, dict) and "error" in reply:
                    return f"the server answered {reply['error']}"
    except TimeoutError:
        return "no answer within 10 s"
    except (WebSocketException, OSError, ValueError) as e:
        return str(e) or type(e).__name__
    return None


def cell(value: float | None, fmt: str = ".1f") -> str:
    return "-" if value is None else format(value, fmt)


HEADER: str = (
    f"{'clients':>7} {'p50':>6} {'p95':>6} {'p99':>6} {'late':>6} {'lost':>6} {'backlog':>8} {'cpu':>5} "
    f"{'resample':>8} {'features':>8} {'infer':>6} {'pool':>6} {'loop':>6}  verdict"
)


def format_step(r: StepResult) -> str:
    lat: LatencyMs | None = r.latency_ms
    srv: ServerStats | None = r.server
    verdict: str = r.verdict + (" " + ",".join(r.reasons) if r.reasons else "")
    return (
        f"{r.clients:7d} {cell(lat.p50 if lat else None):>6} {cell(lat.p95 if lat else None):>6} "
        f"{cell(lat.p99 if lat else None):>6} {r.late_fraction:6.1%} {r.lost_fraction:6.1%} "
        f"{cell(r.backlog_growth_ms):>8} {cell(srv.cpu_cores if srv else None, '.2f'):>5} "
        f"{cell(srv.stage_ms['resample'] if srv else None, '.2f'):>8} "
        f"{cell(srv.stage_ms['features'] if srv else None, '.2f'):>8} "
        f"{cell(srv.stage_ms['inference'] if srv else None, '.2f'):>6} "
        f"{cell(srv.threadpool_wait_ms if srv else None):>6} {cell(srv.loop_lag_ms if srv else None):>6}  "
        f"{verdict}"
    )


def format_ceiling(report: LoadReport) -> str:
    if report.stopped_by is None:
        return f"ceiling >= {report.ceiling} (all steps passed)"
    return f"ceiling {report.ceiling} (stopped by {report.stopped_by})"
