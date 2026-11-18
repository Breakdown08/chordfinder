"""The generator against a real uvicorn on 127.0.0.1 with the real model (checks 27-31 of the load task).

Short on purpose: N <= 2, a few seconds, no dataset. The real ceiling measurement is manual (README, Load).
"""

import asyncio
import json
import socket
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import NamedTuple

import pytest
import uvicorn

from chordfinder import stream
from chordfinder.audio import Audio
from chordfinder.cli import main
from chordfinder.load import LoadConfig, LoadReport, StepResult, ceiling, run_step, synthetic_signal
from chordfinder.model import ChordModel
from chordfinder.stream import Heard, Timings


class Live(NamedTuple):
    url: str
    metrics: str


def free_port() -> int:
    s: socket.socket = socket.socket()
    s.bind(("127.0.0.1", 0))
    port: int = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def live(model: ChordModel) -> Iterator[Live]:
    from chordfinder.server import app

    port: int = free_port()
    server: uvicorn.Server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread: threading.Thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline: float = time.monotonic() + 30.0
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.05)
    yield Live(f"ws://127.0.0.1:{port}/ws/recognize", f"http://127.0.0.1:{port}/metrics")
    server.should_exit = True
    thread.join(timeout=10.0)


def config(live: Live) -> LoadConfig:
    return LoadConfig(url=live.url, metrics=live.metrics, duration=3.0, warmup=0.5, grace=1.0)


def test_two_clients_pass_on_a_free_server(live: Live) -> None:
    result: StepResult = asyncio.run(run_step(config(live), 2, synthetic_signal(48000, 10, 0)))
    assert result.verdict == "pass", result.reasons
    assert result.lost == 0
    assert result.connection_errors == 0
    assert result.measured >= 2 * 18
    assert result.server is not None
    assert result.server.frames >= 2 * 20
    assert all(v is not None for v in result.server.stage_ms.values())


def test_a_slow_server_fails_on_p95(live: Live, monkeypatch: pytest.MonkeyPatch) -> None:
    real: Callable[[ChordModel, Audio, float], tuple[Heard, Timings | None]] = stream.recognize_timed

    def slow(model: ChordModel, y: Audio, sample_rate: float) -> tuple[Heard, Timings | None]:
        time.sleep(0.3)
        return real(model, y, sample_rate)

    monkeypatch.setattr("chordfinder.server.recognize_timed", slow)
    result: StepResult = asyncio.run(run_step(config(live), 1, synthetic_signal(48000, 10, 0)))
    assert result.verdict == "fail"
    assert "p95" in result.reasons


def test_cli_load_writes_a_consistent_report(live: Live, tmp_path: Path) -> None:
    out: Path = tmp_path / "load.json"
    code: int = main(
        [
            "load",
            "--url",
            live.url,
            "--steps",
            "1,2",
            "--duration",
            "2",
            "--warmup",
            "0.5",
            "--grace",
            "1",
            "--pause",
            "0.2",
            "--json",
            str(out),
        ]
    )
    assert code == 0
    report: LoadReport = LoadReport.model_validate(json.loads(out.read_text()))
    assert len(report.steps) >= 1
    assert (report.ceiling, report.stopped_by) == ceiling(report.steps)


def test_cli_load_reports_an_unreachable_server(capsys: pytest.CaptureFixture[str]) -> None:
    url: str = f"ws://127.0.0.1:{free_port()}/ws/recognize"
    assert main(["load", "--url", url, "--steps", "1"]) == 1
    assert "server not reachable" in capsys.readouterr().err


@pytest.mark.parametrize("steps", ["4,2", "0", "2,2", "1,x"])
def test_cli_load_rejects_bad_steps(steps: str, capsys: pytest.CaptureFixture[str]) -> None:
    e: pytest.ExceptionInfo[SystemExit]
    with pytest.raises(SystemExit) as e:
        main(["load", "--steps", steps])
    assert e.value.code == 2
    err: str = capsys.readouterr().err
    assert "--steps" in err
    assert "invalid choice" not in err  # `load` itself must be a known command
