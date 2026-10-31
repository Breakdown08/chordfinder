"""End-to-end: Chromium fake microphone -> Godot web build -> /ws/recognize -> chord on screen.

    godot --headless --path ../frontend --export-release Web build/web/index.html
    uv run --group e2e pytest -m e2e -s

The client prints "status: ..." and "chord: <stable chord>" to the browser console; the tests read it.
Chromium is the system one (CHORDFINDER_CHROMIUM or `chromium` on PATH): Playwright downloads nothing.
"""

import os
import shutil
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import uvicorn
from scipy.io import wavfile

from chordfinder.audio import Audio
from chordfinder.evaluate import load
from chordfinder.server import WEB, app
from tests.conftest import dataset_file, needs_dataset, needs_model, sine
from tests.test_server import CHORDS, first_file

pytestmark: list[pytest.MarkDecorator] = [pytest.mark.e2e, needs_model]
RATE: int = 48000
LEAD_SECONDS: float = 2.0  # silence before the strum: the page settles, the client must show nothing
LISTEN_SECONDS: float = 5.0
os.environ.setdefault("PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS", "1")  # NixOS: its ldd check misfires
CHROMIUM: str | None = os.environ.get("CHORDFINDER_CHROMIUM") or shutil.which("chromium")

if not (WEB / "index.wasm").is_file() or CHROMIUM is None:
    pytest.skip("needs the exported web build and Chromium", allow_module_level=True)


@pytest.fixture(scope="module")
def base() -> Iterator[str]:
    """The real app (model, web build) on a free port, in this process."""
    s: socket.socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
    server: uvicorn.Server = uvicorn.Server(uvicorn.Config(app, port=port, log_level="warning"))
    thread: threading.Thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    yield f"http://localhost:{port}"
    server.should_exit = True
    thread.join(5)


def write_wav(path: Path, y: Audio) -> Path:
    lead: Audio = np.zeros(int(RATE * LEAD_SECONDS), np.float32)
    wavfile.write(path, RATE, (np.concatenate((lead, y)) * 32767).astype(np.int16))
    return path


def console(base: str, wav: Path, seconds: float, deny: bool = False) -> list[str]:
    """Open the client, click to start the microphone, listen `seconds`, return the console lines."""
    from playwright.sync_api import Browser, ConsoleMessage, Page, Playwright, sync_playwright

    lines: list[str] = []
    p: Playwright
    with sync_playwright() as p:
        browser: Browser = p.chromium.launch(
            executable_path=CHROMIUM,
            args=[
                # deny: a fake device exists, so the refusal itself is under test
                "--deny-permission-prompts" if deny else "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-audio-capture={wav}",
                "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader",
            ],
        )
        page: Page = browser.new_page(viewport={"width": 960, "height": 540})

        def on_console(m: ConsoleMessage) -> None:
            lines.append(m.text)

        page.on("console", on_console)
        page.goto(base)
        page.wait_for_function("() => !document.querySelector('#status')", timeout=60000)  # engine loaded
        page.wait_for_timeout(500)
        page.mouse.click(480, 270)
        page.wait_for_timeout(seconds * 1000)
        browser.close()
    return lines


def chords(lines: list[str]) -> list[str]:
    return [line.removeprefix("chord: ") for line in lines if line.startswith("chord: ")]


@needs_dataset
def test_recognition(base: str, tmp_path: Path) -> None:
    """Each test-set recording after 2 s of silence: the first chord the client shows is the right one."""
    shown: dict[str, list[str]] = {}
    chord: str
    for chord in CHORDS:
        wav: Path = write_wav(tmp_path / f"{chord}.wav", load(dataset_file(first_file(chord)), RATE))
        shown[chord] = chords(console(base, wav, LEAD_SECONDS + LISTEN_SECONDS))
    hits: int = sum(bool(shown[c]) and shown[c][0] == c for c in CHORDS)
    print(f"\n{hits}/8 {shown}")
    assert hits >= 7


def test_silence(base: str, tmp_path: Path) -> None:
    """A quiet room (RMS ~0.002, as in the guitar check): no chord and no false microphone warning."""
    room: Audio = (np.random.default_rng(0).normal(0, 0.002, RATE * 4)).astype(np.float32)
    lines: list[str] = console(base, write_wav(tmp_path / "room.wav", room), 6)
    statuses: list[str] = [line for line in lines if line.startswith("status: ")]
    print(f"\n{statuses}")
    assert statuses[-1] == "status: Играй аккорд"  # connected, sent the header, hears the mic
    assert not chords(lines)


def test_mic_denied(base: str, tmp_path: Path) -> None:
    """Godot only logs the refusal and captures zeros; the client says why."""
    wav: Path = write_wav(tmp_path / "sine.wav", sine(440, 1, RATE))
    lines: list[str] = console(base, wav, 5, deny=True)
    statuses: list[str] = [line for line in lines if line.startswith("status: ")]
    print(f"\n{statuses}")
    assert "Доступ к микрофону запрещён" in statuses[-1]
