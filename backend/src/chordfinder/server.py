"""HTTP and WebSocket API of the recognizer.

    uv run uvicorn chordfinder.server:app --port 8000

WebSocket /ws/recognize: first a text frame {"sample_rate": 48000}, then binary frames of mono
float32 little-endian samples at that rate (stream.HOP_SECONDS each, 125 ms). Every frame gets a
StreamReply about the last second of audio. POST /v1/predict answers for one clip.
CHORDFINDER_LOG=manual.jsonl appends every stream reply with a timestamp (manual check, tools/strums.py).
CHORDFINDER_SESSIONS=dir turns on measuring: text frames {"take": "Am"} and {"take": null} start and end
a labelled take, written to a folder under dir (chordfinder.measure). Off by default: a public server
must not keep anyone's audio.
Everything else is the Godot web build (frontend/build/web, or CHORDFINDER_WEB), when exported.
"""

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, TextIO

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.types import Message

from chordfinder import audio
from chordfinder.audio import Audio
from chordfinder.measure import Session
from chordfinder.model import ChordModel
from chordfinder.stream import Heard, Stabilizer, Stream, recognize

MAX_CLIP_SECONDS: float = 10.0
# Single-threaded Godot build: no SharedArrayBuffer, so no COOP/COEP headers are needed.
REPO: Path = Path(__file__).resolve().parents[3]
WEB: Path = Path(os.environ.get("CHORDFINDER_WEB", REPO / "frontend/build/web"))
LOG: Path | None = Path(os.environ["CHORDFINDER_LOG"]) if os.environ.get("CHORDFINDER_LOG") else None
SESSIONS: Path | None = (
    Path(os.environ["CHORDFINDER_SESSIONS"]) if os.environ.get("CHORDFINDER_SESSIONS") else None
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.model = ChordModel()  # load once at start, not inside the first request
    yield


app: FastAPI = FastAPI(title="Chord Finder", version="0.2.0", lifespan=lifespan)


class Recognition(BaseModel):
    chord: str | None = Field(description="None when the window is silence")
    confidence: float | None
    rms: float


class StreamReply(Recognition):
    stable: str | None = Field(description="chord several consecutive windows agree on")
    samples: int = Field(description="samples received so far")
    probabilities: dict[str, float] | None = Field(description="every chord; None when the window is silence")
    take: str | None = Field(default=None, description="chord being recorded in a measuring session")
    strums: int = Field(default=0, description="strums counted in that take")


class StreamError(BaseModel):
    error: str


class Clip(BaseModel):
    sample_rate: float = Field(ge=audio.MIN_RATE, le=audio.MAX_RATE)
    samples: list[float] = Field(min_length=1)


class ClipReply(Recognition):
    probabilities: dict[str, float] | None


def log_reply(reply: StreamReply) -> None:
    if LOG is None:
        return
    f: TextIO
    with LOG.open("a") as f:
        f.write(json.dumps({"ts": time.time(), **reply.model_dump()}) + "\n")


def take_command(text: str, classes: tuple[str, ...]) -> tuple[bool, str | None]:
    """Text frame -> (is it a take command, chord or None to end the take); others are ignored."""
    try:
        command: object = json.loads(text)
    except json.JSONDecodeError:
        return False, None
    if not isinstance(command, dict) or "take" not in command:
        return False, None
    chord: object = command["take"]
    if chord is None:
        return True, None
    if chord not in classes:
        raise audio.ProtocolError(f"unknown chord: {chord}")
    return True, str(chord)


def get_model(request: Request) -> ChordModel:
    model: ChordModel = request.app.state.model
    return model


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/predict")
async def predict(clip: Clip, model: Annotated[ChordModel, Depends(get_model)]) -> ClipReply:
    if len(clip.samples) > clip.sample_rate * MAX_CLIP_SECONDS:
        raise HTTPException(413, f"clip longer than {MAX_CLIP_SECONDS} s")
    y: Audio = np.asarray(clip.samples, np.float32)
    level: float = audio.rms(y)
    if level < audio.SILENCE_RMS:
        return ClipReply(chord=None, confidence=None, rms=level, probabilities=None)
    p: Audio = await asyncio.to_thread(model.probabilities, audio.resample(y, clip.sample_rate))
    i: int = int(np.argmax(p))
    classes: tuple[str, ...] = model.meta.classes
    return ClipReply(
        chord=classes[i],
        confidence=float(p[i]),
        rms=level,
        probabilities={c: float(v) for c, v in zip(classes, p, strict=True)},
    )


@app.websocket("/ws/recognize")
async def recognize_stream(ws: WebSocket) -> None:
    await ws.accept()
    model: ChordModel = ws.app.state.model
    try:
        rate: float = audio.parse_header(await ws.receive_text())
    except audio.ProtocolError as e:
        await ws.send_json(StreamError(error=str(e)).model_dump())
        await ws.close(code=1003)
        return
    stream: Stream = Stream(rate)
    stabilizer: Stabilizer = Stabilizer()
    session: Session | None = None
    try:
        while True:
            msg: Message = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            data: bytes | None = msg.get("bytes")
            if data is None:
                try:
                    is_take: bool
                    chord: str | None
                    is_take, chord = take_command(msg.get("text") or "", model.meta.classes)
                except audio.ProtocolError as e:
                    await ws.send_json(StreamError(error=str(e)).model_dump())
                    continue
                if not is_take:
                    continue  # other text frames after the header (client stats) are ignored
                if SESSIONS is None:
                    await ws.send_json(
                        StreamError(error="measuring is off: set CHORDFINDER_SESSIONS").model_dump()
                    )
                    continue
                session = session or Session(SESSIONS, rate)
                await asyncio.to_thread(session.stop)
                if chord is not None:
                    session.start(chord)
                continue
            try:
                chunk: Audio = audio.decode_frame(data, rate)
            except audio.ProtocolError as e:
                await ws.send_json(StreamError(error=str(e)).model_dump())
                continue
            stream.push(chunk)
            if session is not None:
                session.push(chunk)
            heard: Heard = await asyncio.to_thread(recognize, model, stream.window, rate)
            reply: StreamReply = StreamReply(
                chord=heard.chord,
                confidence=heard.confidence,
                rms=heard.rms,
                stable=stabilizer.push(heard.chord, heard.confidence),
                samples=stream.total,
                probabilities=heard.probabilities,
                take=session.chord if session else None,
                strums=len(session.onsets) if session else 0,
            )
            log_reply(reply)
            await ws.send_json(reply.model_dump())
    except WebSocketDisconnect:
        return


# Last: a mount at "/" would shadow the routes above.
if (WEB / "index.html").is_file():
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
