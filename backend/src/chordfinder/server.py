"""HTTP and WebSocket API of the recognizer.

    uv run uvicorn chordfinder.server:app --port 8000

WebSocket /ws/recognize: first a text frame {"sample_rate": 48000}, then binary frames of mono
float32 little-endian samples at that rate (stream.HOP_SECONDS each, 125 ms). Every frame gets a
StreamReply about the last second of audio. POST /v1/predict answers for one clip.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from starlette.types import Message

from chordfinder import audio
from chordfinder.audio import Audio
from chordfinder.model import ChordModel
from chordfinder.stream import Stabilizer, Stream, recognize

MAX_CLIP_SECONDS: float = 10.0


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


class StreamError(BaseModel):
    error: str


class Clip(BaseModel):
    sample_rate: float = Field(ge=audio.MIN_RATE, le=audio.MAX_RATE)
    samples: list[float] = Field(min_length=1)


class ClipReply(Recognition):
    probabilities: dict[str, float] | None


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
    try:
        while True:
            msg: Message = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            data: bytes | None = msg.get("bytes")
            if data is None:
                continue  # text frames after the header (client stats) are ignored
            try:
                chunk: Audio = audio.decode_frame(data, rate)
            except audio.ProtocolError as e:
                await ws.send_json(StreamError(error=str(e)).model_dump())
                continue
            stream.push(chunk)
            chord: str | None
            confidence: float | None
            level: float
            chord, confidence, level = await asyncio.to_thread(recognize, model, stream.window, rate)
            reply: StreamReply = StreamReply(
                chord=chord,
                confidence=confidence,
                rms=level,
                stable=stabilizer.push(chord, confidence),
                samples=stream.total,
            )
            await ws.send_json(reply.model_dump())
    except WebSocketDisconnect:
        return
