# chordfinder

[![CI](https://github.com/Breakdown08/chordfinder/actions/workflows/ci.yml/badge.svg)](https://github.com/Breakdown08/chordfinder/actions/workflows/ci.yml)

Real-time guitar chord recognition in the browser: a CNN on log-mel spectrograms behind a FastAPI WebSocket server,
and a Godot 4.5 web client that listens to the microphone. The goal is a game where you play real chords on a real
guitar to cross a bridge; for now the client is a listener: microphone → server → chord on screen.

The model comes from my master's thesis (January 2025), where it ran behind a Django app. This repository rebuilds it
around the same weights: ONNX Runtime instead of TensorFlow, a streaming protocol instead of JSON requests, tests and
CI instead of none.

**Status:** work in progress. The recognition server and the client work end to end; the client shows what the model
hears, window by window (below). Retraining is next, then the game.

## What it recognises

Eight chords: Am, Bb, Bdim, C, Dm, Em, F, G. Model input is one second of audio at 16 kHz, 64 mel bands
(`backend/models/meta.json`). Trained on the Kaggle dataset
[guitar-chords-v3](https://www.kaggle.com/datasets/fabianavinci/guitar-chords-v3) (20 instruments plus synthetic).

## Measured

Same scenario before and after the rebuild, reproducible with the tools in the last column.

| | Thesis code | Now | How |
|---|---|---|---|
| Accuracy, test split (320 files) | 95.9% | 95.9%; ONNX matches Keras on 320/320 labels | `chordfinder eval`, `tests/test_parity.py` |
| Inference per window | 53 ms (TensorFlow) | 1.2 ms (ONNX Runtime) | `tools/bench.py` |
| Cold start to first answer | 2.3 s | 0.86 s (first run after install: ~9 s, see below) | `tools/bench.py` |
| Strum to chord, median | 0.31 s, but a wrong chord flashes in 53 of 80 recordings | 0.61 s, wrong chord in 1 of 80 | `tools/latency_cmp.py` |
| Environment | 2.5 GB venv (TensorFlow 1.7 GB) | 539 MB, no TensorFlow at runtime | size of `.venv` |
| Tests | 0 | 62 backend, 34 client checks, 4 browser e2e (2 of them in CI against the Docker image; the others need the dataset) | CI |

Cold start counts the first answer, and librosa compiles its functions through numba on first use: right
after installing, with no numba cache yet, the first answer takes ~9 s; later starts reuse the cache. The Docker
image compiles them at build time.

The extra 0.3 s is the price of smoothing: a chord is shown after three agreeing windows (125 ms step). Two windows
would be faster but bring the false chords back; speed has to come from retraining, not from tuning.

## How it works

```
microphone ─► Godot web client ─► /ws/recognize ─► resample to 16 kHz ─► 1 s windows ─► CNN (ONNX) ─► vote ─► chord
              Float32, 125 ms      (FastAPI)        (rate from client)     every 125 ms                 3 in a row
```

The client draws every window before the vote: one column per chord with its raw probability, a dashed line at the
vote's confidence threshold, input level on the right with a tick at the silence floor. Amber is the window's best
guess, green is the chord the vote let through. The start of a strum is where the model is wrong: on one Em test
recording it says Bb (0.75), then Bdim (0.87), and the vote keeps both off the screen.

To measure on a real guitar, open the client in measuring mode: it asks for each chord in turn, counts three strums and
keeps the raw audio. `tools/takes.py` scores the recordings offline, so the same strums can compare two models. The
server only records when `CHORDFINDER_SESSIONS` is set.

```bash
(cd backend && CHORDFINDER_SESSIONS=manual/sessions uv run uvicorn chordfinder.server:app --port 8000)
# http://localhost:8000/?measure, Space starts each take
(cd backend && uv run python tools/takes.py manual/sessions/<folder>)
```

- `backend/` — Python 3.12, uv. `src/chordfinder/`: `audio` and `features` (log-mel), `model` (ONNX Runtime),
  `stream` (window buffer and voting), `server` (FastAPI; also serves the web build at `/`), `weights`, `evaluate`,
  `cli`. Fully typed: mypy `--strict`, ruff, and a test that requires annotations on variables too.
- `frontend/` — Godot 4.5.1, exported to Web (single-threaded build). Strict GDScript typing: untyped or unsafe code
  does not compile.
- Model weights are not in git. They live in the GitHub Release
  [`weights-v1`](https://github.com/Breakdown08/chordfinder/releases/tag/weights-v1); `chordfinder weights`
  downloads them and checks SHA-256 against `backend/models/weights.json`.

## Run

With Docker — the image builds the web client, installs the server and downloads the weights (SHA-256 checked):

```sh
docker build -t chordfinder .
docker run --rm -p 8000:8000 chordfinder
```

Open http://localhost:8000/, allow the microphone, play a chord. The image is 255 MB to download: Python
runtime dependencies only, no TensorFlow, no Godot.

Without Docker: needs [uv](https://docs.astral.sh/uv/) and [Godot 4.5.1](https://godotengine.org/) with web
export templates.

```sh
(cd backend && uv sync && uv run chordfinder weights)
(cd frontend && mkdir -p build/web && godot --headless --export-release Web build/web/index.html)
(cd backend && uv run uvicorn chordfinder.server:app --port 8000)
```

Open http://localhost:8000/, allow the microphone, play a chord.

## Test

```sh
(cd backend && uv run ruff check && uv run mypy && uv run pytest)
godot --headless --path frontend -s tests/test_client.gd   # client: wire format, replies, strict typing
(cd backend && uv run pytest -m e2e -s)                     # Chromium with a fake microphone → web build → server
(cd backend && CHORDFINDER_URL=http://localhost:8000 uv run pytest -m e2e -s)   # the same against a running container
```

CI builds the image, starts it and runs the browser tests against it.

Accuracy, parity and latency tests need the dataset and skip without it. Its licence is unknown, so it is not
included: download it from Kaggle and point `CHORDFINDER_DATASET` at its `data/` directory
(default `~/datasets/guitar-chords-v3/data`).
