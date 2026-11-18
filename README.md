# chordfinder

[![CI](https://github.com/Breakdown08/chordfinder/actions/workflows/ci.yml/badge.svg)](https://github.com/Breakdown08/chordfinder/actions/workflows/ci.yml)

Real-time guitar chord recognition in the browser: a CNN on log-mel spectrograms behind a FastAPI WebSocket server,
and a Godot 4.5 web client that listens to the microphone. The client is an instrument that shows what the model hears,
and a small game where you play real chords on a real guitar to cross a bridge.

The model comes from my master's thesis (January 2025), where it ran behind a Django app. This repository rebuilds it
around the same weights: ONNX Runtime instead of TensorFlow, a streaming protocol instead of JSON requests, tests and
CI instead of none.

**Status:** work in progress. The recognition server and the client work end to end; the client shows what the model
hears, window by window (below), and the first demo of the bridge game is playable (below). Retraining from scratch is possible (see Train); the runtime still serves the thesis weights.

## What it recognises

Eight chords: Am, Bb, Bdim, C, Dm, Em, F, G. Model input is one second of audio at 16 kHz, 64 mel bands
(`backend/models/meta.json`). Trained on the Kaggle dataset
[guitar-chords-v3](https://www.kaggle.com/datasets/fabianavinci/guitar-chords-v3) (20 instruments plus synthetic) by Fabiana Vinci, licensed
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## Measured

Same scenario before and after the rebuild, reproducible with the tools in the last column.

| | Thesis code | Now | How |
|---|---|---|---|
| Accuracy, test split (320 files) | 95.9% | 95.9%; ONNX matches Keras on 320/320 labels | `chordfinder eval`, `tests/test_parity.py` |
| Inference per window | 53 ms (TensorFlow) | 1.2 ms (ONNX Runtime) | `tools/bench.py` |
| Cold start to first answer | 2.3 s | 0.86 s (first run after install: ~9 s, see below) | `tools/bench.py` |
| Strum to chord, median | 0.31 s, but a wrong chord flashes in 53 of 80 recordings | 0.61 s, wrong chord in 1 of 80 | `tools/latency_cmp.py` |
| Environment | 2.5 GB venv (TensorFlow 1.7 GB) | 539 MB, no TensorFlow at runtime | size of `.venv` |
| Tests | 0 | 179 backend, 35 client checks, 4 browser e2e (2 of them in CI against the Docker image; the others need the dataset) | CI |

Cold start counts the first answer, and librosa compiles its functions through numba on first use: right
after installing, with no numba cache yet, the first answer takes ~9 s; later starts reuse the cache. The Docker
image compiles them at build time.

The extra 0.3 s is the price of smoothing: a chord is shown after three agreeing windows (125 ms step). Two windows
would be faster but bring the false chords back; speed has to come from retraining, not from tuning.

## Load

How many microphones one server process keeps up with, and what gives out first.

**Ceiling: 48 microphones** on two P-cores (4 logical CPUs) of an Intel Core Ultra 7 155H, synthetic loud audio
(no window is silence, so every frame runs the whole pipeline). At 56 the delay grows without bound.

| Run | Clients | p95, ms | CPU cores used | features, ms | thread pool wait, ms | Verdict |
|---|---|---|---|---|---|---|
| synthetic, 4 CPUs | 1 | 5.7 | 2.15 | 1.41 | 0.1 | pass |
| | 32 | 15.5 | 3.64 | 3.95 | 0.2 | pass |
| | 48 | 41.9 | 3.90 | 11.49 | 1.2 | pass |
| | 56 | 2 822 | 3.97 | 57.54 | 59.5 | fail: p95, lost, backlog |
| dataset recordings, 4 CPUs | 56 | 94.5 | 3.94 | 21.74 | 2.8 | pass (recordings have silence) |
| synthetic, no CPU limit | 32 | 749 | 10.52 | 42.70 | 1.7 | fail: p95, backlog |
| synthetic, 4 CPUs, `OPENBLAS_NUM_THREADS=1` | 1 | 5.9 | 0.03 | 1.38 | 0.2 | pass |
| | 48 | 18.9 | 1.19 | 5.94 | 0.1 | pass |
| | 64 | 6 180 | 1.79 | 57.29 | 81.2 | fail: p95, lost, backlog |

What gives out, in two layers:

- OpenBLAS threads in numpy spin while they wait for work. One microphone kept 2.15 cores busy for about
  0.03 cores of real work, and without a CPU limit the server (one BLAS thread per host CPU) failed already at
  32 microphones while burning 10.5 cores. With `OPENBLAS_NUM_THREADS=1` the same microphone costs 0.03 cores.
- Without the spinning, the ceiling is still 48: at 64 microphones the process uses 1.8 of its 4 CPUs while the
  feature stage (librosa) grows from 1.4 to 57 ms and frames wait 81 ms for a worker thread. Cores sit idle,
  work is serialised inside one Python process (the GIL). More throughput needs more processes, not faster code.

No step was `invalid`: the generator kept up throughout. It sends frames without WebSocket compression (a browser
uses deflate), so the server does slightly less work per frame than with real clients.

Reproduce (reports: `backend/manual/load-*.json`):

```sh
docker build -t chordfinder .
docker run --rm --cpuset-cpus=1-4 -p 8000:8000 chordfinder    # add -e OPENBLAS_NUM_THREADS=1 for the last rows
cd backend && taskset -c 6-11 uv run chordfinder load --json manual/load-synthetic.json
uv run chordfinder load --steps 36,40,44,48,56     # finer steps near the ceiling
uv run chordfinder load --source dataset --steps 48,56
```

Real-time criterion for one step (N simulated microphones, each sending a 125 ms frame every 125 ms):
a frame's delay runs from the moment the microphone "captured" it (its scheduled send time) to the reply.
A step passes when, over the measured window (after the warm-up), all of this holds:

1. p95 delay <= 125 ms (one hop);
2. no frame lost: every frame got a reply within the step plus a grace period;
3. the delay does not grow: median in the last third of the window minus the first third <= 25 ms;
4. no connection errors and no `{"error": ...}` replies.

A step is `invalid` (neither pass nor fail) when the generator itself could not keep up: p95 of its own
event-loop lag > 20 ms. The ceiling is the largest N among the consecutive passing steps from the start.

Run the generator against a running server (it sends synthetic loud audio, so no window is silence):

```sh
cd backend && uv run chordfinder load --json manual/load-synthetic.json
# --steps 1,2,4 --duration 30 --warmup 3 --source dataset --no-metrics ...; see --help
```

The server exposes Prometheus metrics (`curl localhost:8000/metrics`):

- `chordfinder_ws_connections`: open `/ws/recognize` connections
- `chordfinder_frames_total`: binary frames accepted
- `chordfinder_frame_errors_total`: binary frames rejected
- `chordfinder_silent_windows_total`: windows taken for silence, where no stage runs
- `chordfinder_frame_seconds`: frame received to reply sent
- `chordfinder_threadpool_wait_seconds`: wait for a worker thread
- `chordfinder_stage_seconds{stage}`: `resample`, `features`, `inference` time per window
- `chordfinder_stream_lag_seconds`: how far a stream's arrival falls behind its audio
- `chordfinder_event_loop_lag_seconds`: event-loop wake-up delay
- `process_cpu_seconds_total` and the other `process_*` series of the client library

`/metrics` has no authentication and exposes no audio or personal data.

Next: one BLAS thread and several uvicorn worker processes (from the numbers above), then Grafana dashboards.

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

## The game

Open `http://localhost:8000/?game` (`/` is still the instrument, `/?measure` the measuring mode). A traveller walks
over a chasm; the next gap shows the chord it needs, and playing that chord on a guitar lays a board. A run is 16 gaps
in four progressions of four chords, and the walker speeds up from 4.0 s to 2.5 s per span, under a minute in all.
A chord counts after three consecutive confident replies of the wanted chord;
a wrong chord costs nothing but time. If the walker reaches a gap with no board, an attempt is lost and
he goes back to the start of that span, with the boards already laid kept. There are three attempts per run, not per gap.
The console prints `game:` lines for tuning the thresholds (`ChordJudge`, `BridgeGame` constants). The game runs in the
browser only.

## Run

With Docker — the image builds the web client, installs the server and downloads the weights (SHA-256 checked):

```sh
docker build -t chordfinder .
docker run --rm -p 8000:8000 chordfinder
```

Open http://localhost:8000/ (or http://localhost:8000/?game for the game), allow the microphone, play a chord. The image is 255 MB to download: Python
runtime dependencies only, no TensorFlow, no Godot.

Without Docker: needs [uv](https://docs.astral.sh/uv/) and [Godot 4.5.1](https://godotengine.org/) with web
export templates.

```sh
(cd backend && uv sync && uv run chordfinder weights)
(cd frontend && mkdir -p build/web && godot --headless --export-release Web build/web/index.html)
(cd backend && uv run uvicorn chordfinder.server:app --port 8000)
```

Open http://localhost:8000/, allow the microphone, play a chord.

## Train

The model can be trained again from the dataset; the runtime (`backend/models/`) is not touched, the result goes
to `backend/models/trained/` (not in git).

```sh
cd backend
uv run chordfinder dataset --archive archive.zip   # unpack and verify a zip downloaded from Kaggle (see below)
uv run chordfinder dataset --check                 # verify an existing directory against dataset.json
uv run --group train chordfinder train             # Keras, CPU; writes chord-cnn.keras, chord-cnn.onnx, meta.json, history.json
uv run chordfinder weights --all
uv run chordfinder compare --json models/trained/compare.json   # thesis model vs the new one on the same 320 test files
```

`compare` prints accuracy, accuracy per class, both confusion matrices and a paired comparison (files only one of
the models gets right, exact McNemar p-value). The ONNX file has the same graph contract as `models/chord-cnn.onnx`,
so `ChordModel("models/trained")` and `tools/takes.py --models models/trained` load it unchanged.

The dataset is pinned by a hash of its content, not of its zip (Kaggle may rebuild the zip and unpackers spell some
file names differently). Kaggle serves the zip only to a signed-in account: download `archive.zip` from the
[dataset page](https://www.kaggle.com/datasets/fabianavinci/guitar-chords-v3) and pass it with `--archive`.

What is kept from the thesis recipe: features, the model (`model_Conv2D`), windows (envelope mask at 0.01, whole
one-second windows, the tail dropped), Adam 1e-3, categorical cross-entropy (in its sparse form), batch 32, 30 epochs,
shuffling, the last epoch is the result (no model selection).

What differs: a seed (`--seed`, default 0) with deterministic TensorFlow ops, so a rerun on the same machine is meant to
give the same weights; and 15 % of the guitars in `Training` are held out for validation (`--val 0.15`; `--val 0` trains
on everything, as the thesis did). A guitar is one group of recordings across all chord folders, the validation split
never cuts a group. `Test` is never read by `train`.

What the test measures: the split has no guitar on both sides (45 guitars in `Training`, 10 in `Test`), so accuracy
is "new guitar, not new player": one performer (JO) appears on both sides with different instruments, and the virtual
AcusticPlug guitars of `Test` come from the same plugin as those of `Training`.

Measured on an Intel Core Ultra 7 155H, CPU only, `--seed 0 --val 0.15`:

- Training takes 56 s wall time (52 s of it the fit itself), 1.2 GB peak memory; 4177 training windows, 1132
  validation windows, validation accuracy 89.3 %. ONNX and Keras outputs differ by at most 2.4e-7.
- A second run with the same seed gives a byte-identical `chord-cnn.onnx`; `history.json` differs only in the
  recorded duration.
- On the test split the new model gets 300 of 320 files right (93.8 %), the thesis weights 307 (95.9 %). Only the
  thesis model is right on 13 files, only the new one on 6, both are wrong on 7: McNemar p = 0.17, so the
  difference is not significant. The new model is weakest on Bdim (82.5 % against 95 %) and gets every G right.

The runtime keeps the thesis weights: the retrained model reproduces them within noise, it is not better.

## Test

```sh
(cd backend && uv run ruff check && uv run mypy && uv run pytest)
godot --headless --path frontend --import                   # once: builds the class cache the tests need
godot --headless --path frontend -s tests/test_client.gd   # client: wire format, replies, strict typing
godot --headless --path frontend -s tests/test_game.gd     # game rules, view, strict typing
(cd backend && uv run pytest -m e2e -s)                     # Chromium with a fake microphone → web build → server
(cd backend && CHORDFINDER_URL=http://localhost:8000 uv run pytest -m e2e -s)   # the same against a running container
```

CI builds the image, starts it and runs the browser tests against it.

Accuracy, parity and latency tests need the dataset and skip without it. It is not included in the repository:
`chordfinder dataset` fetches and verifies it (see Train), or point `CHORDFINDER_DATASET` at its `data/` directory
(default `~/datasets/guitar-chords-v3/data`). `backend/dataset.json` pins it by a content hash.

The tests of the training code that need TensorFlow run in the CI job `train`, or locally with
`(cd backend && uv run --group train pytest tests/test_train.py)`.
