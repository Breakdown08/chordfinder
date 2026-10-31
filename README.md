# chordfinder

Guitar chord recognition (CNN on log-mel spectrograms) and a web game built on it. Work in progress:
`backend/` is the recognition server, `frontend/` is the Godot 4.5.1 web client (for now a listener: microphone →
`/ws/recognize` → chord on screen; the game comes next). Full setup guide — once the project is runnable end to end.

```sh
(cd backend && uv sync && uv run chordfinder weights)            # weights: GitHub Release, SHA-256 pinned
(cd frontend && mkdir -p build/web && godot --headless --export-release Web build/web/index.html)
(cd backend && uv run uvicorn chordfinder.server:app --port 8000)   # http://localhost:8000/
```

Tests:

```sh
godot --headless --path frontend -s tests/test_client.gd   # client: wire format, replies, strict typing
(cd backend && uv run pytest)                              # server
(cd backend && uv run pytest -m e2e -s)        # Chromium fake mic -> web build -> server
```
