# syntax=docker/dockerfile:1
# One image, the whole app: the Godot web build and the recognition server with its weights.
#
#   docker build -t chordfinder .
#   docker run --rm -p 8000:8000 chordfinder       # http://localhost:8000/
#
# GitHub API allows 60 unauthenticated requests per hour; CI passes its token as a build secret:
#   docker build --secret id=github_token,env=GITHUB_TOKEN -t chordfinder .

ARG PYTHON=3.12.12-slim-bookworm

# --- web: export the Godot client (headless; exporting packs files, renders nothing) -----------
FROM debian:bookworm-slim AS web
ARG GODOT=4.5.1-stable
# SHA-512 from the release's SHA512-SUMS.txt
ARG GODOT_SHA512=5bccbed65a94b82c7c319fdb15719ee8113a6e503976cc54e16f1c61fe95f3d74e5e40b8449b5bb89ff7f424574c20af01a4f5ef08b389e4dc338b245185b0b9
ARG TEMPLATES_SHA512=8a65c73541184fbcf1a7afedb37c9c15ed0f3babdaafe36ff47ccc515548ac84bc7563ba5b94c253e6085b121ac608b7e03d35dfc0b2c8c690a646a8755b57e5
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl unzip \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /tmp/godot
RUN set -eu; \
    url="https://github.com/godotengine/godot/releases/download/${GODOT}"; \
    curl -fsSL -o godot.zip "${url}/Godot_v${GODOT}_linux.x86_64.zip"; \
    curl -fsSL -o templates.tpz "${url}/Godot_v${GODOT}_export_templates.tpz"; \
    echo "${GODOT_SHA512}  godot.zip" | sha512sum -c -; \
    echo "${TEMPLATES_SHA512}  templates.tpz" | sha512sum -c -; \
    unzip -q godot.zip && mv "Godot_v${GODOT}_linux.x86_64" /usr/local/bin/godot; \
    # Only the single-threaded web template the preset uses (the archive is ~1 GB of all platforms).
    dir="/root/.local/share/godot/export_templates/$(echo "${GODOT}" | tr - .)"; \
    mkdir -p "${dir}"; \
    unzip -q -j templates.tpz templates/version.txt templates/web_nothreads_release.zip -d "${dir}"; \
    rm -rf /tmp/godot
COPY frontend /src/frontend
WORKDIR /src/frontend
# A script that fails to compile leaves Godot running: the timeout turns that into a failure.
RUN timeout 300 godot --headless --import \
    && mkdir -p build/web \
    && timeout 300 godot --headless --export-release Web build/web/index.html \
    && test -f build/web/index.wasm

# --- server: runtime dependencies and the weights, checked against models/weights.json --------
FROM python:${PYTHON} AS server
COPY --from=ghcr.io/astral-sh/uv:0.12.11 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app/backend
# Dependencies first: this layer survives code changes.
COPY backend/pyproject.toml backend/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --no-install-project
COPY backend/src src
COPY backend/models models
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups
# Runtime weights only (ONNX); the download is written only after its SHA-256 matches the manifest.
RUN --mount=type=secret,id=github_token,env=GITHUB_TOKEN \
    .venv/bin/chordfinder weights && .venv/bin/chordfinder weights --check

# --- runtime --------------------------------------------------------------------------------------
FROM python:${PYTHON}
RUN useradd --system --uid 10001 --no-create-home chordfinder
COPY --from=server /app/backend /app/backend
COPY --from=web /src/frontend/build/web /app/web
# librosa JIT-compiles through numba: the first answer would take ~9 s. Compile once here, into a
# cache the server user owns (by default numba caches next to its sources, read-only at runtime).
ENV CHORDFINDER_WEB=/app/web PATH=/app/backend/.venv/bin:$PATH PYTHONUNBUFFERED=1 NUMBA_CACHE_DIR=/app/numba
RUN python -c "import numpy as np; from chordfinder.model import ChordModel; from chordfinder.stream import recognize; \
t = np.arange(48000, dtype=np.float32) / 48000; \
print('warm-up:', recognize(ChordModel(), (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), 48000))" \
    && chown -R chordfinder /app/numba
USER chordfinder
WORKDIR /app/backend
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --start-interval=2s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]
CMD ["uvicorn", "chordfinder.server:app", "--host", "0.0.0.0", "--port", "8000"]
