"""Cold start and per-window latency of the runtime model (rows of the "before -> target" table).

uv run python tools/bench.py
"""

import time

import numpy as np

from chordfinder.audio import Audio

t0: float = time.perf_counter()
from chordfinder.model import ChordModel  # noqa: E402

rng: np.random.Generator = np.random.default_rng(0)
y: Audio = (0.3 * rng.standard_normal(16000)).astype(np.float32)
model: ChordModel = ChordModel()
model.predict(y)  # librosa is imported lazily here, so the first answer belongs to the cold start
cold: float = time.perf_counter() - t0

times: list[float] = []
_: int
for _ in range(300):
    t: float = time.perf_counter()
    model.predict(y)
    times.append((time.perf_counter() - t) * 1000)
print(f"cold start {cold:.2f} s (import + load + first answer)")
print(f"window: median {np.median(times):.1f} ms, p95 {np.percentile(times, 95):.1f} ms")
