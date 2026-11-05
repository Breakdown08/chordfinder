"""Score a measuring session: replay its recordings through the model, exactly as the server would.

    CHORDFINDER_SESSIONS=manual/sessions uv run uvicorn chordfinder.server:app --port 8000
    open http://localhost:8000/?measure, play every chord three times
    uv run python tools/takes.py manual/sessions/<folder> [--models path/to/other/weights]

Scored from the audio rather than from the live replies, so the same strums can compare two models.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from chordfinder.measure import SESSION_FILE, Strum, Window, load, replay, score
from chordfinder.model import MODELS, ChordModel

STRUMS: int = 3


def main() -> None:
    ap: argparse.ArgumentParser = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("--models", type=Path, default=MODELS)
    a: argparse.Namespace = ap.parse_args()

    folder: Path = a.session
    meta: Any = json.loads((folder / SESSION_FILE).read_text())
    rate: float = meta["sample_rate"]
    takes: dict[str, list[int]] = meta["takes"]
    model: ChordModel = ChordModel(a.models)

    strums: list[Strum] = []
    chord: str
    for chord in model.meta.classes:
        if chord not in takes:
            print(f"{chord:5} not recorded")
            continue
        onsets: list[int] = takes[chord]
        if len(onsets) != STRUMS:
            print(f"! {chord}: {len(onsets)} onsets, expected {STRUMS}; the first {STRUMS} are scored")
        windows: list[Window] = replay(model, load(folder / f"{chord}.wav"), rate, onsets)
        scored: list[Strum] = score(chord, windows, onsets[: STRUMS + 1], rate)[:STRUMS]
        strums += scored
        s: Strum
        for s in scored:
            mark: str = "ok" if s.majority == chord else "XX"
            smark: str = "ok" if s.first_stable == chord else "XX"
            votes: str = " ".join(f"{c}:{n}" for c, n in s.votes.most_common())
            latency: str = f"{s.latency:.2f} s" if s.latency is not None else "-"
            got: str = s.first_stable or "-"
            print(f"{chord:5} {mark} {s.majority or '-':5} | shown {smark} {got:5} {latency:7} | {votes}")

    total: int = len(strums)
    majority: int = sum(s.majority == s.truth for s in strums)
    shown: int = sum(s.first_stable == s.truth for s in strums)
    latencies: list[float] = sorted(s.latency for s in strums if s.latency is not None)
    print(f"\nmajority {majority}/{total}, first shown {shown}/{total} (target ≥ 23/24)")
    if latencies:
        print(f"median strum to right chord {latencies[len(latencies) // 2]:.2f} s")
    wrong: Counter[tuple[str, str | None]] = Counter(
        (s.truth, s.first_stable) for s in strums if s.first_stable != s.truth
    )
    if wrong:
        print(
            "first shown wrong: " + ", ".join(f"{t} → {g or '-'} ×{n}" for (t, g), n in wrong.most_common())
        )


if __name__ == "__main__":
    main()
