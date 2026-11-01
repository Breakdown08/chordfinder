"""Score the manual guitar check from the server log (CHORDFINDER_LOG).

    CHORDFINDER_LOG=manual.jsonl uv run uvicorn chordfinder.server:app --port 8000
    uv run python tools/strums.py manual.jsonl [--rounds 3] [--retry 21,22]

Protocol: chords in ORDER, each strummed once and left to ring ~2 s, then muted for ~2 s;
ORDER repeated --rounds times. A strum is a run of replies with a chord (RMS above the silence
floor); a silent reply or a gap > GAP s ends it. Two answers per strum: the majority chord over
its replies (as in the spike) and the first stable chord (what the game will act on).
--retry: strums (1-based) that repeated the previous chord; listed but not scored.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ORDER: list[str] = ["Am", "Bb", "Bdim", "C", "Dm", "Em", "F", "G"]
GAP: float = 1.0
type Reply = dict[str, Any]


def strums(replies: list[Reply]) -> list[list[Reply]]:
    runs: list[list[Reply]] = []
    cur: list[Reply] = []
    last_ts: float | None = None
    r: Reply
    for r in replies:
        broken: bool = r["chord"] is None or (last_ts is not None and r["ts"] - last_ts > GAP)
        if broken and cur:
            runs.append(cur)
            cur = []
        if r["chord"] is not None:
            cur.append(r)
        last_ts = r["ts"]
    if cur:
        runs.append(cur)
    return runs


def first_stable(run: list[Reply]) -> str | None:
    return next((r["stable"] for r in run if r["stable"] is not None), None)


def main() -> None:
    ap: argparse.ArgumentParser = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--retry", type=lambda v: {int(x) for x in v.split(",")}, default=set())
    a: argparse.Namespace = ap.parse_args()

    log: Path = a.log
    replies: list[Reply] = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    all_runs: list[list[Reply]] = strums(replies)
    retry: set[int] = a.retry
    i: int
    for i in sorted(retry):
        votes: Counter[str] = Counter(r["chord"] for r in all_runs[i - 1])
        print(f"retry {i}: {' '.join(f'{c}:{n}' for c, n in votes.most_common())} (not scored)")
    runs: list[list[Reply]] = [run for i, run in enumerate(all_runs, 1) if i not in retry]
    expected: list[str] = ORDER * a.rounds
    print(f"replies {len(replies)}, strums {len(runs)}, expected {len(expected)}")
    if len(runs) != len(expected):
        print("! strum count differs from the protocol: alignment below is positional, check by eye")

    hits: int = 0
    stable_hits: int = 0
    want: str
    run: list[Reply]
    for i, (want, run) in enumerate(zip(expected, runs, strict=False), 1):
        counts: Counter[str] = Counter(r["chord"] for r in run)
        got: str = counts.most_common(1)[0][0]
        stable: str | None = first_stable(run)
        hits += got == want
        stable_hits += stable == want
        mark: str = "ok" if got == want else "XX"
        smark: str = "ok" if stable == want else "XX"
        votes_s: str = " ".join(f"{c}:{n}" for c, n in counts.most_common())
        shown: str = stable or "-"
        print(f"{i:2} {mark} want {want:4} got {got:4} | stable {smark} {shown:4} ", end="")
        print(f"({len(run)} replies: {votes_s})")
    print(f"\nmajority {hits}/{len(expected)}, first stable {stable_hits}/{len(expected)} (target ≥ 23/24)")


if __name__ == "__main__":
    main()
