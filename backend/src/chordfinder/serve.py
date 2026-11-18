"""Launcher: several uvicorn worker processes, one native thread each, metrics summed over them.

    uv run chordfinder-serve [--host H] [--port P] [--workers N]

Imports no numpy (nor the app): the thread-limit variables must be set before the first import of it.
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import uvicorn

WORKERS_ENV: str = "CHORDFINDER_WORKERS"
APP: str = "chordfinder.server:app"
NATIVE_THREADS: dict[str, str] = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
MULTIPROC_ENV: str = "PROMETHEUS_MULTIPROC_DIR"
MAX_DEFAULT_WORKERS: int = 8  # only for the default; an explicit number is not capped


def available_cpus() -> int:
    """CPUs this process may run on (a container's cpuset counts, a --cpus quota does not)."""
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def workers_from(value: str | None, cpus: int) -> int:
    if value is None or not value.strip():
        return min(cpus, MAX_DEFAULT_WORKERS)
    n: int = int(value.strip())
    if n < 1:
        raise ValueError(f"workers must be >= 1, got {n}")
    return n


def prepare_multiproc_dir(path: Path) -> None:
    """Create the directory and drop stale *.db files left by an earlier run (only directly in it)."""
    path.mkdir(parents=True, exist_ok=True)
    stale: Path
    for stale in path.glob("*.db"):
        if stale.is_file():
            stale.unlink()


def positive_int(text: str) -> int:
    n: int = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
    return n


def main(argv: list[str] | None = None) -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(prog="chordfinder-serve")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=positive_int, default=None)
    args: argparse.Namespace = parser.parse_args(argv)
    k: str
    v: str
    for k, v in NATIVE_THREADS.items():
        os.environ.setdefault(k, v)

    workers: int
    if args.workers is not None:
        workers = args.workers
    else:
        raw: str | None = os.environ.get(WORKERS_ENV)
        try:
            workers = workers_from(raw, available_cpus())
        except ValueError:
            parser.error(f"{WORKERS_ENV} must be a positive integer, got {raw!r}")

    created: Path | None = None
    configured: str | None = os.environ.get(MULTIPROC_ENV)
    if configured is not None and not configured.strip():
        del os.environ[MULTIPROC_ENV]  # prometheus_client turns multiprocess on by the key alone
        configured = None
    if configured is not None:
        prepare_multiproc_dir(Path(configured))
    elif workers > 1:
        created = Path(tempfile.mkdtemp(prefix="chordfinder-metrics-"))
        os.environ[MULTIPROC_ENV] = str(created)
    try:
        uvicorn.run(APP, host=args.host, port=args.port, workers=workers)
    finally:
        if created is not None:
            shutil.rmtree(created, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
