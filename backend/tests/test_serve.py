"""The launcher chordfinder.serve (checks 1-12 of the workers task). uvicorn.run is replaced by a recorder."""

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from chordfinder import serve

MULTIPROC: str = serve.MULTIPROC_ENV
FORGET: tuple[str, ...] = (MULTIPROC, serve.WORKERS_ENV, *serve.NATIVE_THREADS)


def clean_env(monkeypatch: pytest.MonkeyPatch, *keys: str) -> None:
    """Remove keys now and restore their absence afterwards, even when main() sets them."""
    key: str
    for key in keys:
        monkeypatch.setenv(key, "placeholder")  # records the original state for the teardown
        monkeypatch.delenv(key)


@dataclass
class Call:
    app: str
    kwargs: dict[str, object]
    environ: dict[str, str]
    metrics_dir: str | None
    dir_exists: bool
    dir_entries: list[str]


@dataclass
class Recorder:
    calls: list[Call] = field(default_factory=list)
    raises: Exception | None = None

    def __call__(self, app: str, **kwargs: object) -> None:
        path: str | None = os.environ.get(MULTIPROC)
        exists: bool = path is not None and Path(path).is_dir()
        entries: list[str] = sorted(p.name for p in Path(path).iterdir()) if path and exists else []
        self.calls.append(Call(app, dict(kwargs), dict(os.environ), path, exists, entries))
        if self.raises is not None:
            raise self.raises


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    recorder: Recorder = Recorder()
    monkeypatch.setattr("chordfinder.serve.uvicorn.run", recorder)
    clean_env(monkeypatch, *FORGET)
    return recorder


def affinity(monkeypatch: pytest.MonkeyPatch, cpus: set[int]) -> None:
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: cpus)


# 1
def test_available_cpus_uses_affinity(monkeypatch: pytest.MonkeyPatch) -> None:
    affinity(monkeypatch, {1, 2, 3, 4})
    assert serve.available_cpus() == 4


def test_available_cpus_without_affinity_uses_cpu_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 6)
    assert serve.available_cpus() == 6


def test_available_cpus_without_any_count_is_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert serve.available_cpus() == 1


# 2
@pytest.mark.parametrize(
    ("value", "cpus", "expected"),
    [
        (None, 4, 4),
        ("", 4, 4),
        ("  ", 4, 4),
        (None, 22, 8),
        ("", 22, 8),
        ("12", 4, 12),
        ("3", 4, 3),
        (" 3 ", 4, 3),
    ],
)
def test_workers_from(value: str | None, cpus: int, expected: int) -> None:
    assert serve.workers_from(value, cpus) == expected


@pytest.mark.parametrize("value", ["0", "-1", "abc", "2.5"])
def test_workers_from_rejects_bad_values(value: str) -> None:
    with pytest.raises(ValueError):
        serve.workers_from(value, 4)


# 3
def test_prepare_multiproc_dir_removes_only_db_files_in_the_directory(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    name: str
    for name in ("a.db", "b.db", "keep.txt", "sub/c.db"):
        (tmp_path / name).write_text("x")
    serve.prepare_multiproc_dir(tmp_path)
    assert not (tmp_path / "a.db").exists()
    assert not (tmp_path / "b.db").exists()
    assert (tmp_path / "keep.txt").exists()
    assert (tmp_path / "sub" / "c.db").exists()


def test_prepare_multiproc_dir_creates_missing_path(tmp_path: Path) -> None:
    target: Path = tmp_path / "x" / "y"
    serve.prepare_multiproc_dir(target)
    assert target.is_dir()


# 4
def test_main_passes_arguments_and_makes_a_temporary_metrics_dir(run: Recorder) -> None:
    code: int = serve.main(["--workers", "3", "--host", "0.0.0.0", "--port", "9001"])
    assert code == 0
    assert len(run.calls) == 1
    call: Call = run.calls[0]
    assert call.app == "chordfinder.server:app"
    assert call.kwargs["host"] == "0.0.0.0"
    assert call.kwargs["port"] == 9001
    assert call.kwargs["workers"] == 3
    assert call.metrics_dir is not None
    assert Path(call.metrics_dir).name.startswith("chordfinder-metrics-")
    assert call.dir_exists
    assert call.dir_entries == []
    assert not Path(call.metrics_dir).exists()


def test_main_defaults_are_local_port_8000(run: Recorder) -> None:
    assert serve.main(["--workers", "2"]) == 0
    assert run.calls[0].kwargs["host"] == "127.0.0.1"
    assert run.calls[0].kwargs["port"] == 8000


# 5
def test_main_prepares_a_given_metrics_dir_and_keeps_it(
    run: Recorder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prom: Path = tmp_path / "prom"
    prom.mkdir()
    (prom / "counter_1.db").write_text("x")
    (prom / "notes.txt").write_text("x")
    monkeypatch.setenv(MULTIPROC, str(prom))
    assert serve.main(["--workers", "2"]) == 0
    call: Call = run.calls[0]
    assert call.metrics_dir == str(prom)
    assert call.dir_entries == ["notes.txt"]
    assert prom.is_dir()


# 6
def test_one_process_leaves_multiprocess_mode_off(run: Recorder) -> None:
    assert serve.main(["--workers", "1"]) == 0
    assert run.calls[0].metrics_dir is None
    assert run.calls[0].kwargs["workers"] == 1


# 7
def test_empty_metrics_dir_variable_is_dropped_for_one_process(
    run: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(MULTIPROC, "")
    assert serve.main(["--workers", "1"]) == 0
    assert run.calls[0].metrics_dir is None


def test_blank_metrics_dir_variable_is_replaced_for_several_processes(
    run: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(MULTIPROC, "")
    assert serve.main(["--workers", "2"]) == 0
    call: Call = run.calls[0]
    assert call.metrics_dir is not None
    assert Path(call.metrics_dir).name.startswith("chordfinder-metrics-")
    assert call.dir_exists


# 8
def test_workers_from_environment(run: Recorder, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(serve.WORKERS_ENV, "2")
    assert serve.main([]) == 0
    assert run.calls[0].kwargs["workers"] == 2


def test_flag_beats_environment(run: Recorder, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(serve.WORKERS_ENV, "2")
    assert serve.main(["--workers", "5"]) == 0
    assert run.calls[0].kwargs["workers"] == 5


def test_default_is_the_cpu_count(run: Recorder, monkeypatch: pytest.MonkeyPatch) -> None:
    affinity(monkeypatch, {0, 1, 2})
    assert serve.main([]) == 0
    assert run.calls[0].kwargs["workers"] == 3


def test_default_is_capped_at_eight(run: Recorder, monkeypatch: pytest.MonkeyPatch) -> None:
    affinity(monkeypatch, set(range(22)))
    assert serve.main([]) == 0
    assert run.calls[0].kwargs["workers"] == 8


def test_explicit_flag_is_not_capped(run: Recorder) -> None:
    assert serve.main(["--workers", "12"]) == 0
    assert run.calls[0].kwargs["workers"] == 12


# 9
@pytest.mark.parametrize(
    ("argv", "env"),
    [
        ([], "abc"),
        ([], "0"),
        (["--workers", "0"], None),
        (["--workers", "-2"], None),
    ],
)
def test_bad_worker_counts_exit_with_code_2(
    run: Recorder, monkeypatch: pytest.MonkeyPatch, argv: list[str], env: str | None
) -> None:
    if env is not None:
        monkeypatch.setenv(serve.WORKERS_ENV, env)
    raised: pytest.ExceptionInfo[SystemExit]
    with pytest.raises(SystemExit) as raised:
        serve.main(argv)
    assert raised.value.code == 2
    assert run.calls == []


# 10
def test_native_threads_default_to_one(run: Recorder) -> None:
    assert serve.main(["--workers", "1"]) == 0
    environ: dict[str, str] = run.calls[0].environ
    assert environ["OPENBLAS_NUM_THREADS"] == "1"
    assert environ["OMP_NUM_THREADS"] == "1"


def test_native_threads_keep_the_users_value(run: Recorder, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "4")
    assert serve.main(["--workers", "1"]) == 0
    environ: dict[str, str] = run.calls[0].environ
    assert environ["OPENBLAS_NUM_THREADS"] == "4"
    assert environ["OMP_NUM_THREADS"] == "1"


# 11
def test_uvicorn_error_propagates_and_temporary_dir_is_removed(run: Recorder) -> None:
    run.raises = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        serve.main(["--workers", "2"])
    assert len(run.calls) == 1
    path: str | None = run.calls[0].metrics_dir
    assert path is not None
    assert run.calls[0].dir_exists
    assert not Path(path).exists()


# 12
def test_launcher_imports_no_numpy_and_no_app() -> None:
    script: str = (
        "import sys, chordfinder.serve; "
        "assert 'numpy' not in sys.modules; "
        "assert 'chordfinder.server' not in sys.modules"
    )
    done: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
