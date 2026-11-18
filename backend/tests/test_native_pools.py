"""One thread in every native pool (checks 13-14 of the workers task), measured in a fresh process."""

import json
import os
import subprocess
import sys

import pytest

from chordfinder import serve

IMPORTS: str = "import numpy, scipy.linalg, librosa, onnxruntime; from threadpoolctl import threadpool_info; "
SCRIPT: str = IMPORTS + "import json; print(json.dumps(threadpool_info()))"
# The launcher sets the variables itself (uvicorn.run is a no-op here), then the heavy imports follow.
LAUNCHER_SCRIPT: str = (
    "import sys, json, uvicorn; "
    "uvicorn.run = lambda app, **kw: None; "
    "import chordfinder.serve as s; "
    "assert s.main(['--workers', '1']) == 0; "
    "assert 'numpy' not in sys.modules; " + IMPORTS + "print(json.dumps(threadpool_info()))"
)


def pools(script: str, env: dict[str, str]) -> list[dict[str, object]]:
    done: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180, env=env
    )
    assert done.returncode == 0, done.stderr
    found: list[dict[str, object]] = json.loads(done.stdout.strip().splitlines()[-1])
    return found


def without_native(env: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in env.items() if k not in serve.NATIVE_THREADS}


# 13
def test_with_the_variables_every_pool_has_one_thread() -> None:
    found: list[dict[str, object]] = pools(SCRIPT, {**os.environ, **serve.NATIVE_THREADS})
    assert found
    assert any(p["internal_api"] == "openblas" for p in found)
    assert all(p["num_threads"] == 1 for p in found), found


# 14
def test_without_the_variables_some_pool_has_many_threads() -> None:
    if serve.available_cpus() <= 1:
        pytest.skip("one CPU: pools have one thread anyway")
    found: list[dict[str, object]] = pools(SCRIPT, without_native(dict(os.environ)))
    assert any(isinstance(p["num_threads"], int) and p["num_threads"] > 1 for p in found), found


# 13, via the launcher: setdefault in main() is early enough for numpy imported afterwards
def test_launcher_alone_gives_one_thread_in_every_pool() -> None:
    found: list[dict[str, object]] = pools(LAUNCHER_SCRIPT, without_native(dict(os.environ)))
    assert found
    assert all(p["num_threads"] == 1 for p in found), found
