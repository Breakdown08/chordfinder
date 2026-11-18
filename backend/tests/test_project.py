"""Project wiring of the training task: dependency groups, CI steps, ignored output. Reads files only."""

import json
import re
import tomllib
from pathlib import Path
from typing import Any

BACKEND: Path = Path(__file__).resolve().parents[1]
REPO: Path = BACKEND.parent
RUNTIME: list[str] = [
    "numpy>=1.26,<2",
    "scipy>=1.11",
    "librosa==0.10.2.post1",
    "onnxruntime>=1.19",
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "prometheus-client>=0.21",
    "websockets>=13",
]


def pyproject() -> dict[str, Any]:
    data: dict[str, Any] = tomllib.loads((BACKEND / "pyproject.toml").read_text())
    return data


def test_train_group_is_export_and_never_default() -> None:
    p: dict[str, Any] = pyproject()
    assert p["dependency-groups"]["train"] == [{"include-group": "export"}]
    assert "train" not in p["tool"]["uv"]["default-groups"]
    assert "export" not in p["tool"]["uv"]["default-groups"]


def test_ty_is_a_dev_dependency() -> None:
    dev: list[str] = pyproject()["dependency-groups"]["dev"]
    assert any(re.match(r"ty\b", d) for d in dev)


def test_runtime_packages_are_unchanged() -> None:
    assert pyproject()["project"]["dependencies"] == RUNTIME


def test_ty_keeps_its_rules_on() -> None:
    ty: dict[str, Any] = pyproject().get("tool", {}).get("ty", {})
    assert ty.get("rules", {}).get("unresolved-import") not in ("ignore", "off")
    assert ty.get("rules", {}).get("all") not in ("ignore", "off")


def ci_job(name: str) -> str:
    """The text of one job of ci.yml."""
    text: str = (REPO / ".github/workflows/ci.yml").read_text()
    m: re.Match[str] | None = re.search(rf"^  {name}:\n(.*?)(?=^  \S|\Z)", text, re.S | re.M)
    assert m is not None, f"no job {name} in ci.yml"
    return m.group(1)


def test_ci_backend_job_runs_ty_between_format_and_mypy() -> None:
    steps: list[str] = re.findall(r"- run: (uv run [^\n]+)", ci_job("backend"))
    assert steps.index("uv run ty check") == steps.index("uv run ruff format --check") + 1
    assert steps.index("uv run mypy") > steps.index("uv run ty check")


def test_ci_has_a_separate_train_job() -> None:
    job: str = ci_job("train")
    assert "--group train" in job
    assert "pytest" in job
    assert "TensorFlow" not in ci_job("backend") or "--group train" not in ci_job("backend")
    assert "--group train" not in ci_job("frontend")


def test_trained_models_are_not_tracked() -> None:
    ignored: list[str] = (REPO / ".gitignore").read_text().splitlines()
    assert any(line.strip().rstrip("/") in ("backend/models/trained", "models/trained") for line in ignored)


def test_lock_has_prometheus_client() -> None:
    assert 'name = "prometheus-client"' in (BACKEND / "uv.lock").read_text()


def test_ci_docker_job_checks_metrics_and_runs_the_load_generator() -> None:
    job: str = ci_job("docker")
    assert re.search(r"curl[^\n]*/metrics[^\n]*grep[^\n]*chordfinder_ws_connections", job)
    assert re.search(r"chordfinder load[^\n]*--steps 1\b", job)


# --- workers task (checks 27-29) --------------------------------------------------------------


def test_serve_script_and_threadpoolctl_are_declared() -> None:
    p: dict[str, Any] = pyproject()
    assert p["project"]["scripts"]["chordfinder-serve"] == "chordfinder.serve:main"
    assert any(d.startswith("threadpoolctl") for d in p["dependency-groups"]["dev"])
    assert 'name = "threadpoolctl"' in (BACKEND / "uv.lock").read_text()


def final_stage() -> list[str]:
    lines: list[str] = (REPO / "Dockerfile").read_text().splitlines()
    last: int = max(i for i, line in enumerate(lines) if line.startswith("FROM "))
    return lines[last + 1 :]


def test_dockerfile_sets_one_thread_before_the_warm_up_and_starts_the_launcher() -> None:
    from chordfinder import serve

    stage: list[str] = final_stage()
    envs: list[int] = [
        i
        for i, line in enumerate(stage)
        if line.startswith("ENV ") and all(f"{k}={v}" in line for k, v in serve.NATIVE_THREADS.items())
    ]
    warm: list[int] = [i for i, line in enumerate(stage) if line.startswith("RUN python -c")]
    assert envs, "no ENV line with every pair of NATIVE_THREADS"
    assert warm, "no warm-up line"
    assert envs[0] < warm[0]
    cmd: list[str] = [line for line in stage if line.startswith("CMD ")]
    assert len(cmd) == 1
    argv: list[str] = json.loads(cmd[0][len("CMD ") :])
    assert argv[0] == "chordfinder-serve"
    assert ["--host", "0.0.0.0"] == argv[argv.index("--host") : argv.index("--host") + 2]
    assert ["--port", "8000"] == argv[argv.index("--port") : argv.index("--port") + 2]
    assert "--workers" not in argv


def test_ci_docker_job_checks_pools_and_own_cpu_counter() -> None:
    job: str = ci_job("docker")
    assert "threadpool_info" in job
    assert re.search(r"curl[^\n]*/metrics[^\n]*grep[^\n]*chordfinder_cpu_seconds_total", job)
    assert re.search(r"curl[^\n]*/metrics[^\n]*grep[^\n]*chordfinder_ws_connections", job)
    assert re.search(r"chordfinder load[^\n]*--steps 1\b", job)
