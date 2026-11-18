"""Helpers for tests that read Prometheus text: numbers by series name, parsed with prometheus_client."""

import time
from collections.abc import Mapping

from fastapi.testclient import TestClient
from httpx import Response
from prometheus_client.metrics_core import Metric
from prometheus_client.parser import text_string_to_metric_families
from prometheus_client.samples import Sample


def series(name: str, **labels: str) -> str:
    """Key of one series in the dict `parse` returns: name{a="1",b="2"} with sorted labels."""
    if not labels:
        return name
    inner: str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return f"{name}{{{inner}}}"


def parse(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    family: Metric
    for family in text_string_to_metric_families(text):
        sample: Sample
        for sample in family.samples:
            out[series(sample.name, **sample.labels)] = sample.value
    return out


def scrape(client: TestClient) -> dict[str, float]:
    r: Response = client.get("/metrics")
    assert r.status_code == 200, f"GET /metrics answered {r.status_code}"
    return parse(r.text)


def delta(before: Mapping[str, float], after: Mapping[str, float], key: str) -> float:
    """Growth of one series; a series missing on either side counts as 0."""
    return after.get(key, 0.0) - before.get(key, 0.0)


def settle(
    client: TestClient, before: Mapping[str, float], expected: Mapping[str, float], timeout: float = 2.0
) -> dict[str, float]:
    """Scrape until every series grew by at least `expected` (the handler finishes after the last reply)."""
    deadline: float = time.monotonic() + timeout
    now: dict[str, float] = scrape(client)
    while time.monotonic() < deadline and any(delta(before, now, k) < v for k, v in expected.items()):
        time.sleep(0.05)
        now = scrape(client)
    return now
