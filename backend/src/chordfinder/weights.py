"""Model weights live in a GitHub Release, not in git; models/weights.json pins each file by SHA-256.

    uv run chordfinder weights           # download what is missing or damaged (runtime: ONNX only)
    uv run chordfinder weights --all     # also the Keras original (export, tools/latency_cmp.py --keras)
    uv run chordfinder weights --check   # verify only, exit 1 on a missing or damaged file
    uv run chordfinder weights --pin     # after retraining: write the local files' hashes into the manifest

Downloads go through the GitHub API, so a private repository works with GITHUB_TOKEN (CI has one).
"""

import hashlib
import json
import os
import ssl
import urllib.error
import urllib.request
from collections.abc import Callable
from http.client import HTTPResponse
from pathlib import Path

from pydantic import BaseModel

from chordfinder.model import MODELS

MANIFEST: str = "weights.json"
API: str = "https://api.github.com"


class Pin(BaseModel):
    sha256: str
    size: int
    runtime: bool  # needed to serve; the rest only for export and comparisons


class Manifest(BaseModel):
    repo: str  # owner/name
    tag: str  # release that holds the files
    files: dict[str, Pin]

    @classmethod
    def load(cls, models: Path = MODELS) -> "Manifest":
        return cls.model_validate_json((models / MANIFEST).read_text())


class WeightsError(RuntimeError):
    pass


type Fetch = Callable[[str], bytes]  # asset name -> its bytes


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def problem(path: Path, pin: Pin) -> str | None:
    """None when the file is there and matches its pin."""
    if not path.is_file():
        return "missing"
    if path.stat().st_size != pin.size or sha256(path) != pin.sha256:
        return "damaged (hash differs from the manifest)"
    return None


def wanted(manifest: Manifest, everything: bool) -> dict[str, Pin]:
    return {name: pin for name, pin in manifest.files.items() if everything or pin.runtime}


def check(manifest: Manifest, models: Path, everything: bool) -> dict[str, str]:
    """File name -> problem, for the files that have one."""
    found: dict[str, str] = {}
    name: str
    pin: Pin
    for name, pin in wanted(manifest, everything).items():
        p: str | None = problem(models / name, pin)
        if p is not None:
            found[name] = p
    return found


def download(manifest: Manifest, models: Path, everything: bool, fetch: Fetch) -> list[str]:
    """Fetch the files that are missing or damaged; a file is written only after its hash matches."""
    fetched: list[str] = []
    name: str
    for name in check(manifest, models, everything):
        pin: Pin = manifest.files[name]
        data: bytes = fetch(name)
        digest: str = hashlib.sha256(data).hexdigest()
        if len(data) != pin.size or digest != pin.sha256:
            raise WeightsError(
                f"{name}: downloaded {len(data)} bytes with sha256 {digest}, manifest pins {pin}"
            )
        part: Path = models / f"{name}.part"
        part.write_bytes(data)
        part.replace(models / name)
        fetched.append(name)
    return fetched


def pin(manifest: Manifest, models: Path) -> Manifest:
    """Manifest with the hashes and sizes of the local files (after retraining or re-export)."""
    files: dict[str, Pin] = {
        name: Pin(sha256=sha256(models / name), size=(models / name).stat().st_size, runtime=p.runtime)
        for name, p in manifest.files.items()
    }
    return Manifest(repo=manifest.repo, tag=manifest.tag, files=files)


def save(manifest: Manifest, models: Path = MODELS) -> None:
    (models / MANIFEST).write_text(json.dumps(manifest.model_dump(), indent=2) + "\n")


def tls() -> ssl.SSLContext:
    """NixOS has the CA bundle but not the hashed /etc/ssl/certs that Python's default looks for."""
    bundle: str = os.environ.get("SSL_CERT_FILE") or "/etc/ssl/certs/ca-certificates.crt"
    return ssl.create_default_context(cafile=bundle if Path(bundle).is_file() else None)


class Asset(BaseModel):
    name: str
    url: str  # API url: with Accept: application/octet-stream it redirects to the file


class Release(BaseModel):
    assets: list[Asset]


def github_fetch(manifest: Manifest, token: str | None = None) -> Fetch:
    """Release assets by name through the API (a private repository needs GITHUB_TOKEN)."""
    token = token or os.environ.get("GITHUB_TOKEN")
    assets: dict[str, str] = {}  # name -> API url, filled on the first fetch

    def get(url: str, accept: str) -> bytes:
        request: urllib.request.Request = urllib.request.Request(url, headers={"Accept": accept})
        if token:
            # Not carried over the redirect to the storage host: it rejects a second auth method.
            request.add_unredirected_header("Authorization", f"Bearer {token}")
        response: bytes
        r: HTTPResponse
        with urllib.request.urlopen(request, timeout=60, context=tls()) as r:
            response = r.read()
        return response

    def fetch(name: str) -> bytes:
        if not assets:
            url: str = f"{API}/repos/{manifest.repo}/releases/tags/{manifest.tag}"
            try:
                body: bytes = get(url, "application/vnd.github+json")
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    raise
                hint: str = "" if token else "; a private repository needs GITHUB_TOKEN"
                raise WeightsError(f"no release {manifest.tag} in {manifest.repo}{hint}") from e
            release: Release = Release.model_validate_json(body)
            assets.update({a.name: a.url for a in release.assets})
        if name not in assets:
            raise WeightsError(f"release {manifest.tag} of {manifest.repo} has no asset {name}")
        return get(assets[name], "application/octet-stream")

    return fetch
