"""Weights download: hashes decide, a bad download never replaces a file. No network: fetch is a stub."""

import hashlib
from pathlib import Path

import pytest

from chordfinder.model import MODELS
from chordfinder.weights import Manifest, Pin, WeightsError, check, download, pin

ONNX_BYTES: bytes = b"onnx weights"
KERAS_BYTES: bytes = b"keras original"


def pinned(data: bytes, runtime: bool) -> Pin:
    return Pin(sha256=hashlib.sha256(data).hexdigest(), size=len(data), runtime=runtime)


MANIFEST: Manifest = Manifest(
    repo="owner/repo",
    tag="weights-v1",
    files={"m.onnx": pinned(ONNX_BYTES, True), "m.keras": pinned(KERAS_BYTES, False)},
)
ASSETS: dict[str, bytes] = {"m.onnx": ONNX_BYTES, "m.keras": KERAS_BYTES}


class Stub:
    def __init__(self, assets: dict[str, bytes]) -> None:
        self.assets: dict[str, bytes] = assets
        self.calls: list[str] = []

    def __call__(self, name: str) -> bytes:
        self.calls.append(name)
        return self.assets[name]


def test_downloads_runtime_only_by_default(tmp_path: Path) -> None:
    fetch: Stub = Stub(ASSETS)
    assert download(MANIFEST, tmp_path, False, fetch) == ["m.onnx"]
    assert (tmp_path / "m.onnx").read_bytes() == ONNX_BYTES
    assert not (tmp_path / "m.keras").exists()
    assert download(MANIFEST, tmp_path, True, fetch) == ["m.keras"]  # the onnx is not fetched again
    assert fetch.calls == ["m.onnx", "m.keras"]
    assert check(MANIFEST, tmp_path, True) == {}


def test_damaged_file_is_replaced(tmp_path: Path) -> None:
    (tmp_path / "m.onnx").write_bytes(b"onnx weightz")  # same size, other hash
    assert check(MANIFEST, tmp_path, False) == {"m.onnx": "damaged (hash differs from the manifest)"}
    assert download(MANIFEST, tmp_path, False, Stub(ASSETS)) == ["m.onnx"]
    assert (tmp_path / "m.onnx").read_bytes() == ONNX_BYTES


def test_bad_download_keeps_nothing(tmp_path: Path) -> None:
    with pytest.raises(WeightsError, match="m.onnx"):
        download(MANIFEST, tmp_path, False, Stub({"m.onnx": b"truncated"}))
    assert list(tmp_path.iterdir()) == []


def test_pin_takes_local_hashes(tmp_path: Path) -> None:
    (tmp_path / "m.onnx").write_bytes(b"retrained")
    (tmp_path / "m.keras").write_bytes(KERAS_BYTES)
    repinned: Manifest = pin(MANIFEST, tmp_path)
    assert repinned.files["m.onnx"] == pinned(b"retrained", True)
    assert repinned.files["m.keras"] == MANIFEST.files["m.keras"]


def test_committed_manifest_matches_local_weights() -> None:
    """The manifest in git pins the weights the tests run on (skips where they are not downloaded)."""
    manifest: Manifest = Manifest.load()
    if not all((MODELS / name).is_file() for name in manifest.files):
        pytest.skip("weights not downloaded")
    assert check(manifest, MODELS, True) == {}
