"""Dataset pin: content hash, check, safe install, mirror download. No network, no real dataset."""

import hashlib
import importlib
import io
import os
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

from chordfinder import dataset, weights
from tests.synthetic import digest_by_definition, run_cli, write_tree

TREE: dict[str, bytes] = {"Training/Am/a.wav": b"training am", "Test/Am/b.wav": b"test am"}


def manifest_for(root: Path, archive: dataset.ArchivePin | None = None) -> dataset.DatasetManifest:
    full: tuple[str, int] = digest_by_definition(root)
    test: tuple[str, int] = digest_by_definition(root, "Test")
    return dataset.DatasetManifest(
        name="tiny",
        source="https://example.invalid/tiny",
        author="Someone",
        licence="CC BY-SA 4.0",
        files=full[1],
        content_sha256=full[0],
        test_files=test[1],
        test_content_sha256=test[0],
        archive=archive,
    )


def make_zip(path: Path, members: dict[str, bytes]) -> Path:
    name: str
    data: bytes
    z: zipfile.ZipFile
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return path


def snapshot(root: Path) -> dict[str, int]:
    """Relative path -> mtime in ns, for every file and directory."""
    found: dict[str, int] = {}
    p: Path
    for p in [root, *root.rglob("*")]:
        found[str(p.relative_to(root))] = p.stat().st_mtime_ns
    return found


def patch_manifest(monkeypatch: pytest.MonkeyPatch, manifest: dataset.DatasetManifest) -> None:
    def load(
        cls: type[dataset.DatasetManifest], path: Path = dataset.DATASET_MANIFEST
    ) -> dataset.DatasetManifest:
        return manifest

    monkeypatch.setattr(dataset.DatasetManifest, "load", classmethod(load))


# 1


def test_content_digest_follows_the_definition(tmp_path: Path) -> None:
    write_tree(tmp_path, TREE)
    assert dataset.content_digest(tmp_path) == (digest_by_definition(tmp_path)[0], 2)
    test_only: tuple[str, int] = dataset.content_digest(tmp_path, "Test")
    assert test_only == (digest_by_definition(tmp_path, "Test")[0], 1)
    expected: str = hashlib.sha256(f"Test/Am/{hashlib.sha256(b'test am').hexdigest()}\n".encode()).hexdigest()
    assert test_only[0] == expected


# 2


def test_digest_ignores_names_but_not_bytes_or_class(tmp_path: Path) -> None:
    write_tree(tmp_path, TREE)
    before: str = dataset.content_digest(tmp_path)[0]
    (tmp_path / "Training/Am/a.wav").rename(tmp_path / "Training/Am/renamed-Ω.wav")
    assert dataset.content_digest(tmp_path)[0] == before
    (tmp_path / "Training/Am/renamed-Ω.wav").write_bytes(b"training an")  # one byte changed
    assert dataset.content_digest(tmp_path)[0] != before
    (tmp_path / "Training/Am/renamed-Ω.wav").write_bytes(b"training am")
    assert dataset.content_digest(tmp_path)[0] == before
    (tmp_path / "Training/Bb").mkdir()
    (tmp_path / "Training/Am/renamed-Ω.wav").rename(tmp_path / "Training/Bb/a.wav")
    assert dataset.content_digest(tmp_path)[0] != before


# 3


def test_check_reports_the_reason(tmp_path: Path) -> None:
    write_tree(tmp_path, TREE)
    manifest: dataset.DatasetManifest = manifest_for(tmp_path)
    assert dataset.check(tmp_path, manifest) is None
    assert dataset.check(tmp_path / "nowhere", manifest) == "missing"

    (tmp_path / "Training/Am/extra.wav").write_bytes(b"extra")
    assert dataset.check(tmp_path, manifest) == "3 files, manifest pins 2"
    (tmp_path / "Training/Am/extra.wav").unlink()
    (tmp_path / "Training/Am/a.wav").unlink()
    assert dataset.check(tmp_path, manifest) == "1 files, manifest pins 2"

    (tmp_path / "Training/Am/a.wav").write_bytes(b"training an")  # same count, other byte
    assert dataset.check(tmp_path, manifest) == "content differs from the manifest"


# 4, 5, 6, 7


def test_install_unpacks_and_verifies(tmp_path: Path) -> None:
    src: Path = tmp_path / "src"
    write_tree(src, TREE)
    manifest: dataset.DatasetManifest = manifest_for(src)
    zipped: Path = make_zip(tmp_path / "a.zip", TREE)
    dest: Path = tmp_path / "dest"
    dataset.install(zipped, dest, manifest)
    assert (dest / "Training/Am/a.wav").read_bytes() == b"training am"
    assert (dest / "Test/Am/b.wav").read_bytes() == b"test am"
    assert dataset.check(dest, manifest) is None


def test_install_wrong_content_leaves_nothing(tmp_path: Path) -> None:
    src: Path = tmp_path / "src"
    write_tree(src, TREE)
    manifest: dataset.DatasetManifest = manifest_for(src)
    work: Path = tmp_path / "work"
    work.mkdir()
    zipped: Path = make_zip(work / "a.zip", {**TREE, "Test/Am/b.wav": b"other bytes"})
    dest: Path = work / "dest"
    with pytest.raises(dataset.DatasetError):
        dataset.install(zipped, dest, manifest)
    assert not dest.exists()
    assert sorted(p.name for p in work.iterdir()) == ["a.zip"]  # no temp directory left beside dest


@pytest.mark.parametrize("bad", ["../evil.wav", "ABSOLUTE", "Training/Am/readme.txt"])
def test_install_refuses_unsafe_members(tmp_path: Path, bad: str) -> None:
    src: Path = tmp_path / "src"
    write_tree(src, TREE)
    manifest: dataset.DatasetManifest = manifest_for(src)
    work: Path = tmp_path / "work"
    work.mkdir()
    outside: Path = tmp_path / "outside"
    outside.mkdir()
    member: str = str(outside / "evil.wav") if bad == "ABSOLUTE" else bad
    zipped: Path = make_zip(work / "a.zip", {**TREE, member: b"evil"})  # the honest content is all there
    dest: Path = work / "dest"
    with pytest.raises(dataset.DatasetError):
        dataset.install(zipped, dest, manifest)
    assert not dest.exists()
    assert list(outside.iterdir()) == []
    assert not (work / "evil.wav").exists()
    assert not (tmp_path / "evil.wav").exists()
    assert sorted(p.name for p in work.iterdir()) == ["a.zip"]


def test_install_into_a_verified_dest_touches_nothing(tmp_path: Path) -> None:
    dest: Path = tmp_path / "dest"
    write_tree(dest, TREE)
    manifest: dataset.DatasetManifest = manifest_for(dest)
    zipped: Path = make_zip(tmp_path / "a.zip", TREE)
    os.utime(dest / "Test/Am/b.wav", ns=(1_000_000_000, 1_000_000_000))
    before: dict[str, int] = snapshot(dest)
    dataset.install(zipped, dest, manifest)
    assert snapshot(dest) == before


# 8


def test_committed_manifest() -> None:
    m: dataset.DatasetManifest = dataset.DatasetManifest.load()
    assert m.files == 1760
    assert m.test_files == 320
    assert m.licence == "CC BY-SA 4.0"
    assert len(m.content_sha256) == 64 and int(m.content_sha256, 16) >= 0
    assert len(m.test_content_sha256) == 64 and int(m.test_content_sha256, 16) >= 0


# 9


def test_import_does_not_read_the_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The Docker image has no dataset.json: importing the module must not touch it."""
    monkeypatch.setattr(dataset, "DATASET_MANIFEST", tmp_path / "nowhere" / "dataset.json")

    original_read_text: Callable[..., str] = Path.read_text
    original_read_bytes: Callable[[Path], bytes] = Path.read_bytes

    def guarded_text(self: Path, encoding: str | None = None, errors: str | None = None) -> str:
        if self.name == "dataset.json":
            raise AssertionError("dataset.json read at import")
        text: str = original_read_text(self, encoding, errors)
        return text

    def guarded_bytes(self: Path) -> bytes:
        if self.name == "dataset.json":
            raise AssertionError("dataset.json read at import")
        data: bytes = original_read_bytes(self)
        return data

    monkeypatch.setattr(Path, "read_text", guarded_text)
    monkeypatch.setattr(Path, "read_bytes", guarded_bytes)
    importlib.reload(dataset)
    assert isinstance(dataset.DATASET_MANIFEST, Path)


# weights.github_fetch is generalised for the mirror


def test_release_fetch_is_generalised() -> None:
    fetch: weights.Fetch = weights.github_release_fetch(
        "owner/name", "some-tag", token=None
    )  # lazy: no network
    assert callable(fetch)
    assert callable(weights.github_fetch)


# 10


def test_cli_check_on_an_empty_dir_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src: Path = tmp_path / "src"
    write_tree(src, TREE)
    empty: Path = tmp_path / "empty"
    empty.mkdir()
    assert run_cli(["dataset", "--check", "--dest", str(empty)]) == 1
    assert capsys.readouterr().err.strip() != ""  # the reason


def test_cli_without_a_mirror_asks_for_the_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    src: Path = tmp_path / "src"
    write_tree(src, TREE)
    patch_manifest(monkeypatch, manifest_for(src, archive=None))

    def no_network(repo: str, tag: str, token: str | None = None) -> weights.Fetch:
        raise AssertionError("the network must not be touched")

    monkeypatch.setattr(weights, "github_release_fetch", no_network)
    assert run_cli(["dataset", "--dest", str(tmp_path / "dest")]) == 1
    assert "--archive" in capsys.readouterr().err


def test_cli_archive_installs_a_local_zip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src: Path = tmp_path / "src"
    write_tree(src, TREE)
    patch_manifest(monkeypatch, manifest_for(src))
    zipped: Path = make_zip(tmp_path / "archive.zip", TREE)
    dest: Path = tmp_path / "dest"
    assert run_cli(["dataset", "--dest", str(dest), "--archive", str(zipped)]) == 0
    assert run_cli(["dataset", "--check", "--dest", str(dest)]) == 0


def test_cli_verified_dest_downloads_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    dest: Path = tmp_path / "dest"
    write_tree(dest, TREE)
    patch_manifest(monkeypatch, manifest_for(dest))

    def no_network(repo: str, tag: str, token: str | None = None) -> weights.Fetch:
        raise AssertionError("the network must not be touched")

    monkeypatch.setattr(weights, "github_release_fetch", no_network)
    assert run_cli(["dataset", "--dest", str(dest)]) == 0
    assert "dataset is in place" in capsys.readouterr().out


# 11


def mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, served: bytes, pinned: bytes) -> Path:
    src: Path = tmp_path / "src"
    write_tree(src, TREE)
    pin: dataset.ArchivePin = dataset.ArchivePin(
        repo="owner/name",
        tag="dataset-tiny",
        name="tiny.zip",
        sha256=hashlib.sha256(pinned).hexdigest(),
        size=len(pinned),
    )
    patch_manifest(monkeypatch, manifest_for(src, archive=pin))
    asked: list[tuple[str, str, str]] = []

    def release_fetch(repo: str, tag: str, token: str | None = None) -> weights.Fetch:
        def fetch(name: str) -> bytes:
            asked.append((repo, tag, name))
            return served

        return fetch

    monkeypatch.setattr(weights, "github_release_fetch", release_fetch)
    return tmp_path / "dest"


def zip_bytes() -> bytes:
    buf: io.BytesIO = io.BytesIO()
    name: str
    data: bytes
    z: zipfile.ZipFile
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in TREE.items():
            z.writestr(name, data)
    return buf.getvalue()


def test_mirror_download_installs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    good: bytes = zip_bytes()
    dest: Path = mirror(tmp_path, monkeypatch, served=good, pinned=good)
    assert run_cli(["dataset", "--dest", str(dest)]) == 0
    assert (dest / "Training/Am/a.wav").read_bytes() == b"training am"
    assert run_cli(["dataset", "--check", "--dest", str(dest)]) == 0


def test_mirror_download_with_wrong_hash_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    good: bytes = zip_bytes()
    dest: Path = mirror(tmp_path, monkeypatch, served=good + b"\0", pinned=good)  # not the pinned bytes
    assert run_cli(["dataset", "--dest", str(dest)]) == 1
    assert not dest.exists()
    assert capsys.readouterr().err.strip() != ""
