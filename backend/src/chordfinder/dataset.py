"""Dataset pin: content digest, check, safe install, mirror download.

backend/dataset.json pins the dataset by a hash of its content (file names do not matter: the Kaggle
archive has names in cp437 garbage that unpackers spell differently). Nothing here reads the manifest
at import: the Docker image does not carry it.
"""

import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import IO

from pydantic import BaseModel

from chordfinder.weights import Fetch

DATASET_MANIFEST: Path = Path(__file__).resolve().parents[2] / "dataset.json"
SPLITS: tuple[str, str] = ("Training", "Test")


class ArchivePin(BaseModel):
    repo: str
    tag: str
    name: str
    sha256: str
    size: int


class DatasetManifest(BaseModel):
    name: str
    source: str
    author: str
    licence: str
    files: int
    content_sha256: str
    test_files: int
    test_content_sha256: str
    archive: ArchivePin | None

    @classmethod
    def load(cls, path: Path = DATASET_MANIFEST) -> "DatasetManifest":
        return cls.model_validate_json(path.read_text())


class DatasetError(RuntimeError):
    pass


def content_digest(root: Path, split: str | None = None) -> tuple[str, int]:
    """(digest, number of wav files) over root/<split>/<class>/*.wav; split=None: all splits."""
    pattern: str = "*/*/*.wav" if split is None else f"{split}/*/*.wav"
    lines: list[str] = []
    p: Path
    for p in root.glob(pattern):
        if p.is_file():
            lines.append(
                f"{p.parent.parent.name}/{p.parent.name}/{hashlib.sha256(p.read_bytes()).hexdigest()}"
            )
    body: str = "".join(line + "\n" for line in sorted(lines))
    return hashlib.sha256(body.encode()).hexdigest(), len(lines)


def check(root: Path, manifest: DatasetManifest) -> str | None:
    """None if root holds exactly the pinned dataset; otherwise a one-line reason."""
    if not root.is_dir():
        return "missing"
    digest: str
    count: int
    digest, count = content_digest(root)
    if count != manifest.files:
        return f"{count} files, manifest pins {manifest.files}"
    if digest != manifest.content_sha256:
        return "content differs from the manifest"
    return None


def member_target(name: str) -> tuple[str, str, str] | None:
    """(split, class, file name) of a safe archive member, None for a directory entry."""
    if name.endswith("/"):
        parts: tuple[str, ...] = PurePosixPath(name).parts
        if name.startswith("/") or ".." in parts or "\\" in name or len(parts) > 2:
            raise DatasetError(f"unsafe archive member {name!r}")
        return None
    parts = PurePosixPath(name).parts
    safe: bool = (
        not name.startswith("/")
        and "\\" not in name
        and len(parts) == 3
        and parts[0] in SPLITS
        and ".." not in parts
        and parts[1] not in ("", ".")
        and parts[2].endswith(".wav")
    )
    if not safe:
        raise DatasetError(f"archive member {name!r} is not <Training|Test>/<class>/<name>.wav")
    return parts[0], parts[1], parts[2]


def install(archive: Path, dest: Path, manifest: DatasetManifest) -> None:
    """Unpack a zip into dest (dest/Training, dest/Test) and verify; on any problem raise DatasetError
    and leave dest untouched. Unpacks into a sibling temp dir, renames on success."""
    if check(dest, manifest) is None:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path = Path(tempfile.mkdtemp(prefix=f".{dest.name}-", dir=dest.parent))
    try:
        try:
            z: zipfile.ZipFile
            with zipfile.ZipFile(archive) as z:
                info: zipfile.ZipInfo
                targets: list[tuple[zipfile.ZipInfo, tuple[str, str, str]]] = []
                for info in z.infolist():
                    target: tuple[str, str, str] | None = member_target(info.filename)
                    if target is not None:
                        targets.append((info, target))
                for info, target in targets:
                    src: IO[bytes]
                    dst: IO[bytes]
                    out: Path = tmp / target[0] / target[1] / target[2]
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(info) as src, out.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
        except (zipfile.BadZipFile, OSError) as e:
            raise DatasetError(f"cannot unpack {archive}: {e}") from e
        reason: str | None = check(tmp, manifest)
        if reason is not None:
            raise DatasetError(f"{archive} is not the pinned dataset: {reason}")
        try:
            tmp.rename(dest)
        except OSError as e:
            raise DatasetError(f"cannot put the dataset into {dest}: {e}") from e
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def download(manifest: DatasetManifest, dest: Path, fetch: Fetch) -> None:
    """The mirror archive of the manifest: size and SHA-256 are checked before anything is unpacked."""
    pin: ArchivePin | None = manifest.archive
    if pin is None:
        raise DatasetError("the manifest pins no mirror archive")
    data: bytes = fetch(pin.name)
    digest: str = hashlib.sha256(data).hexdigest()
    if len(data) != pin.size or digest != pin.sha256:
        raise DatasetError(
            f"{pin.name}: downloaded {len(data)} bytes with sha256 {digest}, manifest pins {pin}"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    work: Path
    t: str
    with tempfile.TemporaryDirectory(prefix=f".{dest.name}-zip-", dir=dest.parent) as t:
        work = Path(t) / pin.name
        work.write_bytes(data)
        install(work, dest, manifest)
