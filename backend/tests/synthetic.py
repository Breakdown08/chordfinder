"""Tiny synthetic data for the training tests: sines per class, 16 kHz wav files, no dataset, no network."""

import hashlib
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from chordfinder.audio import Audio
from chordfinder.cli import main
from chordfinder.features import Meta

SR: int = 16000
CLASSES: tuple[str, ...] = ("Am", "Bb", "Bdim", "C", "Dm", "Em", "F", "G")


def meta() -> Meta:
    return Meta(sr=SR, n_mels=64, n_fft=1024, hop_length=256, threshold=0.01, classes=CLASSES)


def tone(cls: str, seconds: float, amplitude: float = 0.5) -> Audio:
    """A sine whose pitch depends on the class."""
    hz: float = 200.0 + 60.0 * CLASSES.index(cls)
    t: Audio = np.arange(int(SR * seconds), dtype=np.float32) / SR
    y: Audio = (amplitude * np.sin(2 * np.pi * hz * t)).astype(np.float32)
    return y


def write_wav(path: Path, y: Audio) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, SR, y)


def write_tree(root: Path, files: dict[str, bytes]) -> None:
    """files: "Training/Am/a.wav" -> bytes (not necessarily real audio)."""
    name: str
    data: bytes
    for name, data in files.items():
        p: Path = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


def digest_by_definition(root: Path, split: str | None = None) -> tuple[str, int]:
    """The spec's definition of the content hash, written out independently of the implementation."""
    lines: list[str] = []
    p: Path
    for p in root.glob("*/*/*.wav"):
        if split is not None and p.parent.parent.name != split:
            continue
        lines.append(f"{p.parent.parent.name}/{p.parent.name}/{hashlib.sha256(p.read_bytes()).hexdigest()}")
    body: str = "".join(line + "\n" for line in sorted(lines))
    return hashlib.sha256(body.encode()).hexdigest(), len(lines)


def synthetic_dataset(
    root: Path, groups: int = 2, takes: int = 2, seconds: float = 2.0, test_files: int = 1
) -> list[Path]:
    """Training/<cls>/<cls>_Plug<A,B..>_<take>.wav and Test/<cls>/<cls>_TestPlug_<n>.wav under root."""
    made: list[Path] = []
    cls: str
    for cls in CLASSES:
        g: int
        for g in range(groups):
            k: int
            for k in range(1, takes + 1):
                p: Path = root / "Training" / cls / f"{cls}_Plug{chr(ord('A') + g)}_{k}.wav"
                write_wav(p, tone(cls, seconds))
                made.append(p)
        n: int
        for n in range(1, test_files + 1):
            write_wav(root / "Test" / cls / f"{cls}_TestPlug_{n}.wav", tone(cls, seconds))
    return sorted(made)


def run_cli(argv: list[str]) -> int:
    """main() with argparse's SystemExit turned into a return code."""
    try:
        return main(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1
