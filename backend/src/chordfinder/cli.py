"""chordfinder weights | export | eval | dataset | train | compare | load."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Literal

from chordfinder import dataset, evaluate
from chordfinder import model as model_module
from chordfinder import weights as weights_module
from chordfinder.evaluate import Comparison, Report
from chordfinder.features import Meta
from chordfinder.load import DEFAULT_STEPS, DEFAULT_URL, LoadConfig
from chordfinder.model import MODELS, ONNX

TRAINED: Path = MODELS / "trained"


def main(argv: list[str] | None = None) -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(prog="chordfinder")
    sub: argparse._SubParsersAction[argparse.ArgumentParser] = parser.add_subparsers(
        dest="command", required=True
    )
    w: argparse.ArgumentParser = sub.add_parser(
        "weights", help="download model weights from the GitHub Release"
    )
    w.add_argument("--all", action="store_true", help="also the Keras original, not only the runtime ONNX")
    mode: argparse._MutuallyExclusiveGroup = w.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="verify only; exit 1 on a missing or damaged file")
    mode.add_argument("--pin", action="store_true", help="write the local files' hashes into the manifest")
    sub.add_parser("export", help="convert models/chord-cnn.keras to ONNX (needs --group export)")
    ev: argparse.ArgumentParser = sub.add_parser("eval", help="accuracy on the dataset test split")
    ev.add_argument("--models", type=Path, default=MODELS, help="model directory (default: models/)")
    ev.add_argument("--min-accuracy", type=float, default=0.0, help="exit 1 below this, e.g. 0.95")
    ds: argparse.ArgumentParser = sub.add_parser("dataset", help="fetch and verify the pinned dataset")
    ds.add_argument("--dest", type=Path, help="dataset directory (default: CHORDFINDER_DATASET)")
    ds.add_argument("--archive", type=Path, help="unpack and verify a zip downloaded by hand")
    ds.add_argument("--check", action="store_true", help="verify only; exit 1 with the reason")
    tr: argparse.ArgumentParser = sub.add_parser(
        "train", help="train the model from scratch (needs --group train)"
    )
    tr.add_argument("--dataset", type=Path, help="dataset directory (default: CHORDFINDER_DATASET)")
    tr.add_argument("--out", type=Path, default=TRAINED, help="result directory (default: models/trained)")
    tr.add_argument("--epochs", type=int)
    tr.add_argument("--batch-size", type=int)
    tr.add_argument("--seed", type=int)
    tr.add_argument("--val", type=float, help="fraction of guitars held out for validation; 0: none")
    cp: argparse.ArgumentParser = sub.add_parser("compare", help="two models on the same test files")
    cp.add_argument("--old", type=Path, default=MODELS, help="default: models/ (the thesis weights)")
    cp.add_argument("--new", type=Path, default=TRAINED, help="default: models/trained")
    cp.add_argument("--dataset", type=Path, help="dataset directory (default: CHORDFINDER_DATASET)")
    cp.add_argument("--json", type=Path, help="write the report here")
    ld: argparse.ArgumentParser = sub.add_parser(
        "load", help="step up simulated microphones against a running server and report the ceiling"
    )
    defaults: LoadConfig = LoadConfig()
    ld.add_argument("--url", default=DEFAULT_URL, help="WebSocket address (default: %(default)s)")
    ld.add_argument(
        "--steps",
        type=steps_arg,
        default=DEFAULT_STEPS,
        help="client counts, strictly increasing (default: " + ",".join(map(str, DEFAULT_STEPS)) + ")",
    )
    ld.add_argument("--duration", type=float, default=defaults.duration, help="seconds of sending per step")
    ld.add_argument("--warmup", type=float, default=defaults.warmup, help="first seconds not measured")
    ld.add_argument("--grace", type=float, default=defaults.grace, help="wait for the last replies")
    ld.add_argument("--pause", type=float, default=defaults.pause, help="pause between steps")
    ld.add_argument("--rate", type=int, default=defaults.sample_rate, help="sample rate of the frames, Hz")
    ld.add_argument("--source", choices=("synthetic", "dataset"), default="synthetic")
    ld.add_argument("--dataset", type=Path, help="dataset directory (default: CHORDFINDER_DATASET)")
    ld.add_argument("--seed", type=int, default=defaults.seed)
    metrics_mode: argparse._MutuallyExclusiveGroup = ld.add_mutually_exclusive_group()
    metrics_mode.add_argument("--metrics-url", help="default: /metrics of the same host")
    metrics_mode.add_argument("--no-metrics", action="store_true", help="do not read /metrics")
    ld.add_argument("--json", type=Path, help="write the report here")
    args: argparse.Namespace = parser.parse_args(argv)

    if args.command == "weights":
        return weights(args.all, args.check, args.pin)

    if args.command == "export":
        from chordfinder.export import export

        print("wrote", export(MODELS))
        return 0

    if args.command == "dataset":
        return dataset_command(args.dest or evaluate.DATASET, args.archive, args.check)

    if args.command == "train":
        return train_command(args)

    if args.command == "load":
        return load_command(args)

    if args.command == "compare":
        return compare_command(args.old, args.new, args.dataset or evaluate.DATASET, args.json)

    model: evaluate.Predictor = model_module.ChordModel(args.models)
    report: Report = evaluate.evaluate(model, evaluate.dataset_files())
    print(f"accuracy {report.accuracy:.1%} ({report.hits}/{len(report.predictions)})")
    cls: str
    acc: float
    for cls, acc in report.per_class().items():
        print(f"  {cls:5} {acc:.1%}")
    print_confusion(report, model.meta.classes)
    min_accuracy: float = args.min_accuracy
    if report.accuracy < min_accuracy:
        print(f"below threshold {min_accuracy:.1%}", file=sys.stderr)
        return 1
    return 0


def steps_arg(text: str) -> tuple[int, ...]:
    try:
        steps: tuple[int, ...] = tuple(int(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a comma-separated list of integers: {text!r}") from None
    if any(n <= 0 for n in steps) or any(a >= b for a, b in zip(steps, steps[1:], strict=False)):
        raise argparse.ArgumentTypeError(f"must be positive and strictly increasing: {text!r}")
    return steps


def load_command(args: argparse.Namespace) -> int:
    import asyncio

    from chordfinder import load
    from chordfinder.audio import Audio

    source: Literal["synthetic", "dataset"] = args.source
    signal: Audio
    if source == "dataset":
        root: Path = args.dataset or evaluate.DATASET
        reason: str | None = dataset.check(root, dataset.DatasetManifest.load())
        if reason is not None:
            print(f"dataset {root}: {reason}; run chordfinder dataset", file=sys.stderr)
            return 1
        signal = load.dataset_signal(root, args.rate)
    else:
        signal = load.synthetic_signal(args.rate, 10.0, args.seed)
    why: str | None = asyncio.run(load.preflight(args.url, args.rate, signal))
    if why is not None:
        print(f"load: server not reachable at {args.url}: {why}", file=sys.stderr)
        return 1
    metrics: str | None = None if args.no_metrics else args.metrics_url or load.metrics_url(args.url)
    if metrics is not None and asyncio.run(load.scrape(metrics)) is None:
        print(f"load: {metrics} is not available, continuing without server metrics", file=sys.stderr)
        metrics = None
    config: LoadConfig = LoadConfig(
        url=args.url,
        metrics=metrics,
        sample_rate=args.rate,
        duration=args.duration,
        warmup=args.warmup,
        grace=args.grace,
        pause=args.pause,
        seed=args.seed,
    )
    print(load.HEADER, flush=True)
    report: load.LoadReport = asyncio.run(
        load.run(config, args.steps, signal, source, lambda r: print(load.format_step(r), flush=True))
    )
    print(load.format_ceiling(report))
    if args.json is not None:
        args.json.write_text(report.model_dump_json(indent=2) + "\n")
    return 0


def print_confusion(report: Report, classes: list[str] | tuple[str, ...], indent: str = "") -> None:
    """Rows are the truth, columns the predicted chord."""
    matrix: list[list[int]] = report.confusion(classes)
    width: int = max(5, *(len(c) for c in classes)) + 1
    print(indent + "truth\\pred".ljust(width) + "".join(c.rjust(width) for c in classes))
    c: str
    row: list[int]
    for c, row in zip(classes, matrix, strict=True):
        print(indent + c.ljust(width) + "".join(str(n).rjust(width) for n in row))


def weights(everything: bool, check_only: bool, pin_only: bool) -> int:
    from chordfinder import weights as w

    manifest: w.Manifest = w.Manifest.load()
    if pin_only:
        w.save(w.pin(manifest, MODELS))
        print(f"pinned {', '.join(manifest.files)}: upload to release {manifest.tag}, commit the manifest")
        return 0
    if check_only:
        problems: dict[str, str] = w.check(manifest, MODELS, everything)
        name: str
        problem: str
        for name, problem in problems.items():
            print(f"{name}: {problem}", file=sys.stderr)
        return 1 if problems else 0
    try:
        fetched: list[str] = w.download(manifest, MODELS, everything, w.github_fetch(manifest))
    except (w.WeightsError, OSError) as e:
        print(f"weights: {e}", file=sys.stderr)
        return 1
    print(f"downloaded {', '.join(fetched)}" if fetched else "weights are in place")
    return 0


def dataset_command(dest: Path, archive: Path | None, check_only: bool) -> int:
    manifest: dataset.DatasetManifest = dataset.DatasetManifest.load()
    reason: str | None = dataset.check(dest, manifest)
    if check_only:
        if reason is not None:
            print(f"dataset {dest}: {reason}", file=sys.stderr)
        return 0 if reason is None else 1
    if reason is None:
        print("dataset is in place")
        return 0
    try:
        if archive is not None:
            dataset.install(archive, dest, manifest)
        elif manifest.archive is None:
            print(
                f"no mirror is pinned: download archive.zip from {manifest.source}, "
                "then chordfinder dataset --archive archive.zip",
                file=sys.stderr,
            )
            return 1
        else:
            pin: dataset.ArchivePin = manifest.archive
            fetch: weights_module.Fetch = weights_module.github_release_fetch(pin.repo, pin.tag)
            dataset.download(manifest, dest, fetch)
    except (dataset.DatasetError, weights_module.WeightsError, OSError) as e:
        print(f"dataset: {e}", file=sys.stderr)
        return 1
    print(f"dataset installed into {dest}")
    return 0


def train_command(args: argparse.Namespace) -> int:
    root: Path = args.dataset or evaluate.DATASET
    reason: str | None = dataset.check(root, dataset.DatasetManifest.load())
    if reason is not None:
        print(f"dataset {root}: {reason}; run chordfinder dataset", file=sys.stderr)
        return 1
    try:
        import keras  # noqa: F401
        import tensorflow  # noqa: F401
    except ImportError:
        print(
            "chordfinder train needs the train group: uv run --group train chordfinder train", file=sys.stderr
        )
        return 2
    from chordfinder import train

    given: dict[str, int | float | None] = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "val_fraction": args.val,
    }
    config: train.TrainConfig = train.TrainConfig().model_copy(
        update={k: v for k, v in given.items() if v is not None}
    )
    result: train.TrainResult = train.train(root, args.out, config, Meta.load(MODELS / "meta.json"))
    print(f"windows: {result.train_windows} train, {result.val_windows} validation")
    if result.val_accuracy is not None:
        print(f"validation accuracy {result.val_accuracy:.1%}")
    print(f"onnx_max_abs_diff {result.onnx_max_abs_diff:.2e}")
    print(f"{result.seconds:.0f} s, written to {result.out}")
    return 0


def side(path: Path, report: Report, classes: tuple[str, ...]) -> dict[str, object]:
    return {
        "path": str(path),
        "onnx_sha256": hashlib.sha256((path / ONNX).read_bytes()).hexdigest(),
        "accuracy": report.accuracy,
        "per_class": report.per_class(),
        "confusion": report.confusion(classes),
    }


def compare_command(old_dir: Path, new_dir: Path, root: Path, json_path: Path | None) -> int:
    manifest: dataset.DatasetManifest = dataset.DatasetManifest.load()
    test_digest: str
    test_count: int
    test_digest, test_count = dataset.content_digest(root, "Test")
    if (test_digest, test_count) != (manifest.test_content_sha256, manifest.test_files):
        print(f"dataset {root}: the Test split is not the pinned one ({test_count} files)", file=sys.stderr)
        return 1
    d: Path
    for d in (old_dir, new_dir):
        if not (d / ONNX).is_file():
            print(f"no {ONNX} in {d}", file=sys.stderr)
            return 1
    old_model: evaluate.Predictor = model_module.ChordModel(old_dir)
    new_model: evaluate.Predictor = model_module.ChordModel(new_dir)
    if old_model.meta.classes != new_model.meta.classes:
        print("the two models have different classes", file=sys.stderr)
        return 1
    classes: tuple[str, ...] = old_model.meta.classes
    files: list[Path] = evaluate.dataset_files(root)
    c: Comparison = evaluate.compare(evaluate.evaluate(old_model, files), evaluate.evaluate(new_model, files))

    print(
        f"accuracy  old {c.old.accuracy:.1%} ({c.old.hits}/{len(c.old.predictions)})  "
        f"new {c.new.accuracy:.1%} ({c.new.hits}/{len(c.new.predictions)})"
    )
    old_class: dict[str, float] = c.old.per_class()
    new_class: dict[str, float] = c.new.per_class()
    print(f"  {'':5} {'old':>7} {'new':>7}")
    cls: str
    for cls in classes:
        print(f"  {cls:5} {old_class.get(cls, 0.0):7.1%} {new_class.get(cls, 0.0):7.1%}")
    print("confusion, old (rows: truth, columns: predicted)")
    print_confusion(c.old, classes, "  ")
    print("confusion, new")
    print_confusion(c.new, classes, "  ")
    print(f"old_only {c.old_only}  new_only {c.new_only}  both_wrong {c.both_wrong}", end="  ")
    print(f"mcnemar_p {c.mcnemar_p:.4g}")

    if json_path is not None:
        new_by_file: dict[str, evaluate.Prediction] = {p.file: p for p in c.new.predictions}
        report: dict[str, object] = {
            "dataset": f"{manifest.name} Test",
            "test_content_sha256": test_digest,
            "classes": list(classes),
            "models": {"old": side(old_dir, c.old, classes), "new": side(new_dir, c.new, classes)},
            "paired": {
                "old_only": c.old_only,
                "new_only": c.new_only,
                "both_wrong": c.both_wrong,
                "mcnemar_p": c.mcnemar_p,
            },
            "predictions": [
                {
                    "file": p.file,
                    "truth": p.truth,
                    "old": p.label,
                    "old_confidence": p.confidence,
                    "new": new_by_file[p.file].label,
                    "new_confidence": new_by_file[p.file].confidence,
                }
                for p in c.old.predictions
            ],
        }
        json_path.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
