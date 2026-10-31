"""chordfinder weights | export | eval."""

import argparse
import sys

from chordfinder.model import MODELS


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
    ev.add_argument("--min-accuracy", type=float, default=0.0, help="exit 1 below this, e.g. 0.95")
    args: argparse.Namespace = parser.parse_args(argv)

    if args.command == "weights":
        return weights(args.all, args.check, args.pin)

    if args.command == "export":
        from chordfinder.export import export

        print("wrote", export(MODELS))
        return 0

    from chordfinder.evaluate import Report, dataset_files, evaluate
    from chordfinder.model import ChordModel

    report: Report = evaluate(ChordModel(), dataset_files())
    print(f"accuracy {report.accuracy:.1%} ({report.hits}/{len(report.predictions)})")
    cls: str
    acc: float
    for cls, acc in report.per_class().items():
        print(f"  {cls:5} {acc:.1%}")
    min_accuracy: float = args.min_accuracy
    if report.accuracy < min_accuracy:
        print(f"below threshold {min_accuracy:.1%}", file=sys.stderr)
        return 1
    return 0


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


if __name__ == "__main__":
    sys.exit(main())
