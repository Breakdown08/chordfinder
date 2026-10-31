"""chordfinder export | eval."""

import argparse
import sys

from chordfinder.model import MODELS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chordfinder")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("export", help="convert models/chord-cnn.keras to ONNX (needs --group export)")
    ev = sub.add_parser("eval", help="accuracy on the dataset test split")
    ev.add_argument("--min-accuracy", type=float, default=0.0, help="exit 1 below this, e.g. 0.95")
    args = parser.parse_args(argv)

    if args.command == "export":
        from chordfinder.export import export

        print("wrote", export(MODELS))
        return 0

    from chordfinder.evaluate import evaluate, test_files
    from chordfinder.model import ChordModel

    files = test_files()
    report = evaluate(ChordModel(), files)
    hits = sum(p.label == p.truth for p in report.predictions)
    print(f"accuracy {report.accuracy:.1%} ({hits}/{len(files)})")
    for cls, acc in report.per_class().items():
        print(f"  {cls:5} {acc:.1%}")
    if report.accuracy < args.min_accuracy:
        print(f"below threshold {args.min_accuracy:.1%}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
