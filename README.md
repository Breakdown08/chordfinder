# chordfinder

Guitar chord recognition: a CNN on log-mel spectrograms (8 chords: Am, Bb, Bdim, C, Dm, Em, F, G),
served through ONNX Runtime. Grew out of a master's thesis (2025); being rebuilt into a web game
where chords played on a real guitar build a bridge.

```sh
uv sync
uv run pytest                               # parity with the thesis model + preprocessing tests
uv run chordfinder eval --min-accuracy 0.95 # needs the dataset, see below
uv run python tools/bench.py                # cold start and per-window latency
uv run --group export chordfinder export    # models/chord-cnn.keras -> chord-cnn.onnx (TensorFlow)
```

## Model

- `models/chord-cnn.keras` — the thesis model as trained (Keras 3, Nov 2024); the source of truth.
- `models/chord-cnn.onnx` — its export, the only thing loaded at runtime (no TensorFlow).
- `baseline/old-model.json` — answers of the thesis code itself on the 320 test files; the ONNX port
  must give the same label on every file (`tests/test_parity.py`).

Test accuracy: **95.9%** (307/320) on instruments absent from the training split.

## Dataset

[fabianavinci/guitar-chords-v3](https://www.kaggle.com/datasets/fabianavinci/guitar-chords-v3)
(Kaggle; licence unknown, so it is not in the repository). Unpack it and point
`CHORDFINDER_DATASET` at the `data` directory (default `~/datasets/guitar-chords-v3/data`).
Tests that need it are skipped without it.

The thesis-era Django application lives in git history only.
