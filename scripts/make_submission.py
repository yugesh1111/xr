"""Build the submission bundle in the layout the Track 02 slides ask for:

    <team>.zip
    ├── src/
    │   ├── requirements.txt     packages needed to run predict()
    │   ├── main.py              predict(video_path) -> dict
    │   ├── README.md            how to set up and run
    │   ├── echogesture/         landmarks, features (used by main.py)
    │   ├── models/              INT8 model, word list, MediaPipe Holistic model
    │   └── scripts/, tests/, Dockerfile, requirements-train.txt   (training + conversion, for reproducibility)
    └── results/
        ├── predictions.json     one JSON object per line, one line per test clip
        ├── predictions.jsonl    the same file under the name used on the guidelines slide
        └── results.csv

    python scripts/make_submission.py --team Palmcast
"""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SRC_README = """# {team} - Track 02, EchoGesture: Sign Language Prediction on Edge Device

Python 3.11.

```bash
cd src
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py --videos /path/to/clips --out ../results/predictions.json
```

```python
from main import predict
predict("clip.mp4")
# {{"clip_id": "clip", "prediction": "BIRD", "top5": [...], "confidence": 0.87, "latency_ms": 9.8, ...}}
```

- `main.py` - `predict(video_path) -> dict`. Runs fully on the device: MediaPipe
  Holistic landmarks -> features -> INT8 ONNX model (`models/model_int8.onnx`,
  6.2 MB). No network access at run time; all models are in `models/`.
- `clip_id` is the file name without its extension. `prediction` is an upper-case
  word from the fixed list in `models/model_meta.json` (WLASL100).
- `latency_ms` is features + classifier for the clip; per-frame landmark time
  is reported next to it (`landmark_ms_per_frame`).
- Conversion script: `scripts/export_onnx.py` (checkpoint -> ONNX FP32 -> INT8).
- Training, evaluation and the Android app are described in the repository:
  https://github.com/yugesh1111/xr
- Headless Linux: `apt-get install libegl1 libgles2 libgl1 libglib2.0-0`
  (MediaPipe needs EGL), or use the `Dockerfile`.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="Palmcast")
    ap.add_argument("--out", default="dist")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(exist_ok=True)
    z = out / f"{args.team}.zip"
    skip = {"__pycache__", ".pytest_cache"}
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
        def add(path, arc):
            f.write(path, arc)

        def add_dir(d, arc):
            for p in sorted((ROOT / d).rglob("*")):
                if p.is_file() and not (set(p.parts) & skip) and p.suffix != ".pyc":
                    add(p, f"{arc}/{p.relative_to(ROOT / d)}")

        add(ROOT / "main.py", "src/main.py")
        add(ROOT / "requirements-inference.txt", "src/requirements.txt")
        add(ROOT / "requirements.txt", "src/requirements-train.txt")
        add(ROOT / "Dockerfile", "src/Dockerfile")
        f.writestr("src/README.md", SRC_README.format(team=args.team))
        for d in ("echogesture", "scripts", "tests"):
            add_dir(d, f"src/{d}")
        for name in ("model_int8.onnx", "model_meta.json", "holistic_landmarker.task", "gate_int8.onnx", "gate_meta.json"):
            if (ROOT / "models" / name).exists():
                add(ROOT / "models" / name, f"src/models/{name}")
        add(ROOT / "results" / "predictions.jsonl", "results/predictions.json")
        add(ROOT / "results" / "predictions.jsonl", "results/predictions.jsonl")
        add(ROOT / "results" / "results.csv", "results/results.csv")
    print(f"{z}  {z.stat().st_size / 1e6:.1f} MB")
    with zipfile.ZipFile(z) as f:
        names = f.namelist()
    for n in ("src/main.py", "src/requirements.txt", "src/README.md", "results/predictions.json", "results/results.csv"):
        assert n in names, n
    print("layout ok:", len(names), "files")


if __name__ == "__main__":
    main()
