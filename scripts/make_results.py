"""Build results/results.csv: one row per hackathon target, measured value, pass/fail.

Every number is read from a file another step wrote, so nothing is typed by hand:

    --eval        eval_*.json from scripts/evaluate.py (headline accuracy; use the
                  held-out-signer split, see docs/CLOUD_PROCEDURE.md)
    --eval-test   eval_test_int8.json (official WLASL test clips, reported for reference)
    --robustness  results/robustness_landmark.json and/or robustness_video.json
    --run-key     key of the shipped model inside the robustness files, e.g. E4+E13+E12
    --device      results_device.csv saved by the app's Bench button (phone)
    --predictions predictions.jsonl from main.py (laptop/CPU latency, for reference)

    python scripts/make_results.py --eval runs/ens/eval_holdout_int8.json \\
        --eval-test runs/ens/eval_test_int8.json \\
        --robustness results/robustness_landmark.json results/robustness_video.json \\
        --run-key E4+E13+E12 --device results/results_device.csv \\
        --predictions results/predictions.jsonl

Missing inputs give an empty "measured" cell, never a guess.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def p95(v):
    return float(np.percentile(v, 95)) if len(v) else None


def drop(path, key):
    """Mean drop in points (clean - condition) over all conditions in one robustness file."""
    res = json.loads(Path(path).read_text())["results"]
    r = res[key]
    drops = [100 * (r["clean"] - a) for name, a in r.items() if name != "clean"]
    worst = max(((100 * (r["clean"] - a), n) for n, a in r.items() if n != "clean"), default=(None, None))
    return float(np.mean(drops)), worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval")
    ap.add_argument("--eval-test")
    ap.add_argument("--robustness", nargs="*", default=[])
    ap.add_argument("--run-key", default="E4+E13+E12")
    ap.add_argument("--device")
    ap.add_argument("--predictions")
    ap.add_argument("--model", default=str(ROOT / "models" / "model_int8.onnx"))
    ap.add_argument("--out", default="results/results.csv")
    args = ap.parse_args()

    rows = []

    def add(metric, measured, unit, target, ok, source, note=""):
        m = "" if measured is None else (round(measured, 2) if isinstance(measured, float) else measured)
        status = "" if measured is None or ok is None else ("PASS" if ok else "FAIL")
        rows.append({"metric": metric, "measured": m, "unit": unit, "target": target,
                     "status": status, "source": source, "note": note})

    if args.eval:
        e = json.loads(Path(args.eval).read_text())
        lo, hi = e.get("top1_95ci", [None, None])
        add("top1_accuracy", 100 * e["top1"], "%", ">= 70", 100 * e["top1"] >= 70, Path(args.eval).name,
            f"{e['clips']} clips, split {e['split']}, 95% CI {100 * lo:.1f}-{100 * hi:.1f}")
        add("top5_accuracy", 100 * e["top5"], "%", "", None, Path(args.eval).name)
    else:
        add("top1_accuracy", None, "%", ">= 70", None, "", "pass --eval")
    if args.eval_test:
        e = json.loads(Path(args.eval_test).read_text())
        note = f"{e['clips']} official WLASL100 test clips we could obtain"
        if "unseen_signers" in e:
            note += f"; signers also in train: {e['seen_signers']['clips']} clips, unseen: {e['unseen_signers']['clips']}"
        add("top1_official_test", 100 * e["top1"], "%", "reference", None, Path(args.eval_test).name, note)

    size_mb = Path(args.model).stat().st_size / 1e6 if Path(args.model).exists() else None
    add("model_size", size_mb, "MB", "<= 20", None if size_mb is None else size_mb <= 20,
        Path(args.model).name, "INT8 ONNX")

    for path in args.robustness:
        level = Path(path).stem.replace("robustness_", "")
        mean, (worst, wname) = drop(path, args.run_key)
        add(f"robustness_drop_{level}", mean, "points", "<= 5", mean <= 5, Path(path).name,
            f"mean over conditions; worst {wname} {worst:.1f}")

    dev = {}
    if args.device and Path(args.device).exists():
        with open(args.device) as f:
            dev = {r["metric"]: r for r in csv.DictReader(f)}
    device_name = next(iter(dev.values()))["device"] if dev else "no phone run yet"

    def devval(k):
        try:
            return float(dev[k]["measured"])
        except (KeyError, ValueError):
            return None

    lm, fm = devval("latency_p95_landmarks_per_frame"), devval("latency_p95_features_model")
    add("latency_p95_per_frame_phone", lm, "ms", "<= 50", None if lm is None else lm <= 50,
        "results_device.csv", f"{device_name}: MediaPipe Holistic per camera frame (the real-time cost)")
    add("latency_p95_classifier_phone", fm, "ms", "<= 50", None if fm is None else fm <= 50,
        "results_device.csv", f"{device_name}: features + INT8 model, once per sign")
    pw = devval("avg_power")
    add("avg_power_phone", pw, "mW", "<= 500", None if pw is None else pw <= 500, "results_device.csv",
        f"{device_name}: whole phone, unplugged")
    td = devval("thermal_drift_10min")
    add("thermal_drift_10min_phone", td, "%", "<= 20", None if td is None else td <= 20, "results_device.csv",
        f"{device_name}: landmark latency, first vs last minute")

    if args.predictions:
        preds = [json.loads(line) for line in open(args.predictions) if line.strip()]
        timed = [p for p in preds if p.get("frames")]
        timed = timed[1:] if len(timed) > 1 else timed  # first clip includes model loading
        per_frame = [p["landmark_ms_per_frame"] for p in timed]
        clf = [p["latency_ms"] for p in timed]
        add("latency_p95_per_frame_cpu", p95(per_frame), "ms", "reference", None, Path(args.predictions).name,
            f"{len(timed)} clips (first skipped: warm-up), computer CPU, not the phone")
        add("latency_p95_classifier_cpu", p95(clf), "ms", "reference", None, Path(args.predictions).name,
            "computer CPU, 1 thread")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['metric']:32s} {str(r['measured']):>8s} {r['unit']:6s} {r['target']:10s} {r['status']:4s} {r['note']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
