"""Build results/results.csv in the brief's format (Track 02, page 11):
columns metric, target, measured, unit, device, runtime, pass and the six
metrics top1_accuracy, latency_p95, model_size, accuracy_drop_robustness,
avg_power, thermal_drift_10min. A longer table with the sources and notes
behind every number goes to results/results_detailed.csv.

Every number is read from a file another step wrote, so nothing is typed by hand:

    --eval        one or more eval_*.json from scripts/evaluate.py (headline accuracy;
                  use the held-out-signer splits, see docs/CLOUD_PROCEDURE.md);
                  several files (seeds) are averaged
    --eval-test   eval_test_int8.json (official WLASL test clips, reported for reference)
    --robustness  results/robustness_landmark.json and/or robustness_video.json
    --run-key     key of the shipped model inside the robustness files, e.g. e4+e13+e12
    --robustness-level  which robustness file goes into the brief's single row
                  (default video: pixels degraded, then MediaPipe re-run;
                  video_enhanced: the same through the deployed pipeline,
                  i.e. with adaptive sharpening - robustness.py --enhance)
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
    ap.add_argument("--eval", nargs="*", default=[])
    ap.add_argument("--eval-test")
    ap.add_argument("--robustness", nargs="*", default=[])
    ap.add_argument("--run-key", default="E4+E13+E12")
    ap.add_argument("--device")
    ap.add_argument("--predictions")
    ap.add_argument("--model", default=str(ROOT / "models" / "model_int8.onnx"))
    ap.add_argument("--robustness-level", default="video")
    ap.add_argument("--out", default="results/results.csv")
    ap.add_argument("--sustained", help="results/sustained_cpu.json from scripts/sustained_run.py (used when there is no phone run)")
    ap.add_argument("--cpu-name", default="laptop CPU (no phone run)", help="device label for CPU latency")
    ap.add_argument("--out-detailed", default="results/results_detailed.csv")
    args = ap.parse_args()

    rows = []

    def add(metric, measured, unit, target, ok, source, note=""):
        m = "" if measured is None else (round(measured, 2) if isinstance(measured, float) else measured)
        status = "" if measured is None or ok is None else ("PASS" if ok else "FAIL")
        rows.append({"metric": metric, "measured": m, "unit": unit, "target": target,
                     "status": status, "source": source, "note": note})

    brief = {}  # metric -> (measured, device, runtime)
    if args.eval:
        evals = [json.loads(Path(p).read_text()) for p in args.eval]
        top1 = float(np.mean([e["top1"] for e in evals]))
        parts = []
        for p, e in zip(args.eval, evals):
            lo, hi = e.get("top1_95ci", [0, 0])
            parts.append(f"{Path(p).parent.name}: {100 * e['top1']:.1f}% ({e['clips']} clips, 95% CI {100 * lo:.1f}-{100 * hi:.1f})")
        add("top1_accuracy", 100 * top1, "%", ">= 70", 100 * top1 >= 70, ", ".join(Path(p).name for p in args.eval),
            f"mean over {len(evals)} held-out signer groups: " + "; ".join(parts))
        add("top5_accuracy", 100 * float(np.mean([e["top5"] for e in evals])), "%", "", None, "")
        brief["top1_accuracy"] = (100 * top1, "offline evaluation (computer CPU)", "ONNX Runtime INT8, CPU")
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
    brief["model_size"] = (size_mb, "any", "ONNX INT8 (static, per-channel)")

    for path in args.robustness:
        level = Path(path).stem.replace("robustness_", "")
        mean, (worst, wname) = drop(path, args.run_key)
        add(f"robustness_drop_{level}", mean, "points", "<= 5", mean <= 5, Path(path).name,
            f"mean over conditions; worst {wname} {worst:.1f}")
        if level == args.robustness_level:
            brief["accuracy_drop_robustness"] = (mean, "offline evaluation (computer CPU)",
                                                 "video-level corruptions through the deployed pipeline "
                                                 "(adaptive sharpening + MediaPipe), mean over conditions"
                                                 if level == "video_enhanced" else
                                                 f"{level}-level corruptions, mean over conditions")

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
    runtime = "MediaPipe Holistic + ONNX Runtime INT8"
    brief["latency_p95"] = (lm, device_name if lm is not None else "", runtime + ", per camera frame")
    pw = devval("avg_power")
    add("avg_power_phone", pw, "mW", "<= 500", None if pw is None else pw <= 500, "results_device.csv",
        f"{device_name}: whole phone, unplugged")
    brief["avg_power"] = (pw, device_name if pw is not None else "", "whole phone, unplugged")
    td = devval("thermal_drift_10min")
    add("thermal_drift_10min_phone", td, "%", "<= 20", None if td is None else td <= 20, "results_device.csv",
        f"{device_name}: landmark latency, first vs last minute")

    brief["thermal_drift_10min"] = (td, device_name if td is not None else "", "10-minute live run")
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
        # no phone run: report the same quantity predict() self-reports as latency_ms
        # (features + INT8 model per clip), measured on this laptop, and say so
        if brief.get("latency_p95", (None,))[0] is None and not args.sustained:
            brief["latency_p95"] = (p95(clf), args.cpu_name,
                                    "ONNX Runtime INT8, features + classifier per clip (latency_ms in predictions.jsonl); "
                                    "landmarks per frame reported separately in results_detailed.csv")

    # no phone run: a 10-minute sustained run of the same pipeline on this computer
    # (scripts/sustained_run.py), labelled as such
    if args.sustained and Path(args.sustained).exists() and brief.get("latency_p95", (None,))[0] is None:
        su = json.loads(Path(args.sustained).read_text())
        dev_name = f"{su['device']} - no phone run"
        add("latency_p95_per_frame_sustained", su["landmark_ms_per_frame_p95"], "ms", "<= 50",
            su["landmark_ms_per_frame_p95"] <= 50, Path(args.sustained).name, f"{dev_name}: MediaPipe Holistic per frame, {su['minutes']:.0f} min")
        add("latency_p95_classifier_sustained", su["classifier_ms_p95"], "ms", "<= 50",
            su["classifier_ms_p95"] <= 50, Path(args.sustained).name, f"{dev_name}: features + INT8 model per sign")
        add("thermal_drift_10min_sustained", su["thermal_drift_pct"], "%", "<= 20", su["thermal_drift_pct"] <= 20,
            Path(args.sustained).name, f"{dev_name}: landmark latency, last vs first minute")
        brief["latency_p95"] = (su["landmark_ms_per_frame_p95"], dev_name,
                                f"MediaPipe Holistic + ONNX Runtime INT8 (CPU), per camera frame, {su['minutes']:.0f}-min sustained run; "
                                f"classifier p95 {su['classifier_ms_p95']} ms per sign")
        brief["thermal_drift_10min"] = (su["thermal_drift_pct"], dev_name,
                                        f"{su['minutes']:.0f}-min sustained run, per-frame latency last vs first minute")

    detailed = Path(args.out_detailed)
    detailed.parent.mkdir(parents=True, exist_ok=True)
    with open(detailed, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # the brief's sheet: fixed metrics, targets as written in the brief
    spec = [("top1_accuracy", ">=70", "%", lambda v: v >= 70),
            ("latency_p95", "<=50", "ms", lambda v: v <= 50),
            ("model_size", "<=25", "MB", lambda v: v <= 25),
            ("accuracy_drop_robustness", "<=5", "pt", lambda v: v <= 5),
            ("avg_power", "<=500", "mW", lambda v: v <= 500),
            ("thermal_drift_10min", "<=20", "%", lambda v: v <= 20)]
    out = Path(args.out)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "target", "measured", "unit", "device", "runtime", "pass"])
        for metric, target, unit, ok in spec:
            v, device, runtime = brief.get(metric, (None, "", ""))
            w.writerow([metric, target, "" if v is None else round(v, 2), unit, device, runtime,
                        "" if v is None else ("PASS" if ok(v) else "FAIL")])
    print(out.read_text())
    print(f"details: {detailed}")


if __name__ == "__main__":
    main()
