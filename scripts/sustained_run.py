"""10-minute sustained run of the full pipeline on this computer, the same
measurement the app's Bench makes on a phone: per-frame landmark latency and
features + classifier latency, thermal drift = last minute vs first minute.

    python scripts/sustained_run.py --videos ../dclips --minutes 10 --out results/sustained_cpu.json
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echogesture.extract import extract_video  # noqa: E402
import main  # noqa: E402


def main_():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--out", default="results/sustained_cpu.json")
    a = ap.parse_args()
    clips = sorted(Path(a.videos).glob("*.mp4"))
    sess, meta = main._load()
    t_end = time.time() + 60 * a.minutes
    t0 = time.time()
    rows = []  # (seconds since start, landmark ms per frame, classifier ms)
    while time.time() < t_end:
        for c in clips:
            s = time.perf_counter()
            xyz = extract_video(c, enhance=True)
            lm = (time.perf_counter() - s) * 1000 / max(len(xyz), 1)
            s = time.perf_counter()
            x = main.make_input(xyz, meta["mode"], meta["length"])[None].astype(np.float32)
            sess.run(None, {"x": x})
            rows.append((time.time() - t0, lm, (time.perf_counter() - s) * 1000))
            if time.time() >= t_end:
                break
    r = np.array(rows)
    first, last = r[r[:, 0] < 60], r[r[:, 0] >= r[-1, 0] - 60]
    drift = 100 * (np.median(last[:, 1]) / np.median(first[:, 1]) - 1)
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() or platform.processor()
    out = {"device": f"MacBook Air ({cpu} CPU)", "minutes": a.minutes, "predictions": len(r),
           "landmark_ms_per_frame_p95": round(float(np.percentile(r[:, 1], 95)), 2),
           "classifier_ms_p95": round(float(np.percentile(r[:, 2], 95)), 2),
           "landmark_ms_first_minute_median": round(float(np.median(first[:, 1])), 2),
           "landmark_ms_last_minute_median": round(float(np.median(last[:, 1])), 2),
           "thermal_drift_pct": round(float(drift), 2)}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main_()
