"""Pretty-print a predictions.jsonl (and, if the clip names end in _<WORD>, mark right/wrong).

    python scripts/show_predictions.py results/predictions.jsonl
"""
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
right = 0
for d in rows:
    truth = d["clip_id"].rsplit("_", 1)[-1] if "_" in d["clip_id"] else None
    ok = truth == d["prediction"]
    right += ok
    mark = ("✓" if ok else "✗") if truth else " "
    print(f"{mark} {d['clip_id']:22s} -> {d['prediction']:10s} {d['confidence']:.2f}  {d['latency_ms']:4.1f} ms")
if any("_" in d["clip_id"] for d in rows):
    print(f"\n{right} of {len(rows)} right")
