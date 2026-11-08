"""Checkpoint -> ONNX FP32 -> ONNX INT8 (static, QDQ), with an accuracy and
latency check of both.

    python scripts/export_onnx.py --run runs/e5
    python scripts/export_onnx.py --run runs/e4 runs/e12 --out runs/ens_e4_e12   # ensemble

Writes <run>/model_fp32.onnx, model_int8.onnx, model_meta.json, export_report.json.
Copy model_int8.onnx and model_meta.json into models/ for main.py.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echogesture.data import ClipDataset, load_split  # noqa: E402
from echogesture.model import Ensemble, build_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", nargs="+", required=True, help="one run, or several to export as an ensemble")
ap.add_argument("--out", default=None, help="output folder (default: the run folder)")
ap.add_argument("--manifest", default="data/wlasl/manifest.csv")
ap.add_argument("--landmarks", default="data/wlasl/landmarks")
ap.add_argument("--calib", type=int, default=300, help="training clips used for calibration")
args = ap.parse_args()

members = []
for r in args.run:
    ck = torch.load(Path(r) / "best.pt", map_location="cpu")
    m = build_model(ck["config"])
    m.load_state_dict(ck["model"])
    members.append((m.eval(), ck["config"], ck["vocab"]))
cfg, vocab = members[0][1], members[0][2]
for _, c, v in members[1:]:
    assert v == vocab and c["mode"] == cfg["mode"] and c["length"] == cfg["length"], "ensemble members must match"
model = members[0][0] if len(members) == 1 else Ensemble([m for m, _, _ in members]).eval()
run = Path(args.out or args.run[0])
run.mkdir(parents=True, exist_ok=True)
in_dim = members[0][0].mu.numel()

fp32, pre, int8 = run / "model_fp32.onnx", run / "model_pre.onnx", run / "model_int8.onnx"
torch.onnx.export(model, torch.zeros(1, cfg["length"], in_dim), fp32, input_names=["x"], output_names=["logits"],
                  opset_version=17, dynamo=False)
quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)

tr_paths, tr_y, _, _ = load_split(args.manifest, args.landmarks, "train", vocab)
va_paths, va_y, _, _ = load_split(args.manifest, args.landmarks, "val", vocab)
pick = np.random.default_rng(0).permutation(len(tr_paths))[:args.calib]
calib = ClipDataset([tr_paths[i] for i in pick], tr_y[pick], cfg["mode"], cfg["length"])


class Reader(CalibrationDataReader):
    def __init__(self):
        self.it = (({"x": calib[i][0].numpy()[None]}) for i in range(len(calib)))

    def get_next(self):
        return next(self.it, None)


# only the heavy layers go to INT8; softmax / averaging / normalisation stay in
# float, where quantising them costs accuracy for almost no size or speed
quantize_static(str(pre), str(int8), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                op_types_to_quantize=["Conv", "MatMul", "Gemm"])
pre.unlink()

val = ClipDataset(va_paths, va_y, cfg["mode"], cfg["length"])
X = np.stack([val[i][0].numpy() for i in range(len(val))])


def check(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1  # closer to a phone's single big core than a laptop's many
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    logits = np.concatenate([sess.run(None, {"x": x[None]})[0] for x in X])
    top5 = np.argsort(-logits, 1)[:, :5]
    times = []
    for i in range(300):
        x = X[i % len(X)][None]
        t = time.perf_counter()
        sess.run(None, {"x": x})
        times.append((time.perf_counter() - t) * 1000)
    times = np.array(times[20:])
    return {"size_mb": round(path.stat().st_size / 2**20, 2),
            "top1": round(float((top5[:, 0] == va_y).mean()), 4),
            "top5": round(float((top5 == va_y[:, None]).any(1).mean()), 4),
            "latency_p50_ms": round(float(np.percentile(times, 50)), 3),
            "latency_p95_ms": round(float(np.percentile(times, 95)), 3)}


report = {"fp32": check(fp32), "int8": check(int8), "val_clips": len(X),
          "note": "latency = classifier only, 1 CPU thread on the export machine; re-measure on the phone"}
(run / "export_report.json").write_text(json.dumps(report, indent=1))
(run / "model_meta.json").write_text(json.dumps({"mode": cfg["mode"], "length": cfg["length"], "vocab": vocab,
                                                 "runs": [Path(r).name for r in args.run]}))
print(json.dumps(report, indent=1))
