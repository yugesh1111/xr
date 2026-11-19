"""Embedding model + background bank for "teach the phone a new sign".

    python scripts/export_embed.py --ckpt runs/f_base/best.pt --out models/ \
        --bank-manifest data/kaggle_asl/manifest.csv --bank-landmarks data/kaggle_asl/landmarks
    python scripts/export_embed.py --ckpt runs/isl_a/best.pt --out models/ --prefix isl_ \
        --bank-manifest data/include50/manifest.csv --bank-landmarks data/include50/landmarks

Writes <prefix>embed.onnx (input x: 1 x 32 x 239, output emb: 1 x 192, the
word model's embedding before its last layer) and <prefix>embed_bg.bin: the
embeddings (unit length, float16) of --bank-n training clips of the model's own
words. The app (core/CustomSigns.kt) takes a sign for a taught word only when it
is closer to that word's recordings than to these ordinary signs.
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_dynamic, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echogesture.data import ClipDataset  # noqa: E402
from echogesture.model import build_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", default="models")
ap.add_argument("--prefix", default="")
ap.add_argument("--bank-manifest", required=True)
ap.add_argument("--bank-landmarks", required=True)
ap.add_argument("--bank-n", type=int, default=1000)
ap.add_argument("--calib", type=int, default=200)
ap.add_argument("--quant", choices=["static", "dynamic", "none"], default="static",
                help="static = INT8 like the word model (cosine to float ~0.999); dynamic = INT8 weights only")
a = ap.parse_args()

ck = torch.load(a.ckpt, map_location="cpu")
cfg = ck["config"]
model = build_model(cfg).eval()
model.load_state_dict(ck["model"])


class Embed(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m.embed(x)


out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
fp32, pre, int8 = out / f"{a.prefix}embed_fp32.onnx", out / f"{a.prefix}embed_pre.onnx", out / f"{a.prefix}embed.onnx"
in_dim = model.mu.numel()
torch.onnx.export(Embed(model), torch.zeros(1, cfg["length"], in_dim), fp32, input_names=["x"], output_names=["emb"],
                  opset_version=17, dynamo=False)
quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)

df = pd.read_csv(a.bank_manifest, dtype={"video_id": str})
lm = Path(a.bank_landmarks)
df = df[df.gloss.isin(ck["vocab"]) & np.array([(lm / f"{v}.npz").exists() for v in df.video_id])]
if "split" in df.columns and (df.split == "train").any():
    df = df[df.split == "train"]
pick = df.sample(min(a.bank_n, len(df)), random_state=2)
ds = ClipDataset([lm / f"{v}.npz" for v in pick.video_id], np.zeros(len(pick), int), cfg["mode"], cfg["length"])
X = np.stack([ds[i][0].numpy() for i in range(len(ds))]).astype(np.float32)


class Reader(CalibrationDataReader):
    def __init__(self):
        self.it = ({"x": X[i:i + 1]} for i in range(min(a.calib, len(X))))

    def get_next(self):
        return next(self.it, None)


if a.quant == "static":
    quantize_static(str(pre), str(int8), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, op_types_to_quantize=["Conv", "MatMul", "Gemm"])
elif a.quant == "dynamic":
    quantize_dynamic(str(pre), str(int8), weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul", "Gemm"])
else:
    pre.replace(int8)
pre.unlink(missing_ok=True)
fp32.unlink()

sess = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
E = np.concatenate([sess.run(None, {"x": X[i:i + 1]})[0] for i in range(len(X))])
E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-6)
model.eval()  # torch.onnx.export can leave the model in training mode (dropout on)
with torch.no_grad():
    T = np.concatenate([model.embed(torch.from_numpy(X[i:i + 1])).numpy() for i in range(len(X))])
T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-6)
agree = (E * T).sum(1)
with open(out / f"{a.prefix}embed_bg.bin", "wb") as f:
    f.write(struct.pack("<ii", *E.shape))
    f.write(E.astype("<f2").tobytes())
print(f"{int8}: {int8.stat().st_size / 2**20:.2f} MB; bank {E.shape[0]} x {E.shape[1]} "
      f"({(out / f'{a.prefix}embed_bg.bin').stat().st_size / 1024:.0f} KB); "
      f"{a.quant} vs float cosine: mean {agree.mean():.4f}, min {agree.min():.4f}")
