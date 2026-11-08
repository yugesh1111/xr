"""Export golden test cases for the Android port of the feature pipeline.

Each case is raw landmarks -> the exact model input Python produces. The
Kotlin unit test (android/app/src/test/.../FeaturesTest.kt) must reproduce
them. Binary little-endian: int32 T, float32[T*88*3] xyz,
int32 L, int32 D, float32[L*D] input.

    python scripts/export_golden.py
"""
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echogesture import layout as L  # noqa: E402
from echogesture.features import clip_features  # noqa: E402
from echogesture.inputs import make_input  # noqa: E402

out = Path("android/app/src/test/resources/golden")
out.mkdir(parents=True, exist_ok=True)
for f in out.glob("*.bin"):
    f.unlink()


def write(name, xyz):
    xyz = np.asarray(xyz, dtype=np.float32)
    x = make_input(xyz, "canonical", 32).astype(np.float32)
    with open(out / f"{name}.bin", "wb") as fh:
        fh.write(struct.pack("<i", len(xyz)))
        fh.write(xyz.astype("<f4").tobytes())
        fh.write(struct.pack("<ii", *x.shape))
        fh.write(x.astype("<f4").tobytes())


df = pd.read_csv("data/wlasl/manifest.csv", dtype={"video_id": str})
clips = [np.load(f"data/wlasl/landmarks/{v}.npz")["xyz"].astype(np.float32) for v in df.video_id[:40]]
for i, x in enumerate(clips[:6]):
    write(f"wlasl_{i}", x)
# a left-handed clip (canonicalizer mirrors it)
lefty = next((x for x in clips if clip_features(x)["quality"][0, 5] > 0), None)
if lefty is not None:
    write("wlasl_lefthanded", lefty)
# hidden hand for a stretch + dropped frames
x = clips[0].copy()
x[10:30, L.RHAND] = np.nan
x[::7] = np.nan
write("occluded", x)
# camera flipped horizontally with swapped labels (left-handed signer seen by a mirror)
m = clips[1][:, L.mirror_permutation()].copy()
m[..., 0] = 1.0 - m[..., 0]
write("mirrored", m)
# no body detected at all
x = clips[2].copy()
x[:, L.POSE] = np.nan
write("no_pose", x)
# a Kaggle ASL Signs clip
k = sorted(Path("data/kaggle_asl/landmarks_fixed").glob("*.npz"))[:1]
if k:
    write("kaggle_0", np.load(k[0])["xyz"].astype(np.float32))
print(sorted(p.name for p in out.glob("*.bin")))
