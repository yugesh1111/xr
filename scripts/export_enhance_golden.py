"""Golden cases for the Android port of echogesture/enhance.py.

Real WLASL frames (resized to 200-400 px wide, covering both the up- and down-scaling path of the 256 px resize to keep the files light), clean and under
each test corruption. Binary little-endian per case: int32 w, int32 h,
uint8[h*w*3] rgb, float64 mean grey, float64 sharpness, int32 sharpened (0/1),
uint8[h*w*3] output of enhance(). Checked by android/.../EnhanceTest.kt.

    python scripts/export_enhance_golden.py
"""
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echogesture import corruptions as C  # noqa: E402
from echogesture import enhance as E  # noqa: E402
from echogesture.extract import read_video  # noqa: E402

out = Path("android/app/src/test/resources/enhance")
out.mkdir(parents=True, exist_ok=True)
for f in out.glob("*.bin"):
    f.unlink()
df = pd.read_csv("data/wlasl/manifest.csv", dtype={"video_id": str})
df = df[df.split == "test"].sample(3, random_state=0)
for width, (_, row) in zip([200, 400, 360], df.iterrows()):
    src = next(Path("data/wlasl/raw").glob(f"{row.video_id}.*"))
    rgb = next(f for i, (f, _) in enumerate(read_video(src, int(row.frame_start), int(row.frame_end))) if i == 10)
    rgb = cv2.resize(rgb, (width, int(width * rgb.shape[0] / rgb.shape[1])), interpolation=cv2.INTER_AREA)
    for name, fn in {"clean": lambda r: r, **{k: v for k, v in C.VIDEO.items() if k != "rotate15"}}.items():
        x = np.ascontiguousarray(fn(rgb))
        g = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        y = E.enhance(x)
        sharpened = int(E.needs_sharpening(x))
        with open(out / f"{row.video_id}_{name}.bin", "wb") as f:
            f.write(struct.pack("<ii", x.shape[1], x.shape[0]) + x.tobytes())
            f.write(struct.pack("<ddi", float(g.mean()), E.sharpness(g), sharpened) + np.ascontiguousarray(y).tobytes())
        print(f"{row.video_id}_{name}: mean {g.mean():.1f} sharpness {E.sharpness(g):.1f} sharpened {sharpened}")
