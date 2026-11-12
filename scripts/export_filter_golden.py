"""Golden case for the Android LandmarkFilter (LandmarkFilterTest.kt).

A real test clip with tracker-like jitter, uneven frame spacing and hands that
drop out and come back, filtered by echogesture.live.LandmarkFilter.

    python scripts/export_filter_golden.py
"""
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echogesture import layout as L  # noqa: E402
from echogesture.live import LandmarkFilter  # noqa: E402

rng = np.random.default_rng(3)
src = sorted(Path("data/wlasl/landmarks").glob("*.npz"))[5]
x = np.load(src)["xyz"].astype(np.float32)[:60]
x = x + rng.normal(0, 0.004, x.shape).astype(np.float32)
x[20:24, L.RHAND] = np.nan                       # hand lost for 4 frames
t = np.cumsum(rng.integers(30, 90, len(x))).astype(np.int64)
f = LandmarkFilter()
y = np.stack([f(a, int(b)) for a, b in zip(x, t)])
out = Path("android/app/src/test/resources/filter/golden.bin")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "wb") as fh:
    fh.write(struct.pack("<i", len(x)))
    fh.write(t.astype("<i8").tobytes())
    fh.write(x.astype("<f4").tobytes())
    fh.write(y.astype("<f4").tobytes())
print(f"{out}: {len(x)} frames from {src.name}")
