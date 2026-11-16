"""Architecture shot for the walkthrough: a slow camera move across the
on-device architecture diagram (diagrams/architecture_device.png), stage by
stage, with eased motion. Writes submission/media/arch_shot.mp4 (1600x900).

    python submission/make_arch_shot.py
"""
import subprocess
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "diagrams" / "architecture_device.png"
OUT = ROOT / "media" / "arch_shot.mp4"
W, H, FPS = 1600, 900, 30

# views as (x0, y0, width) in diagram pixels (3830 x 2170); height follows 16:9
FULL = (0, 0, 3830)
CAPTURE_PERCEPTION = (40, 70, 2440)      # 1 Capture + 2 Perception
SIGN_DETECTION = (1250, 70, 2560)        # 2 Perception (right) + 3 Sign detection
FEATURES = (60, 760, 2000)               # 4 Understanding: body frame, hands, 239 features
MODELS = (1800, 760, 2000)               # 4 Understanding: sampling, sign gate, word model
OUTPUT = (60, 1045, 2000)                # 5 Output: caption rule, sentence, HUD
# (view, seconds to hold, seconds to move to the next)
PLAN = [(FULL, 2.0, 1.3), (CAPTURE_PERCEPTION, 2.8, 1.1), (SIGN_DETECTION, 2.8, 1.1),
        (FEATURES, 2.8, 1.1), (MODELS, 2.8, 1.1), (OUTPUT, 2.4, 1.3), (FULL, 1.6, 0)]


def ease(t):
    return t * t * (3 - 2 * t)


def frames():
    for i, (view, hold, move) in enumerate(PLAN):
        for _ in range(int(hold * FPS)):
            yield view
        if move and i + 1 < len(PLAN):
            nxt = PLAN[i + 1][0]
            n = int(move * FPS)
            for k in range(1, n + 1):
                e = ease(k / n)
                yield tuple(a + (b - a) * e for a, b in zip(view, nxt))


def main():
    img = Image.open(SRC).convert("RGB")
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "18",
                             "-preset", "medium", "-pix_fmt", "yuv420p", str(OUT)], stdin=subprocess.PIPE)
    n = 0
    for x0, y0, w in frames():
        h = w * 9 / 16
        x0 = max(0.0, min(x0, img.width - w))
        y0 = max(0.0, min(y0, img.height - h)) if h <= img.height else (img.height - h) / 2
        if h > img.height:  # full view: letterbox on white
            canvas = Image.new("RGB", (int(w), int(h)), (255, 255, 255))
            canvas.paste(img, (0, int((h - img.height) / 2)))
            crop = canvas
        else:
            crop = img.crop((int(x0), int(y0), int(x0 + w), int(y0 + h)))
        proc.stdin.write(crop.resize((W, H), Image.LANCZOS).tobytes())
        n += 1
    proc.stdin.close()
    proc.wait()
    print(f"{OUT}: {n / FPS:.1f} s")


if __name__ == "__main__":
    main()
