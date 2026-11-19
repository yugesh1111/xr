"""Opening title on the demo video: product name, event and tagline on the
empty left side of the first scene (the phone stands on the right). The audio
is copied untouched, so the voice-over timing does not change.

    python submission/add_title.py submission/Palmcast_demo_final.mp4

Writes the file in place (the input is read from a temporary copy).
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"
CYAN, MINT, TEXT, MUTED = (0, 180, 216), (0, 229, 176), (245, 247, 250), (170, 178, 189)
SHOW, IN, OUT = (0.3, 11.0), 0.6, 0.6     # seconds: visible from/to, fade in/out


def font(size, idx):
    return ImageFont.truetype(FONT, size, index=idx)   # 0 regular, 1 bold, 4 light


def card(path):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # soft dark panel behind the text, fading out towards the phone
    panel = Image.new("L", (W, H), 0)
    pd = ImageDraw.Draw(panel)
    for x in range(1120):
        a = 205 if x < 820 else int(205 * (1120 - x) / 300)
        pd.line([(x, 0), (x, 820)], fill=a)
        for y in range(820, 910):             # fade out above the step caption at the bottom
            pd.point((x, y), fill=int(a * (910 - y) / 90))
    img.paste(Image.new("RGBA", (W, H), (10, 14, 26, 255)), (0, 0), panel)
    x, y = 120, 250
    eyebrow = "SAMSUNG PRISM XR HACKATHON 2026  ·  TRACK 02"
    f = font(26, 1)
    cx = x
    for ch in eyebrow:                      # letter-spaced, like the template headers
        d.text((cx, y), ch, font=f, fill=CYAN)
        cx += d.textlength(ch, font=f) + 4
    d.rectangle((x, y + 52, x + 90, y + 57), fill=MINT)
    d.text((x - 6, y + 80), "Palmcast", font=font(150, 1), fill=TEXT)
    d.text((x, y + 270), "Signs in. Words out.", font=font(54, 1), fill=MINT)
    d.text((x, y + 336), "Nothing leaves the phone.", font=font(54, 0), fill=TEXT)
    d.text((x, y + 430), "Live sign language to text and speech, fully offline", font=font(30, 0), fill=MUTED)
    d.text((x, y + 474), "Team WeCode  ·  VIT Chennai", font=font(30, 1), fill=MUTED)
    img.save(path)


def main():
    video = Path(sys.argv[1])
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        src, png = tmp / "in.mp4", tmp / "title.png"
        shutil.copy(video, src)
        card(png)
        a, b = SHOW
        graph = (f"[1:v]format=rgba,fade=t=in:st={a}:d={IN}:alpha=1,fade=t=out:st={b - OUT}:d={OUT}:alpha=1[t];"
                 f"[0:v][t]overlay=0:0:enable='between(t,{a},{b})',format=yuv420p[v]")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-loop", "1", "-t", f"{b + 0.1}", "-i", str(png),
                        "-filter_complex", graph, "-map", "[v]", "-map", "0:a", "-c:v", "libx264", "-crf", "20",
                        "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(video)],
                       check=True)
    print("title added:", video)


if __name__ == "__main__":
    main()
