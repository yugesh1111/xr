"""Assemble the 2-3 minute demo video from the in-app sentence-test recording.

    python submission/make_demo_video.py --recording ../recordings/palmcast_demo_30_clips.mp4 \
        --out submission/Palmcast_demo.mp4

1920x1080: title card, then the phone recording (30 WLASL test clips through
the live pipeline) beside short explanations that change as the test goes,
then the results card and the closing card. Colours from the submission
template. Needs ffmpeg and Pillow.
"""
import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
W, H = 1920, 1080
BG, PANEL, CYAN, MINT, AMBER, TEXT, MUTED = "#0A0E1A", "#0D1B3E", "#00B4D8", "#00E5B0", "#F59E0B", "#E8EDF5", "#8D9AB5"
F = "/System/Library/Fonts/Supplemental/"


def font(size, bold=False, black=False):
    name = "Arial Black.ttf" if black else ("Arial Bold.ttf" if bold else "Arial.ttf")
    return ImageFont.truetype(F + name, size)


def wrap(d, text, f, width):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def para(d, x, y, text, f, fill, width, gap=10):
    for line in wrap(d, text, f, width):
        d.text((x, y), line, font=f, fill=fill)
        y += f.size + gap
    return y


def card_title(out):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 34, H], fill=CYAN)
    d.text((140, 250), "TRACK 02  ·  ECHOGESTURE", font=font(34, bold=True), fill=CYAN)
    d.text((140, 320), "Palmcast", font=font(150, black=True), fill="#FFFFFF")
    para(d, 140, 540, "Sign language to text and speech, live, on the phone.", font(48), TEXT, 1600)
    para(d, 140, 620, "No internet. No cloud. 6.5 MB of INT8 models.", font(48), MINT, 1600)
    d.text((140, 900), "Samsung PRISM XR Hackathon 2026", font=font(32), fill=MUTED)
    im.save(out)


def card_side(out, title, lines, accent):
    """Right-hand explanation panel (the phone video is composited on the left)."""
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    x, w = 900, 900
    d.text((x, 120), title, font=font(54, bold=True), fill=accent)
    y = 230
    for ln in lines:
        d.ellipse([x, y + 14, x + 14, y + 28], fill=accent)
        y = para(d, x + 40, y, ln, font(38), TEXT, w - 40, gap=12) + 26
    d.text((x, 980), "Palmcast · Track 02 · in-app sentence test", font=font(26), fill=MUTED)
    im.save(out)


def card_results(out):
    rows = list(csv.DictReader(open(ROOT / "results" / "results.csv")))
    label = {"top1_accuracy": "Top-1 accuracy, unseen signers", "latency_p95": "Latency p95 per frame",
             "model_size": "Model size (INT8)", "accuracy_drop_robustness": "Robustness drop",
             "avg_power": "Average power", "thermal_drift_10min": "Thermal drift, 10 min"}
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((140, 90), "Results against the brief", font=font(64, bold=True), fill=AMBER)
    cols = [140, 860, 1190, 1590]
    y = 230
    for i, h in enumerate(["Metric", "Target", "Measured", "Status"]):
        d.text((cols[i], y), h, font=font(34, bold=True), fill=MUTED)
    y += 70
    for r in rows:
        d.rounded_rectangle([120, y - 14, 1800, y + 66], radius=12, fill=PANEL)
        meas = f"{r['measured']} {r['unit']}" if r["measured"] else "needs phone run"
        tgt = r["target"].replace(">=", "≥ ").replace("<=", "≤ ") + " " + r["unit"]
        st = r["pass"] or "pending"
        d.text((cols[0], y), label.get(r["metric"], r["metric"]), font=font(38), fill=TEXT)
        d.text((cols[1], y), tgt, font=font(38), fill=TEXT)
        d.text((cols[2], y), meas, font=font(38, bold=True), fill=TEXT)
        d.text((cols[3], y), st, font=font(38, bold=True), fill=MINT if st == "PASS" else AMBER)
        y += 100
    para(d, 140, y + 20, "Accuracy: 3 groups of signers removed from training. Latency and thermal: 10-minute sustained run "
         "on a laptop CPU (no phone run yet). Robustness: dark, blur, tilt, low resolution, occlusion.", font(28), MUTED, 1640)
    im.save(out)


def card_end(out, repo):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((W // 2, 420), "Palmcast", font=font(120, black=True), fill="#FFFFFF", anchor="mm")
    d.text((W // 2, 560), repo, font=font(48), fill=CYAN, anchor="mm")
    d.text((W // 2, 640), "Code, model, results, APK, slides and this video", font=font(36), fill=TEXT, anchor="mm")
    im.save(out)


def run(cmd):
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", required=True)
    ap.add_argument("--out", default=str(ROOT / "submission" / "Palmcast_demo.mp4"))
    ap.add_argument("--repo", default="github.com/yugesh1111/xr")
    ap.add_argument("--speed", type=float, default=1.15, help="speed-up of the recording")
    a = ap.parse_args()
    rec = Path(a.recording)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(rec)],
                               capture_output=True, text=True).stdout)
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        card_title(t / "title.png")
        # explanation panels, each shown for part of the recording (fractions of its length)
        panels = [
            ("What you are seeing", ["WLASL test videos, signers the model never trained on.",
                                     "Played through the app's live pipeline: the same code the camera uses.",
                                     "Hand and body points drawn on each video."], CYAN, 0.0),
            ("One word at a time", ["When a sign ends, the word appears at the top with its confidence.",
                                    "Mint bar: sure. Grey 'word?': only a guess, not added to the sentence.",
                                    "A tiny sign gate keeps fidgets from becoming words."], MINT, 0.3),
            ("Sentences build up", ["Words join the running sentence at the bottom.",
                                    "A pause ends the sentence with a full stop.",
                                    "The same works live from the camera, and can be spoken aloud."], AMBER, 0.6),
            ("The score", ["30 clips in 10 three-word sentences.",
                           "20 words right and in order, 1 wrong, 9 not shown.",
                           "Clips chosen before testing: the first test clip of each word."], MINT, 0.9),
        ]
        segs = []
        rd = dur / a.speed
        for i, (title, lines, col, start) in enumerate(panels):
            card_side(t / f"side{i}.png", title, lines, col)
            end = panels[i + 1][3] if i + 1 < len(panels) else 1.0
            segs.append((t / f"side{i}.png", start * dur, end * dur))
        card_results(t / "results.png")
        card_end(t / "end.png", a.repo)

        def still(img, secs, out):
            run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(img), "-t", str(secs), "-r", "30",
                 "-vf", "format=yuv420p", "-c:v", "libx264", "-crf", "20", str(out)])

        still(t / "title.png", 7, t / "a.mp4")
        parts = [t / "a.mp4"]
        for i, (img, s0, s1) in enumerate(segs):
            o = t / f"b{i}.mp4"
            # phone recording scaled to the frame height, on the left; panel underneath
            run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{s0:.2f}", "-t", f"{s1 - s0:.2f}", "-i", str(rec),
                 "-loop", "1", "-i", str(img),
                 "-filter_complex",
                 f"[0:v]setpts=PTS/{a.speed},scale=-2:1000[v];[1:v][v]overlay=180:40:shortest=1,fps=30,format=yuv420p",
                 "-c:v", "libx264", "-crf", "20", "-an", str(o)])
            parts.append(o)
        still(t / "results.png", 14, t / "c.mp4")
        still(t / "end.png", 6, t / "d.mp4")
        parts += [t / "c.mp4", t / "d.mp4"]
        (t / "list.txt").write_text("".join(f"file '{p}'\n" for p in parts))
        run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(t / "list.txt"),
             "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out])
    d = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", a.out],
                             capture_output=True, text=True).stdout)
    print(f"{a.out}: {d:.0f} s, {Path(a.out).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
