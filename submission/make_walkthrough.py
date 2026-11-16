"""Demo video as a straight feature walkthrough: only real screen recordings,
each with a small label. No slides, no charts, no synthetic narration.

    python submission/make_walkthrough.py --rec <recordings folder> --laptop <live.py --save output>

<recordings folder> holds the phone screen recordings (adb screenrecord):
live_help.mp4, online_0.mp4, sentence.mp4, builtin.mp4, and terminal.mp4.
The laptop recording comes from: python live.py --demo --fast --mute --save laptop.mp4
Writes submission/Palmcast_walkthrough.mp4 (1920x1080) and a .chapters.json.
Needs ffmpeg and Pillow.
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
W, H = 1920, 1080
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"


def font(size, bold=False):
    try:
        return ImageFont.truetype(FONT, size, index=1 if bold else 0)
    except OSError:
        return ImageFont.load_default()


def label(path, n, title, sub):
    """Small lower-left caption: step number, what is shown, one line of detail."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    ft, fs, fn = font(34, True), font(24), font(22, True)
    tw = max(d.textlength(title, font=ft), d.textlength(sub, font=fs))
    x0, y1 = 48, H - 40
    y0 = y1 - 104
    d.rounded_rectangle((x0, y0, x0 + tw + 112, y1), 18, fill=(12, 16, 22, 215))
    d.text((x0 + 24, y0 + 22), f"{n:02d}", font=fn, fill=(0, 229, 176, 255))
    d.text((x0 + 76, y0 + 14), title, font=ft, fill=(245, 245, 245, 255))
    d.text((x0 + 76, y0 + 60), sub, font=fs, fill=(170, 178, 189, 255))
    img.save(path)


def mask(path, w, h, r):
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w - 1, h - 1), r, fill=255)
    m.save(path)


def run(cmd):
    subprocess.run(cmd, check=True)


def dur(p):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                                capture_output=True, text=True).stdout)


def segment(src, out, tmp, n, title, sub, parts, box):
    """parts: [(start, end, speed)] of src, joined; box: (w, h) of the picture on screen."""
    lab, msk = tmp / f"label{n}.png", tmp / f"mask{n}.png"
    label(lab, n, title, sub)
    bw, bh = box
    mask(msk, bw, bh, 36 if bw < bh else 22)
    chains, ins = [], []
    for i, part in enumerate(parts):
        if isinstance(part[0], Path):   # (image, seconds): a still, e.g. a result card
            img, secs = part
            ins += ["-loop", "1", "-t", f"{secs}", "-i", str(img)]
            sp = 1.0
        else:
            a, b, sp = part
            ins += ["-ss", f"{a}", "-t", f"{b - a}", "-i", str(src)]
        chains.append(f"[{i}:v]setpts=PTS/{sp},fps=30,scale={bw}:{bh}:force_original_aspect_ratio=decrease,"
                      f"pad={bw}:{bh}:(ow-iw)/2:(oh-ih)/2:color=0x0c1016,setsar=1[p{i}]")
    k = len(parts)
    cat = "".join(f"[p{i}]" for i in range(k)) + f"concat=n={k}:v=1:a=0[v]"
    x = W - bw - 260 if bw < bh else (W - bw) // 2   # a phone stands right of centre, its label on the left
    y = (H - bh) // 2 if bw < bh else 24
    graph = ";".join(chains) + ";" + cat + ";" + (
        f"[v]split[a][b];"
        f"[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=40:2,eq=brightness=-0.28:saturation=0.7[bg];"
        f"[b][{k}:v]alphamerge[fg];"
        f"[bg][fg]overlay={x}:{y}:shortest=1[c];"
        f"[c][{k + 1}:v]overlay=0:0:shortest=1,format=yuv420p,fade=t=in:st=0:d=0.4[out]")
    run(["ffmpeg", "-y", "-loglevel", "error", *ins, "-loop", "1", "-i", str(msk), "-loop", "1", "-i", str(lab),
         "-filter_complex", graph, "-map", "[out]", "-r", "30", "-c:v", "libx264", "-crf", "20", "-preset", "medium",
         str(out)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True, help="folder with the phone recordings")
    ap.add_argument("--laptop", required=True, help="recording made with live.py --demo --save")
    ap.add_argument("--out", default=str(ROOT / "submission" / "Palmcast_walkthrough.mp4"))
    a = ap.parse_args()
    rec = Path(a.rec)
    phone, wide = (486, 1080), (1600, 900)
    steps = [
        (rec / "live_help.mp4", "Live camera, on the phone", "Hands tracked as they sign HELP: shown at 54%, then a grey guess", [(0, 14, 1.0)], phone),
        (Path(a.laptop), "Sentences building as they are signed", "Test clips from signers the model never saw: man help now, woman drink hot, hat blue white", [(0, 17, 1.0), (28, 41, 1.0), (79, 93, 1.0)], wide),
        # result pop-ups sit over the live camera: only the card itself is shown (result_*.png)
        (rec / "online_0.mp4", "A video from the web, uploaded into the app", "aslbricks.org clip, read frame by frame on the phone: basketball 81%", [(16, 49.5, 1.5), (rec / "result_online_0.png", 8)], phone),
        (rec / "sentence.mp4", "Sentence test on the phone", "30 clips played as 10 sentences through the live pipeline: 20 of 30 words right", [(10, 110.5, 3.6), (rec / "result_sentence.png", 9)], phone),
        (rec / "builtin.mp4", "Built-in test: same answers as the computer", "10 official test clips: 9 of 10 right, identical to Python", [(10, 35, 2.5), (112, 125, 1.0), (rec / "result_builtin.png", 9)], phone),
        (rec / "terminal.mp4", "predict() for the evaluators, offline", "python main.py on 9 web clips: 6 right; model 6.2 MB + 0.23 MB sign gate", [(0, dur(rec / "terminal.mp4"), 1.0)], (1274, 900)),
    ]
    chapters, t0 = [], 0.0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        files = []
        for n, (src, title, sub, parts, box) in enumerate(steps, 1):
            out = tmp / f"s{n}.mp4"
            segment(src, out, tmp, n, title, sub, parts, box)
            d = dur(out)
            chapters.append({"step": n, "title": title, "start": round(t0, 2), "end": round(t0 + d, 2)})
            t0 += d
            files.append(out)
        lst = tmp / "list.txt"
        lst.write_text("".join(f"file '{f}'\n" for f in files))
        run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
             "-c:v", "libx264", "-crf", "22", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out])
    Path(a.out).with_suffix(".chapters.json").write_text(json.dumps(chapters, indent=1))
    print(f"{a.out}: {t0:.1f} s")
    for c in chapters:
        print(f"  {c['start']:6.1f}-{c['end']:6.1f}  {c['step']:02d} {c['title']}")


if __name__ == "__main__":
    main()
