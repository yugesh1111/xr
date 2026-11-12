"""Build the demo videos from real screen recordings (no voice; record a voice-over on top).

    python submission/make_demo_video.py --rec <folder with the recordings> --cut full   # ~6 min, detailed
    python submission/make_demo_video.py --rec <folder> --cut short                      # <= 3 min, for the form

Recordings expected in --rec (all made with `adb shell screenrecord` on the Android
emulator running the release APK, and `asciinema` for the terminal):
    live_help.mp4            live camera, signing HELP
    online_0.mp4 .. online_2 the 9 online ASL clips picked through Test -> Pick a video
    builtin.mp4              Test -> Built-in WLASL test clips
    sentence.mp4             Test -> Sentence test (30 clips)
    terminal.mp4             python main.py + results.csv
    launcher.png             the installed app
1920x1080, colours from the submission template. Needs ffmpeg and Pillow.
"""
import argparse
import csv
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
W, H = 1920, 1080
BG, PANEL, CYAN, MINT, AMBER, TEXT, MUTED = "#0A0E1A", "#0D1B3E", "#00B4D8", "#00E5B0", "#F59E0B", "#E8EDF5", "#8D9AB5"
F = "/System/Library/Fonts/Supplemental/"
REPO = "github.com/yugesh1111/xr"


def font(size, bold=False, black=False, uni=False):
    if uni:
        return ImageFont.truetype(F + "Arial Unicode.ttf", size)
    return ImageFont.truetype(F + ("Arial Black.ttf" if black else "Arial Bold.ttf" if bold else "Arial.ttf"), size)


def wrap(d, text, f, width):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    return lines + ([cur] if cur else [])


def para(d, x, y, text, f, fill, width, gap=10):
    for line in wrap(d, text, f, width):
        d.text((x, y), line, font=f, fill=fill)
        y += f.size + gap
    return y


def chapter_bar(d, chapter):
    d.text((60, 30), chapter, font=font(30, bold=True), fill=CYAN)


def panel(out, chapter, title, lines, accent, x=900, phone=True):
    """Background with a chapter label and an explanation column (video goes on the left)."""
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    chapter_bar(d, chapter)
    if not phone:
        x = 60
    d.text((x, 110), title, font=font(50, bold=True), fill=accent)
    y = 210
    for ln in lines:
        if isinstance(ln, tuple):  # (mark, text, colour)
            mark, t, col = ln
            d.text((x, y), mark, font=font(38, uni=True), fill=col)
            y = para(d, x + 50, y, t, font(34, uni=True), TEXT, W - x - 110, gap=8) + 14
        else:
            d.ellipse([x, y + 14, x + 14, y + 28], fill=accent)
            y = para(d, x + 40, y, ln, font(36), TEXT, W - x - 100, gap=10) + 22
    d.text((x, 1010), f"Palmcast · Track 02 · {REPO}", font=font(24), fill=MUTED)
    im.save(out)


def card_title(out):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 34, H], fill=CYAN)
    d.text((140, 240), "TRACK 02  ·  ECHOGESTURE", font=font(34, bold=True), fill=CYAN)
    d.text((140, 310), "Palmcast", font=font(150, black=True), fill="#FFFFFF")
    para(d, 140, 530, "Sign language to text and speech, live, on the phone.", font(48), TEXT, 1600)
    para(d, 140, 610, "No internet. No cloud. 6.5 MB of INT8 models.", font(48), MINT, 1600)
    d.text((140, 880), "Samsung PRISM XR Hackathon 2026 · " + REPO, font=font(32), fill=MUTED)
    im.save(out)


def card_install(out, launcher):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    chapter_bar(d, "1 · Install")
    if launcher and Path(launcher).exists():
        sh = Image.open(launcher).convert("RGB")
        sh = sh.crop((0, 0, sh.width, int(sh.width * 0.62)))
        sh = sh.resize((760, int(760 * sh.height / sh.width)))
        im.paste(sh, (80, 160))
    x = 920
    d.text((x, 110), "One APK, fully offline", font=font(50, bold=True), fill=MINT)
    y = 210
    for t in [f"Download Palmcast-arm64.apk from {REPO} → Releases.",
              "Any 64-bit Android phone, Android 8 or newer.",
              "Permissions: camera, and the microphone only for Reply. No internet permission at all.",
              "Also runs as Python: pip install, or Docker with --network none."]:
        d.ellipse([x, y + 14, x + 14, y + 28], fill=MINT)
        y = para(d, x + 40, y, t, font(36), TEXT, 900, gap=10) + 22
    im.save(out)


def card_requirements(out):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    chapter_bar(d, "7 · What the brief asked for")
    rows = [("Fully on-device, exocentric: phone camera → text or audio", "Android app, back camera, captions + voice"),
            ("Moving heads, angled signers, low light", "body-relative features, view augmentation, torch / ring light"),
            ("On-device privacy, zero cloud streaming", "no INTERNET permission; nothing stored or sent"),
            ("Pose-invariant 3D hand features", "239 features in the hand's and body's own frame"),
            ("Quantized TFLite / ONNX model, < 20 MB INT8", "ONNX INT8, 6.23 MB (+ 0.23 MB sign gate)"),
            ("≤ 50 ms latency", "14.0 ms per frame p95, word model 5.4 ms"),
            ("Captions or audio for two-way conversation", "captions, offline voice, Reply → spoken answer as captions"),
            ("predict(), predictions.jsonl, results.csv", "main.py; 201 test clips; 5 of 6 metrics PASS"),
            ("Occlusions & HUD clarity", "occlusion test −4.0 pts; confidence bar, hints, conversation")]
    y = 110
    d.text((60, y), "Requirement", font=font(30, bold=True), fill=MUTED)
    d.text((980, y), "How Palmcast meets it", font=font(30, bold=True), fill=MUTED)
    y += 60
    for req, how in rows:
        d.rounded_rectangle([40, y - 8, W - 40, y + 82], radius=12, fill=PANEL)
        d.text((60, y + 18), "✓", font=font(36, uni=True), fill=MINT)
        para(d, 110, y + 6, req, font(28), TEXT, 840, gap=4)
        para(d, 980, y + 6, how, font(28), MINT, 880, gap=4)
        y += 98
    im.save(out)


def card_results(out, chapter="8 · Results"):
    rows = list(csv.DictReader(open(ROOT / "results" / "results.csv")))
    label = {"top1_accuracy": "Top-1 accuracy, unseen signers", "latency_p95": "Latency p95 per frame",
             "model_size": "Model size (INT8)", "accuracy_drop_robustness": "Robustness drop",
             "avg_power": "Average power", "thermal_drift_10min": "Thermal drift, 10 min"}
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    chapter_bar(d, chapter)
    d.text((140, 100), "results.csv against the brief", font=font(56, bold=True), fill=AMBER)
    cols, y = [140, 860, 1190, 1590], 220
    for i, h in enumerate(["Metric", "Target", "Measured", "Status"]):
        d.text((cols[i], y), h, font=font(34, bold=True), fill=MUTED)
    y += 70
    for r in rows:
        d.rounded_rectangle([120, y - 14, 1800, y + 66], radius=12, fill=PANEL)
        meas = f"{r['measured']} {r['unit']}" if r["measured"] else "needs phone run"
        st = r["pass"] or "pending"
        d.text((cols[0], y), label.get(r["metric"], r["metric"]), font=font(38), fill=TEXT)
        d.text((cols[1], y), r["target"].replace(">=", "≥ ").replace("<=", "≤ ") + " " + r["unit"], font=font(38, uni=True), fill=TEXT)
        d.text((cols[2], y), meas, font=font(38, bold=True), fill=TEXT)
        d.text((cols[3], y), st, font=font(38, bold=True), fill=MINT if st == "PASS" else AMBER)
        y += 100
    para(d, 140, y + 20, "Accuracy: 3 groups of signers removed from training. Latency, thermal: 10-minute sustained run on a "
         "laptop CPU (no phone run). Robustness: dark, blur, tilt, low resolution, occlusion.", font(28), MUTED, 1640)
    im.save(out)


def card_end(out):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((W // 2, 400), "Palmcast", font=font(120, black=True), fill="#FFFFFF", anchor="mm")
    d.text((W // 2, 540), REPO, font=font(48), fill=CYAN, anchor="mm")
    d.text((W // 2, 620), "Code · models · results · APK · slides · this video", font=font(36), fill=TEXT, anchor="mm")
    im.save(out)


def run(cmd):
    subprocess.run(cmd, check=True)


def dur(p):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                                capture_output=True, text=True).stdout)


def still(img, secs, out):
    run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(img), "-t", f"{secs:.2f}", "-r", "30",
         "-vf", "format=yuv420p", "-c:v", "libx264", "-crf", "20", str(out)])


def phone_seg(video, bg, out, start=0.0, length=None, speed=1.0, x=160, h=960, y=95):
    """Phone recording scaled to height h at x, over a panel background."""
    length = length if length is not None else dur(video) - start
    run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-t", f"{length:.2f}", "-i", str(video),
         "-loop", "1", "-i", str(bg), "-filter_complex",
         f"[0:v]setpts=PTS/{speed},scale=-2:{h}[v];[1:v][v]overlay={x}:{y}:shortest=1,fps=30,format=yuv420p",
         "-c:v", "libx264", "-crf", "21", "-an", str(out)])


def wide_seg(video, bg, out, speed=1.0):
    """Landscape recording (terminal) centred under the chapter label."""
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-loop", "1", "-i", str(bg), "-filter_complex",
         f"[0:v]setpts=PTS/{speed},scale=1500:-2[v];[1:v][v]overlay=(W-w)/2:150:shortest=1,fps=30,format=yuv420p",
         "-c:v", "libx264", "-crf", "21", "-an", str(out)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--cut", choices=["full", "short"], default="full")
    ap.add_argument("--out")
    a = ap.parse_args()
    rec = Path(a.rec)
    full = a.cut == "full"
    out = a.out or str(ROOT / "submission" / ("Palmcast_demo_full.mp4" if full else "Palmcast_demo.mp4"))
    sp_online, sp_builtin, sp_sentence = (3.0, 2.0, 1.4) if full else (9.0, 4.5, 3.2)

    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        parts = []

        def add(p):
            parts.append(p)

        card_title(t / "title.png"); still(t / "title.png", 15 if full else 8.5, t / "00.mp4"); add(t / "00.mp4")
        if full:
            card_install(t / "install.png", rec / "launcher.png"); still(t / "install.png", 18, t / "01.mp4"); add(t / "01.mp4")

        panel(t / "live.png", "2 · Live camera", "Signing \"help\" to the phone", [
            "Real signing in front of the camera (the laptop webcam is the phone camera here).",
            "Hand outlines follow the fingers; the chips show Signing and Both hands.",
            "Result: help 54%, then a grey help? guess at 30% with alternatives.",
            "Nothing is shown while the hands rest."], MINT)
        phone_seg(rec / "live_help.mp4", t / "live.png", t / "02.mp4", 0, 16, 1.0)
        add(t / "02.mp4")

        res = [("✓", "aslbricks.org · basketball 81%", MINT), ("✓", "asldeafined.com · clothes 86%", MINT),
               ("✓", "aslsearch.com · cow (guess, 33%)", MINT), ("✗", "aslsignbank (Yale) · candy → pizza", AMBER),
               ("✓", "Amazon S3 dictionary · full 42%", MINT), ("✗", "signingsavvy.com · bowling → what", AMBER),
               ("✓", "signstock · but (guess, 20%)", MINT), ("✓", "spreadthesign.com · change 77%", MINT),
               ("✗", "YouTube · accident → what", AMBER)]
        panel(t / "online.png", "3 · Online ASL videos uploaded into the app",
              "9 websites, 6 of 9 right", res, CYAN)
        segs = sorted(rec.glob("online_*.mp4"))
        for i, v in enumerate(segs):
            phone_seg(v, t / "online.png", t / f"03_{i}.mp4", 0, None, sp_online)
            add(t / f"03_{i}.mp4")

        panel(t / "builtin.png", "4 · Built-in test", "10 official test clips: 9 right", [
            "Test → Built-in WLASL test clips: the same pipeline as the camera.",
            "Each answer is compared with the Python version: 10 of 10 the same.",
            "The miss: accident read as wrong, by both."], AMBER)
        phone_seg(rec / "builtin.mp4", t / "builtin.png", t / "04.mp4", 0, None, sp_builtin)
        add(t / "04.mp4")

        panel(t / "sentence.png", "5 · Sentence test", "30 clips, 10 sentences: 20 words right", [
            "Signers the model never trained on, streamed through the live pipeline.",
            "Each word appears, joins the running sentence; a pause ends it.",
            "Clips fixed before testing: the first test clip of each word.",
            "1 wrong word (want → what), 9 not shown, 0 from idle hands."], MINT)
        phone_seg(rec / "sentence.mp4", t / "sentence.png", t / "05.mp4", 9 if (rec / "sentence.mp4").exists() else 0, None, sp_sentence)
        add(t / "05.mp4")

        im = Image.new("RGB", (W, H), BG)
        dd = ImageDraw.Draw(im)
        chapter_bar(dd, "6 · Same models on a computer: predict() and results.csv")
        im.save(t / "term.png")
        wide_seg(rec / "terminal.mp4", t / "term.png", t / "06.mp4", 0.8 if full else 1.0)
        add(t / "06.mp4")

        card_requirements(t / "req.png"); still(t / "req.png", 19 if full else 8, t / "07.mp4"); add(t / "07.mp4")
        card_results(t / "res.png"); still(t / "res.png", 17 if full else 7.5, t / "08.mp4"); add(t / "08.mp4")
        card_end(t / "end.png"); still(t / "end.png", 8 if full else 4.5, t / "09.mp4"); add(t / "09.mp4")

        # chapter timings for the narration (submission/narrate.py)
        names = {"00": "title", "01": "install", "02": "live", "03": "online", "04": "builtin", "05": "sentence",
                 "06": "terminal", "07": "requirements", "08": "results", "09": "end"}
        chapters, t0 = [], 0.0
        for p in parts:
            key = names[p.stem[:2]]
            d_ = dur(p)
            if chapters and chapters[-1]["chapter"] == key:
                chapters[-1]["end"] = round(t0 + d_, 2)
            else:
                chapters.append({"chapter": key, "start": round(t0, 2), "end": round(t0 + d_, 2)})
            t0 += d_
        import json as _json
        Path(out).with_suffix(".chapters.json").write_text(_json.dumps(chapters, indent=1))
        (t / "list.txt").write_text("".join(f"file '{p}'\n" for p in parts))
        run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(t / "list.txt"),
             "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out])
    print(f"{out}: {dur(out):.0f} s, {Path(out).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
