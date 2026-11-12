"""Palmcast on a laptop: real-time sign recognition from the webcam.

The same pipeline as the Android app, in one window:
camera -> MediaPipe Holistic -> One Euro filter -> sign segmenter -> sign gate
-> INT8 word model -> caption rule -> running sentence -> offline voice.

    python live.py                       # webcam 0
    python live.py --camera 1            # another camera
    python live.py --video clip.mp4 ...  # play video files through the live path, in real time
    python live.py --demo                # 10 test sentences from unseen signers, in the live window
    python live.py --check               # 30-clip sentence test, no camera needed (about 1 minute)

Keys: Q or Esc quit, C clear the sentence, V voice on/off, F filter on/off,
Space end the sentence now.

Nothing leaves the laptop: no network calls after the models are on disk.
"""
import os

os.environ.setdefault("GLOG_minloglevel", "2")       # keep MediaPipe's start-up messages out of the way
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse  # noqa: E402
import difflib
import json
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from echogesture import layout as L
from echogesture.enhance import needs_sharpening, unsharp
from echogesture.extract import LandmarkExtractor, read_video
from echogesture.live import LandmarkFilter, Recognizer, Segmenter, Sentence, caption

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples" / "sentence_clips"

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
ARM_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

# colours (BGR)
MINT = (176, 229, 0)
CYAN = (216, 180, 0)
AMBER = (11, 158, 245)
WHITE = (245, 245, 245)
GREY = (150, 150, 150)
INK = (24, 20, 18)


# ------------------------------------------------------------------ voice

class Voice:
    """Speaks words in the background with the computer's own offline voice."""

    def __init__(self, on=True):
        self.cmd = self._find()
        self.on = on and self.cmd is not None
        self.q = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    @staticmethod
    def _find():
        system = platform.system()
        if system == "Darwin" and shutil.which("say"):
            return lambda t: ["say", "-r", "190", t]
        if system == "Windows":
            return lambda t: ["powershell", "-NoProfile", "-Command",
                              "Add-Type -AssemblyName System.Speech; "
                              f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{t}')"]
        for exe in ("espeak-ng", "espeak", "spd-say"):
            if shutil.which(exe):
                return lambda t, e=exe: [e, t]
        return None

    def say(self, text):
        if self.on:
            self.q.put(text.replace("'", ""))

    def _run(self):
        while True:
            t = self.q.get()
            try:
                subprocess.run(self.cmd(t), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
            except Exception:
                pass


# ------------------------------------------------------------------ pipeline

class Pipeline:
    """Landmarks in, words out. One instance per stream."""

    def __init__(self, recognizer, use_filter=True, on_word=None, on_sentence=None):
        self.rec = recognizer
        self.use_filter = use_filter
        self.filter = LandmarkFilter()
        self.seg = Segmenter()
        self.sentence = Sentence(on_close=on_sentence)
        self.on_word = on_word or (lambda r, kind: None)
        self.last = None          # last recognition result shown
        self.last_kind = None
        self.last_at = -1e9
        self.model_ms = 0.0

    def push(self, frame, t_ms):
        f = self.filter(frame, t_ms) if self.use_filter else frame
        self.sentence.tick(t_ms)
        sign = self.seg.push(f, t_ms)
        if sign is None:
            return f
        r = self.rec(sign)
        self.model_ms = r["ms"]
        kind = caption(r)
        if kind:
            self.last, self.last_kind, self.last_at = r, kind, t_ms
            if kind == "sure":
                self.sentence.add(r["words"][0], t_ms)
        self.on_word(r, kind)
        return f


def rest_frame(f):
    """The body of frame f with both hands gone (hands dropped between signs)."""
    r = f.copy()
    r[L.LHAND] = np.nan
    r[L.RHAND] = np.nan
    return r


# ------------------------------------------------------------------ drawing

class Hud:
    def __init__(self):
        self.font = self.small = None
        try:
            from PIL import ImageFont
            for path in ("/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/Helvetica.ttc",
                         "C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
                if Path(path).exists():
                    self.font = {s: ImageFont.truetype(path, s) for s in (16, 20, 26, 34, 56)}
                    break
        except ImportError:
            pass

    @staticmethod
    def _pt(p, w, h, mirror):
        # landmark units: x and z are image heights, y is image height
        x, y = p[0] * h, p[1] * h
        return int(w - x if mirror else x), int(y)

    def skeleton(self, img, f, mirror):
        h, w = img.shape[:2]
        pose = f[L.POSE]
        for a, b in ARM_EDGES:
            if not (np.isnan(pose[a, 0]) or np.isnan(pose[b, 0])):
                cv2.line(img, self._pt(pose[a], w, h, mirror), self._pt(pose[b], w, h, mirror), (90, 90, 90), 2, cv2.LINE_AA)
        for block, col in ((L.RHAND, MINT), (L.LHAND, CYAN)):
            hnd = f[block]
            if np.isnan(hnd[0, 0]):
                continue
            pts = [self._pt(p, w, h, mirror) for p in hnd]
            for a, b in HAND_EDGES:
                cv2.line(img, pts[a], pts[b], col, 2, cv2.LINE_AA)
            for p in pts:
                cv2.circle(img, p, 3, WHITE, -1, cv2.LINE_AA)

    def panel(self, img, x0, y0, x1, y1, alpha=0.62):
        x0, y0 = max(x0, 0), max(y0, 0)
        roi = img[y0:y1, x0:x1]
        roi[:] = (roi * (1 - alpha) + np.array(INK) * alpha).astype(np.uint8)

    def text(self, img, items):
        """items: (x, y, text, size, bgr). Pillow when available, else OpenCV."""
        if self.font:
            from PIL import Image, ImageDraw
            pil = Image.fromarray(img[..., ::-1])
            d = ImageDraw.Draw(pil)
            for x, y, t, s, c in items:
                d.text((x, y), t, font=self.font[s], fill=(c[2], c[1], c[0]))
            img[:] = np.asarray(pil)[..., ::-1]
        else:
            for x, y, t, s, c in items:
                cv2.putText(img, t, (x, y + s), cv2.FONT_HERSHEY_SIMPLEX, s / 32, c, 2 if s > 24 else 1, cv2.LINE_AA)

    def width(self, t, s):
        if self.font:
            return int(self.font[s].getlength(t))
        return cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, s / 32, 2)[0][0]

    def draw(self, img, pipe, frame, now_ms, info):
        h, w = img.shape[:2]
        if frame is not None:
            self.skeleton(img, frame, info["mirror"])
        hands = frame is not None and not (np.isnan(frame[L.RHAND.start, 0]) and np.isnan(frame[L.LHAND.start, 0]))
        status, col = ("Signing", AMBER) if pipe.seg.signing else (("Hands seen", MINT) if hands else ("Listening", GREY))
        items = []
        # top bar
        self.panel(img, 0, 0, w, 44, 0.55)
        cv2.circle(img, (22, 22), 7, col, -1, cv2.LINE_AA)
        items.append((38, 10, status, 20, WHITE))
        right = (f"{info['fps']:.0f} fps   landmarks {info['lm_ms']:.0f} ms   model {pipe.model_ms:.1f} ms   "
                 f"filter {'on' if pipe.use_filter else 'off'}   voice {'on' if info['voice'] else 'off'}")
        items.append((w - self.width(right, 16) - 16, 13, right, 16, GREY))
        items.append((w // 2 - self.width("Palmcast", 20) // 2, 10, "Palmcast", 20, MINT))
        if info.get("label"):
            self.panel(img, 16, 78, 32 + self.width(info["label"], 20), 112, 0.7)
            items.append((24, 83, info["label"], 20, AMBER))
        # sentence card
        lines = [(s, GREY) for s in pipe.sentence.closed] + ([(pipe.sentence.forming() + " ...", WHITE)]
                                                              if pipe.sentence.forming() else [])
        y = h - 190
        if lines:
            self.panel(img, 16, y - 10 - 34 * (len(lines) - 1), w - 16, y + 36)
            for i, (s, c) in enumerate(lines):
                items.append((32, y - 34 * (len(lines) - 1 - i), s, 26, c))
        # caption
        if pipe.last is not None and now_ms - pipe.last_at < 3000:
            r, sure = pipe.last, pipe.last_kind == "sure"
            word = r["words"][0].lower() + ("" if sure else "?")
            self.panel(img, 16, h - 140, w - 16, h - 16)
            items.append((36, h - 132, word, 56, WHITE if sure else GREY))
            p = r["probs"][0]
            bx = 40 + self.width(word, 56) + 30
            cv2.rectangle(img, (bx, h - 92), (bx + 220, h - 80), (70, 70, 70), -1)
            cv2.rectangle(img, (bx, h - 92), (bx + int(220 * p), h - 80), MINT if sure else GREY, -1)
            items.append((bx + 232, h - 100, f"{100 * p:.0f}%", 20, WHITE))
            alts = "   ".join(f"{a.lower()} {100 * q:.0f}%" for a, q in zip(r["words"][1:], r["probs"][1:]))
            items.append((bx, h - 66, "also: " + alts, 16, GREY))
        elif not hands:
            self.panel(img, 16, h - 76, w - 16, h - 16)
            items.append((36, h - 62, "Show your hands and sign. Pause briefly after each sign.", 20, GREY))
        keys = "Q quit   C clear   V voice   F filter   Space end sentence"
        items.append((w - 16 - self.width(keys, 16), 54, keys, 16, GREY))
        self.text(img, items)


# ------------------------------------------------------------------ sources

class Camera:
    """Reads the webcam on its own thread and keeps only the newest frame,
    so a slow frame never builds a queue: what you see is what is analysed."""

    def __init__(self, index, width, height):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            sys.exit(f"Cannot open camera {index}. On a Mac, allow camera access for your terminal app in "
                     "System Settings > Privacy & Security > Camera, then run again.")
        self.frame, self.t, self.n = None, 0.0, 0
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.running:
            ok, bgr = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame, self.t, self.n = bgr, time.monotonic() * 1000, self.n + 1

    def read(self, last_n):
        while self.running:
            with self.lock:
                if self.n != last_n and self.frame is not None:
                    return self.frame, self.t, self.n
            time.sleep(0.002)
        return None, 0, last_n

    def close(self):
        self.running = False
        self.cap.release()


def video_frames(items, realtime=True):
    """(bgr, t_ms, is_gap, label) for clips in a row. items: (path, gap_ms, label).
    After each clip the hands are 'lowered' for gap_ms (landmarks without hands),
    timed like a camera: sleeps to keep real time unless realtime=False."""
    t_off = 0.0
    start = time.monotonic()
    for p, gap_ms, label in items:
        last_t = 0.0
        for rgb, ts in read_video(p):
            t = t_off + ts
            if realtime:
                time.sleep(max(0.0, (start + t / 1000) - time.monotonic()))
            last_t = ts
            yield rgb[..., ::-1].copy(), t, False, label
        t_off += last_t + 40
        for k in range(int(gap_ms // 40)):
            t = t_off + 40 * k
            if realtime:
                time.sleep(max(0.0, (start + t / 1000) - time.monotonic()))
            yield None, t, True, label
        t_off += gap_ms


def demo_items():
    """The 10 test sentences: 1 s hands-down between words, 4.5 s after a sentence."""
    out = []
    for g in json.loads((ROOT / "models" / "sentence_clips.json").read_text()):
        said = " ".join(c["gloss"].lower() for c in g)
        for i, c in enumerate(g):
            out.append((SAMPLES / f"{c['video_id']}.mp4", 4500 if i == len(g) - 1 else 1000, f"Signed: {said}"))
    return out


# ------------------------------------------------------------------ modes

def run_window(args):
    rec = Recognizer(ROOT / "models")
    voice = Voice(on=not args.mute)
    log = []

    def on_word(r, kind):
        tag = {"sure": "", "guess": "?"}.get(kind, " (not shown)")
        print(f"  {r['words'][0].lower()}{tag}  {100 * r['probs'][0]:.0f}%   "
              f"(gate {r['sign_prob']:.2f}, model {r['ms']:.1f} ms)", flush=True)
        if kind == "sure":
            voice.say(r["words"][0].lower())
        log.append({"t": time.time(), "word": r["words"][0], "p": r["probs"][0], "kind": kind})

    def on_sentence(s):
        print(f"Sentence: {s}", flush=True)

    pipe = Pipeline(rec, use_filter=not args.no_filter, on_word=on_word, on_sentence=on_sentence)
    hud = Hud()
    win = "Palmcast"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    writer = None
    fps, lm_ms, last_tick = 0.0, 0.0, time.monotonic()
    sharpen, frame_no = False, 0
    files = demo_items() if args.demo else [(p, 1000, Path(p).name) for p in args.video] if args.video else None
    mirror = files is None   # the webcam is shown as a mirror; a video file as it was filmed
    cam = None if files else Camera(args.camera, args.width, args.height)
    src = video_frames(files, realtime=not args.fast) if files else None
    label = ""
    print("Palmcast is running. Sign in front of the camera; words print here and show in the window.")
    with LandmarkExtractor() as ex:
        n, last_frame, shown = 0, None, None
        while True:
            if cam:
                bgr, t_ms, n = cam.read(n)
                gap = False
            else:
                try:
                    bgr, t_ms, gap, label = next(src)
                except StopIteration:
                    break
            if gap:
                f = rest_frame(last_frame) if last_frame is not None else None
                if f is not None:
                    pipe.push(f, t_ms)
                img = shown if shown is not None else np.zeros((args.height, args.width, 3), np.uint8)
                img = img.copy()
                img[:] = (img * 0.35).astype(np.uint8)
                f_draw = None
            else:
                t0 = time.perf_counter()
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if frame_no % 5 == 0:
                    sharpen = needs_sharpening(rgb)
                frame_no += 1
                raw = ex.frame(unsharp(rgb) if sharpen else rgb, t_ms)
                lm_ms = 0.9 * lm_ms + 0.1 * (time.perf_counter() - t0) * 1000 if lm_ms else (time.perf_counter() - t0) * 1000
                last_frame = raw
                f_draw = pipe.push(raw, t_ms)
                img = cv2.flip(bgr, 1) if mirror else bgr.copy()
                if img.shape[0] != 720:
                    img = cv2.resize(img, (int(img.shape[1] * 720 / img.shape[0]), 720))
                    if f_draw is not None:
                        pass  # landmarks are in image-height units, so they scale with the picture
                shown = img.copy()   # the picture without the HUD, reused (dimmed) while hands are down
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - last_tick, 1e-3) if fps else 1 / max(now - last_tick, 1e-3)
            last_tick = now
            hud.draw(img, pipe, f_draw, t_ms, {"fps": fps, "lm_ms": lm_ms, "voice": voice.on, "mirror": mirror, "label": label})
            if args.save:
                if writer is None:
                    wsize = (img.shape[1], img.shape[0])
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 25 if files else 15, wsize)
                writer.write(img if (img.shape[1], img.shape[0]) == wsize else cv2.resize(img, wsize))
            cv2.imshow(win, img)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("c"):
                pipe.sentence.clear()
                pipe.last = None
            if k == ord("v"):
                voice.on = not voice.on and voice.cmd is not None
            if k == ord("f"):
                pipe.use_filter = not pipe.use_filter
                pipe.filter.reset()
            if k == ord(" "):
                pipe.sentence._close() if pipe.sentence.open else None
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    if pipe.sentence.open:
        pipe.sentence._close()
    if cam:
        cam.close()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Done. {sum(1 for e in log if e['kind'] == 'sure')} words shown.")


def run_check(args):
    """The app's sentence test on this laptop: 30 test clips from signers the
    model never trained on, played as 10 three-word sentences through the full
    live path with real frame timing. Needs no camera."""
    groups = json.loads((ROOT / "models" / "sentence_clips.json").read_text())
    rec = Recognizer(ROOT / "models")
    hits = total = exact = 0
    lm_times = []
    print(f"Sentence test: {len(groups)} sentences, {sum(map(len, groups))} clips, filter "
          f"{'off' if args.no_filter else 'on'}\n")
    for g in groups:
        truth = [c["gloss"].lower() for c in g]
        pipe = Pipeline(rec, use_filter=not args.no_filter)
        t_off, last = 0.0, None
        for c in g:
            path = SAMPLES / f"{c['video_id']}.mp4"
            if not path.exists():
                sys.exit(f"missing {path}")
            with LandmarkExtractor() as ex:  # fresh tracker per clip, like the app
                ts = 0.0
                for rgb, ts in read_video(path):
                    t0 = time.perf_counter()
                    last = ex.frame(rgb, ts)
                    lm_times.append((time.perf_counter() - t0) * 1000)
                    pipe.push(last, t_off + ts)
            t_off += ts + 40
            for k in range(25):  # 1 s with the hands down
                pipe.push(rest_frame(last), t_off + 40 * k)
            t_off += 1000
        for k in range(110):  # 4.4 s pause closes the sentence
            pipe.push(rest_frame(last), t_off + 40 * k)
        said = pipe.sentence.closed[-1].rstrip(".").lower().split() if pipe.sentence.closed else []
        m = sum(b.size for b in difflib.SequenceMatcher(a=truth, b=said).get_matching_blocks())
        hits += m
        total += len(truth)
        exact += said == truth
        print(f"  signed: {' '.join(truth):32s} read: {' '.join(said) or '-':32s} {m}/{len(truth)}")
    lm = np.array(lm_times)
    print(f"\n{hits} of {total} words right and in order ({100 * hits / total:.0f}%), "
          f"{exact} of {len(groups)} sentences exact")
    print(f"landmarks: median {np.median(lm):.1f} ms per frame, p95 {np.percentile(lm, 95):.1f} ms "
          f"-> up to {1000 / np.median(lm):.0f} fps on this computer")
    print("The Android app scores 20 of 30 on the same clips (Test > Sentence test).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="webcam index (0 = built-in)")
    ap.add_argument("--video", nargs="+", help="play these video files through the live path instead of the camera")
    ap.add_argument("--demo", action="store_true", help="play the 10 test sentences (30 clips) through the live window")
    ap.add_argument("--fast", action="store_true", help="with --video/--demo: do not wait for real time (for --save)")
    ap.add_argument("--check", action="store_true", help="run the 30-clip sentence test and print the score")
    ap.add_argument("--no-filter", action="store_true", help="turn the One Euro landmark filter off")
    ap.add_argument("--mute", action="store_true", help="start with the voice off")
    ap.add_argument("--save", help="also record the window to this .mp4")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()
    if args.check:
        run_check(args)
    else:
        run_window(args)


if __name__ == "__main__":
    main()
