"""Speech script for the demo videos, and a narrated version made with a
synthetic voice (macOS `say`), timed to each chapter. Replace it with your own
recording any time: the silent videos are kept.

    python submission/narrate.py --video submission/Palmcast_demo_full.mp4 --cut full
    python submission/narrate.py --video submission/Palmcast_demo.mp4 --cut short

Writes <video>_narrated.mp4 and submission/SPEECH_SCRIPT.md.
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCRIPT = {
    "full": {
        "title": "Hi, we are team WeCode from VIT Chennai. This is Palmcast, our answer to Track 2, EchoGesture. "
                 "Palmcast turns American Sign Language into text and speech, live, on a phone, with no internet and no cloud.",
        "install": "Palmcast ships as one Android app. Download the APK from our GitHub releases page and install it on any "
                   "64-bit Android phone. It asks only for the camera, and the microphone for replies. It has no internet "
                   "permission at all, so video of people never leaves the device.",
        "live": "Here I am signing help to the camera. The app tracks my body and both hands, drawn as outlines. When the "
                "sign ends, the word appears: help, at fifty four percent. When it is less sure, it shows a grey guess with "
                "two alternatives instead of pretending.",
        "online": "Next, real A S L videos from nine different websites, including YouTube, Signing Savvy and Spread the Sign. "
                  "Each one is uploaded into the app through Pick a video, and read with exactly the same pipeline as the camera. "
                  "You can see the signer, the hand outlines, and then the answer. Basketball, clothes, cow, full, but and change "
                  "are right. Candy, bowling and accident are wrong, and we show those too. Six of nine, from sources the model "
                  "has never seen.",
        "builtin": "The built-in test runs ten official WLASL test clips on the phone. Nine are right, and every single answer "
                   "matches our Python version, so the app and the evaluation code agree.",
        "sentence": "The sentence test streams thirty clips, from signers the model never trained on, as ten three-word sentences. "
                    "Each word appears as its sign ends and joins a running sentence. A pause adds the full stop. Twenty of the "
                    "thirty words come out right and in order, and resting hands never produce a word.",
        "terminal": "The same models run on a computer through the predict function the brief asks for, on Python 3.11. It gives "
                    "a word, a confidence and a latency for every clip, and results dot c s v shows five of six targets passing.",
        "requirements": "Every requirement of the track is covered: fully on-device and exocentric, robust to moving heads, angled "
                        "signers and low light, zero cloud streaming, pose-invariant hand features, an INT8 ONNX model far under "
                        "twenty megabytes, and captions and audio for a two-way conversation.",
        "results": "Our numbers: seventy eight percent accuracy on people the model never saw, fourteen milliseconds per frame, "
                   "a six point two megabyte model, and a drop of four point six points under dark, blur, tilt, low resolution "
                   "and occlusion. Power will be measured on a phone.",
        "end": "Everything is on GitHub: code, models, results, the app, the slides and this video. Thank you.",
    },
    "short": {
        "title": "We are team WeCode, VIT Chennai. This is Palmcast: sign language to text and speech, on the phone, offline.",
        "live": "I sign help to the camera. Hands are tracked, and the word appears: help, fifty four percent. "
                "Unsure signs show as a grey guess.",
        "online": "Real A S L videos from nine websites, uploaded into the app and read with the same pipeline as the camera. "
                  "Six of nine are right, and the misses are shown too.",
        "builtin": "Ten official test clips on the phone: nine right, all the same as our Python version.",
        "sentence": "Thirty clips from unseen signers, as ten sentences. Words join a running sentence: twenty of thirty right, "
                    "no false words from resting hands.",
        "terminal": "The same models on a computer: predict on Python 3.11, and results dot c s v.",
        "requirements": "Every track requirement is covered, on-device and private.",
        "results": "Seventy eight percent, fourteen milliseconds, six point two megabytes.",
        "end": "All on GitHub. Thank you.",
    },
}


def dur(p):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                                capture_output=True, text=True).stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--cut", choices=["full", "short"], required=True)
    ap.add_argument("--voice", default="Samantha")
    a = ap.parse_args()
    video = Path(a.video)
    chapters = json.loads(video.with_suffix(".chapters.json").read_text())
    lines = SCRIPT[a.cut]
    out = video.with_name(video.stem + "_narrated.mp4")
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        inputs, filters, rows = [], [], []
        for i, ch in enumerate(chapters):
            txt = lines.get(ch["chapter"])
            if not txt:
                continue
            room = ch["end"] - ch["start"] - 0.6
            rate = 175
            while True:  # speak faster only if the chapter is too short for the line
                wav = t / f"{i}.aiff"
                subprocess.run(["say", "-v", a.voice, "-r", str(rate), "-o", str(wav), txt], check=True)
                if dur(wav) <= room or rate >= 230:
                    break
                rate += 10
            inputs += ["-i", str(wav)]
            k = len(inputs) // 2
            filters.append(f"[{k}:a]adelay={int((ch['start'] + 0.4) * 1000)}|{int((ch['start'] + 0.4) * 1000)}[a{k}]")
            rows.append((ch, txt, dur(wav), rate))
        mix = "".join(f"[a{k}]" for k in range(1, len(rows) + 1))
        fc = ";".join(filters) + f";{mix}amix=inputs={len(rows)}:normalize=0,apad[aout]"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), *inputs, "-filter_complex", fc,
                        "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest",
                        "-movflags", "+faststart", str(out)], check=True)
    # the script, with timings, for recording your own voice
    md = ROOT / "submission" / "SPEECH_SCRIPT.md"
    head = "# Demo video speech script\n\nRead each part while its chapter is on screen. Timings are from the videos in this folder.\n"
    body = md.read_text() if md.exists() else head
    section = f"\n## {video.name} ({dur(video):.0f} s)\n\n| time | chapter | say |\n|---|---|---|\n"
    for ch, txt, d_, rate in rows:
        m0, s0 = divmod(int(ch["start"]), 60)
        m1, s1 = divmod(int(ch["end"]), 60)
        section += f"| {m0}:{s0:02d}-{m1}:{s1:02d} | {ch['chapter']} | {txt} |\n"
    marker = f"\n## {video.name} "
    if marker in body:
        start = body.index(marker)
        nxt = body.find("\n## ", start + 4)
        body = body[:start] + section.rstrip("\n") + ("\n" + body[nxt:] if nxt != -1 else "\n")
    else:
        body = body.rstrip("\n") + "\n" + section
    md.write_text(body)
    print(f"{out}: {dur(out):.0f} s; script -> {md}")


if __name__ == "__main__":
    main()
