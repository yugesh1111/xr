"""Put a recorded voice-over on the walkthrough video, one recording per part.

    python submission/add_voiceover.py --voice ~/Downloads/voiceover

The folder holds part1 ... part6 in any audio format (m4a, mp3, wav, ...).
Each part starts 0.4 s into its chapter (Palmcast_walkthrough.chapters.json).
If a recording is longer than its chapter, the chapter's last frame is held
until the voice finishes, so nothing is cut off. The voice gets a high-pass,
gentle noise reduction and loudness normalisation (-16 LUFS).
Writes submission/Palmcast_walkthrough_voice.mp4.
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEAD, TAIL = 0.4, 0.8          # seconds before the voice starts / after it ends


def run(cmd):
    subprocess.run(cmd, check=True)


def dur(p):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                                capture_output=True, text=True).stdout)


def find(folder, n):
    hits = sorted(p for p in Path(folder).iterdir()
                  if p.stem.lower().replace(" ", "").replace("_", "").startswith(f"part{n}") and p.suffix)
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", required=True, help="folder with part1 ... part6")
    ap.add_argument("--video", default=str(ROOT / "submission" / "Palmcast_walkthrough.mp4"))
    ap.add_argument("--out", default=str(ROOT / "submission" / "Palmcast_walkthrough_voice.mp4"))
    a = ap.parse_args()
    chapters = json.loads(Path(a.video).with_suffix(".chapters.json").read_text())
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        parts = []
        for c in chapters:
            n, v0, v1 = c["step"], c["start"], c["end"]
            vlen = v1 - v0
            voice = find(a.voice, n)
            alen = dur(voice) if voice else 0.0
            hold = max(0.0, LEAD + alen + TAIL - vlen)
            out = tmp / f"p{n}.mp4"
            vf = f"tpad=stop_mode=clone:stop_duration={hold:.2f}" if hold > 0 else "null"
            total = vlen + hold
            if voice:
                af = (f"[1:a]aresample=48000,highpass=f=80,afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11,"
                      f"adelay={int(LEAD * 1000)}:all=1,apad,atrim=0:{total:.3f}[a]")
                run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{v0}", "-t", f"{vlen}", "-i", a.video, "-i", str(voice),
                     "-filter_complex", f"[0:v]{vf}[v];{af}", "-map", "[v]", "-map", "[a]",
                     "-c:v", "libx264", "-crf", "20", "-preset", "medium", "-r", "30", "-c:a", "aac", "-b:a", "160k",
                     "-ar", "48000", str(out)])
            else:
                run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{v0}", "-t", f"{vlen}", "-i", a.video,
                     "-f", "lavfi", "-t", f"{total:.3f}", "-i", "anullsrc=r=48000:cl=mono",
                     "-filter_complex", f"[0:v]{vf}[v]", "-map", "[v]", "-map", "1:a",
                     "-c:v", "libx264", "-crf", "20", "-preset", "medium", "-r", "30", "-c:a", "aac", "-b:a", "160k",
                     str(out)])
            parts.append(out)
            print(f"  part {n}: chapter {vlen:5.1f} s, voice {alen:5.1f} s"
                  f"{'' if voice else ' (no recording found)'}{f', held {hold:.1f} s' if hold else ''}")
        lst = tmp / "list.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in parts))
        run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
             "-c", "copy", "-movflags", "+faststart", a.out])
    print(f"{a.out}: {dur(a.out):.1f} s")


if __name__ == "__main__":
    main()
