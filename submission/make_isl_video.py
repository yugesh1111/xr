"""Short extra video: Indian Sign Language mode and teach-a-sign, in the same
style as the main walkthrough (make_walkthrough.py). The main demo stays at
2:55 because the track asks for 2-3 minutes; this one sits next to it.

    python submission/make_isl_video.py --rec isl_scene_full_take.mp4 [--teach teach.mp4] [--voice isl_voice.m4a]

--rec   adb screenrecord of the emulator: Test > Switch to Indian Sign Language,
        then Test > Built-in INCLUDE (ISL) test clips, until the result card.
--teach optional phone recording of Test > Teach the phone a new sign.
--voice optional voice-over, one file for the whole video. The picture is
        stretched or sped up to fit it; without it the video is silent.
--layout screen (default): only the phone screen, cropped, nothing around it (1080x2400).
--layout walkthrough: 1920x1080 like the main video, phone in the middle and a caption.
Writes submission/Palmcast_ISL_and_teach.mp4.
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import make_walkthrough as mw

ROOT = Path(__file__).resolve().parent.parent
LEAD, TAIL = 0.5, 0.9


def label(path, n, title, sub):
    """Same caption as the walkthrough, with ISL in place of the step number."""
    img = mw.Image.new("RGBA", (mw.W, mw.H), (0, 0, 0, 0))
    d = mw.ImageDraw.Draw(img)
    ft, fs, fn = mw.font(34, True), mw.font(24), mw.font(22, True)
    tw = max(d.textlength(title, font=ft), d.textlength(sub, font=fs))
    x0, y1 = 48, mw.H - 40
    y0 = y1 - 104
    d.rounded_rectangle((x0, y0, x0 + tw + 112, y1), 18, fill=(12, 16, 22, 215))
    d.text((x0 + 20, y0 + 22), n, font=fn, fill=(0, 229, 176, 255))
    d.text((x0 + 76, y0 + 14), title, font=ft, fill=(245, 245, 245, 255))
    d.text((x0 + 76, y0 + 60), sub, font=fs, fill=(170, 178, 189, 255))
    img.save(path)


mw.label = label


def screen(src, out, parts, target):
    """Just the screen recording: the parts joined, sped up or held to fit `target` seconds."""
    parts, hold = mw.fit(parts, target)
    ins, chains = [], []
    for i, (a, b, sp) in enumerate(parts):
        ins += ["-ss", f"{a}", "-t", f"{b - a}", "-i", str(src)]
        chains.append(f"[{i}:v]setpts=PTS/{sp},fps=30,scale=1080:2400:force_original_aspect_ratio=decrease,"
                      f"pad=1080:2400:(ow-iw)/2:(oh-ih)/2,setsar=1[p{i}]")
    k = len(parts)
    graph = ";".join(chains) + ";" + "".join(f"[p{i}]" for i in range(k)) + f"concat=n={k}:v=1:a=0," + \
        f"tpad=stop_mode=clone:stop_duration={hold:.2f},trim=duration={target:.2f},setpts=PTS-STARTPTS,format=yuv420p[v]"
    mw.run(["ffmpeg", "-y", "-loglevel", "error", *ins, "-filter_complex", graph, "-map", "[v]", "-r", "30",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium", str(out)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--teach")
    ap.add_argument("--voice")
    ap.add_argument("--layout", choices=["screen", "walkthrough"], default="screen")
    ap.add_argument("--out", default=str(ROOT / "submission" / "Palmcast_ISL_and_teach.mp4"))
    a = ap.parse_args()
    rec, phone = Path(a.rec), (486, 1080)
    steps = [
        (rec, "Indian Sign Language: one switch", "Test menu: switch to 50 Indian Sign Language words (INCLUDE-50)",
         [(1.5, 14, 1.0)], phone, 11.0),
        (rec, "ISL built-in test: same answers as the computer", "10 INCLUDE test clips played on the phone: 10 of 10 right, identical to Python",
         [(14.5, 22, 1.0), (22, 47, 2.5), (47, 56.9, 1.0)], phone, 26.0),
    ]
    if a.teach:
        steps.append((Path(a.teach), "Teach the phone a new sign", "Sign a new word 3 times; the phone then knows it. No retraining, nothing uploaded",
                      [(0, mw.dur(Path(a.teach)), 1.0)], phone, 18.0))
    targets = [s[5] for s in steps]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        if a.voice:
            voice = tmp / "voice.wav"
            mw.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.voice, "-ac", "1", "-ar", "48000", str(voice)])
            need = LEAD + mw.dur(voice) + TAIL
            targets = [t * need / sum(targets) for t in targets]
        files = []
        for i, (src, title, sub, parts, box, _) in enumerate(steps):
            out = tmp / f"s{i}.mp4"
            if a.layout == "walkthrough":
                mw.segment(src, out, tmp, "ISL", title, sub, parts, box, round(targets[i], 2))
            else:
                screen(src, out, parts, round(targets[i], 2))
            files.append(out)
        lst = tmp / "list.txt"
        lst.write_text("".join(f"file '{f}'\n" for f in files))
        silent = tmp / "silent.mp4"
        mw.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                "-c:v", "libx264", "-crf", "20", "-preset", "slow", "-pix_fmt", "yuv420p", str(silent)])
        total = mw.dur(silent)
        if a.voice:
            # same voice treatment as mix_voice.py
            graph = (f"[1:a]adelay={int(LEAD * 1000)}:all=1,highpass=f=80,afftdn=nf=-28,"
                     "acompressor=threshold=-20dB:ratio=2.5:attack=8:release=120,"
                     f"loudnorm=I=-16:TP=-1.5:LRA=9,apad,atrim=0:{total:.3f}[a]")
            mw.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), "-i", str(voice), "-filter_complex", graph,
                    "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                    "-movflags", "+faststart", a.out])
        else:
            mw.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), "-c", "copy", "-movflags", "+faststart", a.out])
    print(f"{a.out}: {mw.dur(Path(a.out)):.1f} s")


if __name__ == "__main__":
    main()
