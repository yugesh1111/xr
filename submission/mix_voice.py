"""Final demo video: the walkthrough fitted to a recorded voice-over.

    python submission/mix_voice.py --voice voiceover_full.m4a --edit voice_edit.json \
        --rec <phone recordings> --laptop submission/Palmcast_live_pipeline_10_sentences.mp4

voice_edit.json lists, for each of the six script parts, the spans of the
recording to keep, in seconds: {"parts": [[[1.2, 13.7]], [[14.5, 49.7]], ...]}.
Retakes are handled by leaving the first attempt out of the spans.

Each video segment gets a minimum length (enough to see the footage); a
segment group grows when its voice part is longer, so the voice never runs
over the picture. Voice: high-pass, noise reduction, light compression,
-16 LUFS. Writes submission/Palmcast_demo_final.mp4.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEAD, TAIL, XF = 0.5, 0.9, 0.03
# video segments (make_walkthrough.py steps) shown under each voice part, with minimum lengths
GROUPS = [[0], [1, 2], [3], [4], [5], [6]]
MIN_LEN = [13.5, 15.0, 31.0, 27.0, 26.0, 26.0, 18.0]
# with --isl and --teach: 9 video steps, 8 voice parts (ISL and teach-a-sign before predict())
GROUPS_ALL = [[0], [1, 2], [3], [4], [5], [6], [7], [8]]
MIN_LEN_ALL = [11.0, 12.0, 21.0, 17.0, 15.0, 14.0, 22.0, 10.0, 13.0]


def run(cmd):
    subprocess.run(cmd, check=True)


def dur(p):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                                capture_output=True, text=True).stdout)


def cut_part(src, spans, out):
    """Keep `spans` of the recording, joined with 30 ms crossfades."""
    ins, chains = [], []
    for i, (a, b) in enumerate(spans):
        ins += ["-ss", f"{a}", "-t", f"{b - a}", "-i", str(src)]
        chains.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=mono[s{i}]")
    if len(spans) == 1:
        graph = chains[0].replace("[s0]", "[o]")
    else:
        graph, prev = ";".join(chains), "[s0]"
        for i in range(1, len(spans)):
            nxt = f"[x{i}]" if i < len(spans) - 1 else "[o]"
            graph += f";{prev}[s{i}]acrossfade=d={XF}:c1=tri:c2=tri{nxt}"
            prev = nxt
    run(["ffmpeg", "-y", "-loglevel", "error", *ins, "-filter_complex", graph, "-map", "[o]", str(out)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", help="one recording of the whole script (with --edit)")
    ap.add_argument("--edit", help="voice_edit.json: spans of --voice for each part")
    ap.add_argument("--parts", help="folder with part1 ... partN, one recording per part, any audio format")
    ap.add_argument("--insert", nargs=2, metavar=("ISL_VOICE", "TEACH_VOICE"),
                    help="with --voice/--edit (6 parts): new recordings for the ISL and teach-a-sign parts, put before the last part")
    ap.add_argument("--isl")
    ap.add_argument("--teach")
    ap.add_argument("--rec", required=True)
    ap.add_argument("--laptop", required=True)
    ap.add_argument("--out", default=str(ROOT / "submission" / "Palmcast_demo_final.mp4"))
    a = ap.parse_args()
    full = bool(a.isl and a.teach)
    groups, min_len = (GROUPS_ALL, MIN_LEN_ALL) if full else (GROUPS, MIN_LEN)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        lens = []
        if a.parts:
            files = sorted((f for f in Path(a.parts).iterdir() if f.stem.lower().startswith("part")),
                           key=lambda f: int("".join(c for c in f.stem if c.isdigit())))
            parts = files
            for i, f in enumerate(files):
                run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(f), "-ac", "1", "-ar", "48000", str(tmp / f"v{i}.wav")])
                lens.append(dur(tmp / f"v{i}.wav"))
        else:
            parts = json.loads(Path(a.edit).read_text())["parts"]
            for i, spans in enumerate(parts):
                cut_part(a.voice, spans, tmp / f"v{i}.wav")
                lens.append(dur(tmp / f"v{i}.wav"))
        if a.insert:
            # old parts 1-5, then ISL and teach-a-sign, then the old last part
            for i, f in enumerate(a.insert):
                run(["ffmpeg", "-y", "-loglevel", "error", "-i", f, "-ac", "1", "-ar", "48000", str(tmp / f"new{i}.wav")])
            last = len(parts) - 1
            (tmp / f"v{last}.wav").rename(tmp / "last.wav")
            (tmp / "new0.wav").rename(tmp / f"v{last}.wav")
            (tmp / "new1.wav").rename(tmp / f"v{last + 1}.wav")
            (tmp / "last.wav").rename(tmp / f"v{last + 2}.wav")
            lens = lens[:last] + [dur(tmp / f"v{last}.wav"), dur(tmp / f"v{last + 1}.wav"), lens[last]]
            parts = list(parts[:last]) + ["isl", "teach", parts[last]]
        if len(parts) != len(groups):
            sys.exit(f"{len(parts)} voice parts, the video has {len(groups)}")
        GROUPS_ = groups
        targets = list(min_len)
        for g, idx in enumerate(GROUPS_):
            need = LEAD + lens[g] + TAIL
            have = sum(targets[j] for j in idx)
            if need > have:
                targets[idx[-1]] += need - have
        total = sum(targets)
        print("voice parts:", ", ".join(f"{x:.1f}" for x in lens), f"| video {total:.1f} s")
        if total > 180:
            sys.exit(f"too long: {total:.1f} s > 180 s; trim the voice or the minimum lengths")
        silent = tmp / "silent.mp4"
        run([sys.executable, str(ROOT / "submission" / "make_walkthrough.py"), "--rec", a.rec, "--laptop", a.laptop,
             "--out", str(silent), "--targets", json.dumps([round(t, 2) for t in targets])]
            + (["--isl", a.isl, "--teach", a.teach] if full else []))
        # place each voice part at the start of its group
        starts, t = [], 0.0
        for idx in GROUPS_:
            starts.append(t)
            t += sum(targets[j] for j in idx)
        ins, chains = [], []
        for i in range(len(parts)):
            ins += ["-i", str(tmp / f"v{i}.wav")]
            chains.append(f"[{i + 1}:a]adelay={int((starts[i] + LEAD) * 1000)}:all=1[d{i}]")
        mix = "".join(f"[d{i}]" for i in range(len(parts)))
        graph = (";".join(chains) + f";{mix}amix=inputs={len(parts)}:normalize=0:duration=longest,"
                 "highpass=f=80,afftdn=nf=-28,acompressor=threshold=-20dB:ratio=2.5:attack=8:release=120,"
                 f"loudnorm=I=-16:TP=-1.5:LRA=9,apad,atrim=0:{total:.3f}[a]")
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), *ins, "-filter_complex", graph,
             "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
             "-movflags", "+faststart", a.out])
        Path(a.out).with_suffix(".chapters.json").write_text(
            json.dumps([{"part": i + 1, "voice_start": round(starts[i] + LEAD, 2), "voice_len": round(lens[i], 2)}
                        for i in range(len(parts))], indent=1))
    print(f"{a.out}: {dur(a.out):.1f} s")


if __name__ == "__main__":
    main()
