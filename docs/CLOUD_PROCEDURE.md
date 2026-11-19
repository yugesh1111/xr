# Finishing the submission: procedure

Deadline: **18 Nov 2026, 11:59 PM**. This file is the to-do list for the
machine that has the data (`data/wlasl`, `data/kaggle_asl`, `runs/`). Each
step says what to run, what file it produces, and how to know it worked.
Steps 1-6 run on the cloud machine; step 7 needs a real phone; step 8 is
yours (people, not code).

## Where things stand (8 Nov, from the laptop)

Done and pushed:

| item | state |
|---|---|
| App crash on launch (`Stats.kt`) | fixed; app runs on an Android 16 emulator, built-in test clips 6/6 correct, same answers as Python |
| App crashed if a test clip was missing from the build | fixed (`VideoTester.kt` skips it) |
| `main.py` latency | now also reports MediaPipe time per frame and end-to-end time |
| `scripts/make_results.py` | builds `results/results.csv` from evaluation, robustness and phone files |
| `Dockerfile` + `requirements-inference.txt` | inference image; same predictions as the full install |
| Held-out-signer split | `prepare_wlasl.py --holdout-fraction`, tested (`tests/test_split.py`) |
| README | "How to read these numbers" section; status line no longer overstates |

Deliverables checklist (from the brief):

State as of 19 Nov (this file was the working plan from 8 Nov):

| deliverable | state | step |
|---|---|---|
| `predict(video_path) -> dict` in `main.py` | done | - |
| quantized model + conversion script | done (`models/model_int8.onnx`, `scripts/export_onnx.py`) | - |
| honest headline accuracy (new signers) | done: 78.1% on held-out signers | 1 |
| `predictions.jsonl` | done (`results/predictions.jsonl`, 201 clips) | 3 |
| `results.csv` | done (`results/results.csv`, 5 of 6 PASS) | 5 |
| Docker | done, checked with `--network none` | 4 |
| README with reproducible setup | done | 6 |
| phone latency / power / thermal | laptop CPU run done; phone run (Test → Measure) still to do | 7 |
| demo video (max 3 min), presentation | done: [video](https://youtu.be/fKg2cl0Uf9g) (2:58), deck | 8 |

Before starting: `git pull`, then `python -m pytest tests` (13 tests must pass).
Read the brief (pages 9-11) once more: if it gives an exact format for
`predictions.jsonl` or `results.csv`, follow the brief and adjust
`main.py` / `scripts/make_results.py` rather than this file.

---

## Step 1 - honest accuracy on signers nobody looked at (most important)

**Why:** 84% (official test) is mostly signers the model trained on, and
78.2% (pooled) includes the validation signers that picked checkpoints and
the ensemble. Judges test on new people.

**How:** keep ~15% of clips (whole signers) out of everything, retrain the
three members with exactly the deployed recipe, choose nothing on the
held-out clips, then score once.

```bash
# 1a. new manifest with a holdout split (the original manifest.csv is untouched)
python scripts/prepare_wlasl.py --vocab-size 100 --only-available \
    --holdout-fraction 0.15 --holdout-seed 100 --out manifest_holdout.csv
#   check the printout: holdout ~120-150 clips, several signers, no signer in two splits

M=data/wlasl/manifest_holdout.csv

# 1b. retrain the three members (same commands as the deployed ones + --manifest)
python scripts/train.py --manifest $M --fusion gated --out runs/h100_e4
python scripts/train.py --manifest $M --fusion gated --init runs/pretrain/best.pt --lr 5e-4 \
    --extra-manifest data/kaggle_asl/manifest_split.csv \
    --extra-landmarks data/kaggle_asl/landmarks_fixed --extra-per-class 20 --out runs/h100_e12
python scripts/train.py --manifest $M --fusion gated \
    --alt-landmarks data/wlasl/landmarks_aug1 data/wlasl/landmarks_aug2 --out runs/h100_e13
#   runs/pretrain is reused as is: it is Kaggle ASL Signs only, no WLASL clips

# 1c. export the same ensemble; --manifest matters (calibration must not see holdout clips)
python scripts/export_onnx.py --manifest $M --run runs/h100_e4 runs/h100_e13 runs/h100_e12 \
    --out runs/h100_ens

# 1d. score ONCE on holdout (no changes after seeing this number)
python scripts/evaluate.py --manifest $M --run runs/h100_ens --onnx --split holdout
#   -> runs/h100_ens/eval_holdout_int8.json
```

**Repeat 1a-1d with `--holdout-seed 200` and `300`** (`runs/h200_*`,
`runs/h300_*`, `manifest_holdout_200.csv`, ...). One holdout of ~140 clips
has a 95% interval of about +/-7 points; three different signer groups give
a mean and a spread, which is much more convincing.

**Report:** mean of the three holdout top-1 scores, each with its interval,
as the headline. The shipped model stays the one trained on all data
(`models/model_int8.onnx`); the holdout runs estimate how it does on new
people. Say so in one sentence.

**Check:** in each `manifest_holdout*.csv`, no `signer_id` appears in both
`holdout` and `train`/`val`:
```bash
python -c "import pandas as pd,sys; d=pd.read_csv(sys.argv[1]); h=set(d[d.split=='holdout'].signer_id); print('overlap', h & set(d[d.split.isin(['train','val'])].signer_id))" $M
```
It must print `overlap set()`.

## Step 2 - robustness files for the shipped model

If `results/robustness_landmark.json` and `results/robustness_video.json`
already contain the key `E4+E13+E12`, skip this. Otherwise:

```bash
python scripts/robustness.py --runs runs/e4+runs/e13+runs/e12 --level landmark
python scripts/robustness.py --runs runs/e4+runs/e13+runs/e12 --level video
```

(Use the same `--runs` spelling that produced the README tables; the key in
the JSON must match `--run-key` in step 5.)

## Step 3 - `predictions.jsonl`

One line per clip from the shipped INT8 model, through `main.py` (real
videos -> MediaPipe -> model), on the official test clips:

```bash
python - <<'EOF'
import pandas as pd; from pathlib import Path
d = pd.read_csv("data/wlasl/manifest.csv", dtype={"video_id": str})
paths = [p for v in d[d.split == "test"].video_id for p in Path("data/wlasl/raw").glob(f"{v}.*")]
Path("results/test_videos.txt").write_text("\n".join(map(str, paths)))
print(len(paths), "videos")
EOF
python main.py --videos results/test_videos.txt --out results/predictions.jsonl
```

**Check:** about 100 lines; accuracy from the file matches the 84% of
`eval_test_int8.json` within a point or two (MediaPipe versions differ):

```bash
python - <<'EOF'
import json, pandas as pd
d = pd.read_csv("data/wlasl/manifest.csv", dtype={"video_id": str}).set_index("video_id")
p = [json.loads(l) for l in open("results/predictions.jsonl")]
print(sum(x["prediction"] == d.loc[x["clip_id"], "gloss"] for x in p) / len(p))
EOF
```

When the organisers release their test clips, run the same `main.py`
command on their folder; that file is the one to submit.

## Step 4 - Docker check

Already verified on the laptop (8 Nov): the image builds (1.3 GB), runs
with `--network none`, and gives the same words as the normal install on 6
real clips. Confidences differ by a few points (MediaPipe on Linux vs
macOS). Rerun only if the Dockerfile or requirements change:

```bash
docker build -t echogesture .
mkdir -p out
docker run --rm --network none -v "/path/to/clips:/clips:ro" -v "$PWD/out:/app/results" \
    echogesture --videos /clips --out results/predictions.jsonl
```

`--network none` proves nothing is downloaded at run time.

## Step 5 - `results.csv`

```bash
python scripts/make_results.py \
    --eval runs/h100_ens/eval_holdout_int8.json \
    --eval-test runs/ens_e4_e13_e12/eval_test_int8.json \
    --robustness results/robustness_landmark.json results/robustness_video.json \
    --run-key E4+E13+E12 \
    --device results/results_device.csv \
    --predictions results/predictions.jsonl
```

Rows without a file (phone numbers before step 7) stay empty, never
guessed. Rerun this command after step 7 with the phone file.
If three holdout seeds were run, put the mean in the README and pass the
seed-100 file here (or extend `make_results.py` to take several).

## Step 6 - write it up and push

1. README "Results": a new first table "Held-out signers (headline)" with
   the three seeds, mean, intervals; keep the official-test and pooled
   numbers below it, labelled as optimistic (the "How to read" section
   already explains why).
2. Status line at the top: lead with the held-out number.
3. Work log entry for the day.
4. Commit the deliverable files (`results/` is git-ignored, so force-add):
   ```bash
   git add -f results/results.csv results/predictions.jsonl results/robustness_*.json
   git add README.md runs/h*_ens/eval_*.json 2>/dev/null; git commit; git push
   ```
   (`runs/` is ignored too; force-add only the small eval JSONs if wanted.)

## Step 7 - real phone (Galaxy A36, or Samsung Remote Test Lab)

Cannot run in the cloud (no Android there). On the phone:

1. Rebuild the APK from the latest commit (the old APKs crash on launch).
   Copy the 10 test clips into `data/wlasl/raw/` first so they are bundled.
2. Install `app-arm64-v8a-release.apk`.
3. **Test** -> Built-in WLASL test clips: note "Correct x/10, Same as Python".
4. **Bench**: unplug the phone, start Bench, sign or point at a signer for
   10 minutes, stop. The app saves `bench_<time>/results_device.csv`.
5. Get the file: `adb pull /sdcard/Android/data/com.echogesture.app/files/` then
   copy `results_device.csv` to `results/` and rerun step 5.

Key number: `latency_p95_landmarks_per_frame` (MediaPipe per frame) is the
real-time cost; the brief's 50 ms target is most naturally read as this.
If it is above 50 ms: switch MediaPipe to GPU (it tries GPU first already),
lower the camera resolution, or run landmarks every other frame.

## Step 8 - things only you can do

- **Kaggle ASL Signs permission.** E12 (one of three models) was pre-trained
  on it. Email the organisers: "May we use the Google Isolated Sign
  Language Recognition (Kaggle ASL Signs) landmarks for pre-training?" If
  they say no, ship `E4+E13`:
  ```bash
  python scripts/export_onnx.py --run runs/e4 runs/e13 --out runs/ens_e4_e13
  python scripts/evaluate.py --run runs/ens_e4_e13 --onnx --split test
  ```
  (have this ready before the answer comes).
- **AI-assistance disclosure.** Check whether the hackathon rules ask you to
  declare AI-assisted code; the branch names show it.
- **Demo video (max 3 min)** suggested order: problem (10 s) -> live signing
  on the phone with HUD captions (60 s) -> Test mode 10 clips with "Same as
  Python" (30 s) -> robustness: dark room and tilted camera live (30 s) ->
  numbers slide: held-out accuracy, size, phone latency/power (30 s) ->
  honest limits (blur, 100 words) (20 s).
- **Presentation**: same story; one slide for the canonical features
  (+17 points, the main idea), one for honest evaluation (held-out signers,
  why official test is optimistic), one for on-device numbers.
