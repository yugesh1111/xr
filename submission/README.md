# Submission - Team WeCode (VIT Chennai), Palmcast, Track 02 (EchoGesture)

Amritha S (23BEC1368) · Yugeshwaran P (23BEC1404)

Everything the final-submission email asks for, and where it is in this repository.

| Asked for | Where |
|---|---|
| Code | repository root: `main.py` (`predict()`), `echogesture/`, `scripts/`, `android/` |
| Resources | `models/` (INT8 model, sign gate, word list), `results/` |
| Document | [`../README.md`](../README.md), [`../docs/DETAILS.md`](../docs/DETAILS.md) (full technical record), [`../docs/USER_GUIDE.md`](../docs/USER_GUIDE.md) |
| Demo | [YouTube](https://youtu.be/aDAs7t2xOe0) · [`Palmcast_demo_final.mp4`](Palmcast_demo_final.mp4) (2:57, with our voice-over): real screen recordings only, feature by feature, plus the architecture, Indian Sign Language mode and teach-a-sign; voice-over script in [`SPEECH_SCRIPT.md`](SPEECH_SCRIPT.md). Also [`Palmcast_live_pipeline_10_sentences.mp4`](Palmcast_live_pipeline_10_sentences.mp4) (all 10 test sentences through the live pipeline) |
| Diagrams | [`diagrams/`](diagrams): architecture (device, training/evaluation) and per-frame flowchart, draw.io sources + PNG |
| PPT | [`Palmcast_XR_Hackathon_Track02.pptx`](Palmcast_XR_Hackathon_Track02.pptx), built on the organisers' template |
| Package | Android app: `Palmcast-arm64.apk` on the repository's **Releases** page. Evaluation bundle: `python scripts/make_submission.py --team <team name>` writes `dist/<team name>.zip` in the slides' layout (`src/` + `results/`) |
| README | [`../README.md`](../README.md) |

## Track output (from the Track 02 slides)

| Required | Status |
|---|---|
| `predict(video_path) -> dict` in `main.py` | done; returns `clip_id`, `prediction`, `top5`, `confidence`, `latency_ms` |
| `predictions.jsonl`, one line per test clip | `results/predictions.jsonl`, 201 official WLASL100 test clips |
| `results.csv`, the six metrics | `results/results.csv`: 5 of 6 PASS, power needs a phone run |
| Quantized model + conversion script, < 20 MB INT8 | `models/model_int8.onnx` 6.23 MB (+ sign gate 0.23 MB), `scripts/export_onnx.py` |
| Python 3.11, `requirements.txt` | checked on Python 3.11.15; `requirements-inference.txt` (in the bundle as `src/requirements.txt`) |
| Demo video 2-3 min | [YouTube](https://youtu.be/aDAs7t2xOe0), `submission/Palmcast_demo_final.mp4` (2:57) |

## results.csv

| Metric | Target | Measured | |
|---|---|---|---|
| top1_accuracy | >= 70 % | 78.06 % | PASS |
| latency_p95 | <= 50 ms | 13.98 ms per frame | PASS* |
| model_size | <= 25 MB | 6.23 MB | PASS |
| accuracy_drop_robustness | <= 5 pt | 4.64 pt | PASS |
| avg_power | <= 500 mW | - | needs a phone |
| thermal_drift_10min | <= 20 % | 5.28 % | PASS* |

\* 10-minute sustained run on a laptop CPU (Apple M5), `scripts/sustained_run.py`, not a phone.
To fill power from a phone: install the APK, Test -> Measure speed, power and heat, 10 minutes
unplugged, then copy `results_device.csv` into `results/` and rerun `scripts/make_results.py
... --device results/results_device.csv` (command in `docs/USER_GUIDE.md`, section 6).

## Rebuild the slides and the video

```bash
python submission/make_slides.py --template Submission_Final_forGithub.pptx
python live.py --demo --fast --mute --save laptop.mp4
python submission/make_walkthrough.py --rec <recordings folder> --laptop laptop.mp4
# final video: voice-over (mix_voice.py), then the opening title
python submission/add_title.py submission/Palmcast_demo_final.mp4
```
