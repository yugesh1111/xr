"""Detailed architecture diagrams (draw.io): device runtime and training/evaluation.
Run after gen.py's style constants: python gen_arch.py"""
from gen import D, BOX, MODEL, GROUP, OUT, NOTE, EDGE, DEDGE

SMALL = BOX.replace("fontSize=13", "fontSize=12")
MSMALL = MODEL.replace("fontSize=13", "fontSize=12")
OSMALL = OUT.replace("fontSize=13", "fontSize=12")
TAG = "text;html=1;align=center;verticalAlign=middle;fontSize=11;fontColor=#0F766E;fontStyle=1;fontFamily=Helvetica;"
DATA = "text;html=1;align=center;verticalAlign=middle;fontSize=11;fontColor=#555555;fontFamily=Courier New;"
PRIV = "rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#0F766E;strokeWidth=2;dashed=1;verticalAlign=top;align=right;spacingRight=12;spacingTop=6;fontSize=13;fontStyle=1;fontColor=#0F766E;fontFamily=Helvetica;arcSize=2;"

# ================= on-device runtime =================
d = D()
W = 1900
d.v("Android phone - privacy boundary: no INTERNET permission, video never leaves the device", 10, 10, W - 20, 1050, PRIV)

# --- capture lane
d.v("1  Capture  (MainActivity.kt)", 30, 50, 560, 300, GROUP)
c1 = d.v("CameraX\nback camera (exocentric) / front\n1280x720 RGBA, keep-latest", 50, 95, 250, 85, SMALL)
c2 = d.v("Low-light check\nmean luma every 15th frame\n< 0.18 → torch or white screen ring", 320, 95, 250, 85, SMALL)
c3 = d.v("Idle gate\nno hand for 2 s → analyse every\n2nd frame (saves power)", 50, 205, 250, 85, SMALL)
c4 = d.v("Camera fallback + memory\nmissing camera → use the other;\nchoice remembered", 320, 205, 250, 85, SMALL)
d.v("frame: Bitmap 720x1280", 50, 300, 520, 30, DATA)
d.e(c1, c2); d.e(c1, c3)

# --- perception lane
d.v("2  Perception  (Landmarks.kt, Enhance.kt)", 620, 50, 600, 300, GROUP)
p1 = d.v("Adaptive sharpening\nsharpness check every 5th frame;\nunsharp mask only if blurry & not dark", 640, 95, 270, 85, SMALL)
p2 = d.v("MediaPipe Holistic\nGPU delegate, CPU fallback\npose 33 + hands 2x21 + face 13", 930, 95, 270, 85, SMALL)
p3 = d.v("Landmark frame\n88 points x (x, y, z)\nx, z scaled by aspect", 930, 205, 270, 85, SMALL)
p4 = d.v("Hints\nface > 40% of frame → \"move back\"\ntoo dark → \"add light\"", 640, 205, 270, 85, SMALL)
d.v("88 x 3 floats per frame  |  14.0 ms p95 (laptop CPU)", 640, 300, 560, 30, DATA)
d.e(p1, p2); d.e(p2, p3); d.e(p3, p4, style=DEDGE)

# --- segmentation lane
d.v("3  Sign detection  (Segmenter.kt)", 1250, 50, 630, 300, GROUP)
s1 = d.v("Start\nhand seen in 2 frames\n(+5 frames pre-roll)", 1270, 95, 180, 85, SMALL)
s2 = d.v("End (in ms, any fps)\n0.5 s still, 0.5 s no hands,\nor 3.5 s max", 1470, 95, 190, 85, SMALL)
s3 = d.v("Is it a sign?\n≥ 5 frames, ≥ 0.4 s, and wrist\n≥ 0.2 or fingertip ≥ 0.25 face h.", 1680, 95, 185, 85, SMALL)
s4 = d.v("Why ranges, not paths\ntracker jitter adds up along a path;\n5-95% ranges ignore it → 0/600 idle\nstreams produce a word", 1270, 205, 390, 85, SMALL)
d.v("segment: T x 88 x 3", 1680, 230, 185, 30, DATA)
d.e(s1, s2); d.e(s2, s3)

# --- understanding lane
d.v("4  Understanding  (Features.kt, Recognizer.kt, ONNX Runtime)", 30, 380, 1850, 300, GROUP)
u1 = d.v("Canonical body frame\norigin between shoulders, unit =\nshoulder width; roll and yaw removed;\nvirtual shoulders if cut off", 50, 425, 300, 100, SMALL)
u2 = d.v("Hands\nassigned to nearest pose wrist;\nleft-handers mirrored; gaps ≤ 6\nframes interpolated and flagged", 370, 425, 280, 100, SMALL)
u3 = d.v("239 features per frame\nhand shape 170, orientation 12,\nlocation 24, motion 16, face 11,\nquality 6", 670, 425, 280, 100, SMALL)
u4 = d.v("Motion-weighted sampling\ntrim rest pose, pick 32 frames,\nmore where hands move", 970, 425, 260, 100, SMALL)
u5 = d.v("Sign gate (INT8)\nEchoNet 88 k params, 0.23 MB\n0.12 ms; sign vs not-a-sign\nthreshold keeps 90% of signs", 1250, 425, 290, 100, MSMALL)
u6 = d.v("Word model (INT8)\n4-network ensemble, 5.1 M params\n6.2 MB; 2.4 ms per call\n100 words (WLASL100)", 1560, 425, 300, 100, MSMALL)
d.v("input tensor 1 x 32 x 239  (features: same code as Python, golden-tested to ~1e-6)", 50, 540, 1180, 30, DATA)
u7 = d.v("Inside each network\nper-group projection → quality gate (trusts only what was seen) → 4 depthwise-conv blocks + 2 attention blocks → mean pool → 100 logits", 50, 580, 1180, 80, MSMALL)
d.e(u1, u2); d.e(u2, u3); d.e(u3, u4); d.e(u4, u5); d.e(u5, u6)
d.v("runs once per sign, not per frame", 1250, 540, 610, 30, TAG)

# --- output lane
d.v("5  Output and conversation  (MainActivity.kt, Sentence.kt, Conversation.kt, HudView.kt)", 30, 710, 1850, 330, GROUP)
o1 = d.v("Caption rule\nnot a sign → nothing\np ≥ 45% and 20 pts ahead → word\np ≥ 25% → grey 'word?' + 2 options", 50, 755, 300, 110, SMALL)
o2 = d.v("Sentence builder\nwords join; repeat within 2.5 s\ncounted once; 4 s pause → full stop", 370, 755, 280, 110, SMALL)
o3 = d.v("HUD\ncaption + confidence bar,\nlistening / signing, which hands,\nconversation, hints", 670, 755, 280, 110, OSMALL)
o4 = d.v("Voice\noffline TTS voice only\n(network voices skipped)", 970, 755, 240, 110, OSMALL)
o5 = d.v("Reply (two-way)\nmicrophone → on-device speech\nrecognition → blue caption\n(fallback if no offline pack)", 1230, 755, 300, 110, SMALL)
o6 = d.v("Test & measure\nbuilt-in clips, sentence test,\npick a video, 10-min run →\nlatency, power, heat CSV", 1550, 755, 310, 110, SMALL)
d.v("outputs: text on screen, speech; nothing stored, nothing sent", 1230, 905, 630, 30, DATA)
d.v("Edge cases handled: no hands · too close · too dark · blur · hand at rest · fidget · out-of-vocabulary sign · left-handed signer · "
    "shoulders out of frame · signer far away · missing camera · no offline voice / speech pack · app in test mode (camera paused)",
    50, 960, 1810, 60, NOTE.replace("fontSize=11", "fontSize=13"))
d.e(o1, o2); d.e(o2, o3); d.e(o2, o4, pts=[(510, 880), (1090, 880)]); d.e(o5, o3, pts=[(1380, 900), (810, 900)])

# lane links
d.e(c2, p1, pts=[(605, 137), (605, 137)]); d.e(p3, s1, pts=[(1235, 247), (1235, 137)])
d.e(s3, u1, "segment", pts=[(1772, 365), (200, 365)])
d.e(u6, o1, "top-3 words + p", pts=[(1710, 695), (200, 695)])
d.save("architecture_device.drawio", "On the phone", W, 1070)

# ================= training, export, evaluation =================
t = D()
t.v("A  Data", 10, 10, 560, 560, GROUP)
a1 = t.v("WLASL v0.3 → WLASL100\n100 words (the organisers' list)\n1,466 clips found (2,038 listed)", 30, 55, 250, 95, SMALL)
a2 = t.v("Split by signer\ntrain 1,044 / val 221 / test 201\nval signers never in train", 300, 55, 250, 95, SMALL)
a3 = t.v("Kaggle ASL Signs\n94,477 phone clips, 250 signs,\n21 signers (pre-training)", 30, 175, 250, 95, SMALL)
a4 = t.v("MediaPipe Holistic\nsame landmark model as the app\n→ T x 88 x 3 per clip", 300, 175, 250, 95, SMALL)
a5 = t.v("Augmentation (training only)\ncamera yaw ±25°, pitch/roll ±10°,\nhand size ±15%, speed 0.8-1.25x,\none hand lost, dropped frames, noise", 30, 295, 520, 110, SMALL)
a6 = t.v("Extra copies for robustness\nphone close (shoulders cut), signer far,\ndark / blurred / low-res video re-tracked", 30, 425, 520, 95, SMALL)
t.e(a1, a2); t.e(a3, a4); t.e(a2, a4)

t.v("B  Models", 590, 10, 620, 560, GROUP)
b1 = t.v("EchoNet\n1.28 M params; groups → gated\nfusion → 4 conv + 2 attention", 610, 55, 280, 95, MSMALL)
b2 = t.v("Loss\ncross-entropy + confusion-aware\nmargin (separates look-alike signs)", 910, 55, 280, 95, SMALL)
b3 = t.v("Ensemble of 4, averaged\nbase · near/far · degraded ·\nKaggle-pre-trained (25 epochs)", 610, 175, 580, 95, MSMALL)
b4 = t.v("Sign gate\n88 k params; positives: signs;\nnegatives: rest stretches, synthetic\nfidgets, out-of-vocabulary signs", 610, 295, 580, 110, MSMALL)
b5 = t.v("Why landmarks\nsmall, fast, independent of\nbackground, clothing, skin tone", 610, 425, 580, 95, SMALL)
t.e(b1, b3); t.e(b2, b3, style=DEDGE)

t.v("C  Export", 1230, 10, 330, 560, GROUP)
e1 = t.v("PyTorch → ONNX FP32\n(export_onnx.py)", 1250, 55, 290, 80, SMALL)
e2 = t.v("Static INT8 quantization\nQDQ, per-channel,\n300 calibration clips", 1250, 160, 290, 95, MSMALL)
e3 = t.v("Check\nINT8 vs FP32 accuracy and\nlatency on held-out clips", 1250, 280, 290, 95, SMALL)
e4 = t.v("models/\nmodel_int8.onnx 6.23 MB\ngate_int8.onnx 0.23 MB\n(limit 20 MB)", 1250, 400, 290, 110, OSMALL)
t.e(e1, e2); t.e(e2, e3); t.e(e3, e4)
t.e(b3, e1, pts=[(1220, 222), (1220, 95)])

t.v("D  Evaluation (how every number was measured)", 10, 600, 1550, 300, GROUP)
v1 = t.v("Accuracy 78.1%\n3 signer groups removed from\nall training, models retrained,\nscored once (622 clips)", 30, 645, 280, 115, SMALL)
v2 = t.v("Official test 81.6%\n201 clips through main.py\npredict() → predictions.jsonl", 330, 645, 280, 115, SMALL)
v3 = t.v("Robustness drop 4.64\ndark, blur, tilt, low res,\nocclusion; re-tracked through\nthe deployed pipeline", 630, 645, 280, 115, SMALL)
v4 = t.v("Latency 14.0 ms, drift 5.3%\n10-min sustained run\n(laptop CPU; phone run\nby the app's Measure mode)", 930, 645, 280, 115, SMALL)
v5 = t.v("Live path\nsentence test 20/30 words,\nidle hands 0/600, 9 online\nclips 6/9 in the app", 1230, 645, 310, 115, SMALL)
t.v("results/results.csv · results_detailed.csv · robustness_*.json · gate_effect.json · sustained_cpu.json", 30, 790, 1510, 30, DATA)
t.v("Bundle: scripts/make_submission.py → team_name.zip (src/ + results/), checked on Python 3.11; Docker image runs with --network none",
    30, 830, 1510, 40, NOTE.replace("fontSize=11", "fontSize=13"))
t.e(e4, v2, pts=[(1395, 585), (470, 585)])
t.save("architecture_training.drawio", "Training and evaluation", 1570, 910)
print("ok")
