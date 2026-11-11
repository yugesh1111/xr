# EchoGesture - how to use the app

## 1. Install

1. On the phone, open the [APK link](https://github.com/Amritha902/xr/releases/latest/download/EchoGesture-arm64.apk) and download it.
2. Tap the downloaded file. If Android says the source isn't allowed, tap
   **Settings**, turn on **Allow from this source**, go back, tap **Install**.
3. If Play Protect warns about an unknown app, tap **Install anyway**
   (the app is not on the Play Store, that is all the warning means).
4. Open **EchoGesture** and tap **Allow** for the camera. That is the only
   permission it asks for. It never uses the internet.

Needs a 64-bit Android phone with Android 8 or newer.

## 2. Set up

- Prop the phone up (a stand, a cup, a stack of books) so your hands stay free.
- **Front camera** (default) to sign yourself; **Flip** for the back camera to
  caption someone else.
- Distance: your head, chest and both hands in view. Arm's length works;
  further away also works.
- Light: face a window or a lamp. If it's too dark the app turns on the
  torch (back camera) or a white ring around the screen (front camera).
- Plain background helps but is not required.

## 3. Sign

1. Start with your hands down.
2. Sign **one word**, then bring your hands down again.
3. The word appears at the top with a confidence (%). A short pause between
   words is enough; the app finds where each sign starts and ends.

What you see on screen:

| On screen | Meaning |
|---|---|
| Word in **green** | sure (60% or more) |
| Word in **amber** + "or: ..." | fairly sure; the two next guesses are listed |
| Nothing | not sure enough to show a word - sign it again |
| `○ listening` / `● signing` | waiting / sign in progress |
| `hands ✓` / `hands –` | hands found / not found |
| "Move the phone back a little..." | too close, hands out of view |
| "Too dark..." | add light |
| Line of words under the caption | the last few words recognised |

It knows **100 words** (WLASL100): accident, africa, all, apple, basketball,
bed, before, bird, birthday, black, blue, book, bowling, brown, but, can,
candy, chair, change, cheat, city, clothes, color, computer, cook, cool, corn,
cousin, cow, dance, dark, deaf, decide, doctor, dog, drink, eat, enjoy,
family, fine, finish, fish, forget, full, give, go, graduate, hat, hearing,
help, hot, how, jacket, kiss, language, last, later, letter, like, man, many,
medicine, meet, mother, need, no, now, orange, paint, paper, pink, pizza,
play, pull, purple, right, same, school, secretary, shirt, short, son, study,
table, tall, tell, thanksgiving, thin, thursday, time, walk, want, what,
white, who, woman, work, wrong, year, yes.

## 4. Buttons

| Button | What it does |
|---|---|
| **Flip** | front / back camera |
| **Recognise** | give a word now for what you just signed (if it didn't show one by itself) |
| **Voice off / on** | speak each word aloud (offline voice) |
| **Bench** | start / stop a measurement run (see 6) |
| **Test** | run test videos instead of the camera (see 5) |

## 5. Check it works (no signer needed)

**Test -> Built-in WLASL test clips.** Runs 10 test videos through the same
pipeline. You get a list: true word -> app's word (confidence), the answer
of the computer version, and timing. Expected: 9 of 10 correct, all the same
as the computer.

**Test -> Pick a video...** runs any video on the phone (one word per clip).

## 6. Measure speed, power and heat (for results.csv)

1. Charge the phone, then **unplug it** (power is read from the battery).
2. Tap **Bench**, then sign words continuously for **10 minutes**.
3. Tap **Stop bench**. A summary appears and a file is saved to
   `Android/data/com.echogesture.app/files/bench_<time>/results_device.csv`.
4. Copy that file to the computer (USB file transfer, or
   `adb pull /sdcard/Android/data/com.echogesture.app/files/`) into `results/`, then:

```bash
python scripts/make_results.py --eval runs/h100_ens12/eval_holdout_int8.json runs/h200_ens12/eval_holdout_int8.json runs/h300_ens12/eval_holdout_int8.json \
  --eval-test runs/f_ens4/eval_test_int8.json \
  --robustness results/robustness_landmark.json results/robustness_video_enhanced.json \
  --run-key f_base+f_nf+f_deg+f_e12p25 --robustness-level video_enhanced \
  --predictions results/predictions.jsonl --device results/results_device.csv
```

This fills the latency, power and thermal rows of `results/results.csv`.

## 7. If something is off

| Problem | Try |
|---|---|
| Black camera | close and reopen the app; check camera permission in Settings -> Apps -> EchoGesture |
| `hands –` all the time | step back, more light, keep hands in front of your chest |
| No word appears | sign a bit bigger and slower, start and end with hands down, or tap **Recognise** |
| Wrong word | check it is one of the 100 words; sign facing the camera |
| App closes on start | reinstall the latest APK; tell us the phone model |
