# Speech script - Palmcast walkthrough (2:57)

Recorded and edited into `submission/Palmcast_demo_final.mp4` (`submission/mix_voice.py`). Parts 2-5 and 8 are our
first recording, shortened a little; parts 1, 6 and 7 (introduction, Indian Sign Language, teach-a-sign) were recorded later. The track
asks for 2-3 minutes, so a few side remarks were cut to make room.

### Part 1 - Live on the phone  (screen: signing "help" · about 11 s, recorded later by Amritha)

> Hello, this is Palmcast, from team WeCode. Everything you'll see runs on the phone, offline. I sign "help"... and there it is. Help, fifty-four percent.

### Part 2 - How it works, live  (screen: the architecture, then signers building sentences · about 33 s)

> Here's what's happening underneath. On every camera frame, MediaPipe finds eighty-eight points on the body, the hands and the face. We turn those into body-relative features, so it doesn't matter where you stand or what's behind you. When the hands pause for half a second, that stretch is cut out as one sign and handed to the model. Watch the sentence build. Sure words turn white and are spoken; a guess stays grey. And none of these signers were in our training data.

### Part 3 - A clip from the internet  (screen: uploaded web clip, then its result · about 21 s)

> It isn't tied to our camera either. Here we upload a clip from aslbricks dot org. The answer: basketball, eighty-one percent. We tried nine clips from nine different websites. Six came back right, and they're the same six our Python code gets.

### Part 4 - Sentence test  (screen: sentence test on the phone, then the score · about 29 s)

> Next, the sentence test: thirty clips of unseen signers, played as ten short sentences through the live pipeline. It gets twenty out of thirty words right, and in the right order. Before any word reaches the screen, a tiny sign gate, just a quarter of a megabyte, asks: is this a real sign at all? That's why resting or fidgeting hands never turn into words.

### Part 5 - Built-in test  (screen: built-in test, then 9 of 10 · about 24 s)

> The word model itself is four small networks averaged together, quantised to INT8. Six point two megabytes in total. The built-in test runs ten official test clips on the phone. Nine out of ten are correct, and every single answer matches Python.

### Part 6 - Indian Sign Language  (screen: Test menu, switch to ISL, the ISL test, 10 of 10 · about 23 s)

> We also added Indian Sign Language. One tap switches to fifty ISL words. We trained them on INCLUDE, signed by deaf students of St. Louis School for the Deaf, right here in Chennai. It gets ninety-nine percent on its own test set, and forty-five on YouTube signers it has never seen, so there's still work to do there. On the phone: ten out of ten.

### Part 7 - Teach a sign  (screen: the teach dialog, "water", then "sign it, 1 of 3" · about 8 s)

> If you're missing a word, just teach it. Type the word, sign it three times, and the phone remembers it. No retraining, and nothing uploaded.

### Part 8 - For the judges  (screen: terminal with predict() and results · about 22 s)

> For judging, predict() in main.py runs completely offline: about fourteen milliseconds per frame, with less than five points lost on dark, blurry or blocked video. Everything is on GitHub. That's Palmcast: sign language, understood right on the device.

## Numbers you can quote (all measured, in `results/`)

| | |
|---|---|
| Top-1 on signers never seen in training | 78.1% (target 70%) |
| Official WLASL100 test clips | 81.6% |
| Model size, INT8 | 6.2 MB + 0.23 MB sign gate (limit 20 MB) |
| Latency p95 per frame | 14.0 ms (limit 50 ms) |
| Accuracy drop: dark, blur, tilt, low-res, occlusion | 4.6 points (limit 5) |
| Thermal drift over 10 minutes | 5.3% (limit 20%) |
| Resting / fidgeting hands | 0 false words in 600 runs |
| Jitter filter, with webcam-like jitter | false words 140/600 → 0/600; 3-sign sentences 33% → 49% |
