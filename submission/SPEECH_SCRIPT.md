# Speech script - Palmcast walkthrough (2:47)

For `submission/Palmcast_walkthrough.mp4`. Read at a relaxed pace (about 140
words a minute). Each part lines up with the label on screen; the times are
where that part starts. If a part runs short, just pause - the picture keeps
going. Record in a quiet room, phone mic about 20 cm away, one take per part
is fine (send the parts as separate files and they will be placed on the
timeline).

---

### 01 · Live camera, on the phone - 0:00 to 0:14  (about 30 words)

> This is Palmcast, running on an Android phone with no internet at all.
> I sign "help" to the camera. The app tracks my hands, cuts out the sign,
> and shows **help** at fifty-four percent.

### 02 · Sentences building as they are signed - 0:14 to 0:58  (about 95 words)

> Signing is continuous, so Palmcast doesn't wait for a button. A sign ends
> when the hands pause for half a second, the word is recognised, and it
> joins the sentence that is forming at the top.
>
> These signers were never seen in training. "Man help now." "Woman drink
> hot." "Hat blue white."
>
> Words the model is sure about are shown in white and spoken. A fair guess
> is shown in grey with a question mark and is never spoken. And when the hands
> are resting, nothing is shown - a sign gate and a landmark filter stop
> jitter and fidgeting from turning into words.

### 03 · A video from the web, uploaded into the app - 0:58 to 1:29  (about 60 words)

> It also works on video it has never seen. Here I pick a clip downloaded from
> aslbricks.org. The phone reads it frame by frame - you can see the hand
> outline on every frame - and answers **basketball**, eighty-one percent.
> We tried nine clips from nine different websites: six right on the phone,
> the same six as our Python code.

### 04 · Sentence test on the phone - 1:29 to 2:06  (about 70 words)

> To test sentences honestly, the app plays thirty test clips as ten
> three-word sentences through exactly the live pipeline: hands drop between
> words, a long pause ends the sentence. The result is twenty of thirty words
> right and in the right order. Single words from new signers are right
> seventy-eight percent of the time, against a target of seventy.

### 05 · Built-in test - 2:06 to 2:30  (about 45 words)

> The built-in test runs ten official clips on the phone and compares with the
> computer: nine of ten right, and the same answer as Python on every clip.
> The phone and the evaluation code are one pipeline, not two.

### 06 · predict() for the evaluators - 2:30 to 2:47  (about 35 words)

> For the judges, `predict()` in `main.py` runs fully offline. The word model
> is six point two megabytes in INT8, under fourteen milliseconds per frame.
> Palmcast - sign language, on the device, in real time.

---

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
| Landmark filter, with webcam-like jitter | false words 140/600 → 0/600; 3-sign sentences 33% → 49% |
