# Implementation, in pictures

Screenshots of Palmcast running: the release APK on the Android emulator and a phone, plus a terminal.
Numbered in the order of the demo video.

| # | What it shows |
|---|---|
| 01-02 | Live camera on the phone: "help" signed, read as help (54%) |
| 03-04 | Running sentence from signers the model never saw; sure words white, guesses grey |
| 05 | The live pipeline on a laptop (live.py): landmarks and the word as it is signed |
| 06-07 | A clip from aslbricks.org uploaded into the app, read frame by frame: basketball 81% |
| 08-09 | Sentence test: 30 clips as 10 sentences through the live pipeline, 20 of 30 words |
| 10-11 | Built-in ASL test: 9 of 10, every answer the same as Python |
| 12 | The controls: Flip, Voice, Recognise, Test, Reply; hints when too close |
| 13-14 | Test menu, switching to Indian Sign Language (50 words) |
| 15-16 | Built-in ISL test (INCLUDE-50 clips): 10 of 10, same as Python |
| 17-18 | Teach the phone a new sign: type the word, then sign it 3 times |
| 19 | predict() in main.py on 9 web clips, and results.csv |
| 20 | Diagrams: on-device architecture, training and evaluation, per-frame flow |
