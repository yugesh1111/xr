# Demo video speech script

Read each part while its chapter is on screen. Timings are from the videos in this folder.

## Palmcast_demo.mp4 (175 s)

| time | chapter | say |
|---|---|---|
| 0:00-0:08 | title | We are team WeCode, VIT Chennai. This is Palmcast: sign language to text and speech, on the phone, offline. |
| 0:08-0:24 | live | I sign help to the camera. Hands are tracked, and the word appears: help, fifty four percent. Unsure signs show as a grey guess. |
| 0:24-1:11 | online | Real A S L videos from nine websites, uploaded into the app and read with the same pipeline as the camera. Six of nine are right, and the misses are shown too. |
| 1:11-1:43 | builtin | Ten official test clips on the phone: nine right, all the same as our Python version. |
| 1:43-2:19 | sentence | Thirty clips from unseen signers, as ten sentences. Words join a running sentence: twenty of thirty right, no false words from resting hands. |
| 2:19-2:34 | terminal | The same models on a computer: predict on Python 3.11, and results dot c s v. |
| 2:34-2:42 | requirements | Every track requirement is covered, on-device and private. |
| 2:42-2:50 | results | Seventy eight percent, fourteen milliseconds, six point two megabytes. |
| 2:50-2:54 | end | All on GitHub. Thank you. |

## Palmcast_demo_full.mp4 (407 s)

| time | chapter | say |
|---|---|---|
| 0:00-0:15 | title | Hi, we are team WeCode from VIT Chennai. This is Palmcast, our answer to Track 2, EchoGesture. Palmcast turns American Sign Language into text and speech, live, on a phone, with no internet and no cloud. |
| 0:15-0:33 | install | Palmcast ships as one Android app. Download the APK from our GitHub releases page and install it on any 64-bit Android phone. It asks only for the camera, and the microphone for replies. It has no internet permission at all, so video of people never leaves the device. |
| 0:33-0:48 | live | Here I am signing help to the camera. The app tracks my body and both hands, drawn as outlines. When the sign ends, the word appears: help, at fifty four percent. When it is less sure, it shows a grey guess with two alternatives instead of pretending. |
| 0:48-3:09 | online | Next, real A S L videos from nine different websites, including YouTube, Signing Savvy and Spread the Sign. Each one is uploaded into the app through Pick a video, and read with exactly the same pipeline as the camera. You can see the signer, the hand outlines, and then the answer. Basketball, clothes, cow, full, but and change are right. Candy, bowling and accident are wrong, and we show those too. Six of nine, from sources the model has never seen. |
| 3:09-4:21 | builtin | The built-in test runs ten official WLASL test clips on the phone. Nine are right, and every single answer matches our Python version, so the app and the evaluation code agree. |
| 4:21-5:44 | sentence | The sentence test streams thirty clips, from signers the model never trained on, as ten three-word sentences. Each word appears as its sign ends and joins a running sentence. A pause adds the full stop. Twenty of the thirty words come out right and in order, and resting hands never produce a word. |
| 5:44-6:03 | terminal | The same models run on a computer through the predict function the brief asks for, on Python 3.11. It gives a word, a confidence and a latency for every clip, and results dot c s v shows five of six targets passing. |
| 6:03-6:22 | requirements | Every requirement of the track is covered: fully on-device and exocentric, robust to moving heads, angled signers and low light, zero cloud streaming, pose-invariant hand features, an INT8 ONNX model far under twenty megabytes, and captions and audio for a two-way conversation. |
| 6:22-6:39 | results | Our numbers: seventy eight percent accuracy on people the model never saw, fourteen milliseconds per frame, a six point two megabyte model, and a drop of four point six points under dark, blur, tilt, low resolution and occlusion. Power will be measured on a phone. |
| 6:39-6:47 | end | Everything is on GitHub: code, models, results, the app, the slides and this video. Thank you. |
