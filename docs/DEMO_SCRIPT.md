# Demo video plan (max 3 minutes)

The judges score: accuracy (35%), edge latency (20%), model size/quantization
(20%), robustness + HUD clarity (25%). The video should show each one.

## Before recording

1. Start the emulator **from your own Terminal** so macOS can ask for the camera:

   ```bash
   /opt/homebrew/share/android-commandlinetools/emulator/emulator -avd twofold_phone -no-snapshot-load -gpu host
   ```

   Click **Allow** when macOS asks for camera and microphone access. The
   emulator's back camera (the app's default, pointed at the signer as in the
   brief) is the Mac camera; its front camera is a fake room (the emulator can
   give the Mac camera to only one of them).
2. Install the latest APK: drag `android/app/build/outputs/apk/debug/app-arm64-v8a-debug.apk`
   (or the release `Palmcast-arm64.apk`) onto the emulator window.
3. Light: face a window or lamp, plain wall behind you, sleeves that contrast
   with your hands. Sit so your head, chest and both hands are in view.
4. Record the screen with **Cmd+Shift+5** (select the emulator window), or
   QuickTime -> New Screen Recording.
5. Practise each word a few times first and check it shows up. Look up the
   WLASL form of each sign (the model learned those variants).

Give the emulator 8 cores, 4 GB and the host GPU (Device Manager -> Edit ->
Advanced; already set on this Mac): it then runs at about 11 frames per second
instead of 5. Leave audio on so Voice can be heard. The emulator is still slower than a phone, so sign a little
slower than normal and drop your hands between words. Speed and power numbers
in the video come from the phone run (Bench), not the emulator.

## Words to sign

The most reliable words on held-out clips (100% right, high confidence):

**no, help, fish, work, meet, hat, man, now, drink, computer, blue, white,
orange, year, tall, hot**

Good sequences: `man - help - now`, `drink - hot`, `computer - work`,
`no - help`. Show 5-6 words; one clean take each is enough.

## Shot list

| time | show | say (or caption) |
|---|---|---|
| 0:00-0:15 | Title card: Palmcast, team, Track 02 (EchoGesture) | "Sign language to text, live, on the phone, no internet." |
| 0:15-0:30 | App info: only the Camera permission; airplane mode on | "Nothing leaves the device." |
| 0:30-1:30 | Live signing, 5-6 words, letting the sentence build at the bottom (half-second pause between words, then 4 s to end the sentence). Keep the hand outlines, "Signing" chip, caption card and word history in frame. Turn **Voice** on for one or two words. | "Body and hand points from MediaPipe, a 6 MB INT8 model, word in about a second." |
| 1:30-1:45 | Hands resting in view, nothing appears. Then one sign that comes up as grey "word?" with alternatives | "It only shows a word when it's sure; otherwise a marked guess." |
| 1:45-2:05 | Robustness: dim the light (ring light/hint appears), lean in close ("Move the phone back"), sign one word from further away | "Low light, too close, far away." |
| 2:05-2:25 | **Test -> Built-in WLASL test clips** -> "9 of 10 correct, same as the computer" | "Same answers on the phone as offline." |
| 2:25-2:50 | results.csv on screen: 78.1% on unseen signers, 6.2 MB, robustness drop 4.8, phone latency/power/thermal from the Bench run | Read the numbers vs targets. |
| 2:50-3:00 | GitHub link + APK link | |

## If a word doesn't show

- Start and end with hands down, sign a bit bigger.
- Tap **Recognise** right after the sign.
- A grey "word?" with the right word among the options is fine to keep in;
  it shows the app being honest about uncertainty.
