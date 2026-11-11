package com.echogesture.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.content.Intent
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import com.echogesture.core.Conversation
import android.widget.Button
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import android.util.Size
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import android.view.WindowManager
import android.view.View
import android.view.ViewGroup
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.camera.core.Camera
import com.echogesture.core.Layout
import com.echogesture.core.Segmenter
import java.util.Locale
import java.util.concurrent.Executors

private const val SHOW_MIN_PROB = 0.45f
private const val SHOW_MIN_MARGIN = 0.20f
// below the gate but still a real candidate: shown as a grey "word?" with the next two
// guesses, not added to the sentence or spoken. On held-out clips 20% of signs land
// here and the right word is among the three shown 85% of the time.
private const val GUESS_MIN_PROB = 0.25f
// when the face is this big in the frame, most signs happen below the image edge
// (hands visible in 15% of frames at 0.5 vs 55% normally): ask the user to move back
private const val TOO_CLOSE_FACE_SHARE = 0.40f
// below this average brightness the hand tracker starts to lose the hands
private const val TOO_DARK_LUMA = 0.18f
private const val LIGHT_OFF_LUMA = 0.45f

class MainActivity : AppCompatActivity() {
    private lateinit var preview: PreviewView
    private lateinit var hud: HudView
    private val worker = Executors.newSingleThreadExecutor()
    private var landmarks: Landmarks? = null
    private var recognizer: Recognizer? = null
    private val segmenter = Segmenter()
    private var stats = Stats(this)
    private var benchRunning = false
    // back camera: the brief is exocentric - the device films the person signing to you; Flip to sign to it yourself
    private var backCamera = true
    private var camera: Camera? = null
    private var lightOn = false        // ring light (front) or torch (back) switched on for low light
    private var darkChecks = 0
    private var brightChecks = 0
    private var speak = false
    private var tts: TextToSpeech? = null
    @Volatile private var ttsReady = false
    private var ttsProblem: String? = null
    private val conversation = Conversation()
    private val sentence = com.echogesture.core.Sentence(onClose = { conversation.add(Conversation.Who.SIGNER, it); refreshTalk() })
    private var speech: SpeechRecognizer? = null
    private lateinit var replyButton: Button
    private var frameIndex = 0L
    private var lastHandMs = 0L
    private var fps = 0.0
    private var lastFrameMs = 0L
    @Volatile private var testing = false
    @Volatile private var cancelTest = false
    private var cameraProvider: ProcessCameraProvider? = null
    private lateinit var testButton: Button
    private var demoClose = false
    private val pickVideo = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) runTest { t -> t.pick(uri) { msg -> runOnUiThread { hud.status = msg; hud.invalidate() } } }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        preview = findViewById(R.id.preview)
        hud = findViewById(R.id.hud)
        // draw under the status and navigation bars; keep text and buttons clear of them
        WindowCompat.setDecorFitsSystemWindows(window, false)
        val bar = findViewById<View>(R.id.bar)
        val barMargin = (bar.layoutParams as ViewGroup.MarginLayoutParams).bottomMargin
        ViewCompat.setOnApplyWindowInsetsListener(hud) { _, insets ->
            val sys = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            hud.insetTop = sys.top.toFloat()
            (bar.layoutParams as ViewGroup.MarginLayoutParams).bottomMargin = barMargin + sys.bottom
            bar.requestLayout()
            insets
        }
        bar.addOnLayoutChangeListener { v, _, _, _, _, _, _, _, _ -> hud.barTop = v.top.toFloat(); hud.invalidate() }
        findViewById<Button>(R.id.flip).setOnClickListener { backCamera = !backCamera; startCamera() }
        findViewById<Button>(R.id.now).setOnClickListener { recognizeNow() }
        findViewById<Button>(R.id.speak).setOnClickListener {
            speak = !speak
            val b = it as Button
            b.setCompoundDrawablesRelativeWithIntrinsicBounds(0, if (speak) R.drawable.ic_voice_on else R.drawable.ic_voice_off, 0, 0)
            b.contentDescription = "Speak words aloud: " + if (speak) "on" else "off"
            if (speak && !ttsReady) Toast.makeText(this, ttsProblem ?: "Voice is still starting, try again in a second", Toast.LENGTH_LONG).show()
            else { if (speak) say("Voice on") else Toast.makeText(this, "Voice off", Toast.LENGTH_SHORT).show() }
        }
        replyButton = findViewById(R.id.reply)
        replyButton.setOnClickListener { toggleReply() }
        testButton = findViewById(R.id.test)
        testButton.setOnClickListener {
            if (testing) { cancelTest = true; hud.status = "Stopping the test…"; hud.invalidate(); return@setOnClickListener }
            AlertDialog.Builder(this).setTitle("Test and measure")
                .setItems(arrayOf("Built-in WLASL test clips", "Pick a video…",
                    if (benchRunning) "Stop measuring and save results" else "Measure speed, power and heat (10 min)")) { _, which ->
                    when (which) {
                        0 -> runTest { t -> t.builtIn { msg -> runOnUiThread { hud.status = msg; hud.invalidate() } } }
                        1 -> pickVideo.launch(arrayOf("video/*"))
                        else -> toggleBench()
                    }
                }.show()
        }
        // screenshot helper: adb shell am start -n com.echogesture.app/.MainActivity --es demo bird --ef conf 0.52 --ez close true
        intent.getStringExtra("demo")?.let { w ->
            hud.word = w.uppercase(); hud.confidence = intent.getFloatExtra("conf", 0.9f)
            hud.alternatives = listOf("BOOK", "BLUE"); hud.guess = intent.getBooleanExtra("guess", false); hud.shownAtMs = SystemClock.uptimeMillis() + 600_000
            conversation.add(Conversation.Who.SIGNER, "Man help now."); conversation.add(Conversation.Who.SPEAKER, "Sure, what happened?")
            hud.turns = conversation.turns(); hud.forming = "Mother drink ${w.lowercase()}"; demoClose = intent.getBooleanExtra("close", false)
        }
        hud.onClearSentence = {
            sentence.clear(); conversation.clear(); refreshTalk()
            Toast.makeText(this, "Conversation cleared", Toast.LENGTH_SHORT).show()
        }
        tts = TextToSpeech(this) { status -> runOnUiThread { setUpVoice(status) } }

        worker.execute {
            // try the GPU first, fall back to CPU if this phone's GPU delegate fails
            landmarks = runCatching { Landmarks(this, gpu = true) }.getOrElse { Landmarks(this, gpu = false) }
            recognizer = Recognizer(this, nnapi = false)
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) startCamera()
        else ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 1)
    }

    override fun onRequestPermissionsResult(code: Int, perms: Array<out String>, results: IntArray) {
        super.onRequestPermissionsResult(code, perms, results)
        val ok = results.firstOrNull() == PackageManager.PERMISSION_GRANTED
        when {
            code == 2 && ok -> startReply()
            code == 2 -> Toast.makeText(this, "Microphone permission is needed for replies", Toast.LENGTH_LONG).show()
            ok -> startCamera()
            else -> Toast.makeText(this, "Camera permission is needed", Toast.LENGTH_LONG).show()
        }
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            cameraProvider = provider
            if (testing) return@addListener
            val pv = Preview.Builder().build().also { it.surfaceProvider = preview.surfaceProvider }
            // 1280x720 instead of CameraX's default 640x480: hands that are small in the
            // image (signer further away) keep enough pixels for finger landmarks
            val resolution = ResolutionSelector.Builder()
                .setResolutionStrategy(ResolutionStrategy(Size(1280, 720),
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                .build()
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(resolution)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(worker) { analyze(it) }
            // a phone (or emulator) without the requested camera: use the one it has
            fun has(back: Boolean) = runCatching {
                provider.hasCamera(if (back) CameraSelector.DEFAULT_BACK_CAMERA else CameraSelector.DEFAULT_FRONT_CAMERA)
            }.getOrDefault(false)
            if (!has(backCamera)) {
                if (!has(!backCamera)) {
                    Toast.makeText(this, "No camera found. Test mode still works.", Toast.LENGTH_LONG).show()
                    return@addListener
                }
                backCamera = !backCamera
            }
            val selector = if (backCamera) CameraSelector.DEFAULT_BACK_CAMERA else CameraSelector.DEFAULT_FRONT_CAMERA
            hud.mirror = !backCamera
            segmenter.reset()
            provider.unbindAll()
            setLight(false)
            camera = provider.bindToLifecycle(this, selector, pv, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyze(image: ImageProxy) {
        image.use {
            val lm = landmarks ?: return
            if (testing) return
            frameIndex++
            val now = SystemClock.uptimeMillis()
            // power saving: when nobody has signed for 2 s, look at every 2nd frame only
            val idle = segmenter.state == Segmenter.State.IDLE && now - lastHandMs > 2000
            if (idle && frameIndex % 2 != 0L) return
            val bmp = rotate(image.toBitmap(), image.imageInfo.rotationDegrees)
            if (frameIndex % 15 == 0L) checkLight(meanLuma(bmp))
            val t0 = SystemClock.elapsedRealtimeNanos()
            val frame = lm.detect(bmp, image.imageInfo.timestamp / 1_000_000)
            val ms = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
            stats.frame(ms)
            if (lastFrameMs > 0) fps = 0.9 * fps + 0.1 * (1000.0 / (now - lastFrameMs).coerceAtLeast(1))
            lastFrameMs = now
            val left = !frame[Layout.LHAND * 3].isNaN(); val right = !frame[Layout.RHAND * 3].isNaN()
            val hands = left || right
            if (hands) lastHandMs = now
            segmenter.push(frame, now)?.let { show(it) }
            // Sentence is only touched on the UI thread (show() adds words there)
            runOnUiThread { sentence.tick(SystemClock.uptimeMillis()) }
            hud.signing = segmenter.state == Segmenter.State.SIGNING
            hud.leftHand = left; hud.rightHand = right
            hud.tooClose = demoClose || faceShare(frame) > TOO_CLOSE_FACE_SHARE
            hud.perf = "${"%.0f".format(fps)} fps, landmarks ${"%.0f".format(ms)} ms"
            hud.setFrame(frame, bmp.width, bmp.height)
        }
    }

    /** Low light: switch the light on after 2 dark checks (~1 s), off again after 4 bright
     *  ones. "Bright" is well above the dark threshold, because the light itself makes
     *  the picture brighter and must not switch itself off. */
    private fun checkLight(luma: Float) {
        if (!lightOn) {
            darkChecks = if (luma < TOO_DARK_LUMA) darkChecks + 1 else 0
            hud.tooDark = darkChecks > 0
            if (darkChecks >= 2) setLight(true)
        } else {
            brightChecks = if (luma > LIGHT_OFF_LUMA) brightChecks + 1 else 0
            if (brightChecks >= 4) setLight(false)
        }
    }

    /** Back camera: the phone's torch. Front camera (no torch): the screen becomes a
     *  ring light, a white border at full screen brightness around the preview. */
    private fun setLight(on: Boolean) {
        lightOn = on; darkChecks = 0; brightChecks = 0
        runOnUiThread {
            val cam = camera
            val torch = backCamera && cam?.cameraInfo?.hasFlashUnit() == true
            if (torch) cam?.cameraControl?.enableTorch(on)
            hud.ringLight = on && !torch
            hud.tooDark = false
            window.attributes = window.attributes.apply {
                screenBrightness = if (on && !torch) 1f else WindowManager.LayoutParams.BRIGHTNESS_OVERRIDE_NONE
            }
            hud.invalidate()
        }
    }

    /** Average brightness 0..1 from a coarse grid of pixels (cheap, every 15th frame). */
    private fun meanLuma(b: Bitmap): Float {
        var sum = 0f; var n = 0
        val sx = maxOf(1, b.width / 32); val sy = maxOf(1, b.height / 32)
        var y = 0
        while (y < b.height) {
            var x = 0
            while (x < b.width) {
                val c = b.getPixel(x, y)
                sum += (0.299f * ((c shr 16) and 255) + 0.587f * ((c shr 8) and 255) + 0.114f * (c and 255)) / 255f
                n++; x += sx
            }
            y += sy
        }
        return if (n == 0) 1f else sum / n
    }

    /** Face height (forehead to chin) as a share of the image height; 0 if no face. */
    private fun faceShare(f: FloatArray): Float {
        val a = (Layout.FACE + Layout.F_FOREHEAD) * 3; val b = (Layout.FACE + Layout.F_CHIN) * 3
        if (f[a].isNaN() || f[b].isNaN()) return 0f
        // units are image heights (x is scaled by width/height), same measure as augment.close_up
        return Math.hypot((f[a] - f[b]).toDouble(), (f[a + 1] - f[b + 1]).toDouble()).toFloat()
    }

    /** Show a word only when the model clearly prefers it: thresholds chosen on
     *  held-out clips so that shown words are right most of the time (README). */
    private fun confident(r: Recognizer.Result) =
        r.probs[0] >= SHOW_MIN_PROB && r.probs[0] - r.probs.getOrElse(1) { 0f } >= SHOW_MIN_MARGIN

    private fun show(frames: List<FloatArray>) {
        val rec = recognizer ?: return
        val r = rec.recognize(frames)
        stats.inference(r.featureMs, r.modelMs)
        runOnUiThread {
            // a weak guess (below 25%) is not shown at all: a wrong word on screen
            // reads as a mistake, an empty caption reads as "still listening"
            val sure = confident(r)
            if (sure || r.probs[0] >= GUESS_MIN_PROB) {
                hud.word = r.words[0]
                hud.confidence = r.probs[0]
                hud.alternatives = r.words.drop(1)
                hud.guess = !sure
                hud.shownAtMs = SystemClock.uptimeMillis()
            }
            if (sure) {
                sentence.add(r.words[0], SystemClock.uptimeMillis())
                refreshTalk()
                if (speak) say(r.words[0].lowercase())
            }
            hud.perf += ", model ${"%.1f".format(r.modelMs)} ms"
            hud.invalidate()
        }
    }

    /** The app has no internet, so the voice must be an installed (offline) one.
     *  Engines often default to a network voice, which then fails silently. */
    private fun setUpVoice(status: Int) {
        val t = tts ?: return
        if (status != TextToSpeech.SUCCESS) { ttsProblem = "No text-to-speech engine on this phone"; return }
        val offline = runCatching { t.voices }.getOrNull().orEmpty()
            .filter { it.locale.language == "en" && !it.isNetworkConnectionRequired &&
                it.features?.contains(TextToSpeech.Engine.KEY_FEATURE_NOT_INSTALLED) != true }
            .sortedWith(compareBy({ it.locale.country != "US" }, { -it.quality }))
        if (offline.isNotEmpty()) t.voice = offline.first() else t.language = Locale.US
        if (offline.isEmpty() && t.isLanguageAvailable(Locale.US) < TextToSpeech.LANG_AVAILABLE) {
            ttsProblem = "No offline English voice. Install one in Settings > Text-to-speech > Install voice data."
            return
        }
        t.setAudioAttributes(android.media.AudioAttributes.Builder()
            .setUsage(android.media.AudioAttributes.USAGE_ASSISTANT)
            .setContentType(android.media.AudioAttributes.CONTENT_TYPE_SPEECH).build())
        ttsReady = true
    }

    private fun say(text: String) {
        if (!ttsReady) return
        tts?.speak(text, TextToSpeech.QUEUE_ADD, null, "w${SystemClock.uptimeMillis()}")
    }

    private fun runTest(job: (VideoTester) -> List<VideoTester.Row>) {
        val rec = recognizer ?: run {
            Toast.makeText(this, "Still loading the model, try again in a moment", Toast.LENGTH_SHORT).show(); return
        }
        testing = true; cancelTest = false
        // pause the camera: the test then gets the whole CPU, and the live
        // pipeline can't interleave with it on the worker thread
        cameraProvider?.unbindAll(); setLight(false)
        hud.word = ""; hud.signing = false; hud.leftHand = false; hud.rightHand = false
        hud.setFrame(FloatArray(Layout.N_POINTS * 3) { Float.NaN }, 1, 1)
        hud.status = "Starting the test…"; hud.invalidate()
        testButton.text = "Stop"
        testButton.setCompoundDrawablesRelativeWithIntrinsicBounds(0, R.drawable.ic_stop, 0, 0)
        worker.execute {
            val rows = runCatching { job(VideoTester(this, rec) { cancelTest }) }
            val stopped = cancelTest
            testing = false
            runOnUiThread {
                hud.status = ""; hud.invalidate()
                testButton.text = "Test"
                testButton.setCompoundDrawablesRelativeWithIntrinsicBounds(0, R.drawable.ic_test, 0, 0)
                segmenter.reset(); startCamera()
                if (stopped && rows.getOrNull().isNullOrEmpty()) {
                    Toast.makeText(this, "Test stopped", Toast.LENGTH_SHORT).show(); return@runOnUiThread
                }
                val dim = ContextCompat.getColor(this, R.color.paper_dim)
                val sure = ContextCompat.getColor(this, R.color.sure)
                val maybe = ContextCompat.getColor(this, R.color.maybe)
                val text = android.text.SpannableStringBuilder()
                fun add(t: String, color: Int? = null, bold: Boolean = false, rel: Float = 1f) {
                    val st = text.length; text.append(t)
                    val sp = android.text.Spanned.SPAN_EXCLUSIVE_EXCLUSIVE
                    color?.let { text.setSpan(android.text.style.ForegroundColorSpan(it), st, text.length, sp) }
                    if (bold) text.setSpan(android.text.style.StyleSpan(android.graphics.Typeface.BOLD), st, text.length, sp)
                    if (rel != 1f) text.setSpan(android.text.style.RelativeSizeSpan(rel), st, text.length, sp)
                }
                rows.fold({ list ->
                    val right = list.count { it.truth != null && it.truth == it.app }
                    val same = list.count { it.python != null && it.python == it.app }
                    if (list.any { it.truth != null }) {
                        add("$right of ${list.size} correct\n", bold = true, rel = 1.25f)
                        add("Same answer as the computer version: $same of ${list.size}\n\n", dim)
                    }
                    list.forEachIndexed { i, r ->
                        val ok = r.truth == null || r.truth == r.app
                        add(if (r.truth == null) "• " else if (ok) "✓ " else "✗ ", if (ok) sure else maybe, bold = true)
                        if (r.truth != null && !ok) add("${r.truth.lowercase()}, app said ")
                        add(r.app.lowercase(), bold = true)
                        add("  ${(r.confidence * 100).toInt()}%\n", dim)
                        val py = r.python?.let { if (it == r.app) "computer agrees" else "computer said ${it.lowercase()}" }
                        add("    ${listOfNotNull(py, "${r.frames} frames", "${"%.0f".format(r.landmarkMsPerFrame)} ms per frame", "model ${"%.1f".format(r.modelMs)} ms").joinToString(", ")}" +
                            if (i < list.size - 1) "\n\n" else "", dim, rel = 0.85f)
                    }
                }, { add("The test could not run: ${it.message}") })
                AlertDialog.Builder(this).setTitle("Test results").setMessage(text).setPositiveButton("OK", null).show()
            }
        }
    }

    private fun recognizeNow() = worker.execute {
        val seg = segmenter.current()
        if (seg.size >= 8) show(seg) else runOnUiThread { Toast.makeText(this, "Sign first, then tap", Toast.LENGTH_SHORT).show() }
    }

    private fun refreshTalk() {
        hud.turns = conversation.turns(); hud.forming = sentence.forming(); hud.invalidate()
    }

    // --- replies: the hearing person speaks, the deaf user reads ---------------------

    private fun toggleReply() {
        if (hud.listening) { speech?.stopListening(); return }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED)
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), 2)
        else startReply()
    }

    /** On-device recognition where the phone has it (Android 12+), so the reply's
     *  audio stays on the phone like the video does. */
    private fun startReply() {
        val onDevice = Build.VERSION.SDK_INT >= 31 && SpeechRecognizer.isOnDeviceRecognitionAvailable(this)
        val sr = when {
            onDevice -> SpeechRecognizer.createOnDeviceSpeechRecognizer(this)
            SpeechRecognizer.isRecognitionAvailable(this) -> SpeechRecognizer.createSpeechRecognizer(this)
            else -> { Toast.makeText(this, "This phone has no speech recognition", Toast.LENGTH_LONG).show(); return }
        }
        sentence.closeNow()  // the signer's sentence ends when the other person starts talking
        speech?.destroy(); speech = sr
        fun done() {
            hud.listening = false; hud.hearing = null; refreshTalk()
            replyButton.text = "Reply"
            replyButton.setCompoundDrawablesRelativeWithIntrinsicBounds(0, R.drawable.ic_mic, 0, 0)
        }
        fun best(b: Bundle?) = b?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()
        sr.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(p: Bundle?) { hud.listening = true; hud.invalidate() }
            override fun onPartialResults(b: Bundle?) { best(b)?.let { if (it.isNotBlank()) { hud.hearing = it; hud.invalidate() } } }
            override fun onResults(b: Bundle?) {
                best(b)?.let { conversation.add(Conversation.Who.SPEAKER, it.replaceFirstChar { c -> c.uppercase() }) }
                done()
            }
            override fun onError(error: Int) {
                done()
                val msg = when (error) {
                    SpeechRecognizer.ERROR_NO_MATCH, SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "Didn't catch that. Tap Reply and speak again."
                    SpeechRecognizer.ERROR_NETWORK, SpeechRecognizer.ERROR_NETWORK_TIMEOUT, SpeechRecognizer.ERROR_SERVER ->
                        "Offline speech isn't installed. Settings > System > Languages > Speech > download English."
                    SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "Microphone permission is needed for replies"
                    else -> "Speech recognition stopped (code $error)"
                }
                Toast.makeText(this@MainActivity, msg, Toast.LENGTH_LONG).show()
            }
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(v: Float) {}
            override fun onBufferReceived(b: ByteArray?) {}
            override fun onEndOfSpeech() {}
            override fun onEvent(t: Int, b: Bundle?) {}
        })
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
            .putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            .putExtra(RecognizerIntent.EXTRA_LANGUAGE, "en-US")
            .putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            .putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
        hud.listening = true; hud.hearing = null; hud.invalidate()
        replyButton.text = "Done"
        replyButton.setCompoundDrawablesRelativeWithIntrinsicBounds(0, R.drawable.ic_stop, 0, 0)
        sr.startListening(intent)
    }

    private fun toggleBench() {
        if (!benchRunning) {
            stats = Stats(this); benchRunning = true; hud.showPerf = true
            Toast.makeText(this, "Measuring. Unplug the phone and sign for 10 minutes, then Test > Stop measuring.", Toast.LENGTH_LONG).show()
        } else {
            benchRunning = false; hud.showPerf = false
            val dir = stats.save("${Build.MANUFACTURER} ${Build.MODEL}")
            AlertDialog.Builder(this).setTitle("Benchmark").setMessage(stats.summary() + "\n\nSaved to\n$dir")
                .setPositiveButton("OK", null).show()
        }
    }

    private fun rotate(b: Bitmap, deg: Int): Bitmap {
        if (deg == 0) return b
        val m = Matrix().apply { postRotate(deg.toFloat()) }
        return Bitmap.createBitmap(b, 0, 0, b.width, b.height, m, true)
    }

    override fun onDestroy() {
        super.onDestroy()
        worker.execute { landmarks?.close(); recognizer?.close() }
        worker.shutdown()
        tts?.shutdown()
        speech?.destroy()
    }
}
