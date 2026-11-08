package com.echogesture.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.speech.tts.TextToSpeech
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
import com.echogesture.core.Segmenter
import java.util.Locale
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var preview: PreviewView
    private lateinit var hud: HudView
    private val worker = Executors.newSingleThreadExecutor()
    private var landmarks: Landmarks? = null
    private var recognizer: Recognizer? = null
    private val segmenter = Segmenter()
    private var stats = Stats(this)
    private var benchRunning = false
    private var backCamera = true
    private var speak = false
    private var tts: TextToSpeech? = null
    private val transcript = ArrayDeque<String>()
    private var frameIndex = 0L
    private var lastHandMs = 0L
    private var fps = 0.0
    private var lastFrameMs = 0L
    @Volatile private var testing = false
    private val pickVideo = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) runTest { t -> listOf(t.pick(uri)) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        preview = findViewById(R.id.preview)
        hud = findViewById(R.id.hud)
        findViewById<Button>(R.id.flip).setOnClickListener { backCamera = !backCamera; startCamera() }
        findViewById<Button>(R.id.now).setOnClickListener { recognizeNow() }
        findViewById<Button>(R.id.speak).setOnClickListener {
            speak = !speak
            (it as Button).text = if (speak) "Voice on" else "Voice off"
        }
        findViewById<Button>(R.id.bench).setOnClickListener { toggleBench(it as Button) }
        findViewById<Button>(R.id.test).setOnClickListener {
            AlertDialog.Builder(this).setTitle("Test with videos")
                .setItems(arrayOf("Built-in WLASL test clips", "Pick a video…")) { _, which ->
                    if (which == 0) runTest { t -> t.builtIn { msg -> runOnUiThread { hud.status = msg; hud.invalidate() } } }
                    else pickVideo.launch(arrayOf("video/*"))
                }.show()
        }
        tts = TextToSpeech(this) { if (it == TextToSpeech.SUCCESS) tts?.language = Locale.US }

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
        if (results.firstOrNull() == PackageManager.PERMISSION_GRANTED) startCamera()
        else Toast.makeText(this, "Camera permission is needed", Toast.LENGTH_LONG).show()
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val pv = Preview.Builder().build().also { it.surfaceProvider = preview.surfaceProvider }
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(worker) { analyze(it) }
            val selector = if (backCamera) CameraSelector.DEFAULT_BACK_CAMERA else CameraSelector.DEFAULT_FRONT_CAMERA
            hud.mirror = !backCamera
            segmenter.reset()
            provider.unbindAll()
            provider.bindToLifecycle(this, selector, pv, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyze(image: ImageProxy) {
        image.use {
            val lm = landmarks ?: return
            if (testing) return
            frameIndex++
            val now = SystemClock.uptimeMillis()
            // power saving: when nobody has signed for 2 s, look at every 3rd frame only
            val idle = segmenter.state == Segmenter.State.IDLE && now - lastHandMs > 2000
            if (idle && frameIndex % 3 != 0L) return
            val bmp = rotate(image.toBitmap(), image.imageInfo.rotationDegrees)
            val t0 = SystemClock.elapsedRealtimeNanos()
            val frame = lm.detect(bmp, image.imageInfo.timestamp / 1_000_000)
            val ms = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
            stats.frame(ms)
            if (lastFrameMs > 0) fps = 0.9 * fps + 0.1 * (1000.0 / (now - lastFrameMs).coerceAtLeast(1))
            lastFrameMs = now
            val hands = !frame[33 * 3].isNaN() || !frame[54 * 3].isNaN()
            if (hands) lastHandMs = now
            segmenter.push(frame)?.let { show(it) }
            hud.signing = segmenter.state == Segmenter.State.SIGNING
            hud.handsSeen = hands
            hud.status = "${"%.0f".format(fps)} fps  landmarks ${"%.0f".format(ms)} ms" + if (benchRunning) "  [bench]" else ""
            hud.setFrame(frame, bmp.width, bmp.height)
        }
    }

    private fun show(frames: List<FloatArray>) {
        val rec = recognizer ?: return
        val r = rec.recognize(frames)
        stats.inference(r.featureMs, r.modelMs)
        runOnUiThread {
            hud.word = r.words[0]
            hud.confidence = r.probs[0]
            hud.alternatives = r.words.drop(1)
            if (r.probs[0] >= 0.35f) {
                transcript.addLast(r.words[0]); while (transcript.size > 6) transcript.removeFirst()
                hud.transcript = transcript.toList()
                if (speak) tts?.speak(r.words[0].lowercase(), TextToSpeech.QUEUE_ADD, null, "w")
            }
            hud.status += "  model ${"%.1f".format(r.modelMs)} ms"
            hud.invalidate()
        }
    }

    private fun runTest(job: (VideoTester) -> List<VideoTester.Row>) {
        val rec = recognizer ?: return
        testing = true
        Toast.makeText(this, "Running test videos on the device…", Toast.LENGTH_SHORT).show()
        worker.execute {
            val rows = runCatching { job(VideoTester(this, rec)) }
            testing = false
            runOnUiThread {
                val text = rows.fold({ list ->
                    val right = list.count { it.truth != null && it.truth == it.app }
                    val same = list.count { it.python != null && it.python == it.app }
                    val head = if (list.any { it.truth != null })
                        "Correct: $right/${list.size}   Same as Python: $same/${list.size}\n\n" else ""
                    head + list.joinToString("\n") { r ->
                        val mark = when { r.truth == null -> "•"; r.truth == r.app -> "✓"; else -> "✗" }
                        "$mark ${r.truth ?: r.name} → ${r.app} (${(r.confidence * 100).toInt()}%)" +
                            (r.python?.let { "  [python: $it]" } ?: "") +
                            "\n    ${r.frames} frames, landmarks ${"%.0f".format(r.landmarkMsPerFrame)} ms/frame, model ${"%.1f".format(r.modelMs)} ms"
                    }
                }, { "Test failed: ${it.message}" })
                AlertDialog.Builder(this).setTitle("Test results").setMessage(text).setPositiveButton("OK", null).show()
            }
        }
    }

    private fun recognizeNow() = worker.execute {
        val seg = segmenter.current()
        if (seg.size >= 8) show(seg) else runOnUiThread { Toast.makeText(this, "Sign first, then tap", Toast.LENGTH_SHORT).show() }
    }

    private fun toggleBench(b: Button) {
        if (!benchRunning) {
            stats = Stats(this); benchRunning = true; b.text = "Stop bench"
            Toast.makeText(this, "Benchmark running - unplug the phone, sign for 10 minutes", Toast.LENGTH_LONG).show()
        } else {
            benchRunning = false; b.text = "Bench"
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
    }
}
