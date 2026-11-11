package com.echogesture.app

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import org.json.JSONArray

/**
 * Runs whole video clips through the same on-device pipeline as the camera
 * (MediaPipe -> features -> INT8 model), like main.py's predict() does on a
 * computer. Works on the emulator and needs nobody signing.
 *
 * Built-in clips are WLASL test-split videos listed in test_clips.json with
 * their true word and the word our Python pipeline predicted, so the app's
 * answer can be compared with both.
 */
class VideoTester(private val context: Context, private val recognizer: Recognizer,
                  private val cancelled: () -> Boolean = { false }) {
    data class Row(val name: String, val truth: String?, val python: String?, val app: String,
                   val confidence: Float, val frames: Int, val landmarkMsPerFrame: Double, val modelMs: Double)

    fun builtIn(progress: (String) -> Unit): List<Row> {
        val list = JSONArray(context.assets.open("test_clips.json").bufferedReader().readText())
        // skip clips whose video was not packaged (the videos are not in git; a build
        // made without data/wlasl/raw/ has the list but not every file)
        val bundled = context.assets.list("test_clips")?.toSet() ?: emptySet()
        val ids = (0 until list.length()).map { list.getJSONObject(it) }.filter { "${it.getString("video_id")}.mp4" in bundled }
        val rows = ArrayList<Row>()
        for ((i, c) in ids.withIndex()) {
            if (cancelled()) break
            val id = c.getString("video_id")
            val r = MediaMetadataRetriever()
            context.assets.openFd("test_clips/$id.mp4").use { r.setDataSource(it.fileDescriptor, it.startOffset, it.length) }
            rows += run(r, id, c.getString("gloss"), c.getString("python")) { done, total ->
                progress("Testing clip ${i + 1} of ${ids.size}, frame $done of $total")
            } ?: break
        }
        return rows
    }

    fun pick(uri: Uri, progress: (String) -> Unit): List<Row> {
        val r = MediaMetadataRetriever()
        r.setDataSource(context, uri)
        return listOfNotNull(run(r, uri.lastPathSegment ?: "video", null, null) { done, total ->
            progress("Testing your video, frame $done of $total")
        })
    }

    /** Null when the run was cancelled. */
    private fun run(r: MediaMetadataRetriever, name: String, truth: String?, python: String?,
                    progress: (Int, Int) -> Unit): Row? {
        val frames = ArrayList<FloatArray>()
        var landmarkMs = 0.0
        // fresh landmarker per clip: tracking must not carry over between clips (same as Python)
        Landmarks(context, gpu = false).use { lm ->
            val durationMs = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: 0L
            val count = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toInt()
            val n = count ?: (durationMs / 33).toInt()
            var i = 0
            while (i < n) {
                if (cancelled()) { r.release(); return null }
                progress(i, n)
                // decode in batches: one seek per batch instead of one per frame
                val batch: List<Bitmap?> = if (Build.VERSION.SDK_INT >= 28 && count != null)
                    runCatching { r.getFramesAtIndex(i, minOf(BATCH, n - i)) }.getOrElse { List(minOf(BATCH, n - i)) { k -> r.getFrameAtIndex(i + k) } }
                else List(minOf(BATCH, n - i)) { k -> r.getFrameAtTime((i + k) * 33_000L, MediaMetadataRetriever.OPTION_CLOSEST) }
                for ((k, bmp) in batch.withIndex()) {
                    if (bmp == null) continue
                    val argb = if (bmp.config == Bitmap.Config.ARGB_8888) bmp else bmp.copy(Bitmap.Config.ARGB_8888, false)
                    val t0 = System.nanoTime()
                    frames.add(lm.detect(argb, if (durationMs > 0 && n > 0) (i + k) * durationMs / n else (i + k) * 33L))
                    landmarkMs += (System.nanoTime() - t0) / 1e6
                }
                i += batch.size.coerceAtLeast(1)
            }
            progress(n, n)
        }
        r.release()
        if (frames.isEmpty()) return Row(name, truth, python, "(no frames)", 0f, 0, 0.0, 0.0)
        val res = recognizer.recognize(frames)  // same model the camera path uses
        return Row(name, truth, python, res.words[0], res.probs[0], frames.size, landmarkMs / frames.size, res.modelMs)
    }

    private companion object { const val BATCH = 12 }
}
