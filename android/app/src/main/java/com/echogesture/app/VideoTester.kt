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
class VideoTester(private val context: Context, private val recognizer: Recognizer) {
    data class Row(val name: String, val truth: String?, val python: String?, val app: String,
                   val confidence: Float, val frames: Int, val landmarkMsPerFrame: Double, val modelMs: Double)

    fun builtIn(progress: (String) -> Unit): List<Row> {
        val list = JSONArray(context.assets.open("test_clips.json").bufferedReader().readText())
        return (0 until list.length()).map { i ->
            val c = list.getJSONObject(i)
            val id = c.getString("video_id")
            progress("clip ${i + 1}/${list.length()}  ${c.getString("gloss")}")
            val r = MediaMetadataRetriever()
            context.assets.openFd("test_clips/$id.mp4").use { r.setDataSource(it.fileDescriptor, it.startOffset, it.length) }
            run(r, id, c.getString("gloss"), c.getString("python"))
        }
    }

    fun pick(uri: Uri): Row {
        val r = MediaMetadataRetriever()
        r.setDataSource(context, uri)
        return run(r, uri.lastPathSegment ?: "video", null, null)
    }

    private fun run(r: MediaMetadataRetriever, name: String, truth: String?, python: String?): Row {
        val frames = ArrayList<FloatArray>()
        var landmarkMs = 0.0
        // fresh landmarker per clip: tracking must not carry over between clips (same as Python)
        Landmarks(context, gpu = false).use { lm ->
            val durationMs = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: 0L
            val count = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toInt()
            val n = count ?: (durationMs / 33).toInt()
            for (i in 0 until n) {
                val bmp: Bitmap = (if (Build.VERSION.SDK_INT >= 28 && count != null) r.getFrameAtIndex(i)
                else r.getFrameAtTime(i * 33_000L, MediaMetadataRetriever.OPTION_CLOSEST)) ?: continue
                val argb = if (bmp.config == Bitmap.Config.ARGB_8888) bmp else bmp.copy(Bitmap.Config.ARGB_8888, false)
                val t0 = System.nanoTime()
                frames.add(lm.detect(argb, if (durationMs > 0 && n > 0) i * durationMs / n else i * 33L))
                landmarkMs += (System.nanoTime() - t0) / 1e6
            }
        }
        r.release()
        if (frames.isEmpty()) return Row(name, truth, python, "(no frames)", 0f, 0, 0.0, 0.0)
        val res = recognizer.recognize(frames)
        return Row(name, truth, python, res.words[0], res.probs[0], frames.size, landmarkMs / frames.size, res.modelMs)
    }
}
