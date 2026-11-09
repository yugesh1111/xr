package com.echogesture.app

import android.content.Context
import android.graphics.Bitmap
import com.echogesture.core.Enhance
import com.echogesture.core.Layout
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticLandmarker

/** Camera frame -> 88 landmarks, same layout and units as echogesture/extract.py. */
class Landmarks(context: Context, gpu: Boolean, private val enhance: Boolean = true) : AutoCloseable {
    private val landmarker: HolisticLandmarker
    private var lastTs = -1L
    private var pixels = IntArray(0)

    /** Frames sharpened so far (blurry and not dark, see core/Enhance.kt). */
    var sharpened = 0L
        private set

    init {
        val base = BaseOptions.builder()
            .setModelAssetPath("holistic_landmarker.task")
            .setDelegate(if (gpu) Delegate.GPU else Delegate.CPU)
            .build()
        val opts = HolisticLandmarker.HolisticLandmarkerOptions.builder()
            .setBaseOptions(base)
            .setRunningMode(RunningMode.VIDEO)
            .build()
        landmarker = HolisticLandmarker.createFromOptions(context, opts)
    }

    /** Returns 88*3 floats; x and z are scaled by width/height like the Python extractor. */
    fun detect(bitmap: Bitmap, timestampMs: Long): FloatArray {
        val ts = maxOf(timestampMs, lastTs + 1)
        lastTs = ts
        val res = landmarker.detectForVideo(BitmapImageBuilder(prepare(bitmap)).build(), ts)
        val out = FloatArray(Layout.N_POINTS * 3) { Float.NaN }
        fun put(at: Int, lms: List<NormalizedLandmark>?, idx: IntArray? = null) {
            if (lms.isNullOrEmpty()) return
            val n = idx?.size ?: lms.size
            for (i in 0 until n) {
                val p = lms[idx?.get(i) ?: i]
                out[(at + i) * 3] = p.x(); out[(at + i) * 3 + 1] = p.y(); out[(at + i) * 3 + 2] = p.z()
            }
        }
        put(Layout.POSE, res.poseLandmarks())
        put(Layout.LHAND, res.leftHandLandmarks())
        put(Layout.RHAND, res.rightHandLandmarks())
        if (res.faceLandmarks().size >= 468) put(Layout.FACE, res.faceLandmarks(), Layout.FACE_IDX)
        val aspect = bitmap.width.toFloat() / bitmap.height
        for (n in 0 until Layout.N_POINTS) { out[n * 3] *= aspect; out[n * 3 + 2] *= aspect }
        return out
    }

    /** Adaptive sharpening, same rule as echogesture/enhance.py. */
    private fun prepare(bitmap: Bitmap): Bitmap {
        if (!enhance) return bitmap
        val w = bitmap.width; val h = bitmap.height
        if (pixels.size != w * h) pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
        if (!Enhance.apply(pixels, w, h)) return bitmap
        sharpened++
        val out = if (bitmap.isMutable && bitmap.config == Bitmap.Config.ARGB_8888) bitmap
                  else bitmap.copy(Bitmap.Config.ARGB_8888, true)
        out.setPixels(pixels, 0, w, 0, 0, w, h)
        return out
    }

    override fun close() = landmarker.close()
}
