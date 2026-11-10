package com.echogesture.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.echogesture.core.Layout

/**
 * Heads-up display drawn over the camera preview.
 * Caption at the top (away from the signer's hands), coloured by confidence;
 * a low-confidence word is shown with a question mark and its alternatives
 * instead of being presented as certain. Status line at the bottom.
 */
class HudView(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {
    var word = ""; var confidence = 0f; var alternatives: List<String> = emptyList()
    var shownAtMs = 0L  // the caption fades out 4 s after it was set
    var tooClose = false  // face fills too much of the frame: hands are mostly out of view
    var tooDark = false   // picture too dark for reliable hand tracking
    var ringLight = false // front camera in low light: white border lights the signer
    private val ring = Paint().apply { color = Color.WHITE; style = Paint.Style.STROKE }
    var transcript: List<String> = emptyList()
    var status = ""
    var signing = false
    var handsSeen = false
    var mirror = false
    private var frame: FloatArray? = null
    private var aspect = 1f
    private var imgW = 1; private var imgH = 1

    private val dp = resources.displayMetrics.density
    private val pill = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.argb(170, 0, 0, 0) }
    private val big = Paint(Paint.ANTI_ALIAS_FLAG).apply { textSize = 40 * dp; isFakeBoldText = true; textAlign = Paint.Align.CENTER }
    private val small = Paint(Paint.ANTI_ALIAS_FLAG).apply { textSize = 15 * dp; color = Color.WHITE; textAlign = Paint.Align.CENTER }
    private val mono = Paint(Paint.ANTI_ALIAS_FLAG).apply { textSize = 12 * dp; color = Color.WHITE }
    private val bone = Paint(Paint.ANTI_ALIAS_FLAG).apply { strokeWidth = 3 * dp; color = Color.argb(200, 0, 220, 255) }
    private val joint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.argb(230, 255, 255, 255) }

    fun setFrame(f: FloatArray, w: Int, h: Int) { frame = f; imgW = w; imgH = h; aspect = w.toFloat() / h; postInvalidate() }

    override fun onDraw(c: Canvas) {
        super.onDraw(c)
        if (ringLight) {  // drawn first so captions and buttons stay on top
            ring.strokeWidth = 56 * dp
            c.drawRect(0f, 0f, width.toFloat(), height.toFloat(), ring)
        }
        drawSkeleton(c)
        val w = width.toFloat()
        // caption
        if (word.isNotEmpty() && android.os.SystemClock.uptimeMillis() - shownAtMs < 4000) {
            val color = when {
                confidence >= 0.6f -> Color.rgb(80, 230, 120)
                confidence >= 0.35f -> Color.rgb(255, 196, 0)
                else -> Color.rgb(200, 200, 200)
            }
            big.color = color
            val text = if (confidence >= 0.35f) word else "$word?"
            val top = 40 * dp
            c.drawRoundRect(RectF(16 * dp, top, w - 16 * dp, top + 96 * dp), 18 * dp, 18 * dp, pill)
            c.drawText(text, w / 2, top + 50 * dp, big)
            // alternatives only while the model is unsure between a few words
            val alt = if (confidence < 0.6f && alternatives.isNotEmpty()) "or: " + alternatives.joinToString("  ·  ") else ""
            c.drawText("${(confidence * 100).toInt()}%   $alt", w / 2, top + 80 * dp, small)
        }
        val hint = when {
            tooDark -> "Too dark: turn on a light or face a window"
            tooClose -> "Move the phone back a little so your hands are in view"
            else -> null
        }
        if (hint != null) {
            val top = height - 205 * dp
            c.drawRoundRect(RectF(16 * dp, top, w - 16 * dp, top + 44 * dp), 12 * dp, 12 * dp, pill)
            small.color = Color.rgb(255, 196, 0)
            c.drawText(hint, w / 2, top + 28 * dp, small)
            small.color = Color.WHITE
        }
        if (transcript.isNotEmpty()) {
            c.drawText(transcript.joinToString(" "), w / 2, 160 * dp, small)
        }
        // status line
        val state = if (signing) "● signing" else "○ listening"
        val hands = if (handsSeen) "hands ✓" else "hands –"
        c.drawRoundRect(RectF(8 * dp, height - 150 * dp, w - 8 * dp, height - 112 * dp), 10 * dp, 10 * dp, pill)
        c.drawText("$state   $hands   $status", 16 * dp, height - 126 * dp, mono)
    }

    private fun drawSkeleton(c: Canvas) {
        val f = frame ?: return
        // PreviewView FILL_CENTER: scale to cover, centred
        val s = maxOf(width.toFloat() / imgW, height.toFloat() / imgH)
        val ox = (width - imgW * s) / 2; val oy = (height - imgH * s) / 2
        fun px(n: Int): Pair<Float, Float>? {
            val x = f[n * 3]; val y = f[n * 3 + 1]
            if (x.isNaN()) return null
            var vx = x / aspect * imgW * s + ox
            if (mirror) vx = width - vx
            return vx to (y * imgH * s + oy)
        }
        fun line(a: Int, b: Int) { val p = px(a) ?: return; val q = px(b) ?: return; c.drawLine(p.first, p.second, q.first, q.second, bone) }
        for ((a, b) in listOf(11 to 12, 11 to 13, 13 to 15, 12 to 14, 14 to 16)) line(a, b)
        val handBones = listOf(0 to 1, 1 to 2, 2 to 3, 3 to 4, 0 to 5, 5 to 6, 6 to 7, 7 to 8, 5 to 9, 9 to 10, 10 to 11, 11 to 12,
            9 to 13, 13 to 14, 14 to 15, 15 to 16, 13 to 17, 17 to 18, 18 to 19, 19 to 20, 0 to 17)
        for (block in intArrayOf(Layout.LHAND, Layout.RHAND)) {
            for ((a, b) in handBones) line(block + a, block + b)
            for (i in 0 until 21) px(block + i)?.let { c.drawCircle(it.first, it.second, 3 * dp, joint) }
        }
    }
}
