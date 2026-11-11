package com.echogesture.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.os.SystemClock
import android.provider.Settings
import android.text.Layout as TextLayout
import android.text.StaticLayout
import android.text.TextPaint
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import androidx.core.content.res.ResourcesCompat
import com.echogesture.core.Layout

/**
 * Heads-up display drawn over the camera preview.
 *
 * Top: two status chips (listening / signing, and which hands are in view, in the
 * same colours as the hand skeletons), then the caption card: the word in large
 * type with a confidence meter under it (mint = sure, marigold = fairly sure, with
 * the next guesses). The last words run underneath. Hints (too close, too dark)
 * sit just above the button bar so they don't cover the signer's face.
 */
class HudView(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {
    var word = ""; var confidence = 0f; var alternatives: List<String> = emptyList()
    var guess = false     // below the confidence gate: grey "word?" plus the next guesses
    var shownAtMs = 0L  // the caption fades out 4 s after it was set
    var tooClose = false  // face fills too much of the frame: hands are mostly out of view
    var tooDark = false   // picture too dark for reliable hand tracking
    var ringLight = false // front camera in low light: white border lights the signer
    var turns: List<com.echogesture.core.Conversation.Turn> = emptyList()  // signed + spoken, oldest first
    var forming: String? = null                // the sentence still being signed
    var hearing: String? = null                // the reply being spoken right now (partial)
    var listening = false                      // microphone open for a reply
    var onClearSentence: (() -> Unit)? = null  // tap on the sentence card
    private var sentenceRect: RectF? = null
    var status = ""       // progress text (test mode); shown when not empty
    var perf = ""         // fps / latency, shown while a bench run is on
    var showPerf = false
    var signing = false
    var leftHand = false; var rightHand = false
    var mirror = false
    var fit = false       // picture letterboxed (test clips) instead of filling the screen (camera)
    var insetTop = 0f     // status bar height
    var barTop = 0f       // y where the button bar starts

    private var frame: FloatArray? = null
    private var aspect = 1f
    private var imgW = 1; private var imgH = 1

    private val dp = resources.displayMetrics.density
    private val sp = resources.displayMetrics.scaledDensity
    private fun col(id: Int) = ContextCompat.getColor(context, id)
    private val cScrim = col(R.color.scrim); private val cPaper = col(R.color.paper)
    private val cDim = col(R.color.paper_dim); private val cSure = col(R.color.sure)
    private val cMaybe = col(R.color.maybe); private val cLeft = col(R.color.left_hand)
    private val cRight = col(R.color.right_hand)
    private val regular = ResourcesCompat.getFont(context, R.font.atkinson_regular)
    private val bold = ResourcesCompat.getFont(context, R.font.atkinson_bold)
    private val reduceMotion = Settings.Global.getFloat(context.contentResolver, Settings.Global.ANIMATOR_DURATION_SCALE, 1f) == 0f

    private val fill = Paint(Paint.ANTI_ALIAS_FLAG)
    private val ring = Paint().apply { color = Color.WHITE; style = Paint.Style.STROKE }
    private val wordPaint = TextPaint(Paint.ANTI_ALIAS_FLAG).apply { typeface = bold; textSize = 46 * sp; color = cPaper }
    private val body = TextPaint(Paint.ANTI_ALIAS_FLAG).apply { typeface = regular; textSize = 15 * sp; color = cPaper }
    private val chipText = TextPaint(Paint.ANTI_ALIAS_FLAG).apply { typeface = bold; textSize = 13 * sp; color = cPaper }
    private val bone = Paint(Paint.ANTI_ALIAS_FLAG).apply { strokeCap = Paint.Cap.ROUND }

    fun setFrame(f: FloatArray, w: Int, h: Int) { frame = f; imgW = w; imgH = h; aspect = w.toFloat() / h; postInvalidate() }

    override fun onDraw(c: Canvas) {
        super.onDraw(c)
        val now = SystemClock.uptimeMillis()
        if (ringLight) {  // drawn first so everything else stays on top
            ring.strokeWidth = 56 * dp
            c.drawRect(0f, 0f, width.toFloat(), height.toFloat(), ring)
        }
        drawSkeleton(c)
        val m = 16 * dp
        var y = insetTop + 12 * dp
        y = drawChips(c, m, y, now) + 10 * dp
        y = drawCaption(c, m, y, now)
        drawBottom(c, m)
        if (signing || now - shownAtMs in 0..4200) postInvalidateOnAnimation()
    }

    // --- status chips ---------------------------------------------------------------

    private fun chip(c: Canvas, x: Float, y: Float, text: String, dots: List<Pair<Int, Boolean>>): Float {
        val h = 30 * dp; val r = 4 * dp; val gap = 11 * dp
        val w = 12 * dp + dots.size * gap + 4 * dp + chipText.measureText(text) + 12 * dp
        fill.color = cScrim; fill.style = Paint.Style.FILL
        c.drawRoundRect(RectF(x, y, x + w, y + h), h / 2, h / 2, fill)
        var cx = x + 12 * dp + r
        for ((color, on) in dots) {
            fill.color = color
            fill.style = if (on) Paint.Style.FILL else Paint.Style.STROKE
            fill.strokeWidth = 1.5f * dp
            c.drawCircle(cx, y + h / 2, r, fill)
            cx += gap
        }
        fill.style = Paint.Style.FILL
        c.drawText(text, cx - r + 4 * dp, y + h / 2 - (chipText.ascent() + chipText.descent()) / 2, chipText)
        return x + w
    }

    private fun drawChips(c: Canvas, m: Float, y: Float, now: Long): Float {
        // the signing dot breathes, so you can tell the app is following you
        val pulse = if (signing && !reduceMotion) (0.55f + 0.45f * kotlin.math.sin(now / 160.0).toFloat()) else 1f
        val dot = if (signing) Color.argb((255 * pulse).toInt(), Color.red(cSure), Color.green(cSure), Color.blue(cSure)) else cDim
        val x = chip(c, m, y, if (signing) "Signing" else "Listening", listOf(dot to signing))
        val hands = when {
            leftHand && rightHand -> "Both hands"
            leftHand || rightHand -> "One hand"
            else -> "No hands"
        }
        // left dot = blue skeleton, right dot = orange skeleton
        chip(c, x + 8 * dp, y, hands, listOf(cLeft to leftHand, cRight to rightHand))
        return y + 30 * dp
    }

    // --- caption --------------------------------------------------------------------

    private fun drawCaption(c: Canvas, m: Float, top: Float, now: Long): Float {
        val age = (now - shownAtMs).let { if (it < 0) 1000 else it }  // future = pinned (demo)
        if (word.isEmpty() || age >= 4000) return top
        // enter: 180 ms rise and fade in; leave: fade over the last 600 ms
        val enter = if (reduceMotion) 1f else (age / 180f).coerceIn(0f, 1f)
        val ease = 1 - (1 - enter) * (1 - enter) * (1 - enter)
        val leave = if (age > 3400) 1 - (age - 3400) / 600f else 1f
        val a = (ease * leave).coerceIn(0f, 1f)
        val sure = confidence >= 0.6f && !guess
        val accent = if (guess) cDim else if (sure) cSure else cMaybe

        val w = width - 2 * m
        val pad = 20 * dp
        val alt = if (!sure && alternatives.isNotEmpty()) "or " + alternatives.joinToString(", ") { it.lowercase() } else ""
        val h = pad + 52 * sp + 14 * dp + 6 * dp + (if (alt.isNotEmpty()) 26 * sp else 0f) + pad
        val dy = (1 - ease) * 10 * dp
        c.save(); c.translate(0f, dy)
        fill.color = cScrim; fill.alpha = (Color.alpha(cScrim) * a).toInt()
        c.drawRoundRect(RectF(m, top, m + w, top + h), 24 * dp, 24 * dp, fill)

        val shown = word.lowercase() + if (guess) "?" else ""
        wordPaint.color = if (guess) cDim else cPaper
        wordPaint.alpha = (Color.alpha(wordPaint.color) * a).toInt()
        // shrink long words to fit one line
        wordPaint.textSize = 46 * sp
        val maxW = w - 2 * pad - 60 * dp
        if (wordPaint.measureText(shown) > maxW) wordPaint.textSize *= maxW / wordPaint.measureText(shown)
        val base = top + pad + 42 * sp
        c.drawText(shown, m + pad, base, wordPaint)
        body.color = accent; body.alpha = (255 * a).toInt(); body.typeface = bold
        val pct = "${(confidence * 100).toInt()}%"
        c.drawText(pct, m + w - pad - body.measureText(pct), base, body)

        // confidence meter
        val my = top + pad + 52 * sp + 14 * dp
        fill.color = cPaper; fill.alpha = (40 * a).toInt()
        c.drawRoundRect(RectF(m + pad, my, m + w - pad, my + 6 * dp), 3 * dp, 3 * dp, fill)
        fill.color = accent; fill.alpha = (255 * a).toInt()
        val fw = (w - 2 * pad) * (confidence.coerceIn(0f, 1f) * ease)
        c.drawRoundRect(RectF(m + pad, my, m + pad + fw, my + 6 * dp), 3 * dp, 3 * dp, fill)
        fill.alpha = 255

        if (alt.isNotEmpty()) {
            body.color = cDim; body.alpha = (Color.alpha(cDim) * a).toInt(); body.typeface = regular
            c.drawText(alt, m + pad, my + 6 * dp + 22 * sp, body)
        }
        body.typeface = regular; body.color = cPaper
        c.restore()
        return top + h
    }

    // --- hints and stats above the button bar ------------------------------------------

    private fun drawBottom(c: Canvas, m: Float) {
        var bottom = (if (barTop > 0) barTop else height.toFloat()) - 12 * dp
        bottom = drawSentence(c, m, bottom)
        val lines = mutableListOf<Pair<String, Int>>()
        if (showPerf && perf.isNotEmpty()) lines += perf to cDim
        if (status.isNotEmpty()) lines += status to cPaper
        for ((text, color) in lines.asReversed()) {
            body.color = color; body.textSize = 13 * sp
            val w = body.measureText(text) + 24 * dp; val h = 28 * dp
            fill.color = cScrim
            c.drawRoundRect(RectF(m, bottom - h, m + w, bottom), h / 2, h / 2, fill)
            c.drawText(text, m + 12 * dp, bottom - h / 2 - (body.ascent() + body.descent()) / 2, body)
            bottom -= h + 8 * dp
        }
        body.textSize = 15 * sp; body.color = cPaper

        val hint = when {
            tooDark -> "Too dark. Turn on a light or face a window."
            tooClose -> "Move the phone back so both hands fit in the picture."
            else -> null
        } ?: return
        val pad = 16 * dp; val w = (width - 2 * m).toInt()
        val layout = StaticLayout.Builder.obtain(hint, 0, hint.length, body, w - (2 * pad + 6 * dp).toInt())
            .setAlignment(TextLayout.Alignment.ALIGN_NORMAL).build()
        val h = layout.height + 2 * pad
        val top = bottom - h
        fill.color = cScrim
        c.drawRoundRect(RectF(m, top, m + w, bottom), 18 * dp, 18 * dp, fill)
        fill.color = cMaybe
        c.drawRoundRect(RectF(m + 10 * dp, top + 12 * dp, m + 14 * dp, bottom - 12 * dp), 2 * dp, 2 * dp, fill)
        c.save(); c.translate(m + pad + 6 * dp, top + pad); layout.draw(c); c.restore()
    }

    // --- running sentence, subtitle-style above the buttons ----------------------------

    private val sentencePaint = TextPaint(Paint.ANTI_ALIAS_FLAG).apply { typeface = bold; textSize = 22 * sp }

    /** The conversation, subtitle-style: signed sentences (mint marker) and spoken
     *  replies (blue marker), older turns dimmed, the live line bright. Shows the
     *  last 4 lines; tap the card to clear. Returns the new bottom edge. */
    private fun drawSentence(c: Canvas, m: Float, bottom: Float): Float {
        val open = forming; val heard = hearing
        if (turns.isEmpty() && open == null && heard == null && !signing && !listening) { sentenceRect = null; return bottom }
        val text = android.text.SpannableStringBuilder()
        fun line(marker: Int, body: String, color: Int) {
            if (text.isNotEmpty()) text.append('\n')
            var st = text.length; text.append("● ")
            text.setSpan(android.text.style.ForegroundColorSpan(marker), st, text.length, 0)
            text.setSpan(android.text.style.RelativeSizeSpan(0.6f), st, text.length - 1, 0)
            st = text.length; text.append(body)
            text.setSpan(android.text.style.ForegroundColorSpan(color), st, text.length, 0)
        }
        val live = open != null || heard != null || signing || listening
        turns.forEachIndexed { i, t ->
            val latest = i == turns.size - 1 && !live
            val marker = if (t.who == com.echogesture.core.Conversation.Who.SIGNER) cSure else cLeft
            line(marker, t.text, if (latest) cPaper else cDim)
        }
        if (open != null || signing) line(cSure, (open ?: "") + if (signing) (if (open == null) "…" else " …") else "", cPaper)
        if (heard != null || listening) line(cLeft, (heard ?: "") + if (listening) (if (heard == null) "Listening…" else " …") else "", cPaper)
        val pad = 16 * dp; val w = (width - 2 * m).toInt()
        sentencePaint.color = cPaper
        val layout = StaticLayout.Builder.obtain(text, 0, text.length, sentencePaint, w - (2 * pad).toInt())
            .setLineSpacing(0f, 1.15f).build()
        val lines = minOf(4, layout.lineCount)
        val firstLine = layout.lineCount - lines
        val textH = (layout.getLineBottom(layout.lineCount - 1) - layout.getLineTop(firstLine)).toFloat()
        val h = textH + 2 * pad
        val rect = RectF(m, bottom - h, m + w, bottom)
        fill.color = cScrim
        c.drawRoundRect(rect, 20 * dp, 20 * dp, fill)
        c.save()
        c.clipRect(rect.left, rect.top + pad, rect.right, rect.bottom - pad)
        c.translate(m + pad, rect.top + pad - layout.getLineTop(firstLine))
        layout.draw(c)
        c.restore()
        sentenceRect = rect
        return rect.top - 10 * dp
    }

    override fun onTouchEvent(e: android.view.MotionEvent): Boolean {
        val r = sentenceRect ?: return false
        if (!r.contains(e.x, e.y)) return false
        if (e.action == android.view.MotionEvent.ACTION_UP) { onClearSentence?.invoke(); performClick() }
        return true
    }

    override fun performClick(): Boolean { super.performClick(); return true }

    // --- skeleton -------------------------------------------------------------------

    private fun drawSkeleton(c: Canvas) {
        val f = frame ?: return
        // PreviewView FILL_CENTER: scale to cover, centred
        val s = if (fit) minOf(width.toFloat() / imgW, height.toFloat() / imgH) else maxOf(width.toFloat() / imgW, height.toFloat() / imgH)
        val ox = (width - imgW * s) / 2; val oy = (height - imgH * s) / 2
        fun px(n: Int): Pair<Float, Float>? {
            val x = f[n * 3]; val y = f[n * 3 + 1]
            if (x.isNaN()) return null
            var vx = x / aspect * imgW * s + ox
            if (mirror) vx = width - vx
            return vx to (y * imgH * s + oy)
        }
        fun line(a: Int, b: Int) { val p = px(a) ?: return; val q = px(b) ?: return; c.drawLine(p.first, p.second, q.first, q.second, bone) }
        bone.color = Color.argb(150, 247, 248, 250); bone.strokeWidth = 2.5f * dp
        for ((a, b) in listOf(11 to 12, 11 to 13, 13 to 15, 12 to 14, 14 to 16)) line(a, b)
        val handBones = listOf(0 to 1, 1 to 2, 2 to 3, 3 to 4, 0 to 5, 5 to 6, 6 to 7, 7 to 8, 5 to 9, 9 to 10, 10 to 11, 11 to 12,
            9 to 13, 13 to 14, 14 to 15, 15 to 16, 13 to 17, 17 to 18, 18 to 19, 19 to 20, 0 to 17)
        for ((block, color) in listOf(Layout.LHAND to cLeft, Layout.RHAND to cRight)) {
            bone.color = color; bone.strokeWidth = 2.5f * dp
            for ((a, b) in handBones) line(block + a, block + b)
            fill.color = cPaper
            for (i in 0 until 21) px(block + i)?.let { c.drawCircle(it.first, it.second, 2.5f * dp, fill) }
        }
    }
}
