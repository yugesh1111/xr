package com.echogesture.core

/**
 * Turns a stream of frames into sign segments.
 *
 * Idle until a hand has been visible for a few frames, then collects frames.
 * The segment ends when the hands drop out of view, stay still for a while,
 * or the segment gets too long. A few frames from before the start are kept,
 * like the rest pose at the start of a WLASL clip.
 */
class Segmenter(
    private val startFrames: Int = 3,
    private val endMissing: Int = 8,
    private val endStill: Int = 15,
    private val stillSpeed: Double = 0.004,
    private val maxFrames: Int = 90,
    private val preRoll: Int = 5,
    // a segment counts as a sign when the wrist covers this much ground, or a
    // fingertip moves this much relative to the wrist (in face heights, 5-95%
    // range so tracker jitter doesn't add up). Signs made in one place (MOTHER,
    // COLOR, CANDY) move the fingers, not the arm. Resting hands with jitter
    // stay below 0.12 / 0.17; 91% of WLASL test signs pass (scripts/live_sim.py).
    private val minWristRange: Double = 0.3,
    private val minFingerRange: Double = 0.4,
    private val minFrames: Int = 8,
) {
    enum class State { IDLE, SIGNING }

    var state = State.IDLE
        private set
    private val history = ArrayDeque<FloatArray>()
    private val segment = ArrayList<FloatArray>()
    private var seenRun = 0
    private var missingRun = 0
    private var stillRun = 0
    private var moved = false
    private var lastWrist: FloatArray? = null
    private val faceHeights = ArrayList<Double>()

    /** Feed one frame (88*3 floats, NaN = missing). Returns a finished segment or null. */
    fun push(frame: FloatArray): List<FloatArray>? {
        val hand = handSeen(frame, Layout.LHAND) || handSeen(frame, Layout.RHAND)
        val speed = wristSpeed(frame)
        when (state) {
            State.IDLE -> {
                history.addLast(frame)
                while (history.size > preRoll + startFrames) history.removeFirst()
                seenRun = if (hand) seenRun + 1 else 0
                if (seenRun >= startFrames) {
                    state = State.SIGNING
                    segment.clear(); segment.addAll(history)
                    missingRun = 0; stillRun = 0; moved = false
                    faceHeights.clear()
                }
            }
            State.SIGNING -> {
                segment.add(frame)
                faceHeight(frame)?.let { faceHeights.add(it) }
                missingRun = if (hand) 0 else missingRun + 1
                if (speed > stillSpeed * 3) moved = true
                stillRun = if (hand && speed < stillSpeed) stillRun + 1 else 0
                val done = missingRun >= endMissing || (moved && stillRun >= endStill) || segment.size >= maxFrames
                if (done) {
                    state = State.IDLE
                    history.clear(); seenRun = 0
                    val out = ArrayList(segment)
                    segment.clear()
                    return if (isSign(out)) out else null
                }
            }
        }
        return null
    }

    /** Frames collected so far in the current segment (for "recognise now"). */
    fun current(): List<FloatArray> = ArrayList(segment)

    fun reset() { state = State.IDLE; history.clear(); segment.clear(); seenRun = 0 }

    /** Enough frames and enough hand or finger movement relative to the face (scale-free). */
    fun isSign(frames: List<FloatArray>): Boolean {
        if (frames.size < minFrames) return false
        val fh = if (faceHeights.isEmpty()) 0.25 else faceHeights.sorted()[faceHeights.size / 2]
        for (b in intArrayOf(Layout.RHAND, Layout.LHAND)) {
            val seen = frames.filter { !it[b * 3].isNaN() }
            if (seen.size < 3) continue
            if (range(seen.map { it[b * 3] }, seen.map { it[b * 3 + 1] }) / fh >= minWristRange) return true
            for (tip in TIPS) {
                val dx = seen.map { it[(b + tip) * 3] - it[b * 3] }
                val dy = seen.map { it[(b + tip) * 3 + 1] - it[b * 3 + 1] }
                if (range(dx, dy) / fh >= minFingerRange) return true
            }
        }
        return false
    }

    /** Diagonal of the 5-95 percentile box of a 2-D point set. */
    private fun range(xs: List<Float>, ys: List<Float>): Double {
        fun span(v: List<Float>): Double {
            val s = v.sorted(); val n = s.size - 1
            return (pct(s, 0.95, n) - pct(s, 0.05, n))
        }
        return Math.hypot(span(xs), span(ys))
    }

    private fun pct(s: List<Float>, q: Double, n: Int): Double {  // linear, like numpy
        val pos = q * n; val lo = pos.toInt(); val hi = minOf(lo + 1, n)
        return s[lo] + (s[hi] - s[lo]) * (pos - lo)
    }

    private companion object { val TIPS = intArrayOf(4, 8, 12, 16, 20) }

    /** Forehead-chin distance in image units, or null when the face is not found. */
    private fun faceHeight(f: FloatArray): Double? {
        val a = (Layout.FACE + Layout.F_FOREHEAD) * 3; val b = (Layout.FACE + Layout.F_CHIN) * 3
        if (f[a].isNaN() || f[b].isNaN()) return null
        return Math.hypot((f[a] - f[b]).toDouble(), (f[a + 1] - f[b + 1]).toDouble()).takeIf { it > 1e-3 }
    }

    private fun handSeen(f: FloatArray, block: Int) = !f[block * 3].isNaN()

    private fun wristSpeed(f: FloatArray): Double {
        var best = -1
        for (b in intArrayOf(Layout.RHAND, Layout.LHAND)) if (!f[b * 3].isNaN()) { best = b; break }
        val prev = lastWrist
        if (best < 0) { lastWrist = null; return 0.0 }
        val w = floatArrayOf(f[best * 3], f[best * 3 + 1])
        lastWrist = w
        if (prev == null) return 0.0
        return Math.hypot((w[0] - prev[0]).toDouble(), (w[1] - prev[1]).toDouble())
    }
}
