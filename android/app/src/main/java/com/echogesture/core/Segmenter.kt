package com.echogesture.core

/**
 * Turns a stream of frames into sign segments.
 *
 * Idle until a hand has been visible for a few frames, then collects frames.
 * The segment ends when the hands drop out of view, pause briefly, or the
 * segment gets too long. A few frames from before the start are kept, like the
 * rest pose at the start of a WLASL clip.
 *
 * Timing is in milliseconds, not frames, so the same pause ends a sign whether
 * the phone runs at 30 fps or the emulator at 6: you can sign word after word,
 * holding still for half a second in between, without dropping your hands.
 */
class Segmenter(
    private val startFrames: Int = 2,
    private val endMissingMs: Long = 500,
    private val endStillMs: Long = 500,
    private val stillSpeed: Double = 0.1,    // image heights per second (0.004 per frame at 25 fps)
    private val maxMs: Long = 3500,
    private val preRoll: Int = 5,
    // a segment counts as a sign when the wrist covers this much ground, or a
    // fingertip moves this much relative to the wrist (in face heights, 5-95%
    // range so tracker jitter doesn't add up). Signs made in one place (MOTHER,
    // COLOR, CANDY) move the fingers, not the arm. Resting hands with jitter
    // stay below 0.12 / 0.17; 95% of WLASL test signs pass (scripts/live_sim.py).
    private val minWristRange: Double = 0.2,
    private val minFingerRange: Double = 0.25,
    private val minFrames: Int = 5,
    private val minMs: Long = 400,
) {
    enum class State { IDLE, SIGNING }

    var state = State.IDLE
        private set
    private val history = ArrayDeque<FloatArray>()
    private val segment = ArrayList<FloatArray>()
    private var seenRun = 0
    private var missingMs = 0L
    private var stillMs = 0L
    private var startMs = 0L
    private var lastMs = -1L
    private var moved = false
    private var lastWrist: FloatArray? = null
    /** Why the last segment ended (for logs and tuning). */
    var lastEnd = ""
        private set
    var lastRange = ""
        private set
    private val faceHeights = ArrayList<Double>()

    /** Feed one frame (88*3 floats, NaN = missing) taken at [tMs]. Returns a finished segment or null. */
    fun push(frame: FloatArray, tMs: Long): List<FloatArray>? {
        val hand = handSeen(frame, Layout.LHAND) || handSeen(frame, Layout.RHAND)
        val dt = if (lastMs < 0) 40L else (tMs - lastMs).coerceIn(1L, 500L)
        lastMs = tMs
        val speed = wristSpeed(frame) * 1000.0 / dt
        when (state) {
            State.IDLE -> {
                history.addLast(frame)
                while (history.size > preRoll + startFrames) history.removeFirst()
                seenRun = if (hand) seenRun + 1 else 0
                if (seenRun >= startFrames) {
                    state = State.SIGNING
                    segment.clear(); segment.addAll(history)
                    missingMs = 0; stillMs = 0; moved = false; startMs = tMs
                    faceHeights.clear()
                }
            }
            State.SIGNING -> {
                segment.add(frame)
                faceHeight(frame)?.let { faceHeights.add(it) }
                missingMs = if (hand) 0 else missingMs + dt
                if (speed > stillSpeed * 3) moved = true
                stillMs = if (hand && speed < stillSpeed) stillMs + dt else 0
                val done = missingMs >= endMissingMs || (moved && stillMs >= endStillMs) || tMs - startMs >= maxMs
                if (done) {
                    state = State.IDLE
                    history.clear(); seenRun = 0
                    val out = ArrayList(segment)
                    segment.clear()
                    val ok = tMs - startMs >= minMs && isSign(out)
                    lastEnd = "frames=${out.size} ms=${tMs - startMs} missing=$missingMs still=$stillMs moved=$moved accepted=$ok $lastRange"
                    return if (ok) out else null
                }
            }
        }
        return null
    }

    /** Frames collected so far in the current segment (for "recognise now"). */
    fun current(): List<FloatArray> = ArrayList(segment)

    fun reset() { state = State.IDLE; history.clear(); segment.clear(); seenRun = 0; lastMs = -1; lastWrist = null }

    /** Enough frames and enough hand or finger movement relative to the face (scale-free). */
    fun isSign(frames: List<FloatArray>): Boolean {
        if (frames.size < minFrames) return false
        val fh = if (faceHeights.isEmpty()) 0.25 else faceHeights.sorted()[faceHeights.size / 2]
        lastRange = "fh=${"%.3f".format(fh)}"
        for (b in intArrayOf(Layout.RHAND, Layout.LHAND)) {
            val seen = frames.filter { !it[b * 3].isNaN() }
            if (seen.size < 3) continue
            val wr = range(seen.map { it[b * 3] }, seen.map { it[b * 3 + 1] }) / fh
            lastRange += " hand$b seen=${seen.size} wrist=${"%.2f".format(wr)}"
            if (wr >= minWristRange) return true
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
