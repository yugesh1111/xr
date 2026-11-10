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
    // a segment must move the hand at least this far, in face heights, to count as a
    // sign: holding the hands still in view, or small fidgets, never reach the model
    private val minTravelFaces: Double = 0.8,
    private val minFrames: Int = 10,
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
    private var travel = 0.0
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
                    travel = 0.0; faceHeights.clear()
                }
            }
            State.SIGNING -> {
                segment.add(frame)
                travel += speed
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
                    return if (isSign(out.size)) out else null
                }
            }
        }
        return null
    }

    /** Frames collected so far in the current segment (for "recognise now"). */
    fun current(): List<FloatArray> = ArrayList(segment)

    fun reset() { state = State.IDLE; history.clear(); segment.clear(); seenRun = 0 }

    /** Enough frames and enough hand travel relative to the face (scale-free). */
    fun isSign(frames: Int): Boolean {
        if (frames < minFrames || !moved) return false
        val fh = if (faceHeights.isEmpty()) 0.25 else faceHeights.sorted()[faceHeights.size / 2]
        return travel / fh >= minTravelFaces
    }

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
