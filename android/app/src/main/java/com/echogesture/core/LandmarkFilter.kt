package com.echogesture.core

import kotlin.math.PI
import kotlin.math.sqrt

/**
 * One Euro filter (Casiez et al., CHI 2012) on every landmark, the same as
 * echogesture/live.py LandmarkFilter.
 *
 * A live tracker jitters by a few pixels per frame even on still hands. That
 * jitter makes resting hands look like they move (false signs) and keeps a
 * signer's pause from reading as a pause, so signs run together. The filter
 * smooths slow points strongly and fast points hardly at all, so real signing
 * keeps its timing. A point that disappears starts afresh when it comes back.
 *
 * Measured on the 201 test signs with 0.008 image heights of added jitter:
 * resting-hand streams with a word 140/600 -> 0/600; with 0.004 jitter,
 * 3-sign sequences right and in order 32.8% -> 49.4% (scripts/live_sim.py).
 */
class LandmarkFilter(
    private val minCutoff: Double = 1.5,
    private val beta: Double = 8.0,
    private val dCutoff: Double = 1.0,
) {
    private var x: FloatArray? = null
    private var dx = FloatArray(0)
    private var lastT = 0.0

    fun reset() { x = null }

    /** [frame] is 88*3 floats (NaN = not seen); returns the filtered copy. */
    fun filter(frame: FloatArray, tMs: Long): FloatArray {
        val t = tMs / 1000.0
        val prev = x
        if (prev == null || prev.size != frame.size) {
            x = frame.copyOf(); dx = FloatArray(frame.size); lastT = t
            return frame.copyOf()
        }
        val dt = (t - lastT).coerceIn(1e-3, 0.5)
        lastT = t
        val aD = alpha(dCutoff, dt)
        val out = FloatArray(frame.size)
        for (p in 0 until frame.size / 3) {
            val i = p * 3
            var speed2 = 0.0
            val seen = BooleanArray(3); val fresh = BooleanArray(3)
            for (k in 0 until 3) {
                val f = frame[i + k]
                seen[k] = !f.isNaN()
                fresh[k] = seen[k] && prev[i + k].isNaN()
                dx[i + k] = if (seen[k] && !fresh[k]) {
                    val raw = (f - prev[i + k]) / dt
                    val old = if (dx[i + k].isNaN()) 0f else dx[i + k]
                    (aD * raw + (1 - aD) * old).toFloat()
                } else 0f
                speed2 += dx[i + k].toDouble() * dx[i + k]
            }
            val a = alpha(minCutoff + beta * sqrt(speed2), dt)
            for (k in 0 until 3) {
                val f = frame[i + k]
                out[i + k] = when {
                    !seen[k] -> Float.NaN
                    fresh[k] -> f
                    else -> (a * f + (1 - a) * prev[i + k]).toFloat()
                }
            }
        }
        x = out
        return out.copyOf()
    }

    private fun alpha(cutoff: Double, dt: Double) = 1.0 / (1.0 + 1.0 / (2 * PI * cutoff * dt))
}
