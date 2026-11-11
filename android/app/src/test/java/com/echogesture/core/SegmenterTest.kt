package com.echogesture.core

import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test
import java.util.Random

/** Sign / no-sign decisions of the live segmenter on synthetic streams. */
class SegmenterTest {
    private val rnd = Random(0)

    /** Face 0.25 tall at the top, right hand near the chest; [move] shifts wrist and fingertips. */
    private fun frame(wrist: Pair<Float, Float>?, finger: Float = 0f, noise: Float = 0.002f): FloatArray {
        val f = FloatArray(Layout.N_POINTS * 3) { Float.NaN }
        fun set(i: Int, x: Float, y: Float) {
            f[i * 3] = x + noise * rnd.nextGaussian().toFloat(); f[i * 3 + 1] = y + noise * rnd.nextGaussian().toFloat(); f[i * 3 + 2] = 0f
        }
        set(Layout.FACE + Layout.F_FOREHEAD, 0.5f, 0.15f); set(Layout.FACE + Layout.F_CHIN, 0.5f, 0.40f)
        if (wrist != null) for (j in 0 until 21) {
            val tip = j in intArrayOf(4, 8, 12, 16, 20)
            set(Layout.RHAND + j, wrist.first + 0.01f * (j % 5), wrist.second - 0.01f * (j / 5) - if (tip) finger else 0f)
        }
        return f
    }

    private fun run(stream: List<FloatArray>): List<FloatArray>? {
        val s = Segmenter()
        var out: List<FloatArray>? = null
        for ((i, f) in stream.withIndex()) s.push(f, i * 40L)?.let { out = it }
        return out
    }

    private fun rest(n: Int) = List(n) { frame(null) }

    @Test fun stillHandsInViewAreNotASign() {
        val stream = rest(5) + List(100) { frame(0.5f to 0.7f, noise = 0.004f) } + rest(20)
        assertNull(run(stream))
    }

    @Test fun armMovementIsASign() {
        val stream = rest(5) + List(30) { i -> frame(0.4f + 0.008f * i to 0.7f - 0.006f * i) } + rest(20)
        assertNotNull(run(stream))
    }

    @Test fun fingersMovingInPlaceIsASign() {  // e.g. COLOR, MOTHER: wrist stays put
        val stream = rest(5) + List(30) { i -> frame(0.5f to 0.6f, finger = if ((i / 4) % 2 == 0) 0f else 0.12f) } + rest(20)
        assertNotNull(run(stream))
    }
}
