package com.echogesture.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** The Kotlin filter must match echogesture/live.py (scripts/export_filter_golden.py). */
class LandmarkFilterTest {
    @Test
    fun matchesPython() {
        val file = File(javaClass.classLoader!!.getResource("filter/golden.bin")!!.toURI())
        val b = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        val n = b.int
        val t = LongArray(n) { b.long }
        val size = Layout.N_POINTS * 3
        val x = Array(n) { FloatArray(size) { b.float } }
        val y = Array(n) { FloatArray(size) { b.float } }
        val f = LandmarkFilter()
        var worst = 0.0
        for (i in 0 until n) {
            val got = f.filter(x[i], t[i])
            for (k in 0 until size) {
                if (y[i][k].isNaN()) { assertTrue("frame $i value $k should be missing", got[k].isNaN()); continue }
                worst = maxOf(worst, Math.abs((got[k] - y[i][k]).toDouble()))
            }
        }
        println("LandmarkFilter: $n frames, worst difference ${"%.2e".format(worst)}")
        assertTrue("max difference $worst", worst < 1e-4)
    }

    @Test
    fun stillPointIsSmoothedAndMovingPointFollows() {
        val f = LandmarkFilter()
        val size = Layout.N_POINTS * 3
        // still point with +-0.004 jitter: output moves far less than the input
        var outRange = 0f
        var first = Float.NaN
        for (i in 0 until 60) {
            val fr = FloatArray(size) { 0.5f + if (i % 2 == 0) 0.004f else -0.004f }
            val o = f.filter(fr, i * 40L)[0]
            if (i > 10) { if (first.isNaN()) first = o; outRange = maxOf(outRange, Math.abs(o - first)) }
        }
        assertTrue("jitter not reduced: $outRange", outRange < 0.004f)
        // fast move: after 10 frames the output is close to the input
        val g = LandmarkFilter()
        var last = 0f
        for (i in 0 until 10) last = g.filter(FloatArray(size) { 0.1f + 0.05f * i }, i * 40L)[0]
        assertEquals(0.55f, last, 0.06f)
    }
}
