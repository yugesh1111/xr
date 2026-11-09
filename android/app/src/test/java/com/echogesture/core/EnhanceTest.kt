package com.echogesture.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Adaptive sharpening must decide and sharpen like Python (scripts/export_enhance_golden.py). */
class EnhanceTest {
    @Test
    fun matchesPython() {
        val dir = File(javaClass.classLoader!!.getResource("enhance")!!.toURI())
        val files = dir.listFiles { f -> f.name.endsWith(".bin") }!!.sortedBy { it.name }
        assertTrue("no golden cases", files.isNotEmpty())
        for (f in files) {
            val b = ByteBuffer.wrap(f.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
            val w = b.int; val h = b.int
            val argb = IntArray(w * h) {
                val r = b.get().toInt() and 255; val g = b.get().toInt() and 255; val bl = b.get().toInt() and 255
                (0xFF shl 24) or (r shl 16) or (g shl 8) or bl
            }
            val mean = b.double; val sharp = b.double; val decided = b.int == 1
            val expected = IntArray(w * h * 3) { b.get().toInt() and 255 }

            val grey = Enhance.grey(argb)
            val gotSharp = Enhance.sharpness(grey, w, h)
            val gotDecision = Enhance.apply(argb, w, h)
            var worst = 0; var off = 0
            for (i in 0 until w * h) for (c in 0 until 3) {
                val d = Math.abs((argb[i] shr (16 - 8 * c) and 255) - expected[i * 3 + c])
                worst = maxOf(worst, d); if (d > 1) off++
            }
            println("${f.name}: mean ${"%.2f".format(grey.average())}/${"%.2f".format(mean)} " +
                "sharpness ${"%.2f".format(gotSharp)}/${"%.2f".format(sharp)} sharpened $gotDecision/$decided " +
                "max pixel diff $worst, >1: $off of ${w * h * 3}")
            assertEquals(f.name, mean, grey.average(), 0.01)
            assertEquals(f.name, sharp, gotSharp, 0.02 * sharp + 0.5)
            assertEquals(f.name, decided, gotDecision)
            // OpenCV blurs 8-bit images in fixed point: about 1% of values end up 2 grey levels apart
            assertTrue("${f.name}: pixels differ by up to $worst ($off values > 1)", worst <= 2 && off < w * h * 3 / 50)
        }
    }
}
