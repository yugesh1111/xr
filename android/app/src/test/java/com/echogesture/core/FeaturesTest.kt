package com.echogesture.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * The Kotlin feature pipeline must produce the same model input as Python.
 * Golden cases come from scripts/export_golden.py.
 */
class FeaturesTest {
    private fun cases(): List<File> {
        val dir = File(javaClass.classLoader!!.getResource("golden")!!.toURI())
        return dir.listFiles { f -> f.name.endsWith(".bin") }!!.sortedBy { it.name }
    }

    @Test
    fun matchesPython() {
        val files = cases()
        assertTrue("no golden cases", files.isNotEmpty())
        for (f in files) {
            val b = ByteBuffer.wrap(f.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
            val t = b.int
            val xyz = FloatArray(t * Layout.N_POINTS * 3) { b.float }
            val l = b.int; val d = b.int
            val expected = FloatArray(l * d) { b.float }
            assertEquals(Features.DIM, d)
            val got = Features.modelInput(Clip.fromFloats(t, xyz), l)
            var worst = 0.0; var at = -1
            for (i in expected.indices) {
                val e = Math.abs(got[i] - expected[i]) / (1.0 + Math.abs(expected[i]))
                if (e > worst) { worst = e; at = i }
            }
            println("${f.name}: frames=$t worst rel. error=${"%.2e".format(worst)}" +
                if (at >= 0) " (row ${at / d}, feature ${at % d})" else "")
            assertTrue("${f.name}: max error $worst at row ${at / d} feature ${at % d}", worst < 1e-3)
        }
    }
}
