package com.echogesture.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class CustomSignsTest {
    private fun v(vararg x: Float) = floatArrayOf(*x)

    @Test fun nearestTaughtSignWins() {
        val c = CustomSigns(minSimilarity = 0.8f, minMargin = 0.03f)
        c.add("WATER", v(1f, 0f, 0f)); c.add("WATER", v(0.9f, 0.1f, 0f)); c.add("FOOD", v(0f, 1f, 0f))
        assertEquals("WATER", c.match(v(2f, 0.1f, 0f))?.word)   // length does not matter
        assertEquals("FOOD", c.match(v(0.05f, 1f, 0.05f))?.word)
    }

    @Test fun farOrAmbiguousSignsAreRejected() {
        val c = CustomSigns(minSimilarity = 0.8f, minMargin = 0.03f)
        c.add("WATER", v(1f, 0f, 0f)); c.add("FOOD", v(0f, 1f, 0f))
        assertNull(c.match(v(0f, 0f, 1f)))       // not close to anything
        assertNull(c.match(v(1f, 1f, 0f)))       // halfway: 0.71 to both
        assertNull(CustomSigns().match(v(1f)))   // nothing taught
    }

    @Test fun backgroundBankBlocksOrdinarySigns() {
        val c = CustomSigns(minSimilarity = 0.6f, minMargin = 0.0f)
        c.add("WATER", v(1f, 0.2f, 0f))
        assertEquals("WATER", c.match(v(1f, 0f, 0f))?.word)
        c.background = arrayOf(CustomSigns.unit(v(1f, 0f, 0f)))   // an ordinary sign right there
        assertNull(c.match(v(1f, 0f, 0f)))
    }

    @Test fun bankFileIsReadAsHalfFloats() {
        // 1 x 2: [1.0, -0.5] as float16 = 0x3C00, 0xB800
        val bytes = byteArrayOf(1, 0, 0, 0, 2, 0, 0, 0, 0x00, 0x3C, 0x00, 0xB8.toByte())
        val bank = CustomSigns.readBank(bytes)
        assertEquals(1f, bank[0][0], 0f); assertEquals(-0.5f, bank[0][1], 0f)
    }

    @Test fun jsonRoundTripAndForget() {
        val c = CustomSigns()
        c.add("WATER", v(3f, 4f)); c.add("FOOD", v(0f, 1f))
        val d = CustomSigns().apply { loadJson(c.toJson()) }
        assertEquals(listOf("WATER", "FOOD"), d.words)
        assertEquals(1f, d.match(v(3f, 4f))!!.similarity, 1e-5f)
        d.remove("WATER")
        assertEquals(listOf("FOOD"), d.words)
    }
}
