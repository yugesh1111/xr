package com.echogesture.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class SentenceTest {
    @Test fun wordsFormASentenceAndAPauseClosesIt() {
        val s = Sentence(pauseMs = 4000)
        s.add("MAN", 0); s.add("HELP", 1500); s.add("NOW", 3000)
        assertEquals("Man help now", s.forming())
        assertNull(s.tick(5000))
        assertEquals("Man help now.", s.tick(7100))
        assertNull(s.forming())
        s.add("DRINK", 9000)
        assertEquals(listOf("Man help now."), s.finished())
        assertEquals("Drink", s.forming())
    }

    @Test fun quickRepeatCountsOnce() {
        val s = Sentence()
        s.add("BED", 0); s.add("BED", 800)
        assertEquals("Bed", s.forming())
        s.add("BED", 3000)  // signed again later: a real repeat
        assertEquals("Bed bed", s.forming())
    }

    @Test fun keepsTheLastTwoSentences() {
        val s = Sentence(pauseMs = 1000, keep = 2)
        for ((i, w) in listOf("A", "B", "C").withIndex()) { s.add(w, i * 2000L); s.tick(i * 2000L + 1500) }
        assertEquals(listOf("B.", "C."), s.finished())
    }
}
