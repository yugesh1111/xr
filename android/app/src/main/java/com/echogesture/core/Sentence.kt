package com.echogesture.core

/**
 * Builds a running sentence from recognised words, the way live captioning
 * tools do for continuous signing: each sign is cut out at a pause and
 * classified, the word joins the open sentence, and a longer pause (no new word
 * for [pauseMs]) closes it with a full stop. The last [keep] sentences stay on
 * screen.
 *
 * ASL glosses carry no English grammar, so the text is the signs in order:
 * first word capitalised, the rest lower case. A word repeated within
 * [repeatMs] is taken as one sign seen twice and kept once.
 */
class Sentence(
    private val pauseMs: Long = 4000,
    private val repeatMs: Long = 1500,
    private val keep: Int = 2,
    private val maxWords: Int = 12,
    /** Called with each sentence as it closes (pause, length cap or [closeNow]). */
    private val onClose: (String) -> Unit = {},
) {
    private val closed = ArrayDeque<String>()
    private val open = ArrayList<String>()
    private var lastWordMs = 0L

    /** Add a recognised word at time [tMs]. Returns false if it was a repeat. */
    fun add(word: String, tMs: Long): Boolean {
        tick(tMs)
        val w = word.lowercase()
        if (open.lastOrNull() == w && tMs - lastWordMs < repeatMs) { lastWordMs = tMs; return false }
        open += w
        lastWordMs = tMs
        if (open.size >= maxWords) close()
        return true
    }

    /** Close the open sentence after a long pause. Returns the sentence if it just closed. */
    fun tick(tMs: Long): String? =
        if (open.isNotEmpty() && tMs - lastWordMs >= pauseMs) close() else null

    fun clear() { closed.clear(); open.clear() }

    /** Close the open sentence now (e.g. the other person starts replying). */
    fun closeNow(): String? = if (open.isNotEmpty()) close() else null

    /** Finished sentences (oldest first), then the one still forming, or null. */
    fun finished(): List<String> = closed.toList()
    fun forming(): String? = if (open.isEmpty()) null else format(open, false)
    val isEmpty get() = closed.isEmpty() && open.isEmpty()

    private fun close(): String {
        val s = format(open, true)
        closed.addLast(s); open.clear()
        while (closed.size > keep) closed.removeFirst()
        onClose(s)
        return s
    }

    private fun format(words: List<String>, end: Boolean): String {
        val s = words.joinToString(" ")
        return s.replaceFirstChar { it.uppercase() } + if (end) "." else ""
    }
}
