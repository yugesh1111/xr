package com.echogesture.core

/**
 * Two-way conversation history: sentences signed to the camera and spoken
 * replies (on-device speech recognition), in the order they happened.
 */
class Conversation(private val keep: Int = 4) {
    enum class Who { SIGNER, SPEAKER }
    data class Turn(val who: Who, val text: String)

    private val turns = ArrayDeque<Turn>()

    fun add(who: Who, text: String) {
        val t = text.trim()
        if (t.isEmpty()) return
        turns.addLast(Turn(who, t))
        while (turns.size > keep) turns.removeFirst()
    }

    fun clear() = turns.clear()
    fun turns(): List<Turn> = turns.toList()
}
