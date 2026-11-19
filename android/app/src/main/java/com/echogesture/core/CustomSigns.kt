package com.echogesture.core

import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Signs the user teaches the phone: a word plus a few recordings of it.
 *
 * Each recording is kept as the word model's embedding of the sign (the 192
 * numbers just before its last layer, unit length). A new sign is compared with
 * every stored recording by cosine similarity; the nearest one wins when it is
 * close enough and not beaten (by more than the margin) by the nearest
 * recording of another taught word or by the [background] bank: embeddings of
 * ordinary signs of the built-in words, shipped with the app (embed_bg.bin).
 * The bank is what keeps everyday signing from being taken for a taught word.
 * Nothing is retrained, so teaching takes a few seconds and stays on the phone.
 * Thresholds: scripts/eval_teach.py (Kaggle ASL signs outside the vocabulary,
 * 10 words taught with 3 recordings each, tested on the same signer's other
 * clips; at most 5% of untaught signs may be taken for a taught word).
 */
class CustomSigns(var minSimilarity: Float = MIN_SIMILARITY, var minMargin: Float = MIN_MARGIN) {
    data class Match(val word: String, val similarity: Float, val margin: Float)

    private val samples = ArrayList<Pair<String, FloatArray>>()
    /** Unit-length embeddings of ordinary (not taught) signs. */
    var background: Array<FloatArray> = emptyArray()

    val words: List<String> @Synchronized get() = samples.map { it.first }.distinct()
    @Synchronized fun count(word: String) = samples.count { it.first == word }
    val isEmpty get() = samples.isEmpty()

    @Synchronized fun add(word: String, embedding: FloatArray) { samples += word to unit(embedding) }
    @Synchronized fun remove(word: String) { samples.removeAll { it.first == word } }
    @Synchronized fun clear() = samples.clear()

    @Synchronized fun match(embedding: FloatArray): Match? {
        if (samples.isEmpty()) return null
        val e = unit(embedding)
        val best = HashMap<String, Float>()
        for ((w, s) in samples) {
            val sim = dot(e, s)
            if (sim > (best[w] ?: -2f)) best[w] = sim
        }
        val ranked = best.entries.sortedByDescending { it.value }
        val top = ranked[0]
        var rival = ranked.getOrNull(1)?.value ?: -1f
        for (b in background) rival = maxOf(rival, dot(e, b))
        val margin = top.value - rival
        return if (top.value >= minSimilarity && margin >= minMargin) Match(top.key, top.value, margin) else null
    }

    @Synchronized fun toJson(): String = JSONArray().apply {
        for ((w, s) in samples) put(JSONObject().put("word", w).put("e", JSONArray().apply { s.forEach { put(it.toDouble()) } }))
    }.toString()

    @Synchronized fun loadJson(text: String) {
        samples.clear()
        val a = JSONArray(text)
        for (i in 0 until a.length()) {
            val o = a.getJSONObject(i); val e = o.getJSONArray("e")
            samples += o.getString("word") to FloatArray(e.length()) { e.getDouble(it).toFloat() }
        }
    }

    companion object {
        const val MIN_SIMILARITY = 0.65f
        const val MIN_MARGIN = -0.05f
        const val SHOTS = 3

        fun unit(v: FloatArray): FloatArray {
            var n = 0f
            for (x in v) n += x * x
            n = sqrt(n).coerceAtLeast(1e-6f)
            return FloatArray(v.size) { v[it] / n }
        }

        /** embed_bg.bin: int32 count, int32 dim, then count x dim float16 (little endian). */
        fun readBank(bytes: ByteArray): Array<FloatArray> {
            val b = java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN)
            val n = b.int; val d = b.int
            return Array(n) { FloatArray(d) { halfToFloat(b.short) } }
        }

        fun halfToFloat(h: Short): Float {
            val x = h.toInt() and 0xffff
            val sign = (x ushr 15) and 1; val exp = (x ushr 10) and 0x1f; val mant = x and 0x3ff
            val v = when (exp) {
                0 -> mant / 1024f * 2f.pow(-14)
                31 -> if (mant == 0) Float.POSITIVE_INFINITY else Float.NaN
                else -> (1 + mant / 1024f) * 2f.pow(exp - 15)
            }
            return if (sign == 1) -v else v
        }

        fun dot(a: FloatArray, b: FloatArray): Float {
            var s = 0f
            for (i in a.indices) s += a[i] * b[i]
            return s
        }
    }
}
