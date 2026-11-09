package com.echogesture.core

import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Adaptive sharpening before MediaPipe - same rule as echogesture/enhance.py:
 * sharpen a frame only when it is blurry (variance of the Laplacian of the
 * grey image, resized to 256 px high, below 35) and not dark (mean grey >= 50).
 * Sharpening = unsharp mask, out = 3 * rgb - 2 * gaussian(rgb, sigma 2).
 *
 * Works on ARGB pixel arrays (Bitmap.getPixels) so it runs in plain JVM unit tests.
 */
object Enhance {
    const val SHARPNESS_MAX = 35.0
    const val MIN_LIGHT = 50.0
    const val AMOUNT = 2.0
    const val SIGMA = 2.0

    /** OpenCV COLOR_RGB2GRAY (fixed point, rounded). */
    fun grey(argb: IntArray): IntArray = IntArray(argb.size) {
        val p = argb[it]
        ((p shr 16 and 255) * 4899 + (p shr 8 and 255) * 9617 + (p and 255) * 1868 + 8192) shr 14
    }

    /** Area-average resize of a grey image to height 256 (OpenCV INTER_AREA), rounded to 8 bit. */
    fun resize256(g: IntArray, w: Int, h: Int): Triple<DoubleArray, Int, Int> {
        val nh = 256
        val nw = max((w * 256.0 / h).toInt(), 1)
        val area = w >= nw && h >= nh // OpenCV: true area averaging only when shrinking both axes
        val wx = weights(w, nw, area)
        val wy = weights(h, nh, area)
        val tmp = DoubleArray(h * nw)
        for (y in 0 until h) for ((x, ws) in wx.withIndex()) {
            var s = 0.0
            for ((i, a) in ws) s += g[y * w + i] * a
            tmp[y * nw + x] = s
        }
        val out = DoubleArray(nw * nh)
        for ((y, ws) in wy.withIndex()) for (x in 0 until nw) {
            var s = 0.0
            for ((i, a) in ws) s += tmp[i * nw + x] * a
            out[y * nw + x] = s.roundToInt().coerceIn(0, 255).toDouble()
        }
        return Triple(out, nw, nh)
    }

    /** For each output index, the source indices and their weights (sum 1), as OpenCV INTER_AREA. */
    private val cache = HashMap<Triple<Int, Int, Boolean>, List<List<Pair<Int, Double>>>>()

    private fun weights(src: Int, dst: Int, area: Boolean): List<List<Pair<Int, Double>>> =
        synchronized(cache) { cache.getOrPut(Triple(src, dst, area)) { computeWeights(src, dst, area) } }

    private fun computeWeights(src: Int, dst: Int, area: Boolean): List<List<Pair<Int, Double>>> {
        val scale = src.toDouble() / dst
        return List(dst) { d ->
            if (!area) { // OpenCV's INTER_AREA when enlarging: linear, with its own coefficients
                val sx = Math.floor(d * scale).toInt()
                var fx = (d + 1) - (sx + 1) / scale
                fx = if (fx <= 0) 0.0 else fx - Math.floor(fx)
                listOf(min(sx, src - 1) to 1 - fx, min(sx + 1, src - 1) to fx)
            } else {
                val a = d * scale; val b = a + scale
                val list = ArrayList<Pair<Int, Double>>()
                var i = a.toInt()
                while (i < b && i < src) {
                    val ov = min(b, i + 1.0) - max(a, i.toDouble())
                    if (ov > 1e-9) list.add(i to ov / scale)
                    i++
                }
                list
            }
        }
    }

    private fun reflect(i: Int, n: Int): Int = when {
        n == 1 -> 0
        i < 0 -> -i
        i >= n -> 2 * n - 2 - i
        else -> i
    }

    /** Variance of the 3x3 Laplacian (OpenCV ksize=1, BORDER_REFLECT_101). */
    fun sharpness(g: IntArray, w: Int, h: Int): Double {
        val (s, nw, nh) = resize256(g, w, h)
        var sum = 0.0; var sum2 = 0.0
        for (y in 0 until nh) for (x in 0 until nw) {
            val c = s[y * nw + x]
            val l = s[reflect(y - 1, nh) * nw + x] + s[reflect(y + 1, nh) * nw + x] +
                s[y * nw + reflect(x - 1, nw)] + s[y * nw + reflect(x + 1, nw)] - 4 * c
            sum += l; sum2 += l * l
        }
        val n = (nw * nh).toDouble()
        val mean = sum / n
        return sum2 / n - mean * mean
    }

    fun needsSharpening(argb: IntArray, w: Int, h: Int): Boolean {
        val g = grey(argb)
        if (g.average() < MIN_LIGHT) return false
        return sharpness(g, w, h) < SHARPNESS_MAX
    }

    private val kernel: DoubleArray by lazy {
        val k = 13 // OpenCV: round(sigma * 3 * 2 + 1) | 1 for 8-bit images
        val r = k / 2
        val v = DoubleArray(k) { exp(-((it - r) * (it - r)) / (2 * SIGMA * SIGMA)) }
        val s = v.sum()
        DoubleArray(k) { v[it] / s }
    }

    /** Unsharp mask, in place on an ARGB array. */
    fun unsharp(argb: IntArray, w: Int, h: Int) {
        val r = kernel.size / 2
        val ch = Array(3) { c -> DoubleArray(w * h) { (argb[it] shr (16 - 8 * c) and 255).toDouble() } }
        for (c in 0 until 3) {
            val src = ch[c]
            val tmp = DoubleArray(w * h)
            for (y in 0 until h) for (x in 0 until w) {
                var s = 0.0
                for (k in kernel.indices) s += src[y * w + reflect(x + k - r, w)] * kernel[k]
                tmp[y * w + x] = s // no rounding between the two passes, as OpenCV
            }
            val soft = DoubleArray(w * h)
            for (y in 0 until h) for (x in 0 until w) {
                var s = 0.0
                for (k in kernel.indices) s += tmp[reflect(y + k - r, h) * w + x] * kernel[k]
                soft[y * w + x] = s.roundToInt().toDouble()
            }
            for (i in src.indices) src[i] = ((1 + AMOUNT) * src[i] - AMOUNT * soft[i]).roundToInt().coerceIn(0, 255).toDouble()
        }
        for (i in argb.indices) {
            argb[i] = (argb[i] and 0xFF000000.toInt()) or
                (ch[0][i].toInt() shl 16) or (ch[1][i].toInt() shl 8) or ch[2][i].toInt()
        }
    }

    /** Applies the rule; returns true if the frame was sharpened. */
    fun apply(argb: IntArray, w: Int, h: Int): Boolean {
        if (!needsSharpening(argb, w, h)) return false
        unsharp(argb, w, h)
        return true
    }
}
