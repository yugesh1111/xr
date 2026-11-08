package com.echogesture.core

import kotlin.math.PI
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin

/**
 * Kotlin port of echogesture/features.py + sampling.py + inputs.canonical_input.
 * Must stay numerically identical to the Python version: FeaturesTest checks
 * it against golden vectors exported by scripts/export_golden.py.
 */
object Features {
    // feature group sizes, in model input order
    const val HANDSHAPE = 170
    const val ORIENTATION = 12
    const val LOCATION = 24
    const val MOTION = 16
    const val NONMANUAL = 11
    const val QUALITY = 6
    const val DIM = HANDSHAPE + ORIENTATION + LOCATION + MOTION + NONMANUAL + QUALITY // 239

    private const val O_ORIENT = HANDSHAPE
    private const val O_LOC = O_ORIENT + ORIENTATION
    private const val O_MOTION = O_LOC + LOCATION
    private const val O_NONMAN = O_MOTION + MOTION
    private const val O_QUAL = O_NONMAN + NONMANUAL

    /** Raw clip -> model input [length * DIM], row-major. */
    fun modelInput(raw: Clip, length: Int = 32, uniformWeight: Double = 0.5): FloatArray {
        require(raw.frames > 0) { "clip has no frames" }
        val f = clipFeatures(raw)
        val idx = sampleIndices(f, length, uniformWeight)
        val out = FloatArray(length * DIM)
        for ((i, t) in idx.withIndex()) {
            for (d in 0 until DIM) out[i * DIM + d] = f[t][d].toFloat()
        }
        return out
    }

    // ------------------------------------------------------------ preprocessing

    fun fixHandSlots(x: Clip): Clip {
        val out = x.copy()
        for (t in 0 until x.frames) {
            val lw = xy(x, t, Layout.L_WRIST); val rw = xy(x, t, Layout.R_WRIST)
            val lh = xy(x, t, Layout.LHAND); val rh = xy(x, t, Layout.RHAND)
            val keep = nansum(dist2(lh, lw), dist2(rh, rw))
            val swap = nansum(dist2(lh, rw), dist2(rh, lw))
            val known = !lw.anyNaN() && !rw.anyNaN() && !(lh.anyNaN() && rh.anyNaN())
            if (known && swap < keep) {
                for (i in 0 until 21) for (c in 0 until 3) {
                    out[t, Layout.LHAND + i, c] = x[t, Layout.RHAND + i, c]
                    out[t, Layout.RHAND + i, c] = x[t, Layout.LHAND + i, c]
                }
            }
        }
        return out
    }

    fun canonicalize(x: Clip, useYaw: Boolean = true, maxYawDeg: Double = 60.0): Clip {
        val p = x.copy()
        for (t in 0 until p.frames) for (n in 0 until Layout.N_POINTS) p[t, n, 1] = -p[t, n, 1]
        val l = nanMedianPoint(p, Layout.L_SHOULDER)
        val r = nanMedianPoint(p, Layout.R_SHOULDER)
        if (l == null || r == null) {
            // no body: centre on everything we have
            val center = DoubleArray(3) { c -> nanMean(p, c) }
            val scale = nanStdXY(p) * 4 + 1e-6
            for (t in 0 until p.frames) for (n in 0 until Layout.N_POINTS) for (c in 0 until 3)
                p[t, n, c] = (p[t, n, c] - center[c]) / scale
            return p
        }
        val center = (l + r) * 0.5
        val across = l - r
        var rot = rotZ(-atan2(across.y, across.x))
        if (useYaw) {
            val a = mul(rot, across)
            val lim = Math.toRadians(maxYawDeg)
            val yaw = atan2(a.z, a.x).coerceIn(-lim, lim)
            rot = matmul(rotY(yaw), rot)
        }
        val scale = Math.sqrt(across.x * across.x + across.y * across.y) + 1e-6
        for (t in 0 until p.frames) for (n in 0 until Layout.N_POINTS) {
            val v = mul(rot, p.point(t, n) - center) / scale
            p.setPoint(t, n, v)
        }
        // Python returns float32 here
        for (i in p.data.indices) p.data[i] = p.data[i].toFloat().toDouble()
        return p
    }

    private fun motionEnergy(x: Clip, block: Int): Double {
        var s = 0.0
        for (t in 1 until x.frames) for (i in 0 until 21) {
            val d = (x.point(t, block + i) - x.point(t - 1, block + i)).norm()
            if (!d.isNaN()) s += d
        }
        var seen = 0
        for (t in 0 until x.frames) if (x.seen(t, block)) seen++
        return s + 0.05 * seen
    }

    /** Mirror left-handed signing so the dominant hand is in the RHAND slot. */
    fun toRightDominant(x: Clip): Pair<Clip, Boolean> {
        if (motionEnergy(x, Layout.LHAND) <= motionEnergy(x, Layout.RHAND)) return x to false
        val m = Clip(x.frames)
        for (t in 0 until x.frames) for (n in 0 until Layout.N_POINTS) {
            val src = Layout.MIRROR_PERM[n]
            m[t, n, 0] = -x[t, src, 0]; m[t, n, 1] = x[t, src, 1]; m[t, n, 2] = x[t, src, 2]
        }
        return m to true
    }

    /** Interpolate gaps of <= maxGap frames per point; returns the interpolated mask [t][n]. */
    fun fillGaps(x: Clip, maxGap: Int = 6): Pair<Clip, Array<BooleanArray>> {
        val out = x.copy()
        val interp = Array(x.frames) { BooleanArray(Layout.N_POINTS) }
        for (n in 0 until Layout.N_POINTS) {
            val idx = (0 until x.frames).filter { x.seen(it, n) }
            for (k in 0 until idx.size - 1) {
                val a = idx[k]; val b = idx[k + 1]; val gap = b - a - 1
                if (gap in 1..maxGap) for (j in 1..gap) {
                    val w = j.toDouble() / (gap + 1)
                    for (c in 0 until 3) out[a + j, n, c] = x[a, n, c] * (1 - w) + x[b, n, c] * w
                    interp[a + j][n] = true
                }
            }
        }
        return out to interp
    }

    // ------------------------------------------------------------ features

    /** Features per frame: [T][DIM], NaN-free. */
    fun clipFeatures(raw: Clip, maxGap: Int = 6): Array<DoubleArray> {
        var x = fixHandSlots(raw)
        x = canonicalize(x)
        val (m, mirrored) = toRightDominant(x)
        val (xyz, interp) = fillGaps(m, maxGap)
        val T = xyz.frames
        val f = Array(T) { DoubleArray(DIM) }

        val nose = Array(T) { t ->
            val fn = xyz.point(t, Layout.FACE + Layout.F_NOSE); val pn = xyz.point(t, Layout.NOSE)
            Vec(if (fn.x.isNaN()) pn.x else fn.x, if (fn.y.isNaN()) pn.y else fn.y, if (fn.z.isNaN()) pn.z else fn.z)
        }
        val mouth = Array(T) { t ->
            (xyz.point(t, Layout.FACE + Layout.F_UPPER_LIP) + xyz.point(t, Layout.FACE + Layout.F_LOWER_LIP)) * 0.5
        }

        // dominant hand first, then non-dominant (mirrored to look like a right hand)
        for ((h, pair) in listOf(Layout.RHAND to false, Layout.LHAND to true).withIndex()) {
            val (block, mirrorX) = pair
            val wrist = Array(T) { xyz.point(it, block) }
            val tipsRel = Array(T) { t ->
                var s = Vec(0.0, 0.0, 0.0)
                for (tip in Layout.TIPS) s += xyz.point(t, block + tip)
                s / 5.0 - wrist[t]
            }
            val v = vel(wrist); val acc = vel(v); val tipV = vel(tipsRel)
            for (t in 0 until T) {
                val row = f[t]
                handFeatures(xyz, t, block, mirrorX, row, h * 85, O_ORIENT + h * 6)
                val w = wrist[t]
                var o = O_LOC + h * 10
                o = put(row, o, w); o = put(row, o, w - nose[t]); o = put(row, o, w - mouth[t])
                row[o] = (w - nose[t]).norm()
                o = O_MOTION + h * 8
                o = put(row, o, v[t]); row[o++] = v[t].norm(); row[o++] = acc[t].norm(); put(row, o, tipV[t])
                var ok = true
                var anyInterp = false
                for (i in 0 until 21) {
                    for (c in 0 until 3) if (xyz[t, block + i, c].isNaN()) ok = false
                    if (interp[t][block + i]) anyInterp = true
                }
                row[O_QUAL + h * 2] = if (ok) 1.0 else 0.0
                row[O_QUAL + h * 2 + 1] = if (anyInterp) 1.0 else 0.0
            }
        }
        for (t in 0 until T) {
            val row = f[t]
            val d = xyz.point(t, Layout.RHAND) - xyz.point(t, Layout.LHAND)
            var o = put(row, O_LOC + 20, d); row[o] = d.norm()
            o = put(row, O_NONMAN, xyz.point(t, Layout.L_ELBOW))
            o = put(row, o, xyz.point(t, Layout.R_ELBOW))
            o = put(row, o, nose[t])
            row[o++] = (xyz.point(t, Layout.FACE + Layout.F_UPPER_LIP) - xyz.point(t, Layout.FACE + Layout.F_LOWER_LIP)).norm()
            row[o] = (xyz.point(t, Layout.FACE + Layout.F_MOUTH_L) - xyz.point(t, Layout.FACE + Layout.F_MOUTH_R)).norm()
            var poseOk = true
            for (n in intArrayOf(Layout.L_SHOULDER, Layout.R_SHOULDER)) for (c in 0 until 3) if (xyz[t, n, c].isNaN()) poseOk = false
            row[O_QUAL + 4] = if (poseOk) 1.0 else 0.0
            row[O_QUAL + 5] = if (mirrored) 1.0 else 0.0
            for (i in row.indices) {
                val y = row[i].toFloat().toDouble() // Python casts groups to float32
                row[i] = if (y.isNaN() || y.isInfinite()) 0.0 else y
            }
        }
        return f
    }

    private fun handFeatures(x: Clip, t: Int, block: Int, mirrorX: Boolean, row: DoubleArray, oShape: Int, oOrient: Int) {
        val p = Array(21) { x.point(t, block + it) }
        val yAx = (p[Layout.MIDDLE_MCP] - p[Layout.WRIST]).unit()
        val across = p[Layout.INDEX_MCP] - p[Layout.PINKY_MCP]
        val zAx = (across cross yAx).unit()
        val xAx = yAx cross zAx
        val size = (p[Layout.MIDDLE_MCP] - p[Layout.WRIST]).norm() + 1e-6
        var o = oShape
        for (i in 1 until 21) {
            val d = p[i] - p[0]
            row[o++] = (xAx dot d) / size * (if (mirrorX) -1.0 else 1.0)
            row[o++] = (yAx dot d) / size
            row[o++] = (zAx dot d) / size
        }
        for (a in 0 until 5) for (b in a + 1 until 5) row[o++] = (p[Layout.TIPS[a]] - p[Layout.TIPS[b]]).norm() / size
        for (j in Layout.FINGER_JOINTS) {
            val u = (p[j[0]] - p[j[1]]).unit(); val v = (p[j[2]] - p[j[1]]).unit()
            val c = u dot v
            row[o++] = acos(if (c.isNaN()) c else c.coerceIn(-1.0, 1.0)) / PI
        }
        put(row, put(row, oOrient, zAx), yAx)
    }

    // ------------------------------------------------------------ sampling

    fun sampleIndices(f: Array<DoubleArray>, length: Int, uniformWeight: Double): IntArray {
        val T = f.size
        var first = -1; var last = -1
        for (t in 0 until T) if (f[t][O_QUAL] > 0 || f[t][O_QUAL + 2] > 0) { if (first < 0) first = t; last = t }
        val a = if (first < 0) 0 else maxOf(first - 2, 0)
        val b = if (first < 0) T - 1 else minOf(last + 2, T - 1)
        val speed = DoubleArray(b - a + 1) { f[a + it][O_MOTION + 3] + f[a + it][O_MOTION + 11] }
        return adaptiveIndices(speed, length, uniformWeight).map { it + a }.toIntArray()
    }

    fun adaptiveIndices(speed: DoubleArray, length: Int, uniformWeight: Double): IntArray {
        val T = speed.size
        val s = DoubleArray(T) { if (speed[it].isNaN()) 0.0 else speed[it] }
        val sum = s.sum()
        for (i in 0 until T) s[i] = if (sum > 0) s[i] / sum else 1.0 / T
        val w = DoubleArray(T) { uniformWeight / T + (1 - uniformWeight) * s[it] }
        val total = w.sum()
        val cdf = DoubleArray(T)
        var acc = 0.0
        for (i in 0 until T) { acc += w[i]; cdf[i] = acc / total }
        return IntArray(length) { k ->
            val q = (k + 0.5) / length
            var i = 0
            while (i < T && cdf[i] < q) i++ // numpy searchsorted, side="left"
            i.coerceIn(0, T - 1)
        }
    }

    // ------------------------------------------------------------ helpers

    private fun put(row: DoubleArray, o: Int, v: Vec): Int { row[o] = v.x; row[o + 1] = v.y; row[o + 2] = v.z; return o + 3 }

    private fun vel(p: Array<Vec>) = Array(p.size) { if (it == 0) Vec(0.0, 0.0, 0.0) else p[it] - p[it - 1] }

    private fun xy(x: Clip, t: Int, n: Int) = doubleArrayOf(x[t, n, 0], x[t, n, 1])
    private fun DoubleArray.anyNaN() = any { it.isNaN() }
    private fun dist2(a: DoubleArray, b: DoubleArray) = Math.hypot(a[0] - b[0], a[1] - b[1])
    private fun nansum(vararg v: Double) = v.filter { !it.isNaN() }.sum()

    private fun nanMedianPoint(p: Clip, n: Int): Vec? {
        val c = DoubleArray(3)
        for (k in 0 until 3) {
            val v = (0 until p.frames).map { p[it, n, k] }.filter { !it.isNaN() }.sorted()
            if (v.isEmpty()) return null
            c[k] = if (v.size % 2 == 1) v[v.size / 2] else (v[v.size / 2 - 1] + v[v.size / 2]) / 2
        }
        return Vec(c[0], c[1], c[2])
    }

    private fun nanMean(p: Clip, c: Int): Double {
        var s = 0.0; var n = 0
        for (t in 0 until p.frames) for (i in 0 until Layout.N_POINTS) { val v = p[t, i, c]; if (!v.isNaN()) { s += v; n++ } }
        return s / n
    }

    private fun nanStdXY(p: Clip): Double {
        val v = ArrayList<Double>()
        for (t in 0 until p.frames) for (i in 0 until Layout.N_POINTS) for (c in 0 until 2) { val a = p[t, i, c]; if (!a.isNaN()) v.add(a) }
        val m = v.average()
        return Math.sqrt(v.sumOf { (it - m) * (it - m) } / v.size)
    }

    private fun rotZ(a: Double) = arrayOf(doubleArrayOf(cos(a), -sin(a), 0.0), doubleArrayOf(sin(a), cos(a), 0.0), doubleArrayOf(0.0, 0.0, 1.0))
    private fun rotY(a: Double) = arrayOf(doubleArrayOf(cos(a), 0.0, sin(a)), doubleArrayOf(0.0, 1.0, 0.0), doubleArrayOf(-sin(a), 0.0, cos(a)))
    private fun mul(m: Array<DoubleArray>, v: Vec) = Vec(
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
    )
    private fun matmul(a: Array<DoubleArray>, b: Array<DoubleArray>) =
        Array(3) { i -> DoubleArray(3) { j -> (0 until 3).sumOf { k -> a[i][k] * b[k][j] } } }
}
