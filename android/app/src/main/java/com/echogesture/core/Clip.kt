package com.echogesture.core

/**
 * Landmarks of one clip: T frames x 88 points x (x, y, z), NaN = not detected.
 * Stored flat in a DoubleArray: index = (t * 88 + n) * 3 + c.
 */
class Clip(val frames: Int, val data: DoubleArray = DoubleArray(frames * Layout.N_POINTS * 3) { Double.NaN }) {
    operator fun get(t: Int, n: Int, c: Int) = data[(t * Layout.N_POINTS + n) * 3 + c]
    operator fun set(t: Int, n: Int, c: Int, v: Double) { data[(t * Layout.N_POINTS + n) * 3 + c] = v }
    fun point(t: Int, n: Int) = Vec(this[t, n, 0], this[t, n, 1], this[t, n, 2])
    fun setPoint(t: Int, n: Int, v: Vec) { this[t, n, 0] = v.x; this[t, n, 1] = v.y; this[t, n, 2] = v.z }
    fun seen(t: Int, n: Int) = !this[t, n, 0].isNaN()
    fun copy() = Clip(frames, data.copyOf())

    companion object {
        fun fromFloats(frames: Int, values: FloatArray) =
            Clip(frames, DoubleArray(values.size) { values[it].toDouble() })
    }
}

data class Vec(val x: Double, val y: Double, val z: Double) {
    operator fun plus(o: Vec) = Vec(x + o.x, y + o.y, z + o.z)
    operator fun minus(o: Vec) = Vec(x - o.x, y - o.y, z - o.z)
    operator fun times(s: Double) = Vec(x * s, y * s, z * s)
    operator fun div(s: Double) = Vec(x / s, y / s, z / s)
    infix fun dot(o: Vec) = x * o.x + y * o.y + z * o.z
    infix fun cross(o: Vec) = Vec(y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x)
    fun norm() = Math.sqrt(x * x + y * y + z * z)
    fun unit() = this / (norm() + 1e-8)
    fun hasNaN() = x.isNaN() || y.isNaN() || z.isNaN()

    companion object {
        val NAN = Vec(Double.NaN, Double.NaN, Double.NaN)
    }
}
