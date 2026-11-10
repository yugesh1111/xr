package com.echogesture.core

/** Same 88-point layout as echogesture/layout.py. */
object Layout {
    const val N_POINTS = 88
    const val POSE = 0
    const val LHAND = 33
    const val RHAND = 54
    const val FACE = 75
    val FACE_IDX = intArrayOf(1, 10, 152, 33, 263, 61, 291, 0, 17, 13, 14, 78, 308)

    const val NOSE = 0
    const val L_SHOULDER = 11
    const val R_SHOULDER = 12
    const val L_ELBOW = 13
    const val R_ELBOW = 14
    const val L_WRIST = 15
    const val R_WRIST = 16

    const val F_NOSE = 0
    const val F_FOREHEAD = 1
    const val F_CHIN = 2
    const val F_EYE_R = 3
    const val F_EYE_L = 4
    const val F_UPPER_LIP = 9
    const val F_LOWER_LIP = 10
    const val F_MOUTH_L = 5
    const val F_MOUTH_R = 6

    const val WRIST = 0
    const val INDEX_MCP = 5
    const val MIDDLE_MCP = 9
    const val PINKY_MCP = 17
    val TIPS = intArrayOf(4, 8, 12, 16, 20)
    val FINGER_JOINTS = arrayOf(
        intArrayOf(1, 2, 3), intArrayOf(2, 3, 4),
        intArrayOf(0, 5, 6), intArrayOf(5, 6, 7), intArrayOf(6, 7, 8),
        intArrayOf(0, 9, 10), intArrayOf(9, 10, 11), intArrayOf(10, 11, 12),
        intArrayOf(0, 13, 14), intArrayOf(13, 14, 15), intArrayOf(14, 15, 16),
        intArrayOf(0, 17, 18), intArrayOf(17, 18, 19), intArrayOf(18, 19, 20),
        intArrayOf(0, 1, 2),
    )
    private val POSE_LR = arrayOf(
        1 to 4, 2 to 5, 3 to 6, 7 to 8, 9 to 10, 11 to 12, 13 to 14, 15 to 16,
        17 to 18, 19 to 20, 21 to 22, 23 to 24, 25 to 26, 27 to 28, 29 to 30, 31 to 32,
    )
    private val FACE_LR = arrayOf(3 to 4, 5 to 6, 11 to 12)

    /** perm[i] = source index of point i after a left/right mirror. */
    val MIRROR_PERM: IntArray = IntArray(N_POINTS) { it }.also { p ->
        for ((a, b) in POSE_LR) { p[a] = b; p[b] = a }
        for (i in 0 until 21) { p[LHAND + i] = RHAND + i; p[RHAND + i] = LHAND + i }
        for ((a, b) in FACE_LR) { p[FACE + a] = FACE + b; p[FACE + b] = FACE + a }
    }
}
