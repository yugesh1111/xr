package com.echogesture.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import com.echogesture.core.Clip
import com.echogesture.core.Features
import com.echogesture.core.Layout
import org.json.JSONObject
import java.nio.FloatBuffer
import kotlin.math.exp

/** Segment of frames -> top-k glosses with the INT8 ONNX model. */
class Recognizer(context: Context, nnapi: Boolean) : AutoCloseable {
    /** [isSign]: the sign gate's verdict (true when no gate model is packaged). */
    data class Result(val words: List<String>, val probs: List<Float>, val featureMs: Double, val modelMs: Double,
                      val signProb: Float = 1f, val isSign: Boolean = true)

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    val vocab: List<String>
    private val length: Int
    // sign gate (scripts/train_gate.py): "is this a sign from the vocabulary at all?"
    private var gate: OrtSession? = null
    private var gateThreshold = 0f

    init {
        val meta = JSONObject(context.assets.open("model_meta.json").bufferedReader().readText())
        length = meta.getInt("length")
        val v = meta.getJSONArray("vocab")
        vocab = List(v.length()) { v.getString(it) }
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
            if (nnapi) runCatching { addNnapi() }
        }
        session = env.createSession(context.assets.open("model_int8.onnx").readBytes(), opts)
        runCatching {
            gateThreshold = JSONObject(context.assets.open("gate_meta.json").bufferedReader().readText()).getDouble("threshold").toFloat()
            gate = env.createSession(context.assets.open("gate_int8.onnx").readBytes(), opts)
        }
    }

    fun recognize(frames: List<FloatArray>, k: Int = 3): Result {
        val t0 = System.nanoTime()
        val flat = FloatArray(frames.size * Layout.N_POINTS * 3)
        for ((i, f) in frames.withIndex()) System.arraycopy(f, 0, flat, i * f.size, f.size)
        val input = Features.modelInput(Clip.fromFloats(frames.size, flat), length)
        val t1 = System.nanoTime()
        var signProb = 1f
        val logits = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1, length.toLong(), Features.DIM.toLong())).use { x ->
            gate?.run(mapOf("x" to x))?.use { out ->
                @Suppress("UNCHECKED_CAST")
                val g = (out[0].value as Array<FloatArray>)[0]
                signProb = (1.0 / (1.0 + exp((g[0] - g[1]).toDouble()))).toFloat()
            }
            session.run(mapOf("x" to x)).use { out ->
                @Suppress("UNCHECKED_CAST")
                (out[0].value as Array<FloatArray>)[0]
            }
        }
        val t2 = System.nanoTime()
        val m = logits.max()
        val e = logits.map { exp((it - m).toDouble()) }
        val s = e.sum()
        val top = e.indices.sortedByDescending { e[it] }.take(k)
        return Result(top.map { vocab[it] }, top.map { (e[it] / s).toFloat() }, (t1 - t0) / 1e6, (t2 - t1) / 1e6,
            signProb, signProb >= gateThreshold)
    }

    override fun close() { session.close(); gate?.close() }
}
