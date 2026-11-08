package com.echogesture.app

import android.content.Context
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import java.io.File

/**
 * Latency, power and thermal log for results.csv.
 * Power = battery current x voltage, only meaningful when the phone is
 * unplugged. Thermal drift = change in landmark latency between the first and
 * last minute of a run (throttling shows up as slower frames).
 */
class Stats(private val context: Context) {
    private val landmarkMs = ArrayList<Double>()
    private val e2eMs = ArrayList<Double>()
    private val modelMs = ArrayList<Double>()
    private val samples = ArrayList<String>()
    private val powerMw = ArrayList<Double>()
    private val start = System.currentTimeMillis()
    private var lastSample = 0L
    private val bm = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
    private val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager

    fun frame(landmark: Double) { synchronized(this) { landmarkMs.add(landmark) }; maybeSample() }
    fun inference(features: Double, model: Double) {
        synchronized(this) { modelMs.add(model); e2eMs.add(features + model) }
    }

    private fun maybeSample() {
        val now = System.currentTimeMillis()
        if (now - lastSample < 1000) return
        lastSample = now
        val currentUa = bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
        val voltageMv = context.registerReceiver(null, android.content.IntentFilter(android.content.Intent.ACTION_BATTERY_CHANGED))
            ?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, 0) ?: 0
        // devices disagree on the sign and on uA vs mA; treat values < 10000 as mA
        val ua = Math.abs(currentUa).let { if (it in 1..9_999) it * 1000 else it }
        val mw = ua / 1000.0 * voltageMv / 1000.0
        val thermal = if (Build.VERSION.SDK_INT >= 29) pm.currentThermalStatus else -1
        val headroom = if (Build.VERSION.SDK_INT >= 30) pm.getThermalHeadroom(0) else Float.NaN
        synchronized(this) {
            powerMw.add(mw)
            val recent = landmarkMs.takeLast(30)
            samples.add("${(now - start) / 1000},${"%.1f".format(mw)},$thermal,$headroom,${"%.2f".format(recent.average())}")
        }
    }

    fun summary(): String = synchronized(this) {
        fun p(v: List<Double>, q: Double) = if (v.isEmpty()) Double.NaN else v.sorted()[((v.size - 1) * q).toInt()]
        val n = landmarkMs.size
        val first = landmarkMs.take(minOf(n, 900)); val last = landmarkMs.takeLast(minOf(n, 900))
        val drift = if (n > 1800) 100 * (last.average() - first.average()) / first.average() else Double.NaN
        "frames $n | landmarks p50 ${"%.1f".format(p(landmarkMs, .5))} p95 ${"%.1f".format(p(landmarkMs, .95))} ms\n" +
            "model p95 ${"%.2f".format(p(modelMs, .95))} ms | features+model p95 ${"%.2f".format(p(e2eMs, .95))} ms\n" +
            "avg power ${"%.0f".format(powerMw.average())} mW | thermal drift ${"%.1f".format(drift)} %"
    }

    /** Writes the per-second log and a results.csv-style summary; returns the folder. */
    fun save(device: String): File {
        val dir = File(context.getExternalFilesDir(null), "bench_${start}")
        dir.mkdirs()
        synchronized(this) {
            File(dir, "per_second.csv").writeText("t_s,power_mw,thermal_status,thermal_headroom,landmark_ms_avg\n" + samples.joinToString("\n"))
            fun p(v: List<Double>, q: Double) = if (v.isEmpty()) Double.NaN else v.sorted()[((v.size - 1) * q).toInt()]
            val n = landmarkMs.size
            val first = landmarkMs.take(minOf(n, 900)); val last = landmarkMs.takeLast(minOf(n, 900))
            val drift = if (n > 1800) 100 * (last.average() - first.average()) / first.average() else Double.NaN
            File(dir, "results_device.csv").writeText(
                "metric,measured,unit,device,runtime\n" +
                    "latency_p95_model,${"%.2f".format(p(modelMs, .95))},ms,$device,ONNX Runtime INT8\n" +
                    "latency_p95_features_model,${"%.2f".format(p(e2eMs, .95))},ms,$device,ONNX Runtime INT8\n" +
                    "latency_p95_landmarks_per_frame,${"%.2f".format(p(landmarkMs, .95))},ms,$device,MediaPipe Holistic\n" +
                    "avg_power,${"%.0f".format(powerMw.average())},mW,$device,whole phone while running\n" +
                    "thermal_drift_10min,${"%.1f".format(drift)},%,$device,landmark latency first vs last minute\n"
            )
        }
        return dir
    }
}
