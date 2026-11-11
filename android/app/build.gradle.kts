import java.net.URI

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.echogesture.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.echogesture.app"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
    }
    // one APK per CPU type: arm64-v8a for current Galaxy phones/tablets,
    // armeabi-v7a for older 32-bit devices, x86_64 for the Android emulator
    splits {
        abi {
            isEnable = true
            reset()
            include("arm64-v8a", "armeabi-v7a", "x86_64")
            isUniversalApk = false
        }
    }
    buildTypes {
        release {
            isMinifyEnabled = false
            // debug key so the APK installs without extra setup; use a real key for store builds
            signingConfig = signingConfigs.getByName("debug")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    androidResources { noCompress += listOf("task", "onnx", "mp4") }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.camera:camera-core:1.4.1")
    implementation("androidx.camera:camera-camera2:1.4.1")
    implementation("androidx.camera:camera-lifecycle:1.4.1")
    implementation("androidx.camera:camera-view:1.4.1")
    implementation("com.google.mediapipe:tasks-vision:1.2.2")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
    testImplementation("junit:junit:4.13.2")
}

// Copy the trained model + vocabulary from the repo's models/ folder, and the
// MediaPipe Holistic model (downloaded once, ~14 MB), into the app's assets.
val modelsDir = rootProject.projectDir.resolve("../models")
val assetsDir = projectDir.resolve("src/main/assets")
val prepareAssets by tasks.registering {
    doLast {
        assetsDir.mkdirs()
        for (name in listOf("model_int8.onnx", "model_meta.json")) {
            modelsDir.resolve(name).copyTo(assetsDir.resolve(name), overwrite = true)
        }
        // optional sign gate (scripts/train_gate.py)
        for (name in listOf("gate_int8.onnx", "gate_meta.json")) {
            val f = modelsDir.resolve(name)
            if (f.exists()) f.copyTo(assetsDir.resolve(name), overwrite = true)
        }
        // WLASL test clips for the in-app test mode (only if the data is on this machine;
        // the videos are not stored in git)
        // test_clips.json: single-word test (whole WLASL clips).
        // sentence_clips.json: 3-word sentences; clips cut to the sign by scripts/cut_sentence_clips.py
        for ((listName, srcDir, dstDir) in listOf(
            Triple("test_clips.json", "../data/wlasl/raw", "test_clips"),
            Triple("sentence_clips.json", "../data/wlasl/sentence_clips", "sentence_clips"))) {
            val clipList = modelsDir.resolve(listName)
            if (!clipList.exists()) continue
            clipList.copyTo(assetsDir.resolve(listName), overwrite = true)
            val ids = Regex("\"video_id\": \"(\\w+)\"").findAll(clipList.readText()).map { it.groupValues[1] }.toSet()
            val clipDir = assetsDir.resolve(dstDir).apply { mkdirs() }
            clipDir.listFiles()?.filter { it.nameWithoutExtension !in ids }?.forEach { it.delete() }
            for (id in ids) {
                val src = rootProject.projectDir.resolve("$srcDir/$id.mp4")
                if (src.exists()) src.copyTo(clipDir.resolve("$id.mp4"), overwrite = true)
            }
        }
        val task = assetsDir.resolve("holistic_landmarker.task")
        val local = modelsDir.resolve("holistic_landmarker.task")
        if (!task.exists()) {
            if (local.exists()) local.copyTo(task)
            else URI("https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task")
                .toURL().openStream().use { input -> task.outputStream().use { input.copyTo(it) } }
        }
    }
}
tasks.named("preBuild") { dependsOn(prepareAssets) }
