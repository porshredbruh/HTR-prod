plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.example.htrapp_v2"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.htrapp_v2"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
}

dependencies {

    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.activity)
    implementation(libs.constraintlayout)
    testImplementation(libs.junit)
    androidTestImplementation(libs.ext.junit)
    androidTestImplementation(libs.espresso.core)
}

dependencies {
    // OkHttp для HTTP-запросов
    implementation("com.squareup.okhttp3:okhttp:4.9.3")

    // Для работы с изображениями (например, Glide)
    implementation("com.github.bumptech.glide:glide:4.15.1")
    implementation("com.github.bumptech.glide:glide:4.15.1")

    // Для работы с разрешениями
    implementation("androidx.core:core-ktx:1.9.0")

    // Для работы с UI-элементами
    implementation("androidx.appcompat:appcompat:1.6.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
}