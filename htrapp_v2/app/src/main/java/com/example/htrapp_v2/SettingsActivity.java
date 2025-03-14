package com.example.htrapp_v2;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

public class SettingsActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Установка языка перед загрузкой вида
        String language = LocaleHelper.getAppLanguage(this);
        LocaleHelper.setLocale(this, language);

        setContentView(R.layout.activity_settings);

        ImageButton backButton = findViewById(R.id.back_bt);
        Button ruButton = findViewById(R.id.ru_bt);
        Button engButton = findViewById(R.id.eng_bt);

        backButton.setOnClickListener(v -> {
            Intent intent = new Intent(SettingsActivity.this, MainActivity.class);
            startActivity(intent);
            finish();
        });

        ruButton.setOnClickListener(v -> {
            LocaleHelper.setLocale(this, "ru"); // Устанавливаем язык как русский
            recreate(); // Перезагружаем активность для применения изменений

            Toast.makeText(this, "Язык: Русский", Toast.LENGTH_SHORT).show();
        });

        engButton.setOnClickListener(v -> {
            LocaleHelper.setLocale(this, "en"); // Устанавливаем язык как английский
            recreate(); // Перезагружаем активность для применения изменений

            Toast.makeText(this, "Language: English", Toast.LENGTH_SHORT).show();
        });
    }
}
