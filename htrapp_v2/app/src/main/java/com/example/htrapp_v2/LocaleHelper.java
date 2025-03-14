package com.example.htrapp_v2;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.res.Configuration;
import android.util.DisplayMetrics;

import java.util.Locale;

public class LocaleHelper {

    private static final String LANGUAGE_KEY = "AppLanguage";

    // Метод для установки языка
    public static void setLocale(Context context, String language) {
        setAppLanguage(context, language);

        Locale locale = new Locale(language);
        Locale.setDefault(locale);

        Configuration config = new Configuration();
        config.locale = locale;
        context.getResources().updateConfiguration(config, context.getResources().getDisplayMetrics());
    }

    // Метод для получения выбранного языка
    public static String getAppLanguage(Context context) {
        SharedPreferences preferences = context.getSharedPreferences("app_prefs", Context.MODE_PRIVATE);
        return preferences.getString(LANGUAGE_KEY, "en");
    }

    // Метод для сохранения языка
    public static void setAppLanguage(Context context, String language) {
        SharedPreferences preferences = context.getSharedPreferences("app_prefs", Context.MODE_PRIVATE);
        SharedPreferences.Editor editor = preferences.edit();
        editor.putString(LANGUAGE_KEY, language);
        editor.apply();
    }
}
