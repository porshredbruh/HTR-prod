package com.example.htrapp_v2;

import android.content.Intent;
import android.os.Bundle;
import android.view.ViewGroup;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;

import androidx.appcompat.app.AppCompatActivity;
import androidx.viewpager2.widget.ViewPager2;

public class GuideActivity extends AppCompatActivity {
    private ViewPager2 viewPager;
    private SliderAdapter sliderAdapter;
    private LinearLayout dotsIndicator;

    private final int[] images = {
            R.drawable.camera,
            R.drawable.time,
            R.drawable.storage
    };

    // Инициализация заголовков в методе onCreate
    private String[] titles;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_guide);

        // Получите строки из ресурсов
        titles = new String[]{
                getString(R.string.guide1),
                getString(R.string.guide2),
                getString(R.string.guide3)
        };

        ImageButton backButton = findViewById(R.id.back_bt);

        backButton.setOnClickListener(v -> {
            Intent intent = new Intent(GuideActivity.this, MainActivity.class);
            startActivity(intent);
            this.finish();
        });

        viewPager = findViewById(R.id.viewPager);
        dotsIndicator = findViewById(R.id.dotsIndicator);
        sliderAdapter = new SliderAdapter(this, images, titles); // Передаем контекст, изображения и тексты
        viewPager.setAdapter(sliderAdapter);

        setupDotsIndicator();
        viewPager.registerOnPageChangeCallback(new ViewPager2.OnPageChangeCallback() {
            @Override
            public void onPageSelected(int position) {
                updateDotsIndicator(position);
                super.onPageSelected(position);
            }
        });
    }

    public void setupDotsIndicator() {
        int dotSize = 50; // Укажите размер индикаторов (например, 24dp)

        for (int i = 0; i < images.length; i++) {
            ImageView dot = new ImageView(this);
            dot.setImageResource(R.drawable.dot_inactive);

            // Настройка параметров для индикатора
            LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                    dotSize, // Ширина
                    dotSize  // Высота
            );

            // Установите отступы (включая отступ снизу)
            params.setMargins(8, 10, 8, 15); // Отступы: слева, сверху, справа, снизу
            dotsIndicator.addView(dot, params);
        }
        updateDotsIndicator(0);
    }



    public void updateDotsIndicator(int position) {
        for (int i = 0; i < dotsIndicator.getChildCount(); i++) {
            ImageView dot = (ImageView) dotsIndicator.getChildAt(i);
            dot.setImageResource(i == position ? R.drawable.dot_active : R.drawable.dot_inactive);
        }
    }
}
