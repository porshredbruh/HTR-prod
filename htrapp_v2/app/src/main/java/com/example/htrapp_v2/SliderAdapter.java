package com.example.htrapp_v2;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

public class SliderAdapter extends RecyclerView.Adapter<SliderAdapter.SliderViewHolder> {
    private final int[] images; // Массив изображений
    private final String[] titles; // Массив заголовков
    private final Context context; // Контекст для работы с ресурсами

    public SliderAdapter(Context context, int[] images, String[] titles) {
        this.context = context;
        this.images = images;
        this.titles = titles;
    }

    @NonNull
    @Override
    public SliderViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(context).inflate(R.layout.slide_item, parent, false);
        return new SliderViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull SliderViewHolder holder, int position) {
        // Устанавливаем изображение и текст для текущего слайда
        holder.imageView.setImageResource(images[position]);
        holder.titleTextView.setText(titles[position]);
    }

    @Override
    public int getItemCount() {
        // Возвращаем количество слайдов
        return images.length;
    }

    static class SliderViewHolder extends RecyclerView.ViewHolder {
        ImageView imageView;
        TextView titleTextView;

        SliderViewHolder(@NonNull View itemView) {
            super(itemView);
            // Находим представления в макете
            imageView = itemView.findViewById(R.id.slide_image);
            titleTextView = itemView.findViewById(R.id.slide_title);
        }
    }
}
