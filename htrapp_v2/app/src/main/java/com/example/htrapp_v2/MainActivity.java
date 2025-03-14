
package com.example.htrapp_v2;

import android.Manifest;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.preference.PreferenceManager;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_IMAGE_CAPTURE = 1;
    private static final int REQUEST_IMAGE_PICK = 2;
    private static final int REQUEST_PERMISSIONS = 100;

    private Uri imageUri;

    private static final String TAG = "MainActivity";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Установка языка перед загрузкой вида
        String language = LocaleHelper.getAppLanguage(this); // Получаем язык из SharedPreferences
        LocaleHelper.setLocale(this, language);

        setContentView(R.layout.activity_main);

        // Запрос разрешений
        requestPermissions();

        Button makePhotoButton = findViewById(R.id.tPhoto_bt);
        Button selectGalleryButton = findViewById(R.id.slPhoto_bt);

        makePhotoButton.setOnClickListener(v -> openCamera());
        selectGalleryButton.setOnClickListener(v -> openGallery());

        findViewById(R.id.set_bt).setOnClickListener(v -> {
            Intent intent = new Intent(MainActivity.this, SettingsActivity.class);
            startActivity(intent);

        });

        findViewById(R.id.guide_bt).setOnClickListener(v -> {
            Intent intent = new Intent(MainActivity.this, GuideActivity.class);
            startActivity(intent);

        });
    }

    private OkHttpClient getOkHttpClient() {
        return new OkHttpClient.Builder()
                .connectTimeout(60, java.util.concurrent.TimeUnit.SECONDS)
                .readTimeout(300, java.util.concurrent.TimeUnit.SECONDS)
                .writeTimeout(60, java.util.concurrent.TimeUnit.SECONDS)
                .build();
    }

    private void openCamera() {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(intent, REQUEST_IMAGE_CAPTURE);
    }

    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, REQUEST_IMAGE_PICK);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK && data != null) {
            if (requestCode == REQUEST_IMAGE_CAPTURE) {
                imageUri = data.getData();
            } else if (requestCode == REQUEST_IMAGE_PICK) {
                imageUri = data.getData();
            }
            Log.d(TAG, "Image URI: " + imageUri);
            uploadImage(imageUri);
        }
    }


    private String getRealPathFromURI(Uri uri) {
        String[] projection = {MediaStore.Images.Media.DATA};
        Cursor cursor = getContentResolver().query(uri, projection, null, null, null);
        if (cursor != null) {
            try {
                cursor.moveToFirst();
                int columnIndex = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
                String filePath = cursor.getString(columnIndex);
                return filePath;
            } finally {
                cursor.close();
            }
        }
        return null;
    }

    private void uploadImage(Uri imageUri) {
        String filePath = getRealPathFromURI(imageUri);
        if (filePath == null) {
            Log.e(TAG, "Failed to get file path from URI.");
            Toast.makeText(this, "Не удалось получить путь к файлу изображения", Toast.LENGTH_SHORT).show();
            return;
        }

        File imageFile = new File(filePath);
        if (!imageFile.exists()) {
            Log.e(TAG, "File does not exist: " + filePath);
            Toast.makeText(this, "Файл не существует", Toast.LENGTH_SHORT).show();
            return;
        }

        Log.d(TAG, "Uploading file: " + imageFile.getAbsolutePath());

        // Создаем RequestBody для изображения
        RequestBody requestBody = RequestBody.create(MediaType.parse("image/*"), imageFile);
        MultipartBody.Part body = MultipartBody.Part.createFormData("file", imageFile.getName(), requestBody);

        // URL сервера FastAPI
        String serverUrl = ".../upload_image/"; //YOUR URL

        // Используем единый OkHttpClient с увеличенными таймаутами
        OkHttpClient client = getOkHttpClient();
        RequestBody multipartBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addPart(body)
                .build();

        Request request = new Request.Builder()
                .url(serverUrl)
                .post(multipartBody)
                .build();

        // Отправляем запрос на сервер
        client.newCall(request).enqueue(new okhttp3.Callback() {
            @Override
            public void onFailure(@NonNull okhttp3.Call call, @NonNull IOException e) {
                Log.e(TAG, "Upload failed: " + e.getMessage());
                runOnUiThread(() -> Toast.makeText(MainActivity.this, "Не удалось загрузить изображение: " + e.getMessage(), Toast.LENGTH_LONG).show());
            }

            @Override
            public void onResponse(@NonNull okhttp3.Call call, @NonNull Response response) throws IOException {
                if (response.isSuccessful()) {
                    String responseBody = response.body().string();
                    Log.d(TAG, "Image uploaded successfully: " + responseBody);
                    runOnUiThread(() -> Toast.makeText(MainActivity.this, "Изображение успешно загружено", Toast.LENGTH_SHORT).show());

                    // Парсим ответ сервера
                    try {
                        JSONObject jsonResponse = new JSONObject(responseBody);
                        String ocrResultPath = jsonResponse.getString("ocr_result");

                        // Извлекаем имена файлов
                        String ocrFileName = new File(ocrResultPath).getName();

                        // Формируем URL для скачивания файлов
                        String downloadUrlBase = ".../download_file/"; //YOUR URL
                        String ocrDownloadUrl = downloadUrlBase + ocrFileName;

                        // Скачиваем файлы
                        downloadFile(ocrDownloadUrl, ocrFileName);


                    } catch (JSONException e) {
                        Log.e(TAG, "Failed to parse JSON response: " + e.getMessage());
                        runOnUiThread(() -> Toast.makeText(MainActivity.this, "Ошибка обработки ответа сервера", Toast.LENGTH_LONG).show());
                    }

                } else {
                    Log.e(TAG, "Failed to upload image: " + response.code() + " " + response.message());
                    runOnUiThread(() -> Toast.makeText(MainActivity.this, "Не удалось загрузить изображение: " + response.code() + " " + response.message(), Toast.LENGTH_LONG).show());
                }
            }
        });
    }

    private void downloadFile(String fileUrl, String fileName) {
        // Используем единый OkHttpClient с увеличенными таймаутами
        OkHttpClient client = getOkHttpClient();

        // Создаем запрос
        Request request = new Request.Builder()
                .url(fileUrl)
                .build();

        // Выполняем запрос асинхронно
        client.newCall(request).enqueue(new okhttp3.Callback() {
            @Override
            public void onFailure(@NonNull okhttp3.Call call, @NonNull IOException e) {
                Log.e(TAG, "Download failed for " + fileName + ": " + e.getMessage());
                runOnUiThread(() -> Toast.makeText(MainActivity.this, "Не удалось скачать файл: " + fileName, Toast.LENGTH_LONG).show());
            }

            @Override
            public void onResponse(@NonNull okhttp3.Call call, @NonNull Response response) throws IOException {
                if (!response.isSuccessful()) {
                    Log.e(TAG, "Failed to download file: " + fileName + " Response code: " + response.code());
                    runOnUiThread(() -> Toast.makeText(MainActivity.this, "Не удалось скачать файл: " + fileName, Toast.LENGTH_LONG).show());
                    return;
                }

                // Получаем InputStream из ответа
                InputStream inputStream = response.body().byteStream();

                // Путь к папке загрузок
                File downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);
                if (!downloadsDir.exists()) {
                    boolean dirCreated = downloadsDir.mkdirs();
                    if (!dirCreated) {
                        Log.e(TAG, "Failed to create Downloads directory");
                        runOnUiThread(() -> Toast.makeText(MainActivity.this, "Не удалось создать папку загрузок", Toast.LENGTH_LONG).show());
                        return;
                    }
                }

                // Создаем файл
                File file = new File(downloadsDir, fileName);
                OutputStream outputStream = new FileOutputStream(file);

                // Буфер для чтения данных
                byte[] buffer = new byte[4096];
                int bytesRead;

                // Читаем данные и записываем в файл
                while ((bytesRead = inputStream.read(buffer)) != -1) {
                    outputStream.write(buffer, 0, bytesRead);
                }

                // Закрываем потоки
                outputStream.close();
                inputStream.close();

                Log.d(TAG, "File downloaded: " + file.getAbsolutePath());
                runOnUiThread(() -> Toast.makeText(MainActivity.this, "Файл скачан: " + fileName, Toast.LENGTH_SHORT).show());
            }
        });
    }


    // Запрос разрешений на использование камеры и чтение/запись файлов
    private void requestPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED ||
                ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                        != PackageManager.PERMISSION_GRANTED ||
                ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                        != PackageManager.PERMISSION_GRANTED) {

            ActivityCompat.requestPermissions(this, new String[]{
                    Manifest.permission.READ_EXTERNAL_STORAGE,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE,
                    Manifest.permission.CAMERA
            }, REQUEST_PERMISSIONS);
        }
    }

    // Обработка результатов запроса разрешений
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_PERMISSIONS) {
            boolean allGranted = true;
            for (int result : grantResults) {
                if (result != PackageManager.PERMISSION_GRANTED) {
                    allGranted = false;
                    break;
                }
            }
        }
    }
}
