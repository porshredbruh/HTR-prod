# HTR-prod
<p align="center">
      <img src="https://i.ibb.co/fYtd9nHX/htr.png" alt="htr" border="0"></a>
</p>

## О проекте

В наше время люди в различных сферах деятельности сталкиваются с проблемой необходимости эффективного управления объёмными массивами информации, включая рукописные заметки, лекции и другие документы.

Целью проекта стало решении проблемы оцифровки рукописного текста с использованием технологий искусственного интеллекта для автоматизации процесса сегментации и распознавания рукописных записей.

## Что уникально?

- Реализовано распознавание чертежей(или прочих рисунков), рукописного текста и математических выражений
- Реализована оцифровка текста и математических выражений
- Создано простое для использования и интуитивно понятное Android приложение
- Датасет, состоящий более чем из 1000 самостоятельно аннотированных изображений.

## Принцип работы

1. Изображение приходит на сервер от пользователя.
2. Определение объектов(text, math и image) моделью Object detection.
3. Объекты класса text отправляются на распознавание в модель №1, а объекты класса math - в №2.
4. Собирается файл docx с содержанием аналогичным по расположению объектов на исходной фотографии.
5. Файл docx сохраняется в хранилище телефона пользователя.
   
## Технический стек

- Machine Learning:
    - Object detection: Faster R-CNN
    - OCR: кастомный трансформер на TransformerModelOCR
- Мобильная разработка: Android Studio, Java
- Backend: Uvicorn

## Документация

Documentation Here

## План развития

1. Краткосрочные цели
    - Улучшение точности распознавания
    - Оптимизация производительности
    - Расширение поддерживаемых форматов вывода

2. Долгосрочные цели

    - Интеграция с облачными сервисами
    - Добавление функций перевода языков
    - Автоматический анализ текста, контекстные добавления и исправления
    - Расширение обучающей выборки
    - Расширение функционала приложения

## License
