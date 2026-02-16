# Модуль подсчёта объектов (цветов) на изображении для мониторинга экосистем.
# Используется обученная модель классификации и скользящее окно по изображению.
# Запуск: из корня проекта или с путём к модели и изображению.

import argparse
import sys
from pathlib import Path

import numpy as np


def load_model_and_classes(model_path: Path, class_names_path: Path = None):
    """Загружает сохранённую модель Keras и список имён классов."""
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        raise ImportError("Установите tensorflow: pip install tensorflow")
    model = keras.models.load_model(model_path)
    # Имена классов можно сохранить отдельно или захардкодить для 5 цветов
    if class_names_path and class_names_path.exists():
        classes = class_names_path.read_text(encoding="utf-8").strip().split("\n")
    else:
        classes = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
    return model, classes


def count_flowers_sliding_window(model, image_path: Path, class_names: list,
                                 img_size=(224, 224), stride=112, confidence_threshold=0.75):
    """
    Оценка количества цветов на большом изображении методом скользящего окна.
    Каждое окно классифицируется; окна с уверенностью выше порога считаются детекцией.
    Возвращает число срабатываний по классам и список предсказаний с координатами.
    """
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    img = load_img(image_path)
    img_arr = np.array(img) / 255.0
    h, w = img_arr.shape[:2]
    results = []
    count_by_class = {c: 0 for c in class_names}
    for y in range(0, max(1, h - img_size[0] + 1), stride):
        for x in range(0, max(1, w - img_size[1] + 1), stride):
            patch = img_arr[y : y + img_size[0], x : x + img_size[1]]
            if patch.shape[0] != img_size[0] or patch.shape[1] != img_size[1]:
                continue
            preds = model.predict(np.expand_dims(patch, 0), verbose=0)[0]
            pred_class_idx = int(np.argmax(preds))
            confidence = float(preds[pred_class_idx])
            if confidence >= confidence_threshold:
                count_by_class[class_names[pred_class_idx]] += 1
            results.append({
                "x": x, "y": y,
                "class": class_names[pred_class_idx],
                "confidence": confidence,
            })
    return count_by_class, results


def count_flowers_single(model, image_path: Path, class_names: list,
                          img_size=(224, 224), confidence_threshold=0.75):
    """
    Классификация одного изображения (одно «окно»). Для мониторинга — один снимок = один объект.
    """
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    img = load_img(image_path, target_size=img_size)
    img_arr = np.array(img) / 255.0
    preds = model.predict(np.expand_dims(img_arr, 0), verbose=0)[0]
    pred_class_idx = int(np.argmax(preds))
    confidence = float(preds[pred_class_idx])
    return {
        "class": class_names[pred_class_idx],
        "confidence": confidence,
        "needs_manual_review": confidence < confidence_threshold,
    }


def main():
    parser = argparse.ArgumentParser(description="Подсчёт цветов на изображении для мониторинга экосистем")
    parser.add_argument("model", type=Path, help="Путь к файлу модели (.keras)")
    parser.add_argument("image", type=Path, help="Путь к изображению")
    parser.add_argument("--sliding", action="store_true", help="Использовать скользящее окно по изображению")
    parser.add_argument("--stride", type=int, default=112, help="Шаг скользящего окна (по умолчанию 112)")
    parser.add_argument("--threshold", type=float, default=0.75, help="Порог уверенности для учёта объекта")
    args = parser.parse_args()
    if not args.model.exists():
        print("Ошибка: файл модели не найден:", args.model, file=sys.stderr)
        sys.exit(1)
    if not args.image.exists():
        print("Ошибка: файл изображения не найден:", args.image, file=sys.stderr)
        sys.exit(1)
    model, class_names = load_model_and_classes(args.model)
    if args.sliding:
        count_by_class, results = count_flowers_sliding_window(
            model, args.image, class_names, confidence_threshold=args.threshold, stride=args.stride
        )
        print("Оценка количества по классам (скользящее окно):", count_by_class)
        print("Всего предсказаний выше порога:", sum(count_by_class.values()))
    else:
        r = count_flowers_single(model, args.image, class_names, confidence_threshold=args.threshold)
        print("Класс:", r["class"], "| Уверенность:", f"{r['confidence']:.4f}", "| Ручная проверка:", r["needs_manual_review"])


if __name__ == "__main__":
    main()
