# Скрипты проекта

## count_objects.py

Модуль для **подсчёта объектов (цветов) на изображении** в рамках мониторинга экосистем.

- **Один кадр:** классификация одного изображения, вывод класса и уверенности; флаг «требуется ручная проверка» при низкой уверенности.
- **Скользящее окно:** разбиение большого снимка на патчи 224×224 с заданным шагом (stride), классификация каждого патча и подсчёт срабатываний по классам выше порога уверенности.

### Использование

Из корня проекта (после обучения модели):

```bash
# Классификация одного изображения
python scripts/count_objects.py checkpoints/flowers_effnet_best.keras path/to/image.jpg

# Скользящее окно по большому изображению
python scripts/count_objects.py checkpoints/flowers_effnet_best.keras path/to/large_image.jpg --sliding --stride 112

# Порог уверенности (по умолчанию 0.75)
python scripts/count_objects.py checkpoints/flowers_effnet_best.keras image.jpg --threshold 0.8
```

Требуется установленные зависимости из `requirements.txt` в корне проекта.
