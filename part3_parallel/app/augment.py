"""Аугментация изображений (CPU-bound операции).

Функции модуля выполняются в дочерних процессах, поэтому они объявлены на
верхнем уровне модуля и принимают только сериализуемые аргументы.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def read_image(path, size):
    """Читает изображение, приводит к RGB и квадрату size x size.

    :param path: путь к файлу.
    :param size: сторона изображения в пикселях.
    :return: массив uint8 формы (size, size, 3).
    """
    with Image.open(Path(path)) as image:
        return np.asarray(image.convert("RGB").resize((size, size)))


def augment_image(pixels, seed, variants):
    """Создаёт несколько аугментированных копий изображения.

    Каждая копия получает случайный поворот, возможное зеркальное
    отражение, размытие и гауссов шум. Результат нормализован в [0, 1].

    :param pixels: массив uint8 формы (H, W, 3).
    :param seed: зерно генератора (для воспроизводимости).
    :param variants: число создаваемых копий.
    :return: массив float32 формы (variants, H, W, 3).
    """
    rng = np.random.default_rng(seed)
    source = Image.fromarray(pixels)
    results = []
    for _ in range(variants):
        angle = float(rng.uniform(-30, 30))
        image = source.rotate(angle, resample=Image.BICUBIC,
                              fillcolor=(255, 255, 255))
        if rng.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if rng.random() < 0.3:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image = image.filter(ImageFilter.GaussianBlur(
            radius=float(rng.uniform(0, 1.5))))
        array = np.asarray(image, dtype=np.float32) / 255.0
        array += rng.normal(0, 0.03, array.shape).astype(np.float32)
        results.append(np.clip(array, 0.0, 1.0))
    return np.stack(results)


def augment_task(index, pixels, seed, variants):
    """Обёртка для пула процессов: возвращает индекс вместе с результатом.

    :param index: порядковый номер файла.
    :param pixels: исходное изображение.
    :param seed: зерно генератора.
    :param variants: число копий.
    :return: кортеж (index, массив копий).
    """
    return index, augment_image(pixels, seed, variants)
