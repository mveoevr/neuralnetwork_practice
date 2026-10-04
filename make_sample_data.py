"""Генерирует тестовые данные для всех частей работы.

Создаёт:
  * data/input/moons.csv — две «луны» (2 признака + метка 0/1) для части 1;
  * data/input/images/<класс>/*.png — синтетические изображения фигур;
  * data/input/tables/*.csv — небольшие таблицы для команды prepare.

Запуск:  python make_sample_data.py --root data/input --images 60 --seed 7
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def make_moons(samples, noise, rng):
    """Возвращает признаки и метки датасета «две луны».

    :param samples: общее число точек.
    :param noise: стандартное отклонение шума.
    :param rng: генератор случайных чисел NumPy.
    :return: кортеж (features, labels).
    """
    half = samples // 2
    angles = rng.uniform(0, np.pi, half)
    upper = np.column_stack([np.cos(angles), np.sin(angles)])
    lower = np.column_stack([1 - np.cos(angles), 0.5 - np.sin(angles)])
    features = np.vstack([upper, lower]) + rng.normal(0, noise,
                                                      (2 * half, 2))
    labels = np.concatenate([np.zeros(half), np.ones(half)])
    return features, labels


def write_moons(path, samples, rng):
    """Сохраняет датасет «две луны» в CSV с заголовком.

    :param path: путь к файлу.
    :param samples: число строк.
    :param rng: генератор случайных чисел.
    """
    features, labels = make_moons(samples, 0.15, rng)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x1", "x2", "label"])
        for (first, second), label in zip(features, labels):
            writer.writerow([f"{first:.4f}", f"{second:.4f}", int(label)])


def draw_shape(shape, size, rng):
    """Рисует фигуру со случайным положением, цветом и шумом.

    :param shape: "circle", "square" или "triangle".
    :param size: сторона изображения в пикселях.
    :param rng: генератор случайных чисел.
    :return: объект PIL.Image.
    """
    background = tuple(int(v) for v in rng.integers(200, 256, 3))
    image = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(image)
    color = tuple(int(v) for v in rng.integers(0, 150, 3))
    margin = int(rng.integers(size // 8, size // 4))
    box = [margin, margin, size - margin, size - margin]
    if shape == "circle":
        draw.ellipse(box, fill=color)
    elif shape == "square":
        draw.rectangle(box, fill=color)
    else:
        draw.polygon([(size // 2, margin), (size - margin, size - margin),
                      (margin, size - margin)], fill=color)
    pixels = np.asarray(image, dtype=float)
    pixels += rng.normal(0, 8, pixels.shape)
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def main():
    """Разбирает аргументы и создаёт все тестовые файлы."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="data/input",
                        help="каталог для входных данных")
    parser.add_argument("--images", type=int, default=60,
                        help="число изображений на класс")
    parser.add_argument("--size", type=int, default=64,
                        help="сторона изображения, пикселей")
    parser.add_argument("--seed", type=int, default=7, help="зерно")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    root = Path(args.root)
    write_moons(root / "moons.csv", 400, rng)
    for shape in ("circle", "square", "triangle"):
        folder = root / "images" / shape
        folder.mkdir(parents=True, exist_ok=True)
        for number in range(args.images):
            extension = "png" if number % 3 else "jpg"
            draw_shape(shape, args.size, rng).save(
                folder / f"{shape}_{number:03d}.{extension}")
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    for number in range(3):
        values = rng.normal(0, 1, (50, 4))
        np.savetxt(tables / f"sensor_{number}.csv", values, delimiter=",",
                   header="s1,s2,s3,s4", comments="", fmt="%.4f")
    print(f"Тестовые данные созданы в {root.resolve()}")


if __name__ == "__main__":
    main()
