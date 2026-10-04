"""Обход каталога и преобразование файлов в массивы NumPy."""

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

LOGGER = logging.getLogger("scanner")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
TABLE_EXTENSIONS = {".csv"}


class PrepareError(Exception):
    """Ошибка подготовки данных (нет файлов, разные форматы и т.п.)."""


@dataclass
class PrepareStats:
    """Статистика выполнения команды prepare."""

    files_found: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    total_bytes: int = 0
    shapes: dict = field(default_factory=dict)
    seconds: float = 0.0


def normalize_extensions(extensions):
    """Приводит расширения к виду ".png" в нижнем регистре.

    :param extensions: список строк вида "png", ".JPG", "csv".
    :return: множество нормализованных расширений.
    """
    return {"." + ext.lower().lstrip(".") for ext in extensions}


def scan_directory(root, extensions):
    """Рекурсивно находит файлы с заданными расширениями.

    :param root: корневой каталог.
    :param extensions: множество расширений.
    :return: отсортированный список путей.
    :raises PrepareError: если каталог не существует.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise PrepareError(f"Каталог не найден: {root_path}")
    return sorted(path for path in root_path.rglob("*")
                  if path.is_file() and path.suffix.lower() in extensions)


def load_image(path, size):
    """Читает изображение, приводит к RGB, size x size и диапазону [0, 1].

    :param path: путь к изображению.
    :param size: сторона итогового квадрата в пикселях.
    :return: массив формы (size, size, 3) типа float32.
    """
    with Image.open(path) as image:
        resized = image.convert("RGB").resize((size, size))
        return np.asarray(resized, dtype=np.float32) / 255.0


def load_table(path):
    """Читает CSV с заголовком в двумерный массив.

    :param path: путь к CSV.
    :return: массив формы (строк, столбцов) типа float32.
    """
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2,
                      dtype=np.float32)


def min_max_scale(table):
    """Масштабирует столбцы таблицы в диапазон [0, 1].

    :param table: двумерный массив.
    :return: масштабированный массив.
    """
    low, high = table.min(axis=0), table.max(axis=0)
    span = np.where(high - low == 0, 1, high - low)
    return (table - low) / span


def prepare(root, extensions, output, image_size):
    """Преобразует найденные файлы в единый массив и сохраняет в .npy.

    :param root: каталог с данными.
    :param extensions: список расширений.
    :param output: путь к итоговому .npy.
    :param image_size: сторона изображения после изменения размера.
    :return: объект PrepareStats.
    :raises PrepareError: если файлов нет или типы смешаны.
    """
    started = time.perf_counter()
    wanted = normalize_extensions(extensions)
    if wanted & IMAGE_EXTENSIONS and wanted & TABLE_EXTENSIONS:
        raise PrepareError("Нельзя смешивать изображения и CSV в одном "
                           "массиве — запустите prepare дважды")
    files = scan_directory(root, wanted)
    stats = PrepareStats(files_found=len(files))
    if not files:
        raise PrepareError(f"В {root} нет файлов с расширениями "
                           f"{sorted(wanted)}")
    LOGGER.info("Найдено файлов: %d", len(files))
    arrays = []
    for path in files:
        try:
            if path.suffix.lower() in TABLE_EXTENSIONS:
                array = load_table(path)
            else:
                array = load_image(path, image_size)
        except (OSError, UnidentifiedImageError, ValueError) as error:
            stats.files_skipped += 1
            LOGGER.warning("Пропущен %s: %s", path.name, error)
            continue
        stats.files_processed += 1
        stats.total_bytes += path.stat().st_size
        shape = "x".join(str(dim) for dim in array.shape)
        stats.shapes[shape] = stats.shapes.get(shape, 0) + 1
        LOGGER.debug("%s -> %s", path.relative_to(root), shape)
        arrays.append(array)
    if not arrays:
        raise PrepareError("Ни один файл не удалось прочитать")
    if wanted & TABLE_EXTENSIONS:
        combined = min_max_scale(np.vstack(arrays))
    else:
        combined = np.stack(arrays)
    save_array(combined, output)
    stats.seconds = time.perf_counter() - started
    LOGGER.info("Итоговый массив %s сохранён в %s", combined.shape, output)
    return stats


def save_array(array, output):
    """Сохраняет массив, предварительно сделав резервную копию старого файла.

    :param array: массив NumPy.
    :param output: путь к .npy.
    """
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(target.parent).free
    if free_bytes < array.nbytes:
        raise PrepareError("Недостаточно места на диске для сохранения")
    if target.exists():
        backup = target.with_suffix(".bak.npy")
        shutil.copy2(target, backup)
        LOGGER.info("Старый файл сохранён как %s", backup.name)
    np.save(target, array)
