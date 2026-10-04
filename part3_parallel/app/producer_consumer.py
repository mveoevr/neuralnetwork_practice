"""Конвейер «производитель — потребители» с пулом процессов.

Схема работы:
  1. поток-производитель обходит каталог и кладёт пути в path_queue;
  2. несколько потоков-читателей берут пути, читают изображения и кладут
     массивы в array_queue;
  3. главный поток передаёт массивы в multiprocessing.Pool через
     apply_async(); результаты собирает callback-функция.
Общие счётчики защищены threading.Lock. Завершение организовано через
«сигнальные» объекты (sentinel), поэтому зависших потоков не остаётся.
"""

import logging
import multiprocessing
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.augment import augment_image, augment_task, read_image

LOGGER = logging.getLogger("pipeline")
SENTINEL = None


@dataclass
class PipelineStats:
    """Счётчики и замеры конвейера."""

    files_found: int = 0
    files_read: int = 0
    files_augmented: int = 0
    read_errors: int = 0
    augment_errors: int = 0
    seconds: float = 0.0
    per_thread: dict = field(default_factory=dict)


def find_files(root, extensions):
    """Возвращает отсортированный список файлов с нужными расширениями.

    :param root: каталог с изображениями.
    :param extensions: список расширений вида "png".
    :return: список объектов Path.
    """
    wanted = {"." + ext.lower().lstrip(".") for ext in extensions}
    return sorted(path for path in Path(root).rglob("*")
                  if path.is_file() and path.suffix.lower() in wanted)


class ParallelAugmentationPipeline:
    """Многопоточное чтение и многопроцессорная аугментация изображений."""

    def __init__(self, root, extensions, readers=4, workers=None,
                 variants=4, size=64, seed=0):
        """Сохраняет параметры конвейера.

        :param root: каталог с изображениями.
        :param extensions: список расширений.
        :param readers: число потоков-читателей.
        :param workers: число процессов (по умолчанию — число ядер).
        :param variants: число аугментированных копий на изображение.
        :param size: сторона изображения.
        :param seed: базовое зерно генератора.
        """
        self.root = Path(root)
        self.extensions = extensions
        self.readers = readers
        self.workers = workers or multiprocessing.cpu_count()
        self.variants = variants
        self.size = size
        self.seed = seed
        self.stats = PipelineStats()
        self._lock = threading.Lock()
        self._results = {}
        self._path_queue = queue.Queue(maxsize=64)
        self._array_queue = queue.Queue(maxsize=64)

    def _producer(self):
        """Поток-производитель: кладёт пути в очередь и сигналы завершения."""
        files = find_files(self.root, self.extensions)
        with self._lock:
            self.stats.files_found = len(files)
        LOGGER.info("Производитель нашёл %d файлов", len(files))
        for index, path in enumerate(files):
            self._path_queue.put((index, path))
        for _ in range(self.readers):
            self._path_queue.put(SENTINEL)
        LOGGER.info("Производитель завершил работу")

    def _reader(self):
        """Поток-потребитель: читает изображения и передаёт массивы дальше."""
        name = threading.current_thread().name
        processed = 0
        while True:
            item = self._path_queue.get()
            if item is SENTINEL:
                self._array_queue.put(SENTINEL)
                break
            index, path = item
            try:
                pixels = read_image(path, self.size)
            except OSError as error:
                LOGGER.warning("%s: не удалось прочитать %s (%s)",
                               name, path.name, error)
                with self._lock:
                    self.stats.read_errors += 1
                continue
            self._array_queue.put((index, pixels))
            processed += 1
            with self._lock:
                self.stats.files_read += 1
                count = self.stats.files_read
            if count % 50 == 0:
                LOGGER.info("%s: прочитано всего %d файлов", name, count)
        with self._lock:
            self.stats.per_thread[name] = processed
        LOGGER.info("%s завершён, обработал %d файлов", name, processed)

    def _on_done(self, result):
        """Callback пула: сохраняет результат аугментации.

        :param result: кортеж (index, массив копий).
        """
        index, augmented = result
        with self._lock:
            self._results[index] = augmented
            self.stats.files_augmented += 1

    def _on_error(self, error):
        """Callback пула при ошибке в дочернем процессе.

        :param error: исключение из процесса.
        """
        LOGGER.error("Ошибка аугментации: %s", error)
        with self._lock:
            self.stats.augment_errors += 1

    def run(self):
        """Запускает конвейер и дожидается завершения всех потоков и процессов.

        :return: массив float32 формы (N * variants, size, size, 3).
        """
        started = time.perf_counter()
        producer = threading.Thread(target=self._producer, name="Producer")
        readers = [threading.Thread(target=self._reader, name=f"Reader-{n}")
                   for n in range(1, self.readers + 1)]
        producer.start()
        for thread in readers:
            thread.start()
        LOGGER.info("Пул из %d процессов запущен", self.workers)
        finished_readers = 0
        with multiprocessing.Pool(self.workers) as pool:
            while finished_readers < self.readers:
                item = self._array_queue.get()
                if item is SENTINEL:
                    finished_readers += 1
                    continue
                index, pixels = item
                pool.apply_async(
                    augment_task,
                    (index, pixels, self.seed + index, self.variants),
                    callback=self._on_done, error_callback=self._on_error)
            pool.close()
            pool.join()
        producer.join()
        for thread in readers:
            thread.join()
        self.stats.seconds = time.perf_counter() - started
        alive = [t.name for t in [producer, *readers] if t.is_alive()]
        LOGGER.info("Все потоки завершены: %s", "да" if not alive else alive)
        ordered = [self._results[key] for key in sorted(self._results)]
        return np.concatenate(ordered) if ordered else np.empty((0,))


def run_sequential(root, extensions, variants=4, size=64, seed=0):
    """Последовательная версия той же обработки (для сравнения).

    :param root: каталог с изображениями.
    :param extensions: список расширений.
    :param variants: число копий на изображение.
    :param size: сторона изображения.
    :param seed: базовое зерно генератора.
    :return: кортеж (массив результатов, время в секундах, число файлов).
    """
    started = time.perf_counter()
    files = find_files(root, extensions)
    results = [augment_image(read_image(path, size), seed + index, variants)
               for index, path in enumerate(files)]
    seconds = time.perf_counter() - started
    data = np.concatenate(results) if results else np.empty((0,))
    return data, seconds, len(files)
