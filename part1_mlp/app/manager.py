"""Загрузка, нормализация и разбиение набора данных."""

import csv
from pathlib import Path

import numpy as np

from app.exceptions import (
    DataFileError,
    MismatchedDataError,
    ModelNotReadyError,
)


class DatasetManager:
    """Управляет набором данных из CSV-файла.

    Первая строка файла — заголовки, последний столбец — целевая переменная.
    """

    def __init__(self):
        """Создаёт пустой менеджер данных."""
        self.headers = []
        self.features = None
        self.targets = None
        self._min = None
        self._max = None
        self._mean = None
        self._std = None
        self.method = None

    def load_csv(self, path):
        """Загружает данные из CSV-файла.

        :param path: путь к файлу.
        :return: кортеж (число строк, число признаков).
        :raises DataFileError: если файл не найден или содержит ошибки.
        """
        source = Path(path).expanduser()
        if not source.is_file():
            raise DataFileError(f"Файл не найден: {source}")
        try:
            with source.open(encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                self.headers = next(reader)
                rows = [[float(cell) for cell in row] for row in reader if row]
        except (StopIteration, ValueError) as error:
            raise DataFileError(f"Ошибка чтения CSV: {error}") from error
        if not rows:
            raise DataFileError("CSV-файл не содержит данных")
        table = np.array(rows, dtype=float)
        self.features = table[:, :-1]
        self.targets = table[:, -1:]
        self.method = None
        return self.features.shape

    def normalize(self, method="standard"):
        """Нормализует признаки и запоминает параметры преобразования.

        :param method: "standard" (z-оценка) или "minmax".
        :raises ModelNotReadyError: если данные не загружены.
        """
        self._require_data()
        if method == "minmax":
            self._min = self.features.min(axis=0)
            self._max = self.features.max(axis=0)
        else:
            method = "standard"
            self._mean = self.features.mean(axis=0)
            self._std = self.features.std(axis=0)
        self.method = method
        self.features = self.transform(self.features)

    def transform(self, features):
        """Применяет сохранённую нормализацию к новым данным.

        :param features: матрица признаков.
        :return: нормализованная матрица.
        :raises MismatchedDataError: если число признаков не совпадает.
        """
        matrix = np.atleast_2d(np.asarray(features, dtype=float))
        if self.features is not None and \
                matrix.shape[1] != self.features.shape[1]:
            raise MismatchedDataError(
                f"Ожидалось {self.features.shape[1]} признака(ов), "
                f"получено {matrix.shape[1]}"
            )
        if self.method == "minmax":
            span = np.where(self._max - self._min == 0, 1,
                            self._max - self._min)
            return (matrix - self._min) / span
        if self.method == "standard":
            std = np.where(self._std == 0, 1, self._std)
            return (matrix - self._mean) / std
        return matrix

    def train_test_split(self, test_ratio=0.2, seed=42):
        """Разбивает выборку на обучающую и тестовую.

        :param test_ratio: доля тестовой выборки (0..1).
        :param seed: зерно перемешивания.
        :return: кортеж (x_train, x_test, y_train, y_test).
        """
        self._require_data()
        if not 0 < test_ratio < 1:
            raise ValueError("Доля тестовой выборки должна быть от 0 до 1")
        order = np.random.default_rng(seed).permutation(len(self.features))
        test_size = max(1, int(len(order) * test_ratio))
        test_index, train_index = order[:test_size], order[test_size:]
        return (self.features[train_index], self.features[test_index],
                self.targets[train_index], self.targets[test_index])

    def _require_data(self):
        """Проверяет, что данные загружены."""
        if self.features is None:
            raise ModelNotReadyError("Сначала загрузите данные (пункт 2)")
