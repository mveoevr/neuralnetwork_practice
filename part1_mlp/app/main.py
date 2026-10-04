"""Консольное меню для работы с многослойным перцептроном.

Запуск из каталога part1_mlp:  python -m app.main
Пути по умолчанию берутся из переменных окружения NN_DATA_FILE и
NN_OUTPUT_DIR (см. .env.example в корне проекта).
"""

import os
import sys
from pathlib import Path

import numpy as np

from app.exceptions import (
    MismatchedDataError,
    ModelNotReadyError,
    NeuralNetworkError,
)
from app.manager import DatasetManager
from app.models import NeuralNetwork
from app.utils import ask_float, ask_int

MENU = """
========== Многослойный перцептрон ==========
1. Создать сеть
2. Загрузить данные из CSV
3. Обучить сеть
4. Сделать предсказание
5. Сохранить / загрузить веса
6. Показать график ошибки
7. Выход
"""


class Application:
    """Связывает сеть, данные и пользовательский ввод."""

    def __init__(self):
        """Читает настройки из окружения и создаёт пустое состояние."""
        self.output_dir = Path(os.getenv("NN_OUTPUT_DIR", "output"))
        self.default_data = os.getenv("NN_DATA_FILE", "")
        self.network = None
        self.dataset = DatasetManager()
        self.split = None

    def create_network(self):
        """Пункт 1: создание сети с заданной архитектурой."""
        raw = input("Размеры слоёв через пробел (например 2 8 1): ")
        sizes = [int(part) for part in raw.split()]
        activation = input("Активация скрытых слоёв [sigmoid/relu]: ") \
            or "sigmoid"
        self.network = NeuralNetwork(sizes, activation, seed=1)
        print(f"Создана сеть: {self.network}")

    def load_data(self):
        """Пункт 2: загрузка, нормализация и разбиение данных."""
        prompt = "Путь к CSV"
        if self.default_data:
            prompt += f" [{self.default_data}]"
        path = input(prompt + ": ").strip() or self.default_data
        rows, columns = self.dataset.load_csv(path)
        method = input("Нормализация [standard/minmax]: ") or "standard"
        self.dataset.normalize(method)
        ratio = ask_float("Доля тестовой выборки [0.2]: ", default=0.2)
        self.split = self.dataset.train_test_split(ratio)
        print(f"Загружено {rows} строк, {columns} признака(ов); "
              f"обучающая выборка: {len(self.split[0])}, "
              f"тестовая: {len(self.split[1])}")

    def train(self):
        """Пункт 3: обучение сети и оценка точности на тесте."""
        self._require_ready()
        learning_rate = ask_float("Скорость обучения [0.5]: ", default=0.5)
        epochs = ask_int("Число эпох [200]: ", default=200)
        batch_size = ask_int("Размер батча [16]: ", default=16)
        x_train, x_test, y_train, y_test = self.split
        self.network.train(x_train, y_train, learning_rate, epochs,
                           batch_size, verbose_every=max(1, epochs // 10))
        _, labels = self.network.predict(x_test)
        accuracy = float(np.mean(labels == y_test))
        print(f"Точность на тестовой выборке: {accuracy:.1%}")

    def predict(self):
        """Пункт 4: предсказание по введённым значениям или файлу."""
        if self.network is None:
            raise ModelNotReadyError("Сначала создайте или загрузите сеть")
        raw = input("Признаки через запятую или путь к CSV без целевого "
                    "столбца: ").strip()
        if Path(raw).is_file():
            features = np.loadtxt(raw, delimiter=",", skiprows=1, ndmin=2)
        else:
            features = np.array([[float(item) for item in raw.split(",")]])
        if features.shape[1] != self.network.layer_sizes[0]:
            raise MismatchedDataError(
                f"Сеть ожидает {self.network.layer_sizes[0]} признака(ов), "
                f"введено {features.shape[1]}"
            )
        normalized = self.dataset.transform(features)
        probabilities, labels = self.network.predict(normalized)
        for row, prob, label in zip(features, probabilities, labels):
            print(f"  {row.tolist()} -> вероятность класса 1 = "
                  f"{prob[0]:.3f}, класс = {label[0]}")

    def save_or_load(self):
        """Пункт 5: сохранение или загрузка весов в JSON."""
        action = input("s — сохранить, l — загрузить: ").strip().lower()
        default_path = self.output_dir / "weights.json"
        path = input(f"Файл весов [{default_path}]: ").strip() or default_path
        if action == "s":
            if self.network is None:
                raise ModelNotReadyError("Нет сети для сохранения")
            self.network.save(path)
            print(f"Веса сохранены в {path}")
        elif action == "l":
            self.network = NeuralNetwork.load(path)
            print(f"Загружена сеть: {self.network}")
        else:
            print("Неизвестное действие")

    def show_loss_plot(self):
        """Пункт 6: строит график ошибки и сохраняет его в PNG."""
        if self.network is None or not self.network.loss_history:
            raise ModelNotReadyError("Сеть ещё не обучалась")
        import matplotlib
        if not os.getenv("DISPLAY"):
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.output_dir.mkdir(parents=True, exist_ok=True)
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot(self.network.loss_history, color="#2b5c9e")
        axis.set_xlabel("Эпоха")
        axis.set_ylabel("MSE")
        axis.set_title("Ошибка обучения")
        axis.grid(alpha=0.3)
        target = self.output_dir / "loss.png"
        figure.savefig(target, dpi=120, bbox_inches="tight")
        print(f"График сохранён в {target}")
        if os.getenv("DISPLAY"):
            plt.show()
        plt.close(figure)

    def _require_ready(self):
        """Проверяет наличие сети и данных."""
        if self.network is None:
            raise ModelNotReadyError("Сначала создайте сеть (пункт 1)")
        if self.split is None:
            raise ModelNotReadyError("Сначала загрузите данные (пункт 2)")

    def run(self):
        """Главный цикл меню."""
        actions = {
            "1": self.create_network, "2": self.load_data, "3": self.train,
            "4": self.predict, "5": self.save_or_load,
            "6": self.show_loss_plot,
        }
        while True:
            print(MENU)
            choice = input("Выберите пункт: ").strip()
            if choice == "7":
                print("Работа завершена.")
                return
            action = actions.get(choice)
            if action is None:
                print("Нет такого пункта меню")
                continue
            try:
                action()
            except NeuralNetworkError as error:
                print(f"Ошибка: {error}")
            except ValueError as error:
                print(f"Некорректный ввод: {error}")


def main():
    """Точка входа."""
    try:
        Application().run()
    except (KeyboardInterrupt, EOFError):
        print("\nПрервано пользователем")
        sys.exit(0)


if __name__ == "__main__":
    main()
