"""Полносвязная нейронная сеть (многослойный перцептрон)."""

import json
from pathlib import Path

import numpy as np

from app.exceptions import (
    DataFileError,
    InvalidLayerSizeError,
    MismatchedDataError,
)
from app.utils import get_activation, sigmoid, sigmoid_derivative


class NeuralNetwork:
    """Многослойный перцептрон с обучением методом обратного распространения.

    Скрытые слои используют выбранную активацию (sigmoid или relu),
    выходной слой — сигмоиду, функция потерь — среднеквадратичная ошибка.
    """

    def __init__(self, layer_sizes, activation="sigmoid", seed=None):
        """Создаёт сеть и инициализирует веса случайными числами.

        :param layer_sizes: список размеров слоёв, например [2, 8, 1].
        :param activation: активация скрытых слоёв.
        :param seed: зерно генератора случайных чисел (необязательно).
        :raises InvalidLayerSizeError: при некорректных размерах слоёв.
        """
        if len(layer_sizes) < 3:
            raise InvalidLayerSizeError(
                "Нужно минимум три слоя: входной, скрытый и выходной"
            )
        if any(int(size) < 1 for size in layer_sizes):
            raise InvalidLayerSizeError("Размер каждого слоя должен быть >= 1")
        self.layer_sizes = [int(size) for size in layer_sizes]
        self.activation_name = activation
        self._activation, self._activation_derivative = get_activation(
            activation
        )
        self._rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        self._init_weights()
        self.loss_history = []
        self._layer_outputs = []

    def _init_weights(self):
        """Инициализирует веса по схеме He или Xavier."""
        self.weights.clear()
        self.biases.clear()
        for fan_in, fan_out in zip(self.layer_sizes[:-1],
                                   self.layer_sizes[1:]):
            scale = np.sqrt(2.0 / fan_in) if self.activation_name == "relu" \
                else np.sqrt(1.0 / fan_in)
            self.weights.append(self._rng.normal(0.0, scale,
                                                 (fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out)))

    def set_activation(self, name):
        """Меняет активацию скрытых слоёв и переинициализирует веса.

        :param name: "sigmoid" или "relu".
        """
        self._activation, self._activation_derivative = get_activation(name)
        self.activation_name = name
        self._init_weights()

    def _check_input(self, features):
        """Проверяет размерность входных данных.

        :param features: матрица признаков (n_samples, n_features).
        :return: матрица в виде двумерного массива float.
        :raises MismatchedDataError: если число признаков не совпадает.
        """
        matrix = np.atleast_2d(np.asarray(features, dtype=float))
        if matrix.shape[1] != self.layer_sizes[0]:
            raise MismatchedDataError(
                f"Сеть ожидает {self.layer_sizes[0]} признаков, "
                f"получено {matrix.shape[1]}"
            )
        return matrix

    def forward(self, features):
        """Прямое распространение сигнала.

        :param features: матрица признаков (n_samples, n_features).
        :return: выход сети (n_samples, n_outputs).
        """
        layer_output = self._check_input(features)
        self._layer_outputs = [layer_output]
        last_index = len(self.weights) - 1
        for index, (weight, bias) in enumerate(zip(self.weights,
                                                   self.biases)):
            weighted_sum = layer_output @ weight + bias
            if index == last_index:
                layer_output = sigmoid(weighted_sum)
            else:
                layer_output = self._activation(weighted_sum)
            self._layer_outputs.append(layer_output)
        return layer_output

    def backward(self, targets, learning_rate):
        """Обратное распространение ошибки и шаг градиентного спуска.

        Использует выходы слоёв, сохранённые при последнем forward().

        :param targets: истинные значения (n_samples, n_outputs).
        :param learning_rate: скорость обучения.
        """
        batch_size = targets.shape[0]
        prediction = self._layer_outputs[-1]
        # dL/dz для выходного слоя: (y_pred - y) * sigma'(z), L = MSE
        delta = (prediction - targets) * sigmoid_derivative(prediction)
        for index in range(len(self.weights) - 1, -1, -1):
            previous_output = self._layer_outputs[index]
            grad_weight = previous_output.T @ delta / batch_size
            grad_bias = delta.mean(axis=0, keepdims=True)
            if index > 0:
                delta = (delta @ self.weights[index].T) * \
                    self._activation_derivative(previous_output)
            self.weights[index] -= learning_rate * grad_weight
            self.biases[index] -= learning_rate * grad_bias

    def train(self, features, targets, learning_rate=0.1, epochs=100,
              batch_size=16, verbose_every=10):
        """Обучает сеть мини-батчами.

        :param features: матрица признаков.
        :param targets: целевые значения (n_samples,) или (n_samples, k).
        :param learning_rate: скорость обучения.
        :param epochs: число эпох.
        :param batch_size: размер мини-батча.
        :param verbose_every: как часто печатать ошибку (0 — не печатать).
        :return: список значений ошибки по эпохам.
        """
        matrix = self._check_input(features)
        target_matrix = np.asarray(targets, dtype=float).reshape(
            matrix.shape[0], -1
        )
        if target_matrix.shape[1] != self.layer_sizes[-1]:
            raise MismatchedDataError(
                f"Выходной слой имеет {self.layer_sizes[-1]} нейрон(ов), "
                f"а целевая переменная — {target_matrix.shape[1]} столбец(ов)"
            )
        for epoch in range(1, epochs + 1):
            order = self._rng.permutation(matrix.shape[0])
            for start in range(0, matrix.shape[0], batch_size):
                batch_index = order[start:start + batch_size]
                self.forward(matrix[batch_index])
                self.backward(target_matrix[batch_index], learning_rate)
            epoch_loss = float(np.mean(
                (self.forward(matrix) - target_matrix) ** 2
            ))
            self.loss_history.append(epoch_loss)
            if verbose_every and (epoch % verbose_every == 0 or epoch == 1):
                print(f"  эпоха {epoch:>4}/{epochs}  MSE = {epoch_loss:.5f}")
        return self.loss_history

    def predict(self, features, threshold=0.5):
        """Возвращает вероятности и метки классов.

        :param features: матрица признаков.
        :param threshold: порог отнесения к классу 1.
        :return: кортеж (вероятности, метки).
        """
        probabilities = self.forward(features)
        return probabilities, (probabilities >= threshold).astype(int)

    def save(self, path):
        """Сохраняет архитектуру и веса в JSON-файл.

        :param path: путь к файлу.
        """
        payload = {
            "layer_sizes": self.layer_sizes,
            "activation": self.activation_name,
            "weights": [weight.tolist() for weight in self.weights],
            "biases": [bias.tolist() for bias in self.biases],
            "loss_history": self.loss_history,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path):
        """Создаёт сеть по сохранённому JSON-файлу.

        :param path: путь к файлу весов.
        :return: экземпляр NeuralNetwork.
        :raises DataFileError: если файл отсутствует или повреждён.
        """
        source = Path(path)
        if not source.is_file():
            raise DataFileError(f"Файл весов не найден: {source}")
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            network = cls(payload["layer_sizes"], payload["activation"])
            network.weights = [np.array(w) for w in payload["weights"]]
            network.biases = [np.array(b) for b in payload["biases"]]
            network.loss_history = payload.get("loss_history", [])
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise DataFileError(f"Файл весов повреждён: {error}") from error
        return network

    def __repr__(self):
        """Краткое описание сети."""
        return (f"NeuralNetwork(layers={self.layer_sizes}, "
                f"activation='{self.activation_name}')")
