"""Функции активации и вспомогательные функции ввода."""

import numpy as np

from app.exceptions import UnknownActivationError


def sigmoid(values):
    """Вычисляет сигмоиду поэлементно.

    :param values: массив NumPy.
    :return: массив значений в интервале (0, 1).
    """
    clipped = np.clip(values, -500, 500)
    return 1.0 / (1.0 + np.exp(-clipped))


def sigmoid_derivative(activated):
    """Производная сигмоиды, выраженная через её значение.

    :param activated: результат sigmoid(z).
    :return: sigmoid(z) * (1 - sigmoid(z)).
    """
    return activated * (1.0 - activated)


def relu(values):
    """Вычисляет ReLU: max(0, z).

    :param values: массив NumPy.
    :return: массив с обнулёнными отрицательными значениями.
    """
    return np.maximum(0.0, values)


def relu_derivative(activated):
    """Производная ReLU по её значению.

    :param activated: результат relu(z).
    :return: 1 там, где значение больше нуля, иначе 0.
    """
    return (activated > 0).astype(float)


ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "relu": (relu, relu_derivative),
}


def get_activation(name):
    """Возвращает пару (функция, производная) по имени активации.

    :param name: "sigmoid" или "relu".
    :return: кортеж из двух функций.
    :raises UnknownActivationError: если имя неизвестно.
    """
    key = name.strip().lower()
    if key not in ACTIVATIONS:
        raise UnknownActivationError(
            f"Неизвестная активация '{name}'. Доступны: "
            f"{', '.join(ACTIVATIONS)}"
        )
    return ACTIVATIONS[key]


def ask_int(prompt, default=None, minimum=1):
    """Запрашивает у пользователя целое число.

    :param prompt: текст приглашения.
    :param default: значение при пустом вводе.
    :param minimum: минимально допустимое значение.
    :return: введённое целое число.
    :raises ValueError: если ввод не является допустимым числом.
    """
    raw = input(prompt).strip()
    if not raw and default is not None:
        return default
    number = int(raw)
    if number < minimum:
        raise ValueError(f"Число должно быть не меньше {minimum}")
    return number


def ask_float(prompt, default=None):
    """Запрашивает у пользователя положительное вещественное число.

    :param prompt: текст приглашения.
    :param default: значение при пустом вводе.
    :return: введённое число.
    :raises ValueError: если ввод некорректен.
    """
    raw = input(prompt).strip().replace(",", ".")
    if not raw and default is not None:
        return default
    number = float(raw)
    if number <= 0:
        raise ValueError("Число должно быть больше нуля")
    return number
