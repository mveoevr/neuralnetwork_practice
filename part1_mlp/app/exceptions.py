"""Собственные исключения части 1 (многослойный перцептрон)."""


class NeuralNetworkError(Exception):
    """Базовое исключение для всех ошибок приложения."""


class InvalidLayerSizeError(NeuralNetworkError):
    """Размеры слоёв заданы неверно (меньше двух слоёв или размер < 1)."""


class UnknownActivationError(NeuralNetworkError):
    """Запрошена неизвестная функция активации."""


class MismatchedDataError(NeuralNetworkError):
    """Размерность данных не совпадает с архитектурой сети."""


class DataFileError(NeuralNetworkError):
    """Файл с данными или весами отсутствует либо повреждён."""


class ModelNotReadyError(NeuralNetworkError):
    """Действие требует созданной сети или загруженных данных."""
