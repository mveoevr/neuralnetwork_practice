"""Настройки утилиты: значения по умолчанию, переменные окружения, .env.

Приоритет: аргумент командной строки > переменная окружения (в том числе
из файла .env) > значение по умолчанию.
"""

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv необязателен
    load_dotenv = None

DEFAULT_DATA_DIR = "data/input"
DEFAULT_OUTPUT_DIR = "data/output"
DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_IMAGE_SIZE = 32


@dataclass
class Settings:
    """Набор настроек утилиты."""

    data_dir: Path
    output_dir: Path
    log_dir: Path
    log_level: str
    image_size: int


def load_settings(env_file=None):
    """Читает настройки из окружения (и .env, если он есть).

    :param env_file: путь к .env-файлу; по умолчанию ищется в текущем и
        родительских каталогах.
    :return: объект Settings.
    """
    if load_dotenv is not None:
        load_dotenv(env_file)
    return Settings(
        data_dir=Path(os.getenv("NN_DATA_DIR", DEFAULT_DATA_DIR)),
        output_dir=Path(os.getenv("NN_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)),
        log_dir=Path(os.getenv("NN_LOG_DIR", DEFAULT_LOG_DIR)),
        log_level=os.getenv("NN_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper(),
        image_size=int(os.getenv("NN_IMAGE_SIZE", DEFAULT_IMAGE_SIZE)),
    )
