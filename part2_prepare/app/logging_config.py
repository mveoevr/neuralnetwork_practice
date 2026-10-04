"""Настройка журналирования: вывод в консоль и в файл."""

import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(level, log_dir):
    """Настраивает корневой логгер.

    :param level: уровень журналирования (DEBUG, INFO, WARNING, ...).
    :param log_dir: каталог для файла prepare.log.
    :return: путь к файлу журнала.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / "prepare.log"
    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format=LOG_FORMAT,
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )
    return log_file
