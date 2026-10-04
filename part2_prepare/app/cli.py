"""CLI-утилита: команды prepare и doctor.

Примеры (из каталога part2_prepare):
    python -m app.cli prepare --path ../data/input/images --ext png jpg \
        --output ../data/output/images.npy
    python -m app.cli doctor
"""

import argparse
import importlib.metadata
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from app.config import load_settings
from app.logging_config import setup_logging
from app.scanner import PrepareError, prepare

LOGGER = logging.getLogger("cli")
REQUIRED_PACKAGES = ("numpy", "pillow", "python-dotenv")
OPTIONAL_PACKAGES = ("matplotlib", "torch", "tensorflow")


def build_parser(settings):
    """Создаёт парсер аргументов с подкомандами.

    :param settings: настройки по умолчанию.
    :return: объект argparse.ArgumentParser.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--log-level", default=settings.log_level,
                        help="уровень журнала (по умолчанию из NN_LOG_LEVEL)")
    common.add_argument("--log-dir", default=str(settings.log_dir),
                        help="каталог журналов (NN_LOG_DIR)")
    parser = argparse.ArgumentParser(
        prog="prepare-tool",
        description="Подготовка данных для нейронной сети")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser(
        "prepare", parents=[common],
        help="собрать файлы каталога в единый .npy")
    prepare_parser.add_argument("--path", default=str(settings.data_dir),
                                help="каталог с данными (NN_DATA_DIR)")
    prepare_parser.add_argument("--ext", nargs="+", required=True,
                                help="расширения, например png jpg или csv")
    prepare_parser.add_argument(
        "--output", default=str(settings.output_dir / "dataset.npy"),
        help="итоговый .npy (по умолчанию NN_OUTPUT_DIR/dataset.npy)")
    prepare_parser.add_argument("--size", type=int,
                                default=settings.image_size,
                                help="размер изображения (NN_IMAGE_SIZE)")

    commands.add_parser("doctor", parents=[common],
                        help="проверить окружение")
    return parser


def run_prepare(args):
    """Выполняет команду prepare и печатает статистику.

    :param args: разобранные аргументы.
    :return: код завершения.
    """
    try:
        stats = prepare(args.path, args.ext, args.output, args.size)
    except PrepareError as error:
        LOGGER.error("%s", error)
        return 2
    print("\n----- Статистика prepare -----")
    print(f"Каталог:            {Path(args.path).resolve()}")
    print(f"Найдено файлов:     {stats.files_found}")
    print(f"Обработано:         {stats.files_processed}")
    print(f"Пропущено:          {stats.files_skipped}")
    print(f"Объём исходных:     {stats.total_bytes / 1024:.1f} КБ")
    for shape, count in stats.shapes.items():
        print(f"Форма {shape:<14} {count} шт.")
    print(f"Время обработки:    {stats.seconds:.2f} с")
    print(f"Результат:          {Path(args.output).resolve()}")
    return 0


def run_command(command):
    """Запускает внешнюю команду и возвращает её вывод.

    :param command: список аргументов команды.
    :return: строка вывода или описание ошибки.
    """
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=15, check=False)
    except FileNotFoundError:
        return "не найдено"
    except subprocess.TimeoutExpired:
        return "превышено время ожидания"
    output = (result.stdout or result.stderr).strip()
    return output.splitlines()[0] if output else f"код {result.returncode}"


def package_version(name):
    """Возвращает версию установленного пакета или «не установлен».

    :param name: имя дистрибутива pip.
    :return: строка с версией.
    """
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "не установлен"


def run_doctor(_args):
    """Проверяет окружение и печатает отчёт.

    :param _args: разобранные аргументы (не используются).
    :return: код завершения (1, если нет обязательных пакетов).
    """
    print("----- Проверка окружения (doctor) -----")
    print(f"Python:             {platform.python_version()} "
          f"({sys.executable})")
    print(f"ОС:                 {platform.system()} {platform.release()}")
    print(f"Рабочий каталог:    {Path.cwd()}")
    print(f"pip:                {run_command([sys.executable, '-m', 'pip', '--version'])}")  # noqa: E501
    missing = []
    print("\nОбязательные пакеты:")
    for name in REQUIRED_PACKAGES:
        version = package_version(name)
        missing += [name] if version == "не установлен" else []
        print(f"  {name:<16}{version}")
    print("Дополнительные пакеты:")
    for name in OPTIONAL_PACKAGES:
        print(f"  {name:<16}{package_version(name)}")
    print("\nGPU / CUDA:")
    if shutil.which("nvidia-smi"):
        print("  " + run_command(["nvidia-smi", "--query-gpu=name,"
                                  "driver_version", "--format=csv,noheader"]))
    else:
        print("  nvidia-smi не найден — CUDA недоступна, вычисления на CPU")
    print(f"\nПеременная NN_DATA_DIR: {os.getenv('NN_DATA_DIR', 'не задана')}")
    if missing:
        print(f"\nНе хватает пакетов: {', '.join(missing)}. "
              f"Выполните: pip install -r requirements.txt")
        return 1
    print("\nОкружение готово к работе.")
    return 0


def main(argv=None):
    """Точка входа CLI.

    :param argv: список аргументов (для тестов).
    :return: код завершения.
    """
    settings = load_settings()
    args = build_parser(settings).parse_args(argv)
    log_file = setup_logging(args.log_level, args.log_dir)
    LOGGER.debug("Журнал пишется в %s", log_file)
    handlers = {"prepare": run_prepare, "doctor": run_doctor}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
