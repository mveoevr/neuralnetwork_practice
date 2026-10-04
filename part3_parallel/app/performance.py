"""Сравнение последовательной и параллельной аугментации.

Запуск из каталога part3_parallel:
    python -m app.performance --path ../data/input/images --ext png jpg

Параметры можно задать через переменные окружения NN_DATA_DIR,
NN_OUTPUT_DIR, NN_READERS, NN_WORKERS, NN_VARIANTS, NN_IMAGE_SIZE.
"""

import argparse
import json
import logging
import multiprocessing
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from app.producer_consumer import ParallelAugmentationPipeline, run_sequential

LOGGER = logging.getLogger("performance")


def build_parser():
    """Создаёт парсер аргументов со значениями по умолчанию из окружения.

    :return: объект argparse.ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="Параллельная аугментация и замер производительности")
    parser.add_argument("--path",
                        default=os.getenv("NN_DATA_DIR", "data/input"),
                        help="каталог с изображениями (NN_DATA_DIR)")
    parser.add_argument("--ext", nargs="+", default=["png", "jpg"],
                        help="расширения файлов")
    parser.add_argument("--output-dir",
                        default=os.getenv("NN_OUTPUT_DIR", "data/output"),
                        help="каталог для результатов (NN_OUTPUT_DIR)")
    parser.add_argument("--readers", type=int,
                        default=int(os.getenv("NN_READERS", "4")),
                        help="число потоков-читателей")
    parser.add_argument("--workers", type=int,
                        default=int(os.getenv("NN_WORKERS",
                                              multiprocessing.cpu_count())),
                        help="число процессов аугментации")
    parser.add_argument("--variants", type=int,
                        default=int(os.getenv("NN_VARIANTS", "4")),
                        help="копий на одно изображение")
    parser.add_argument("--size", type=int,
                        default=int(os.getenv("NN_IMAGE_SIZE", "64")),
                        help="сторона изображения")
    parser.add_argument("--log-level", default=os.getenv("NN_LOG_LEVEL",
                                                         "INFO"))
    return parser


def setup_logging(level):
    """Настраивает журнал с именами потоков и процессов.

    :param level: уровень журналирования.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(threadName)-10s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv=None):
    """Запускает обе версии, проверяет результаты и пишет отчёт.

    :param argv: список аргументов (для тестов).
    :return: код завершения.
    """
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    if not Path(args.path).is_dir():
        LOGGER.error("Каталог не найден: %s", args.path)
        return 2
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Последовательная обработка...")
    sequential_data, sequential_seconds, files = run_sequential(
        args.path, args.ext, args.variants, args.size)
    LOGGER.info("Последовательно: %d файлов за %.2f с", files,
                sequential_seconds)

    LOGGER.info("Параллельная обработка...")
    pipeline = ParallelAugmentationPipeline(
        args.path, args.ext, args.readers, args.workers, args.variants,
        args.size)
    parallel_data = pipeline.run()
    stats = pipeline.stats
    LOGGER.info("Параллельно: %d файлов за %.2f с", stats.files_augmented,
                stats.seconds)

    identical = bool(sequential_data.shape == parallel_data.shape and
                     np.allclose(sequential_data, parallel_data))
    speedup = sequential_seconds / stats.seconds if stats.seconds else 0.0
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": multiprocessing.cpu_count(),
        "files": files,
        "variants_per_file": args.variants,
        "image_size": args.size,
        "output_samples": int(parallel_data.shape[0]),
        "readers": args.readers,
        "workers": args.workers,
        "sequential_seconds": round(sequential_seconds, 3),
        "parallel_seconds": round(stats.seconds, 3),
        "speedup": round(speedup, 2),
        "results_identical": identical,
        "read_errors": stats.read_errors,
        "augment_errors": stats.augment_errors,
        "files_per_reader": stats.per_thread,
    }
    report_path = output_dir / "performance_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    np.save(output_dir / "augmented.npy", parallel_data)
    print("\n----- Сравнение производительности -----")
    print(f"Последовательно:  {sequential_seconds:.2f} с")
    print(f"Параллельно:      {stats.seconds:.2f} с "
          f"({args.readers} потока чтения, {args.workers} процесса)")
    print(f"Ускорение:        {speedup:.2f}x")
    print(f"Результаты совпадают: {'да' if identical else 'нет'}")
    print(f"Отчёт:            {report_path.resolve()}")
    alive = [p.name for p in multiprocessing.active_children()]
    print(f"Активных дочерних процессов после завершения: {len(alive)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
