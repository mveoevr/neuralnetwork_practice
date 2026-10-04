# Часть 2. CLI-утилита подготовки данных

```bash
cd part2_prepare
# изображения -> массив (N, size, size, 3), значения делятся на 255
python -m app.cli prepare --path ../data/input/images --ext png jpg \
    --output ../data/output/images.npy --size 32
# CSV-таблицы -> массив (строки, столбцы) с min-max нормализацией
python -m app.cli prepare --path ../data/input/tables --ext csv \
    --output ../data/output/tables.npy --log-level DEBUG
# проверка окружения
python -m app.cli doctor
```

Приоритет настроек: аргументы CLI > переменные окружения (`NN_DATA_DIR`,
`NN_OUTPUT_DIR`, `NN_LOG_DIR`, `NN_LOG_LEVEL`, `NN_IMAGE_SIZE`, в том числе из
`.env`) > значения по умолчанию в `app/config.py`.

| Модуль | Назначение |
| --- | --- |
| `app/cli.py` | argparse с подкомандами prepare и doctor; внешние команды через subprocess.run |
| `app/config.py` | чтение настроек через os.getenv и python-dotenv |
| `app/scanner.py` | рекурсивный обход (pathlib.rglob), чтение и нормализация, сохранение .npy, резервная копия и проверка места на диске через shutil |
| `app/logging_config.py` | журнал в консоль и в `logs/prepare.log` |
