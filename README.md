# Практическое задание «Основы нейронных сетей»

Три самостоятельные части:

| Часть | Каталог | Запуск |
| --- | --- | --- |
| 1. ООП-реализация многослойного перцептрона | `part1_mlp/` | `python -m app.main` |
| 2. CLI-утилита подготовки данных | `part2_prepare/` | `python -m app.cli prepare ...`, `python -m app.cli doctor` |
| 3. Многопоточная и многопроцессорная аугментация | `part3_parallel/` | `python -m app.performance ...` |

## Установка

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # Windows: copy .env.example .env
python make_sample_data.py       # создаёт тестовые данные в data/input
```

Все пути и параметры задаются аргументами командной строки, переменными
окружения (см. `.env.example`) или берутся по умолчанию — в коде нет
захардкоженных путей.

## Структура

```
neural_network_practice/
├── part1_mlp/app/        main.py, models.py, manager.py, exceptions.py, utils.py
├── part2_prepare/app/    cli.py, config.py, scanner.py, logging_config.py
├── part3_parallel/app/   producer_consumer.py, augment.py, performance.py
├── data/input/           тестовые данные (создаются make_sample_data.py)
├── data/output/          результаты (в .gitignore)
├── logs/                 журналы (в .gitignore)
├── make_sample_data.py
├── requirements.txt
├── .gitignore
└── .env.example
```

Проверка стиля: `flake8 .`
