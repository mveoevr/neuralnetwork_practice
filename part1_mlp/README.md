# Часть 1. Многослойный перцептрон (ООП)

Полносвязная нейронная сеть с обратным распространением ошибки,
менеджер данных и консольное меню.

## Структура

| Файл | Назначение |
| --- | --- |
| `app/models.py` | класс `NeuralNetwork`: forward, backward, train, predict, save/load |
| `app/manager.py` | класс `DatasetManager`: чтение CSV, нормализация, train/test split |
| `app/exceptions.py` | собственные исключения (`InvalidLayerSizeError`, `MismatchedDataError` и др.) |
| `app/utils.py` | функции активации (sigmoid, ReLU) и функции ввода |
| `app/main.py` | консольное меню |

## Запуск

```bash
cd part1_mlp
export NN_DATA_FILE=../data/input/moons.csv   # необязательно
export NN_OUTPUT_DIR=../data/output           # необязательно
python -m app.main
```

Пример сессии: создать сеть `2 8 1` с активацией `relu`, загрузить
`../data/input/moons.csv`, обучить (скорость 0.5, 300 эпох, батч 16),
сделать предсказание `1.0,-0.4`, сохранить веса и построить график ошибки.
