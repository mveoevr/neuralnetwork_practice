# Часть 3. Многопоточное чтение и многопроцессорная аугментация

```bash
cd part3_parallel
python -m app.performance --path ../data/input/images --ext png jpg \
    --output-dir ../data/output --readers 4 --workers 2 --variants 6 --size 128
```

Результаты: `data/output/performance_report.json` (сравнение времени
последовательной и параллельной версий) и `data/output/augmented.npy`
(расширенный набор данных).

| Модуль | Назначение |
| --- | --- |
| `app/producer_consumer.py` | поток-производитель, потоки-читатели (queue.Queue), пул процессов (Pool.apply_async + callback), счётчики под threading.Lock |
| `app/augment.py` | поворот, отражения, размытие, гауссов шум |
| `app/performance.py` | запуск обеих версий, проверка совпадения результатов, отчёт JSON |
