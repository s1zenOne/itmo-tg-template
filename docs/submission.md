# Материалы для сдачи лабораторных работ

Отчёты и результаты экспериментов находятся в соответствующих метках кода:

| Работа | Сдаваемая метка | Отчёт и результаты |
| --- | --- | --- |
| № 1 | [`lab1`](https://github.com/s1zenOne/itmo-tg-template/tree/lab1) | [Отчёт](https://github.com/s1zenOne/itmo-tg-template/blob/lab1/docs/lab1-report.md), [обезличенные измерения](https://github.com/s1zenOne/itmo-tg-template/blob/lab1/docs/evaluation/lab1-temperature-results.json) |
| № 2 | [`lab2`](https://github.com/s1zenOne/itmo-tg-template/tree/lab2) | [Отчёт](https://github.com/s1zenOne/itmo-tg-template/blob/lab2/docs/lab2-report.md), [результаты эксперимента](https://github.com/s1zenOne/itmo-tg-template/blob/lab2/docs/evaluation/lab2-routing-results.md) |
| № 3 | [`lab3`](https://github.com/s1zenOne/itmo-tg-template/tree/lab3) | [Отчёт](https://github.com/s1zenOne/itmo-tg-template/blob/lab3/docs/lab3-report.md), [исходная серия](https://github.com/s1zenOne/itmo-tg-template/blob/lab3/docs/evaluation/lab3-routing-baseline.json), [повтор](https://github.com/s1zenOne/itmo-tg-template/blob/lab3/docs/evaluation/lab3-routing-rerun.json), [итоговая серия](https://github.com/s1zenOne/itmo-tg-template/blob/lab3/docs/evaluation/lab3-routing-final.json) |

Проверка `RUN_LAB1_SUBMISSION_DEMO=1 .venv/bin/python -m pytest -q tests/test_lab1_submission_demo.py` воспроизводит два учебных аккаунта на временной PostgreSQL, восстановление состояния сервисов, изоляцию `/reset`, таймаут и разбиение длинного ответа. Она прошла на коде каждой из трёх работ. Telegram и LLM подменены; на версии № 2 проверяется базовый сервис без агента. Это не живая демонстрация двух реальных аккаунтов и действительной модели.

В Telegram дополнительно проверены работа с городом, погода и сводка. Для создания тестового напоминания применена подмена решения модели после сообщения об исчерпании бесплатного лимита OpenRouter. Фактические результаты и ограничения каждого испытания приведены в отчётах. Не включайте в сдаваемые материалы токены, пароли, полные личные диалоги или локальные базы данных.
