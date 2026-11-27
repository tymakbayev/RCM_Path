# Код

Модули лежат в одной папке и подключают друг друга по имени, как на эталонном стенде. Имена файлов совпадают с
упоминаниями в протоколах и отчётах. Запуск из этой папки: `cd rcm_address_engine`.

## API и демо

| Файл | Назначение |
|---|---|
| [`rcm.py`](rcm.py) | API для собственной таблицы: `build`, `Address.open`, `check`, `get`, `query` ([описание](../docs/integration.md)) |
| [`demo.py`](demo.py) | сравнение адреса RCM и B-tree на одной таблице ([запуск](../README.md#быстрый-запуск)) |
| [`rcm_pgtest.py`](rcm_pgtest.py) | проверка `rcm.py` на работающем PostgreSQL |
| [`rcm_selftest.py`](rcm_selftest.py) | проверка `rcm.py` без базы |

## Движок

| Файл | Назначение |
|---|---|
| [`a2_core.py`](a2_core.py) | хеш ключа; адрес `JLMADR01`: каскад минимальной совершенной хеш-функции, отпечаток и номер страницы (вариант А); раскладка по хеш-корзинам (вариант Б) |
| [`a4_lsm.py`](a4_lsm.py) | изменяющиеся данные: журнал хвоста, адреса-прогоны, ярусные слияния (A4) |
| [`a5_lsm.py`](a5_lsm.py) | большое слияние шагами по хеш-секциям адреса (A5) |
| [`a6_lsm.py`](a6_lsm.py) | шаг слияния равен одной хеш-партиции таблицы, без рабочего файла (A6) |
| [`a6_pghash.py`](a6_pghash.py) | клиентская копия хеш-маршрутизации секций PostgreSQL, сверенная с сервером |
| [`a2_pg.py`](a2_pg.py) | слой PostgreSQL: настройки стенда из переменных окружения, `pg_ctl`, загрузка через COPY, клиенты поиска, EXPLAIN, `pg_statio` |
| [`hoststats.py`](hoststats.py) | память процесса и счётчики диска (Windows и Linux) |

## Эксперименты A2-A8

| Файл | Назначение |
|---|---|
| [`a2.py`](a2.py), [`a2_runs.py`](a2_runs.py), [`a2_dyn.py`](a2_dyn.py) | A2 и A3: `pilot`, `selfcheck`, `static N1/N10`, `dynamics`, `dynamics-a3` |
| [`a2_summary.py`](a2_summary.py) | краткая сводка статического прогона A2 |
| [`a4_run.py`](a4_run.py) … [`a8_run.py`](a8_run.py) | A4-A8: `selfcheck` (сценарии и мутанты), `dynamics`, `replay`, `open-probe`; для A8 `run` |
| [`a7_chain.cmd`](a7_chain.cmd), [`a8_chain.cmd`](a8_chain.cmd) | цепочки запуска A7 и A8 для Windows |
| [`probes/`](probes/) | пробы этапа 1: размер совершенного хеша, база SQLite |

Протоколы и отчёты лежат в [`docs/experiments/`](../docs/experiments/), результаты в
[`results/`](../results/README.md), порядок запуска описан в [`docs/reproduce.md`](../docs/reproduce.md).

## Данные

| Файл | Назначение |
|---|---|
| [`a2_gen.py`](a2_gen.py) | генератор потока записей: COPY-текст для PostgreSQL и эталонные массивы для проверки ответов |
| [`corpus.py`](corpus.py) | дословная копия генератора корпуса из проекта автора JARVIS-LM |
| [`prepare_data.py`](prepare_data.py) | входные файлы экспериментов (`dyn.copy` и др.) со сверкой по sha256 |
| [`verify_corpus.py`](verify_corpus.py) | повторная генерация потока и сверка с [`results/data_manifest.sha256`](../results/data_manifest.sha256) |
| `data/` | создаваемые файлы (адреса, эталонные массивы), в git не входят |

## Тесты

| Файл | Проверка | База |
|---|---|---|
| [`a2_selftest_core.py`](a2_selftest_core.py) | ядро адреса | не нужна |
| [`a5_selftest_lsm.py`](a5_selftest_lsm.py), [`a6_selftest_lsm.py`](a6_selftest_lsm.py) | LSM адресов, слияние шагами, рестарты посреди раунда, мутанты | не нужна |
| [`rcm_selftest.py`](rcm_selftest.py) | API `rcm.py` | не нужна |
| [`rcm_pgtest.py`](rcm_pgtest.py) | API `rcm.py` на собственной таблице | нужна, с правом создавать таблицы |
| `python aN_run.py selfcheck` | самопроверки экспериментов со сценариями и мутантами | нужна, тестовый кластер |

Комментарии в коде экспериментов на английском языке: файлы опубликованы в том виде, в каком шли прогоны.
