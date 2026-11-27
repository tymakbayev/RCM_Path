# Эталонные результаты

Сырые результаты прогонов A2–A8 на эталонном стенде ([описание стенда](../docs/results.md#эталонный-стенд)): JSON с
полными числами и логи. Сводка — в [`docs/results.md`](../docs/results.md), разбор — в отчётах
[`docs/experiments/`](../docs/experiments/).

| Эксперимент | Файлы | Отчёт |
|---|---|---|
| A2 | `a2_p0.*` (пилот), `a2_selfcheck.*`, `a2_static_n1.*`, `a2_static_n10.*`, `a2_dynamics.*` | [a2_report.md](../docs/experiments/a2_report.md) |
| A3 | `a3_dynamics.*` | [a3_report.md](../docs/experiments/a3_report.md) |
| A4 | `a4_selfcheck.*`, `a4_dynamics_n1.*`, `a4_dynamics_n10.*`, `a4_replay_n1.*`, `a4_replay_n10.*` | [a4_report.md](../docs/experiments/a4_report.md) |
| A5 | `a5_selfcheck.*`, `a5_dynamics_n1.*`, `a5_dynamics_n10.*`, `a5_replay_n1.*`, `a5_replay_n10.*` | [a5_report.md](../docs/experiments/a5_report.md) |
| A6 | `a6_selfcheck.*`, `a6_dynamics_n1.*`, `a6_dynamics_n10.*`, `a6_replay_n1.*`, `a6_replay_n10.*`, `a6_summary.json`; `a6_dynamics_n1_smoke.*` — пробный прогон, в отчёт не входит | [a6_report.md](../docs/experiments/a6_report.md) |
| A7 | `a7_selfcheck.*`, `a7_n1_r1.*`, `a7_n1_r2.*`, `a7_n10_r1.*`; `a7_extract.json` — выжимка для ревью | [a7_report.md](../docs/experiments/a7_report.md) |
| A8 | `a8_selfcheck.*`, `a8_selfcheck_extract.json`; `a8_debug_small.*` — отладочный прогон. Прогон на 10M не выполнен | [A8_protocol.md](../docs/experiments/A8_protocol.md) |
| Демо | `demo_reference_5k.json`, `demo_reference_200k.json` — `demo.py` на эталонном стенде, роль без прав суперпользователя | [README](../README.md#попробовать-за-несколько-минут) |

Служебные файлы:

- `aN_protocol.sha256` — sha256 протоколов, зафиксированные до прогонов; скрипты запуска сверяют с ними протоколы;
- `data_manifest.sha256` — sha256 эталонных потоков записей; по ним сверяют `verify_corpus.py` и `prepare_data.py`.

Ваши прогоны пишут сюда же, с теми же именами файлов: `git diff` покажет ваши числа рядом с нашими. Демо пишет
`demo_result.json`, он в git не входит. Поделиться результатами — issue по шаблону **Результаты**.
