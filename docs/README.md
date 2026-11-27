# Документация

| Документ | О чём |
|---|---|
| [results.md](results.md) | все результаты A2–A8: сводная таблица, что установлено и что нет, условия работы прототипа, стенд |
| [integration.md](integration.md) | подключение к своей таблице: API `rcm.py`, что поддерживается, что означает адрес, статус интеграции |
| [design.md](design.md) | как устроен адрес и как от этапа к этапу решалась динамика |
| [reproduce.md](reproduce.md) | как повторить эксперименты: стенд, настройка, шаги, что проверено после подготовки к публикации |
| [experiments/](experiments/) | протоколы, приложения и отчёты A2–A8 и этапа 1 — в том виде, в каком велись |
| [theory/](theory/README.md) | модель RCM Path «генератор + траектория + остаток», из которой выросла идея |
| [method/](method/README.md) | эссе автора о методе исследования: несколько LLM, эксперимент как арбитр |

## Протоколы и отчёты

| | Протокол | Приложения | Отчёт |
|---|---|---|---|
| Этап 1 | [STAGE1_postgresql_branch.md](experiments/STAGE1_postgresql_branch.md) (действующий), [STAGE1.md](experiments/STAGE1.md) (вариант «своё хранилище») | | |
| A2 | [A2_protocol.md](experiments/A2_protocol.md); черновик [A2_protocol_ownstore_draft.md](experiments/A2_protocol_ownstore_draft.md) | [A2_addendum_2026-11-16.md](experiments/A2_addendum_2026-11-16.md) | [a2_report.md](experiments/a2_report.md) |
| A3 | [A3_protocol.md](experiments/A3_protocol.md) | | [a3_report.md](experiments/a3_report.md) |
| A4 | [A4_protocol.md](experiments/A4_protocol.md) | | [a4_report.md](experiments/a4_report.md) |
| A5 | [A5_protocol.md](experiments/A5_protocol.md) | [A5_addendum_2026-11-16.md](experiments/A5_addendum_2026-11-16.md) | [a5_report.md](experiments/a5_report.md) |
| A6 | [A6_protocol.md](experiments/A6_protocol.md) | [A6_addendum_2026-11-16.md](experiments/A6_addendum_2026-11-16.md) | [a6_report.md](experiments/a6_report.md) |
| A7 | [A7_protocol.md](experiments/A7_protocol.md) | [A7_addendum_2026-11-16.md](experiments/A7_addendum_2026-11-16.md) | [a7_report.md](experiments/a7_report.md) |
| A8 | [A8_protocol.md](experiments/A8_protocol.md) | [A8_addendum_2026-11-17.md](experiments/A8_addendum_2026-11-17.md) | прогон на 10M не выполнен |

Протоколы зафиксированы по sha256 до прогонов (хеши — в [`results/`](../results/README.md)) и не менялись; более
поздние факты — в датированных приложениях. Пути внутри протоколов и отчётов (`results/…`, `rcm_address_engine/…`)
указаны так, как было на стенде: сейчас результаты лежат в [`results/`](../results/README.md) в корне репозитория,
а документы — здесь, в `docs/experiments/`.
