# RCM Path theory

The two documents here are the theoretical basis that the RCM Address Engine grew out of. Both were written for the
author's JARVIS-LM project (long-term memory of a local voice assistant) and are published **unchanged**, as they were
on 2026-10-05.

| File | What it is | sha256 |
|---|---|---|
| `memory_math.md` | Mathematical model of compact reconstructive memory with path-dependent addressing: **generator + trajectory (path) + residual**, x̂ = Unfold_G(p) ⊕ r, with every claim labelled proved / estimate / to be checked | `82047c08b14677f8625b55f26da8322fb025449ac2e54edfc6d92045c8c1da1a` |
| `architecture_breakthroughs.md` | Summary of the measured results of that project (experiments E1–E23) and of what was refuted | `fca4df8d3df5c4dac42ecaa4cda548e3a062683ab98f4264115b9c95b46fdf1f` |

How the address engine relates to it: the theory applies G + P + R to the stored content. In November 2026 the author
moved the same principle to the address (index) layer of an existing database (`rcm_address_engine/STAGE1.md` §2,
`STAGE1_postgresql_branch.md`):

    h = hash64(key, seed)                 G: generated from the key, not stored
    slot = P(h)                           P: cascade of bitmaps (minimal perfect hash), ~3 bits per key
    (fingerprint, page) = D[slot]         R: the residual, a fixed-width field per key (8 + ⌈log2 pages⌉ bits)

Both STAGE1 reports start from `memory_math.md` §19.2c, which concluded that the search index should be derived rather
than stored: a stored 16-byte index per object cut that project's compression ratio from 11.5× to 3.5×. They also
correct the claim "index = 0 bytes on disk". When placement does not depend on the key, an exact map from key to row
position needs about log2(n) − 1.44 bits per record, and a map from key to page about log2(pages) bits
(`STAGE1.md` §2.1).

Both documents are in Russian. They refer to files of the JARVIS project (`design-docs/`, `results/e*.json`,
`src/jarvis_mem/`) that are not part of this repository.

Copyright (c) 2026 Abu (github.com/tymakbayev). MIT License, see `../LICENSE`.

---

# Теория RCM Path

Здесь два документа, из которых вырос адресный движок. Оба написаны для проекта JARVIS-LM (долговременная память
локального голосового ассистента) и публикуются **без изменений**, в состоянии на 2026-10-05.

- `memory_math.md` — математическая модель компактной реконструктивной памяти с path-dependent адресацией:
  **генератор + траектория + остаток**. Каждое утверждение помечено: доказано, оценка или проверить.
- `architecture_breakthroughs.md` — сводка измеренных результатов этого проекта (эксперименты E1–E23) и
  опровергнутого.

Связь с движком: теория применяет G + P + R к содержимому. В ноябре 2026 тот же принцип перенесён на адресный
(индексный) слой существующей СУБД: G — хеш ключа порождается и не хранится; P — каскад битовых карт (минимальная
совершенная хеш-функция), около 3 бит на ключ; R — остаток, поле фиксированной ширины на ключ: отпечаток 8 бит и номер
страницы. Подробнее — `rcm_address_engine/STAGE1.md` §2 и `STAGE1_postgresql_branch.md`. Оба отчёта отталкиваются от
вывода §19.2c `memory_math.md`: индекс поиска лучше выводить, а не хранить (хранимый индекс по 16 байт на объект снижал
коэффициент сжатия того проекта с 11,5× до 3,5×). И уточняют тезис «индекс = 0 байт на диске»: если место записи не
зависит от ключа, точное отображение «ключ → позиция строки» стоит около log2(n) − 1,44 бита на запись, а
«ключ → страница» — около log2(число страниц) бит (`STAGE1.md` §2.1).

Документы ссылаются на файлы проекта JARVIS (`design-docs/`, `results/e*.json`, `src/jarvis_mem/`), которых в этом
репозитории нет.
