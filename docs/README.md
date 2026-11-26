# Method

[`LLM_engineering_research_T.A.Astana2026.pdf`](LLM_engineering_research_T.A.Astana2026.pdf): the author's essay on
how this research was done (in Russian). Its title translates as "From 'nobody does it this way' to an executable
hypothesis: how to use several LLMs for engineering research". It is published unchanged, in the edition of
2026-10-03. sha256 `2639e1524e75e06637198d80815e60658cc269e4b655a1292740d8ed2b8e2f01`.

The essay describes these practices:

- **Assume → Build → Break**: assume the required properties are achievable, build an honest prototype, then try to
  refute it.
- **Models propose, the human routes, experiments decide.** Several models (GPT, Claude, Gemini, Qwen, DeepSeek) were
  orchestrated by hand.
- **Negative results are part of the knowledge.**

It also covers where the method is dangerous: biased selection by the human, false consensus of models, unsuitable
comparisons, and shared errors of the code and of its test.

Its worked example is the earlier, content-compression stage of the same project (JARVIS-LM experiments E20–E23). The
E20–E23 numbers come from the results of that time, which are not part of this repository. The address engine
(experiments A2–A8) came later and follows the same practice: frozen protocols, a second model as reviewer, and
failures reported as failures.

Copyright (c) 2026 Abu (github.com/tymakbayev).

---

# Метод

`LLM_engineering_research_T.A.Astana2026.pdf` — эссе автора о том, как велось это исследование: «От "так никто не
делает" к исполняемой гипотезе: как использовать несколько LLM для инженерных исследований». Публикуется без
изменений, в редакции от 2026-10-03.

Принципы, которые в нём описаны:

- Assume → Build → Break;
- модели предлагают, человек направляет, эксперимент решает;
- ручная оркестрация разнородных моделей;
- отрицательные результаты — часть знания.

Там же разобрано, где метод опасен.

Сквозной пример эссе — более ранний этап проекта: сжатие содержимого, эксперименты JARVIS-LM E20–E23. Их числа взяты
из результатов того времени, которых в этом репозитории нет. Адресный движок (A2–A8) сделан позже по той же практике.
