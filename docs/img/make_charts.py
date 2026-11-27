"""Графики для README и docs/results.md: SVG в светлом и тёмном вариантах (docs/img/*-light.svg, *-dark.svg).

    python docs/img/make_charts.py

Числа взяты из отчётов (источник указан у каждого набора). Значения для 100 млн и 1 млрд строк являются расчётом:
размер адреса по формуле формата JLMADR01, число обращений B-tree по его высоте (см. docs/results.md).
Цвета: проверенная палитра (категориальная и порядковая), валидатор палитры прошёл в обеих темах.
"""

from __future__ import annotations

import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "rcm": "#2a78d6", "btree": "#eb6834", "hash": "#1baf7a",
              "n1": "#6da7ec", "n10": "#1c5cab"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "rcm": "#3987e5", "btree": "#d95926", "hash": "#199e70",
             "n1": "#256abf", "n10": "#86b6ef"},
}


def num(v: float, digits: int = 2) -> str:
    return f"{v:.{digits}f}".replace(".", ",")


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=12, fill="#000", anchor="start", weight=400) -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}">{esc(s)}</text>')


def column(x, y_base, w, h, fill) -> str:
    """Столбец со скруглённым верхним краем (4 px) и прямым основанием."""
    r = min(4.0, w / 2, h)
    top = y_base - h
    return (f'<path d="M{x:.1f},{y_base:.1f} V{top + r:.1f} Q{x:.1f},{top:.1f} {x + r:.1f},{top:.1f} '
            f'H{x + w - r:.1f} Q{x + w:.1f},{top:.1f} {x + w:.1f},{top + r:.1f} V{y_base:.1f} Z" fill="{fill}"/>')


def legend(items, x, y, t) -> str:
    out, cx = [], x
    for label, color, kind in items:
        if kind == "swatch":
            out.append(f'<rect x="{cx}" y="{y - 9}" width="10" height="10" rx="2" fill="{color}"/>')
        elif kind == "filled":
            out.append(f'<circle cx="{cx + 5}" cy="{y - 4}" r="4" fill="{color}"/>')
        elif kind == "hollow":
            out.append(f'<circle cx="{cx + 5}" cy="{y - 4}" r="3.5" fill="{t["surface"]}" stroke="{color}" stroke-width="2"/>')
        out.append(text(cx + 16, y, label, 12, t["ink2"]))
        cx += 16 + 7.2 * len(label) + 18
    return "".join(out)


def frame(width, height, t, body, title, subtitle) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            f'<rect width="{width}" height="{height}" rx="8" fill="{t["surface"]}"/>'
            + text(20, 28, title, 15, t["ink"], weight=600) + text(20, 47, subtitle, 12, t["ink2"]) + body + "</svg>")


def grouped_columns(t, groups, series, values, labels, ymax, ticks, title, subtitle, height=340, width=720,
                    plot_top=96) -> str:
    """groups: подписи групп (строки через \\n); series: [(имя, ключ цвета)]; values[g][s]; labels[g][s]."""
    left, right, bottom = 56, 20, height - 54
    plot_w = width - left - right
    body = [legend([(name, t[c], "swatch") for name, c in series], 20, 74, t)]
    for v in ticks:
        y = bottom - (bottom - plot_top) * v / ymax
        body.append(f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="{t["grid"]}" stroke-width="1"/>')
        body.append(text(left - 8, y + 4, num(v, 0), 11, t["muted"], "end"))
    body.append(f'<line x1="{left}" x2="{width - right}" y1="{bottom}" y2="{bottom}" stroke="{t["axis"]}" stroke-width="1"/>')
    band = plot_w / len(groups)
    bw, gap = 24, 2
    for gi, g in enumerate(groups):
        n = len(series)
        gx = left + band * gi + (band - (n * bw + (n - 1) * gap)) / 2
        for si, (_, c) in enumerate(series):
            v = values[gi][si]
            h = (bottom - plot_top) * v / ymax
            x = gx + si * (bw + gap)
            body.append(column(x, bottom, bw, h, t[c]))
            body.append(text(x + bw / 2, bottom - h - 6, labels[gi][si], 11, t["ink2"], "middle"))
        for li, line in enumerate(g.split("\n")):
            body.append(text(left + band * gi + band / 2, bottom + 18 + li * 14, line, 12, t["ink2"], "middle"))
    return frame(width, height, t, "".join(body), title, subtitle)


# ------------------------------------------------------------------ расчёт для 100 млн и 1 млрд строк
ROWS_PER_PAGE = 1_000_000 / 76_232          # A2: 76 232 страницы кучи на 1 млн строк
P_BITS = 3.3                                 # каскад MPHF вместе с опорами ранга (STAGE1, A2)


def rcm_bytes(n: float) -> float:
    """Байт на запись: (каскад + отпечаток 8 бит + номер страницы) / 8, плюс поправка, измеренная на 10 млн (A2)."""
    r = math.ceil(math.log2(n / ROWS_PER_PAGE))
    base = (P_BITS + 8 + r) / 8
    r10 = math.ceil(math.log2(10_000_000 / ROWS_PER_PAGE))
    return base + (3.94 - (P_BITS + 8 + r10) / 8)


def btree_lookup_buffers(n: float, leaf=260, fanout=290) -> int:
    """Обращений на поиск = высота B-tree + 1 страница кучи; ключ uuid, лист ≈260 ключей, внутренняя ≈290 ссылок."""
    pages, height = math.ceil(n / leaf), 1
    while pages > 1:
        pages, height = math.ceil(pages / fanout), height + 1
    return height + 1


def projection(t) -> str:
    width, height = 780, 380
    ns = [1e6, 1e7, 1e8, 1e9]
    xl = ["1 млн", "10 млн", "100 млн", "1 млрд"]
    measured = 2  # первые две точки являются замером
    panels = [
        ("Байт на запись", 40, [0, 10, 20, 30, 40],
         [("RCM", "rcm", [3.57, 3.94] + [rcm_bytes(n) for n in ns[2:]]),
          ("B-tree", "btree", [31.56, 31.54, 31.5, 31.5])], 1),
        ("Обращений к буферам на поиск", 6, [0, 2, 4, 6],
         [("RCM", "rcm", [1, 1, 1, 1]),
          ("B-tree", "btree", [4, 4] + [btree_lookup_buffers(n) for n in ns[2:]])], 0),
    ]
    body = [legend([("RCM", t["rcm"], "swatch"), ("B-tree", t["btree"], "swatch"),
                    ("замер", t["ink2"], "filled"), ("расчёт", t["ink2"], "hollow")], 20, 74, t)]
    pw, top, bottom = 300, 120, height - 70
    for pi, (ptitle, ymax, ticks, series, digits) in enumerate(panels):
        left = 56 + pi * (pw + 84)
        body.append(text(left, top - 14, ptitle, 13, t["ink"], weight=600))
        for v in ticks:
            y = bottom - (bottom - top) * v / ymax
            body.append(f'<line x1="{left}" x2="{left + pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="{t["grid"]}" stroke-width="1"/>')
            body.append(text(left - 8, y + 4, num(v, 0), 11, t["muted"], "end"))
        body.append(f'<line x1="{left}" x2="{left + pw}" y1="{bottom}" y2="{bottom}" stroke="{t["axis"]}" stroke-width="1"/>')
        xs = [left + 24 + i * (pw - 48) / 3 for i in range(4)]
        for i, lab in enumerate(xl):
            body.append(text(xs[i], bottom + 18, lab, 12, t["ink2"], "middle"))
        for name, c, vals in series:
            ys = [bottom - (bottom - top) * v / ymax for v in vals]
            solid = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs[:measured], ys[:measured]))
            dashed = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs[measured - 1:], ys[measured - 1:]))
            body.append(f'<polyline points="{solid}" fill="none" stroke="{t[c]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
            body.append(f'<polyline points="{dashed}" fill="none" stroke="{t[c]}" stroke-width="2" stroke-dasharray="5 4" stroke-linecap="round"/>')
            for i, (x, y) in enumerate(zip(xs, ys)):
                if i < measured:
                    body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{t[c]}" stroke="{t["surface"]}" stroke-width="2"/>')
                else:
                    body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{t["surface"]}" stroke="{t[c]}" stroke-width="2"/>')
            body.append(text(xs[-1] + 10, ys[-1] + 4, num(vals[-1], digits), 11, t["ink2"]))
    body.append(text(20, height - 30, "1 и 10 млн строк: замер (A2-A6). 100 млн и 1 млрд: расчёт по формуле размера адреса и высоте B-tree", 11, t["muted"]))
    body.append(text(20, height - 14, "(ключ uuid, около 260 ключей на лист и 290 ссылок во внутренней странице), не замер.", 11, t["muted"]))
    return frame(width, height, t, "".join(body), "Ожидания при росте таблицы",
                 "Размер адреса растёт на доли байта, число обращений RCM не зависит от высоты дерева")


def build() -> None:
    for mode, t in THEMES.items():
        charts = {
            # A2, static: bytes per row (a2_report §3, §4)
            "size": grouped_columns(
                t, ["1 млн строк", "10 млн строк"],
                [("RCM", "rcm"), ("B-tree", "btree"), ("Хеш-индекс", "hash")],
                [[3.566, 31.56, 33.57], [3.940, 31.54, 26.85]],
                [["3,57", "31,56", "33,57"], ["3,94", "31,54", "26,85"]],
                40, [0, 10, 20, 30, 40], "Размер адреса, байт на запись",
                "Статический адрес RCM против индексов PostgreSQL на тех же таблицах (A2)"),
            # A2 static: exactly 1.000 vs 4.000; dynamics A4-A6: RCM at most 1.02 (1M) / 1.04 (10M), B-tree 4.00-4.06
            "buffers": grouped_columns(
                t, ["Без изменений\n1 млн", "Без изменений\n10 млн", "Поток изменений\n1 млн", "Поток изменений\n10 млн"],
                [("RCM", "rcm"), ("B-tree", "btree")],
                [[1.0, 4.0], [1.0, 4.0], [1.02, 4.0], [1.04, 4.0]],
                [["1,00", "4,00"], ["1,00", "4,00"], ["≤1,02", "4,00"], ["≤1,04", "4,00"]],
                5, [0, 1, 2, 3, 4, 5], "Обращений к буферам PostgreSQL на поиск существующего ключа",
                "Без изменений: A2. Поток изменений: наибольшее значение RCM по точкам замера A4-A6", height=360),
            # longest maintenance pause: a4/a5/a6 reports
            "pauses": grouped_columns(
                t, ["A4: одно большое слияние", "A5: слияние шагами", "A6: шаг = партиция"],
                [("1 млн строк", "n1"), ("10 млн строк", "n10")],
                [[5.0, 18.5], [1.13, 0.40], [0.62, 0.36]],
                [["5,0", "18,5"], ["1,13", "0,40"], ["0,62", "0,36"]],
                20, [0, 5, 10, 15, 20], "Самая длинная пауза обслуживания адреса, секунды",
                "Обслуживание изменяющихся данных: от одного большого слияния к шагам по партициям"),
            "projection": projection(t),
        }
        for name, svg in charts.items():
            (OUT / f"{name}-{mode}.svg").write_text(svg, encoding="utf-8")
    print("charts written to", OUT)
    print("RCM bytes/row: 100M", round(rcm_bytes(1e8), 2), "1B", round(rcm_bytes(1e9), 2),
          "| B-tree buffers: 1M", btree_lookup_buffers(1e6), "10M", btree_lookup_buffers(1e7),
          "100M", btree_lookup_buffers(1e8), "1B", btree_lookup_buffers(1e9))


if __name__ == "__main__":
    build()
