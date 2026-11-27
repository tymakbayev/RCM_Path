"""Print the key numbers of results/a2_static_<n>.json (no computation beyond reading)."""
import json
import sys
from pathlib import Path

d = json.loads((Path(__file__).parent.parent / "results" / f"a2_static_{sys.argv[1]}.json").read_text(encoding="utf-8"))
print("rows", d["rows"], "pages", d["table"]["pages"], "rows/page %.2f" % d["table"]["rows_per_page"],
      "toast", d["table"]["toast_rows"])
print("A build:", {k: d["A"]["build"][k] for k in ("bytes_per_row", "levels", "exceptions", "c1_errors", "r_bits",
                                                   "read_s", "build_s", "keys_match_ref")})
print("SORT", d["SORT"], "EF %.3f" % d["EF"]["bytes_per_row"])
for k in ("btree", "hash"):
    print(k, "B/row %.2f total %.2f create %.1fs" % (d[k]["bytes_per_row"], d[k]["total_bytes_per_row"], d[k]["create_s"]),
          d[k]["size"]["reloptions"])
print("B build:", {k: d["B"]["build"][k] for k in ("bytes_per_row", "b", "misplaced_rows", "c1_violations",
                                                   "pages_per_bucket", "load_s")})
for k in ("A", "B", "btree", "hash"):
    m = d[k]["measure"]
    e, ea = m["explain_present"], m["explain_absent"]
    print(f"{k:6s} C2 err {m['c2_errors']} C3 err {m['c3_errors']} | client p50/p95/p99 "
          f"{m['client_us']['p50']:.0f}/{m['client_us']['p95']:.0f}/{m['client_us']['p99']:.0f} us"
          + (f" addr p50 {m['address_us']['p50']:.1f} us" if m["address_us"] else "")
          + f" | exec p50 {e['exec_us']['p50']:.1f} us p99 {e['exec_us']['p99']:.1f} | buffers {e['buffers_mean']:.3f}"
          f" (max {e['buffers_max']}) | statio heap {m['statio_present']['heap_per_q']:.3f} idx "
          f"{m['statio_present']['index_per_q']:.3f} toast {m['statio_present']['toast_per_q']:.3f} | absent heap "
          f"{m['statio_absent']['heap_per_q']:.4f} idx {m['statio_absent']['index_per_q']:.3f} nosql "
          f"{ea['no_sql_fraction']:.3f}")
    print("        plan", e["plan"])
    if "restart" in d[k]:
        r = d[k]["restart"]
        print("        restart first buffers", [x["first_buffers"] for x in r], "read", [x["first_read"] for x in r],
              "next200 read mean", [round(x["next200_read_mean"], 2) for x in r])
    if "open_probe" in d[k]:
        o = d[k]["open_probe"]
        print("        open ms p50 %.2f max %.2f | first answer p50 %.2f | connect p50 %.1f | pages %s" % (
            o["open_ms"]["p50"], o["open_ms"]["max"], o["first_answer_ms"]["p50"], o["connect_ms"]["p50"],
            o["sidecar_pages_touched"]))
if "BRIN" in d:
    for p in ("ppr1", "ppr4"):
        b = d["BRIN"][p]
        m = b["measure"]
        print(f"BRIN {p}: B/row {b['bytes_per_row']:.3f} buffers {m['explain_present']['buffers_mean']:.1f} heap "
              f"{m['statio_present']['heap_per_q']:.2f} idx {m['statio_present']['index_per_q']:.2f} exec p50 "
              f"{m['explain_present']['exec_us']['p50']:.0f} us C2 err {m['c2_errors']} plan {m['explain_present']['plan']}")
