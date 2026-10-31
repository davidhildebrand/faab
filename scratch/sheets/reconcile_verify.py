"""Offline check of a reconciled workbook against its input (2026-10-31): per tab, data rows and hyperlinks before / after;
cells (root_id_curr) on several rows whose status / proofreader / notes still differ (yellow MISMATCH rows left out);
the PRs-tab changes from the sync log. No CAVE. Usage: reconcile_verify.py BEFORE.xlsx AFTER.xlsx CHANGES.tsv"""
import re
import sys

import openpyxl
import pandas as pd

from faab.partners import reconcile as R
from faab.partners.sheet import cell_text as s, scan_book

HL = "FFFFF9C4"


def summary(path):
    wb = openpyxl.load_workbook(path)
    rows = [x for x in scan_book(wb) if not re.search(r"_b\d{6}d$", x["tab"])]
    links = {ws.title: sum(1 for row in ws.iter_rows() for c in row if c.hyperlink) for ws in wb.worksheets}
    return wb, rows, links


wb0, r0, l0 = summary(sys.argv[1])
wb1, r1, l1 = summary(sys.argv[2])
for tab in sorted({x["tab"] for x in r0}):
    print(f"{tab}: rows {sum(x['tab'] == tab for x in r0)} -> {sum(x['tab'] == tab for x in r1)}; hyperlinks {l0.get(tab)} -> {l1.get(tab)}")
yellow = lambda x: any(wb1[x["tab"]].cell(x["row"], c).fill is not None and wb1[x["tab"]].cell(x["row"], c).fill.fill_type == "solid"
                       and wb1[x["tab"]].cell(x["row"], c).fill.fgColor.rgb == HL for c in x["cols"].values())
by = {}
for x in r1:
    c = s(x["values"].get("root_id_curr"))
    if c and "status" in x["cols"] and not yellow(x):
        by.setdefault(c, []).append(x)
bad = 0
for c, g in by.items():
    if len(g) < 2 or not any(s(x["values"].get("status")).lower() not in ("", "new", "new-dup", "candidate") for x in g):
        continue
    shared = lambda t: "; ".join(p for p in R.parts(t) if not R.OWN.search(p)).lower()  # row-own notes may differ
    diff = [f for f in ("status", "proofreader", "notes")
            if len({(shared if f == "notes" else str.lower)(s(x["values"].get(f))) for x in g}) > 1]
    if diff:
        bad += 1
        if bad <= 10:
            print(f"  still differs {c} {diff}: " + "; ".join(f"{x['tab']} r{x['row']} [{s(x['values'].get('status'))}] "
                                                          f"'{s(x['values'].get('notes'))[:50]}'" for x in g))
print(f"cells with a status on several rows that still differ (PRs notes ignored): {bad}")
L = pd.read_csv(sys.argv[3], sep="\t", dtype=str).fillna("")
P = L[L.tab == "PRs"]
print(f"\nPRs changes ({len(P)}):")
for x in P.to_dict("records"):
    print(f"  r{x['row']} {x['column']}: '{x['old'][:60]}' -> '{x['new'][:60]}'")
