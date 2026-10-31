"""Cells that the PRs tab lists twice - a 'candidate' row (lead / L-side groups) and a row elsewhere with a status, or two
PRs rows (2026-10-31, from the cell_consistency.py report). Prints each row's group (nearest 'group N' line above it) and
section (the nearest '>>>' line above). Usage: prs_candidate_dups.py CONSISTENCY.tsv BOOK.xlsx"""
import sys

import openpyxl
import pandas as pd

from faab.partners.sheet import cell_text as s

D = pd.read_csv(sys.argv[1], sep="\t", dtype=str).fillna("")
ws = openpyxl.load_workbook(sys.argv[2])["PRs"]
col_a = [s(ws.cell(r, 1).value) for r in range(1, ws.max_row + 1)]


def where(r):
    grp = next((col_a[i] for i in range(r - 1, -1, -1) if col_a[i].startswith("group ")), "")
    sec = next((col_a[i][:40] for i in range(r - 1, -1, -1) if col_a[i].startswith(">>>") or col_a[i].startswith("NOT PROOF")), "")
    return f"{grp} / {sec}"


cells = sorted({c for c, g in D[D.n_rows.astype(int) > 1].groupby("cell") if (g.tab == "PRs").any() and len(g) > 1
                and ((g.tab == "PRs").sum() > 1 or (g.status.str.lower() == "candidate").any())})
for c in cells:
    g = D[D.cell == c]
    print(c)
    for x in g.to_dict("records"):
        loc = where(int(x["row"])) if x["tab"] == "PRs" else ""
        print(f"   {x['tab']} r{x['row']} [{x['status']}] {loc}  '{x['notes'][:50]}'")
