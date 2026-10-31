"""For some rows of one tab: the block each sits in (nearest title / '>>>' line above its header row) and the row's key
columns, to tell same-block twins from rows in different blocks (2026-10-31, PRg1hop2 r162 vs the other L4 rows).
Usage: row_blocks.py BOOK.xlsx TAB ROW [ROW ...]"""
import sys

import openpyxl

from faab.partners.sheet import cell_text as s, scan_book

wb = openpyxl.load_workbook(sys.argv[1])
tab, want = sys.argv[2], [int(r) for r in sys.argv[3:]]
ws = wb[tab]
rows = {x["row"]: x for x in scan_book(wb) if x["tab"] == tab}
for r in want:
    x = rows[r]
    hdr = next(rr for rr in range(r, 0, -1) if s(ws.cell(rr, x["cols"]["xyz"]).value) == "xyz")
    title = next((s(ws.cell(rr, 1).value) for rr in range(hdr - 1, 0, -1) if s(ws.cell(rr, 1).value)
                  and not s(ws.cell(rr, 1).value).isdigit()), "")
    v = x["values"]
    keys = [k for k in ("upstream_PR", "PR_root_id", "hop1_root_id", "nsyn", "h1_nsyn", "status", "proofreader",
                        "pr_date", "checker", "chk_date", "notes") if k in v]
    print(f"r{r} (header r{hdr}, block '{title[:90]}'): " + "; ".join(f"{k}={s(v[k])}" for k in keys))
