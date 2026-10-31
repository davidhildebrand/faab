"""Every row of the given cells in a workbook (root_id_curr or root_id_orig match), with status, names, dates, notes,
SeaTable, link (and whether the link cell is a hyperlink), to check a reconciled export (2026-10-31).
Usage: cells_rows.py BOOK.xlsx CELL [CELL ...]"""
import re
import sys

import openpyxl

from faab.partners.sheet import cell_text as s, scan_book

wb = openpyxl.load_workbook(sys.argv[1])
want = set(sys.argv[2:])
for x in scan_book(wb):
    if re.search(r"_b\d{6}d$", x["tab"]):
        continue
    v = x["values"]
    ids = {s(v.get("root_id_curr")), s(v.get("root_id_orig"))}
    if not ids & want:
        continue
    lc = x["cols"].get("link")
    hl = bool(lc and wb[x["tab"]].cell(x["row"], lc).hyperlink)
    print(f"{x['tab']} r{x['row']} {s(v.get('root_id_curr'))} nsyn={s(v.get('nsyn'))} [{s(v.get('status'))}] "
          f"pr={s(v.get('proofreader'))} {s(v.get('pr_date'))} chk={s(v.get('checker'))} {s(v.get('chk_date'))} "
          f"notes='{s(v.get('notes'))}' sea='{s(v.get('SeaTable'))}' link={'yes' if hl else s(v.get('link'))[:20]}"
          + (f" tracked_in='{s(v.get('tracked_in'))}'" if 'tracked_in' in v else ""))
