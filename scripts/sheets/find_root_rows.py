"""Rows of a workbook tab that mention a root id anywhere (any column), with the block line above them.
Usage: find_root_rows.py BOOK.xlsx TAB ROOT [ROOT ...]"""
import sys

import openpyxl

from faab.partners.sheet import cell_text as s

wb = openpyxl.load_workbook(sys.argv[1])
ws = wb[sys.argv[2]]
hdr, block = None, ""
for r in range(1, ws.max_row + 1):
    vals = [s(ws.cell(r, c).value) for c in range(1, ws.max_column + 1)]
    if "root_id_orig" in vals:
        hdr = vals
        continue
    if vals and vals[0] and not vals[0].isdigit():
        block = vals[0][:60]
    for root in sys.argv[3:]:
        if any(root in v for v in vals):
            row = {h: v for h, v in zip(hdr or [], vals) if h and h != "link"}
            print(f"{root} in block {block!r}: {row}")
