"""Network-free: rows of each tab with solid fills, by colour (to find highlighted rows). Usage: fill_scan.py BOOK.xlsx"""
import sys
from collections import defaultdict

import openpyxl

wb = openpyxl.load_workbook(sys.argv[1])
for ws in wb.worksheets:
    by = defaultdict(set)
    for row in ws.iter_rows():
        for c in row:
            if c.fill is not None and c.fill.fill_type == "solid" and isinstance(c.fill.fgColor.rgb, str):
                by[c.fill.fgColor.rgb].add(c.row)
    for col, rows in sorted(by.items(), key=lambda kv: -len(kv[1])):
        rs = sorted(rows)
        print(f"{ws.title} {col}: {len(rs)} rows: {rs[:40]}{' ...' if len(rs) > 40 else ''}")
