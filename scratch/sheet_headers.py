"""The first lines of every tab of a workbook (titles, header rows), to match a new tab's layout to the existing ones.
Network-free. Usage: sheet_headers.py BOOK.xlsx [N]"""
import sys

import openpyxl

n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
wb = openpyxl.load_workbook(sys.argv[1], read_only=True)
for ws in wb.worksheets:
    print(f"== {ws.title} ({ws.max_row} rows)")
    for k, row in enumerate(ws.iter_rows(values_only=True)):
        if k >= n:
            break
        vals = [str(v) for v in row if v not in (None, "")]
        if vals:
            print("   " + " | ".join(vals)[:230])
