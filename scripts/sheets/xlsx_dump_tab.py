"""Print one tab of an .xlsx as rows (dates yy-mm-dd, whole floats as ints), with each cell's fill noted, so the tab's
layout can be read. Usage: xlsx_dump_tab.py BOOK.xlsx TAB [MAX_ROWS]"""
import datetime
import sys

import openpyxl

ws = openpyxl.load_workbook(sys.argv[1])[sys.argv[2]]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 200


def show(v):
    if v is None:
        return ""
    if isinstance(v, datetime.datetime):
        return f"{v:%y-%m-%d}"
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else f"{v:g}"
    return str(v)


print(f"{ws.title}: {ws.max_row} rows x {ws.max_column} cols")
for r, row in enumerate(ws.iter_rows(max_row=n), 1):
    vals = [show(c.value) for c in row]
    if not any(vals):
        print(f"{r:4d} |")
        continue
    fills = {c.column_letter: c.fill.fgColor.rgb for c in row
             if c.fill is not None and c.fill.fill_type == "solid" and isinstance(c.fill.fgColor.rgb, str)}
    links = [c.column_letter for c in row if c.hyperlink]
    print(f"{r:4d} | " + " | ".join(vals).rstrip(" |") + (f"   [fill {fills}]" if fills else "")
          + (f"   [links {links}]" if links else ""))
