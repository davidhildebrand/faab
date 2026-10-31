"""Network-free: list every date cell in a workbook that is not in 2026 (wrong-year entries such as 20-10-26 typed as
d-m-y), and every date-column cell holding text. Usage: date_scan.py BOOK.xlsx"""
import datetime
import sys

import openpyxl

wb = openpyxl.load_workbook(sys.argv[1])
for ws in wb.worksheets:
    hdr = {}
    for row in ws.iter_rows():
        names = [str(c.value).strip() if c.value is not None else "" for c in row]
        if "xyz" in names:
            hdr = {c.column: n for c, n in zip(row, names) if n}
            continue
        for c in row:
            name = hdr.get(c.column, "")
            if isinstance(c.value, datetime.datetime) and c.value.year != 2026:
                print(f"{ws.title} {c.coordinate} ({name}): {c.value:%Y-%m-%d}; row proofreader "
                      f"{next((x.value for x in row if hdr.get(x.column) == 'proofreader'), '')}")
            elif name.endswith("date") and isinstance(c.value, str) and c.value.strip():
                print(f"{ws.title} {c.coordinate} ({name}): text {c.value!r}")
