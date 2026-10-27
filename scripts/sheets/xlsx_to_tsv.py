"""Network-free: export tabs of an .xlsx as block TSVs the way Google Sheets exports them (dates yy-mm-dd, whole floats as
ints, ids as text, empty cells blank, every row padded to the tab width). Usage: xlsx_to_tsv.py BOOK.xlsx OUTDIR TAB [TAB ...]"""
import datetime
import os
import sys

import openpyxl

book, out = sys.argv[1], sys.argv[2]
wb = openpyxl.load_workbook(sys.argv[1])
base = os.path.splitext(os.path.basename(book))[0]


def show(v):
    if v is None:
        return ""
    if isinstance(v, datetime.datetime):
        return f"{v:%y-%m-%d}"
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else f"{v:g}"
    return str(v)


for tab in sys.argv[3:]:
    ws = wb[tab]
    rows = [[show(c.value) for c in row] for row in ws.iter_rows()]
    while rows and not any(rows[-1]):
        rows.pop()
    path = os.path.join(out, f"{base}-{tab}.tsv")
    open(path, "w", encoding="utf-8").write("\n".join("\t".join(r) for r in rows) + "\n")
    data = sum(1 for r in rows if r and r[0].strip().isdigit())
    print(f"{tab}: {len(rows)} lines, {ws.max_column} columns, {data} data lines -> {path}")
