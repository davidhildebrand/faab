"""Where ids appear in workbooks (every tab, every cell, as text): tab, cell, the row's first cells.
Network-free. Usage: find_in_book.py BOOK.xlsx [BOOK2.xlsx ...] -- ID [ID ...]"""
import sys

import openpyxl

args = sys.argv[1:]
books, ids = args[:args.index("--")], args[args.index("--") + 1:]
for path in books:
    wb = openpyxl.load_workbook(path, read_only=True)
    hits = 0
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for c in row:
                v = c.value
                if v is not None and any(i in str(v) for i in ids):
                    hits += 1
                    head = [str(x.value) for x in row[:3] if x.value is not None]
                    print(f"{path.rsplit('/', 1)[-1]} | {ws.title} | {c.coordinate}: {str(v)[:60]} | row starts {' / '.join(head)[:80]}")
    if not hits:
        print(f"{path.rsplit('/', 1)[-1]}: not found")
