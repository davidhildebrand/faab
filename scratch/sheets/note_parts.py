"""Network-free: the '; '-separated parts of every notes cell in a workbook's tracking tabs, grouped by a pattern
(ids, numbers and dates masked), with counts and one example each - to see which parts are generated text.
Usage: note_parts.py BOOK.xlsx [TAB ...]"""
import re
import sys
from collections import Counter, defaultdict

import openpyxl

from faab.partners.sheet import cell_text as s, scan_book

wb = openpyxl.load_workbook(sys.argv[1])
tabs = set(sys.argv[2:]) or None
cnt, ex, where = Counter(), {}, defaultdict(Counter)
for x in scan_book(wb):
    if tabs and x["tab"] not in tabs:
        continue
    for part in (p.strip() for p in s(x["values"].get("notes")).split(";")):
        if not part:
            continue
        key = re.sub(r"\d{15,}", "<id>", part)
        key = re.sub(r"[-+]?\d+(\.\d+)?", "#", key)[:110]
        cnt[key] += 1
        ex.setdefault(key, f"{x['tab']} r{x['row']}: {part[:160]}")
        where[key][x["tab"]] += 1
for key, n in cnt.most_common():
    print(f"{n:4d}  {key}\n        {dict(where[key])}  e.g. {ex[key]}")
