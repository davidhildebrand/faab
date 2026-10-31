"""Header columns of every tracking tab (distinct header rows per tab, with the number of rows under each kind), to see
which columns hold the upstream cell and nsyn (2026-10-31, reconciliation). Usage: tab_headers.py BOOK.xlsx"""
import re
import sys
from collections import Counter

import openpyxl

from faab.partners.sheet import scan_book

wb = openpyxl.load_workbook(sys.argv[1])
kinds = Counter()
for x in scan_book(wb):
    if not re.search(r"_b\d{6}d$", x["tab"]):
        kinds[(x["tab"], tuple(x["cols"]))] += 1
for (tab, cols), n in sorted(kinds.items()):
    print(f"{tab} ({n} rows): {', '.join(cols)}")
