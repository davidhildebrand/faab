"""Survey of a PRs tab (.xlsx): per group its section, rows, classes and status; the section lines in column A; where
given roots sit. Network-free. Usage: prs_tab_survey.py BOOK [ROOT ...]"""
import sys
from collections import Counter

from faab.pr import prs_tab as T

wb, ws = T.load(sys.argv[1])
want = set(sys.argv[2:])
print("column-A lines that are not group / header / member rows:")
for r in range(1, ws.max_row + 1):
    a = T.text(ws, r, 1)
    if a and not a.isdigit() and not a.startswith("group ") and a != "root_id_orig":
        print(f"  row {r}: {a[:100]}")
bl = T.blocks(ws)
print(f"\n{len(bl)} groups")
for b in bl:
    cls = Counter(T.get(ws, b, r, "class") for r in b.rows)
    st = Counter(T.get(ws, b, r, "status") or "-" for r in b.rows)
    print(f"group {b.group:>3} {b.section:<7} title row {b.title_row:>4}: {len(b.rows)} rows; "
          f"{dict(cls)}; status {dict(st)}")
    for r in b.rows:
        hit = {T.get(ws, b, r, "root_id_orig"), T.get(ws, b, r, "root_id_curr")} & want
        if hit:
            print(f"    row {r}: {', '.join(hit)} class {T.get(ws, b, r, 'class')}; notes {T.get(ws, b, r, 'notes')[:160]}")
print("\nheader of the first leads group:", list(next(b for b in bl if b.section == "leads").cols))
