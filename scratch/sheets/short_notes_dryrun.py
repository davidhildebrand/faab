"""Network-free dry run of faab.partners.short_notes on a workbook: per tab, cells changed and characters saved; every
distinct rule outcome (part -> short form) with counts; and the parts KEPT verbatim in the PRs tab's leads section
(groups 9+), to spot generated text the rules miss. Usage: short_notes_dryrun.py BOOK.xlsx"""
import re
import sys
from collections import Counter, defaultdict

import openpyxl

from faab.partners.sheet import cell_text as s, scan_book
from faab.partners.short_notes import shorten, split_parts

wb = openpyxl.load_workbook(sys.argv[1])
rows = scan_book(wb)
by_orig = defaultdict(list)
for x in rows:
    by_orig[(x["tab"], s(x["values"].get("root_id_orig")))].append(x["row"])
mask = lambda t: re.sub(r"[-+]?\d+(\.\d+)?", "#", re.sub(r"\d{15,}", "<id>", t))[:100]
outcomes, kept_prs, per_tab = Counter(), Counter(), defaultdict(lambda: [0, 0, 0])
unresolved = Counter()
for x in rows:
    old = s(x["values"].get("notes"))
    if not old:
        continue

    def row_of(orig, tab=x["tab"]):
        r = by_orig.get((tab, orig))
        if r:
            return f"r{r[0]}"
        hit = next(((t, rr) for (t, o), rr in by_orig.items() if o == orig), None)
        return f"{hit[0]} r{hit[1][0]}" if hit else None

    new, changed = shorten(old, row_of)
    for p, q in changed:
        outcomes[(mask(p), mask(q) if q else "(dropped)")] += 1
    for p in split_parts(old):
        if re.match(r"(duplicate target of|merged into) \d{15,} \(", p) and p in new:
            unresolved[p] += 1
    if new != old:
        per_tab[x["tab"]][0] += 1
        per_tab[x["tab"]][1] += len(old)
        per_tab[x["tab"]][2] += len(new)
    if x["tab"] == "PRs" and x["row"] > 96:
        for p in split_parts(new):
            if (p, "") not in [(a, "") for a, _ in changed]:
                kept_prs[mask(p)] += 1
for t, (n, a, b) in per_tab.items():
    print(f"{t}: {n} cells shortened, {a} -> {b} characters")
print("\nrule outcomes:")
for (p, q), n in outcomes.most_common():
    print(f"  {n:4d}  {p}  ->  {q}")
print("\nunresolved row references (kept as written):", dict(unresolved))
print("\nPRs leads: parts in the new notes:")
for k, n in kept_prs.most_common():
    print(f"  {n:4d}  {k}")
