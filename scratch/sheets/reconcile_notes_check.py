"""Offline check of faab.partners.reconcile's note rules on the cell_consistency.py TSV (2026-10-31): for every cell on
several rows with a status, the notes each row would carry, next to its notes now. No CAVE. Usage:
reconcile_notes_check.py CONSISTENCY.tsv [CELL ...]"""
import sys

import pandas as pd

from faab.partners import reconcile as R

D = pd.read_csv(sys.argv[1], sep="\t", dtype=str).fillna("")
sizes = {c: int(n) for c, n in zip(D.cell, D.l2_chunks) if n.isdigit()}
want = set(sys.argv[2:])
n_changed = 0
for c, g in D[D.n_rows.astype(int) > 1].groupby("cell"):
    if want and c not in want:
        continue
    g = g.assign(rk=g.status.map(R.rank)).sort_values(["rk", "tab", "row"])
    if g.rk.iloc[0] >= len(R.RANK):
        continue
    status = g.status.iloc[0].lower()
    whole = sizes.get(c, 0) >= R.WHOLE_L2
    shared, seen = [], set()
    for n in g.notes:
        for p in R.parts(n):
            q = R.shared_part(p, whole, status == "checked")
            if q and R.key(q) not in seen:
                seen.add(R.key(q))
                shared.append(q)
    shared = R.dedupe(shared)
    lines = []
    for x in g.to_dict("records"):
        own = [p for p in R.parts(x["notes"]) if R.OWN.search(p) and R.key(p) not in seen]
        new = "; ".join(shared + own)
        if new != x["notes"]:
            lines.append(f"    {x['tab']} r{x['row']} [{x['status']}]: '{x['notes']}' -> '{new}'")
    if lines:
        n_changed += 1
        if want or n_changed <= 25:
            print(f"{c} -> {status}{' (whole)' if whole else ''}")
            print("\n".join(lines))
print(f"\ncells whose notes change: {n_changed}")
