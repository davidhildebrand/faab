"""Break down the cell_consistency.py disagreements (2026-10-31): for cells listed on several rows, which fields differ and
how (status: one real status vs NEW / blank, or two real statuses; notes: proofreader text vs our row-specific notes).
Usage: consistency_review.py figures/sheets/<stamp>_cell_consistency.tsv"""
import re
import sys

import pandas as pd

D = pd.read_csv(sys.argv[1], sep="\t", dtype=str).fillna("")
D["n_rows"] = D.n_rows.astype(int)
REAL = ["checked", "proofread", "adequate", "review", "to_review", "needs_extending", "in progress", "stopped"]
OURS = re.compile(r"^(duplicate( target)? of|merged into (r|PRg|PRs|Home|Mi1)|xyz off the cell|census bundle|T4/T5 candidate|"
                  r"pair |terminal|split-off piece)", re.I)
M = D[D.n_rows > 1]
cat = {}
for cell, g in M.groupby("cell"):
    st = [x.lower() for x in g.status]
    real = sorted({x for x in st if x in REAL}, key=REAL.index)
    other = sorted({x for x in st if x not in REAL})
    if len(set(st)) == 1:
        k = "same"
    elif len(real) == 1:
        k = f"one status ({real[0]}) + {'/'.join(other) or 'blank'}"
    elif not real:
        k = "no status: " + "/".join(other)
    else:
        k = "several statuses: " + " / ".join(real) + (" + " + "/".join(other) if other else "")
    cat.setdefault(k, []).append(cell)
print(f"{M.cell.nunique()} cells on several rows; status:")
for k, v in sorted(cat.items(), key=lambda kv: -len(kv[1])):
    print(f"  {len(v):3d}  {k}")
show = [c for k, v in cat.items() if k.startswith("several") for c in v]
print("\ncells with several real statuses:")
for cell in show:
    g = M[M.cell == cell]
    print(f"  {cell}: " + "; ".join(f"{t} r{r} [{s}] {p} {d} '{n[:60]}'" for t, r, s, p, d, n in
                                    zip(g.tab, g.row, g.status, g.proofreader, g.pr_date, g.notes)))


def parts(n):
    return {p.strip() for p in re.split(r";\s*", n) if p.strip() and not OURS.search(p.strip())}


nd = []
for cell, g in M.groupby("cell"):
    ps = [frozenset(parts(n)) for n in g.notes]
    if len(set(ps)) > 1:
        nd.append(cell)
print(f"\nnotes: cells whose proofreader notes differ between rows (our row-specific notes ignored): {len(nd)}")
for cell in nd[:12]:
    g = M[M.cell == cell]
    print(f"  {cell}: " + " | ".join(f"{t} r{r}: '{n[:70]}'" for t, r, n in zip(g.tab, g.row, g.notes)))
sea = M.groupby("cell").SeaTable.apply(lambda v: tuple(sorted(set(v))))
sea = sea[sea.map(len) > 1]
print(f"\nSeaTable column differs: {len(sea)}; e.g.")
for cell, v in list(sea.items())[:8]:
    print(f"  {cell}: {v}")
pr = M.groupby("cell").apply(lambda g: g.proofreader.nunique() > 1 or g.pr_date.nunique() > 1)
print(f"\nproofreader or pr_date differs: {int(pr.sum())}; e.g.")
for cell in pr[pr].index[:8]:
    g = M[M.cell == cell]
    print(f"  {cell}: " + "; ".join(f"{t} r{r} [{s}] '{p}' {d}" for t, r, s, p, d in zip(g.tab, g.row, g.status, g.proofreader, g.pr_date)))
