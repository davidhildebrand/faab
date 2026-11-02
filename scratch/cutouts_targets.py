"""Target cells available for the new cutouts (2026-11-01): R7s in the PRs tab (class R7, by group, side and status),
annotated LMCs in the LMCs tab TSV (by type, side and status), and T4/T5 cells in SeaTable (by type, side and status)
plus the PRg1toT4T5 targets in the workbook. Network-free. Usage: cutouts_targets.py SEATABLE.tsv BOOK.xlsx LMC_TAB.tsv"""
import os
import re
import sys

import openpyxl
import pandas as pd

from faab.config import fig

S = pd.read_csv(sys.argv[1], sep="\t", dtype=str).fillna("")
tab = pd.read_csv(os.path.join(fig("ommatidia/20261101"), "tab_roots.tsv"), sep="\t", dtype=str).fillna("")
r7 = tab[tab.cls == "R7"]
print("R7 rows in the PRs tab (as of the 14:15 UTC lookup), by side x status:")
print(pd.crosstab(r7.side, r7.notes.str.len() >= 0).to_string() if False else
      r7.assign(g=r7.group.astype(int)).groupby("side").g.apply(lambda s: f"{len(s)} rows, groups {sorted(s)[:12]}...").to_string())
lines = [l.split("\t") for l in open(sys.argv[3]).read().split("\n") if l and l.split("\t")[0].isdigit()]
L = pd.DataFrame(lines, columns=open(sys.argv[3]).read().split("\n")[2].split("\t"))
ann = L[L.confidence == "annotated"]
print("\nannotated LMCs in the LMCs tab, type x side (status adequate / proofread / checked):")
good = ann[ann.status.str.lower().isin(["adequate", "proofread", "checked"])]
print(pd.crosstab(good.putative_type, good.side).to_string())
lab = S[["class", "subclass", "subsubclass", "type"]].agg(" ".join, axis=1)
t45 = S[lab.str.contains(r"\bT[45]", regex=True)].assign(label=lab)
print("\nSeaTable T4/T5 rows, label x side:")
print(pd.crosstab(t45.label.str.strip(), t45.side).to_string())
print("status:", t45.status.value_counts().to_dict())
wb = openpyxl.load_workbook(sys.argv[2], read_only=True)
ws = wb["PRg1toT4T5"]
rows = list(ws.iter_rows(values_only=True))
hdr = next(r for r in rows if r and r[0] == "hop")
tgt = [dict(zip(hdr, r)) for r in rows if r and str(r[0]).startswith("hop-3")]
print(f"\nPRg1toT4T5 hop-3 (T4/T5 targets): {len(tgt)}; status "
      f"{pd.Series([str(t.get('status')) for t in tgt]).value_counts().to_dict()}; annotation "
      f"{pd.Series([str(t.get('annotation')) for t in tgt]).value_counts().head(6).to_dict()}")
