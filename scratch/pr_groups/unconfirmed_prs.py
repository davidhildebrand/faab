"""PRs rows that failed the 2026-10-29 mesh-share check (pr_classes_mesh_shares: R1-R6 must end in the lamina, so a
large medulla share means not an R1-R6), with their newest SeaTable label. Network-free.
Usage: unconfirmed_prs.py MESH_SHARES.tsv SEATABLE.tsv"""
import sys

import pandas as pd

M = pd.read_csv(sys.argv[1], sep="\t", dtype=str).fillna("")
S = pd.read_csv(sys.argv[2], sep="\t", dtype=str).fillna("")
S["label"] = S[["superclass", "class", "subclass", "type"]].agg(lambda r: " / ".join(x for x in r if x), axis=1)
lab = dict(zip(S.root_now, S.label))
for c in ("LA", "ME", "n_pre", "n_post"):
    M[c] = pd.to_numeric(M[c], errors="coerce")
print(f"{len(M)} rows; confirmed {M.confirmed.value_counts().to_dict()}")
print("by class and source:\n", M.groupby(["class", M.source.str.slice(0, 30), "confirmed"]).size().to_string())
r16 = M[M["class"] == "R1-R6"]
print("\nR1-R6 medulla share by 'confirmed' (proofread in the PRs tab):")
print(r16.groupby("confirmed").ME.describe().round(2).to_string())
bad = r16[r16.ME >= 0.3].copy()
bad["SeaTable"] = bad.root_now.map(lab).fillna("")
bad["in_per_out"] = (bad.n_post / bad.n_pre).round(2)
print(f"\nR1-R6 rows with >= 30% in the medulla (an R1-R6 ends in the lamina): {len(bad)}")
print(bad[["group", "root_now", "status", "LA", "ME", "n_pre", "in_per_out", "SeaTable", "confirmed"]]
      .sort_values("ME", ascending=False).to_string(index=False))
