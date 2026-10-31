"""Network-free review of the newest figures/*_seatable_sync_changes.tsv: status transitions, annotation replacements,
dates, hand edits and MISMATCH rows. Usage: sync_log_review.py"""
import glob
import os

import pandas as pd

from faab.config import FIGURES

path = sorted(glob.glob(os.path.join(FIGURES, "*_seatable_sync_changes.tsv")))[-1]
L = pd.read_csv(path, sep="\t", dtype=str).fillna("")
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_colwidth", 90)
print(os.path.basename(path), len(L))
st = L[L.why == "SeaTable status"]
print("\nstatus transitions (all tabs):")
print((st.old.replace("", "(blank)") + " -> " + st.new).value_counts().to_string())
an = L[L.why == "SeaTable label"]
print("\nannotation replacements, top 25:")
print((an.old.replace("", "(blank)") + " -> " + an.new).value_counts().head(25).to_string())
print("\nnotes added, examples of the added text:")
na = L[L.why == "notes added"].copy()
na["added"] = [n[len(o):].lstrip("; ") if n.startswith(o) else n for o, n in zip(na.old, na.new)]
for part, k in pd.Series([p for a in na.added for p in a.split("; ")]).str.replace(r"\d{15,}", "<id>", regex=True) \
        .str.slice(0, 70).value_counts().head(25).items():
    print(f"  {k:4d}  {part}")
print("\nproofreader changes, top 20:")
pr = L[L.why == "SeaTable proofreaders added"]
print((pr.old.replace("", "(blank)") + " -> " + pr.new).value_counts().head(20).to_string())
print("\ndates fixed:")
print(L[L.why.str.contains("day-month|text in")][["tab", "row", "column", "old", "new"]].to_string(index=False))
print("\nhand edits, MISMATCH, checker moves, spelling:")
print(L[L.why.isin(["hand edit", "MISMATCH", "'added' was typed into checker", "status lowercase / spelling",
                    "SeaTable mark spelling / formula leftover"])][["tab", "row", "column", "old", "new", "why"]]
      .assign(old=lambda d: d.old.str.slice(0, 60), new=lambda d: d.new.str.slice(0, 60)).to_string(index=False))
