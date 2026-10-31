"""Network-free: every notes cell the newest seatable_sync run wrote (from its change log), plus the notes of the
L side rows it added to the PRs tab, with lengths - to check that notes stay free of generated text.
Usage: notes_audit.py"""
import glob
import os

import pandas as pd

from faab.config import FIGURES
from faab.config import fig

log_path = sorted(glob.glob(fig("*_seatable_sync_changes.tsv")))[-1]
L = pd.read_csv(log_path, sep="\t", dtype=str).fillna("")
n = L[L.column == "notes"]
print(f"{os.path.basename(log_path)}: {len(n)} notes cells written")
for r in n.itertuples():
    print(f"  {r.tab} r{r.row} [{r.why}] ({len(r.new)} chars): {r.new}")
left = pd.read_csv(sorted(glob.glob(fig("*_pr_left_groups.tsv")))[-1], sep="\t", dtype=str).fillna("")
print(f"\nL side rows: {len(left)}; notes length p50 {left.notes.str.len().median():.0f}, max {left.notes.str.len().max()}; e.g.")
for x in left.notes.drop_duplicates().head(3):
    print(f"  {x}")
