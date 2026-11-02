"""Readable summary of an orphan_fragments.py table: rows asking for a review first, then the rest; per row the tab,
row, block, root ids, status, names, size now, and whether the piece is now part of a tracked cell.
Usage: orphan_fragments_report.py ORPHANS.tsv"""
import sys

import pandas as pd

D = pd.read_csv(sys.argv[1], sep="\t", dtype=str).fillna("")
D["asks_review"] = D.orphan_text.str.contains("review", case=False) | D.notes.str.contains("review", case=False)
D["changed"] = D.root_now != D.root_id_curr
D["l2"] = pd.to_numeric(D.l2_chunks_now, errors="coerce")
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 70)
cols = ["tab", "row", "root_id_orig", "root_now", "status", "proofreader", "l2", "now_tracked_on", "orphan_text"]
for flag, title in [(True, "ASK FOR A REVIEW"), (False, "OTHER ORPHAN ROWS")]:
    S = D[D.asks_review == flag]
    print(f"\n== {title}: {len(S)} rows, {S.root_now.nunique()} cells now ==")
    print(S[cols].to_string(index=False))
print("\nsize now (L2 chunks):", D.l2.describe().round(1).to_dict())
print("root changed since the sheet:", int(D.changed.sum()), "| now on a tracked row:", int((D.now_tracked_on != "").sum()))
print("blocks:", D.groupby("tab").block.nunique().to_dict())
open_ = D[D.asks_review | ~D.status.str.lower().eq("stopped") | (D.l2 >= 5)]
print("\nblocks and full notes of the rows asking for a review, not stopped, or >= 5 L2 chunks:")
print(open_[["tab", "row", "block", "root_id_curr", "notes"]].to_string(index=False))
dup = D[D.duplicated("root_now", keep=False) & (D.root_now != "")]
print("same cell on several orphan rows:\n" + dup[["tab", "row", "root_id_orig", "root_now"]].to_string(index=False))
