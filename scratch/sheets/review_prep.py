"""Network-free: compact review files for checking the newest seatable_sync run (tmp/review_*):
  review_refs.tsv       every 'duplicate of rN' / 'merged into rN' written, with this row's and the referenced row's
                        live root (root at xyz, from the newest seatable_match.tsv) and whether they are the same cell
  review_prg_notes.tsv  every shortened notes cell outside the PRs tab (tab, row, old, new)
  review_prs_notes.tsv  every shortened notes cell in the PRs tab (row, old, new)
  review_other.tsv      every other change (not notes shortening)
Usage: review_prep.py"""
import glob
import os
import re

import pandas as pd

from faab.config import FIGURES, TMP

log = pd.read_csv(sorted(glob.glob(os.path.join(FIGURES, "*_seatable_sync_changes.tsv")))[-1], sep="\t", dtype=str).fillna("")
m = pd.read_csv(sorted(glob.glob(os.path.join(FIGURES, "*_seatable_match.tsv")))[-1], sep="\t", dtype=str).fillna("")
cell = {(t, int(r)): a for t, r, a in zip(m.tab, m.row, m.root_at_xyz)}
orig = {(t, int(r)): o for t, r, o in zip(m.tab, m.row, m.root_id_orig)}
sh = log[log.why == "generated notes shortened"].copy()
refs = []
for x in sh.itertuples():
    for kind, ref in re.findall(r"(duplicate of|merged into) ((?:PR\w+ )?r\d+)\b", x.new):
        t, rr = (ref.split(" r")[0], int(ref.split(" r")[1])) if " r" in ref else (x.tab, int(ref[1:]))
        old_id = re.search(rf"{'duplicate target of' if kind == 'duplicate of' else 'merged into'} (\d{{15,}})", x.old)
        refs.append(dict(tab=x.tab, row=x.row, kind=kind, ref=ref, this_cell=cell.get((x.tab, int(x.row)), ""),
                         ref_cell=cell.get((t, rr), ""), ref_root_id_orig=orig.get((t, rr), ""),
                         old_target=old_id.group(1) if old_id else "",
                         same_cell=cell.get((x.tab, int(x.row)), "") == cell.get((t, rr), "x"),
                         orig_matches=(old_id.group(1) if old_id else "") == orig.get((t, rr), "")))
R = pd.DataFrame(refs)
R.to_csv(os.path.join(TMP, "review_refs.tsv"), sep="\t", index=False)
sh[sh.tab != "PRs"][["tab", "row", "old", "new"]].to_csv(os.path.join(TMP, "review_prg_notes.tsv"), sep="\t", index=False)
sh[sh.tab == "PRs"][["row", "old", "new"]].to_csv(os.path.join(TMP, "review_prs_notes.tsv"), sep="\t", index=False)
log[log.why != "generated notes shortened"].to_csv(os.path.join(TMP, "review_other.tsv"), sep="\t", index=False)
print(f"refs {len(R)}: same live cell {int(R.same_cell.sum()) if len(R) else 0}, record id matches "
      f"{int(R.orig_matches.sum()) if len(R) else 0}; prg notes {int((sh.tab != 'PRs').sum())}; prs notes "
      f"{int((sh.tab == 'PRs').sum())}; other changes {int((log.why != 'generated notes shortened').sum())}")
if len(R):
    print(R[~(R.same_cell & R.orig_matches)].to_string(index=False))
